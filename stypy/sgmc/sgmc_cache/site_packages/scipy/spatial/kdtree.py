
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright Anne M. Archibald 2008
2: # Released under the scipy license
3: from __future__ import division, print_function, absolute_import
4: 
5: import sys
6: import numpy as np
7: from heapq import heappush, heappop
8: import scipy.sparse
9: 
10: __all__ = ['minkowski_distance_p', 'minkowski_distance',
11:            'distance_matrix',
12:            'Rectangle', 'KDTree']
13: 
14: 
15: def minkowski_distance_p(x, y, p=2):
16:     '''
17:     Compute the p-th power of the L**p distance between two arrays.
18: 
19:     For efficiency, this function computes the L**p distance but does
20:     not extract the pth root. If `p` is 1 or infinity, this is equal to
21:     the actual L**p distance.
22: 
23:     Parameters
24:     ----------
25:     x : (M, K) array_like
26:         Input array.
27:     y : (N, K) array_like
28:         Input array.
29:     p : float, 1 <= p <= infinity
30:         Which Minkowski p-norm to use.
31: 
32:     Examples
33:     --------
34:     >>> from scipy.spatial import minkowski_distance_p
35:     >>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
36:     array([2, 1])
37: 
38:     '''
39:     x = np.asarray(x)
40:     y = np.asarray(y)
41:     if p == np.inf:
42:         return np.amax(np.abs(y-x), axis=-1)
43:     elif p == 1:
44:         return np.sum(np.abs(y-x), axis=-1)
45:     else:
46:         return np.sum(np.abs(y-x)**p, axis=-1)
47: 
48: 
49: def minkowski_distance(x, y, p=2):
50:     '''
51:     Compute the L**p distance between two arrays.
52: 
53:     Parameters
54:     ----------
55:     x : (M, K) array_like
56:         Input array.
57:     y : (N, K) array_like
58:         Input array.
59:     p : float, 1 <= p <= infinity
60:         Which Minkowski p-norm to use.
61: 
62:     Examples
63:     --------
64:     >>> from scipy.spatial import minkowski_distance
65:     >>> minkowski_distance([[0,0],[0,0]], [[1,1],[0,1]])
66:     array([ 1.41421356,  1.        ])
67: 
68:     '''
69:     x = np.asarray(x)
70:     y = np.asarray(y)
71:     if p == np.inf or p == 1:
72:         return minkowski_distance_p(x, y, p)
73:     else:
74:         return minkowski_distance_p(x, y, p)**(1./p)
75: 
76: 
77: class Rectangle(object):
78:     '''Hyperrectangle class.
79: 
80:     Represents a Cartesian product of intervals.
81:     '''
82:     def __init__(self, maxes, mins):
83:         '''Construct a hyperrectangle.'''
84:         self.maxes = np.maximum(maxes,mins).astype(float)
85:         self.mins = np.minimum(maxes,mins).astype(float)
86:         self.m, = self.maxes.shape
87: 
88:     def __repr__(self):
89:         return "<Rectangle %s>" % list(zip(self.mins, self.maxes))
90: 
91:     def volume(self):
92:         '''Total volume.'''
93:         return np.prod(self.maxes-self.mins)
94: 
95:     def split(self, d, split):
96:         '''
97:         Produce two hyperrectangles by splitting.
98: 
99:         In general, if you need to compute maximum and minimum
100:         distances to the children, it can be done more efficiently
101:         by updating the maximum and minimum distances to the parent.
102: 
103:         Parameters
104:         ----------
105:         d : int
106:             Axis to split hyperrectangle along.
107:         split : float
108:             Position along axis `d` to split at.
109: 
110:         '''
111:         mid = np.copy(self.maxes)
112:         mid[d] = split
113:         less = Rectangle(self.mins, mid)
114:         mid = np.copy(self.mins)
115:         mid[d] = split
116:         greater = Rectangle(mid, self.maxes)
117:         return less, greater
118: 
119:     def min_distance_point(self, x, p=2.):
120:         '''
121:         Return the minimum distance between input and points in the hyperrectangle.
122: 
123:         Parameters
124:         ----------
125:         x : array_like
126:             Input.
127:         p : float, optional
128:             Input.
129: 
130:         '''
131:         return minkowski_distance(0, np.maximum(0,np.maximum(self.mins-x,x-self.maxes)),p)
132: 
133:     def max_distance_point(self, x, p=2.):
134:         '''
135:         Return the maximum distance between input and points in the hyperrectangle.
136: 
137:         Parameters
138:         ----------
139:         x : array_like
140:             Input array.
141:         p : float, optional
142:             Input.
143: 
144:         '''
145:         return minkowski_distance(0, np.maximum(self.maxes-x,x-self.mins),p)
146: 
147:     def min_distance_rectangle(self, other, p=2.):
148:         '''
149:         Compute the minimum distance between points in the two hyperrectangles.
150: 
151:         Parameters
152:         ----------
153:         other : hyperrectangle
154:             Input.
155:         p : float
156:             Input.
157: 
158:         '''
159:         return minkowski_distance(0, np.maximum(0,np.maximum(self.mins-other.maxes,other.mins-self.maxes)),p)
160: 
161:     def max_distance_rectangle(self, other, p=2.):
162:         '''
163:         Compute the maximum distance between points in the two hyperrectangles.
164: 
165:         Parameters
166:         ----------
167:         other : hyperrectangle
168:             Input.
169:         p : float, optional
170:             Input.
171: 
172:         '''
173:         return minkowski_distance(0, np.maximum(self.maxes-other.mins,other.maxes-self.mins),p)
174: 
175: 
176: class KDTree(object):
177:     '''
178:     kd-tree for quick nearest-neighbor lookup
179: 
180:     This class provides an index into a set of k-dimensional points which
181:     can be used to rapidly look up the nearest neighbors of any point.
182: 
183:     Parameters
184:     ----------
185:     data : (N,K) array_like
186:         The data points to be indexed. This array is not copied, and
187:         so modifying this data will result in bogus results.
188:     leafsize : int, optional
189:         The number of points at which the algorithm switches over to
190:         brute-force.  Has to be positive.
191: 
192:     Raises
193:     ------
194:     RuntimeError
195:         The maximum recursion limit can be exceeded for large data
196:         sets.  If this happens, either increase the value for the `leafsize`
197:         parameter or increase the recursion limit by::
198: 
199:             >>> import sys
200:             >>> sys.setrecursionlimit(10000)
201: 
202:     See Also
203:     --------
204:     cKDTree : Implementation of `KDTree` in Cython
205: 
206:     Notes
207:     -----
208:     The algorithm used is described in Maneewongvatana and Mount 1999.
209:     The general idea is that the kd-tree is a binary tree, each of whose
210:     nodes represents an axis-aligned hyperrectangle. Each node specifies
211:     an axis and splits the set of points based on whether their coordinate
212:     along that axis is greater than or less than a particular value.
213: 
214:     During construction, the axis and splitting point are chosen by the
215:     "sliding midpoint" rule, which ensures that the cells do not all
216:     become long and thin.
217: 
218:     The tree can be queried for the r closest neighbors of any given point
219:     (optionally returning only those within some maximum distance of the
220:     point). It can also be queried, with a substantial gain in efficiency,
221:     for the r approximate closest neighbors.
222: 
223:     For large dimensions (20 is already large) do not expect this to run
224:     significantly faster than brute force. High-dimensional nearest-neighbor
225:     queries are a substantial open problem in computer science.
226: 
227:     The tree also supports all-neighbors queries, both with arrays of points
228:     and with other kd-trees. These do use a reasonably efficient algorithm,
229:     but the kd-tree is not necessarily the best data structure for this
230:     sort of calculation.
231: 
232:     '''
233:     def __init__(self, data, leafsize=10):
234:         self.data = np.asarray(data)
235:         self.n, self.m = np.shape(self.data)
236:         self.leafsize = int(leafsize)
237:         if self.leafsize < 1:
238:             raise ValueError("leafsize must be at least 1")
239:         self.maxes = np.amax(self.data,axis=0)
240:         self.mins = np.amin(self.data,axis=0)
241: 
242:         self.tree = self.__build(np.arange(self.n), self.maxes, self.mins)
243: 
244:     class node(object):
245:         if sys.version_info[0] >= 3:
246:             def __lt__(self, other):
247:                 return id(self) < id(other)
248: 
249:             def __gt__(self, other):
250:                 return id(self) > id(other)
251: 
252:             def __le__(self, other):
253:                 return id(self) <= id(other)
254: 
255:             def __ge__(self, other):
256:                 return id(self) >= id(other)
257: 
258:             def __eq__(self, other):
259:                 return id(self) == id(other)
260: 
261:     class leafnode(node):
262:         def __init__(self, idx):
263:             self.idx = idx
264:             self.children = len(idx)
265: 
266:     class innernode(node):
267:         def __init__(self, split_dim, split, less, greater):
268:             self.split_dim = split_dim
269:             self.split = split
270:             self.less = less
271:             self.greater = greater
272:             self.children = less.children+greater.children
273: 
274:     def __build(self, idx, maxes, mins):
275:         if len(idx) <= self.leafsize:
276:             return KDTree.leafnode(idx)
277:         else:
278:             data = self.data[idx]
279:             # maxes = np.amax(data,axis=0)
280:             # mins = np.amin(data,axis=0)
281:             d = np.argmax(maxes-mins)
282:             maxval = maxes[d]
283:             minval = mins[d]
284:             if maxval == minval:
285:                 # all points are identical; warn user?
286:                 return KDTree.leafnode(idx)
287:             data = data[:,d]
288: 
289:             # sliding midpoint rule; see Maneewongvatana and Mount 1999
290:             # for arguments that this is a good idea.
291:             split = (maxval+minval)/2
292:             less_idx = np.nonzero(data <= split)[0]
293:             greater_idx = np.nonzero(data > split)[0]
294:             if len(less_idx) == 0:
295:                 split = np.amin(data)
296:                 less_idx = np.nonzero(data <= split)[0]
297:                 greater_idx = np.nonzero(data > split)[0]
298:             if len(greater_idx) == 0:
299:                 split = np.amax(data)
300:                 less_idx = np.nonzero(data < split)[0]
301:                 greater_idx = np.nonzero(data >= split)[0]
302:             if len(less_idx) == 0:
303:                 # _still_ zero? all must have the same value
304:                 if not np.all(data == data[0]):
305:                     raise ValueError("Troublesome data array: %s" % data)
306:                 split = data[0]
307:                 less_idx = np.arange(len(data)-1)
308:                 greater_idx = np.array([len(data)-1])
309: 
310:             lessmaxes = np.copy(maxes)
311:             lessmaxes[d] = split
312:             greatermins = np.copy(mins)
313:             greatermins[d] = split
314:             return KDTree.innernode(d, split,
315:                     self.__build(idx[less_idx],lessmaxes,mins),
316:                     self.__build(idx[greater_idx],maxes,greatermins))
317: 
318:     def __query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
319: 
320:         side_distances = np.maximum(0,np.maximum(x-self.maxes,self.mins-x))
321:         if p != np.inf:
322:             side_distances **= p
323:             min_distance = np.sum(side_distances)
324:         else:
325:             min_distance = np.amax(side_distances)
326: 
327:         # priority queue for chasing nodes
328:         # entries are:
329:         #  minimum distance between the cell and the target
330:         #  distances between the nearest side of the cell and the target
331:         #  the head node of the cell
332:         q = [(min_distance,
333:               tuple(side_distances),
334:               self.tree)]
335:         # priority queue for the nearest neighbors
336:         # furthest known neighbor first
337:         # entries are (-distance**p, i)
338:         neighbors = []
339: 
340:         if eps == 0:
341:             epsfac = 1
342:         elif p == np.inf:
343:             epsfac = 1/(1+eps)
344:         else:
345:             epsfac = 1/(1+eps)**p
346: 
347:         if p != np.inf and distance_upper_bound != np.inf:
348:             distance_upper_bound = distance_upper_bound**p
349: 
350:         while q:
351:             min_distance, side_distances, node = heappop(q)
352:             if isinstance(node, KDTree.leafnode):
353:                 # brute-force
354:                 data = self.data[node.idx]
355:                 ds = minkowski_distance_p(data,x[np.newaxis,:],p)
356:                 for i in range(len(ds)):
357:                     if ds[i] < distance_upper_bound:
358:                         if len(neighbors) == k:
359:                             heappop(neighbors)
360:                         heappush(neighbors, (-ds[i], node.idx[i]))
361:                         if len(neighbors) == k:
362:                             distance_upper_bound = -neighbors[0][0]
363:             else:
364:                 # we don't push cells that are too far onto the queue at all,
365:                 # but since the distance_upper_bound decreases, we might get
366:                 # here even if the cell's too far
367:                 if min_distance > distance_upper_bound*epsfac:
368:                     # since this is the nearest cell, we're done, bail out
369:                     break
370:                 # compute minimum distances to the children and push them on
371:                 if x[node.split_dim] < node.split:
372:                     near, far = node.less, node.greater
373:                 else:
374:                     near, far = node.greater, node.less
375: 
376:                 # near child is at the same distance as the current node
377:                 heappush(q,(min_distance, side_distances, near))
378: 
379:                 # far child is further by an amount depending only
380:                 # on the split value
381:                 sd = list(side_distances)
382:                 if p == np.inf:
383:                     min_distance = max(min_distance, abs(node.split-x[node.split_dim]))
384:                 elif p == 1:
385:                     sd[node.split_dim] = np.abs(node.split-x[node.split_dim])
386:                     min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]
387:                 else:
388:                     sd[node.split_dim] = np.abs(node.split-x[node.split_dim])**p
389:                     min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]
390: 
391:                 # far child might be too far, if so, don't bother pushing it
392:                 if min_distance <= distance_upper_bound*epsfac:
393:                     heappush(q,(min_distance, tuple(sd), far))
394: 
395:         if p == np.inf:
396:             return sorted([(-d,i) for (d,i) in neighbors])
397:         else:
398:             return sorted([((-d)**(1./p),i) for (d,i) in neighbors])
399: 
400:     def query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):
401:         '''
402:         Query the kd-tree for nearest neighbors
403: 
404:         Parameters
405:         ----------
406:         x : array_like, last dimension self.m
407:             An array of points to query.
408:         k : int, optional
409:             The number of nearest neighbors to return.
410:         eps : nonnegative float, optional
411:             Return approximate nearest neighbors; the kth returned value
412:             is guaranteed to be no further than (1+eps) times the
413:             distance to the real kth nearest neighbor.
414:         p : float, 1<=p<=infinity, optional
415:             Which Minkowski p-norm to use.
416:             1 is the sum-of-absolute-values "Manhattan" distance
417:             2 is the usual Euclidean distance
418:             infinity is the maximum-coordinate-difference distance
419:         distance_upper_bound : nonnegative float, optional
420:             Return only neighbors within this distance. This is used to prune
421:             tree searches, so if you are doing a series of nearest-neighbor
422:             queries, it may help to supply the distance to the nearest neighbor
423:             of the most recent point.
424: 
425:         Returns
426:         -------
427:         d : float or array of floats
428:             The distances to the nearest neighbors.
429:             If x has shape tuple+(self.m,), then d has shape tuple if
430:             k is one, or tuple+(k,) if k is larger than one. Missing
431:             neighbors (e.g. when k > n or distance_upper_bound is
432:             given) are indicated with infinite distances.  If k is None,
433:             then d is an object array of shape tuple, containing lists
434:             of distances. In either case the hits are sorted by distance
435:             (nearest first).
436:         i : integer or array of integers
437:             The locations of the neighbors in self.data. i is the same
438:             shape as d.
439: 
440:         Examples
441:         --------
442:         >>> from scipy import spatial
443:         >>> x, y = np.mgrid[0:5, 2:8]
444:         >>> tree = spatial.KDTree(list(zip(x.ravel(), y.ravel())))
445:         >>> tree.data
446:         array([[0, 2],
447:                [0, 3],
448:                [0, 4],
449:                [0, 5],
450:                [0, 6],
451:                [0, 7],
452:                [1, 2],
453:                [1, 3],
454:                [1, 4],
455:                [1, 5],
456:                [1, 6],
457:                [1, 7],
458:                [2, 2],
459:                [2, 3],
460:                [2, 4],
461:                [2, 5],
462:                [2, 6],
463:                [2, 7],
464:                [3, 2],
465:                [3, 3],
466:                [3, 4],
467:                [3, 5],
468:                [3, 6],
469:                [3, 7],
470:                [4, 2],
471:                [4, 3],
472:                [4, 4],
473:                [4, 5],
474:                [4, 6],
475:                [4, 7]])
476:         >>> pts = np.array([[0, 0], [2.1, 2.9]])
477:         >>> tree.query(pts)
478:         (array([ 2.        ,  0.14142136]), array([ 0, 13]))
479:         >>> tree.query(pts[0])
480:         (2.0, 0)
481: 
482:         '''
483:         x = np.asarray(x)
484:         if np.shape(x)[-1] != self.m:
485:             raise ValueError("x must consist of vectors of length %d but has shape %s" % (self.m, np.shape(x)))
486:         if p < 1:
487:             raise ValueError("Only p-norms with 1<=p<=infinity permitted")
488:         retshape = np.shape(x)[:-1]
489:         if retshape != ():
490:             if k is None:
491:                 dd = np.empty(retshape,dtype=object)
492:                 ii = np.empty(retshape,dtype=object)
493:             elif k > 1:
494:                 dd = np.empty(retshape+(k,),dtype=float)
495:                 dd.fill(np.inf)
496:                 ii = np.empty(retshape+(k,),dtype=int)
497:                 ii.fill(self.n)
498:             elif k == 1:
499:                 dd = np.empty(retshape,dtype=float)
500:                 dd.fill(np.inf)
501:                 ii = np.empty(retshape,dtype=int)
502:                 ii.fill(self.n)
503:             else:
504:                 raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
505:             for c in np.ndindex(retshape):
506:                 hits = self.__query(x[c], k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)
507:                 if k is None:
508:                     dd[c] = [d for (d,i) in hits]
509:                     ii[c] = [i for (d,i) in hits]
510:                 elif k > 1:
511:                     for j in range(len(hits)):
512:                         dd[c+(j,)], ii[c+(j,)] = hits[j]
513:                 elif k == 1:
514:                     if len(hits) > 0:
515:                         dd[c], ii[c] = hits[0]
516:                     else:
517:                         dd[c] = np.inf
518:                         ii[c] = self.n
519:             return dd, ii
520:         else:
521:             hits = self.__query(x, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)
522:             if k is None:
523:                 return [d for (d,i) in hits], [i for (d,i) in hits]
524:             elif k == 1:
525:                 if len(hits) > 0:
526:                     return hits[0]
527:                 else:
528:                     return np.inf, self.n
529:             elif k > 1:
530:                 dd = np.empty(k,dtype=float)
531:                 dd.fill(np.inf)
532:                 ii = np.empty(k,dtype=int)
533:                 ii.fill(self.n)
534:                 for j in range(len(hits)):
535:                     dd[j], ii[j] = hits[j]
536:                 return dd, ii
537:             else:
538:                 raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
539: 
540:     def __query_ball_point(self, x, r, p=2., eps=0):
541:         R = Rectangle(self.maxes, self.mins)
542: 
543:         def traverse_checking(node, rect):
544:             if rect.min_distance_point(x, p) > r / (1. + eps):
545:                 return []
546:             elif rect.max_distance_point(x, p) < r * (1. + eps):
547:                 return traverse_no_checking(node)
548:             elif isinstance(node, KDTree.leafnode):
549:                 d = self.data[node.idx]
550:                 return node.idx[minkowski_distance(d, x, p) <= r].tolist()
551:             else:
552:                 less, greater = rect.split(node.split_dim, node.split)
553:                 return traverse_checking(node.less, less) + \
554:                        traverse_checking(node.greater, greater)
555: 
556:         def traverse_no_checking(node):
557:             if isinstance(node, KDTree.leafnode):
558:                 return node.idx.tolist()
559:             else:
560:                 return traverse_no_checking(node.less) + \
561:                        traverse_no_checking(node.greater)
562: 
563:         return traverse_checking(self.tree, R)
564: 
565:     def query_ball_point(self, x, r, p=2., eps=0):
566:         '''Find all points within distance r of point(s) x.
567: 
568:         Parameters
569:         ----------
570:         x : array_like, shape tuple + (self.m,)
571:             The point or points to search for neighbors of.
572:         r : positive float
573:             The radius of points to return.
574:         p : float, optional
575:             Which Minkowski p-norm to use.  Should be in the range [1, inf].
576:         eps : nonnegative float, optional
577:             Approximate search. Branches of the tree are not explored if their
578:             nearest points are further than ``r / (1 + eps)``, and branches are
579:             added in bulk if their furthest points are nearer than
580:             ``r * (1 + eps)``.
581: 
582:         Returns
583:         -------
584:         results : list or array of lists
585:             If `x` is a single point, returns a list of the indices of the
586:             neighbors of `x`. If `x` is an array of points, returns an object
587:             array of shape tuple containing lists of neighbors.
588: 
589:         Notes
590:         -----
591:         If you have many points whose neighbors you want to find, you may save
592:         substantial amounts of time by putting them in a KDTree and using
593:         query_ball_tree.
594: 
595:         Examples
596:         --------
597:         >>> from scipy import spatial
598:         >>> x, y = np.mgrid[0:5, 0:5]
599:         >>> points = np.c_[x.ravel(), y.ravel()]
600:         >>> tree = spatial.KDTree(points)
601:         >>> tree.query_ball_point([2, 0], 1)
602:         [5, 10, 11, 15]
603: 
604:         Query multiple points and plot the results:
605: 
606:         >>> import matplotlib.pyplot as plt
607:         >>> points = np.asarray(points)
608:         >>> plt.plot(points[:,0], points[:,1], '.')
609:         >>> for results in tree.query_ball_point(([2, 0], [3, 3]), 1):
610:         ...     nearby_points = points[results]
611:         ...     plt.plot(nearby_points[:,0], nearby_points[:,1], 'o')
612:         >>> plt.margins(0.1, 0.1)
613:         >>> plt.show()
614: 
615:         '''
616:         x = np.asarray(x)
617:         if x.shape[-1] != self.m:
618:             raise ValueError("Searching for a %d-dimensional point in a "
619:                              "%d-dimensional KDTree" % (x.shape[-1], self.m))
620:         if len(x.shape) == 1:
621:             return self.__query_ball_point(x, r, p, eps)
622:         else:
623:             retshape = x.shape[:-1]
624:             result = np.empty(retshape, dtype=object)
625:             for c in np.ndindex(retshape):
626:                 result[c] = self.__query_ball_point(x[c], r, p=p, eps=eps)
627:             return result
628: 
629:     def query_ball_tree(self, other, r, p=2., eps=0):
630:         '''Find all pairs of points whose distance is at most r
631: 
632:         Parameters
633:         ----------
634:         other : KDTree instance
635:             The tree containing points to search against.
636:         r : float
637:             The maximum distance, has to be positive.
638:         p : float, optional
639:             Which Minkowski norm to use.  `p` has to meet the condition
640:             ``1 <= p <= infinity``.
641:         eps : float, optional
642:             Approximate search.  Branches of the tree are not explored
643:             if their nearest points are further than ``r/(1+eps)``, and
644:             branches are added in bulk if their furthest points are nearer
645:             than ``r * (1+eps)``.  `eps` has to be non-negative.
646: 
647:         Returns
648:         -------
649:         results : list of lists
650:             For each element ``self.data[i]`` of this tree, ``results[i]`` is a
651:             list of the indices of its neighbors in ``other.data``.
652: 
653:         '''
654:         results = [[] for i in range(self.n)]
655: 
656:         def traverse_checking(node1, rect1, node2, rect2):
657:             if rect1.min_distance_rectangle(rect2, p) > r/(1.+eps):
658:                 return
659:             elif rect1.max_distance_rectangle(rect2, p) < r*(1.+eps):
660:                 traverse_no_checking(node1, node2)
661:             elif isinstance(node1, KDTree.leafnode):
662:                 if isinstance(node2, KDTree.leafnode):
663:                     d = other.data[node2.idx]
664:                     for i in node1.idx:
665:                         results[i] += node2.idx[minkowski_distance(d,self.data[i],p) <= r].tolist()
666:                 else:
667:                     less, greater = rect2.split(node2.split_dim, node2.split)
668:                     traverse_checking(node1,rect1,node2.less,less)
669:                     traverse_checking(node1,rect1,node2.greater,greater)
670:             elif isinstance(node2, KDTree.leafnode):
671:                 less, greater = rect1.split(node1.split_dim, node1.split)
672:                 traverse_checking(node1.less,less,node2,rect2)
673:                 traverse_checking(node1.greater,greater,node2,rect2)
674:             else:
675:                 less1, greater1 = rect1.split(node1.split_dim, node1.split)
676:                 less2, greater2 = rect2.split(node2.split_dim, node2.split)
677:                 traverse_checking(node1.less,less1,node2.less,less2)
678:                 traverse_checking(node1.less,less1,node2.greater,greater2)
679:                 traverse_checking(node1.greater,greater1,node2.less,less2)
680:                 traverse_checking(node1.greater,greater1,node2.greater,greater2)
681: 
682:         def traverse_no_checking(node1, node2):
683:             if isinstance(node1, KDTree.leafnode):
684:                 if isinstance(node2, KDTree.leafnode):
685:                     for i in node1.idx:
686:                         results[i] += node2.idx.tolist()
687:                 else:
688:                     traverse_no_checking(node1, node2.less)
689:                     traverse_no_checking(node1, node2.greater)
690:             else:
691:                 traverse_no_checking(node1.less, node2)
692:                 traverse_no_checking(node1.greater, node2)
693: 
694:         traverse_checking(self.tree, Rectangle(self.maxes, self.mins),
695:                           other.tree, Rectangle(other.maxes, other.mins))
696:         return results
697: 
698:     def query_pairs(self, r, p=2., eps=0):
699:         '''
700:         Find all pairs of points within a distance.
701: 
702:         Parameters
703:         ----------
704:         r : positive float
705:             The maximum distance.
706:         p : float, optional
707:             Which Minkowski norm to use.  `p` has to meet the condition
708:             ``1 <= p <= infinity``.
709:         eps : float, optional
710:             Approximate search.  Branches of the tree are not explored
711:             if their nearest points are further than ``r/(1+eps)``, and
712:             branches are added in bulk if their furthest points are nearer
713:             than ``r * (1+eps)``.  `eps` has to be non-negative.
714: 
715:         Returns
716:         -------
717:         results : set
718:             Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding
719:             positions are close.
720: 
721:         '''
722:         results = set()
723: 
724:         def traverse_checking(node1, rect1, node2, rect2):
725:             if rect1.min_distance_rectangle(rect2, p) > r/(1.+eps):
726:                 return
727:             elif rect1.max_distance_rectangle(rect2, p) < r*(1.+eps):
728:                 traverse_no_checking(node1, node2)
729:             elif isinstance(node1, KDTree.leafnode):
730:                 if isinstance(node2, KDTree.leafnode):
731:                     # Special care to avoid duplicate pairs
732:                     if id(node1) == id(node2):
733:                         d = self.data[node2.idx]
734:                         for i in node1.idx:
735:                             for j in node2.idx[minkowski_distance(d,self.data[i],p) <= r]:
736:                                 if i < j:
737:                                     results.add((i,j))
738:                     else:
739:                         d = self.data[node2.idx]
740:                         for i in node1.idx:
741:                             for j in node2.idx[minkowski_distance(d,self.data[i],p) <= r]:
742:                                 if i < j:
743:                                     results.add((i,j))
744:                                 elif j < i:
745:                                     results.add((j,i))
746:                 else:
747:                     less, greater = rect2.split(node2.split_dim, node2.split)
748:                     traverse_checking(node1,rect1,node2.less,less)
749:                     traverse_checking(node1,rect1,node2.greater,greater)
750:             elif isinstance(node2, KDTree.leafnode):
751:                 less, greater = rect1.split(node1.split_dim, node1.split)
752:                 traverse_checking(node1.less,less,node2,rect2)
753:                 traverse_checking(node1.greater,greater,node2,rect2)
754:             else:
755:                 less1, greater1 = rect1.split(node1.split_dim, node1.split)
756:                 less2, greater2 = rect2.split(node2.split_dim, node2.split)
757:                 traverse_checking(node1.less,less1,node2.less,less2)
758:                 traverse_checking(node1.less,less1,node2.greater,greater2)
759: 
760:                 # Avoid traversing (node1.less, node2.greater) and
761:                 # (node1.greater, node2.less) (it's the same node pair twice
762:                 # over, which is the source of the complication in the
763:                 # original KDTree.query_pairs)
764:                 if id(node1) != id(node2):
765:                     traverse_checking(node1.greater,greater1,node2.less,less2)
766: 
767:                 traverse_checking(node1.greater,greater1,node2.greater,greater2)
768: 
769:         def traverse_no_checking(node1, node2):
770:             if isinstance(node1, KDTree.leafnode):
771:                 if isinstance(node2, KDTree.leafnode):
772:                     # Special care to avoid duplicate pairs
773:                     if id(node1) == id(node2):
774:                         for i in node1.idx:
775:                             for j in node2.idx:
776:                                 if i < j:
777:                                     results.add((i,j))
778:                     else:
779:                         for i in node1.idx:
780:                             for j in node2.idx:
781:                                 if i < j:
782:                                     results.add((i,j))
783:                                 elif j < i:
784:                                     results.add((j,i))
785:                 else:
786:                     traverse_no_checking(node1, node2.less)
787:                     traverse_no_checking(node1, node2.greater)
788:             else:
789:                 # Avoid traversing (node1.less, node2.greater) and
790:                 # (node1.greater, node2.less) (it's the same node pair twice
791:                 # over, which is the source of the complication in the
792:                 # original KDTree.query_pairs)
793:                 if id(node1) == id(node2):
794:                     traverse_no_checking(node1.less, node2.less)
795:                     traverse_no_checking(node1.less, node2.greater)
796:                     traverse_no_checking(node1.greater, node2.greater)
797:                 else:
798:                     traverse_no_checking(node1.less, node2)
799:                     traverse_no_checking(node1.greater, node2)
800: 
801:         traverse_checking(self.tree, Rectangle(self.maxes, self.mins),
802:                           self.tree, Rectangle(self.maxes, self.mins))
803:         return results
804: 
805:     def count_neighbors(self, other, r, p=2.):
806:         '''
807:         Count how many nearby pairs can be formed.
808: 
809:         Count the number of pairs (x1,x2) can be formed, with x1 drawn
810:         from self and x2 drawn from `other`, and where
811:         ``distance(x1, x2, p) <= r``.
812:         This is the "two-point correlation" described in Gray and Moore 2000,
813:         "N-body problems in statistical learning", and the code here is based
814:         on their algorithm.
815: 
816:         Parameters
817:         ----------
818:         other : KDTree instance
819:             The other tree to draw points from.
820:         r : float or one-dimensional array of floats
821:             The radius to produce a count for. Multiple radii are searched with
822:             a single tree traversal.
823:         p : float, 1<=p<=infinity, optional
824:             Which Minkowski p-norm to use
825: 
826:         Returns
827:         -------
828:         result : int or 1-D array of ints
829:             The number of pairs. Note that this is internally stored in a numpy
830:             int, and so may overflow if very large (2e9).
831: 
832:         '''
833:         def traverse(node1, rect1, node2, rect2, idx):
834:             min_r = rect1.min_distance_rectangle(rect2,p)
835:             max_r = rect1.max_distance_rectangle(rect2,p)
836:             c_greater = r[idx] > max_r
837:             result[idx[c_greater]] += node1.children*node2.children
838:             idx = idx[(min_r <= r[idx]) & (r[idx] <= max_r)]
839:             if len(idx) == 0:
840:                 return
841: 
842:             if isinstance(node1,KDTree.leafnode):
843:                 if isinstance(node2,KDTree.leafnode):
844:                     ds = minkowski_distance(self.data[node1.idx][:,np.newaxis,:],
845:                                   other.data[node2.idx][np.newaxis,:,:],
846:                                   p).ravel()
847:                     ds.sort()
848:                     result[idx] += np.searchsorted(ds,r[idx],side='right')
849:                 else:
850:                     less, greater = rect2.split(node2.split_dim, node2.split)
851:                     traverse(node1, rect1, node2.less, less, idx)
852:                     traverse(node1, rect1, node2.greater, greater, idx)
853:             else:
854:                 if isinstance(node2,KDTree.leafnode):
855:                     less, greater = rect1.split(node1.split_dim, node1.split)
856:                     traverse(node1.less, less, node2, rect2, idx)
857:                     traverse(node1.greater, greater, node2, rect2, idx)
858:                 else:
859:                     less1, greater1 = rect1.split(node1.split_dim, node1.split)
860:                     less2, greater2 = rect2.split(node2.split_dim, node2.split)
861:                     traverse(node1.less,less1,node2.less,less2,idx)
862:                     traverse(node1.less,less1,node2.greater,greater2,idx)
863:                     traverse(node1.greater,greater1,node2.less,less2,idx)
864:                     traverse(node1.greater,greater1,node2.greater,greater2,idx)
865: 
866:         R1 = Rectangle(self.maxes, self.mins)
867:         R2 = Rectangle(other.maxes, other.mins)
868:         if np.shape(r) == ():
869:             r = np.array([r])
870:             result = np.zeros(1,dtype=int)
871:             traverse(self.tree, R1, other.tree, R2, np.arange(1))
872:             return result[0]
873:         elif len(np.shape(r)) == 1:
874:             r = np.asarray(r)
875:             n, = r.shape
876:             result = np.zeros(n,dtype=int)
877:             traverse(self.tree, R1, other.tree, R2, np.arange(n))
878:             return result
879:         else:
880:             raise ValueError("r must be either a single value or a one-dimensional array of values")
881: 
882:     def sparse_distance_matrix(self, other, max_distance, p=2.):
883:         '''
884:         Compute a sparse distance matrix
885: 
886:         Computes a distance matrix between two KDTrees, leaving as zero
887:         any distance greater than max_distance.
888: 
889:         Parameters
890:         ----------
891:         other : KDTree
892: 
893:         max_distance : positive float
894: 
895:         p : float, optional
896: 
897:         Returns
898:         -------
899:         result : dok_matrix
900:             Sparse matrix representing the results in "dictionary of keys" format.
901: 
902:         '''
903:         result = scipy.sparse.dok_matrix((self.n,other.n))
904: 
905:         def traverse(node1, rect1, node2, rect2):
906:             if rect1.min_distance_rectangle(rect2, p) > max_distance:
907:                 return
908:             elif isinstance(node1, KDTree.leafnode):
909:                 if isinstance(node2, KDTree.leafnode):
910:                     for i in node1.idx:
911:                         for j in node2.idx:
912:                             d = minkowski_distance(self.data[i],other.data[j],p)
913:                             if d <= max_distance:
914:                                 result[i,j] = d
915:                 else:
916:                     less, greater = rect2.split(node2.split_dim, node2.split)
917:                     traverse(node1,rect1,node2.less,less)
918:                     traverse(node1,rect1,node2.greater,greater)
919:             elif isinstance(node2, KDTree.leafnode):
920:                 less, greater = rect1.split(node1.split_dim, node1.split)
921:                 traverse(node1.less,less,node2,rect2)
922:                 traverse(node1.greater,greater,node2,rect2)
923:             else:
924:                 less1, greater1 = rect1.split(node1.split_dim, node1.split)
925:                 less2, greater2 = rect2.split(node2.split_dim, node2.split)
926:                 traverse(node1.less,less1,node2.less,less2)
927:                 traverse(node1.less,less1,node2.greater,greater2)
928:                 traverse(node1.greater,greater1,node2.less,less2)
929:                 traverse(node1.greater,greater1,node2.greater,greater2)
930:         traverse(self.tree, Rectangle(self.maxes, self.mins),
931:                  other.tree, Rectangle(other.maxes, other.mins))
932: 
933:         return result
934: 
935: 
936: def distance_matrix(x, y, p=2, threshold=1000000):
937:     '''
938:     Compute the distance matrix.
939: 
940:     Returns the matrix of all pair-wise distances.
941: 
942:     Parameters
943:     ----------
944:     x : (M, K) array_like
945:         Matrix of M vectors in K dimensions.
946:     y : (N, K) array_like
947:         Matrix of N vectors in K dimensions.
948:     p : float, 1 <= p <= infinity
949:         Which Minkowski p-norm to use.
950:     threshold : positive int
951:         If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead
952:         of large temporary arrays.
953: 
954:     Returns
955:     -------
956:     result : (M, N) ndarray
957:         Matrix containing the distance from every vector in `x` to every vector
958:         in `y`.
959: 
960:     Examples
961:     --------
962:     >>> from scipy.spatial import distance_matrix
963:     >>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])
964:     array([[ 1.        ,  1.41421356],
965:            [ 1.41421356,  1.        ]])
966: 
967:     '''
968: 
969:     x = np.asarray(x)
970:     m, k = x.shape
971:     y = np.asarray(y)
972:     n, kk = y.shape
973: 
974:     if k != kk:
975:         raise ValueError("x contains %d-dimensional vectors but y contains %d-dimensional vectors" % (k, kk))
976: 
977:     if m*n*k <= threshold:
978:         return minkowski_distance(x[:,np.newaxis,:],y[np.newaxis,:,:],p)
979:     else:
980:         result = np.empty((m,n),dtype=float)  # FIXME: figure out the best dtype
981:         if m < n:
982:             for i in range(m):
983:                 result[i,:] = minkowski_distance(x[i],y,p)
984:         else:
985:             for j in range(n):
986:                 result[:,j] = minkowski_distance(x,y[j],p)
987:         return result
988: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_466815 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_466815) is not StypyTypeError):

    if (import_466815 != 'pyd_module'):
        __import__(import_466815)
        sys_modules_466816 = sys.modules[import_466815]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_466816.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_466815)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from heapq import heappush, heappop' statement (line 7)
try:
    from heapq import heappush, heappop

except:
    heappush = UndefinedType
    heappop = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'heapq', None, module_type_store, ['heappush', 'heappop'], [heappush, heappop])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import scipy.sparse' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_466817 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse')

if (type(import_466817) is not StypyTypeError):

    if (import_466817 != 'pyd_module'):
        __import__(import_466817)
        sys_modules_466818 = sys.modules[import_466817]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse', sys_modules_466818.module_type_store, module_type_store)
    else:
        import scipy.sparse

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.sparse', import_466817)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')


# Assigning a List to a Name (line 10):

# Assigning a List to a Name (line 10):
__all__ = ['minkowski_distance_p', 'minkowski_distance', 'distance_matrix', 'Rectangle', 'KDTree']
module_type_store.set_exportable_members(['minkowski_distance_p', 'minkowski_distance', 'distance_matrix', 'Rectangle', 'KDTree'])

# Obtaining an instance of the builtin type 'list' (line 10)
list_466819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_466820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'minkowski_distance_p')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_466819, str_466820)
# Adding element type (line 10)
str_466821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 35), 'str', 'minkowski_distance')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_466819, str_466821)
# Adding element type (line 10)
str_466822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'str', 'distance_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_466819, str_466822)
# Adding element type (line 10)
str_466823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'Rectangle')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_466819, str_466823)
# Adding element type (line 10)
str_466824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 24), 'str', 'KDTree')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_466819, str_466824)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', list_466819)

@norecursion
def minkowski_distance_p(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_466825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 33), 'int')
    defaults = [int_466825]
    # Create a new context for function 'minkowski_distance_p'
    module_type_store = module_type_store.open_function_context('minkowski_distance_p', 15, 0, False)
    
    # Passed parameters checking function
    minkowski_distance_p.stypy_localization = localization
    minkowski_distance_p.stypy_type_of_self = None
    minkowski_distance_p.stypy_type_store = module_type_store
    minkowski_distance_p.stypy_function_name = 'minkowski_distance_p'
    minkowski_distance_p.stypy_param_names_list = ['x', 'y', 'p']
    minkowski_distance_p.stypy_varargs_param_name = None
    minkowski_distance_p.stypy_kwargs_param_name = None
    minkowski_distance_p.stypy_call_defaults = defaults
    minkowski_distance_p.stypy_call_varargs = varargs
    minkowski_distance_p.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'minkowski_distance_p', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'minkowski_distance_p', localization, ['x', 'y', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'minkowski_distance_p(...)' code ##################

    str_466826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'str', '\n    Compute the p-th power of the L**p distance between two arrays.\n\n    For efficiency, this function computes the L**p distance but does\n    not extract the pth root. If `p` is 1 or infinity, this is equal to\n    the actual L**p distance.\n\n    Parameters\n    ----------\n    x : (M, K) array_like\n        Input array.\n    y : (N, K) array_like\n        Input array.\n    p : float, 1 <= p <= infinity\n        Which Minkowski p-norm to use.\n\n    Examples\n    --------\n    >>> from scipy.spatial import minkowski_distance_p\n    >>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])\n    array([2, 1])\n\n    ')
    
    # Assigning a Call to a Name (line 39):
    
    # Assigning a Call to a Name (line 39):
    
    # Call to asarray(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'x' (line 39)
    x_466829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'x', False)
    # Processing the call keyword arguments (line 39)
    kwargs_466830 = {}
    # Getting the type of 'np' (line 39)
    np_466827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 39)
    asarray_466828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), np_466827, 'asarray')
    # Calling asarray(args, kwargs) (line 39)
    asarray_call_result_466831 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), asarray_466828, *[x_466829], **kwargs_466830)
    
    # Assigning a type to the variable 'x' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'x', asarray_call_result_466831)
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to asarray(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'y' (line 40)
    y_466834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'y', False)
    # Processing the call keyword arguments (line 40)
    kwargs_466835 = {}
    # Getting the type of 'np' (line 40)
    np_466832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 40)
    asarray_466833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), np_466832, 'asarray')
    # Calling asarray(args, kwargs) (line 40)
    asarray_call_result_466836 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), asarray_466833, *[y_466834], **kwargs_466835)
    
    # Assigning a type to the variable 'y' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'y', asarray_call_result_466836)
    
    
    # Getting the type of 'p' (line 41)
    p_466837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 7), 'p')
    # Getting the type of 'np' (line 41)
    np_466838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'np')
    # Obtaining the member 'inf' of a type (line 41)
    inf_466839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), np_466838, 'inf')
    # Applying the binary operator '==' (line 41)
    result_eq_466840 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 7), '==', p_466837, inf_466839)
    
    # Testing the type of an if condition (line 41)
    if_condition_466841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), result_eq_466840)
    # Assigning a type to the variable 'if_condition_466841' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_466841', if_condition_466841)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to amax(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Call to abs(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'y' (line 42)
    y_466846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 30), 'y', False)
    # Getting the type of 'x' (line 42)
    x_466847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 32), 'x', False)
    # Applying the binary operator '-' (line 42)
    result_sub_466848 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 30), '-', y_466846, x_466847)
    
    # Processing the call keyword arguments (line 42)
    kwargs_466849 = {}
    # Getting the type of 'np' (line 42)
    np_466844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'np', False)
    # Obtaining the member 'abs' of a type (line 42)
    abs_466845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 23), np_466844, 'abs')
    # Calling abs(args, kwargs) (line 42)
    abs_call_result_466850 = invoke(stypy.reporting.localization.Localization(__file__, 42, 23), abs_466845, *[result_sub_466848], **kwargs_466849)
    
    # Processing the call keyword arguments (line 42)
    int_466851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 41), 'int')
    keyword_466852 = int_466851
    kwargs_466853 = {'axis': keyword_466852}
    # Getting the type of 'np' (line 42)
    np_466842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'np', False)
    # Obtaining the member 'amax' of a type (line 42)
    amax_466843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), np_466842, 'amax')
    # Calling amax(args, kwargs) (line 42)
    amax_call_result_466854 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), amax_466843, *[abs_call_result_466850], **kwargs_466853)
    
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', amax_call_result_466854)
    # SSA branch for the else part of an if statement (line 41)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'p' (line 43)
    p_466855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 9), 'p')
    int_466856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 14), 'int')
    # Applying the binary operator '==' (line 43)
    result_eq_466857 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 9), '==', p_466855, int_466856)
    
    # Testing the type of an if condition (line 43)
    if_condition_466858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 9), result_eq_466857)
    # Assigning a type to the variable 'if_condition_466858' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 9), 'if_condition_466858', if_condition_466858)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to sum(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Call to abs(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'y' (line 44)
    y_466863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 29), 'y', False)
    # Getting the type of 'x' (line 44)
    x_466864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 31), 'x', False)
    # Applying the binary operator '-' (line 44)
    result_sub_466865 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 29), '-', y_466863, x_466864)
    
    # Processing the call keyword arguments (line 44)
    kwargs_466866 = {}
    # Getting the type of 'np' (line 44)
    np_466861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'np', False)
    # Obtaining the member 'abs' of a type (line 44)
    abs_466862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 22), np_466861, 'abs')
    # Calling abs(args, kwargs) (line 44)
    abs_call_result_466867 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), abs_466862, *[result_sub_466865], **kwargs_466866)
    
    # Processing the call keyword arguments (line 44)
    int_466868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 40), 'int')
    keyword_466869 = int_466868
    kwargs_466870 = {'axis': keyword_466869}
    # Getting the type of 'np' (line 44)
    np_466859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'np', False)
    # Obtaining the member 'sum' of a type (line 44)
    sum_466860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 15), np_466859, 'sum')
    # Calling sum(args, kwargs) (line 44)
    sum_call_result_466871 = invoke(stypy.reporting.localization.Localization(__file__, 44, 15), sum_466860, *[abs_call_result_466867], **kwargs_466870)
    
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', sum_call_result_466871)
    # SSA branch for the else part of an if statement (line 43)
    module_type_store.open_ssa_branch('else')
    
    # Call to sum(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Call to abs(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'y' (line 46)
    y_466876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'y', False)
    # Getting the type of 'x' (line 46)
    x_466877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'x', False)
    # Applying the binary operator '-' (line 46)
    result_sub_466878 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 29), '-', y_466876, x_466877)
    
    # Processing the call keyword arguments (line 46)
    kwargs_466879 = {}
    # Getting the type of 'np' (line 46)
    np_466874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'np', False)
    # Obtaining the member 'abs' of a type (line 46)
    abs_466875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 22), np_466874, 'abs')
    # Calling abs(args, kwargs) (line 46)
    abs_call_result_466880 = invoke(stypy.reporting.localization.Localization(__file__, 46, 22), abs_466875, *[result_sub_466878], **kwargs_466879)
    
    # Getting the type of 'p' (line 46)
    p_466881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 35), 'p', False)
    # Applying the binary operator '**' (line 46)
    result_pow_466882 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 22), '**', abs_call_result_466880, p_466881)
    
    # Processing the call keyword arguments (line 46)
    int_466883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 43), 'int')
    keyword_466884 = int_466883
    kwargs_466885 = {'axis': keyword_466884}
    # Getting the type of 'np' (line 46)
    np_466872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'np', False)
    # Obtaining the member 'sum' of a type (line 46)
    sum_466873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 15), np_466872, 'sum')
    # Calling sum(args, kwargs) (line 46)
    sum_call_result_466886 = invoke(stypy.reporting.localization.Localization(__file__, 46, 15), sum_466873, *[result_pow_466882], **kwargs_466885)
    
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stypy_return_type', sum_call_result_466886)
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'minkowski_distance_p(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'minkowski_distance_p' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_466887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_466887)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'minkowski_distance_p'
    return stypy_return_type_466887

# Assigning a type to the variable 'minkowski_distance_p' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'minkowski_distance_p', minkowski_distance_p)

@norecursion
def minkowski_distance(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_466888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 31), 'int')
    defaults = [int_466888]
    # Create a new context for function 'minkowski_distance'
    module_type_store = module_type_store.open_function_context('minkowski_distance', 49, 0, False)
    
    # Passed parameters checking function
    minkowski_distance.stypy_localization = localization
    minkowski_distance.stypy_type_of_self = None
    minkowski_distance.stypy_type_store = module_type_store
    minkowski_distance.stypy_function_name = 'minkowski_distance'
    minkowski_distance.stypy_param_names_list = ['x', 'y', 'p']
    minkowski_distance.stypy_varargs_param_name = None
    minkowski_distance.stypy_kwargs_param_name = None
    minkowski_distance.stypy_call_defaults = defaults
    minkowski_distance.stypy_call_varargs = varargs
    minkowski_distance.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'minkowski_distance', ['x', 'y', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'minkowski_distance', localization, ['x', 'y', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'minkowski_distance(...)' code ##################

    str_466889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', '\n    Compute the L**p distance between two arrays.\n\n    Parameters\n    ----------\n    x : (M, K) array_like\n        Input array.\n    y : (N, K) array_like\n        Input array.\n    p : float, 1 <= p <= infinity\n        Which Minkowski p-norm to use.\n\n    Examples\n    --------\n    >>> from scipy.spatial import minkowski_distance\n    >>> minkowski_distance([[0,0],[0,0]], [[1,1],[0,1]])\n    array([ 1.41421356,  1.        ])\n\n    ')
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to asarray(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'x' (line 69)
    x_466892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'x', False)
    # Processing the call keyword arguments (line 69)
    kwargs_466893 = {}
    # Getting the type of 'np' (line 69)
    np_466890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 69)
    asarray_466891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), np_466890, 'asarray')
    # Calling asarray(args, kwargs) (line 69)
    asarray_call_result_466894 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), asarray_466891, *[x_466892], **kwargs_466893)
    
    # Assigning a type to the variable 'x' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'x', asarray_call_result_466894)
    
    # Assigning a Call to a Name (line 70):
    
    # Assigning a Call to a Name (line 70):
    
    # Call to asarray(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'y' (line 70)
    y_466897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'y', False)
    # Processing the call keyword arguments (line 70)
    kwargs_466898 = {}
    # Getting the type of 'np' (line 70)
    np_466895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 70)
    asarray_466896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), np_466895, 'asarray')
    # Calling asarray(args, kwargs) (line 70)
    asarray_call_result_466899 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), asarray_466896, *[y_466897], **kwargs_466898)
    
    # Assigning a type to the variable 'y' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'y', asarray_call_result_466899)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'p' (line 71)
    p_466900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 7), 'p')
    # Getting the type of 'np' (line 71)
    np_466901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'np')
    # Obtaining the member 'inf' of a type (line 71)
    inf_466902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), np_466901, 'inf')
    # Applying the binary operator '==' (line 71)
    result_eq_466903 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 7), '==', p_466900, inf_466902)
    
    
    # Getting the type of 'p' (line 71)
    p_466904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'p')
    int_466905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 27), 'int')
    # Applying the binary operator '==' (line 71)
    result_eq_466906 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 22), '==', p_466904, int_466905)
    
    # Applying the binary operator 'or' (line 71)
    result_or_keyword_466907 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 7), 'or', result_eq_466903, result_eq_466906)
    
    # Testing the type of an if condition (line 71)
    if_condition_466908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), result_or_keyword_466907)
    # Assigning a type to the variable 'if_condition_466908' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'if_condition_466908', if_condition_466908)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to minkowski_distance_p(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'x' (line 72)
    x_466910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 36), 'x', False)
    # Getting the type of 'y' (line 72)
    y_466911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 39), 'y', False)
    # Getting the type of 'p' (line 72)
    p_466912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), 'p', False)
    # Processing the call keyword arguments (line 72)
    kwargs_466913 = {}
    # Getting the type of 'minkowski_distance_p' (line 72)
    minkowski_distance_p_466909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'minkowski_distance_p', False)
    # Calling minkowski_distance_p(args, kwargs) (line 72)
    minkowski_distance_p_call_result_466914 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), minkowski_distance_p_466909, *[x_466910, y_466911, p_466912], **kwargs_466913)
    
    # Assigning a type to the variable 'stypy_return_type' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', minkowski_distance_p_call_result_466914)
    # SSA branch for the else part of an if statement (line 71)
    module_type_store.open_ssa_branch('else')
    
    # Call to minkowski_distance_p(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'x' (line 74)
    x_466916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 36), 'x', False)
    # Getting the type of 'y' (line 74)
    y_466917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 39), 'y', False)
    # Getting the type of 'p' (line 74)
    p_466918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 42), 'p', False)
    # Processing the call keyword arguments (line 74)
    kwargs_466919 = {}
    # Getting the type of 'minkowski_distance_p' (line 74)
    minkowski_distance_p_466915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'minkowski_distance_p', False)
    # Calling minkowski_distance_p(args, kwargs) (line 74)
    minkowski_distance_p_call_result_466920 = invoke(stypy.reporting.localization.Localization(__file__, 74, 15), minkowski_distance_p_466915, *[x_466916, y_466917, p_466918], **kwargs_466919)
    
    float_466921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 47), 'float')
    # Getting the type of 'p' (line 74)
    p_466922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 50), 'p')
    # Applying the binary operator 'div' (line 74)
    result_div_466923 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 47), 'div', float_466921, p_466922)
    
    # Applying the binary operator '**' (line 74)
    result_pow_466924 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 15), '**', minkowski_distance_p_call_result_466920, result_div_466923)
    
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'stypy_return_type', result_pow_466924)
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'minkowski_distance(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'minkowski_distance' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_466925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_466925)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'minkowski_distance'
    return stypy_return_type_466925

# Assigning a type to the variable 'minkowski_distance' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'minkowski_distance', minkowski_distance)
# Declaration of the 'Rectangle' class

class Rectangle(object, ):
    str_466926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', 'Hyperrectangle class.\n\n    Represents a Cartesian product of intervals.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rectangle.__init__', ['maxes', 'mins'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['maxes', 'mins'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_466927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'str', 'Construct a hyperrectangle.')
        
        # Assigning a Call to a Attribute (line 84):
        
        # Assigning a Call to a Attribute (line 84):
        
        # Call to astype(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'float' (line 84)
        float_466935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 51), 'float', False)
        # Processing the call keyword arguments (line 84)
        kwargs_466936 = {}
        
        # Call to maximum(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'maxes' (line 84)
        maxes_466930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'maxes', False)
        # Getting the type of 'mins' (line 84)
        mins_466931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 38), 'mins', False)
        # Processing the call keyword arguments (line 84)
        kwargs_466932 = {}
        # Getting the type of 'np' (line 84)
        np_466928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'np', False)
        # Obtaining the member 'maximum' of a type (line 84)
        maximum_466929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), np_466928, 'maximum')
        # Calling maximum(args, kwargs) (line 84)
        maximum_call_result_466933 = invoke(stypy.reporting.localization.Localization(__file__, 84, 21), maximum_466929, *[maxes_466930, mins_466931], **kwargs_466932)
        
        # Obtaining the member 'astype' of a type (line 84)
        astype_466934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), maximum_call_result_466933, 'astype')
        # Calling astype(args, kwargs) (line 84)
        astype_call_result_466937 = invoke(stypy.reporting.localization.Localization(__file__, 84, 21), astype_466934, *[float_466935], **kwargs_466936)
        
        # Getting the type of 'self' (line 84)
        self_466938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self')
        # Setting the type of the member 'maxes' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_466938, 'maxes', astype_call_result_466937)
        
        # Assigning a Call to a Attribute (line 85):
        
        # Assigning a Call to a Attribute (line 85):
        
        # Call to astype(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'float' (line 85)
        float_466946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 50), 'float', False)
        # Processing the call keyword arguments (line 85)
        kwargs_466947 = {}
        
        # Call to minimum(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'maxes' (line 85)
        maxes_466941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 31), 'maxes', False)
        # Getting the type of 'mins' (line 85)
        mins_466942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 37), 'mins', False)
        # Processing the call keyword arguments (line 85)
        kwargs_466943 = {}
        # Getting the type of 'np' (line 85)
        np_466939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'np', False)
        # Obtaining the member 'minimum' of a type (line 85)
        minimum_466940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), np_466939, 'minimum')
        # Calling minimum(args, kwargs) (line 85)
        minimum_call_result_466944 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), minimum_466940, *[maxes_466941, mins_466942], **kwargs_466943)
        
        # Obtaining the member 'astype' of a type (line 85)
        astype_466945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), minimum_call_result_466944, 'astype')
        # Calling astype(args, kwargs) (line 85)
        astype_call_result_466948 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), astype_466945, *[float_466946], **kwargs_466947)
        
        # Getting the type of 'self' (line 85)
        self_466949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self')
        # Setting the type of the member 'mins' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_466949, 'mins', astype_call_result_466948)
        
        # Assigning a Attribute to a Tuple (line 86):
        
        # Assigning a Subscript to a Name (line 86):
        
        # Obtaining the type of the subscript
        int_466950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
        # Getting the type of 'self' (line 86)
        self_466951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'self')
        # Obtaining the member 'maxes' of a type (line 86)
        maxes_466952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 18), self_466951, 'maxes')
        # Obtaining the member 'shape' of a type (line 86)
        shape_466953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 18), maxes_466952, 'shape')
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___466954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), shape_466953, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_466955 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___466954, int_466950)
        
        # Assigning a type to the variable 'tuple_var_assignment_466760' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_466760', subscript_call_result_466955)
        
        # Assigning a Name to a Attribute (line 86):
        # Getting the type of 'tuple_var_assignment_466760' (line 86)
        tuple_var_assignment_466760_466956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_466760')
        # Getting the type of 'self' (line 86)
        self_466957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self')
        # Setting the type of the member 'm' of a type (line 86)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_466957, 'm', tuple_var_assignment_466760_466956)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rectangle.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Rectangle.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rectangle.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rectangle.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Rectangle.stypy__repr__')
        Rectangle.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Rectangle.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rectangle.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rectangle.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rectangle.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rectangle.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rectangle.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rectangle.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_466958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 15), 'str', '<Rectangle %s>')
        
        # Call to list(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to zip(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'self' (line 89)
        self_466961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 43), 'self', False)
        # Obtaining the member 'mins' of a type (line 89)
        mins_466962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 43), self_466961, 'mins')
        # Getting the type of 'self' (line 89)
        self_466963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 54), 'self', False)
        # Obtaining the member 'maxes' of a type (line 89)
        maxes_466964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 54), self_466963, 'maxes')
        # Processing the call keyword arguments (line 89)
        kwargs_466965 = {}
        # Getting the type of 'zip' (line 89)
        zip_466960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 39), 'zip', False)
        # Calling zip(args, kwargs) (line 89)
        zip_call_result_466966 = invoke(stypy.reporting.localization.Localization(__file__, 89, 39), zip_466960, *[mins_466962, maxes_466964], **kwargs_466965)
        
        # Processing the call keyword arguments (line 89)
        kwargs_466967 = {}
        # Getting the type of 'list' (line 89)
        list_466959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 34), 'list', False)
        # Calling list(args, kwargs) (line 89)
        list_call_result_466968 = invoke(stypy.reporting.localization.Localization(__file__, 89, 34), list_466959, *[zip_call_result_466966], **kwargs_466967)
        
        # Applying the binary operator '%' (line 89)
        result_mod_466969 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 15), '%', str_466958, list_call_result_466968)
        
        # Assigning a type to the variable 'stypy_return_type' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'stypy_return_type', result_mod_466969)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_466970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_466970)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_466970


    @norecursion
    def volume(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'volume'
        module_type_store = module_type_store.open_function_context('volume', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rectangle.volume.__dict__.__setitem__('stypy_localization', localization)
        Rectangle.volume.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rectangle.volume.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rectangle.volume.__dict__.__setitem__('stypy_function_name', 'Rectangle.volume')
        Rectangle.volume.__dict__.__setitem__('stypy_param_names_list', [])
        Rectangle.volume.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rectangle.volume.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rectangle.volume.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rectangle.volume.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rectangle.volume.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rectangle.volume.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rectangle.volume', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'volume', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'volume(...)' code ##################

        str_466971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'str', 'Total volume.')
        
        # Call to prod(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'self' (line 93)
        self_466974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'self', False)
        # Obtaining the member 'maxes' of a type (line 93)
        maxes_466975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 23), self_466974, 'maxes')
        # Getting the type of 'self' (line 93)
        self_466976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 34), 'self', False)
        # Obtaining the member 'mins' of a type (line 93)
        mins_466977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 34), self_466976, 'mins')
        # Applying the binary operator '-' (line 93)
        result_sub_466978 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 23), '-', maxes_466975, mins_466977)
        
        # Processing the call keyword arguments (line 93)
        kwargs_466979 = {}
        # Getting the type of 'np' (line 93)
        np_466972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'np', False)
        # Obtaining the member 'prod' of a type (line 93)
        prod_466973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 15), np_466972, 'prod')
        # Calling prod(args, kwargs) (line 93)
        prod_call_result_466980 = invoke(stypy.reporting.localization.Localization(__file__, 93, 15), prod_466973, *[result_sub_466978], **kwargs_466979)
        
        # Assigning a type to the variable 'stypy_return_type' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'stypy_return_type', prod_call_result_466980)
        
        # ################# End of 'volume(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'volume' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_466981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_466981)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'volume'
        return stypy_return_type_466981


    @norecursion
    def split(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'split'
        module_type_store = module_type_store.open_function_context('split', 95, 4, False)
        # Assigning a type to the variable 'self' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rectangle.split.__dict__.__setitem__('stypy_localization', localization)
        Rectangle.split.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rectangle.split.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rectangle.split.__dict__.__setitem__('stypy_function_name', 'Rectangle.split')
        Rectangle.split.__dict__.__setitem__('stypy_param_names_list', ['d', 'split'])
        Rectangle.split.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rectangle.split.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rectangle.split.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rectangle.split.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rectangle.split.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rectangle.split.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rectangle.split', ['d', 'split'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'split', localization, ['d', 'split'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'split(...)' code ##################

        str_466982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, (-1)), 'str', '\n        Produce two hyperrectangles by splitting.\n\n        In general, if you need to compute maximum and minimum\n        distances to the children, it can be done more efficiently\n        by updating the maximum and minimum distances to the parent.\n\n        Parameters\n        ----------\n        d : int\n            Axis to split hyperrectangle along.\n        split : float\n            Position along axis `d` to split at.\n\n        ')
        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to copy(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 111)
        self_466985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'self', False)
        # Obtaining the member 'maxes' of a type (line 111)
        maxes_466986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 22), self_466985, 'maxes')
        # Processing the call keyword arguments (line 111)
        kwargs_466987 = {}
        # Getting the type of 'np' (line 111)
        np_466983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'np', False)
        # Obtaining the member 'copy' of a type (line 111)
        copy_466984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 14), np_466983, 'copy')
        # Calling copy(args, kwargs) (line 111)
        copy_call_result_466988 = invoke(stypy.reporting.localization.Localization(__file__, 111, 14), copy_466984, *[maxes_466986], **kwargs_466987)
        
        # Assigning a type to the variable 'mid' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'mid', copy_call_result_466988)
        
        # Assigning a Name to a Subscript (line 112):
        
        # Assigning a Name to a Subscript (line 112):
        # Getting the type of 'split' (line 112)
        split_466989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'split')
        # Getting the type of 'mid' (line 112)
        mid_466990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'mid')
        # Getting the type of 'd' (line 112)
        d_466991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'd')
        # Storing an element on a container (line 112)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 8), mid_466990, (d_466991, split_466989))
        
        # Assigning a Call to a Name (line 113):
        
        # Assigning a Call to a Name (line 113):
        
        # Call to Rectangle(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'self' (line 113)
        self_466993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'self', False)
        # Obtaining the member 'mins' of a type (line 113)
        mins_466994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), self_466993, 'mins')
        # Getting the type of 'mid' (line 113)
        mid_466995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 36), 'mid', False)
        # Processing the call keyword arguments (line 113)
        kwargs_466996 = {}
        # Getting the type of 'Rectangle' (line 113)
        Rectangle_466992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 113)
        Rectangle_call_result_466997 = invoke(stypy.reporting.localization.Localization(__file__, 113, 15), Rectangle_466992, *[mins_466994, mid_466995], **kwargs_466996)
        
        # Assigning a type to the variable 'less' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'less', Rectangle_call_result_466997)
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to copy(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'self' (line 114)
        self_467000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'self', False)
        # Obtaining the member 'mins' of a type (line 114)
        mins_467001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 22), self_467000, 'mins')
        # Processing the call keyword arguments (line 114)
        kwargs_467002 = {}
        # Getting the type of 'np' (line 114)
        np_466998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'np', False)
        # Obtaining the member 'copy' of a type (line 114)
        copy_466999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 14), np_466998, 'copy')
        # Calling copy(args, kwargs) (line 114)
        copy_call_result_467003 = invoke(stypy.reporting.localization.Localization(__file__, 114, 14), copy_466999, *[mins_467001], **kwargs_467002)
        
        # Assigning a type to the variable 'mid' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'mid', copy_call_result_467003)
        
        # Assigning a Name to a Subscript (line 115):
        
        # Assigning a Name to a Subscript (line 115):
        # Getting the type of 'split' (line 115)
        split_467004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'split')
        # Getting the type of 'mid' (line 115)
        mid_467005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'mid')
        # Getting the type of 'd' (line 115)
        d_467006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'd')
        # Storing an element on a container (line 115)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 8), mid_467005, (d_467006, split_467004))
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to Rectangle(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'mid' (line 116)
        mid_467008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'mid', False)
        # Getting the type of 'self' (line 116)
        self_467009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 33), 'self', False)
        # Obtaining the member 'maxes' of a type (line 116)
        maxes_467010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 33), self_467009, 'maxes')
        # Processing the call keyword arguments (line 116)
        kwargs_467011 = {}
        # Getting the type of 'Rectangle' (line 116)
        Rectangle_467007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 116)
        Rectangle_call_result_467012 = invoke(stypy.reporting.localization.Localization(__file__, 116, 18), Rectangle_467007, *[mid_467008, maxes_467010], **kwargs_467011)
        
        # Assigning a type to the variable 'greater' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'greater', Rectangle_call_result_467012)
        
        # Obtaining an instance of the builtin type 'tuple' (line 117)
        tuple_467013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 117)
        # Adding element type (line 117)
        # Getting the type of 'less' (line 117)
        less_467014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'less')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 15), tuple_467013, less_467014)
        # Adding element type (line 117)
        # Getting the type of 'greater' (line 117)
        greater_467015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'greater')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 15), tuple_467013, greater_467015)
        
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', tuple_467013)
        
        # ################# End of 'split(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'split' in the type store
        # Getting the type of 'stypy_return_type' (line 95)
        stypy_return_type_467016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_467016)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'split'
        return stypy_return_type_467016


    @norecursion
    def min_distance_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_467017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 38), 'float')
        defaults = [float_467017]
        # Create a new context for function 'min_distance_point'
        module_type_store = module_type_store.open_function_context('min_distance_point', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rectangle.min_distance_point.__dict__.__setitem__('stypy_localization', localization)
        Rectangle.min_distance_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rectangle.min_distance_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rectangle.min_distance_point.__dict__.__setitem__('stypy_function_name', 'Rectangle.min_distance_point')
        Rectangle.min_distance_point.__dict__.__setitem__('stypy_param_names_list', ['x', 'p'])
        Rectangle.min_distance_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rectangle.min_distance_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rectangle.min_distance_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rectangle.min_distance_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rectangle.min_distance_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rectangle.min_distance_point.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rectangle.min_distance_point', ['x', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'min_distance_point', localization, ['x', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'min_distance_point(...)' code ##################

        str_467018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, (-1)), 'str', '\n        Return the minimum distance between input and points in the hyperrectangle.\n\n        Parameters\n        ----------\n        x : array_like\n            Input.\n        p : float, optional\n            Input.\n\n        ')
        
        # Call to minkowski_distance(...): (line 131)
        # Processing the call arguments (line 131)
        int_467020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 34), 'int')
        
        # Call to maximum(...): (line 131)
        # Processing the call arguments (line 131)
        int_467023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 48), 'int')
        
        # Call to maximum(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'self' (line 131)
        self_467026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 61), 'self', False)
        # Obtaining the member 'mins' of a type (line 131)
        mins_467027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 61), self_467026, 'mins')
        # Getting the type of 'x' (line 131)
        x_467028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 71), 'x', False)
        # Applying the binary operator '-' (line 131)
        result_sub_467029 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 61), '-', mins_467027, x_467028)
        
        # Getting the type of 'x' (line 131)
        x_467030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 73), 'x', False)
        # Getting the type of 'self' (line 131)
        self_467031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 75), 'self', False)
        # Obtaining the member 'maxes' of a type (line 131)
        maxes_467032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 75), self_467031, 'maxes')
        # Applying the binary operator '-' (line 131)
        result_sub_467033 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 73), '-', x_467030, maxes_467032)
        
        # Processing the call keyword arguments (line 131)
        kwargs_467034 = {}
        # Getting the type of 'np' (line 131)
        np_467024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 50), 'np', False)
        # Obtaining the member 'maximum' of a type (line 131)
        maximum_467025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 50), np_467024, 'maximum')
        # Calling maximum(args, kwargs) (line 131)
        maximum_call_result_467035 = invoke(stypy.reporting.localization.Localization(__file__, 131, 50), maximum_467025, *[result_sub_467029, result_sub_467033], **kwargs_467034)
        
        # Processing the call keyword arguments (line 131)
        kwargs_467036 = {}
        # Getting the type of 'np' (line 131)
        np_467021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 37), 'np', False)
        # Obtaining the member 'maximum' of a type (line 131)
        maximum_467022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 37), np_467021, 'maximum')
        # Calling maximum(args, kwargs) (line 131)
        maximum_call_result_467037 = invoke(stypy.reporting.localization.Localization(__file__, 131, 37), maximum_467022, *[int_467023, maximum_call_result_467035], **kwargs_467036)
        
        # Getting the type of 'p' (line 131)
        p_467038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 88), 'p', False)
        # Processing the call keyword arguments (line 131)
        kwargs_467039 = {}
        # Getting the type of 'minkowski_distance' (line 131)
        minkowski_distance_467019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'minkowski_distance', False)
        # Calling minkowski_distance(args, kwargs) (line 131)
        minkowski_distance_call_result_467040 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), minkowski_distance_467019, *[int_467020, maximum_call_result_467037, p_467038], **kwargs_467039)
        
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', minkowski_distance_call_result_467040)
        
        # ################# End of 'min_distance_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'min_distance_point' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_467041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_467041)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'min_distance_point'
        return stypy_return_type_467041


    @norecursion
    def max_distance_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_467042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 38), 'float')
        defaults = [float_467042]
        # Create a new context for function 'max_distance_point'
        module_type_store = module_type_store.open_function_context('max_distance_point', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rectangle.max_distance_point.__dict__.__setitem__('stypy_localization', localization)
        Rectangle.max_distance_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rectangle.max_distance_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rectangle.max_distance_point.__dict__.__setitem__('stypy_function_name', 'Rectangle.max_distance_point')
        Rectangle.max_distance_point.__dict__.__setitem__('stypy_param_names_list', ['x', 'p'])
        Rectangle.max_distance_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rectangle.max_distance_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rectangle.max_distance_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rectangle.max_distance_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rectangle.max_distance_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rectangle.max_distance_point.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rectangle.max_distance_point', ['x', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'max_distance_point', localization, ['x', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'max_distance_point(...)' code ##################

        str_467043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'str', '\n        Return the maximum distance between input and points in the hyperrectangle.\n\n        Parameters\n        ----------\n        x : array_like\n            Input array.\n        p : float, optional\n            Input.\n\n        ')
        
        # Call to minkowski_distance(...): (line 145)
        # Processing the call arguments (line 145)
        int_467045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 34), 'int')
        
        # Call to maximum(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'self' (line 145)
        self_467048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 48), 'self', False)
        # Obtaining the member 'maxes' of a type (line 145)
        maxes_467049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 48), self_467048, 'maxes')
        # Getting the type of 'x' (line 145)
        x_467050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 59), 'x', False)
        # Applying the binary operator '-' (line 145)
        result_sub_467051 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 48), '-', maxes_467049, x_467050)
        
        # Getting the type of 'x' (line 145)
        x_467052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 61), 'x', False)
        # Getting the type of 'self' (line 145)
        self_467053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 63), 'self', False)
        # Obtaining the member 'mins' of a type (line 145)
        mins_467054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 63), self_467053, 'mins')
        # Applying the binary operator '-' (line 145)
        result_sub_467055 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 61), '-', x_467052, mins_467054)
        
        # Processing the call keyword arguments (line 145)
        kwargs_467056 = {}
        # Getting the type of 'np' (line 145)
        np_467046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 37), 'np', False)
        # Obtaining the member 'maximum' of a type (line 145)
        maximum_467047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 37), np_467046, 'maximum')
        # Calling maximum(args, kwargs) (line 145)
        maximum_call_result_467057 = invoke(stypy.reporting.localization.Localization(__file__, 145, 37), maximum_467047, *[result_sub_467051, result_sub_467055], **kwargs_467056)
        
        # Getting the type of 'p' (line 145)
        p_467058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 74), 'p', False)
        # Processing the call keyword arguments (line 145)
        kwargs_467059 = {}
        # Getting the type of 'minkowski_distance' (line 145)
        minkowski_distance_467044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'minkowski_distance', False)
        # Calling minkowski_distance(args, kwargs) (line 145)
        minkowski_distance_call_result_467060 = invoke(stypy.reporting.localization.Localization(__file__, 145, 15), minkowski_distance_467044, *[int_467045, maximum_call_result_467057, p_467058], **kwargs_467059)
        
        # Assigning a type to the variable 'stypy_return_type' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'stypy_return_type', minkowski_distance_call_result_467060)
        
        # ################# End of 'max_distance_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'max_distance_point' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_467061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_467061)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'max_distance_point'
        return stypy_return_type_467061


    @norecursion
    def min_distance_rectangle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_467062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 46), 'float')
        defaults = [float_467062]
        # Create a new context for function 'min_distance_rectangle'
        module_type_store = module_type_store.open_function_context('min_distance_rectangle', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rectangle.min_distance_rectangle.__dict__.__setitem__('stypy_localization', localization)
        Rectangle.min_distance_rectangle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rectangle.min_distance_rectangle.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rectangle.min_distance_rectangle.__dict__.__setitem__('stypy_function_name', 'Rectangle.min_distance_rectangle')
        Rectangle.min_distance_rectangle.__dict__.__setitem__('stypy_param_names_list', ['other', 'p'])
        Rectangle.min_distance_rectangle.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rectangle.min_distance_rectangle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rectangle.min_distance_rectangle.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rectangle.min_distance_rectangle.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rectangle.min_distance_rectangle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rectangle.min_distance_rectangle.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rectangle.min_distance_rectangle', ['other', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'min_distance_rectangle', localization, ['other', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'min_distance_rectangle(...)' code ##################

        str_467063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'str', '\n        Compute the minimum distance between points in the two hyperrectangles.\n\n        Parameters\n        ----------\n        other : hyperrectangle\n            Input.\n        p : float\n            Input.\n\n        ')
        
        # Call to minkowski_distance(...): (line 159)
        # Processing the call arguments (line 159)
        int_467065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 34), 'int')
        
        # Call to maximum(...): (line 159)
        # Processing the call arguments (line 159)
        int_467068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 48), 'int')
        
        # Call to maximum(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'self' (line 159)
        self_467071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 61), 'self', False)
        # Obtaining the member 'mins' of a type (line 159)
        mins_467072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 61), self_467071, 'mins')
        # Getting the type of 'other' (line 159)
        other_467073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 71), 'other', False)
        # Obtaining the member 'maxes' of a type (line 159)
        maxes_467074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 71), other_467073, 'maxes')
        # Applying the binary operator '-' (line 159)
        result_sub_467075 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 61), '-', mins_467072, maxes_467074)
        
        # Getting the type of 'other' (line 159)
        other_467076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 83), 'other', False)
        # Obtaining the member 'mins' of a type (line 159)
        mins_467077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 83), other_467076, 'mins')
        # Getting the type of 'self' (line 159)
        self_467078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 94), 'self', False)
        # Obtaining the member 'maxes' of a type (line 159)
        maxes_467079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 94), self_467078, 'maxes')
        # Applying the binary operator '-' (line 159)
        result_sub_467080 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 83), '-', mins_467077, maxes_467079)
        
        # Processing the call keyword arguments (line 159)
        kwargs_467081 = {}
        # Getting the type of 'np' (line 159)
        np_467069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 50), 'np', False)
        # Obtaining the member 'maximum' of a type (line 159)
        maximum_467070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 50), np_467069, 'maximum')
        # Calling maximum(args, kwargs) (line 159)
        maximum_call_result_467082 = invoke(stypy.reporting.localization.Localization(__file__, 159, 50), maximum_467070, *[result_sub_467075, result_sub_467080], **kwargs_467081)
        
        # Processing the call keyword arguments (line 159)
        kwargs_467083 = {}
        # Getting the type of 'np' (line 159)
        np_467066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 37), 'np', False)
        # Obtaining the member 'maximum' of a type (line 159)
        maximum_467067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 37), np_467066, 'maximum')
        # Calling maximum(args, kwargs) (line 159)
        maximum_call_result_467084 = invoke(stypy.reporting.localization.Localization(__file__, 159, 37), maximum_467067, *[int_467068, maximum_call_result_467082], **kwargs_467083)
        
        # Getting the type of 'p' (line 159)
        p_467085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 107), 'p', False)
        # Processing the call keyword arguments (line 159)
        kwargs_467086 = {}
        # Getting the type of 'minkowski_distance' (line 159)
        minkowski_distance_467064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'minkowski_distance', False)
        # Calling minkowski_distance(args, kwargs) (line 159)
        minkowski_distance_call_result_467087 = invoke(stypy.reporting.localization.Localization(__file__, 159, 15), minkowski_distance_467064, *[int_467065, maximum_call_result_467084, p_467085], **kwargs_467086)
        
        # Assigning a type to the variable 'stypy_return_type' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type', minkowski_distance_call_result_467087)
        
        # ################# End of 'min_distance_rectangle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'min_distance_rectangle' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_467088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_467088)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'min_distance_rectangle'
        return stypy_return_type_467088


    @norecursion
    def max_distance_rectangle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_467089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 46), 'float')
        defaults = [float_467089]
        # Create a new context for function 'max_distance_rectangle'
        module_type_store = module_type_store.open_function_context('max_distance_rectangle', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rectangle.max_distance_rectangle.__dict__.__setitem__('stypy_localization', localization)
        Rectangle.max_distance_rectangle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rectangle.max_distance_rectangle.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rectangle.max_distance_rectangle.__dict__.__setitem__('stypy_function_name', 'Rectangle.max_distance_rectangle')
        Rectangle.max_distance_rectangle.__dict__.__setitem__('stypy_param_names_list', ['other', 'p'])
        Rectangle.max_distance_rectangle.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rectangle.max_distance_rectangle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rectangle.max_distance_rectangle.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rectangle.max_distance_rectangle.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rectangle.max_distance_rectangle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rectangle.max_distance_rectangle.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rectangle.max_distance_rectangle', ['other', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'max_distance_rectangle', localization, ['other', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'max_distance_rectangle(...)' code ##################

        str_467090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, (-1)), 'str', '\n        Compute the maximum distance between points in the two hyperrectangles.\n\n        Parameters\n        ----------\n        other : hyperrectangle\n            Input.\n        p : float, optional\n            Input.\n\n        ')
        
        # Call to minkowski_distance(...): (line 173)
        # Processing the call arguments (line 173)
        int_467092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 34), 'int')
        
        # Call to maximum(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'self' (line 173)
        self_467095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 48), 'self', False)
        # Obtaining the member 'maxes' of a type (line 173)
        maxes_467096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 48), self_467095, 'maxes')
        # Getting the type of 'other' (line 173)
        other_467097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 59), 'other', False)
        # Obtaining the member 'mins' of a type (line 173)
        mins_467098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 59), other_467097, 'mins')
        # Applying the binary operator '-' (line 173)
        result_sub_467099 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 48), '-', maxes_467096, mins_467098)
        
        # Getting the type of 'other' (line 173)
        other_467100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 70), 'other', False)
        # Obtaining the member 'maxes' of a type (line 173)
        maxes_467101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 70), other_467100, 'maxes')
        # Getting the type of 'self' (line 173)
        self_467102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 82), 'self', False)
        # Obtaining the member 'mins' of a type (line 173)
        mins_467103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 82), self_467102, 'mins')
        # Applying the binary operator '-' (line 173)
        result_sub_467104 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 70), '-', maxes_467101, mins_467103)
        
        # Processing the call keyword arguments (line 173)
        kwargs_467105 = {}
        # Getting the type of 'np' (line 173)
        np_467093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 37), 'np', False)
        # Obtaining the member 'maximum' of a type (line 173)
        maximum_467094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 37), np_467093, 'maximum')
        # Calling maximum(args, kwargs) (line 173)
        maximum_call_result_467106 = invoke(stypy.reporting.localization.Localization(__file__, 173, 37), maximum_467094, *[result_sub_467099, result_sub_467104], **kwargs_467105)
        
        # Getting the type of 'p' (line 173)
        p_467107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 93), 'p', False)
        # Processing the call keyword arguments (line 173)
        kwargs_467108 = {}
        # Getting the type of 'minkowski_distance' (line 173)
        minkowski_distance_467091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'minkowski_distance', False)
        # Calling minkowski_distance(args, kwargs) (line 173)
        minkowski_distance_call_result_467109 = invoke(stypy.reporting.localization.Localization(__file__, 173, 15), minkowski_distance_467091, *[int_467092, maximum_call_result_467106, p_467107], **kwargs_467108)
        
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', minkowski_distance_call_result_467109)
        
        # ################# End of 'max_distance_rectangle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'max_distance_rectangle' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_467110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_467110)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'max_distance_rectangle'
        return stypy_return_type_467110


# Assigning a type to the variable 'Rectangle' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'Rectangle', Rectangle)
# Declaration of the 'KDTree' class

class KDTree(object, ):
    str_467111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, (-1)), 'str', '\n    kd-tree for quick nearest-neighbor lookup\n\n    This class provides an index into a set of k-dimensional points which\n    can be used to rapidly look up the nearest neighbors of any point.\n\n    Parameters\n    ----------\n    data : (N,K) array_like\n        The data points to be indexed. This array is not copied, and\n        so modifying this data will result in bogus results.\n    leafsize : int, optional\n        The number of points at which the algorithm switches over to\n        brute-force.  Has to be positive.\n\n    Raises\n    ------\n    RuntimeError\n        The maximum recursion limit can be exceeded for large data\n        sets.  If this happens, either increase the value for the `leafsize`\n        parameter or increase the recursion limit by::\n\n            >>> import sys\n            >>> sys.setrecursionlimit(10000)\n\n    See Also\n    --------\n    cKDTree : Implementation of `KDTree` in Cython\n\n    Notes\n    -----\n    The algorithm used is described in Maneewongvatana and Mount 1999.\n    The general idea is that the kd-tree is a binary tree, each of whose\n    nodes represents an axis-aligned hyperrectangle. Each node specifies\n    an axis and splits the set of points based on whether their coordinate\n    along that axis is greater than or less than a particular value.\n\n    During construction, the axis and splitting point are chosen by the\n    "sliding midpoint" rule, which ensures that the cells do not all\n    become long and thin.\n\n    The tree can be queried for the r closest neighbors of any given point\n    (optionally returning only those within some maximum distance of the\n    point). It can also be queried, with a substantial gain in efficiency,\n    for the r approximate closest neighbors.\n\n    For large dimensions (20 is already large) do not expect this to run\n    significantly faster than brute force. High-dimensional nearest-neighbor\n    queries are a substantial open problem in computer science.\n\n    The tree also supports all-neighbors queries, both with arrays of points\n    and with other kd-trees. These do use a reasonably efficient algorithm,\n    but the kd-tree is not necessarily the best data structure for this\n    sort of calculation.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_467112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 38), 'int')
        defaults = [int_467112]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KDTree.__init__', ['data', 'leafsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data', 'leafsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 234):
        
        # Assigning a Call to a Attribute (line 234):
        
        # Call to asarray(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'data' (line 234)
        data_467115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 31), 'data', False)
        # Processing the call keyword arguments (line 234)
        kwargs_467116 = {}
        # Getting the type of 'np' (line 234)
        np_467113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'np', False)
        # Obtaining the member 'asarray' of a type (line 234)
        asarray_467114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 20), np_467113, 'asarray')
        # Calling asarray(args, kwargs) (line 234)
        asarray_call_result_467117 = invoke(stypy.reporting.localization.Localization(__file__, 234, 20), asarray_467114, *[data_467115], **kwargs_467116)
        
        # Getting the type of 'self' (line 234)
        self_467118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'self')
        # Setting the type of the member 'data' of a type (line 234)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), self_467118, 'data', asarray_call_result_467117)
        
        # Assigning a Call to a Tuple (line 235):
        
        # Assigning a Subscript to a Name (line 235):
        
        # Obtaining the type of the subscript
        int_467119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 8), 'int')
        
        # Call to shape(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'self' (line 235)
        self_467122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 34), 'self', False)
        # Obtaining the member 'data' of a type (line 235)
        data_467123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 34), self_467122, 'data')
        # Processing the call keyword arguments (line 235)
        kwargs_467124 = {}
        # Getting the type of 'np' (line 235)
        np_467120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'np', False)
        # Obtaining the member 'shape' of a type (line 235)
        shape_467121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 25), np_467120, 'shape')
        # Calling shape(args, kwargs) (line 235)
        shape_call_result_467125 = invoke(stypy.reporting.localization.Localization(__file__, 235, 25), shape_467121, *[data_467123], **kwargs_467124)
        
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___467126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), shape_call_result_467125, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_467127 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), getitem___467126, int_467119)
        
        # Assigning a type to the variable 'tuple_var_assignment_466761' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_466761', subscript_call_result_467127)
        
        # Assigning a Subscript to a Name (line 235):
        
        # Obtaining the type of the subscript
        int_467128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 8), 'int')
        
        # Call to shape(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'self' (line 235)
        self_467131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 34), 'self', False)
        # Obtaining the member 'data' of a type (line 235)
        data_467132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 34), self_467131, 'data')
        # Processing the call keyword arguments (line 235)
        kwargs_467133 = {}
        # Getting the type of 'np' (line 235)
        np_467129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'np', False)
        # Obtaining the member 'shape' of a type (line 235)
        shape_467130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 25), np_467129, 'shape')
        # Calling shape(args, kwargs) (line 235)
        shape_call_result_467134 = invoke(stypy.reporting.localization.Localization(__file__, 235, 25), shape_467130, *[data_467132], **kwargs_467133)
        
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___467135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), shape_call_result_467134, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_467136 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), getitem___467135, int_467128)
        
        # Assigning a type to the variable 'tuple_var_assignment_466762' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_466762', subscript_call_result_467136)
        
        # Assigning a Name to a Attribute (line 235):
        # Getting the type of 'tuple_var_assignment_466761' (line 235)
        tuple_var_assignment_466761_467137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_466761')
        # Getting the type of 'self' (line 235)
        self_467138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self')
        # Setting the type of the member 'n' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_467138, 'n', tuple_var_assignment_466761_467137)
        
        # Assigning a Name to a Attribute (line 235):
        # Getting the type of 'tuple_var_assignment_466762' (line 235)
        tuple_var_assignment_466762_467139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_466762')
        # Getting the type of 'self' (line 235)
        self_467140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'self')
        # Setting the type of the member 'm' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 16), self_467140, 'm', tuple_var_assignment_466762_467139)
        
        # Assigning a Call to a Attribute (line 236):
        
        # Assigning a Call to a Attribute (line 236):
        
        # Call to int(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'leafsize' (line 236)
        leafsize_467142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 28), 'leafsize', False)
        # Processing the call keyword arguments (line 236)
        kwargs_467143 = {}
        # Getting the type of 'int' (line 236)
        int_467141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 24), 'int', False)
        # Calling int(args, kwargs) (line 236)
        int_call_result_467144 = invoke(stypy.reporting.localization.Localization(__file__, 236, 24), int_467141, *[leafsize_467142], **kwargs_467143)
        
        # Getting the type of 'self' (line 236)
        self_467145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self')
        # Setting the type of the member 'leafsize' of a type (line 236)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_467145, 'leafsize', int_call_result_467144)
        
        
        # Getting the type of 'self' (line 237)
        self_467146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'self')
        # Obtaining the member 'leafsize' of a type (line 237)
        leafsize_467147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 11), self_467146, 'leafsize')
        int_467148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 27), 'int')
        # Applying the binary operator '<' (line 237)
        result_lt_467149 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 11), '<', leafsize_467147, int_467148)
        
        # Testing the type of an if condition (line 237)
        if_condition_467150 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 8), result_lt_467149)
        # Assigning a type to the variable 'if_condition_467150' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'if_condition_467150', if_condition_467150)
        # SSA begins for if statement (line 237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 238)
        # Processing the call arguments (line 238)
        str_467152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 29), 'str', 'leafsize must be at least 1')
        # Processing the call keyword arguments (line 238)
        kwargs_467153 = {}
        # Getting the type of 'ValueError' (line 238)
        ValueError_467151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 238)
        ValueError_call_result_467154 = invoke(stypy.reporting.localization.Localization(__file__, 238, 18), ValueError_467151, *[str_467152], **kwargs_467153)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 238, 12), ValueError_call_result_467154, 'raise parameter', BaseException)
        # SSA join for if statement (line 237)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 239):
        
        # Assigning a Call to a Attribute (line 239):
        
        # Call to amax(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'self' (line 239)
        self_467157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 29), 'self', False)
        # Obtaining the member 'data' of a type (line 239)
        data_467158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 29), self_467157, 'data')
        # Processing the call keyword arguments (line 239)
        int_467159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 44), 'int')
        keyword_467160 = int_467159
        kwargs_467161 = {'axis': keyword_467160}
        # Getting the type of 'np' (line 239)
        np_467155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 21), 'np', False)
        # Obtaining the member 'amax' of a type (line 239)
        amax_467156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 21), np_467155, 'amax')
        # Calling amax(args, kwargs) (line 239)
        amax_call_result_467162 = invoke(stypy.reporting.localization.Localization(__file__, 239, 21), amax_467156, *[data_467158], **kwargs_467161)
        
        # Getting the type of 'self' (line 239)
        self_467163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'self')
        # Setting the type of the member 'maxes' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), self_467163, 'maxes', amax_call_result_467162)
        
        # Assigning a Call to a Attribute (line 240):
        
        # Assigning a Call to a Attribute (line 240):
        
        # Call to amin(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'self' (line 240)
        self_467166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 28), 'self', False)
        # Obtaining the member 'data' of a type (line 240)
        data_467167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 28), self_467166, 'data')
        # Processing the call keyword arguments (line 240)
        int_467168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 43), 'int')
        keyword_467169 = int_467168
        kwargs_467170 = {'axis': keyword_467169}
        # Getting the type of 'np' (line 240)
        np_467164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'np', False)
        # Obtaining the member 'amin' of a type (line 240)
        amin_467165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 20), np_467164, 'amin')
        # Calling amin(args, kwargs) (line 240)
        amin_call_result_467171 = invoke(stypy.reporting.localization.Localization(__file__, 240, 20), amin_467165, *[data_467167], **kwargs_467170)
        
        # Getting the type of 'self' (line 240)
        self_467172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'self')
        # Setting the type of the member 'mins' of a type (line 240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), self_467172, 'mins', amin_call_result_467171)
        
        # Assigning a Call to a Attribute (line 242):
        
        # Assigning a Call to a Attribute (line 242):
        
        # Call to __build(...): (line 242)
        # Processing the call arguments (line 242)
        
        # Call to arange(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'self' (line 242)
        self_467177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 43), 'self', False)
        # Obtaining the member 'n' of a type (line 242)
        n_467178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 43), self_467177, 'n')
        # Processing the call keyword arguments (line 242)
        kwargs_467179 = {}
        # Getting the type of 'np' (line 242)
        np_467175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 33), 'np', False)
        # Obtaining the member 'arange' of a type (line 242)
        arange_467176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 33), np_467175, 'arange')
        # Calling arange(args, kwargs) (line 242)
        arange_call_result_467180 = invoke(stypy.reporting.localization.Localization(__file__, 242, 33), arange_467176, *[n_467178], **kwargs_467179)
        
        # Getting the type of 'self' (line 242)
        self_467181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 52), 'self', False)
        # Obtaining the member 'maxes' of a type (line 242)
        maxes_467182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 52), self_467181, 'maxes')
        # Getting the type of 'self' (line 242)
        self_467183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 64), 'self', False)
        # Obtaining the member 'mins' of a type (line 242)
        mins_467184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 64), self_467183, 'mins')
        # Processing the call keyword arguments (line 242)
        kwargs_467185 = {}
        # Getting the type of 'self' (line 242)
        self_467173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'self', False)
        # Obtaining the member '__build' of a type (line 242)
        build_467174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 20), self_467173, '__build')
        # Calling __build(args, kwargs) (line 242)
        build_call_result_467186 = invoke(stypy.reporting.localization.Localization(__file__, 242, 20), build_467174, *[arange_call_result_467180, maxes_467182, mins_467184], **kwargs_467185)
        
        # Getting the type of 'self' (line 242)
        self_467187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self')
        # Setting the type of the member 'tree' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_467187, 'tree', build_call_result_467186)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()

    # Declaration of the 'node' class

    class node(object, ):
        
        
        
        # Obtaining the type of the subscript
        int_467188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 28), 'int')
        # Getting the type of 'sys' (line 245)
        sys_467189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 11), 'sys')
        # Obtaining the member 'version_info' of a type (line 245)
        version_info_467190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 11), sys_467189, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___467191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 11), version_info_467190, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_467192 = invoke(stypy.reporting.localization.Localization(__file__, 245, 11), getitem___467191, int_467188)
        
        int_467193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 34), 'int')
        # Applying the binary operator '>=' (line 245)
        result_ge_467194 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 11), '>=', subscript_call_result_467192, int_467193)
        
        # Testing the type of an if condition (line 245)
        if_condition_467195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 8), result_ge_467194)
        # Assigning a type to the variable 'if_condition_467195' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'if_condition_467195', if_condition_467195)
        # SSA begins for if statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def __lt__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__lt__'
            module_type_store = module_type_store.open_function_context('__lt__', 246, 12, False)
            # Assigning a type to the variable 'self' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'self', type_of_self)
            
            # Passed parameters checking function
            node.__lt__.__dict__.__setitem__('stypy_localization', localization)
            node.__lt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            node.__lt__.__dict__.__setitem__('stypy_type_store', module_type_store)
            node.__lt__.__dict__.__setitem__('stypy_function_name', 'node.__lt__')
            node.__lt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            node.__lt__.__dict__.__setitem__('stypy_varargs_param_name', None)
            node.__lt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            node.__lt__.__dict__.__setitem__('stypy_call_defaults', defaults)
            node.__lt__.__dict__.__setitem__('stypy_call_varargs', varargs)
            node.__lt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            node.__lt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'node.__lt__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__lt__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__lt__(...)' code ##################

            
            
            # Call to id(...): (line 247)
            # Processing the call arguments (line 247)
            # Getting the type of 'self' (line 247)
            self_467197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 26), 'self', False)
            # Processing the call keyword arguments (line 247)
            kwargs_467198 = {}
            # Getting the type of 'id' (line 247)
            id_467196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'id', False)
            # Calling id(args, kwargs) (line 247)
            id_call_result_467199 = invoke(stypy.reporting.localization.Localization(__file__, 247, 23), id_467196, *[self_467197], **kwargs_467198)
            
            
            # Call to id(...): (line 247)
            # Processing the call arguments (line 247)
            # Getting the type of 'other' (line 247)
            other_467201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 37), 'other', False)
            # Processing the call keyword arguments (line 247)
            kwargs_467202 = {}
            # Getting the type of 'id' (line 247)
            id_467200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 34), 'id', False)
            # Calling id(args, kwargs) (line 247)
            id_call_result_467203 = invoke(stypy.reporting.localization.Localization(__file__, 247, 34), id_467200, *[other_467201], **kwargs_467202)
            
            # Applying the binary operator '<' (line 247)
            result_lt_467204 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 23), '<', id_call_result_467199, id_call_result_467203)
            
            # Assigning a type to the variable 'stypy_return_type' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'stypy_return_type', result_lt_467204)
            
            # ################# End of '__lt__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__lt__' in the type store
            # Getting the type of 'stypy_return_type' (line 246)
            stypy_return_type_467205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_467205)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__lt__'
            return stypy_return_type_467205


        @norecursion
        def __gt__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__gt__'
            module_type_store = module_type_store.open_function_context('__gt__', 249, 12, False)
            # Assigning a type to the variable 'self' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'self', type_of_self)
            
            # Passed parameters checking function
            node.__gt__.__dict__.__setitem__('stypy_localization', localization)
            node.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            node.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
            node.__gt__.__dict__.__setitem__('stypy_function_name', 'node.__gt__')
            node.__gt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            node.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
            node.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            node.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
            node.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
            node.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            node.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'node.__gt__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__gt__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__gt__(...)' code ##################

            
            
            # Call to id(...): (line 250)
            # Processing the call arguments (line 250)
            # Getting the type of 'self' (line 250)
            self_467207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 26), 'self', False)
            # Processing the call keyword arguments (line 250)
            kwargs_467208 = {}
            # Getting the type of 'id' (line 250)
            id_467206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 23), 'id', False)
            # Calling id(args, kwargs) (line 250)
            id_call_result_467209 = invoke(stypy.reporting.localization.Localization(__file__, 250, 23), id_467206, *[self_467207], **kwargs_467208)
            
            
            # Call to id(...): (line 250)
            # Processing the call arguments (line 250)
            # Getting the type of 'other' (line 250)
            other_467211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 37), 'other', False)
            # Processing the call keyword arguments (line 250)
            kwargs_467212 = {}
            # Getting the type of 'id' (line 250)
            id_467210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 34), 'id', False)
            # Calling id(args, kwargs) (line 250)
            id_call_result_467213 = invoke(stypy.reporting.localization.Localization(__file__, 250, 34), id_467210, *[other_467211], **kwargs_467212)
            
            # Applying the binary operator '>' (line 250)
            result_gt_467214 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 23), '>', id_call_result_467209, id_call_result_467213)
            
            # Assigning a type to the variable 'stypy_return_type' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'stypy_return_type', result_gt_467214)
            
            # ################# End of '__gt__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__gt__' in the type store
            # Getting the type of 'stypy_return_type' (line 249)
            stypy_return_type_467215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_467215)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__gt__'
            return stypy_return_type_467215


        @norecursion
        def __le__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__le__'
            module_type_store = module_type_store.open_function_context('__le__', 252, 12, False)
            # Assigning a type to the variable 'self' (line 253)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'self', type_of_self)
            
            # Passed parameters checking function
            node.__le__.__dict__.__setitem__('stypy_localization', localization)
            node.__le__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            node.__le__.__dict__.__setitem__('stypy_type_store', module_type_store)
            node.__le__.__dict__.__setitem__('stypy_function_name', 'node.__le__')
            node.__le__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            node.__le__.__dict__.__setitem__('stypy_varargs_param_name', None)
            node.__le__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            node.__le__.__dict__.__setitem__('stypy_call_defaults', defaults)
            node.__le__.__dict__.__setitem__('stypy_call_varargs', varargs)
            node.__le__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            node.__le__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'node.__le__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__le__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__le__(...)' code ##################

            
            
            # Call to id(...): (line 253)
            # Processing the call arguments (line 253)
            # Getting the type of 'self' (line 253)
            self_467217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'self', False)
            # Processing the call keyword arguments (line 253)
            kwargs_467218 = {}
            # Getting the type of 'id' (line 253)
            id_467216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 23), 'id', False)
            # Calling id(args, kwargs) (line 253)
            id_call_result_467219 = invoke(stypy.reporting.localization.Localization(__file__, 253, 23), id_467216, *[self_467217], **kwargs_467218)
            
            
            # Call to id(...): (line 253)
            # Processing the call arguments (line 253)
            # Getting the type of 'other' (line 253)
            other_467221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 38), 'other', False)
            # Processing the call keyword arguments (line 253)
            kwargs_467222 = {}
            # Getting the type of 'id' (line 253)
            id_467220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 35), 'id', False)
            # Calling id(args, kwargs) (line 253)
            id_call_result_467223 = invoke(stypy.reporting.localization.Localization(__file__, 253, 35), id_467220, *[other_467221], **kwargs_467222)
            
            # Applying the binary operator '<=' (line 253)
            result_le_467224 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 23), '<=', id_call_result_467219, id_call_result_467223)
            
            # Assigning a type to the variable 'stypy_return_type' (line 253)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'stypy_return_type', result_le_467224)
            
            # ################# End of '__le__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__le__' in the type store
            # Getting the type of 'stypy_return_type' (line 252)
            stypy_return_type_467225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_467225)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__le__'
            return stypy_return_type_467225


        @norecursion
        def __ge__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__ge__'
            module_type_store = module_type_store.open_function_context('__ge__', 255, 12, False)
            # Assigning a type to the variable 'self' (line 256)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'self', type_of_self)
            
            # Passed parameters checking function
            node.__ge__.__dict__.__setitem__('stypy_localization', localization)
            node.__ge__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            node.__ge__.__dict__.__setitem__('stypy_type_store', module_type_store)
            node.__ge__.__dict__.__setitem__('stypy_function_name', 'node.__ge__')
            node.__ge__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            node.__ge__.__dict__.__setitem__('stypy_varargs_param_name', None)
            node.__ge__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            node.__ge__.__dict__.__setitem__('stypy_call_defaults', defaults)
            node.__ge__.__dict__.__setitem__('stypy_call_varargs', varargs)
            node.__ge__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            node.__ge__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'node.__ge__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__ge__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__ge__(...)' code ##################

            
            
            # Call to id(...): (line 256)
            # Processing the call arguments (line 256)
            # Getting the type of 'self' (line 256)
            self_467227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 26), 'self', False)
            # Processing the call keyword arguments (line 256)
            kwargs_467228 = {}
            # Getting the type of 'id' (line 256)
            id_467226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 'id', False)
            # Calling id(args, kwargs) (line 256)
            id_call_result_467229 = invoke(stypy.reporting.localization.Localization(__file__, 256, 23), id_467226, *[self_467227], **kwargs_467228)
            
            
            # Call to id(...): (line 256)
            # Processing the call arguments (line 256)
            # Getting the type of 'other' (line 256)
            other_467231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 38), 'other', False)
            # Processing the call keyword arguments (line 256)
            kwargs_467232 = {}
            # Getting the type of 'id' (line 256)
            id_467230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 35), 'id', False)
            # Calling id(args, kwargs) (line 256)
            id_call_result_467233 = invoke(stypy.reporting.localization.Localization(__file__, 256, 35), id_467230, *[other_467231], **kwargs_467232)
            
            # Applying the binary operator '>=' (line 256)
            result_ge_467234 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 23), '>=', id_call_result_467229, id_call_result_467233)
            
            # Assigning a type to the variable 'stypy_return_type' (line 256)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'stypy_return_type', result_ge_467234)
            
            # ################# End of '__ge__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__ge__' in the type store
            # Getting the type of 'stypy_return_type' (line 255)
            stypy_return_type_467235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_467235)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__ge__'
            return stypy_return_type_467235


        @norecursion
        def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__eq__'
            module_type_store = module_type_store.open_function_context('__eq__', 258, 12, False)
            # Assigning a type to the variable 'self' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'self', type_of_self)
            
            # Passed parameters checking function
            node.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
            node.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            node.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
            node.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'node.stypy__eq__')
            node.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            node.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
            node.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            node.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
            node.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
            node.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            node.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'node.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__eq__(...)' code ##################

            
            
            # Call to id(...): (line 259)
            # Processing the call arguments (line 259)
            # Getting the type of 'self' (line 259)
            self_467237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 26), 'self', False)
            # Processing the call keyword arguments (line 259)
            kwargs_467238 = {}
            # Getting the type of 'id' (line 259)
            id_467236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 23), 'id', False)
            # Calling id(args, kwargs) (line 259)
            id_call_result_467239 = invoke(stypy.reporting.localization.Localization(__file__, 259, 23), id_467236, *[self_467237], **kwargs_467238)
            
            
            # Call to id(...): (line 259)
            # Processing the call arguments (line 259)
            # Getting the type of 'other' (line 259)
            other_467241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 38), 'other', False)
            # Processing the call keyword arguments (line 259)
            kwargs_467242 = {}
            # Getting the type of 'id' (line 259)
            id_467240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 35), 'id', False)
            # Calling id(args, kwargs) (line 259)
            id_call_result_467243 = invoke(stypy.reporting.localization.Localization(__file__, 259, 35), id_467240, *[other_467241], **kwargs_467242)
            
            # Applying the binary operator '==' (line 259)
            result_eq_467244 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 23), '==', id_call_result_467239, id_call_result_467243)
            
            # Assigning a type to the variable 'stypy_return_type' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'stypy_return_type', result_eq_467244)
            
            # ################# End of '__eq__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__eq__' in the type store
            # Getting the type of 'stypy_return_type' (line 258)
            stypy_return_type_467245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_467245)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__eq__'
            return stypy_return_type_467245

        # SSA join for if statement (line 245)
        module_type_store = module_type_store.join_ssa_context()
        
    
    # Assigning a type to the variable 'node' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'node', node)
    # Declaration of the 'leafnode' class
    # Getting the type of 'node' (line 261)
    node_467246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'node')

    class leafnode(node_467246, ):

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 262, 8, False)
            # Assigning a type to the variable 'self' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'leafnode.__init__', ['idx'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['idx'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Assigning a Name to a Attribute (line 263):
            
            # Assigning a Name to a Attribute (line 263):
            # Getting the type of 'idx' (line 263)
            idx_467247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 23), 'idx')
            # Getting the type of 'self' (line 263)
            self_467248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'self')
            # Setting the type of the member 'idx' of a type (line 263)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 12), self_467248, 'idx', idx_467247)
            
            # Assigning a Call to a Attribute (line 264):
            
            # Assigning a Call to a Attribute (line 264):
            
            # Call to len(...): (line 264)
            # Processing the call arguments (line 264)
            # Getting the type of 'idx' (line 264)
            idx_467250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 32), 'idx', False)
            # Processing the call keyword arguments (line 264)
            kwargs_467251 = {}
            # Getting the type of 'len' (line 264)
            len_467249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 28), 'len', False)
            # Calling len(args, kwargs) (line 264)
            len_call_result_467252 = invoke(stypy.reporting.localization.Localization(__file__, 264, 28), len_467249, *[idx_467250], **kwargs_467251)
            
            # Getting the type of 'self' (line 264)
            self_467253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'self')
            # Setting the type of the member 'children' of a type (line 264)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), self_467253, 'children', len_call_result_467252)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'leafnode' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'leafnode', leafnode)
    # Declaration of the 'innernode' class
    # Getting the type of 'node' (line 266)
    node_467254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 'node')

    class innernode(node_467254, ):

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 267, 8, False)
            # Assigning a type to the variable 'self' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'innernode.__init__', ['split_dim', 'split', 'less', 'greater'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['split_dim', 'split', 'less', 'greater'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Assigning a Name to a Attribute (line 268):
            
            # Assigning a Name to a Attribute (line 268):
            # Getting the type of 'split_dim' (line 268)
            split_dim_467255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 29), 'split_dim')
            # Getting the type of 'self' (line 268)
            self_467256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'self')
            # Setting the type of the member 'split_dim' of a type (line 268)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 12), self_467256, 'split_dim', split_dim_467255)
            
            # Assigning a Name to a Attribute (line 269):
            
            # Assigning a Name to a Attribute (line 269):
            # Getting the type of 'split' (line 269)
            split_467257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 25), 'split')
            # Getting the type of 'self' (line 269)
            self_467258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'self')
            # Setting the type of the member 'split' of a type (line 269)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), self_467258, 'split', split_467257)
            
            # Assigning a Name to a Attribute (line 270):
            
            # Assigning a Name to a Attribute (line 270):
            # Getting the type of 'less' (line 270)
            less_467259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 24), 'less')
            # Getting the type of 'self' (line 270)
            self_467260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'self')
            # Setting the type of the member 'less' of a type (line 270)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 12), self_467260, 'less', less_467259)
            
            # Assigning a Name to a Attribute (line 271):
            
            # Assigning a Name to a Attribute (line 271):
            # Getting the type of 'greater' (line 271)
            greater_467261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 27), 'greater')
            # Getting the type of 'self' (line 271)
            self_467262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'self')
            # Setting the type of the member 'greater' of a type (line 271)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), self_467262, 'greater', greater_467261)
            
            # Assigning a BinOp to a Attribute (line 272):
            
            # Assigning a BinOp to a Attribute (line 272):
            # Getting the type of 'less' (line 272)
            less_467263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 28), 'less')
            # Obtaining the member 'children' of a type (line 272)
            children_467264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 28), less_467263, 'children')
            # Getting the type of 'greater' (line 272)
            greater_467265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 42), 'greater')
            # Obtaining the member 'children' of a type (line 272)
            children_467266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 42), greater_467265, 'children')
            # Applying the binary operator '+' (line 272)
            result_add_467267 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 28), '+', children_467264, children_467266)
            
            # Getting the type of 'self' (line 272)
            self_467268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'self')
            # Setting the type of the member 'children' of a type (line 272)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 12), self_467268, 'children', result_add_467267)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'innernode' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'innernode', innernode)

    @norecursion
    def __build(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__build'
        module_type_store = module_type_store.open_function_context('__build', 274, 4, False)
        # Assigning a type to the variable 'self' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KDTree.__build.__dict__.__setitem__('stypy_localization', localization)
        KDTree.__build.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KDTree.__build.__dict__.__setitem__('stypy_type_store', module_type_store)
        KDTree.__build.__dict__.__setitem__('stypy_function_name', 'KDTree.__build')
        KDTree.__build.__dict__.__setitem__('stypy_param_names_list', ['idx', 'maxes', 'mins'])
        KDTree.__build.__dict__.__setitem__('stypy_varargs_param_name', None)
        KDTree.__build.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KDTree.__build.__dict__.__setitem__('stypy_call_defaults', defaults)
        KDTree.__build.__dict__.__setitem__('stypy_call_varargs', varargs)
        KDTree.__build.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KDTree.__build.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KDTree.__build', ['idx', 'maxes', 'mins'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__build', localization, ['idx', 'maxes', 'mins'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__build(...)' code ##################

        
        
        
        # Call to len(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'idx' (line 275)
        idx_467270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'idx', False)
        # Processing the call keyword arguments (line 275)
        kwargs_467271 = {}
        # Getting the type of 'len' (line 275)
        len_467269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'len', False)
        # Calling len(args, kwargs) (line 275)
        len_call_result_467272 = invoke(stypy.reporting.localization.Localization(__file__, 275, 11), len_467269, *[idx_467270], **kwargs_467271)
        
        # Getting the type of 'self' (line 275)
        self_467273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 23), 'self')
        # Obtaining the member 'leafsize' of a type (line 275)
        leafsize_467274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 23), self_467273, 'leafsize')
        # Applying the binary operator '<=' (line 275)
        result_le_467275 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 11), '<=', len_call_result_467272, leafsize_467274)
        
        # Testing the type of an if condition (line 275)
        if_condition_467276 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 8), result_le_467275)
        # Assigning a type to the variable 'if_condition_467276' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'if_condition_467276', if_condition_467276)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to leafnode(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'idx' (line 276)
        idx_467279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 35), 'idx', False)
        # Processing the call keyword arguments (line 276)
        kwargs_467280 = {}
        # Getting the type of 'KDTree' (line 276)
        KDTree_467277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 19), 'KDTree', False)
        # Obtaining the member 'leafnode' of a type (line 276)
        leafnode_467278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 19), KDTree_467277, 'leafnode')
        # Calling leafnode(args, kwargs) (line 276)
        leafnode_call_result_467281 = invoke(stypy.reporting.localization.Localization(__file__, 276, 19), leafnode_467278, *[idx_467279], **kwargs_467280)
        
        # Assigning a type to the variable 'stypy_return_type' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'stypy_return_type', leafnode_call_result_467281)
        # SSA branch for the else part of an if statement (line 275)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 278):
        
        # Assigning a Subscript to a Name (line 278):
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 278)
        idx_467282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 29), 'idx')
        # Getting the type of 'self' (line 278)
        self_467283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'self')
        # Obtaining the member 'data' of a type (line 278)
        data_467284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 19), self_467283, 'data')
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___467285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 19), data_467284, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_467286 = invoke(stypy.reporting.localization.Localization(__file__, 278, 19), getitem___467285, idx_467282)
        
        # Assigning a type to the variable 'data' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'data', subscript_call_result_467286)
        
        # Assigning a Call to a Name (line 281):
        
        # Assigning a Call to a Name (line 281):
        
        # Call to argmax(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'maxes' (line 281)
        maxes_467289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 26), 'maxes', False)
        # Getting the type of 'mins' (line 281)
        mins_467290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 32), 'mins', False)
        # Applying the binary operator '-' (line 281)
        result_sub_467291 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 26), '-', maxes_467289, mins_467290)
        
        # Processing the call keyword arguments (line 281)
        kwargs_467292 = {}
        # Getting the type of 'np' (line 281)
        np_467287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'np', False)
        # Obtaining the member 'argmax' of a type (line 281)
        argmax_467288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), np_467287, 'argmax')
        # Calling argmax(args, kwargs) (line 281)
        argmax_call_result_467293 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), argmax_467288, *[result_sub_467291], **kwargs_467292)
        
        # Assigning a type to the variable 'd' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'd', argmax_call_result_467293)
        
        # Assigning a Subscript to a Name (line 282):
        
        # Assigning a Subscript to a Name (line 282):
        
        # Obtaining the type of the subscript
        # Getting the type of 'd' (line 282)
        d_467294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 27), 'd')
        # Getting the type of 'maxes' (line 282)
        maxes_467295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 21), 'maxes')
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___467296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 21), maxes_467295, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_467297 = invoke(stypy.reporting.localization.Localization(__file__, 282, 21), getitem___467296, d_467294)
        
        # Assigning a type to the variable 'maxval' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'maxval', subscript_call_result_467297)
        
        # Assigning a Subscript to a Name (line 283):
        
        # Assigning a Subscript to a Name (line 283):
        
        # Obtaining the type of the subscript
        # Getting the type of 'd' (line 283)
        d_467298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 26), 'd')
        # Getting the type of 'mins' (line 283)
        mins_467299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 21), 'mins')
        # Obtaining the member '__getitem__' of a type (line 283)
        getitem___467300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 21), mins_467299, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 283)
        subscript_call_result_467301 = invoke(stypy.reporting.localization.Localization(__file__, 283, 21), getitem___467300, d_467298)
        
        # Assigning a type to the variable 'minval' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'minval', subscript_call_result_467301)
        
        
        # Getting the type of 'maxval' (line 284)
        maxval_467302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 15), 'maxval')
        # Getting the type of 'minval' (line 284)
        minval_467303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 'minval')
        # Applying the binary operator '==' (line 284)
        result_eq_467304 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 15), '==', maxval_467302, minval_467303)
        
        # Testing the type of an if condition (line 284)
        if_condition_467305 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 12), result_eq_467304)
        # Assigning a type to the variable 'if_condition_467305' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'if_condition_467305', if_condition_467305)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to leafnode(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'idx' (line 286)
        idx_467308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 39), 'idx', False)
        # Processing the call keyword arguments (line 286)
        kwargs_467309 = {}
        # Getting the type of 'KDTree' (line 286)
        KDTree_467306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 23), 'KDTree', False)
        # Obtaining the member 'leafnode' of a type (line 286)
        leafnode_467307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 23), KDTree_467306, 'leafnode')
        # Calling leafnode(args, kwargs) (line 286)
        leafnode_call_result_467310 = invoke(stypy.reporting.localization.Localization(__file__, 286, 23), leafnode_467307, *[idx_467308], **kwargs_467309)
        
        # Assigning a type to the variable 'stypy_return_type' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'stypy_return_type', leafnode_call_result_467310)
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 287):
        
        # Assigning a Subscript to a Name (line 287):
        
        # Obtaining the type of the subscript
        slice_467311 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 287, 19), None, None, None)
        # Getting the type of 'd' (line 287)
        d_467312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 26), 'd')
        # Getting the type of 'data' (line 287)
        data_467313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'data')
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___467314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 19), data_467313, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_467315 = invoke(stypy.reporting.localization.Localization(__file__, 287, 19), getitem___467314, (slice_467311, d_467312))
        
        # Assigning a type to the variable 'data' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'data', subscript_call_result_467315)
        
        # Assigning a BinOp to a Name (line 291):
        
        # Assigning a BinOp to a Name (line 291):
        # Getting the type of 'maxval' (line 291)
        maxval_467316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 21), 'maxval')
        # Getting the type of 'minval' (line 291)
        minval_467317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 28), 'minval')
        # Applying the binary operator '+' (line 291)
        result_add_467318 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 21), '+', maxval_467316, minval_467317)
        
        int_467319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 36), 'int')
        # Applying the binary operator 'div' (line 291)
        result_div_467320 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 20), 'div', result_add_467318, int_467319)
        
        # Assigning a type to the variable 'split' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'split', result_div_467320)
        
        # Assigning a Subscript to a Name (line 292):
        
        # Assigning a Subscript to a Name (line 292):
        
        # Obtaining the type of the subscript
        int_467321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 49), 'int')
        
        # Call to nonzero(...): (line 292)
        # Processing the call arguments (line 292)
        
        # Getting the type of 'data' (line 292)
        data_467324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 34), 'data', False)
        # Getting the type of 'split' (line 292)
        split_467325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 42), 'split', False)
        # Applying the binary operator '<=' (line 292)
        result_le_467326 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 34), '<=', data_467324, split_467325)
        
        # Processing the call keyword arguments (line 292)
        kwargs_467327 = {}
        # Getting the type of 'np' (line 292)
        np_467322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 23), 'np', False)
        # Obtaining the member 'nonzero' of a type (line 292)
        nonzero_467323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 23), np_467322, 'nonzero')
        # Calling nonzero(args, kwargs) (line 292)
        nonzero_call_result_467328 = invoke(stypy.reporting.localization.Localization(__file__, 292, 23), nonzero_467323, *[result_le_467326], **kwargs_467327)
        
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___467329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 23), nonzero_call_result_467328, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_467330 = invoke(stypy.reporting.localization.Localization(__file__, 292, 23), getitem___467329, int_467321)
        
        # Assigning a type to the variable 'less_idx' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'less_idx', subscript_call_result_467330)
        
        # Assigning a Subscript to a Name (line 293):
        
        # Assigning a Subscript to a Name (line 293):
        
        # Obtaining the type of the subscript
        int_467331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 51), 'int')
        
        # Call to nonzero(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Getting the type of 'data' (line 293)
        data_467334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 37), 'data', False)
        # Getting the type of 'split' (line 293)
        split_467335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 44), 'split', False)
        # Applying the binary operator '>' (line 293)
        result_gt_467336 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 37), '>', data_467334, split_467335)
        
        # Processing the call keyword arguments (line 293)
        kwargs_467337 = {}
        # Getting the type of 'np' (line 293)
        np_467332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'np', False)
        # Obtaining the member 'nonzero' of a type (line 293)
        nonzero_467333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 26), np_467332, 'nonzero')
        # Calling nonzero(args, kwargs) (line 293)
        nonzero_call_result_467338 = invoke(stypy.reporting.localization.Localization(__file__, 293, 26), nonzero_467333, *[result_gt_467336], **kwargs_467337)
        
        # Obtaining the member '__getitem__' of a type (line 293)
        getitem___467339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 26), nonzero_call_result_467338, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 293)
        subscript_call_result_467340 = invoke(stypy.reporting.localization.Localization(__file__, 293, 26), getitem___467339, int_467331)
        
        # Assigning a type to the variable 'greater_idx' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'greater_idx', subscript_call_result_467340)
        
        
        
        # Call to len(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'less_idx' (line 294)
        less_idx_467342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 'less_idx', False)
        # Processing the call keyword arguments (line 294)
        kwargs_467343 = {}
        # Getting the type of 'len' (line 294)
        len_467341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'len', False)
        # Calling len(args, kwargs) (line 294)
        len_call_result_467344 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), len_467341, *[less_idx_467342], **kwargs_467343)
        
        int_467345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 32), 'int')
        # Applying the binary operator '==' (line 294)
        result_eq_467346 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 15), '==', len_call_result_467344, int_467345)
        
        # Testing the type of an if condition (line 294)
        if_condition_467347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 12), result_eq_467346)
        # Assigning a type to the variable 'if_condition_467347' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'if_condition_467347', if_condition_467347)
        # SSA begins for if statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to amin(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'data' (line 295)
        data_467350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 32), 'data', False)
        # Processing the call keyword arguments (line 295)
        kwargs_467351 = {}
        # Getting the type of 'np' (line 295)
        np_467348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'np', False)
        # Obtaining the member 'amin' of a type (line 295)
        amin_467349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 24), np_467348, 'amin')
        # Calling amin(args, kwargs) (line 295)
        amin_call_result_467352 = invoke(stypy.reporting.localization.Localization(__file__, 295, 24), amin_467349, *[data_467350], **kwargs_467351)
        
        # Assigning a type to the variable 'split' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'split', amin_call_result_467352)
        
        # Assigning a Subscript to a Name (line 296):
        
        # Assigning a Subscript to a Name (line 296):
        
        # Obtaining the type of the subscript
        int_467353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 53), 'int')
        
        # Call to nonzero(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Getting the type of 'data' (line 296)
        data_467356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 38), 'data', False)
        # Getting the type of 'split' (line 296)
        split_467357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 46), 'split', False)
        # Applying the binary operator '<=' (line 296)
        result_le_467358 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 38), '<=', data_467356, split_467357)
        
        # Processing the call keyword arguments (line 296)
        kwargs_467359 = {}
        # Getting the type of 'np' (line 296)
        np_467354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 27), 'np', False)
        # Obtaining the member 'nonzero' of a type (line 296)
        nonzero_467355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 27), np_467354, 'nonzero')
        # Calling nonzero(args, kwargs) (line 296)
        nonzero_call_result_467360 = invoke(stypy.reporting.localization.Localization(__file__, 296, 27), nonzero_467355, *[result_le_467358], **kwargs_467359)
        
        # Obtaining the member '__getitem__' of a type (line 296)
        getitem___467361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 27), nonzero_call_result_467360, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 296)
        subscript_call_result_467362 = invoke(stypy.reporting.localization.Localization(__file__, 296, 27), getitem___467361, int_467353)
        
        # Assigning a type to the variable 'less_idx' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'less_idx', subscript_call_result_467362)
        
        # Assigning a Subscript to a Name (line 297):
        
        # Assigning a Subscript to a Name (line 297):
        
        # Obtaining the type of the subscript
        int_467363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 55), 'int')
        
        # Call to nonzero(...): (line 297)
        # Processing the call arguments (line 297)
        
        # Getting the type of 'data' (line 297)
        data_467366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 41), 'data', False)
        # Getting the type of 'split' (line 297)
        split_467367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 48), 'split', False)
        # Applying the binary operator '>' (line 297)
        result_gt_467368 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 41), '>', data_467366, split_467367)
        
        # Processing the call keyword arguments (line 297)
        kwargs_467369 = {}
        # Getting the type of 'np' (line 297)
        np_467364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 30), 'np', False)
        # Obtaining the member 'nonzero' of a type (line 297)
        nonzero_467365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 30), np_467364, 'nonzero')
        # Calling nonzero(args, kwargs) (line 297)
        nonzero_call_result_467370 = invoke(stypy.reporting.localization.Localization(__file__, 297, 30), nonzero_467365, *[result_gt_467368], **kwargs_467369)
        
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___467371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 30), nonzero_call_result_467370, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_467372 = invoke(stypy.reporting.localization.Localization(__file__, 297, 30), getitem___467371, int_467363)
        
        # Assigning a type to the variable 'greater_idx' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'greater_idx', subscript_call_result_467372)
        # SSA join for if statement (line 294)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'greater_idx' (line 298)
        greater_idx_467374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 19), 'greater_idx', False)
        # Processing the call keyword arguments (line 298)
        kwargs_467375 = {}
        # Getting the type of 'len' (line 298)
        len_467373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'len', False)
        # Calling len(args, kwargs) (line 298)
        len_call_result_467376 = invoke(stypy.reporting.localization.Localization(__file__, 298, 15), len_467373, *[greater_idx_467374], **kwargs_467375)
        
        int_467377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 35), 'int')
        # Applying the binary operator '==' (line 298)
        result_eq_467378 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 15), '==', len_call_result_467376, int_467377)
        
        # Testing the type of an if condition (line 298)
        if_condition_467379 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 12), result_eq_467378)
        # Assigning a type to the variable 'if_condition_467379' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'if_condition_467379', if_condition_467379)
        # SSA begins for if statement (line 298)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to amax(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'data' (line 299)
        data_467382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 32), 'data', False)
        # Processing the call keyword arguments (line 299)
        kwargs_467383 = {}
        # Getting the type of 'np' (line 299)
        np_467380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'np', False)
        # Obtaining the member 'amax' of a type (line 299)
        amax_467381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 24), np_467380, 'amax')
        # Calling amax(args, kwargs) (line 299)
        amax_call_result_467384 = invoke(stypy.reporting.localization.Localization(__file__, 299, 24), amax_467381, *[data_467382], **kwargs_467383)
        
        # Assigning a type to the variable 'split' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'split', amax_call_result_467384)
        
        # Assigning a Subscript to a Name (line 300):
        
        # Assigning a Subscript to a Name (line 300):
        
        # Obtaining the type of the subscript
        int_467385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 52), 'int')
        
        # Call to nonzero(...): (line 300)
        # Processing the call arguments (line 300)
        
        # Getting the type of 'data' (line 300)
        data_467388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 38), 'data', False)
        # Getting the type of 'split' (line 300)
        split_467389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 45), 'split', False)
        # Applying the binary operator '<' (line 300)
        result_lt_467390 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 38), '<', data_467388, split_467389)
        
        # Processing the call keyword arguments (line 300)
        kwargs_467391 = {}
        # Getting the type of 'np' (line 300)
        np_467386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 'np', False)
        # Obtaining the member 'nonzero' of a type (line 300)
        nonzero_467387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 27), np_467386, 'nonzero')
        # Calling nonzero(args, kwargs) (line 300)
        nonzero_call_result_467392 = invoke(stypy.reporting.localization.Localization(__file__, 300, 27), nonzero_467387, *[result_lt_467390], **kwargs_467391)
        
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___467393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 27), nonzero_call_result_467392, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
        subscript_call_result_467394 = invoke(stypy.reporting.localization.Localization(__file__, 300, 27), getitem___467393, int_467385)
        
        # Assigning a type to the variable 'less_idx' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'less_idx', subscript_call_result_467394)
        
        # Assigning a Subscript to a Name (line 301):
        
        # Assigning a Subscript to a Name (line 301):
        
        # Obtaining the type of the subscript
        int_467395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 56), 'int')
        
        # Call to nonzero(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Getting the type of 'data' (line 301)
        data_467398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 41), 'data', False)
        # Getting the type of 'split' (line 301)
        split_467399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 49), 'split', False)
        # Applying the binary operator '>=' (line 301)
        result_ge_467400 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 41), '>=', data_467398, split_467399)
        
        # Processing the call keyword arguments (line 301)
        kwargs_467401 = {}
        # Getting the type of 'np' (line 301)
        np_467396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 30), 'np', False)
        # Obtaining the member 'nonzero' of a type (line 301)
        nonzero_467397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 30), np_467396, 'nonzero')
        # Calling nonzero(args, kwargs) (line 301)
        nonzero_call_result_467402 = invoke(stypy.reporting.localization.Localization(__file__, 301, 30), nonzero_467397, *[result_ge_467400], **kwargs_467401)
        
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___467403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 30), nonzero_call_result_467402, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 301)
        subscript_call_result_467404 = invoke(stypy.reporting.localization.Localization(__file__, 301, 30), getitem___467403, int_467395)
        
        # Assigning a type to the variable 'greater_idx' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'greater_idx', subscript_call_result_467404)
        # SSA join for if statement (line 298)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'less_idx' (line 302)
        less_idx_467406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 19), 'less_idx', False)
        # Processing the call keyword arguments (line 302)
        kwargs_467407 = {}
        # Getting the type of 'len' (line 302)
        len_467405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'len', False)
        # Calling len(args, kwargs) (line 302)
        len_call_result_467408 = invoke(stypy.reporting.localization.Localization(__file__, 302, 15), len_467405, *[less_idx_467406], **kwargs_467407)
        
        int_467409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 32), 'int')
        # Applying the binary operator '==' (line 302)
        result_eq_467410 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 15), '==', len_call_result_467408, int_467409)
        
        # Testing the type of an if condition (line 302)
        if_condition_467411 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 12), result_eq_467410)
        # Assigning a type to the variable 'if_condition_467411' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'if_condition_467411', if_condition_467411)
        # SSA begins for if statement (line 302)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to all(...): (line 304)
        # Processing the call arguments (line 304)
        
        # Getting the type of 'data' (line 304)
        data_467414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 30), 'data', False)
        
        # Obtaining the type of the subscript
        int_467415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 43), 'int')
        # Getting the type of 'data' (line 304)
        data_467416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 38), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___467417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 38), data_467416, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_467418 = invoke(stypy.reporting.localization.Localization(__file__, 304, 38), getitem___467417, int_467415)
        
        # Applying the binary operator '==' (line 304)
        result_eq_467419 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 30), '==', data_467414, subscript_call_result_467418)
        
        # Processing the call keyword arguments (line 304)
        kwargs_467420 = {}
        # Getting the type of 'np' (line 304)
        np_467412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 23), 'np', False)
        # Obtaining the member 'all' of a type (line 304)
        all_467413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 23), np_467412, 'all')
        # Calling all(args, kwargs) (line 304)
        all_call_result_467421 = invoke(stypy.reporting.localization.Localization(__file__, 304, 23), all_467413, *[result_eq_467419], **kwargs_467420)
        
        # Applying the 'not' unary operator (line 304)
        result_not__467422 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 19), 'not', all_call_result_467421)
        
        # Testing the type of an if condition (line 304)
        if_condition_467423 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 16), result_not__467422)
        # Assigning a type to the variable 'if_condition_467423' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'if_condition_467423', if_condition_467423)
        # SSA begins for if statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 305)
        # Processing the call arguments (line 305)
        str_467425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 37), 'str', 'Troublesome data array: %s')
        # Getting the type of 'data' (line 305)
        data_467426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 68), 'data', False)
        # Applying the binary operator '%' (line 305)
        result_mod_467427 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 37), '%', str_467425, data_467426)
        
        # Processing the call keyword arguments (line 305)
        kwargs_467428 = {}
        # Getting the type of 'ValueError' (line 305)
        ValueError_467424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 305)
        ValueError_call_result_467429 = invoke(stypy.reporting.localization.Localization(__file__, 305, 26), ValueError_467424, *[result_mod_467427], **kwargs_467428)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 305, 20), ValueError_call_result_467429, 'raise parameter', BaseException)
        # SSA join for if statement (line 304)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 306):
        
        # Assigning a Subscript to a Name (line 306):
        
        # Obtaining the type of the subscript
        int_467430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 29), 'int')
        # Getting the type of 'data' (line 306)
        data_467431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 24), 'data')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___467432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 24), data_467431, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_467433 = invoke(stypy.reporting.localization.Localization(__file__, 306, 24), getitem___467432, int_467430)
        
        # Assigning a type to the variable 'split' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'split', subscript_call_result_467433)
        
        # Assigning a Call to a Name (line 307):
        
        # Assigning a Call to a Name (line 307):
        
        # Call to arange(...): (line 307)
        # Processing the call arguments (line 307)
        
        # Call to len(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'data' (line 307)
        data_467437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 41), 'data', False)
        # Processing the call keyword arguments (line 307)
        kwargs_467438 = {}
        # Getting the type of 'len' (line 307)
        len_467436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 37), 'len', False)
        # Calling len(args, kwargs) (line 307)
        len_call_result_467439 = invoke(stypy.reporting.localization.Localization(__file__, 307, 37), len_467436, *[data_467437], **kwargs_467438)
        
        int_467440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 47), 'int')
        # Applying the binary operator '-' (line 307)
        result_sub_467441 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 37), '-', len_call_result_467439, int_467440)
        
        # Processing the call keyword arguments (line 307)
        kwargs_467442 = {}
        # Getting the type of 'np' (line 307)
        np_467434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 27), 'np', False)
        # Obtaining the member 'arange' of a type (line 307)
        arange_467435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 27), np_467434, 'arange')
        # Calling arange(args, kwargs) (line 307)
        arange_call_result_467443 = invoke(stypy.reporting.localization.Localization(__file__, 307, 27), arange_467435, *[result_sub_467441], **kwargs_467442)
        
        # Assigning a type to the variable 'less_idx' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'less_idx', arange_call_result_467443)
        
        # Assigning a Call to a Name (line 308):
        
        # Assigning a Call to a Name (line 308):
        
        # Call to array(...): (line 308)
        # Processing the call arguments (line 308)
        
        # Obtaining an instance of the builtin type 'list' (line 308)
        list_467446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 308)
        # Adding element type (line 308)
        
        # Call to len(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'data' (line 308)
        data_467448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 44), 'data', False)
        # Processing the call keyword arguments (line 308)
        kwargs_467449 = {}
        # Getting the type of 'len' (line 308)
        len_467447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 40), 'len', False)
        # Calling len(args, kwargs) (line 308)
        len_call_result_467450 = invoke(stypy.reporting.localization.Localization(__file__, 308, 40), len_467447, *[data_467448], **kwargs_467449)
        
        int_467451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 50), 'int')
        # Applying the binary operator '-' (line 308)
        result_sub_467452 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 40), '-', len_call_result_467450, int_467451)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 39), list_467446, result_sub_467452)
        
        # Processing the call keyword arguments (line 308)
        kwargs_467453 = {}
        # Getting the type of 'np' (line 308)
        np_467444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 30), 'np', False)
        # Obtaining the member 'array' of a type (line 308)
        array_467445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 30), np_467444, 'array')
        # Calling array(args, kwargs) (line 308)
        array_call_result_467454 = invoke(stypy.reporting.localization.Localization(__file__, 308, 30), array_467445, *[list_467446], **kwargs_467453)
        
        # Assigning a type to the variable 'greater_idx' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'greater_idx', array_call_result_467454)
        # SSA join for if statement (line 302)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 310):
        
        # Assigning a Call to a Name (line 310):
        
        # Call to copy(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'maxes' (line 310)
        maxes_467457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 32), 'maxes', False)
        # Processing the call keyword arguments (line 310)
        kwargs_467458 = {}
        # Getting the type of 'np' (line 310)
        np_467455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 24), 'np', False)
        # Obtaining the member 'copy' of a type (line 310)
        copy_467456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 24), np_467455, 'copy')
        # Calling copy(args, kwargs) (line 310)
        copy_call_result_467459 = invoke(stypy.reporting.localization.Localization(__file__, 310, 24), copy_467456, *[maxes_467457], **kwargs_467458)
        
        # Assigning a type to the variable 'lessmaxes' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'lessmaxes', copy_call_result_467459)
        
        # Assigning a Name to a Subscript (line 311):
        
        # Assigning a Name to a Subscript (line 311):
        # Getting the type of 'split' (line 311)
        split_467460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 27), 'split')
        # Getting the type of 'lessmaxes' (line 311)
        lessmaxes_467461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'lessmaxes')
        # Getting the type of 'd' (line 311)
        d_467462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 22), 'd')
        # Storing an element on a container (line 311)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 12), lessmaxes_467461, (d_467462, split_467460))
        
        # Assigning a Call to a Name (line 312):
        
        # Assigning a Call to a Name (line 312):
        
        # Call to copy(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'mins' (line 312)
        mins_467465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 34), 'mins', False)
        # Processing the call keyword arguments (line 312)
        kwargs_467466 = {}
        # Getting the type of 'np' (line 312)
        np_467463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 26), 'np', False)
        # Obtaining the member 'copy' of a type (line 312)
        copy_467464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 26), np_467463, 'copy')
        # Calling copy(args, kwargs) (line 312)
        copy_call_result_467467 = invoke(stypy.reporting.localization.Localization(__file__, 312, 26), copy_467464, *[mins_467465], **kwargs_467466)
        
        # Assigning a type to the variable 'greatermins' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'greatermins', copy_call_result_467467)
        
        # Assigning a Name to a Subscript (line 313):
        
        # Assigning a Name to a Subscript (line 313):
        # Getting the type of 'split' (line 313)
        split_467468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 29), 'split')
        # Getting the type of 'greatermins' (line 313)
        greatermins_467469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'greatermins')
        # Getting the type of 'd' (line 313)
        d_467470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 24), 'd')
        # Storing an element on a container (line 313)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 12), greatermins_467469, (d_467470, split_467468))
        
        # Call to innernode(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'd' (line 314)
        d_467473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 36), 'd', False)
        # Getting the type of 'split' (line 314)
        split_467474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 39), 'split', False)
        
        # Call to __build(...): (line 315)
        # Processing the call arguments (line 315)
        
        # Obtaining the type of the subscript
        # Getting the type of 'less_idx' (line 315)
        less_idx_467477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 37), 'less_idx', False)
        # Getting the type of 'idx' (line 315)
        idx_467478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 33), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 315)
        getitem___467479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 33), idx_467478, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 315)
        subscript_call_result_467480 = invoke(stypy.reporting.localization.Localization(__file__, 315, 33), getitem___467479, less_idx_467477)
        
        # Getting the type of 'lessmaxes' (line 315)
        lessmaxes_467481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 47), 'lessmaxes', False)
        # Getting the type of 'mins' (line 315)
        mins_467482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 57), 'mins', False)
        # Processing the call keyword arguments (line 315)
        kwargs_467483 = {}
        # Getting the type of 'self' (line 315)
        self_467475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 20), 'self', False)
        # Obtaining the member '__build' of a type (line 315)
        build_467476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 20), self_467475, '__build')
        # Calling __build(args, kwargs) (line 315)
        build_call_result_467484 = invoke(stypy.reporting.localization.Localization(__file__, 315, 20), build_467476, *[subscript_call_result_467480, lessmaxes_467481, mins_467482], **kwargs_467483)
        
        
        # Call to __build(...): (line 316)
        # Processing the call arguments (line 316)
        
        # Obtaining the type of the subscript
        # Getting the type of 'greater_idx' (line 316)
        greater_idx_467487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 37), 'greater_idx', False)
        # Getting the type of 'idx' (line 316)
        idx_467488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 33), 'idx', False)
        # Obtaining the member '__getitem__' of a type (line 316)
        getitem___467489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 33), idx_467488, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 316)
        subscript_call_result_467490 = invoke(stypy.reporting.localization.Localization(__file__, 316, 33), getitem___467489, greater_idx_467487)
        
        # Getting the type of 'maxes' (line 316)
        maxes_467491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 50), 'maxes', False)
        # Getting the type of 'greatermins' (line 316)
        greatermins_467492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 56), 'greatermins', False)
        # Processing the call keyword arguments (line 316)
        kwargs_467493 = {}
        # Getting the type of 'self' (line 316)
        self_467485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 20), 'self', False)
        # Obtaining the member '__build' of a type (line 316)
        build_467486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 20), self_467485, '__build')
        # Calling __build(args, kwargs) (line 316)
        build_call_result_467494 = invoke(stypy.reporting.localization.Localization(__file__, 316, 20), build_467486, *[subscript_call_result_467490, maxes_467491, greatermins_467492], **kwargs_467493)
        
        # Processing the call keyword arguments (line 314)
        kwargs_467495 = {}
        # Getting the type of 'KDTree' (line 314)
        KDTree_467471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 19), 'KDTree', False)
        # Obtaining the member 'innernode' of a type (line 314)
        innernode_467472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 19), KDTree_467471, 'innernode')
        # Calling innernode(args, kwargs) (line 314)
        innernode_call_result_467496 = invoke(stypy.reporting.localization.Localization(__file__, 314, 19), innernode_467472, *[d_467473, split_467474, build_call_result_467484, build_call_result_467494], **kwargs_467495)
        
        # Assigning a type to the variable 'stypy_return_type' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'stypy_return_type', innernode_call_result_467496)
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__build(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__build' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_467497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_467497)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__build'
        return stypy_return_type_467497


    @norecursion
    def __query(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_467498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 27), 'int')
        int_467499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 34), 'int')
        int_467500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 39), 'int')
        # Getting the type of 'np' (line 318)
        np_467501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 63), 'np')
        # Obtaining the member 'inf' of a type (line 318)
        inf_467502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 63), np_467501, 'inf')
        defaults = [int_467498, int_467499, int_467500, inf_467502]
        # Create a new context for function '__query'
        module_type_store = module_type_store.open_function_context('__query', 318, 4, False)
        # Assigning a type to the variable 'self' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KDTree.__query.__dict__.__setitem__('stypy_localization', localization)
        KDTree.__query.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KDTree.__query.__dict__.__setitem__('stypy_type_store', module_type_store)
        KDTree.__query.__dict__.__setitem__('stypy_function_name', 'KDTree.__query')
        KDTree.__query.__dict__.__setitem__('stypy_param_names_list', ['x', 'k', 'eps', 'p', 'distance_upper_bound'])
        KDTree.__query.__dict__.__setitem__('stypy_varargs_param_name', None)
        KDTree.__query.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KDTree.__query.__dict__.__setitem__('stypy_call_defaults', defaults)
        KDTree.__query.__dict__.__setitem__('stypy_call_varargs', varargs)
        KDTree.__query.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KDTree.__query.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KDTree.__query', ['x', 'k', 'eps', 'p', 'distance_upper_bound'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__query', localization, ['x', 'k', 'eps', 'p', 'distance_upper_bound'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__query(...)' code ##################

        
        # Assigning a Call to a Name (line 320):
        
        # Assigning a Call to a Name (line 320):
        
        # Call to maximum(...): (line 320)
        # Processing the call arguments (line 320)
        int_467505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 36), 'int')
        
        # Call to maximum(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'x' (line 320)
        x_467508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 49), 'x', False)
        # Getting the type of 'self' (line 320)
        self_467509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 51), 'self', False)
        # Obtaining the member 'maxes' of a type (line 320)
        maxes_467510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 51), self_467509, 'maxes')
        # Applying the binary operator '-' (line 320)
        result_sub_467511 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 49), '-', x_467508, maxes_467510)
        
        # Getting the type of 'self' (line 320)
        self_467512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 62), 'self', False)
        # Obtaining the member 'mins' of a type (line 320)
        mins_467513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 62), self_467512, 'mins')
        # Getting the type of 'x' (line 320)
        x_467514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 72), 'x', False)
        # Applying the binary operator '-' (line 320)
        result_sub_467515 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 62), '-', mins_467513, x_467514)
        
        # Processing the call keyword arguments (line 320)
        kwargs_467516 = {}
        # Getting the type of 'np' (line 320)
        np_467506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 38), 'np', False)
        # Obtaining the member 'maximum' of a type (line 320)
        maximum_467507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 38), np_467506, 'maximum')
        # Calling maximum(args, kwargs) (line 320)
        maximum_call_result_467517 = invoke(stypy.reporting.localization.Localization(__file__, 320, 38), maximum_467507, *[result_sub_467511, result_sub_467515], **kwargs_467516)
        
        # Processing the call keyword arguments (line 320)
        kwargs_467518 = {}
        # Getting the type of 'np' (line 320)
        np_467503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 25), 'np', False)
        # Obtaining the member 'maximum' of a type (line 320)
        maximum_467504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 25), np_467503, 'maximum')
        # Calling maximum(args, kwargs) (line 320)
        maximum_call_result_467519 = invoke(stypy.reporting.localization.Localization(__file__, 320, 25), maximum_467504, *[int_467505, maximum_call_result_467517], **kwargs_467518)
        
        # Assigning a type to the variable 'side_distances' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'side_distances', maximum_call_result_467519)
        
        
        # Getting the type of 'p' (line 321)
        p_467520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 11), 'p')
        # Getting the type of 'np' (line 321)
        np_467521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'np')
        # Obtaining the member 'inf' of a type (line 321)
        inf_467522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 16), np_467521, 'inf')
        # Applying the binary operator '!=' (line 321)
        result_ne_467523 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 11), '!=', p_467520, inf_467522)
        
        # Testing the type of an if condition (line 321)
        if_condition_467524 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 8), result_ne_467523)
        # Assigning a type to the variable 'if_condition_467524' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'if_condition_467524', if_condition_467524)
        # SSA begins for if statement (line 321)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'side_distances' (line 322)
        side_distances_467525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'side_distances')
        # Getting the type of 'p' (line 322)
        p_467526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 31), 'p')
        # Applying the binary operator '**=' (line 322)
        result_ipow_467527 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 12), '**=', side_distances_467525, p_467526)
        # Assigning a type to the variable 'side_distances' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'side_distances', result_ipow_467527)
        
        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Call to sum(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'side_distances' (line 323)
        side_distances_467530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 34), 'side_distances', False)
        # Processing the call keyword arguments (line 323)
        kwargs_467531 = {}
        # Getting the type of 'np' (line 323)
        np_467528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 27), 'np', False)
        # Obtaining the member 'sum' of a type (line 323)
        sum_467529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 27), np_467528, 'sum')
        # Calling sum(args, kwargs) (line 323)
        sum_call_result_467532 = invoke(stypy.reporting.localization.Localization(__file__, 323, 27), sum_467529, *[side_distances_467530], **kwargs_467531)
        
        # Assigning a type to the variable 'min_distance' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'min_distance', sum_call_result_467532)
        # SSA branch for the else part of an if statement (line 321)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 325):
        
        # Assigning a Call to a Name (line 325):
        
        # Call to amax(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'side_distances' (line 325)
        side_distances_467535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 35), 'side_distances', False)
        # Processing the call keyword arguments (line 325)
        kwargs_467536 = {}
        # Getting the type of 'np' (line 325)
        np_467533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 27), 'np', False)
        # Obtaining the member 'amax' of a type (line 325)
        amax_467534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 27), np_467533, 'amax')
        # Calling amax(args, kwargs) (line 325)
        amax_call_result_467537 = invoke(stypy.reporting.localization.Localization(__file__, 325, 27), amax_467534, *[side_distances_467535], **kwargs_467536)
        
        # Assigning a type to the variable 'min_distance' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'min_distance', amax_call_result_467537)
        # SSA join for if statement (line 321)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 332):
        
        # Assigning a List to a Name (line 332):
        
        # Obtaining an instance of the builtin type 'list' (line 332)
        list_467538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 332)
        # Adding element type (line 332)
        
        # Obtaining an instance of the builtin type 'tuple' (line 332)
        tuple_467539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 332)
        # Adding element type (line 332)
        # Getting the type of 'min_distance' (line 332)
        min_distance_467540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), 'min_distance')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 14), tuple_467539, min_distance_467540)
        # Adding element type (line 332)
        
        # Call to tuple(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'side_distances' (line 333)
        side_distances_467542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 20), 'side_distances', False)
        # Processing the call keyword arguments (line 333)
        kwargs_467543 = {}
        # Getting the type of 'tuple' (line 333)
        tuple_467541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 14), 'tuple', False)
        # Calling tuple(args, kwargs) (line 333)
        tuple_call_result_467544 = invoke(stypy.reporting.localization.Localization(__file__, 333, 14), tuple_467541, *[side_distances_467542], **kwargs_467543)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 14), tuple_467539, tuple_call_result_467544)
        # Adding element type (line 332)
        # Getting the type of 'self' (line 334)
        self_467545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 14), 'self')
        # Obtaining the member 'tree' of a type (line 334)
        tree_467546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 14), self_467545, 'tree')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 14), tuple_467539, tree_467546)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 12), list_467538, tuple_467539)
        
        # Assigning a type to the variable 'q' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'q', list_467538)
        
        # Assigning a List to a Name (line 338):
        
        # Assigning a List to a Name (line 338):
        
        # Obtaining an instance of the builtin type 'list' (line 338)
        list_467547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 338)
        
        # Assigning a type to the variable 'neighbors' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'neighbors', list_467547)
        
        
        # Getting the type of 'eps' (line 340)
        eps_467548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 11), 'eps')
        int_467549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 18), 'int')
        # Applying the binary operator '==' (line 340)
        result_eq_467550 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 11), '==', eps_467548, int_467549)
        
        # Testing the type of an if condition (line 340)
        if_condition_467551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 8), result_eq_467550)
        # Assigning a type to the variable 'if_condition_467551' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'if_condition_467551', if_condition_467551)
        # SSA begins for if statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 341):
        
        # Assigning a Num to a Name (line 341):
        int_467552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 21), 'int')
        # Assigning a type to the variable 'epsfac' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'epsfac', int_467552)
        # SSA branch for the else part of an if statement (line 340)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'p' (line 342)
        p_467553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 13), 'p')
        # Getting the type of 'np' (line 342)
        np_467554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 18), 'np')
        # Obtaining the member 'inf' of a type (line 342)
        inf_467555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 18), np_467554, 'inf')
        # Applying the binary operator '==' (line 342)
        result_eq_467556 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 13), '==', p_467553, inf_467555)
        
        # Testing the type of an if condition (line 342)
        if_condition_467557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 13), result_eq_467556)
        # Assigning a type to the variable 'if_condition_467557' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 13), 'if_condition_467557', if_condition_467557)
        # SSA begins for if statement (line 342)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 343):
        
        # Assigning a BinOp to a Name (line 343):
        int_467558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 21), 'int')
        int_467559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 24), 'int')
        # Getting the type of 'eps' (line 343)
        eps_467560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 26), 'eps')
        # Applying the binary operator '+' (line 343)
        result_add_467561 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 24), '+', int_467559, eps_467560)
        
        # Applying the binary operator 'div' (line 343)
        result_div_467562 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 21), 'div', int_467558, result_add_467561)
        
        # Assigning a type to the variable 'epsfac' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'epsfac', result_div_467562)
        # SSA branch for the else part of an if statement (line 342)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 345):
        
        # Assigning a BinOp to a Name (line 345):
        int_467563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 21), 'int')
        int_467564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 24), 'int')
        # Getting the type of 'eps' (line 345)
        eps_467565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 26), 'eps')
        # Applying the binary operator '+' (line 345)
        result_add_467566 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 24), '+', int_467564, eps_467565)
        
        # Getting the type of 'p' (line 345)
        p_467567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 32), 'p')
        # Applying the binary operator '**' (line 345)
        result_pow_467568 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 23), '**', result_add_467566, p_467567)
        
        # Applying the binary operator 'div' (line 345)
        result_div_467569 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 21), 'div', int_467563, result_pow_467568)
        
        # Assigning a type to the variable 'epsfac' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'epsfac', result_div_467569)
        # SSA join for if statement (line 342)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 340)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'p' (line 347)
        p_467570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'p')
        # Getting the type of 'np' (line 347)
        np_467571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'np')
        # Obtaining the member 'inf' of a type (line 347)
        inf_467572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 16), np_467571, 'inf')
        # Applying the binary operator '!=' (line 347)
        result_ne_467573 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 11), '!=', p_467570, inf_467572)
        
        
        # Getting the type of 'distance_upper_bound' (line 347)
        distance_upper_bound_467574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 27), 'distance_upper_bound')
        # Getting the type of 'np' (line 347)
        np_467575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 51), 'np')
        # Obtaining the member 'inf' of a type (line 347)
        inf_467576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 51), np_467575, 'inf')
        # Applying the binary operator '!=' (line 347)
        result_ne_467577 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 27), '!=', distance_upper_bound_467574, inf_467576)
        
        # Applying the binary operator 'and' (line 347)
        result_and_keyword_467578 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 11), 'and', result_ne_467573, result_ne_467577)
        
        # Testing the type of an if condition (line 347)
        if_condition_467579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 347, 8), result_and_keyword_467578)
        # Assigning a type to the variable 'if_condition_467579' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'if_condition_467579', if_condition_467579)
        # SSA begins for if statement (line 347)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 348):
        
        # Assigning a BinOp to a Name (line 348):
        # Getting the type of 'distance_upper_bound' (line 348)
        distance_upper_bound_467580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 35), 'distance_upper_bound')
        # Getting the type of 'p' (line 348)
        p_467581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 57), 'p')
        # Applying the binary operator '**' (line 348)
        result_pow_467582 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 35), '**', distance_upper_bound_467580, p_467581)
        
        # Assigning a type to the variable 'distance_upper_bound' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'distance_upper_bound', result_pow_467582)
        # SSA join for if statement (line 347)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'q' (line 350)
        q_467583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 14), 'q')
        # Testing the type of an if condition (line 350)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 8), q_467583)
        # SSA begins for while statement (line 350)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Tuple (line 351):
        
        # Assigning a Subscript to a Name (line 351):
        
        # Obtaining the type of the subscript
        int_467584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 12), 'int')
        
        # Call to heappop(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'q' (line 351)
        q_467586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 57), 'q', False)
        # Processing the call keyword arguments (line 351)
        kwargs_467587 = {}
        # Getting the type of 'heappop' (line 351)
        heappop_467585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 49), 'heappop', False)
        # Calling heappop(args, kwargs) (line 351)
        heappop_call_result_467588 = invoke(stypy.reporting.localization.Localization(__file__, 351, 49), heappop_467585, *[q_467586], **kwargs_467587)
        
        # Obtaining the member '__getitem__' of a type (line 351)
        getitem___467589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 12), heappop_call_result_467588, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 351)
        subscript_call_result_467590 = invoke(stypy.reporting.localization.Localization(__file__, 351, 12), getitem___467589, int_467584)
        
        # Assigning a type to the variable 'tuple_var_assignment_466763' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'tuple_var_assignment_466763', subscript_call_result_467590)
        
        # Assigning a Subscript to a Name (line 351):
        
        # Obtaining the type of the subscript
        int_467591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 12), 'int')
        
        # Call to heappop(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'q' (line 351)
        q_467593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 57), 'q', False)
        # Processing the call keyword arguments (line 351)
        kwargs_467594 = {}
        # Getting the type of 'heappop' (line 351)
        heappop_467592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 49), 'heappop', False)
        # Calling heappop(args, kwargs) (line 351)
        heappop_call_result_467595 = invoke(stypy.reporting.localization.Localization(__file__, 351, 49), heappop_467592, *[q_467593], **kwargs_467594)
        
        # Obtaining the member '__getitem__' of a type (line 351)
        getitem___467596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 12), heappop_call_result_467595, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 351)
        subscript_call_result_467597 = invoke(stypy.reporting.localization.Localization(__file__, 351, 12), getitem___467596, int_467591)
        
        # Assigning a type to the variable 'tuple_var_assignment_466764' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'tuple_var_assignment_466764', subscript_call_result_467597)
        
        # Assigning a Subscript to a Name (line 351):
        
        # Obtaining the type of the subscript
        int_467598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 12), 'int')
        
        # Call to heappop(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'q' (line 351)
        q_467600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 57), 'q', False)
        # Processing the call keyword arguments (line 351)
        kwargs_467601 = {}
        # Getting the type of 'heappop' (line 351)
        heappop_467599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 49), 'heappop', False)
        # Calling heappop(args, kwargs) (line 351)
        heappop_call_result_467602 = invoke(stypy.reporting.localization.Localization(__file__, 351, 49), heappop_467599, *[q_467600], **kwargs_467601)
        
        # Obtaining the member '__getitem__' of a type (line 351)
        getitem___467603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 12), heappop_call_result_467602, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 351)
        subscript_call_result_467604 = invoke(stypy.reporting.localization.Localization(__file__, 351, 12), getitem___467603, int_467598)
        
        # Assigning a type to the variable 'tuple_var_assignment_466765' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'tuple_var_assignment_466765', subscript_call_result_467604)
        
        # Assigning a Name to a Name (line 351):
        # Getting the type of 'tuple_var_assignment_466763' (line 351)
        tuple_var_assignment_466763_467605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'tuple_var_assignment_466763')
        # Assigning a type to the variable 'min_distance' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'min_distance', tuple_var_assignment_466763_467605)
        
        # Assigning a Name to a Name (line 351):
        # Getting the type of 'tuple_var_assignment_466764' (line 351)
        tuple_var_assignment_466764_467606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'tuple_var_assignment_466764')
        # Assigning a type to the variable 'side_distances' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 26), 'side_distances', tuple_var_assignment_466764_467606)
        
        # Assigning a Name to a Name (line 351):
        # Getting the type of 'tuple_var_assignment_466765' (line 351)
        tuple_var_assignment_466765_467607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'tuple_var_assignment_466765')
        # Assigning a type to the variable 'node' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 42), 'node', tuple_var_assignment_466765_467607)
        
        
        # Call to isinstance(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'node' (line 352)
        node_467609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 26), 'node', False)
        # Getting the type of 'KDTree' (line 352)
        KDTree_467610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 32), 'KDTree', False)
        # Obtaining the member 'leafnode' of a type (line 352)
        leafnode_467611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 32), KDTree_467610, 'leafnode')
        # Processing the call keyword arguments (line 352)
        kwargs_467612 = {}
        # Getting the type of 'isinstance' (line 352)
        isinstance_467608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 352)
        isinstance_call_result_467613 = invoke(stypy.reporting.localization.Localization(__file__, 352, 15), isinstance_467608, *[node_467609, leafnode_467611], **kwargs_467612)
        
        # Testing the type of an if condition (line 352)
        if_condition_467614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 12), isinstance_call_result_467613)
        # Assigning a type to the variable 'if_condition_467614' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'if_condition_467614', if_condition_467614)
        # SSA begins for if statement (line 352)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 354):
        
        # Assigning a Subscript to a Name (line 354):
        
        # Obtaining the type of the subscript
        # Getting the type of 'node' (line 354)
        node_467615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 33), 'node')
        # Obtaining the member 'idx' of a type (line 354)
        idx_467616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 33), node_467615, 'idx')
        # Getting the type of 'self' (line 354)
        self_467617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 23), 'self')
        # Obtaining the member 'data' of a type (line 354)
        data_467618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 23), self_467617, 'data')
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___467619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 23), data_467618, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_467620 = invoke(stypy.reporting.localization.Localization(__file__, 354, 23), getitem___467619, idx_467616)
        
        # Assigning a type to the variable 'data' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'data', subscript_call_result_467620)
        
        # Assigning a Call to a Name (line 355):
        
        # Assigning a Call to a Name (line 355):
        
        # Call to minkowski_distance_p(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'data' (line 355)
        data_467622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 42), 'data', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'np' (line 355)
        np_467623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 49), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 355)
        newaxis_467624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 49), np_467623, 'newaxis')
        slice_467625 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 355, 47), None, None, None)
        # Getting the type of 'x' (line 355)
        x_467626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 47), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 355)
        getitem___467627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 47), x_467626, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 355)
        subscript_call_result_467628 = invoke(stypy.reporting.localization.Localization(__file__, 355, 47), getitem___467627, (newaxis_467624, slice_467625))
        
        # Getting the type of 'p' (line 355)
        p_467629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 63), 'p', False)
        # Processing the call keyword arguments (line 355)
        kwargs_467630 = {}
        # Getting the type of 'minkowski_distance_p' (line 355)
        minkowski_distance_p_467621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 21), 'minkowski_distance_p', False)
        # Calling minkowski_distance_p(args, kwargs) (line 355)
        minkowski_distance_p_call_result_467631 = invoke(stypy.reporting.localization.Localization(__file__, 355, 21), minkowski_distance_p_467621, *[data_467622, subscript_call_result_467628, p_467629], **kwargs_467630)
        
        # Assigning a type to the variable 'ds' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'ds', minkowski_distance_p_call_result_467631)
        
        
        # Call to range(...): (line 356)
        # Processing the call arguments (line 356)
        
        # Call to len(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'ds' (line 356)
        ds_467634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 35), 'ds', False)
        # Processing the call keyword arguments (line 356)
        kwargs_467635 = {}
        # Getting the type of 'len' (line 356)
        len_467633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 31), 'len', False)
        # Calling len(args, kwargs) (line 356)
        len_call_result_467636 = invoke(stypy.reporting.localization.Localization(__file__, 356, 31), len_467633, *[ds_467634], **kwargs_467635)
        
        # Processing the call keyword arguments (line 356)
        kwargs_467637 = {}
        # Getting the type of 'range' (line 356)
        range_467632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 25), 'range', False)
        # Calling range(args, kwargs) (line 356)
        range_call_result_467638 = invoke(stypy.reporting.localization.Localization(__file__, 356, 25), range_467632, *[len_call_result_467636], **kwargs_467637)
        
        # Testing the type of a for loop iterable (line 356)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 356, 16), range_call_result_467638)
        # Getting the type of the for loop variable (line 356)
        for_loop_var_467639 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 356, 16), range_call_result_467638)
        # Assigning a type to the variable 'i' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 16), 'i', for_loop_var_467639)
        # SSA begins for a for statement (line 356)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 357)
        i_467640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 26), 'i')
        # Getting the type of 'ds' (line 357)
        ds_467641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 23), 'ds')
        # Obtaining the member '__getitem__' of a type (line 357)
        getitem___467642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 23), ds_467641, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 357)
        subscript_call_result_467643 = invoke(stypy.reporting.localization.Localization(__file__, 357, 23), getitem___467642, i_467640)
        
        # Getting the type of 'distance_upper_bound' (line 357)
        distance_upper_bound_467644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 31), 'distance_upper_bound')
        # Applying the binary operator '<' (line 357)
        result_lt_467645 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 23), '<', subscript_call_result_467643, distance_upper_bound_467644)
        
        # Testing the type of an if condition (line 357)
        if_condition_467646 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 20), result_lt_467645)
        # Assigning a type to the variable 'if_condition_467646' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'if_condition_467646', if_condition_467646)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'neighbors' (line 358)
        neighbors_467648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 31), 'neighbors', False)
        # Processing the call keyword arguments (line 358)
        kwargs_467649 = {}
        # Getting the type of 'len' (line 358)
        len_467647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'len', False)
        # Calling len(args, kwargs) (line 358)
        len_call_result_467650 = invoke(stypy.reporting.localization.Localization(__file__, 358, 27), len_467647, *[neighbors_467648], **kwargs_467649)
        
        # Getting the type of 'k' (line 358)
        k_467651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 45), 'k')
        # Applying the binary operator '==' (line 358)
        result_eq_467652 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 27), '==', len_call_result_467650, k_467651)
        
        # Testing the type of an if condition (line 358)
        if_condition_467653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 358, 24), result_eq_467652)
        # Assigning a type to the variable 'if_condition_467653' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 24), 'if_condition_467653', if_condition_467653)
        # SSA begins for if statement (line 358)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to heappop(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'neighbors' (line 359)
        neighbors_467655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 36), 'neighbors', False)
        # Processing the call keyword arguments (line 359)
        kwargs_467656 = {}
        # Getting the type of 'heappop' (line 359)
        heappop_467654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 28), 'heappop', False)
        # Calling heappop(args, kwargs) (line 359)
        heappop_call_result_467657 = invoke(stypy.reporting.localization.Localization(__file__, 359, 28), heappop_467654, *[neighbors_467655], **kwargs_467656)
        
        # SSA join for if statement (line 358)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to heappush(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'neighbors' (line 360)
        neighbors_467659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 33), 'neighbors', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 360)
        tuple_467660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 360)
        # Adding element type (line 360)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 360)
        i_467661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 49), 'i', False)
        # Getting the type of 'ds' (line 360)
        ds_467662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 46), 'ds', False)
        # Obtaining the member '__getitem__' of a type (line 360)
        getitem___467663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 46), ds_467662, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 360)
        subscript_call_result_467664 = invoke(stypy.reporting.localization.Localization(__file__, 360, 46), getitem___467663, i_467661)
        
        # Applying the 'usub' unary operator (line 360)
        result___neg___467665 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 45), 'usub', subscript_call_result_467664)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 45), tuple_467660, result___neg___467665)
        # Adding element type (line 360)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 360)
        i_467666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 62), 'i', False)
        # Getting the type of 'node' (line 360)
        node_467667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 53), 'node', False)
        # Obtaining the member 'idx' of a type (line 360)
        idx_467668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 53), node_467667, 'idx')
        # Obtaining the member '__getitem__' of a type (line 360)
        getitem___467669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 53), idx_467668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 360)
        subscript_call_result_467670 = invoke(stypy.reporting.localization.Localization(__file__, 360, 53), getitem___467669, i_467666)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 45), tuple_467660, subscript_call_result_467670)
        
        # Processing the call keyword arguments (line 360)
        kwargs_467671 = {}
        # Getting the type of 'heappush' (line 360)
        heappush_467658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 24), 'heappush', False)
        # Calling heappush(args, kwargs) (line 360)
        heappush_call_result_467672 = invoke(stypy.reporting.localization.Localization(__file__, 360, 24), heappush_467658, *[neighbors_467659, tuple_467660], **kwargs_467671)
        
        
        
        
        # Call to len(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'neighbors' (line 361)
        neighbors_467674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 31), 'neighbors', False)
        # Processing the call keyword arguments (line 361)
        kwargs_467675 = {}
        # Getting the type of 'len' (line 361)
        len_467673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 27), 'len', False)
        # Calling len(args, kwargs) (line 361)
        len_call_result_467676 = invoke(stypy.reporting.localization.Localization(__file__, 361, 27), len_467673, *[neighbors_467674], **kwargs_467675)
        
        # Getting the type of 'k' (line 361)
        k_467677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 45), 'k')
        # Applying the binary operator '==' (line 361)
        result_eq_467678 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 27), '==', len_call_result_467676, k_467677)
        
        # Testing the type of an if condition (line 361)
        if_condition_467679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 24), result_eq_467678)
        # Assigning a type to the variable 'if_condition_467679' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 24), 'if_condition_467679', if_condition_467679)
        # SSA begins for if statement (line 361)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a UnaryOp to a Name (line 362):
        
        # Assigning a UnaryOp to a Name (line 362):
        
        
        # Obtaining the type of the subscript
        int_467680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 65), 'int')
        
        # Obtaining the type of the subscript
        int_467681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 62), 'int')
        # Getting the type of 'neighbors' (line 362)
        neighbors_467682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 52), 'neighbors')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___467683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 52), neighbors_467682, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_467684 = invoke(stypy.reporting.localization.Localization(__file__, 362, 52), getitem___467683, int_467681)
        
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___467685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 52), subscript_call_result_467684, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_467686 = invoke(stypy.reporting.localization.Localization(__file__, 362, 52), getitem___467685, int_467680)
        
        # Applying the 'usub' unary operator (line 362)
        result___neg___467687 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 51), 'usub', subscript_call_result_467686)
        
        # Assigning a type to the variable 'distance_upper_bound' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 28), 'distance_upper_bound', result___neg___467687)
        # SSA join for if statement (line 361)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 352)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'min_distance' (line 367)
        min_distance_467688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 19), 'min_distance')
        # Getting the type of 'distance_upper_bound' (line 367)
        distance_upper_bound_467689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 34), 'distance_upper_bound')
        # Getting the type of 'epsfac' (line 367)
        epsfac_467690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 55), 'epsfac')
        # Applying the binary operator '*' (line 367)
        result_mul_467691 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 34), '*', distance_upper_bound_467689, epsfac_467690)
        
        # Applying the binary operator '>' (line 367)
        result_gt_467692 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 19), '>', min_distance_467688, result_mul_467691)
        
        # Testing the type of an if condition (line 367)
        if_condition_467693 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 16), result_gt_467692)
        # Assigning a type to the variable 'if_condition_467693' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'if_condition_467693', if_condition_467693)
        # SSA begins for if statement (line 367)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 367)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'node' (line 371)
        node_467694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 21), 'node')
        # Obtaining the member 'split_dim' of a type (line 371)
        split_dim_467695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 21), node_467694, 'split_dim')
        # Getting the type of 'x' (line 371)
        x_467696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 19), 'x')
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___467697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 19), x_467696, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 371)
        subscript_call_result_467698 = invoke(stypy.reporting.localization.Localization(__file__, 371, 19), getitem___467697, split_dim_467695)
        
        # Getting the type of 'node' (line 371)
        node_467699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 39), 'node')
        # Obtaining the member 'split' of a type (line 371)
        split_467700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 39), node_467699, 'split')
        # Applying the binary operator '<' (line 371)
        result_lt_467701 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 19), '<', subscript_call_result_467698, split_467700)
        
        # Testing the type of an if condition (line 371)
        if_condition_467702 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 16), result_lt_467701)
        # Assigning a type to the variable 'if_condition_467702' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'if_condition_467702', if_condition_467702)
        # SSA begins for if statement (line 371)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Tuple (line 372):
        
        # Assigning a Attribute to a Name (line 372):
        # Getting the type of 'node' (line 372)
        node_467703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 32), 'node')
        # Obtaining the member 'less' of a type (line 372)
        less_467704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 32), node_467703, 'less')
        # Assigning a type to the variable 'tuple_assignment_466766' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'tuple_assignment_466766', less_467704)
        
        # Assigning a Attribute to a Name (line 372):
        # Getting the type of 'node' (line 372)
        node_467705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 43), 'node')
        # Obtaining the member 'greater' of a type (line 372)
        greater_467706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 43), node_467705, 'greater')
        # Assigning a type to the variable 'tuple_assignment_466767' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'tuple_assignment_466767', greater_467706)
        
        # Assigning a Name to a Name (line 372):
        # Getting the type of 'tuple_assignment_466766' (line 372)
        tuple_assignment_466766_467707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'tuple_assignment_466766')
        # Assigning a type to the variable 'near' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'near', tuple_assignment_466766_467707)
        
        # Assigning a Name to a Name (line 372):
        # Getting the type of 'tuple_assignment_466767' (line 372)
        tuple_assignment_466767_467708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'tuple_assignment_466767')
        # Assigning a type to the variable 'far' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 26), 'far', tuple_assignment_466767_467708)
        # SSA branch for the else part of an if statement (line 371)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Tuple to a Tuple (line 374):
        
        # Assigning a Attribute to a Name (line 374):
        # Getting the type of 'node' (line 374)
        node_467709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 32), 'node')
        # Obtaining the member 'greater' of a type (line 374)
        greater_467710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 32), node_467709, 'greater')
        # Assigning a type to the variable 'tuple_assignment_466768' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 20), 'tuple_assignment_466768', greater_467710)
        
        # Assigning a Attribute to a Name (line 374):
        # Getting the type of 'node' (line 374)
        node_467711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 46), 'node')
        # Obtaining the member 'less' of a type (line 374)
        less_467712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 46), node_467711, 'less')
        # Assigning a type to the variable 'tuple_assignment_466769' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 20), 'tuple_assignment_466769', less_467712)
        
        # Assigning a Name to a Name (line 374):
        # Getting the type of 'tuple_assignment_466768' (line 374)
        tuple_assignment_466768_467713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 20), 'tuple_assignment_466768')
        # Assigning a type to the variable 'near' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 20), 'near', tuple_assignment_466768_467713)
        
        # Assigning a Name to a Name (line 374):
        # Getting the type of 'tuple_assignment_466769' (line 374)
        tuple_assignment_466769_467714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 20), 'tuple_assignment_466769')
        # Assigning a type to the variable 'far' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 26), 'far', tuple_assignment_466769_467714)
        # SSA join for if statement (line 371)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to heappush(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'q' (line 377)
        q_467716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 25), 'q', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 377)
        tuple_467717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 377)
        # Adding element type (line 377)
        # Getting the type of 'min_distance' (line 377)
        min_distance_467718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 28), 'min_distance', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 28), tuple_467717, min_distance_467718)
        # Adding element type (line 377)
        # Getting the type of 'side_distances' (line 377)
        side_distances_467719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 42), 'side_distances', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 28), tuple_467717, side_distances_467719)
        # Adding element type (line 377)
        # Getting the type of 'near' (line 377)
        near_467720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 58), 'near', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 28), tuple_467717, near_467720)
        
        # Processing the call keyword arguments (line 377)
        kwargs_467721 = {}
        # Getting the type of 'heappush' (line 377)
        heappush_467715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 'heappush', False)
        # Calling heappush(args, kwargs) (line 377)
        heappush_call_result_467722 = invoke(stypy.reporting.localization.Localization(__file__, 377, 16), heappush_467715, *[q_467716, tuple_467717], **kwargs_467721)
        
        
        # Assigning a Call to a Name (line 381):
        
        # Assigning a Call to a Name (line 381):
        
        # Call to list(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'side_distances' (line 381)
        side_distances_467724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 26), 'side_distances', False)
        # Processing the call keyword arguments (line 381)
        kwargs_467725 = {}
        # Getting the type of 'list' (line 381)
        list_467723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'list', False)
        # Calling list(args, kwargs) (line 381)
        list_call_result_467726 = invoke(stypy.reporting.localization.Localization(__file__, 381, 21), list_467723, *[side_distances_467724], **kwargs_467725)
        
        # Assigning a type to the variable 'sd' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 16), 'sd', list_call_result_467726)
        
        
        # Getting the type of 'p' (line 382)
        p_467727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 19), 'p')
        # Getting the type of 'np' (line 382)
        np_467728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 24), 'np')
        # Obtaining the member 'inf' of a type (line 382)
        inf_467729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 24), np_467728, 'inf')
        # Applying the binary operator '==' (line 382)
        result_eq_467730 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 19), '==', p_467727, inf_467729)
        
        # Testing the type of an if condition (line 382)
        if_condition_467731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 16), result_eq_467730)
        # Assigning a type to the variable 'if_condition_467731' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 16), 'if_condition_467731', if_condition_467731)
        # SSA begins for if statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 383):
        
        # Assigning a Call to a Name (line 383):
        
        # Call to max(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'min_distance' (line 383)
        min_distance_467733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 39), 'min_distance', False)
        
        # Call to abs(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'node' (line 383)
        node_467735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 57), 'node', False)
        # Obtaining the member 'split' of a type (line 383)
        split_467736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 57), node_467735, 'split')
        
        # Obtaining the type of the subscript
        # Getting the type of 'node' (line 383)
        node_467737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 70), 'node', False)
        # Obtaining the member 'split_dim' of a type (line 383)
        split_dim_467738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 70), node_467737, 'split_dim')
        # Getting the type of 'x' (line 383)
        x_467739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 68), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 383)
        getitem___467740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 68), x_467739, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 383)
        subscript_call_result_467741 = invoke(stypy.reporting.localization.Localization(__file__, 383, 68), getitem___467740, split_dim_467738)
        
        # Applying the binary operator '-' (line 383)
        result_sub_467742 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 57), '-', split_467736, subscript_call_result_467741)
        
        # Processing the call keyword arguments (line 383)
        kwargs_467743 = {}
        # Getting the type of 'abs' (line 383)
        abs_467734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 53), 'abs', False)
        # Calling abs(args, kwargs) (line 383)
        abs_call_result_467744 = invoke(stypy.reporting.localization.Localization(__file__, 383, 53), abs_467734, *[result_sub_467742], **kwargs_467743)
        
        # Processing the call keyword arguments (line 383)
        kwargs_467745 = {}
        # Getting the type of 'max' (line 383)
        max_467732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 35), 'max', False)
        # Calling max(args, kwargs) (line 383)
        max_call_result_467746 = invoke(stypy.reporting.localization.Localization(__file__, 383, 35), max_467732, *[min_distance_467733, abs_call_result_467744], **kwargs_467745)
        
        # Assigning a type to the variable 'min_distance' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 20), 'min_distance', max_call_result_467746)
        # SSA branch for the else part of an if statement (line 382)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'p' (line 384)
        p_467747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 21), 'p')
        int_467748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 26), 'int')
        # Applying the binary operator '==' (line 384)
        result_eq_467749 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 21), '==', p_467747, int_467748)
        
        # Testing the type of an if condition (line 384)
        if_condition_467750 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 384, 21), result_eq_467749)
        # Assigning a type to the variable 'if_condition_467750' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 21), 'if_condition_467750', if_condition_467750)
        # SSA begins for if statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 385):
        
        # Assigning a Call to a Subscript (line 385):
        
        # Call to abs(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'node' (line 385)
        node_467753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 48), 'node', False)
        # Obtaining the member 'split' of a type (line 385)
        split_467754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 48), node_467753, 'split')
        
        # Obtaining the type of the subscript
        # Getting the type of 'node' (line 385)
        node_467755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 61), 'node', False)
        # Obtaining the member 'split_dim' of a type (line 385)
        split_dim_467756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 61), node_467755, 'split_dim')
        # Getting the type of 'x' (line 385)
        x_467757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 59), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 385)
        getitem___467758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 59), x_467757, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 385)
        subscript_call_result_467759 = invoke(stypy.reporting.localization.Localization(__file__, 385, 59), getitem___467758, split_dim_467756)
        
        # Applying the binary operator '-' (line 385)
        result_sub_467760 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 48), '-', split_467754, subscript_call_result_467759)
        
        # Processing the call keyword arguments (line 385)
        kwargs_467761 = {}
        # Getting the type of 'np' (line 385)
        np_467751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 41), 'np', False)
        # Obtaining the member 'abs' of a type (line 385)
        abs_467752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 41), np_467751, 'abs')
        # Calling abs(args, kwargs) (line 385)
        abs_call_result_467762 = invoke(stypy.reporting.localization.Localization(__file__, 385, 41), abs_467752, *[result_sub_467760], **kwargs_467761)
        
        # Getting the type of 'sd' (line 385)
        sd_467763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'sd')
        # Getting the type of 'node' (line 385)
        node_467764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 23), 'node')
        # Obtaining the member 'split_dim' of a type (line 385)
        split_dim_467765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 23), node_467764, 'split_dim')
        # Storing an element on a container (line 385)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 20), sd_467763, (split_dim_467765, abs_call_result_467762))
        
        # Assigning a BinOp to a Name (line 386):
        
        # Assigning a BinOp to a Name (line 386):
        # Getting the type of 'min_distance' (line 386)
        min_distance_467766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 35), 'min_distance')
        
        # Obtaining the type of the subscript
        # Getting the type of 'node' (line 386)
        node_467767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 65), 'node')
        # Obtaining the member 'split_dim' of a type (line 386)
        split_dim_467768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 65), node_467767, 'split_dim')
        # Getting the type of 'side_distances' (line 386)
        side_distances_467769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 50), 'side_distances')
        # Obtaining the member '__getitem__' of a type (line 386)
        getitem___467770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 50), side_distances_467769, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 386)
        subscript_call_result_467771 = invoke(stypy.reporting.localization.Localization(__file__, 386, 50), getitem___467770, split_dim_467768)
        
        # Applying the binary operator '-' (line 386)
        result_sub_467772 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 35), '-', min_distance_467766, subscript_call_result_467771)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'node' (line 386)
        node_467773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 86), 'node')
        # Obtaining the member 'split_dim' of a type (line 386)
        split_dim_467774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 86), node_467773, 'split_dim')
        # Getting the type of 'sd' (line 386)
        sd_467775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 83), 'sd')
        # Obtaining the member '__getitem__' of a type (line 386)
        getitem___467776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 83), sd_467775, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 386)
        subscript_call_result_467777 = invoke(stypy.reporting.localization.Localization(__file__, 386, 83), getitem___467776, split_dim_467774)
        
        # Applying the binary operator '+' (line 386)
        result_add_467778 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 81), '+', result_sub_467772, subscript_call_result_467777)
        
        # Assigning a type to the variable 'min_distance' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 20), 'min_distance', result_add_467778)
        # SSA branch for the else part of an if statement (line 384)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Subscript (line 388):
        
        # Assigning a BinOp to a Subscript (line 388):
        
        # Call to abs(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'node' (line 388)
        node_467781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 48), 'node', False)
        # Obtaining the member 'split' of a type (line 388)
        split_467782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 48), node_467781, 'split')
        
        # Obtaining the type of the subscript
        # Getting the type of 'node' (line 388)
        node_467783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 61), 'node', False)
        # Obtaining the member 'split_dim' of a type (line 388)
        split_dim_467784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 61), node_467783, 'split_dim')
        # Getting the type of 'x' (line 388)
        x_467785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 59), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___467786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 59), x_467785, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 388)
        subscript_call_result_467787 = invoke(stypy.reporting.localization.Localization(__file__, 388, 59), getitem___467786, split_dim_467784)
        
        # Applying the binary operator '-' (line 388)
        result_sub_467788 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 48), '-', split_467782, subscript_call_result_467787)
        
        # Processing the call keyword arguments (line 388)
        kwargs_467789 = {}
        # Getting the type of 'np' (line 388)
        np_467779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 41), 'np', False)
        # Obtaining the member 'abs' of a type (line 388)
        abs_467780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 41), np_467779, 'abs')
        # Calling abs(args, kwargs) (line 388)
        abs_call_result_467790 = invoke(stypy.reporting.localization.Localization(__file__, 388, 41), abs_467780, *[result_sub_467788], **kwargs_467789)
        
        # Getting the type of 'p' (line 388)
        p_467791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 79), 'p')
        # Applying the binary operator '**' (line 388)
        result_pow_467792 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 41), '**', abs_call_result_467790, p_467791)
        
        # Getting the type of 'sd' (line 388)
        sd_467793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'sd')
        # Getting the type of 'node' (line 388)
        node_467794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 23), 'node')
        # Obtaining the member 'split_dim' of a type (line 388)
        split_dim_467795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 23), node_467794, 'split_dim')
        # Storing an element on a container (line 388)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 20), sd_467793, (split_dim_467795, result_pow_467792))
        
        # Assigning a BinOp to a Name (line 389):
        
        # Assigning a BinOp to a Name (line 389):
        # Getting the type of 'min_distance' (line 389)
        min_distance_467796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 35), 'min_distance')
        
        # Obtaining the type of the subscript
        # Getting the type of 'node' (line 389)
        node_467797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 65), 'node')
        # Obtaining the member 'split_dim' of a type (line 389)
        split_dim_467798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 65), node_467797, 'split_dim')
        # Getting the type of 'side_distances' (line 389)
        side_distances_467799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 50), 'side_distances')
        # Obtaining the member '__getitem__' of a type (line 389)
        getitem___467800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 50), side_distances_467799, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 389)
        subscript_call_result_467801 = invoke(stypy.reporting.localization.Localization(__file__, 389, 50), getitem___467800, split_dim_467798)
        
        # Applying the binary operator '-' (line 389)
        result_sub_467802 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 35), '-', min_distance_467796, subscript_call_result_467801)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'node' (line 389)
        node_467803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 86), 'node')
        # Obtaining the member 'split_dim' of a type (line 389)
        split_dim_467804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 86), node_467803, 'split_dim')
        # Getting the type of 'sd' (line 389)
        sd_467805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 83), 'sd')
        # Obtaining the member '__getitem__' of a type (line 389)
        getitem___467806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 83), sd_467805, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 389)
        subscript_call_result_467807 = invoke(stypy.reporting.localization.Localization(__file__, 389, 83), getitem___467806, split_dim_467804)
        
        # Applying the binary operator '+' (line 389)
        result_add_467808 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 81), '+', result_sub_467802, subscript_call_result_467807)
        
        # Assigning a type to the variable 'min_distance' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 20), 'min_distance', result_add_467808)
        # SSA join for if statement (line 384)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 382)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'min_distance' (line 392)
        min_distance_467809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'min_distance')
        # Getting the type of 'distance_upper_bound' (line 392)
        distance_upper_bound_467810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 35), 'distance_upper_bound')
        # Getting the type of 'epsfac' (line 392)
        epsfac_467811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 56), 'epsfac')
        # Applying the binary operator '*' (line 392)
        result_mul_467812 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 35), '*', distance_upper_bound_467810, epsfac_467811)
        
        # Applying the binary operator '<=' (line 392)
        result_le_467813 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 19), '<=', min_distance_467809, result_mul_467812)
        
        # Testing the type of an if condition (line 392)
        if_condition_467814 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 16), result_le_467813)
        # Assigning a type to the variable 'if_condition_467814' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'if_condition_467814', if_condition_467814)
        # SSA begins for if statement (line 392)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to heappush(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'q' (line 393)
        q_467816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 29), 'q', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 393)
        tuple_467817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 393)
        # Adding element type (line 393)
        # Getting the type of 'min_distance' (line 393)
        min_distance_467818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 32), 'min_distance', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 32), tuple_467817, min_distance_467818)
        # Adding element type (line 393)
        
        # Call to tuple(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'sd' (line 393)
        sd_467820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 52), 'sd', False)
        # Processing the call keyword arguments (line 393)
        kwargs_467821 = {}
        # Getting the type of 'tuple' (line 393)
        tuple_467819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 46), 'tuple', False)
        # Calling tuple(args, kwargs) (line 393)
        tuple_call_result_467822 = invoke(stypy.reporting.localization.Localization(__file__, 393, 46), tuple_467819, *[sd_467820], **kwargs_467821)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 32), tuple_467817, tuple_call_result_467822)
        # Adding element type (line 393)
        # Getting the type of 'far' (line 393)
        far_467823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 57), 'far', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 32), tuple_467817, far_467823)
        
        # Processing the call keyword arguments (line 393)
        kwargs_467824 = {}
        # Getting the type of 'heappush' (line 393)
        heappush_467815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 20), 'heappush', False)
        # Calling heappush(args, kwargs) (line 393)
        heappush_call_result_467825 = invoke(stypy.reporting.localization.Localization(__file__, 393, 20), heappush_467815, *[q_467816, tuple_467817], **kwargs_467824)
        
        # SSA join for if statement (line 392)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 352)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 350)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'p' (line 395)
        p_467826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 11), 'p')
        # Getting the type of 'np' (line 395)
        np_467827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'np')
        # Obtaining the member 'inf' of a type (line 395)
        inf_467828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 16), np_467827, 'inf')
        # Applying the binary operator '==' (line 395)
        result_eq_467829 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 11), '==', p_467826, inf_467828)
        
        # Testing the type of an if condition (line 395)
        if_condition_467830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 8), result_eq_467829)
        # Assigning a type to the variable 'if_condition_467830' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'if_condition_467830', if_condition_467830)
        # SSA begins for if statement (line 395)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to sorted(...): (line 396)
        # Processing the call arguments (line 396)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'neighbors' (line 396)
        neighbors_467836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 47), 'neighbors', False)
        comprehension_467837 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 27), neighbors_467836)
        # Assigning a type to the variable 'd' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 27), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 27), comprehension_467837))
        # Assigning a type to the variable 'i' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 27), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 27), comprehension_467837))
        
        # Obtaining an instance of the builtin type 'tuple' (line 396)
        tuple_467832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 396)
        # Adding element type (line 396)
        
        # Getting the type of 'd' (line 396)
        d_467833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 29), 'd', False)
        # Applying the 'usub' unary operator (line 396)
        result___neg___467834 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 28), 'usub', d_467833)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 28), tuple_467832, result___neg___467834)
        # Adding element type (line 396)
        # Getting the type of 'i' (line 396)
        i_467835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 31), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 28), tuple_467832, i_467835)
        
        list_467838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 27), list_467838, tuple_467832)
        # Processing the call keyword arguments (line 396)
        kwargs_467839 = {}
        # Getting the type of 'sorted' (line 396)
        sorted_467831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'sorted', False)
        # Calling sorted(args, kwargs) (line 396)
        sorted_call_result_467840 = invoke(stypy.reporting.localization.Localization(__file__, 396, 19), sorted_467831, *[list_467838], **kwargs_467839)
        
        # Assigning a type to the variable 'stypy_return_type' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'stypy_return_type', sorted_call_result_467840)
        # SSA branch for the else part of an if statement (line 395)
        module_type_store.open_ssa_branch('else')
        
        # Call to sorted(...): (line 398)
        # Processing the call arguments (line 398)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'neighbors' (line 398)
        neighbors_467850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 57), 'neighbors', False)
        comprehension_467851 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 27), neighbors_467850)
        # Assigning a type to the variable 'd' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 27), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 27), comprehension_467851))
        # Assigning a type to the variable 'i' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 27), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 27), comprehension_467851))
        
        # Obtaining an instance of the builtin type 'tuple' (line 398)
        tuple_467842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 398)
        # Adding element type (line 398)
        
        # Getting the type of 'd' (line 398)
        d_467843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 30), 'd', False)
        # Applying the 'usub' unary operator (line 398)
        result___neg___467844 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 29), 'usub', d_467843)
        
        float_467845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 35), 'float')
        # Getting the type of 'p' (line 398)
        p_467846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 38), 'p', False)
        # Applying the binary operator 'div' (line 398)
        result_div_467847 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 35), 'div', float_467845, p_467846)
        
        # Applying the binary operator '**' (line 398)
        result_pow_467848 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 28), '**', result___neg___467844, result_div_467847)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 28), tuple_467842, result_pow_467848)
        # Adding element type (line 398)
        # Getting the type of 'i' (line 398)
        i_467849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 41), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 28), tuple_467842, i_467849)
        
        list_467852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 27), list_467852, tuple_467842)
        # Processing the call keyword arguments (line 398)
        kwargs_467853 = {}
        # Getting the type of 'sorted' (line 398)
        sorted_467841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), 'sorted', False)
        # Calling sorted(args, kwargs) (line 398)
        sorted_call_result_467854 = invoke(stypy.reporting.localization.Localization(__file__, 398, 19), sorted_467841, *[list_467852], **kwargs_467853)
        
        # Assigning a type to the variable 'stypy_return_type' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'stypy_return_type', sorted_call_result_467854)
        # SSA join for if statement (line 395)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__query(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__query' in the type store
        # Getting the type of 'stypy_return_type' (line 318)
        stypy_return_type_467855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_467855)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__query'
        return stypy_return_type_467855


    @norecursion
    def query(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_467856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 25), 'int')
        int_467857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 32), 'int')
        int_467858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 37), 'int')
        # Getting the type of 'np' (line 400)
        np_467859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 61), 'np')
        # Obtaining the member 'inf' of a type (line 400)
        inf_467860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 61), np_467859, 'inf')
        defaults = [int_467856, int_467857, int_467858, inf_467860]
        # Create a new context for function 'query'
        module_type_store = module_type_store.open_function_context('query', 400, 4, False)
        # Assigning a type to the variable 'self' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KDTree.query.__dict__.__setitem__('stypy_localization', localization)
        KDTree.query.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KDTree.query.__dict__.__setitem__('stypy_type_store', module_type_store)
        KDTree.query.__dict__.__setitem__('stypy_function_name', 'KDTree.query')
        KDTree.query.__dict__.__setitem__('stypy_param_names_list', ['x', 'k', 'eps', 'p', 'distance_upper_bound'])
        KDTree.query.__dict__.__setitem__('stypy_varargs_param_name', None)
        KDTree.query.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KDTree.query.__dict__.__setitem__('stypy_call_defaults', defaults)
        KDTree.query.__dict__.__setitem__('stypy_call_varargs', varargs)
        KDTree.query.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KDTree.query.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KDTree.query', ['x', 'k', 'eps', 'p', 'distance_upper_bound'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'query', localization, ['x', 'k', 'eps', 'p', 'distance_upper_bound'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'query(...)' code ##################

        str_467861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, (-1)), 'str', '\n        Query the kd-tree for nearest neighbors\n\n        Parameters\n        ----------\n        x : array_like, last dimension self.m\n            An array of points to query.\n        k : int, optional\n            The number of nearest neighbors to return.\n        eps : nonnegative float, optional\n            Return approximate nearest neighbors; the kth returned value\n            is guaranteed to be no further than (1+eps) times the\n            distance to the real kth nearest neighbor.\n        p : float, 1<=p<=infinity, optional\n            Which Minkowski p-norm to use.\n            1 is the sum-of-absolute-values "Manhattan" distance\n            2 is the usual Euclidean distance\n            infinity is the maximum-coordinate-difference distance\n        distance_upper_bound : nonnegative float, optional\n            Return only neighbors within this distance. This is used to prune\n            tree searches, so if you are doing a series of nearest-neighbor\n            queries, it may help to supply the distance to the nearest neighbor\n            of the most recent point.\n\n        Returns\n        -------\n        d : float or array of floats\n            The distances to the nearest neighbors.\n            If x has shape tuple+(self.m,), then d has shape tuple if\n            k is one, or tuple+(k,) if k is larger than one. Missing\n            neighbors (e.g. when k > n or distance_upper_bound is\n            given) are indicated with infinite distances.  If k is None,\n            then d is an object array of shape tuple, containing lists\n            of distances. In either case the hits are sorted by distance\n            (nearest first).\n        i : integer or array of integers\n            The locations of the neighbors in self.data. i is the same\n            shape as d.\n\n        Examples\n        --------\n        >>> from scipy import spatial\n        >>> x, y = np.mgrid[0:5, 2:8]\n        >>> tree = spatial.KDTree(list(zip(x.ravel(), y.ravel())))\n        >>> tree.data\n        array([[0, 2],\n               [0, 3],\n               [0, 4],\n               [0, 5],\n               [0, 6],\n               [0, 7],\n               [1, 2],\n               [1, 3],\n               [1, 4],\n               [1, 5],\n               [1, 6],\n               [1, 7],\n               [2, 2],\n               [2, 3],\n               [2, 4],\n               [2, 5],\n               [2, 6],\n               [2, 7],\n               [3, 2],\n               [3, 3],\n               [3, 4],\n               [3, 5],\n               [3, 6],\n               [3, 7],\n               [4, 2],\n               [4, 3],\n               [4, 4],\n               [4, 5],\n               [4, 6],\n               [4, 7]])\n        >>> pts = np.array([[0, 0], [2.1, 2.9]])\n        >>> tree.query(pts)\n        (array([ 2.        ,  0.14142136]), array([ 0, 13]))\n        >>> tree.query(pts[0])\n        (2.0, 0)\n\n        ')
        
        # Assigning a Call to a Name (line 483):
        
        # Assigning a Call to a Name (line 483):
        
        # Call to asarray(...): (line 483)
        # Processing the call arguments (line 483)
        # Getting the type of 'x' (line 483)
        x_467864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 23), 'x', False)
        # Processing the call keyword arguments (line 483)
        kwargs_467865 = {}
        # Getting the type of 'np' (line 483)
        np_467862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 483)
        asarray_467863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 12), np_467862, 'asarray')
        # Calling asarray(args, kwargs) (line 483)
        asarray_call_result_467866 = invoke(stypy.reporting.localization.Localization(__file__, 483, 12), asarray_467863, *[x_467864], **kwargs_467865)
        
        # Assigning a type to the variable 'x' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'x', asarray_call_result_467866)
        
        
        
        # Obtaining the type of the subscript
        int_467867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 23), 'int')
        
        # Call to shape(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'x' (line 484)
        x_467870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 20), 'x', False)
        # Processing the call keyword arguments (line 484)
        kwargs_467871 = {}
        # Getting the type of 'np' (line 484)
        np_467868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 11), 'np', False)
        # Obtaining the member 'shape' of a type (line 484)
        shape_467869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 11), np_467868, 'shape')
        # Calling shape(args, kwargs) (line 484)
        shape_call_result_467872 = invoke(stypy.reporting.localization.Localization(__file__, 484, 11), shape_467869, *[x_467870], **kwargs_467871)
        
        # Obtaining the member '__getitem__' of a type (line 484)
        getitem___467873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 11), shape_call_result_467872, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 484)
        subscript_call_result_467874 = invoke(stypy.reporting.localization.Localization(__file__, 484, 11), getitem___467873, int_467867)
        
        # Getting the type of 'self' (line 484)
        self_467875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 30), 'self')
        # Obtaining the member 'm' of a type (line 484)
        m_467876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 30), self_467875, 'm')
        # Applying the binary operator '!=' (line 484)
        result_ne_467877 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 11), '!=', subscript_call_result_467874, m_467876)
        
        # Testing the type of an if condition (line 484)
        if_condition_467878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 484, 8), result_ne_467877)
        # Assigning a type to the variable 'if_condition_467878' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'if_condition_467878', if_condition_467878)
        # SSA begins for if statement (line 484)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 485)
        # Processing the call arguments (line 485)
        str_467880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 29), 'str', 'x must consist of vectors of length %d but has shape %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 485)
        tuple_467881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 90), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 485)
        # Adding element type (line 485)
        # Getting the type of 'self' (line 485)
        self_467882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 90), 'self', False)
        # Obtaining the member 'm' of a type (line 485)
        m_467883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 90), self_467882, 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 90), tuple_467881, m_467883)
        # Adding element type (line 485)
        
        # Call to shape(...): (line 485)
        # Processing the call arguments (line 485)
        # Getting the type of 'x' (line 485)
        x_467886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 107), 'x', False)
        # Processing the call keyword arguments (line 485)
        kwargs_467887 = {}
        # Getting the type of 'np' (line 485)
        np_467884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 98), 'np', False)
        # Obtaining the member 'shape' of a type (line 485)
        shape_467885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 98), np_467884, 'shape')
        # Calling shape(args, kwargs) (line 485)
        shape_call_result_467888 = invoke(stypy.reporting.localization.Localization(__file__, 485, 98), shape_467885, *[x_467886], **kwargs_467887)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 90), tuple_467881, shape_call_result_467888)
        
        # Applying the binary operator '%' (line 485)
        result_mod_467889 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 29), '%', str_467880, tuple_467881)
        
        # Processing the call keyword arguments (line 485)
        kwargs_467890 = {}
        # Getting the type of 'ValueError' (line 485)
        ValueError_467879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 485)
        ValueError_call_result_467891 = invoke(stypy.reporting.localization.Localization(__file__, 485, 18), ValueError_467879, *[result_mod_467889], **kwargs_467890)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 485, 12), ValueError_call_result_467891, 'raise parameter', BaseException)
        # SSA join for if statement (line 484)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'p' (line 486)
        p_467892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 11), 'p')
        int_467893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 15), 'int')
        # Applying the binary operator '<' (line 486)
        result_lt_467894 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 11), '<', p_467892, int_467893)
        
        # Testing the type of an if condition (line 486)
        if_condition_467895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 486, 8), result_lt_467894)
        # Assigning a type to the variable 'if_condition_467895' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'if_condition_467895', if_condition_467895)
        # SSA begins for if statement (line 486)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 487)
        # Processing the call arguments (line 487)
        str_467897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 29), 'str', 'Only p-norms with 1<=p<=infinity permitted')
        # Processing the call keyword arguments (line 487)
        kwargs_467898 = {}
        # Getting the type of 'ValueError' (line 487)
        ValueError_467896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 487)
        ValueError_call_result_467899 = invoke(stypy.reporting.localization.Localization(__file__, 487, 18), ValueError_467896, *[str_467897], **kwargs_467898)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 487, 12), ValueError_call_result_467899, 'raise parameter', BaseException)
        # SSA join for if statement (line 486)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 488):
        
        # Assigning a Subscript to a Name (line 488):
        
        # Obtaining the type of the subscript
        int_467900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 32), 'int')
        slice_467901 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 488, 19), None, int_467900, None)
        
        # Call to shape(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 'x' (line 488)
        x_467904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 28), 'x', False)
        # Processing the call keyword arguments (line 488)
        kwargs_467905 = {}
        # Getting the type of 'np' (line 488)
        np_467902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 19), 'np', False)
        # Obtaining the member 'shape' of a type (line 488)
        shape_467903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 19), np_467902, 'shape')
        # Calling shape(args, kwargs) (line 488)
        shape_call_result_467906 = invoke(stypy.reporting.localization.Localization(__file__, 488, 19), shape_467903, *[x_467904], **kwargs_467905)
        
        # Obtaining the member '__getitem__' of a type (line 488)
        getitem___467907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 19), shape_call_result_467906, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 488)
        subscript_call_result_467908 = invoke(stypy.reporting.localization.Localization(__file__, 488, 19), getitem___467907, slice_467901)
        
        # Assigning a type to the variable 'retshape' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'retshape', subscript_call_result_467908)
        
        
        # Getting the type of 'retshape' (line 489)
        retshape_467909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 11), 'retshape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 489)
        tuple_467910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 489)
        
        # Applying the binary operator '!=' (line 489)
        result_ne_467911 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 11), '!=', retshape_467909, tuple_467910)
        
        # Testing the type of an if condition (line 489)
        if_condition_467912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 489, 8), result_ne_467911)
        # Assigning a type to the variable 'if_condition_467912' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'if_condition_467912', if_condition_467912)
        # SSA begins for if statement (line 489)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 490)
        # Getting the type of 'k' (line 490)
        k_467913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'k')
        # Getting the type of 'None' (line 490)
        None_467914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 20), 'None')
        
        (may_be_467915, more_types_in_union_467916) = may_be_none(k_467913, None_467914)

        if may_be_467915:

            if more_types_in_union_467916:
                # Runtime conditional SSA (line 490)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 491):
            
            # Assigning a Call to a Name (line 491):
            
            # Call to empty(...): (line 491)
            # Processing the call arguments (line 491)
            # Getting the type of 'retshape' (line 491)
            retshape_467919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 30), 'retshape', False)
            # Processing the call keyword arguments (line 491)
            # Getting the type of 'object' (line 491)
            object_467920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 45), 'object', False)
            keyword_467921 = object_467920
            kwargs_467922 = {'dtype': keyword_467921}
            # Getting the type of 'np' (line 491)
            np_467917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 21), 'np', False)
            # Obtaining the member 'empty' of a type (line 491)
            empty_467918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 21), np_467917, 'empty')
            # Calling empty(args, kwargs) (line 491)
            empty_call_result_467923 = invoke(stypy.reporting.localization.Localization(__file__, 491, 21), empty_467918, *[retshape_467919], **kwargs_467922)
            
            # Assigning a type to the variable 'dd' (line 491)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 16), 'dd', empty_call_result_467923)
            
            # Assigning a Call to a Name (line 492):
            
            # Assigning a Call to a Name (line 492):
            
            # Call to empty(...): (line 492)
            # Processing the call arguments (line 492)
            # Getting the type of 'retshape' (line 492)
            retshape_467926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 30), 'retshape', False)
            # Processing the call keyword arguments (line 492)
            # Getting the type of 'object' (line 492)
            object_467927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 45), 'object', False)
            keyword_467928 = object_467927
            kwargs_467929 = {'dtype': keyword_467928}
            # Getting the type of 'np' (line 492)
            np_467924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 21), 'np', False)
            # Obtaining the member 'empty' of a type (line 492)
            empty_467925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 21), np_467924, 'empty')
            # Calling empty(args, kwargs) (line 492)
            empty_call_result_467930 = invoke(stypy.reporting.localization.Localization(__file__, 492, 21), empty_467925, *[retshape_467926], **kwargs_467929)
            
            # Assigning a type to the variable 'ii' (line 492)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 16), 'ii', empty_call_result_467930)

            if more_types_in_union_467916:
                # Runtime conditional SSA for else branch (line 490)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_467915) or more_types_in_union_467916):
            
            
            # Getting the type of 'k' (line 493)
            k_467931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 17), 'k')
            int_467932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 21), 'int')
            # Applying the binary operator '>' (line 493)
            result_gt_467933 = python_operator(stypy.reporting.localization.Localization(__file__, 493, 17), '>', k_467931, int_467932)
            
            # Testing the type of an if condition (line 493)
            if_condition_467934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 493, 17), result_gt_467933)
            # Assigning a type to the variable 'if_condition_467934' (line 493)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 17), 'if_condition_467934', if_condition_467934)
            # SSA begins for if statement (line 493)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 494):
            
            # Assigning a Call to a Name (line 494):
            
            # Call to empty(...): (line 494)
            # Processing the call arguments (line 494)
            # Getting the type of 'retshape' (line 494)
            retshape_467937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 30), 'retshape', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 494)
            tuple_467938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 40), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 494)
            # Adding element type (line 494)
            # Getting the type of 'k' (line 494)
            k_467939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 40), 'k', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 494, 40), tuple_467938, k_467939)
            
            # Applying the binary operator '+' (line 494)
            result_add_467940 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 30), '+', retshape_467937, tuple_467938)
            
            # Processing the call keyword arguments (line 494)
            # Getting the type of 'float' (line 494)
            float_467941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 50), 'float', False)
            keyword_467942 = float_467941
            kwargs_467943 = {'dtype': keyword_467942}
            # Getting the type of 'np' (line 494)
            np_467935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 21), 'np', False)
            # Obtaining the member 'empty' of a type (line 494)
            empty_467936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 21), np_467935, 'empty')
            # Calling empty(args, kwargs) (line 494)
            empty_call_result_467944 = invoke(stypy.reporting.localization.Localization(__file__, 494, 21), empty_467936, *[result_add_467940], **kwargs_467943)
            
            # Assigning a type to the variable 'dd' (line 494)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 16), 'dd', empty_call_result_467944)
            
            # Call to fill(...): (line 495)
            # Processing the call arguments (line 495)
            # Getting the type of 'np' (line 495)
            np_467947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 24), 'np', False)
            # Obtaining the member 'inf' of a type (line 495)
            inf_467948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 24), np_467947, 'inf')
            # Processing the call keyword arguments (line 495)
            kwargs_467949 = {}
            # Getting the type of 'dd' (line 495)
            dd_467945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 16), 'dd', False)
            # Obtaining the member 'fill' of a type (line 495)
            fill_467946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 16), dd_467945, 'fill')
            # Calling fill(args, kwargs) (line 495)
            fill_call_result_467950 = invoke(stypy.reporting.localization.Localization(__file__, 495, 16), fill_467946, *[inf_467948], **kwargs_467949)
            
            
            # Assigning a Call to a Name (line 496):
            
            # Assigning a Call to a Name (line 496):
            
            # Call to empty(...): (line 496)
            # Processing the call arguments (line 496)
            # Getting the type of 'retshape' (line 496)
            retshape_467953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 30), 'retshape', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 496)
            tuple_467954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 40), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 496)
            # Adding element type (line 496)
            # Getting the type of 'k' (line 496)
            k_467955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 40), 'k', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 40), tuple_467954, k_467955)
            
            # Applying the binary operator '+' (line 496)
            result_add_467956 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 30), '+', retshape_467953, tuple_467954)
            
            # Processing the call keyword arguments (line 496)
            # Getting the type of 'int' (line 496)
            int_467957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 50), 'int', False)
            keyword_467958 = int_467957
            kwargs_467959 = {'dtype': keyword_467958}
            # Getting the type of 'np' (line 496)
            np_467951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 21), 'np', False)
            # Obtaining the member 'empty' of a type (line 496)
            empty_467952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 21), np_467951, 'empty')
            # Calling empty(args, kwargs) (line 496)
            empty_call_result_467960 = invoke(stypy.reporting.localization.Localization(__file__, 496, 21), empty_467952, *[result_add_467956], **kwargs_467959)
            
            # Assigning a type to the variable 'ii' (line 496)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 16), 'ii', empty_call_result_467960)
            
            # Call to fill(...): (line 497)
            # Processing the call arguments (line 497)
            # Getting the type of 'self' (line 497)
            self_467963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 24), 'self', False)
            # Obtaining the member 'n' of a type (line 497)
            n_467964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 24), self_467963, 'n')
            # Processing the call keyword arguments (line 497)
            kwargs_467965 = {}
            # Getting the type of 'ii' (line 497)
            ii_467961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'ii', False)
            # Obtaining the member 'fill' of a type (line 497)
            fill_467962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 16), ii_467961, 'fill')
            # Calling fill(args, kwargs) (line 497)
            fill_call_result_467966 = invoke(stypy.reporting.localization.Localization(__file__, 497, 16), fill_467962, *[n_467964], **kwargs_467965)
            
            # SSA branch for the else part of an if statement (line 493)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'k' (line 498)
            k_467967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 17), 'k')
            int_467968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 22), 'int')
            # Applying the binary operator '==' (line 498)
            result_eq_467969 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 17), '==', k_467967, int_467968)
            
            # Testing the type of an if condition (line 498)
            if_condition_467970 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 498, 17), result_eq_467969)
            # Assigning a type to the variable 'if_condition_467970' (line 498)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 17), 'if_condition_467970', if_condition_467970)
            # SSA begins for if statement (line 498)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 499):
            
            # Assigning a Call to a Name (line 499):
            
            # Call to empty(...): (line 499)
            # Processing the call arguments (line 499)
            # Getting the type of 'retshape' (line 499)
            retshape_467973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 30), 'retshape', False)
            # Processing the call keyword arguments (line 499)
            # Getting the type of 'float' (line 499)
            float_467974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 45), 'float', False)
            keyword_467975 = float_467974
            kwargs_467976 = {'dtype': keyword_467975}
            # Getting the type of 'np' (line 499)
            np_467971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 21), 'np', False)
            # Obtaining the member 'empty' of a type (line 499)
            empty_467972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 21), np_467971, 'empty')
            # Calling empty(args, kwargs) (line 499)
            empty_call_result_467977 = invoke(stypy.reporting.localization.Localization(__file__, 499, 21), empty_467972, *[retshape_467973], **kwargs_467976)
            
            # Assigning a type to the variable 'dd' (line 499)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'dd', empty_call_result_467977)
            
            # Call to fill(...): (line 500)
            # Processing the call arguments (line 500)
            # Getting the type of 'np' (line 500)
            np_467980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 24), 'np', False)
            # Obtaining the member 'inf' of a type (line 500)
            inf_467981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 24), np_467980, 'inf')
            # Processing the call keyword arguments (line 500)
            kwargs_467982 = {}
            # Getting the type of 'dd' (line 500)
            dd_467978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 16), 'dd', False)
            # Obtaining the member 'fill' of a type (line 500)
            fill_467979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 16), dd_467978, 'fill')
            # Calling fill(args, kwargs) (line 500)
            fill_call_result_467983 = invoke(stypy.reporting.localization.Localization(__file__, 500, 16), fill_467979, *[inf_467981], **kwargs_467982)
            
            
            # Assigning a Call to a Name (line 501):
            
            # Assigning a Call to a Name (line 501):
            
            # Call to empty(...): (line 501)
            # Processing the call arguments (line 501)
            # Getting the type of 'retshape' (line 501)
            retshape_467986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 30), 'retshape', False)
            # Processing the call keyword arguments (line 501)
            # Getting the type of 'int' (line 501)
            int_467987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 45), 'int', False)
            keyword_467988 = int_467987
            kwargs_467989 = {'dtype': keyword_467988}
            # Getting the type of 'np' (line 501)
            np_467984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 21), 'np', False)
            # Obtaining the member 'empty' of a type (line 501)
            empty_467985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 21), np_467984, 'empty')
            # Calling empty(args, kwargs) (line 501)
            empty_call_result_467990 = invoke(stypy.reporting.localization.Localization(__file__, 501, 21), empty_467985, *[retshape_467986], **kwargs_467989)
            
            # Assigning a type to the variable 'ii' (line 501)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 16), 'ii', empty_call_result_467990)
            
            # Call to fill(...): (line 502)
            # Processing the call arguments (line 502)
            # Getting the type of 'self' (line 502)
            self_467993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 24), 'self', False)
            # Obtaining the member 'n' of a type (line 502)
            n_467994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 24), self_467993, 'n')
            # Processing the call keyword arguments (line 502)
            kwargs_467995 = {}
            # Getting the type of 'ii' (line 502)
            ii_467991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'ii', False)
            # Obtaining the member 'fill' of a type (line 502)
            fill_467992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 16), ii_467991, 'fill')
            # Calling fill(args, kwargs) (line 502)
            fill_call_result_467996 = invoke(stypy.reporting.localization.Localization(__file__, 502, 16), fill_467992, *[n_467994], **kwargs_467995)
            
            # SSA branch for the else part of an if statement (line 498)
            module_type_store.open_ssa_branch('else')
            
            # Call to ValueError(...): (line 504)
            # Processing the call arguments (line 504)
            str_467998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 33), 'str', 'Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None')
            # Processing the call keyword arguments (line 504)
            kwargs_467999 = {}
            # Getting the type of 'ValueError' (line 504)
            ValueError_467997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 504)
            ValueError_call_result_468000 = invoke(stypy.reporting.localization.Localization(__file__, 504, 22), ValueError_467997, *[str_467998], **kwargs_467999)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 504, 16), ValueError_call_result_468000, 'raise parameter', BaseException)
            # SSA join for if statement (line 498)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 493)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_467915 and more_types_in_union_467916):
                # SSA join for if statement (line 490)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to ndindex(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'retshape' (line 505)
        retshape_468003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 32), 'retshape', False)
        # Processing the call keyword arguments (line 505)
        kwargs_468004 = {}
        # Getting the type of 'np' (line 505)
        np_468001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 21), 'np', False)
        # Obtaining the member 'ndindex' of a type (line 505)
        ndindex_468002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 21), np_468001, 'ndindex')
        # Calling ndindex(args, kwargs) (line 505)
        ndindex_call_result_468005 = invoke(stypy.reporting.localization.Localization(__file__, 505, 21), ndindex_468002, *[retshape_468003], **kwargs_468004)
        
        # Testing the type of a for loop iterable (line 505)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 505, 12), ndindex_call_result_468005)
        # Getting the type of the for loop variable (line 505)
        for_loop_var_468006 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 505, 12), ndindex_call_result_468005)
        # Assigning a type to the variable 'c' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'c', for_loop_var_468006)
        # SSA begins for a for statement (line 505)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 506):
        
        # Assigning a Call to a Name (line 506):
        
        # Call to __query(...): (line 506)
        # Processing the call arguments (line 506)
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 506)
        c_468009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 38), 'c', False)
        # Getting the type of 'x' (line 506)
        x_468010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 36), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 506)
        getitem___468011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 36), x_468010, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 506)
        subscript_call_result_468012 = invoke(stypy.reporting.localization.Localization(__file__, 506, 36), getitem___468011, c_468009)
        
        # Processing the call keyword arguments (line 506)
        # Getting the type of 'k' (line 506)
        k_468013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 44), 'k', False)
        keyword_468014 = k_468013
        # Getting the type of 'eps' (line 506)
        eps_468015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 51), 'eps', False)
        keyword_468016 = eps_468015
        # Getting the type of 'p' (line 506)
        p_468017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 58), 'p', False)
        keyword_468018 = p_468017
        # Getting the type of 'distance_upper_bound' (line 506)
        distance_upper_bound_468019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 82), 'distance_upper_bound', False)
        keyword_468020 = distance_upper_bound_468019
        kwargs_468021 = {'p': keyword_468018, 'k': keyword_468014, 'eps': keyword_468016, 'distance_upper_bound': keyword_468020}
        # Getting the type of 'self' (line 506)
        self_468007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 23), 'self', False)
        # Obtaining the member '__query' of a type (line 506)
        query_468008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 23), self_468007, '__query')
        # Calling __query(args, kwargs) (line 506)
        query_call_result_468022 = invoke(stypy.reporting.localization.Localization(__file__, 506, 23), query_468008, *[subscript_call_result_468012], **kwargs_468021)
        
        # Assigning a type to the variable 'hits' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 16), 'hits', query_call_result_468022)
        
        # Type idiom detected: calculating its left and rigth part (line 507)
        # Getting the type of 'k' (line 507)
        k_468023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 19), 'k')
        # Getting the type of 'None' (line 507)
        None_468024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 24), 'None')
        
        (may_be_468025, more_types_in_union_468026) = may_be_none(k_468023, None_468024)

        if may_be_468025:

            if more_types_in_union_468026:
                # Runtime conditional SSA (line 507)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a ListComp to a Subscript (line 508):
            
            # Assigning a ListComp to a Subscript (line 508):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'hits' (line 508)
            hits_468028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 44), 'hits')
            comprehension_468029 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 29), hits_468028)
            # Assigning a type to the variable 'd' (line 508)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 29), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 29), comprehension_468029))
            # Assigning a type to the variable 'i' (line 508)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 29), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 29), comprehension_468029))
            # Getting the type of 'd' (line 508)
            d_468027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 29), 'd')
            list_468030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 29), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 29), list_468030, d_468027)
            # Getting the type of 'dd' (line 508)
            dd_468031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 20), 'dd')
            # Getting the type of 'c' (line 508)
            c_468032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 23), 'c')
            # Storing an element on a container (line 508)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 508, 20), dd_468031, (c_468032, list_468030))
            
            # Assigning a ListComp to a Subscript (line 509):
            
            # Assigning a ListComp to a Subscript (line 509):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'hits' (line 509)
            hits_468034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 44), 'hits')
            comprehension_468035 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 29), hits_468034)
            # Assigning a type to the variable 'd' (line 509)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 29), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 29), comprehension_468035))
            # Assigning a type to the variable 'i' (line 509)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 29), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 29), comprehension_468035))
            # Getting the type of 'i' (line 509)
            i_468033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 29), 'i')
            list_468036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 29), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 29), list_468036, i_468033)
            # Getting the type of 'ii' (line 509)
            ii_468037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 20), 'ii')
            # Getting the type of 'c' (line 509)
            c_468038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 23), 'c')
            # Storing an element on a container (line 509)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 20), ii_468037, (c_468038, list_468036))

            if more_types_in_union_468026:
                # Runtime conditional SSA for else branch (line 507)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_468025) or more_types_in_union_468026):
            
            
            # Getting the type of 'k' (line 510)
            k_468039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 21), 'k')
            int_468040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 25), 'int')
            # Applying the binary operator '>' (line 510)
            result_gt_468041 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 21), '>', k_468039, int_468040)
            
            # Testing the type of an if condition (line 510)
            if_condition_468042 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 510, 21), result_gt_468041)
            # Assigning a type to the variable 'if_condition_468042' (line 510)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 21), 'if_condition_468042', if_condition_468042)
            # SSA begins for if statement (line 510)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to range(...): (line 511)
            # Processing the call arguments (line 511)
            
            # Call to len(...): (line 511)
            # Processing the call arguments (line 511)
            # Getting the type of 'hits' (line 511)
            hits_468045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 39), 'hits', False)
            # Processing the call keyword arguments (line 511)
            kwargs_468046 = {}
            # Getting the type of 'len' (line 511)
            len_468044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 35), 'len', False)
            # Calling len(args, kwargs) (line 511)
            len_call_result_468047 = invoke(stypy.reporting.localization.Localization(__file__, 511, 35), len_468044, *[hits_468045], **kwargs_468046)
            
            # Processing the call keyword arguments (line 511)
            kwargs_468048 = {}
            # Getting the type of 'range' (line 511)
            range_468043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 29), 'range', False)
            # Calling range(args, kwargs) (line 511)
            range_call_result_468049 = invoke(stypy.reporting.localization.Localization(__file__, 511, 29), range_468043, *[len_call_result_468047], **kwargs_468048)
            
            # Testing the type of a for loop iterable (line 511)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 511, 20), range_call_result_468049)
            # Getting the type of the for loop variable (line 511)
            for_loop_var_468050 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 511, 20), range_call_result_468049)
            # Assigning a type to the variable 'j' (line 511)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 20), 'j', for_loop_var_468050)
            # SSA begins for a for statement (line 511)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Tuple (line 512):
            
            # Assigning a Subscript to a Name (line 512):
            
            # Obtaining the type of the subscript
            int_468051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 24), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 512)
            j_468052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 54), 'j')
            # Getting the type of 'hits' (line 512)
            hits_468053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 49), 'hits')
            # Obtaining the member '__getitem__' of a type (line 512)
            getitem___468054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 49), hits_468053, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 512)
            subscript_call_result_468055 = invoke(stypy.reporting.localization.Localization(__file__, 512, 49), getitem___468054, j_468052)
            
            # Obtaining the member '__getitem__' of a type (line 512)
            getitem___468056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 24), subscript_call_result_468055, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 512)
            subscript_call_result_468057 = invoke(stypy.reporting.localization.Localization(__file__, 512, 24), getitem___468056, int_468051)
            
            # Assigning a type to the variable 'tuple_var_assignment_466770' (line 512)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 24), 'tuple_var_assignment_466770', subscript_call_result_468057)
            
            # Assigning a Subscript to a Name (line 512):
            
            # Obtaining the type of the subscript
            int_468058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 24), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 512)
            j_468059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 54), 'j')
            # Getting the type of 'hits' (line 512)
            hits_468060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 49), 'hits')
            # Obtaining the member '__getitem__' of a type (line 512)
            getitem___468061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 49), hits_468060, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 512)
            subscript_call_result_468062 = invoke(stypy.reporting.localization.Localization(__file__, 512, 49), getitem___468061, j_468059)
            
            # Obtaining the member '__getitem__' of a type (line 512)
            getitem___468063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 24), subscript_call_result_468062, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 512)
            subscript_call_result_468064 = invoke(stypy.reporting.localization.Localization(__file__, 512, 24), getitem___468063, int_468058)
            
            # Assigning a type to the variable 'tuple_var_assignment_466771' (line 512)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 24), 'tuple_var_assignment_466771', subscript_call_result_468064)
            
            # Assigning a Name to a Subscript (line 512):
            # Getting the type of 'tuple_var_assignment_466770' (line 512)
            tuple_var_assignment_466770_468065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 24), 'tuple_var_assignment_466770')
            # Getting the type of 'dd' (line 512)
            dd_468066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 24), 'dd')
            # Getting the type of 'c' (line 512)
            c_468067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 27), 'c')
            
            # Obtaining an instance of the builtin type 'tuple' (line 512)
            tuple_468068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 512)
            # Adding element type (line 512)
            # Getting the type of 'j' (line 512)
            j_468069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 30), 'j')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 30), tuple_468068, j_468069)
            
            # Applying the binary operator '+' (line 512)
            result_add_468070 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 27), '+', c_468067, tuple_468068)
            
            # Storing an element on a container (line 512)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 24), dd_468066, (result_add_468070, tuple_var_assignment_466770_468065))
            
            # Assigning a Name to a Subscript (line 512):
            # Getting the type of 'tuple_var_assignment_466771' (line 512)
            tuple_var_assignment_466771_468071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 24), 'tuple_var_assignment_466771')
            # Getting the type of 'ii' (line 512)
            ii_468072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 36), 'ii')
            # Getting the type of 'c' (line 512)
            c_468073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 39), 'c')
            
            # Obtaining an instance of the builtin type 'tuple' (line 512)
            tuple_468074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 42), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 512)
            # Adding element type (line 512)
            # Getting the type of 'j' (line 512)
            j_468075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 42), 'j')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 42), tuple_468074, j_468075)
            
            # Applying the binary operator '+' (line 512)
            result_add_468076 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 39), '+', c_468073, tuple_468074)
            
            # Storing an element on a container (line 512)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 36), ii_468072, (result_add_468076, tuple_var_assignment_466771_468071))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 510)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'k' (line 513)
            k_468077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 21), 'k')
            int_468078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 26), 'int')
            # Applying the binary operator '==' (line 513)
            result_eq_468079 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 21), '==', k_468077, int_468078)
            
            # Testing the type of an if condition (line 513)
            if_condition_468080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 513, 21), result_eq_468079)
            # Assigning a type to the variable 'if_condition_468080' (line 513)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 21), 'if_condition_468080', if_condition_468080)
            # SSA begins for if statement (line 513)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            
            # Call to len(...): (line 514)
            # Processing the call arguments (line 514)
            # Getting the type of 'hits' (line 514)
            hits_468082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 27), 'hits', False)
            # Processing the call keyword arguments (line 514)
            kwargs_468083 = {}
            # Getting the type of 'len' (line 514)
            len_468081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 23), 'len', False)
            # Calling len(args, kwargs) (line 514)
            len_call_result_468084 = invoke(stypy.reporting.localization.Localization(__file__, 514, 23), len_468081, *[hits_468082], **kwargs_468083)
            
            int_468085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 35), 'int')
            # Applying the binary operator '>' (line 514)
            result_gt_468086 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 23), '>', len_call_result_468084, int_468085)
            
            # Testing the type of an if condition (line 514)
            if_condition_468087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 514, 20), result_gt_468086)
            # Assigning a type to the variable 'if_condition_468087' (line 514)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 20), 'if_condition_468087', if_condition_468087)
            # SSA begins for if statement (line 514)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Tuple (line 515):
            
            # Assigning a Subscript to a Name (line 515):
            
            # Obtaining the type of the subscript
            int_468088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 24), 'int')
            
            # Obtaining the type of the subscript
            int_468089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 44), 'int')
            # Getting the type of 'hits' (line 515)
            hits_468090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 39), 'hits')
            # Obtaining the member '__getitem__' of a type (line 515)
            getitem___468091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 39), hits_468090, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 515)
            subscript_call_result_468092 = invoke(stypy.reporting.localization.Localization(__file__, 515, 39), getitem___468091, int_468089)
            
            # Obtaining the member '__getitem__' of a type (line 515)
            getitem___468093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 24), subscript_call_result_468092, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 515)
            subscript_call_result_468094 = invoke(stypy.reporting.localization.Localization(__file__, 515, 24), getitem___468093, int_468088)
            
            # Assigning a type to the variable 'tuple_var_assignment_466772' (line 515)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 24), 'tuple_var_assignment_466772', subscript_call_result_468094)
            
            # Assigning a Subscript to a Name (line 515):
            
            # Obtaining the type of the subscript
            int_468095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 24), 'int')
            
            # Obtaining the type of the subscript
            int_468096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 44), 'int')
            # Getting the type of 'hits' (line 515)
            hits_468097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 39), 'hits')
            # Obtaining the member '__getitem__' of a type (line 515)
            getitem___468098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 39), hits_468097, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 515)
            subscript_call_result_468099 = invoke(stypy.reporting.localization.Localization(__file__, 515, 39), getitem___468098, int_468096)
            
            # Obtaining the member '__getitem__' of a type (line 515)
            getitem___468100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 24), subscript_call_result_468099, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 515)
            subscript_call_result_468101 = invoke(stypy.reporting.localization.Localization(__file__, 515, 24), getitem___468100, int_468095)
            
            # Assigning a type to the variable 'tuple_var_assignment_466773' (line 515)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 24), 'tuple_var_assignment_466773', subscript_call_result_468101)
            
            # Assigning a Name to a Subscript (line 515):
            # Getting the type of 'tuple_var_assignment_466772' (line 515)
            tuple_var_assignment_466772_468102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 24), 'tuple_var_assignment_466772')
            # Getting the type of 'dd' (line 515)
            dd_468103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 24), 'dd')
            # Getting the type of 'c' (line 515)
            c_468104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 27), 'c')
            # Storing an element on a container (line 515)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 24), dd_468103, (c_468104, tuple_var_assignment_466772_468102))
            
            # Assigning a Name to a Subscript (line 515):
            # Getting the type of 'tuple_var_assignment_466773' (line 515)
            tuple_var_assignment_466773_468105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 24), 'tuple_var_assignment_466773')
            # Getting the type of 'ii' (line 515)
            ii_468106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 31), 'ii')
            # Getting the type of 'c' (line 515)
            c_468107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 34), 'c')
            # Storing an element on a container (line 515)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 31), ii_468106, (c_468107, tuple_var_assignment_466773_468105))
            # SSA branch for the else part of an if statement (line 514)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Subscript (line 517):
            
            # Assigning a Attribute to a Subscript (line 517):
            # Getting the type of 'np' (line 517)
            np_468108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 32), 'np')
            # Obtaining the member 'inf' of a type (line 517)
            inf_468109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 32), np_468108, 'inf')
            # Getting the type of 'dd' (line 517)
            dd_468110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 24), 'dd')
            # Getting the type of 'c' (line 517)
            c_468111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 27), 'c')
            # Storing an element on a container (line 517)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 24), dd_468110, (c_468111, inf_468109))
            
            # Assigning a Attribute to a Subscript (line 518):
            
            # Assigning a Attribute to a Subscript (line 518):
            # Getting the type of 'self' (line 518)
            self_468112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 32), 'self')
            # Obtaining the member 'n' of a type (line 518)
            n_468113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 32), self_468112, 'n')
            # Getting the type of 'ii' (line 518)
            ii_468114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 24), 'ii')
            # Getting the type of 'c' (line 518)
            c_468115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 27), 'c')
            # Storing an element on a container (line 518)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 24), ii_468114, (c_468115, n_468113))
            # SSA join for if statement (line 514)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 513)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 510)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_468025 and more_types_in_union_468026):
                # SSA join for if statement (line 507)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 519)
        tuple_468116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 519)
        # Adding element type (line 519)
        # Getting the type of 'dd' (line 519)
        dd_468117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 19), 'dd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 19), tuple_468116, dd_468117)
        # Adding element type (line 519)
        # Getting the type of 'ii' (line 519)
        ii_468118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 23), 'ii')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 19), tuple_468116, ii_468118)
        
        # Assigning a type to the variable 'stypy_return_type' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'stypy_return_type', tuple_468116)
        # SSA branch for the else part of an if statement (line 489)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 521):
        
        # Assigning a Call to a Name (line 521):
        
        # Call to __query(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'x' (line 521)
        x_468121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 32), 'x', False)
        # Processing the call keyword arguments (line 521)
        # Getting the type of 'k' (line 521)
        k_468122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 37), 'k', False)
        keyword_468123 = k_468122
        # Getting the type of 'eps' (line 521)
        eps_468124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 44), 'eps', False)
        keyword_468125 = eps_468124
        # Getting the type of 'p' (line 521)
        p_468126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 51), 'p', False)
        keyword_468127 = p_468126
        # Getting the type of 'distance_upper_bound' (line 521)
        distance_upper_bound_468128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 75), 'distance_upper_bound', False)
        keyword_468129 = distance_upper_bound_468128
        kwargs_468130 = {'p': keyword_468127, 'k': keyword_468123, 'eps': keyword_468125, 'distance_upper_bound': keyword_468129}
        # Getting the type of 'self' (line 521)
        self_468119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 19), 'self', False)
        # Obtaining the member '__query' of a type (line 521)
        query_468120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 19), self_468119, '__query')
        # Calling __query(args, kwargs) (line 521)
        query_call_result_468131 = invoke(stypy.reporting.localization.Localization(__file__, 521, 19), query_468120, *[x_468121], **kwargs_468130)
        
        # Assigning a type to the variable 'hits' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'hits', query_call_result_468131)
        
        # Type idiom detected: calculating its left and rigth part (line 522)
        # Getting the type of 'k' (line 522)
        k_468132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'k')
        # Getting the type of 'None' (line 522)
        None_468133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 20), 'None')
        
        (may_be_468134, more_types_in_union_468135) = may_be_none(k_468132, None_468133)

        if may_be_468134:

            if more_types_in_union_468135:
                # Runtime conditional SSA (line 522)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Obtaining an instance of the builtin type 'tuple' (line 523)
            tuple_468136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 23), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 523)
            # Adding element type (line 523)
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'hits' (line 523)
            hits_468138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 39), 'hits')
            comprehension_468139 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 24), hits_468138)
            # Assigning a type to the variable 'd' (line 523)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 24), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 24), comprehension_468139))
            # Assigning a type to the variable 'i' (line 523)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 24), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 24), comprehension_468139))
            # Getting the type of 'd' (line 523)
            d_468137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 24), 'd')
            list_468140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 24), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 24), list_468140, d_468137)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 23), tuple_468136, list_468140)
            # Adding element type (line 523)
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'hits' (line 523)
            hits_468142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 62), 'hits')
            comprehension_468143 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 47), hits_468142)
            # Assigning a type to the variable 'd' (line 523)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 47), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 47), comprehension_468143))
            # Assigning a type to the variable 'i' (line 523)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 47), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 47), comprehension_468143))
            # Getting the type of 'i' (line 523)
            i_468141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 47), 'i')
            list_468144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 47), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 47), list_468144, i_468141)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 23), tuple_468136, list_468144)
            
            # Assigning a type to the variable 'stypy_return_type' (line 523)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'stypy_return_type', tuple_468136)

            if more_types_in_union_468135:
                # Runtime conditional SSA for else branch (line 522)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_468134) or more_types_in_union_468135):
            
            
            # Getting the type of 'k' (line 524)
            k_468145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 17), 'k')
            int_468146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 22), 'int')
            # Applying the binary operator '==' (line 524)
            result_eq_468147 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 17), '==', k_468145, int_468146)
            
            # Testing the type of an if condition (line 524)
            if_condition_468148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 524, 17), result_eq_468147)
            # Assigning a type to the variable 'if_condition_468148' (line 524)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 17), 'if_condition_468148', if_condition_468148)
            # SSA begins for if statement (line 524)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            
            # Call to len(...): (line 525)
            # Processing the call arguments (line 525)
            # Getting the type of 'hits' (line 525)
            hits_468150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 23), 'hits', False)
            # Processing the call keyword arguments (line 525)
            kwargs_468151 = {}
            # Getting the type of 'len' (line 525)
            len_468149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 19), 'len', False)
            # Calling len(args, kwargs) (line 525)
            len_call_result_468152 = invoke(stypy.reporting.localization.Localization(__file__, 525, 19), len_468149, *[hits_468150], **kwargs_468151)
            
            int_468153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 31), 'int')
            # Applying the binary operator '>' (line 525)
            result_gt_468154 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 19), '>', len_call_result_468152, int_468153)
            
            # Testing the type of an if condition (line 525)
            if_condition_468155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 525, 16), result_gt_468154)
            # Assigning a type to the variable 'if_condition_468155' (line 525)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'if_condition_468155', if_condition_468155)
            # SSA begins for if statement (line 525)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_468156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 32), 'int')
            # Getting the type of 'hits' (line 526)
            hits_468157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 27), 'hits')
            # Obtaining the member '__getitem__' of a type (line 526)
            getitem___468158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 27), hits_468157, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 526)
            subscript_call_result_468159 = invoke(stypy.reporting.localization.Localization(__file__, 526, 27), getitem___468158, int_468156)
            
            # Assigning a type to the variable 'stypy_return_type' (line 526)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 20), 'stypy_return_type', subscript_call_result_468159)
            # SSA branch for the else part of an if statement (line 525)
            module_type_store.open_ssa_branch('else')
            
            # Obtaining an instance of the builtin type 'tuple' (line 528)
            tuple_468160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 528)
            # Adding element type (line 528)
            # Getting the type of 'np' (line 528)
            np_468161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 27), 'np')
            # Obtaining the member 'inf' of a type (line 528)
            inf_468162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 27), np_468161, 'inf')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 27), tuple_468160, inf_468162)
            # Adding element type (line 528)
            # Getting the type of 'self' (line 528)
            self_468163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 35), 'self')
            # Obtaining the member 'n' of a type (line 528)
            n_468164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 35), self_468163, 'n')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 27), tuple_468160, n_468164)
            
            # Assigning a type to the variable 'stypy_return_type' (line 528)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 20), 'stypy_return_type', tuple_468160)
            # SSA join for if statement (line 525)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 524)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'k' (line 529)
            k_468165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 17), 'k')
            int_468166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 21), 'int')
            # Applying the binary operator '>' (line 529)
            result_gt_468167 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 17), '>', k_468165, int_468166)
            
            # Testing the type of an if condition (line 529)
            if_condition_468168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 529, 17), result_gt_468167)
            # Assigning a type to the variable 'if_condition_468168' (line 529)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 17), 'if_condition_468168', if_condition_468168)
            # SSA begins for if statement (line 529)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 530):
            
            # Assigning a Call to a Name (line 530):
            
            # Call to empty(...): (line 530)
            # Processing the call arguments (line 530)
            # Getting the type of 'k' (line 530)
            k_468171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 30), 'k', False)
            # Processing the call keyword arguments (line 530)
            # Getting the type of 'float' (line 530)
            float_468172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 38), 'float', False)
            keyword_468173 = float_468172
            kwargs_468174 = {'dtype': keyword_468173}
            # Getting the type of 'np' (line 530)
            np_468169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 21), 'np', False)
            # Obtaining the member 'empty' of a type (line 530)
            empty_468170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 21), np_468169, 'empty')
            # Calling empty(args, kwargs) (line 530)
            empty_call_result_468175 = invoke(stypy.reporting.localization.Localization(__file__, 530, 21), empty_468170, *[k_468171], **kwargs_468174)
            
            # Assigning a type to the variable 'dd' (line 530)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 16), 'dd', empty_call_result_468175)
            
            # Call to fill(...): (line 531)
            # Processing the call arguments (line 531)
            # Getting the type of 'np' (line 531)
            np_468178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 24), 'np', False)
            # Obtaining the member 'inf' of a type (line 531)
            inf_468179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 24), np_468178, 'inf')
            # Processing the call keyword arguments (line 531)
            kwargs_468180 = {}
            # Getting the type of 'dd' (line 531)
            dd_468176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 16), 'dd', False)
            # Obtaining the member 'fill' of a type (line 531)
            fill_468177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 16), dd_468176, 'fill')
            # Calling fill(args, kwargs) (line 531)
            fill_call_result_468181 = invoke(stypy.reporting.localization.Localization(__file__, 531, 16), fill_468177, *[inf_468179], **kwargs_468180)
            
            
            # Assigning a Call to a Name (line 532):
            
            # Assigning a Call to a Name (line 532):
            
            # Call to empty(...): (line 532)
            # Processing the call arguments (line 532)
            # Getting the type of 'k' (line 532)
            k_468184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 30), 'k', False)
            # Processing the call keyword arguments (line 532)
            # Getting the type of 'int' (line 532)
            int_468185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 38), 'int', False)
            keyword_468186 = int_468185
            kwargs_468187 = {'dtype': keyword_468186}
            # Getting the type of 'np' (line 532)
            np_468182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 21), 'np', False)
            # Obtaining the member 'empty' of a type (line 532)
            empty_468183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 21), np_468182, 'empty')
            # Calling empty(args, kwargs) (line 532)
            empty_call_result_468188 = invoke(stypy.reporting.localization.Localization(__file__, 532, 21), empty_468183, *[k_468184], **kwargs_468187)
            
            # Assigning a type to the variable 'ii' (line 532)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 16), 'ii', empty_call_result_468188)
            
            # Call to fill(...): (line 533)
            # Processing the call arguments (line 533)
            # Getting the type of 'self' (line 533)
            self_468191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 24), 'self', False)
            # Obtaining the member 'n' of a type (line 533)
            n_468192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 24), self_468191, 'n')
            # Processing the call keyword arguments (line 533)
            kwargs_468193 = {}
            # Getting the type of 'ii' (line 533)
            ii_468189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 16), 'ii', False)
            # Obtaining the member 'fill' of a type (line 533)
            fill_468190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 16), ii_468189, 'fill')
            # Calling fill(args, kwargs) (line 533)
            fill_call_result_468194 = invoke(stypy.reporting.localization.Localization(__file__, 533, 16), fill_468190, *[n_468192], **kwargs_468193)
            
            
            
            # Call to range(...): (line 534)
            # Processing the call arguments (line 534)
            
            # Call to len(...): (line 534)
            # Processing the call arguments (line 534)
            # Getting the type of 'hits' (line 534)
            hits_468197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 35), 'hits', False)
            # Processing the call keyword arguments (line 534)
            kwargs_468198 = {}
            # Getting the type of 'len' (line 534)
            len_468196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 31), 'len', False)
            # Calling len(args, kwargs) (line 534)
            len_call_result_468199 = invoke(stypy.reporting.localization.Localization(__file__, 534, 31), len_468196, *[hits_468197], **kwargs_468198)
            
            # Processing the call keyword arguments (line 534)
            kwargs_468200 = {}
            # Getting the type of 'range' (line 534)
            range_468195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 25), 'range', False)
            # Calling range(args, kwargs) (line 534)
            range_call_result_468201 = invoke(stypy.reporting.localization.Localization(__file__, 534, 25), range_468195, *[len_call_result_468199], **kwargs_468200)
            
            # Testing the type of a for loop iterable (line 534)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 534, 16), range_call_result_468201)
            # Getting the type of the for loop variable (line 534)
            for_loop_var_468202 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 534, 16), range_call_result_468201)
            # Assigning a type to the variable 'j' (line 534)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'j', for_loop_var_468202)
            # SSA begins for a for statement (line 534)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Tuple (line 535):
            
            # Assigning a Subscript to a Name (line 535):
            
            # Obtaining the type of the subscript
            int_468203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 20), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 535)
            j_468204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 40), 'j')
            # Getting the type of 'hits' (line 535)
            hits_468205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 35), 'hits')
            # Obtaining the member '__getitem__' of a type (line 535)
            getitem___468206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 35), hits_468205, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 535)
            subscript_call_result_468207 = invoke(stypy.reporting.localization.Localization(__file__, 535, 35), getitem___468206, j_468204)
            
            # Obtaining the member '__getitem__' of a type (line 535)
            getitem___468208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 20), subscript_call_result_468207, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 535)
            subscript_call_result_468209 = invoke(stypy.reporting.localization.Localization(__file__, 535, 20), getitem___468208, int_468203)
            
            # Assigning a type to the variable 'tuple_var_assignment_466774' (line 535)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 20), 'tuple_var_assignment_466774', subscript_call_result_468209)
            
            # Assigning a Subscript to a Name (line 535):
            
            # Obtaining the type of the subscript
            int_468210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 20), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 535)
            j_468211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 40), 'j')
            # Getting the type of 'hits' (line 535)
            hits_468212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 35), 'hits')
            # Obtaining the member '__getitem__' of a type (line 535)
            getitem___468213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 35), hits_468212, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 535)
            subscript_call_result_468214 = invoke(stypy.reporting.localization.Localization(__file__, 535, 35), getitem___468213, j_468211)
            
            # Obtaining the member '__getitem__' of a type (line 535)
            getitem___468215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 20), subscript_call_result_468214, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 535)
            subscript_call_result_468216 = invoke(stypy.reporting.localization.Localization(__file__, 535, 20), getitem___468215, int_468210)
            
            # Assigning a type to the variable 'tuple_var_assignment_466775' (line 535)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 20), 'tuple_var_assignment_466775', subscript_call_result_468216)
            
            # Assigning a Name to a Subscript (line 535):
            # Getting the type of 'tuple_var_assignment_466774' (line 535)
            tuple_var_assignment_466774_468217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 20), 'tuple_var_assignment_466774')
            # Getting the type of 'dd' (line 535)
            dd_468218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 20), 'dd')
            # Getting the type of 'j' (line 535)
            j_468219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 23), 'j')
            # Storing an element on a container (line 535)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 20), dd_468218, (j_468219, tuple_var_assignment_466774_468217))
            
            # Assigning a Name to a Subscript (line 535):
            # Getting the type of 'tuple_var_assignment_466775' (line 535)
            tuple_var_assignment_466775_468220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 20), 'tuple_var_assignment_466775')
            # Getting the type of 'ii' (line 535)
            ii_468221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 27), 'ii')
            # Getting the type of 'j' (line 535)
            j_468222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 30), 'j')
            # Storing an element on a container (line 535)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 27), ii_468221, (j_468222, tuple_var_assignment_466775_468220))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 536)
            tuple_468223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 23), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 536)
            # Adding element type (line 536)
            # Getting the type of 'dd' (line 536)
            dd_468224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 23), 'dd')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 23), tuple_468223, dd_468224)
            # Adding element type (line 536)
            # Getting the type of 'ii' (line 536)
            ii_468225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 27), 'ii')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 23), tuple_468223, ii_468225)
            
            # Assigning a type to the variable 'stypy_return_type' (line 536)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'stypy_return_type', tuple_468223)
            # SSA branch for the else part of an if statement (line 529)
            module_type_store.open_ssa_branch('else')
            
            # Call to ValueError(...): (line 538)
            # Processing the call arguments (line 538)
            str_468227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 33), 'str', 'Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None')
            # Processing the call keyword arguments (line 538)
            kwargs_468228 = {}
            # Getting the type of 'ValueError' (line 538)
            ValueError_468226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 538)
            ValueError_call_result_468229 = invoke(stypy.reporting.localization.Localization(__file__, 538, 22), ValueError_468226, *[str_468227], **kwargs_468228)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 538, 16), ValueError_call_result_468229, 'raise parameter', BaseException)
            # SSA join for if statement (line 529)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 524)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_468134 and more_types_in_union_468135):
                # SSA join for if statement (line 522)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 489)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'query(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'query' in the type store
        # Getting the type of 'stypy_return_type' (line 400)
        stypy_return_type_468230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_468230)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'query'
        return stypy_return_type_468230


    @norecursion
    def __query_ball_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_468231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 41), 'float')
        int_468232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 49), 'int')
        defaults = [float_468231, int_468232]
        # Create a new context for function '__query_ball_point'
        module_type_store = module_type_store.open_function_context('__query_ball_point', 540, 4, False)
        # Assigning a type to the variable 'self' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KDTree.__query_ball_point.__dict__.__setitem__('stypy_localization', localization)
        KDTree.__query_ball_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KDTree.__query_ball_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        KDTree.__query_ball_point.__dict__.__setitem__('stypy_function_name', 'KDTree.__query_ball_point')
        KDTree.__query_ball_point.__dict__.__setitem__('stypy_param_names_list', ['x', 'r', 'p', 'eps'])
        KDTree.__query_ball_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        KDTree.__query_ball_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KDTree.__query_ball_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        KDTree.__query_ball_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        KDTree.__query_ball_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KDTree.__query_ball_point.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KDTree.__query_ball_point', ['x', 'r', 'p', 'eps'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__query_ball_point', localization, ['x', 'r', 'p', 'eps'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__query_ball_point(...)' code ##################

        
        # Assigning a Call to a Name (line 541):
        
        # Assigning a Call to a Name (line 541):
        
        # Call to Rectangle(...): (line 541)
        # Processing the call arguments (line 541)
        # Getting the type of 'self' (line 541)
        self_468234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 22), 'self', False)
        # Obtaining the member 'maxes' of a type (line 541)
        maxes_468235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 22), self_468234, 'maxes')
        # Getting the type of 'self' (line 541)
        self_468236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 34), 'self', False)
        # Obtaining the member 'mins' of a type (line 541)
        mins_468237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 34), self_468236, 'mins')
        # Processing the call keyword arguments (line 541)
        kwargs_468238 = {}
        # Getting the type of 'Rectangle' (line 541)
        Rectangle_468233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 541)
        Rectangle_call_result_468239 = invoke(stypy.reporting.localization.Localization(__file__, 541, 12), Rectangle_468233, *[maxes_468235, mins_468237], **kwargs_468238)
        
        # Assigning a type to the variable 'R' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'R', Rectangle_call_result_468239)

        @norecursion
        def traverse_checking(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'traverse_checking'
            module_type_store = module_type_store.open_function_context('traverse_checking', 543, 8, False)
            
            # Passed parameters checking function
            traverse_checking.stypy_localization = localization
            traverse_checking.stypy_type_of_self = None
            traverse_checking.stypy_type_store = module_type_store
            traverse_checking.stypy_function_name = 'traverse_checking'
            traverse_checking.stypy_param_names_list = ['node', 'rect']
            traverse_checking.stypy_varargs_param_name = None
            traverse_checking.stypy_kwargs_param_name = None
            traverse_checking.stypy_call_defaults = defaults
            traverse_checking.stypy_call_varargs = varargs
            traverse_checking.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'traverse_checking', ['node', 'rect'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'traverse_checking', localization, ['node', 'rect'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'traverse_checking(...)' code ##################

            
            
            
            # Call to min_distance_point(...): (line 544)
            # Processing the call arguments (line 544)
            # Getting the type of 'x' (line 544)
            x_468242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 39), 'x', False)
            # Getting the type of 'p' (line 544)
            p_468243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 42), 'p', False)
            # Processing the call keyword arguments (line 544)
            kwargs_468244 = {}
            # Getting the type of 'rect' (line 544)
            rect_468240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 15), 'rect', False)
            # Obtaining the member 'min_distance_point' of a type (line 544)
            min_distance_point_468241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 15), rect_468240, 'min_distance_point')
            # Calling min_distance_point(args, kwargs) (line 544)
            min_distance_point_call_result_468245 = invoke(stypy.reporting.localization.Localization(__file__, 544, 15), min_distance_point_468241, *[x_468242, p_468243], **kwargs_468244)
            
            # Getting the type of 'r' (line 544)
            r_468246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 47), 'r')
            float_468247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 52), 'float')
            # Getting the type of 'eps' (line 544)
            eps_468248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 57), 'eps')
            # Applying the binary operator '+' (line 544)
            result_add_468249 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 52), '+', float_468247, eps_468248)
            
            # Applying the binary operator 'div' (line 544)
            result_div_468250 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 47), 'div', r_468246, result_add_468249)
            
            # Applying the binary operator '>' (line 544)
            result_gt_468251 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 15), '>', min_distance_point_call_result_468245, result_div_468250)
            
            # Testing the type of an if condition (line 544)
            if_condition_468252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 544, 12), result_gt_468251)
            # Assigning a type to the variable 'if_condition_468252' (line 544)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'if_condition_468252', if_condition_468252)
            # SSA begins for if statement (line 544)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'list' (line 545)
            list_468253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 545)
            
            # Assigning a type to the variable 'stypy_return_type' (line 545)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'stypy_return_type', list_468253)
            # SSA branch for the else part of an if statement (line 544)
            module_type_store.open_ssa_branch('else')
            
            
            
            # Call to max_distance_point(...): (line 546)
            # Processing the call arguments (line 546)
            # Getting the type of 'x' (line 546)
            x_468256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 41), 'x', False)
            # Getting the type of 'p' (line 546)
            p_468257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 44), 'p', False)
            # Processing the call keyword arguments (line 546)
            kwargs_468258 = {}
            # Getting the type of 'rect' (line 546)
            rect_468254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 17), 'rect', False)
            # Obtaining the member 'max_distance_point' of a type (line 546)
            max_distance_point_468255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 17), rect_468254, 'max_distance_point')
            # Calling max_distance_point(args, kwargs) (line 546)
            max_distance_point_call_result_468259 = invoke(stypy.reporting.localization.Localization(__file__, 546, 17), max_distance_point_468255, *[x_468256, p_468257], **kwargs_468258)
            
            # Getting the type of 'r' (line 546)
            r_468260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 49), 'r')
            float_468261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 54), 'float')
            # Getting the type of 'eps' (line 546)
            eps_468262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 59), 'eps')
            # Applying the binary operator '+' (line 546)
            result_add_468263 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 54), '+', float_468261, eps_468262)
            
            # Applying the binary operator '*' (line 546)
            result_mul_468264 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 49), '*', r_468260, result_add_468263)
            
            # Applying the binary operator '<' (line 546)
            result_lt_468265 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 17), '<', max_distance_point_call_result_468259, result_mul_468264)
            
            # Testing the type of an if condition (line 546)
            if_condition_468266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 546, 17), result_lt_468265)
            # Assigning a type to the variable 'if_condition_468266' (line 546)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 17), 'if_condition_468266', if_condition_468266)
            # SSA begins for if statement (line 546)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to traverse_no_checking(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'node' (line 547)
            node_468268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 44), 'node', False)
            # Processing the call keyword arguments (line 547)
            kwargs_468269 = {}
            # Getting the type of 'traverse_no_checking' (line 547)
            traverse_no_checking_468267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 23), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 547)
            traverse_no_checking_call_result_468270 = invoke(stypy.reporting.localization.Localization(__file__, 547, 23), traverse_no_checking_468267, *[node_468268], **kwargs_468269)
            
            # Assigning a type to the variable 'stypy_return_type' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'stypy_return_type', traverse_no_checking_call_result_468270)
            # SSA branch for the else part of an if statement (line 546)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 548)
            # Processing the call arguments (line 548)
            # Getting the type of 'node' (line 548)
            node_468272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 28), 'node', False)
            # Getting the type of 'KDTree' (line 548)
            KDTree_468273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 34), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 548)
            leafnode_468274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 34), KDTree_468273, 'leafnode')
            # Processing the call keyword arguments (line 548)
            kwargs_468275 = {}
            # Getting the type of 'isinstance' (line 548)
            isinstance_468271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 548)
            isinstance_call_result_468276 = invoke(stypy.reporting.localization.Localization(__file__, 548, 17), isinstance_468271, *[node_468272, leafnode_468274], **kwargs_468275)
            
            # Testing the type of an if condition (line 548)
            if_condition_468277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 17), isinstance_call_result_468276)
            # Assigning a type to the variable 'if_condition_468277' (line 548)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 17), 'if_condition_468277', if_condition_468277)
            # SSA begins for if statement (line 548)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 549):
            
            # Assigning a Subscript to a Name (line 549):
            
            # Obtaining the type of the subscript
            # Getting the type of 'node' (line 549)
            node_468278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 30), 'node')
            # Obtaining the member 'idx' of a type (line 549)
            idx_468279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 30), node_468278, 'idx')
            # Getting the type of 'self' (line 549)
            self_468280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'self')
            # Obtaining the member 'data' of a type (line 549)
            data_468281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 20), self_468280, 'data')
            # Obtaining the member '__getitem__' of a type (line 549)
            getitem___468282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 20), data_468281, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 549)
            subscript_call_result_468283 = invoke(stypy.reporting.localization.Localization(__file__, 549, 20), getitem___468282, idx_468279)
            
            # Assigning a type to the variable 'd' (line 549)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'd', subscript_call_result_468283)
            
            # Call to tolist(...): (line 550)
            # Processing the call keyword arguments (line 550)
            kwargs_468297 = {}
            
            # Obtaining the type of the subscript
            
            
            # Call to minkowski_distance(...): (line 550)
            # Processing the call arguments (line 550)
            # Getting the type of 'd' (line 550)
            d_468285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 51), 'd', False)
            # Getting the type of 'x' (line 550)
            x_468286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 54), 'x', False)
            # Getting the type of 'p' (line 550)
            p_468287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 57), 'p', False)
            # Processing the call keyword arguments (line 550)
            kwargs_468288 = {}
            # Getting the type of 'minkowski_distance' (line 550)
            minkowski_distance_468284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 32), 'minkowski_distance', False)
            # Calling minkowski_distance(args, kwargs) (line 550)
            minkowski_distance_call_result_468289 = invoke(stypy.reporting.localization.Localization(__file__, 550, 32), minkowski_distance_468284, *[d_468285, x_468286, p_468287], **kwargs_468288)
            
            # Getting the type of 'r' (line 550)
            r_468290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 63), 'r', False)
            # Applying the binary operator '<=' (line 550)
            result_le_468291 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 32), '<=', minkowski_distance_call_result_468289, r_468290)
            
            # Getting the type of 'node' (line 550)
            node_468292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 23), 'node', False)
            # Obtaining the member 'idx' of a type (line 550)
            idx_468293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 23), node_468292, 'idx')
            # Obtaining the member '__getitem__' of a type (line 550)
            getitem___468294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 23), idx_468293, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 550)
            subscript_call_result_468295 = invoke(stypy.reporting.localization.Localization(__file__, 550, 23), getitem___468294, result_le_468291)
            
            # Obtaining the member 'tolist' of a type (line 550)
            tolist_468296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 23), subscript_call_result_468295, 'tolist')
            # Calling tolist(args, kwargs) (line 550)
            tolist_call_result_468298 = invoke(stypy.reporting.localization.Localization(__file__, 550, 23), tolist_468296, *[], **kwargs_468297)
            
            # Assigning a type to the variable 'stypy_return_type' (line 550)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 16), 'stypy_return_type', tolist_call_result_468298)
            # SSA branch for the else part of an if statement (line 548)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 552):
            
            # Assigning a Subscript to a Name (line 552):
            
            # Obtaining the type of the subscript
            int_468299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 16), 'int')
            
            # Call to split(...): (line 552)
            # Processing the call arguments (line 552)
            # Getting the type of 'node' (line 552)
            node_468302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 43), 'node', False)
            # Obtaining the member 'split_dim' of a type (line 552)
            split_dim_468303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 43), node_468302, 'split_dim')
            # Getting the type of 'node' (line 552)
            node_468304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 59), 'node', False)
            # Obtaining the member 'split' of a type (line 552)
            split_468305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 59), node_468304, 'split')
            # Processing the call keyword arguments (line 552)
            kwargs_468306 = {}
            # Getting the type of 'rect' (line 552)
            rect_468300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 32), 'rect', False)
            # Obtaining the member 'split' of a type (line 552)
            split_468301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 32), rect_468300, 'split')
            # Calling split(args, kwargs) (line 552)
            split_call_result_468307 = invoke(stypy.reporting.localization.Localization(__file__, 552, 32), split_468301, *[split_dim_468303, split_468305], **kwargs_468306)
            
            # Obtaining the member '__getitem__' of a type (line 552)
            getitem___468308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 16), split_call_result_468307, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 552)
            subscript_call_result_468309 = invoke(stypy.reporting.localization.Localization(__file__, 552, 16), getitem___468308, int_468299)
            
            # Assigning a type to the variable 'tuple_var_assignment_466776' (line 552)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'tuple_var_assignment_466776', subscript_call_result_468309)
            
            # Assigning a Subscript to a Name (line 552):
            
            # Obtaining the type of the subscript
            int_468310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 16), 'int')
            
            # Call to split(...): (line 552)
            # Processing the call arguments (line 552)
            # Getting the type of 'node' (line 552)
            node_468313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 43), 'node', False)
            # Obtaining the member 'split_dim' of a type (line 552)
            split_dim_468314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 43), node_468313, 'split_dim')
            # Getting the type of 'node' (line 552)
            node_468315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 59), 'node', False)
            # Obtaining the member 'split' of a type (line 552)
            split_468316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 59), node_468315, 'split')
            # Processing the call keyword arguments (line 552)
            kwargs_468317 = {}
            # Getting the type of 'rect' (line 552)
            rect_468311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 32), 'rect', False)
            # Obtaining the member 'split' of a type (line 552)
            split_468312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 32), rect_468311, 'split')
            # Calling split(args, kwargs) (line 552)
            split_call_result_468318 = invoke(stypy.reporting.localization.Localization(__file__, 552, 32), split_468312, *[split_dim_468314, split_468316], **kwargs_468317)
            
            # Obtaining the member '__getitem__' of a type (line 552)
            getitem___468319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 16), split_call_result_468318, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 552)
            subscript_call_result_468320 = invoke(stypy.reporting.localization.Localization(__file__, 552, 16), getitem___468319, int_468310)
            
            # Assigning a type to the variable 'tuple_var_assignment_466777' (line 552)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'tuple_var_assignment_466777', subscript_call_result_468320)
            
            # Assigning a Name to a Name (line 552):
            # Getting the type of 'tuple_var_assignment_466776' (line 552)
            tuple_var_assignment_466776_468321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'tuple_var_assignment_466776')
            # Assigning a type to the variable 'less' (line 552)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'less', tuple_var_assignment_466776_468321)
            
            # Assigning a Name to a Name (line 552):
            # Getting the type of 'tuple_var_assignment_466777' (line 552)
            tuple_var_assignment_466777_468322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'tuple_var_assignment_466777')
            # Assigning a type to the variable 'greater' (line 552)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 22), 'greater', tuple_var_assignment_466777_468322)
            
            # Call to traverse_checking(...): (line 553)
            # Processing the call arguments (line 553)
            # Getting the type of 'node' (line 553)
            node_468324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 41), 'node', False)
            # Obtaining the member 'less' of a type (line 553)
            less_468325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 41), node_468324, 'less')
            # Getting the type of 'less' (line 553)
            less_468326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 52), 'less', False)
            # Processing the call keyword arguments (line 553)
            kwargs_468327 = {}
            # Getting the type of 'traverse_checking' (line 553)
            traverse_checking_468323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 23), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 553)
            traverse_checking_call_result_468328 = invoke(stypy.reporting.localization.Localization(__file__, 553, 23), traverse_checking_468323, *[less_468325, less_468326], **kwargs_468327)
            
            
            # Call to traverse_checking(...): (line 554)
            # Processing the call arguments (line 554)
            # Getting the type of 'node' (line 554)
            node_468330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 41), 'node', False)
            # Obtaining the member 'greater' of a type (line 554)
            greater_468331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 41), node_468330, 'greater')
            # Getting the type of 'greater' (line 554)
            greater_468332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 55), 'greater', False)
            # Processing the call keyword arguments (line 554)
            kwargs_468333 = {}
            # Getting the type of 'traverse_checking' (line 554)
            traverse_checking_468329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 23), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 554)
            traverse_checking_call_result_468334 = invoke(stypy.reporting.localization.Localization(__file__, 554, 23), traverse_checking_468329, *[greater_468331, greater_468332], **kwargs_468333)
            
            # Applying the binary operator '+' (line 553)
            result_add_468335 = python_operator(stypy.reporting.localization.Localization(__file__, 553, 23), '+', traverse_checking_call_result_468328, traverse_checking_call_result_468334)
            
            # Assigning a type to the variable 'stypy_return_type' (line 553)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 16), 'stypy_return_type', result_add_468335)
            # SSA join for if statement (line 548)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 546)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 544)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'traverse_checking(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'traverse_checking' in the type store
            # Getting the type of 'stypy_return_type' (line 543)
            stypy_return_type_468336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_468336)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'traverse_checking'
            return stypy_return_type_468336

        # Assigning a type to the variable 'traverse_checking' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'traverse_checking', traverse_checking)

        @norecursion
        def traverse_no_checking(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'traverse_no_checking'
            module_type_store = module_type_store.open_function_context('traverse_no_checking', 556, 8, False)
            
            # Passed parameters checking function
            traverse_no_checking.stypy_localization = localization
            traverse_no_checking.stypy_type_of_self = None
            traverse_no_checking.stypy_type_store = module_type_store
            traverse_no_checking.stypy_function_name = 'traverse_no_checking'
            traverse_no_checking.stypy_param_names_list = ['node']
            traverse_no_checking.stypy_varargs_param_name = None
            traverse_no_checking.stypy_kwargs_param_name = None
            traverse_no_checking.stypy_call_defaults = defaults
            traverse_no_checking.stypy_call_varargs = varargs
            traverse_no_checking.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'traverse_no_checking', ['node'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'traverse_no_checking', localization, ['node'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'traverse_no_checking(...)' code ##################

            
            
            # Call to isinstance(...): (line 557)
            # Processing the call arguments (line 557)
            # Getting the type of 'node' (line 557)
            node_468338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 26), 'node', False)
            # Getting the type of 'KDTree' (line 557)
            KDTree_468339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 32), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 557)
            leafnode_468340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 32), KDTree_468339, 'leafnode')
            # Processing the call keyword arguments (line 557)
            kwargs_468341 = {}
            # Getting the type of 'isinstance' (line 557)
            isinstance_468337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 557)
            isinstance_call_result_468342 = invoke(stypy.reporting.localization.Localization(__file__, 557, 15), isinstance_468337, *[node_468338, leafnode_468340], **kwargs_468341)
            
            # Testing the type of an if condition (line 557)
            if_condition_468343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 557, 12), isinstance_call_result_468342)
            # Assigning a type to the variable 'if_condition_468343' (line 557)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'if_condition_468343', if_condition_468343)
            # SSA begins for if statement (line 557)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to tolist(...): (line 558)
            # Processing the call keyword arguments (line 558)
            kwargs_468347 = {}
            # Getting the type of 'node' (line 558)
            node_468344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 23), 'node', False)
            # Obtaining the member 'idx' of a type (line 558)
            idx_468345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 23), node_468344, 'idx')
            # Obtaining the member 'tolist' of a type (line 558)
            tolist_468346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 23), idx_468345, 'tolist')
            # Calling tolist(args, kwargs) (line 558)
            tolist_call_result_468348 = invoke(stypy.reporting.localization.Localization(__file__, 558, 23), tolist_468346, *[], **kwargs_468347)
            
            # Assigning a type to the variable 'stypy_return_type' (line 558)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 16), 'stypy_return_type', tolist_call_result_468348)
            # SSA branch for the else part of an if statement (line 557)
            module_type_store.open_ssa_branch('else')
            
            # Call to traverse_no_checking(...): (line 560)
            # Processing the call arguments (line 560)
            # Getting the type of 'node' (line 560)
            node_468350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 44), 'node', False)
            # Obtaining the member 'less' of a type (line 560)
            less_468351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 44), node_468350, 'less')
            # Processing the call keyword arguments (line 560)
            kwargs_468352 = {}
            # Getting the type of 'traverse_no_checking' (line 560)
            traverse_no_checking_468349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 23), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 560)
            traverse_no_checking_call_result_468353 = invoke(stypy.reporting.localization.Localization(__file__, 560, 23), traverse_no_checking_468349, *[less_468351], **kwargs_468352)
            
            
            # Call to traverse_no_checking(...): (line 561)
            # Processing the call arguments (line 561)
            # Getting the type of 'node' (line 561)
            node_468355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 44), 'node', False)
            # Obtaining the member 'greater' of a type (line 561)
            greater_468356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 44), node_468355, 'greater')
            # Processing the call keyword arguments (line 561)
            kwargs_468357 = {}
            # Getting the type of 'traverse_no_checking' (line 561)
            traverse_no_checking_468354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 23), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 561)
            traverse_no_checking_call_result_468358 = invoke(stypy.reporting.localization.Localization(__file__, 561, 23), traverse_no_checking_468354, *[greater_468356], **kwargs_468357)
            
            # Applying the binary operator '+' (line 560)
            result_add_468359 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 23), '+', traverse_no_checking_call_result_468353, traverse_no_checking_call_result_468358)
            
            # Assigning a type to the variable 'stypy_return_type' (line 560)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'stypy_return_type', result_add_468359)
            # SSA join for if statement (line 557)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'traverse_no_checking(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'traverse_no_checking' in the type store
            # Getting the type of 'stypy_return_type' (line 556)
            stypy_return_type_468360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_468360)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'traverse_no_checking'
            return stypy_return_type_468360

        # Assigning a type to the variable 'traverse_no_checking' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'traverse_no_checking', traverse_no_checking)
        
        # Call to traverse_checking(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'self' (line 563)
        self_468362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 33), 'self', False)
        # Obtaining the member 'tree' of a type (line 563)
        tree_468363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 33), self_468362, 'tree')
        # Getting the type of 'R' (line 563)
        R_468364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 44), 'R', False)
        # Processing the call keyword arguments (line 563)
        kwargs_468365 = {}
        # Getting the type of 'traverse_checking' (line 563)
        traverse_checking_468361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 15), 'traverse_checking', False)
        # Calling traverse_checking(args, kwargs) (line 563)
        traverse_checking_call_result_468366 = invoke(stypy.reporting.localization.Localization(__file__, 563, 15), traverse_checking_468361, *[tree_468363, R_468364], **kwargs_468365)
        
        # Assigning a type to the variable 'stypy_return_type' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'stypy_return_type', traverse_checking_call_result_468366)
        
        # ################# End of '__query_ball_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__query_ball_point' in the type store
        # Getting the type of 'stypy_return_type' (line 540)
        stypy_return_type_468367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_468367)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__query_ball_point'
        return stypy_return_type_468367


    @norecursion
    def query_ball_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_468368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 39), 'float')
        int_468369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 47), 'int')
        defaults = [float_468368, int_468369]
        # Create a new context for function 'query_ball_point'
        module_type_store = module_type_store.open_function_context('query_ball_point', 565, 4, False)
        # Assigning a type to the variable 'self' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KDTree.query_ball_point.__dict__.__setitem__('stypy_localization', localization)
        KDTree.query_ball_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KDTree.query_ball_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        KDTree.query_ball_point.__dict__.__setitem__('stypy_function_name', 'KDTree.query_ball_point')
        KDTree.query_ball_point.__dict__.__setitem__('stypy_param_names_list', ['x', 'r', 'p', 'eps'])
        KDTree.query_ball_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        KDTree.query_ball_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KDTree.query_ball_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        KDTree.query_ball_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        KDTree.query_ball_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KDTree.query_ball_point.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KDTree.query_ball_point', ['x', 'r', 'p', 'eps'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'query_ball_point', localization, ['x', 'r', 'p', 'eps'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'query_ball_point(...)' code ##################

        str_468370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, (-1)), 'str', "Find all points within distance r of point(s) x.\n\n        Parameters\n        ----------\n        x : array_like, shape tuple + (self.m,)\n            The point or points to search for neighbors of.\n        r : positive float\n            The radius of points to return.\n        p : float, optional\n            Which Minkowski p-norm to use.  Should be in the range [1, inf].\n        eps : nonnegative float, optional\n            Approximate search. Branches of the tree are not explored if their\n            nearest points are further than ``r / (1 + eps)``, and branches are\n            added in bulk if their furthest points are nearer than\n            ``r * (1 + eps)``.\n\n        Returns\n        -------\n        results : list or array of lists\n            If `x` is a single point, returns a list of the indices of the\n            neighbors of `x`. If `x` is an array of points, returns an object\n            array of shape tuple containing lists of neighbors.\n\n        Notes\n        -----\n        If you have many points whose neighbors you want to find, you may save\n        substantial amounts of time by putting them in a KDTree and using\n        query_ball_tree.\n\n        Examples\n        --------\n        >>> from scipy import spatial\n        >>> x, y = np.mgrid[0:5, 0:5]\n        >>> points = np.c_[x.ravel(), y.ravel()]\n        >>> tree = spatial.KDTree(points)\n        >>> tree.query_ball_point([2, 0], 1)\n        [5, 10, 11, 15]\n\n        Query multiple points and plot the results:\n\n        >>> import matplotlib.pyplot as plt\n        >>> points = np.asarray(points)\n        >>> plt.plot(points[:,0], points[:,1], '.')\n        >>> for results in tree.query_ball_point(([2, 0], [3, 3]), 1):\n        ...     nearby_points = points[results]\n        ...     plt.plot(nearby_points[:,0], nearby_points[:,1], 'o')\n        >>> plt.margins(0.1, 0.1)\n        >>> plt.show()\n\n        ")
        
        # Assigning a Call to a Name (line 616):
        
        # Assigning a Call to a Name (line 616):
        
        # Call to asarray(...): (line 616)
        # Processing the call arguments (line 616)
        # Getting the type of 'x' (line 616)
        x_468373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 23), 'x', False)
        # Processing the call keyword arguments (line 616)
        kwargs_468374 = {}
        # Getting the type of 'np' (line 616)
        np_468371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 616)
        asarray_468372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 12), np_468371, 'asarray')
        # Calling asarray(args, kwargs) (line 616)
        asarray_call_result_468375 = invoke(stypy.reporting.localization.Localization(__file__, 616, 12), asarray_468372, *[x_468373], **kwargs_468374)
        
        # Assigning a type to the variable 'x' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'x', asarray_call_result_468375)
        
        
        
        # Obtaining the type of the subscript
        int_468376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 19), 'int')
        # Getting the type of 'x' (line 617)
        x_468377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 11), 'x')
        # Obtaining the member 'shape' of a type (line 617)
        shape_468378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 11), x_468377, 'shape')
        # Obtaining the member '__getitem__' of a type (line 617)
        getitem___468379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 11), shape_468378, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 617)
        subscript_call_result_468380 = invoke(stypy.reporting.localization.Localization(__file__, 617, 11), getitem___468379, int_468376)
        
        # Getting the type of 'self' (line 617)
        self_468381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 26), 'self')
        # Obtaining the member 'm' of a type (line 617)
        m_468382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 26), self_468381, 'm')
        # Applying the binary operator '!=' (line 617)
        result_ne_468383 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 11), '!=', subscript_call_result_468380, m_468382)
        
        # Testing the type of an if condition (line 617)
        if_condition_468384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 617, 8), result_ne_468383)
        # Assigning a type to the variable 'if_condition_468384' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'if_condition_468384', if_condition_468384)
        # SSA begins for if statement (line 617)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 618)
        # Processing the call arguments (line 618)
        str_468386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 29), 'str', 'Searching for a %d-dimensional point in a %d-dimensional KDTree')
        
        # Obtaining an instance of the builtin type 'tuple' (line 619)
        tuple_468387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 619)
        # Adding element type (line 619)
        
        # Obtaining the type of the subscript
        int_468388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 64), 'int')
        # Getting the type of 'x' (line 619)
        x_468389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 56), 'x', False)
        # Obtaining the member 'shape' of a type (line 619)
        shape_468390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 56), x_468389, 'shape')
        # Obtaining the member '__getitem__' of a type (line 619)
        getitem___468391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 56), shape_468390, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 619)
        subscript_call_result_468392 = invoke(stypy.reporting.localization.Localization(__file__, 619, 56), getitem___468391, int_468388)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 56), tuple_468387, subscript_call_result_468392)
        # Adding element type (line 619)
        # Getting the type of 'self' (line 619)
        self_468393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 69), 'self', False)
        # Obtaining the member 'm' of a type (line 619)
        m_468394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 69), self_468393, 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 56), tuple_468387, m_468394)
        
        # Applying the binary operator '%' (line 618)
        result_mod_468395 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 29), '%', str_468386, tuple_468387)
        
        # Processing the call keyword arguments (line 618)
        kwargs_468396 = {}
        # Getting the type of 'ValueError' (line 618)
        ValueError_468385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 618)
        ValueError_call_result_468397 = invoke(stypy.reporting.localization.Localization(__file__, 618, 18), ValueError_468385, *[result_mod_468395], **kwargs_468396)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 618, 12), ValueError_call_result_468397, 'raise parameter', BaseException)
        # SSA join for if statement (line 617)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 620)
        # Processing the call arguments (line 620)
        # Getting the type of 'x' (line 620)
        x_468399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'x', False)
        # Obtaining the member 'shape' of a type (line 620)
        shape_468400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 15), x_468399, 'shape')
        # Processing the call keyword arguments (line 620)
        kwargs_468401 = {}
        # Getting the type of 'len' (line 620)
        len_468398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 11), 'len', False)
        # Calling len(args, kwargs) (line 620)
        len_call_result_468402 = invoke(stypy.reporting.localization.Localization(__file__, 620, 11), len_468398, *[shape_468400], **kwargs_468401)
        
        int_468403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 27), 'int')
        # Applying the binary operator '==' (line 620)
        result_eq_468404 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 11), '==', len_call_result_468402, int_468403)
        
        # Testing the type of an if condition (line 620)
        if_condition_468405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 8), result_eq_468404)
        # Assigning a type to the variable 'if_condition_468405' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'if_condition_468405', if_condition_468405)
        # SSA begins for if statement (line 620)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __query_ball_point(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'x' (line 621)
        x_468408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 43), 'x', False)
        # Getting the type of 'r' (line 621)
        r_468409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 46), 'r', False)
        # Getting the type of 'p' (line 621)
        p_468410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 49), 'p', False)
        # Getting the type of 'eps' (line 621)
        eps_468411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 52), 'eps', False)
        # Processing the call keyword arguments (line 621)
        kwargs_468412 = {}
        # Getting the type of 'self' (line 621)
        self_468406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 19), 'self', False)
        # Obtaining the member '__query_ball_point' of a type (line 621)
        query_ball_point_468407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 19), self_468406, '__query_ball_point')
        # Calling __query_ball_point(args, kwargs) (line 621)
        query_ball_point_call_result_468413 = invoke(stypy.reporting.localization.Localization(__file__, 621, 19), query_ball_point_468407, *[x_468408, r_468409, p_468410, eps_468411], **kwargs_468412)
        
        # Assigning a type to the variable 'stypy_return_type' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 12), 'stypy_return_type', query_ball_point_call_result_468413)
        # SSA branch for the else part of an if statement (line 620)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 623):
        
        # Assigning a Subscript to a Name (line 623):
        
        # Obtaining the type of the subscript
        int_468414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 32), 'int')
        slice_468415 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 623, 23), None, int_468414, None)
        # Getting the type of 'x' (line 623)
        x_468416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 23), 'x')
        # Obtaining the member 'shape' of a type (line 623)
        shape_468417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 23), x_468416, 'shape')
        # Obtaining the member '__getitem__' of a type (line 623)
        getitem___468418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 23), shape_468417, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 623)
        subscript_call_result_468419 = invoke(stypy.reporting.localization.Localization(__file__, 623, 23), getitem___468418, slice_468415)
        
        # Assigning a type to the variable 'retshape' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'retshape', subscript_call_result_468419)
        
        # Assigning a Call to a Name (line 624):
        
        # Assigning a Call to a Name (line 624):
        
        # Call to empty(...): (line 624)
        # Processing the call arguments (line 624)
        # Getting the type of 'retshape' (line 624)
        retshape_468422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 30), 'retshape', False)
        # Processing the call keyword arguments (line 624)
        # Getting the type of 'object' (line 624)
        object_468423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 46), 'object', False)
        keyword_468424 = object_468423
        kwargs_468425 = {'dtype': keyword_468424}
        # Getting the type of 'np' (line 624)
        np_468420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 21), 'np', False)
        # Obtaining the member 'empty' of a type (line 624)
        empty_468421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 21), np_468420, 'empty')
        # Calling empty(args, kwargs) (line 624)
        empty_call_result_468426 = invoke(stypy.reporting.localization.Localization(__file__, 624, 21), empty_468421, *[retshape_468422], **kwargs_468425)
        
        # Assigning a type to the variable 'result' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 12), 'result', empty_call_result_468426)
        
        
        # Call to ndindex(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'retshape' (line 625)
        retshape_468429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 32), 'retshape', False)
        # Processing the call keyword arguments (line 625)
        kwargs_468430 = {}
        # Getting the type of 'np' (line 625)
        np_468427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 21), 'np', False)
        # Obtaining the member 'ndindex' of a type (line 625)
        ndindex_468428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 21), np_468427, 'ndindex')
        # Calling ndindex(args, kwargs) (line 625)
        ndindex_call_result_468431 = invoke(stypy.reporting.localization.Localization(__file__, 625, 21), ndindex_468428, *[retshape_468429], **kwargs_468430)
        
        # Testing the type of a for loop iterable (line 625)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 625, 12), ndindex_call_result_468431)
        # Getting the type of the for loop variable (line 625)
        for_loop_var_468432 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 625, 12), ndindex_call_result_468431)
        # Assigning a type to the variable 'c' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'c', for_loop_var_468432)
        # SSA begins for a for statement (line 625)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Subscript (line 626):
        
        # Assigning a Call to a Subscript (line 626):
        
        # Call to __query_ball_point(...): (line 626)
        # Processing the call arguments (line 626)
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 626)
        c_468435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 54), 'c', False)
        # Getting the type of 'x' (line 626)
        x_468436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 52), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 626)
        getitem___468437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 52), x_468436, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 626)
        subscript_call_result_468438 = invoke(stypy.reporting.localization.Localization(__file__, 626, 52), getitem___468437, c_468435)
        
        # Getting the type of 'r' (line 626)
        r_468439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 58), 'r', False)
        # Processing the call keyword arguments (line 626)
        # Getting the type of 'p' (line 626)
        p_468440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 63), 'p', False)
        keyword_468441 = p_468440
        # Getting the type of 'eps' (line 626)
        eps_468442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 70), 'eps', False)
        keyword_468443 = eps_468442
        kwargs_468444 = {'p': keyword_468441, 'eps': keyword_468443}
        # Getting the type of 'self' (line 626)
        self_468433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 28), 'self', False)
        # Obtaining the member '__query_ball_point' of a type (line 626)
        query_ball_point_468434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 28), self_468433, '__query_ball_point')
        # Calling __query_ball_point(args, kwargs) (line 626)
        query_ball_point_call_result_468445 = invoke(stypy.reporting.localization.Localization(__file__, 626, 28), query_ball_point_468434, *[subscript_call_result_468438, r_468439], **kwargs_468444)
        
        # Getting the type of 'result' (line 626)
        result_468446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 16), 'result')
        # Getting the type of 'c' (line 626)
        c_468447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 23), 'c')
        # Storing an element on a container (line 626)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 626, 16), result_468446, (c_468447, query_ball_point_call_result_468445))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 627)
        result_468448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 19), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'stypy_return_type', result_468448)
        # SSA join for if statement (line 620)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'query_ball_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'query_ball_point' in the type store
        # Getting the type of 'stypy_return_type' (line 565)
        stypy_return_type_468449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_468449)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'query_ball_point'
        return stypy_return_type_468449


    @norecursion
    def query_ball_tree(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_468450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 42), 'float')
        int_468451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 50), 'int')
        defaults = [float_468450, int_468451]
        # Create a new context for function 'query_ball_tree'
        module_type_store = module_type_store.open_function_context('query_ball_tree', 629, 4, False)
        # Assigning a type to the variable 'self' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KDTree.query_ball_tree.__dict__.__setitem__('stypy_localization', localization)
        KDTree.query_ball_tree.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KDTree.query_ball_tree.__dict__.__setitem__('stypy_type_store', module_type_store)
        KDTree.query_ball_tree.__dict__.__setitem__('stypy_function_name', 'KDTree.query_ball_tree')
        KDTree.query_ball_tree.__dict__.__setitem__('stypy_param_names_list', ['other', 'r', 'p', 'eps'])
        KDTree.query_ball_tree.__dict__.__setitem__('stypy_varargs_param_name', None)
        KDTree.query_ball_tree.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KDTree.query_ball_tree.__dict__.__setitem__('stypy_call_defaults', defaults)
        KDTree.query_ball_tree.__dict__.__setitem__('stypy_call_varargs', varargs)
        KDTree.query_ball_tree.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KDTree.query_ball_tree.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KDTree.query_ball_tree', ['other', 'r', 'p', 'eps'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'query_ball_tree', localization, ['other', 'r', 'p', 'eps'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'query_ball_tree(...)' code ##################

        str_468452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, (-1)), 'str', 'Find all pairs of points whose distance is at most r\n\n        Parameters\n        ----------\n        other : KDTree instance\n            The tree containing points to search against.\n        r : float\n            The maximum distance, has to be positive.\n        p : float, optional\n            Which Minkowski norm to use.  `p` has to meet the condition\n            ``1 <= p <= infinity``.\n        eps : float, optional\n            Approximate search.  Branches of the tree are not explored\n            if their nearest points are further than ``r/(1+eps)``, and\n            branches are added in bulk if their furthest points are nearer\n            than ``r * (1+eps)``.  `eps` has to be non-negative.\n\n        Returns\n        -------\n        results : list of lists\n            For each element ``self.data[i]`` of this tree, ``results[i]`` is a\n            list of the indices of its neighbors in ``other.data``.\n\n        ')
        
        # Assigning a ListComp to a Name (line 654):
        
        # Assigning a ListComp to a Name (line 654):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 654)
        # Processing the call arguments (line 654)
        # Getting the type of 'self' (line 654)
        self_468455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 37), 'self', False)
        # Obtaining the member 'n' of a type (line 654)
        n_468456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 37), self_468455, 'n')
        # Processing the call keyword arguments (line 654)
        kwargs_468457 = {}
        # Getting the type of 'range' (line 654)
        range_468454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 31), 'range', False)
        # Calling range(args, kwargs) (line 654)
        range_call_result_468458 = invoke(stypy.reporting.localization.Localization(__file__, 654, 31), range_468454, *[n_468456], **kwargs_468457)
        
        comprehension_468459 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 19), range_call_result_468458)
        # Assigning a type to the variable 'i' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 19), 'i', comprehension_468459)
        
        # Obtaining an instance of the builtin type 'list' (line 654)
        list_468453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 654)
        
        list_468460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 19), list_468460, list_468453)
        # Assigning a type to the variable 'results' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'results', list_468460)

        @norecursion
        def traverse_checking(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'traverse_checking'
            module_type_store = module_type_store.open_function_context('traverse_checking', 656, 8, False)
            
            # Passed parameters checking function
            traverse_checking.stypy_localization = localization
            traverse_checking.stypy_type_of_self = None
            traverse_checking.stypy_type_store = module_type_store
            traverse_checking.stypy_function_name = 'traverse_checking'
            traverse_checking.stypy_param_names_list = ['node1', 'rect1', 'node2', 'rect2']
            traverse_checking.stypy_varargs_param_name = None
            traverse_checking.stypy_kwargs_param_name = None
            traverse_checking.stypy_call_defaults = defaults
            traverse_checking.stypy_call_varargs = varargs
            traverse_checking.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'traverse_checking', ['node1', 'rect1', 'node2', 'rect2'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'traverse_checking', localization, ['node1', 'rect1', 'node2', 'rect2'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'traverse_checking(...)' code ##################

            
            
            
            # Call to min_distance_rectangle(...): (line 657)
            # Processing the call arguments (line 657)
            # Getting the type of 'rect2' (line 657)
            rect2_468463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 44), 'rect2', False)
            # Getting the type of 'p' (line 657)
            p_468464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 51), 'p', False)
            # Processing the call keyword arguments (line 657)
            kwargs_468465 = {}
            # Getting the type of 'rect1' (line 657)
            rect1_468461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 15), 'rect1', False)
            # Obtaining the member 'min_distance_rectangle' of a type (line 657)
            min_distance_rectangle_468462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 15), rect1_468461, 'min_distance_rectangle')
            # Calling min_distance_rectangle(args, kwargs) (line 657)
            min_distance_rectangle_call_result_468466 = invoke(stypy.reporting.localization.Localization(__file__, 657, 15), min_distance_rectangle_468462, *[rect2_468463, p_468464], **kwargs_468465)
            
            # Getting the type of 'r' (line 657)
            r_468467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 56), 'r')
            float_468468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 59), 'float')
            # Getting the type of 'eps' (line 657)
            eps_468469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 62), 'eps')
            # Applying the binary operator '+' (line 657)
            result_add_468470 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 59), '+', float_468468, eps_468469)
            
            # Applying the binary operator 'div' (line 657)
            result_div_468471 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 56), 'div', r_468467, result_add_468470)
            
            # Applying the binary operator '>' (line 657)
            result_gt_468472 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 15), '>', min_distance_rectangle_call_result_468466, result_div_468471)
            
            # Testing the type of an if condition (line 657)
            if_condition_468473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 657, 12), result_gt_468472)
            # Assigning a type to the variable 'if_condition_468473' (line 657)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'if_condition_468473', if_condition_468473)
            # SSA begins for if statement (line 657)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 658)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 16), 'stypy_return_type', types.NoneType)
            # SSA branch for the else part of an if statement (line 657)
            module_type_store.open_ssa_branch('else')
            
            
            
            # Call to max_distance_rectangle(...): (line 659)
            # Processing the call arguments (line 659)
            # Getting the type of 'rect2' (line 659)
            rect2_468476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 46), 'rect2', False)
            # Getting the type of 'p' (line 659)
            p_468477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 53), 'p', False)
            # Processing the call keyword arguments (line 659)
            kwargs_468478 = {}
            # Getting the type of 'rect1' (line 659)
            rect1_468474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 17), 'rect1', False)
            # Obtaining the member 'max_distance_rectangle' of a type (line 659)
            max_distance_rectangle_468475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 17), rect1_468474, 'max_distance_rectangle')
            # Calling max_distance_rectangle(args, kwargs) (line 659)
            max_distance_rectangle_call_result_468479 = invoke(stypy.reporting.localization.Localization(__file__, 659, 17), max_distance_rectangle_468475, *[rect2_468476, p_468477], **kwargs_468478)
            
            # Getting the type of 'r' (line 659)
            r_468480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 58), 'r')
            float_468481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 61), 'float')
            # Getting the type of 'eps' (line 659)
            eps_468482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 64), 'eps')
            # Applying the binary operator '+' (line 659)
            result_add_468483 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 61), '+', float_468481, eps_468482)
            
            # Applying the binary operator '*' (line 659)
            result_mul_468484 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 58), '*', r_468480, result_add_468483)
            
            # Applying the binary operator '<' (line 659)
            result_lt_468485 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 17), '<', max_distance_rectangle_call_result_468479, result_mul_468484)
            
            # Testing the type of an if condition (line 659)
            if_condition_468486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 659, 17), result_lt_468485)
            # Assigning a type to the variable 'if_condition_468486' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 17), 'if_condition_468486', if_condition_468486)
            # SSA begins for if statement (line 659)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to traverse_no_checking(...): (line 660)
            # Processing the call arguments (line 660)
            # Getting the type of 'node1' (line 660)
            node1_468488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 37), 'node1', False)
            # Getting the type of 'node2' (line 660)
            node2_468489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 44), 'node2', False)
            # Processing the call keyword arguments (line 660)
            kwargs_468490 = {}
            # Getting the type of 'traverse_no_checking' (line 660)
            traverse_no_checking_468487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 16), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 660)
            traverse_no_checking_call_result_468491 = invoke(stypy.reporting.localization.Localization(__file__, 660, 16), traverse_no_checking_468487, *[node1_468488, node2_468489], **kwargs_468490)
            
            # SSA branch for the else part of an if statement (line 659)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 661)
            # Processing the call arguments (line 661)
            # Getting the type of 'node1' (line 661)
            node1_468493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 28), 'node1', False)
            # Getting the type of 'KDTree' (line 661)
            KDTree_468494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 35), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 661)
            leafnode_468495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 35), KDTree_468494, 'leafnode')
            # Processing the call keyword arguments (line 661)
            kwargs_468496 = {}
            # Getting the type of 'isinstance' (line 661)
            isinstance_468492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 661)
            isinstance_call_result_468497 = invoke(stypy.reporting.localization.Localization(__file__, 661, 17), isinstance_468492, *[node1_468493, leafnode_468495], **kwargs_468496)
            
            # Testing the type of an if condition (line 661)
            if_condition_468498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 661, 17), isinstance_call_result_468497)
            # Assigning a type to the variable 'if_condition_468498' (line 661)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 17), 'if_condition_468498', if_condition_468498)
            # SSA begins for if statement (line 661)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to isinstance(...): (line 662)
            # Processing the call arguments (line 662)
            # Getting the type of 'node2' (line 662)
            node2_468500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 30), 'node2', False)
            # Getting the type of 'KDTree' (line 662)
            KDTree_468501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 37), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 662)
            leafnode_468502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 37), KDTree_468501, 'leafnode')
            # Processing the call keyword arguments (line 662)
            kwargs_468503 = {}
            # Getting the type of 'isinstance' (line 662)
            isinstance_468499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 662)
            isinstance_call_result_468504 = invoke(stypy.reporting.localization.Localization(__file__, 662, 19), isinstance_468499, *[node2_468500, leafnode_468502], **kwargs_468503)
            
            # Testing the type of an if condition (line 662)
            if_condition_468505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 662, 16), isinstance_call_result_468504)
            # Assigning a type to the variable 'if_condition_468505' (line 662)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'if_condition_468505', if_condition_468505)
            # SSA begins for if statement (line 662)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 663):
            
            # Assigning a Subscript to a Name (line 663):
            
            # Obtaining the type of the subscript
            # Getting the type of 'node2' (line 663)
            node2_468506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 35), 'node2')
            # Obtaining the member 'idx' of a type (line 663)
            idx_468507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 35), node2_468506, 'idx')
            # Getting the type of 'other' (line 663)
            other_468508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 24), 'other')
            # Obtaining the member 'data' of a type (line 663)
            data_468509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 24), other_468508, 'data')
            # Obtaining the member '__getitem__' of a type (line 663)
            getitem___468510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 24), data_468509, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 663)
            subscript_call_result_468511 = invoke(stypy.reporting.localization.Localization(__file__, 663, 24), getitem___468510, idx_468507)
            
            # Assigning a type to the variable 'd' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 20), 'd', subscript_call_result_468511)
            
            # Getting the type of 'node1' (line 664)
            node1_468512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 29), 'node1')
            # Obtaining the member 'idx' of a type (line 664)
            idx_468513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 29), node1_468512, 'idx')
            # Testing the type of a for loop iterable (line 664)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 664, 20), idx_468513)
            # Getting the type of the for loop variable (line 664)
            for_loop_var_468514 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 664, 20), idx_468513)
            # Assigning a type to the variable 'i' (line 664)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 20), 'i', for_loop_var_468514)
            # SSA begins for a for statement (line 664)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'results' (line 665)
            results_468515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 24), 'results')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 665)
            i_468516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 32), 'i')
            # Getting the type of 'results' (line 665)
            results_468517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 24), 'results')
            # Obtaining the member '__getitem__' of a type (line 665)
            getitem___468518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 24), results_468517, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 665)
            subscript_call_result_468519 = invoke(stypy.reporting.localization.Localization(__file__, 665, 24), getitem___468518, i_468516)
            
            
            # Call to tolist(...): (line 665)
            # Processing the call keyword arguments (line 665)
            kwargs_468537 = {}
            
            # Obtaining the type of the subscript
            
            
            # Call to minkowski_distance(...): (line 665)
            # Processing the call arguments (line 665)
            # Getting the type of 'd' (line 665)
            d_468521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 67), 'd', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 665)
            i_468522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 79), 'i', False)
            # Getting the type of 'self' (line 665)
            self_468523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 69), 'self', False)
            # Obtaining the member 'data' of a type (line 665)
            data_468524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 69), self_468523, 'data')
            # Obtaining the member '__getitem__' of a type (line 665)
            getitem___468525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 69), data_468524, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 665)
            subscript_call_result_468526 = invoke(stypy.reporting.localization.Localization(__file__, 665, 69), getitem___468525, i_468522)
            
            # Getting the type of 'p' (line 665)
            p_468527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 82), 'p', False)
            # Processing the call keyword arguments (line 665)
            kwargs_468528 = {}
            # Getting the type of 'minkowski_distance' (line 665)
            minkowski_distance_468520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 48), 'minkowski_distance', False)
            # Calling minkowski_distance(args, kwargs) (line 665)
            minkowski_distance_call_result_468529 = invoke(stypy.reporting.localization.Localization(__file__, 665, 48), minkowski_distance_468520, *[d_468521, subscript_call_result_468526, p_468527], **kwargs_468528)
            
            # Getting the type of 'r' (line 665)
            r_468530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 88), 'r', False)
            # Applying the binary operator '<=' (line 665)
            result_le_468531 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 48), '<=', minkowski_distance_call_result_468529, r_468530)
            
            # Getting the type of 'node2' (line 665)
            node2_468532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 38), 'node2', False)
            # Obtaining the member 'idx' of a type (line 665)
            idx_468533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 38), node2_468532, 'idx')
            # Obtaining the member '__getitem__' of a type (line 665)
            getitem___468534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 38), idx_468533, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 665)
            subscript_call_result_468535 = invoke(stypy.reporting.localization.Localization(__file__, 665, 38), getitem___468534, result_le_468531)
            
            # Obtaining the member 'tolist' of a type (line 665)
            tolist_468536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 38), subscript_call_result_468535, 'tolist')
            # Calling tolist(args, kwargs) (line 665)
            tolist_call_result_468538 = invoke(stypy.reporting.localization.Localization(__file__, 665, 38), tolist_468536, *[], **kwargs_468537)
            
            # Applying the binary operator '+=' (line 665)
            result_iadd_468539 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 24), '+=', subscript_call_result_468519, tolist_call_result_468538)
            # Getting the type of 'results' (line 665)
            results_468540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 24), 'results')
            # Getting the type of 'i' (line 665)
            i_468541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 32), 'i')
            # Storing an element on a container (line 665)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 24), results_468540, (i_468541, result_iadd_468539))
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 662)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 667):
            
            # Assigning a Subscript to a Name (line 667):
            
            # Obtaining the type of the subscript
            int_468542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 20), 'int')
            
            # Call to split(...): (line 667)
            # Processing the call arguments (line 667)
            # Getting the type of 'node2' (line 667)
            node2_468545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 48), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 667)
            split_dim_468546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 48), node2_468545, 'split_dim')
            # Getting the type of 'node2' (line 667)
            node2_468547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 65), 'node2', False)
            # Obtaining the member 'split' of a type (line 667)
            split_468548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 65), node2_468547, 'split')
            # Processing the call keyword arguments (line 667)
            kwargs_468549 = {}
            # Getting the type of 'rect2' (line 667)
            rect2_468543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 36), 'rect2', False)
            # Obtaining the member 'split' of a type (line 667)
            split_468544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 36), rect2_468543, 'split')
            # Calling split(args, kwargs) (line 667)
            split_call_result_468550 = invoke(stypy.reporting.localization.Localization(__file__, 667, 36), split_468544, *[split_dim_468546, split_468548], **kwargs_468549)
            
            # Obtaining the member '__getitem__' of a type (line 667)
            getitem___468551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 20), split_call_result_468550, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 667)
            subscript_call_result_468552 = invoke(stypy.reporting.localization.Localization(__file__, 667, 20), getitem___468551, int_468542)
            
            # Assigning a type to the variable 'tuple_var_assignment_466778' (line 667)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 20), 'tuple_var_assignment_466778', subscript_call_result_468552)
            
            # Assigning a Subscript to a Name (line 667):
            
            # Obtaining the type of the subscript
            int_468553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 20), 'int')
            
            # Call to split(...): (line 667)
            # Processing the call arguments (line 667)
            # Getting the type of 'node2' (line 667)
            node2_468556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 48), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 667)
            split_dim_468557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 48), node2_468556, 'split_dim')
            # Getting the type of 'node2' (line 667)
            node2_468558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 65), 'node2', False)
            # Obtaining the member 'split' of a type (line 667)
            split_468559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 65), node2_468558, 'split')
            # Processing the call keyword arguments (line 667)
            kwargs_468560 = {}
            # Getting the type of 'rect2' (line 667)
            rect2_468554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 36), 'rect2', False)
            # Obtaining the member 'split' of a type (line 667)
            split_468555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 36), rect2_468554, 'split')
            # Calling split(args, kwargs) (line 667)
            split_call_result_468561 = invoke(stypy.reporting.localization.Localization(__file__, 667, 36), split_468555, *[split_dim_468557, split_468559], **kwargs_468560)
            
            # Obtaining the member '__getitem__' of a type (line 667)
            getitem___468562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 20), split_call_result_468561, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 667)
            subscript_call_result_468563 = invoke(stypy.reporting.localization.Localization(__file__, 667, 20), getitem___468562, int_468553)
            
            # Assigning a type to the variable 'tuple_var_assignment_466779' (line 667)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 20), 'tuple_var_assignment_466779', subscript_call_result_468563)
            
            # Assigning a Name to a Name (line 667):
            # Getting the type of 'tuple_var_assignment_466778' (line 667)
            tuple_var_assignment_466778_468564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 20), 'tuple_var_assignment_466778')
            # Assigning a type to the variable 'less' (line 667)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 20), 'less', tuple_var_assignment_466778_468564)
            
            # Assigning a Name to a Name (line 667):
            # Getting the type of 'tuple_var_assignment_466779' (line 667)
            tuple_var_assignment_466779_468565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 20), 'tuple_var_assignment_466779')
            # Assigning a type to the variable 'greater' (line 667)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 26), 'greater', tuple_var_assignment_466779_468565)
            
            # Call to traverse_checking(...): (line 668)
            # Processing the call arguments (line 668)
            # Getting the type of 'node1' (line 668)
            node1_468567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 38), 'node1', False)
            # Getting the type of 'rect1' (line 668)
            rect1_468568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 44), 'rect1', False)
            # Getting the type of 'node2' (line 668)
            node2_468569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 50), 'node2', False)
            # Obtaining the member 'less' of a type (line 668)
            less_468570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 50), node2_468569, 'less')
            # Getting the type of 'less' (line 668)
            less_468571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 61), 'less', False)
            # Processing the call keyword arguments (line 668)
            kwargs_468572 = {}
            # Getting the type of 'traverse_checking' (line 668)
            traverse_checking_468566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 20), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 668)
            traverse_checking_call_result_468573 = invoke(stypy.reporting.localization.Localization(__file__, 668, 20), traverse_checking_468566, *[node1_468567, rect1_468568, less_468570, less_468571], **kwargs_468572)
            
            
            # Call to traverse_checking(...): (line 669)
            # Processing the call arguments (line 669)
            # Getting the type of 'node1' (line 669)
            node1_468575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 38), 'node1', False)
            # Getting the type of 'rect1' (line 669)
            rect1_468576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 44), 'rect1', False)
            # Getting the type of 'node2' (line 669)
            node2_468577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 50), 'node2', False)
            # Obtaining the member 'greater' of a type (line 669)
            greater_468578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 50), node2_468577, 'greater')
            # Getting the type of 'greater' (line 669)
            greater_468579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 64), 'greater', False)
            # Processing the call keyword arguments (line 669)
            kwargs_468580 = {}
            # Getting the type of 'traverse_checking' (line 669)
            traverse_checking_468574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 20), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 669)
            traverse_checking_call_result_468581 = invoke(stypy.reporting.localization.Localization(__file__, 669, 20), traverse_checking_468574, *[node1_468575, rect1_468576, greater_468578, greater_468579], **kwargs_468580)
            
            # SSA join for if statement (line 662)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 661)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 670)
            # Processing the call arguments (line 670)
            # Getting the type of 'node2' (line 670)
            node2_468583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 28), 'node2', False)
            # Getting the type of 'KDTree' (line 670)
            KDTree_468584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 35), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 670)
            leafnode_468585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 35), KDTree_468584, 'leafnode')
            # Processing the call keyword arguments (line 670)
            kwargs_468586 = {}
            # Getting the type of 'isinstance' (line 670)
            isinstance_468582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 670)
            isinstance_call_result_468587 = invoke(stypy.reporting.localization.Localization(__file__, 670, 17), isinstance_468582, *[node2_468583, leafnode_468585], **kwargs_468586)
            
            # Testing the type of an if condition (line 670)
            if_condition_468588 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 670, 17), isinstance_call_result_468587)
            # Assigning a type to the variable 'if_condition_468588' (line 670)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 17), 'if_condition_468588', if_condition_468588)
            # SSA begins for if statement (line 670)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 671):
            
            # Assigning a Subscript to a Name (line 671):
            
            # Obtaining the type of the subscript
            int_468589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 16), 'int')
            
            # Call to split(...): (line 671)
            # Processing the call arguments (line 671)
            # Getting the type of 'node1' (line 671)
            node1_468592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 44), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 671)
            split_dim_468593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 44), node1_468592, 'split_dim')
            # Getting the type of 'node1' (line 671)
            node1_468594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 61), 'node1', False)
            # Obtaining the member 'split' of a type (line 671)
            split_468595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 61), node1_468594, 'split')
            # Processing the call keyword arguments (line 671)
            kwargs_468596 = {}
            # Getting the type of 'rect1' (line 671)
            rect1_468590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 32), 'rect1', False)
            # Obtaining the member 'split' of a type (line 671)
            split_468591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 32), rect1_468590, 'split')
            # Calling split(args, kwargs) (line 671)
            split_call_result_468597 = invoke(stypy.reporting.localization.Localization(__file__, 671, 32), split_468591, *[split_dim_468593, split_468595], **kwargs_468596)
            
            # Obtaining the member '__getitem__' of a type (line 671)
            getitem___468598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 16), split_call_result_468597, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 671)
            subscript_call_result_468599 = invoke(stypy.reporting.localization.Localization(__file__, 671, 16), getitem___468598, int_468589)
            
            # Assigning a type to the variable 'tuple_var_assignment_466780' (line 671)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'tuple_var_assignment_466780', subscript_call_result_468599)
            
            # Assigning a Subscript to a Name (line 671):
            
            # Obtaining the type of the subscript
            int_468600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 16), 'int')
            
            # Call to split(...): (line 671)
            # Processing the call arguments (line 671)
            # Getting the type of 'node1' (line 671)
            node1_468603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 44), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 671)
            split_dim_468604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 44), node1_468603, 'split_dim')
            # Getting the type of 'node1' (line 671)
            node1_468605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 61), 'node1', False)
            # Obtaining the member 'split' of a type (line 671)
            split_468606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 61), node1_468605, 'split')
            # Processing the call keyword arguments (line 671)
            kwargs_468607 = {}
            # Getting the type of 'rect1' (line 671)
            rect1_468601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 32), 'rect1', False)
            # Obtaining the member 'split' of a type (line 671)
            split_468602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 32), rect1_468601, 'split')
            # Calling split(args, kwargs) (line 671)
            split_call_result_468608 = invoke(stypy.reporting.localization.Localization(__file__, 671, 32), split_468602, *[split_dim_468604, split_468606], **kwargs_468607)
            
            # Obtaining the member '__getitem__' of a type (line 671)
            getitem___468609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 16), split_call_result_468608, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 671)
            subscript_call_result_468610 = invoke(stypy.reporting.localization.Localization(__file__, 671, 16), getitem___468609, int_468600)
            
            # Assigning a type to the variable 'tuple_var_assignment_466781' (line 671)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'tuple_var_assignment_466781', subscript_call_result_468610)
            
            # Assigning a Name to a Name (line 671):
            # Getting the type of 'tuple_var_assignment_466780' (line 671)
            tuple_var_assignment_466780_468611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'tuple_var_assignment_466780')
            # Assigning a type to the variable 'less' (line 671)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'less', tuple_var_assignment_466780_468611)
            
            # Assigning a Name to a Name (line 671):
            # Getting the type of 'tuple_var_assignment_466781' (line 671)
            tuple_var_assignment_466781_468612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'tuple_var_assignment_466781')
            # Assigning a type to the variable 'greater' (line 671)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 22), 'greater', tuple_var_assignment_466781_468612)
            
            # Call to traverse_checking(...): (line 672)
            # Processing the call arguments (line 672)
            # Getting the type of 'node1' (line 672)
            node1_468614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 34), 'node1', False)
            # Obtaining the member 'less' of a type (line 672)
            less_468615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 34), node1_468614, 'less')
            # Getting the type of 'less' (line 672)
            less_468616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 45), 'less', False)
            # Getting the type of 'node2' (line 672)
            node2_468617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 50), 'node2', False)
            # Getting the type of 'rect2' (line 672)
            rect2_468618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 56), 'rect2', False)
            # Processing the call keyword arguments (line 672)
            kwargs_468619 = {}
            # Getting the type of 'traverse_checking' (line 672)
            traverse_checking_468613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 16), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 672)
            traverse_checking_call_result_468620 = invoke(stypy.reporting.localization.Localization(__file__, 672, 16), traverse_checking_468613, *[less_468615, less_468616, node2_468617, rect2_468618], **kwargs_468619)
            
            
            # Call to traverse_checking(...): (line 673)
            # Processing the call arguments (line 673)
            # Getting the type of 'node1' (line 673)
            node1_468622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 34), 'node1', False)
            # Obtaining the member 'greater' of a type (line 673)
            greater_468623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 34), node1_468622, 'greater')
            # Getting the type of 'greater' (line 673)
            greater_468624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 48), 'greater', False)
            # Getting the type of 'node2' (line 673)
            node2_468625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 56), 'node2', False)
            # Getting the type of 'rect2' (line 673)
            rect2_468626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 62), 'rect2', False)
            # Processing the call keyword arguments (line 673)
            kwargs_468627 = {}
            # Getting the type of 'traverse_checking' (line 673)
            traverse_checking_468621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 16), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 673)
            traverse_checking_call_result_468628 = invoke(stypy.reporting.localization.Localization(__file__, 673, 16), traverse_checking_468621, *[greater_468623, greater_468624, node2_468625, rect2_468626], **kwargs_468627)
            
            # SSA branch for the else part of an if statement (line 670)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 675):
            
            # Assigning a Subscript to a Name (line 675):
            
            # Obtaining the type of the subscript
            int_468629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 16), 'int')
            
            # Call to split(...): (line 675)
            # Processing the call arguments (line 675)
            # Getting the type of 'node1' (line 675)
            node1_468632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 46), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 675)
            split_dim_468633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 46), node1_468632, 'split_dim')
            # Getting the type of 'node1' (line 675)
            node1_468634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 63), 'node1', False)
            # Obtaining the member 'split' of a type (line 675)
            split_468635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 63), node1_468634, 'split')
            # Processing the call keyword arguments (line 675)
            kwargs_468636 = {}
            # Getting the type of 'rect1' (line 675)
            rect1_468630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 34), 'rect1', False)
            # Obtaining the member 'split' of a type (line 675)
            split_468631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 34), rect1_468630, 'split')
            # Calling split(args, kwargs) (line 675)
            split_call_result_468637 = invoke(stypy.reporting.localization.Localization(__file__, 675, 34), split_468631, *[split_dim_468633, split_468635], **kwargs_468636)
            
            # Obtaining the member '__getitem__' of a type (line 675)
            getitem___468638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 16), split_call_result_468637, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 675)
            subscript_call_result_468639 = invoke(stypy.reporting.localization.Localization(__file__, 675, 16), getitem___468638, int_468629)
            
            # Assigning a type to the variable 'tuple_var_assignment_466782' (line 675)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'tuple_var_assignment_466782', subscript_call_result_468639)
            
            # Assigning a Subscript to a Name (line 675):
            
            # Obtaining the type of the subscript
            int_468640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 16), 'int')
            
            # Call to split(...): (line 675)
            # Processing the call arguments (line 675)
            # Getting the type of 'node1' (line 675)
            node1_468643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 46), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 675)
            split_dim_468644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 46), node1_468643, 'split_dim')
            # Getting the type of 'node1' (line 675)
            node1_468645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 63), 'node1', False)
            # Obtaining the member 'split' of a type (line 675)
            split_468646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 63), node1_468645, 'split')
            # Processing the call keyword arguments (line 675)
            kwargs_468647 = {}
            # Getting the type of 'rect1' (line 675)
            rect1_468641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 34), 'rect1', False)
            # Obtaining the member 'split' of a type (line 675)
            split_468642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 34), rect1_468641, 'split')
            # Calling split(args, kwargs) (line 675)
            split_call_result_468648 = invoke(stypy.reporting.localization.Localization(__file__, 675, 34), split_468642, *[split_dim_468644, split_468646], **kwargs_468647)
            
            # Obtaining the member '__getitem__' of a type (line 675)
            getitem___468649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 16), split_call_result_468648, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 675)
            subscript_call_result_468650 = invoke(stypy.reporting.localization.Localization(__file__, 675, 16), getitem___468649, int_468640)
            
            # Assigning a type to the variable 'tuple_var_assignment_466783' (line 675)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'tuple_var_assignment_466783', subscript_call_result_468650)
            
            # Assigning a Name to a Name (line 675):
            # Getting the type of 'tuple_var_assignment_466782' (line 675)
            tuple_var_assignment_466782_468651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'tuple_var_assignment_466782')
            # Assigning a type to the variable 'less1' (line 675)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'less1', tuple_var_assignment_466782_468651)
            
            # Assigning a Name to a Name (line 675):
            # Getting the type of 'tuple_var_assignment_466783' (line 675)
            tuple_var_assignment_466783_468652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'tuple_var_assignment_466783')
            # Assigning a type to the variable 'greater1' (line 675)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 23), 'greater1', tuple_var_assignment_466783_468652)
            
            # Assigning a Call to a Tuple (line 676):
            
            # Assigning a Subscript to a Name (line 676):
            
            # Obtaining the type of the subscript
            int_468653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 16), 'int')
            
            # Call to split(...): (line 676)
            # Processing the call arguments (line 676)
            # Getting the type of 'node2' (line 676)
            node2_468656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 46), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 676)
            split_dim_468657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 46), node2_468656, 'split_dim')
            # Getting the type of 'node2' (line 676)
            node2_468658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 63), 'node2', False)
            # Obtaining the member 'split' of a type (line 676)
            split_468659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 63), node2_468658, 'split')
            # Processing the call keyword arguments (line 676)
            kwargs_468660 = {}
            # Getting the type of 'rect2' (line 676)
            rect2_468654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 34), 'rect2', False)
            # Obtaining the member 'split' of a type (line 676)
            split_468655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 34), rect2_468654, 'split')
            # Calling split(args, kwargs) (line 676)
            split_call_result_468661 = invoke(stypy.reporting.localization.Localization(__file__, 676, 34), split_468655, *[split_dim_468657, split_468659], **kwargs_468660)
            
            # Obtaining the member '__getitem__' of a type (line 676)
            getitem___468662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 16), split_call_result_468661, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 676)
            subscript_call_result_468663 = invoke(stypy.reporting.localization.Localization(__file__, 676, 16), getitem___468662, int_468653)
            
            # Assigning a type to the variable 'tuple_var_assignment_466784' (line 676)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_466784', subscript_call_result_468663)
            
            # Assigning a Subscript to a Name (line 676):
            
            # Obtaining the type of the subscript
            int_468664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 16), 'int')
            
            # Call to split(...): (line 676)
            # Processing the call arguments (line 676)
            # Getting the type of 'node2' (line 676)
            node2_468667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 46), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 676)
            split_dim_468668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 46), node2_468667, 'split_dim')
            # Getting the type of 'node2' (line 676)
            node2_468669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 63), 'node2', False)
            # Obtaining the member 'split' of a type (line 676)
            split_468670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 63), node2_468669, 'split')
            # Processing the call keyword arguments (line 676)
            kwargs_468671 = {}
            # Getting the type of 'rect2' (line 676)
            rect2_468665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 34), 'rect2', False)
            # Obtaining the member 'split' of a type (line 676)
            split_468666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 34), rect2_468665, 'split')
            # Calling split(args, kwargs) (line 676)
            split_call_result_468672 = invoke(stypy.reporting.localization.Localization(__file__, 676, 34), split_468666, *[split_dim_468668, split_468670], **kwargs_468671)
            
            # Obtaining the member '__getitem__' of a type (line 676)
            getitem___468673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 16), split_call_result_468672, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 676)
            subscript_call_result_468674 = invoke(stypy.reporting.localization.Localization(__file__, 676, 16), getitem___468673, int_468664)
            
            # Assigning a type to the variable 'tuple_var_assignment_466785' (line 676)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_466785', subscript_call_result_468674)
            
            # Assigning a Name to a Name (line 676):
            # Getting the type of 'tuple_var_assignment_466784' (line 676)
            tuple_var_assignment_466784_468675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_466784')
            # Assigning a type to the variable 'less2' (line 676)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'less2', tuple_var_assignment_466784_468675)
            
            # Assigning a Name to a Name (line 676):
            # Getting the type of 'tuple_var_assignment_466785' (line 676)
            tuple_var_assignment_466785_468676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 16), 'tuple_var_assignment_466785')
            # Assigning a type to the variable 'greater2' (line 676)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 23), 'greater2', tuple_var_assignment_466785_468676)
            
            # Call to traverse_checking(...): (line 677)
            # Processing the call arguments (line 677)
            # Getting the type of 'node1' (line 677)
            node1_468678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 34), 'node1', False)
            # Obtaining the member 'less' of a type (line 677)
            less_468679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 34), node1_468678, 'less')
            # Getting the type of 'less1' (line 677)
            less1_468680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 45), 'less1', False)
            # Getting the type of 'node2' (line 677)
            node2_468681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 51), 'node2', False)
            # Obtaining the member 'less' of a type (line 677)
            less_468682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 51), node2_468681, 'less')
            # Getting the type of 'less2' (line 677)
            less2_468683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 62), 'less2', False)
            # Processing the call keyword arguments (line 677)
            kwargs_468684 = {}
            # Getting the type of 'traverse_checking' (line 677)
            traverse_checking_468677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 677)
            traverse_checking_call_result_468685 = invoke(stypy.reporting.localization.Localization(__file__, 677, 16), traverse_checking_468677, *[less_468679, less1_468680, less_468682, less2_468683], **kwargs_468684)
            
            
            # Call to traverse_checking(...): (line 678)
            # Processing the call arguments (line 678)
            # Getting the type of 'node1' (line 678)
            node1_468687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 34), 'node1', False)
            # Obtaining the member 'less' of a type (line 678)
            less_468688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 34), node1_468687, 'less')
            # Getting the type of 'less1' (line 678)
            less1_468689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 45), 'less1', False)
            # Getting the type of 'node2' (line 678)
            node2_468690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 51), 'node2', False)
            # Obtaining the member 'greater' of a type (line 678)
            greater_468691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 51), node2_468690, 'greater')
            # Getting the type of 'greater2' (line 678)
            greater2_468692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 65), 'greater2', False)
            # Processing the call keyword arguments (line 678)
            kwargs_468693 = {}
            # Getting the type of 'traverse_checking' (line 678)
            traverse_checking_468686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 16), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 678)
            traverse_checking_call_result_468694 = invoke(stypy.reporting.localization.Localization(__file__, 678, 16), traverse_checking_468686, *[less_468688, less1_468689, greater_468691, greater2_468692], **kwargs_468693)
            
            
            # Call to traverse_checking(...): (line 679)
            # Processing the call arguments (line 679)
            # Getting the type of 'node1' (line 679)
            node1_468696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 34), 'node1', False)
            # Obtaining the member 'greater' of a type (line 679)
            greater_468697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 34), node1_468696, 'greater')
            # Getting the type of 'greater1' (line 679)
            greater1_468698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 48), 'greater1', False)
            # Getting the type of 'node2' (line 679)
            node2_468699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 57), 'node2', False)
            # Obtaining the member 'less' of a type (line 679)
            less_468700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 57), node2_468699, 'less')
            # Getting the type of 'less2' (line 679)
            less2_468701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 68), 'less2', False)
            # Processing the call keyword arguments (line 679)
            kwargs_468702 = {}
            # Getting the type of 'traverse_checking' (line 679)
            traverse_checking_468695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 679)
            traverse_checking_call_result_468703 = invoke(stypy.reporting.localization.Localization(__file__, 679, 16), traverse_checking_468695, *[greater_468697, greater1_468698, less_468700, less2_468701], **kwargs_468702)
            
            
            # Call to traverse_checking(...): (line 680)
            # Processing the call arguments (line 680)
            # Getting the type of 'node1' (line 680)
            node1_468705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 34), 'node1', False)
            # Obtaining the member 'greater' of a type (line 680)
            greater_468706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 34), node1_468705, 'greater')
            # Getting the type of 'greater1' (line 680)
            greater1_468707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 48), 'greater1', False)
            # Getting the type of 'node2' (line 680)
            node2_468708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 57), 'node2', False)
            # Obtaining the member 'greater' of a type (line 680)
            greater_468709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 57), node2_468708, 'greater')
            # Getting the type of 'greater2' (line 680)
            greater2_468710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 71), 'greater2', False)
            # Processing the call keyword arguments (line 680)
            kwargs_468711 = {}
            # Getting the type of 'traverse_checking' (line 680)
            traverse_checking_468704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 16), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 680)
            traverse_checking_call_result_468712 = invoke(stypy.reporting.localization.Localization(__file__, 680, 16), traverse_checking_468704, *[greater_468706, greater1_468707, greater_468709, greater2_468710], **kwargs_468711)
            
            # SSA join for if statement (line 670)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 661)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 659)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 657)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'traverse_checking(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'traverse_checking' in the type store
            # Getting the type of 'stypy_return_type' (line 656)
            stypy_return_type_468713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_468713)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'traverse_checking'
            return stypy_return_type_468713

        # Assigning a type to the variable 'traverse_checking' (line 656)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'traverse_checking', traverse_checking)

        @norecursion
        def traverse_no_checking(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'traverse_no_checking'
            module_type_store = module_type_store.open_function_context('traverse_no_checking', 682, 8, False)
            
            # Passed parameters checking function
            traverse_no_checking.stypy_localization = localization
            traverse_no_checking.stypy_type_of_self = None
            traverse_no_checking.stypy_type_store = module_type_store
            traverse_no_checking.stypy_function_name = 'traverse_no_checking'
            traverse_no_checking.stypy_param_names_list = ['node1', 'node2']
            traverse_no_checking.stypy_varargs_param_name = None
            traverse_no_checking.stypy_kwargs_param_name = None
            traverse_no_checking.stypy_call_defaults = defaults
            traverse_no_checking.stypy_call_varargs = varargs
            traverse_no_checking.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'traverse_no_checking', ['node1', 'node2'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'traverse_no_checking', localization, ['node1', 'node2'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'traverse_no_checking(...)' code ##################

            
            
            # Call to isinstance(...): (line 683)
            # Processing the call arguments (line 683)
            # Getting the type of 'node1' (line 683)
            node1_468715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 26), 'node1', False)
            # Getting the type of 'KDTree' (line 683)
            KDTree_468716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 33), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 683)
            leafnode_468717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 33), KDTree_468716, 'leafnode')
            # Processing the call keyword arguments (line 683)
            kwargs_468718 = {}
            # Getting the type of 'isinstance' (line 683)
            isinstance_468714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 683)
            isinstance_call_result_468719 = invoke(stypy.reporting.localization.Localization(__file__, 683, 15), isinstance_468714, *[node1_468715, leafnode_468717], **kwargs_468718)
            
            # Testing the type of an if condition (line 683)
            if_condition_468720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 683, 12), isinstance_call_result_468719)
            # Assigning a type to the variable 'if_condition_468720' (line 683)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 12), 'if_condition_468720', if_condition_468720)
            # SSA begins for if statement (line 683)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to isinstance(...): (line 684)
            # Processing the call arguments (line 684)
            # Getting the type of 'node2' (line 684)
            node2_468722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 30), 'node2', False)
            # Getting the type of 'KDTree' (line 684)
            KDTree_468723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 37), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 684)
            leafnode_468724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 37), KDTree_468723, 'leafnode')
            # Processing the call keyword arguments (line 684)
            kwargs_468725 = {}
            # Getting the type of 'isinstance' (line 684)
            isinstance_468721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 684)
            isinstance_call_result_468726 = invoke(stypy.reporting.localization.Localization(__file__, 684, 19), isinstance_468721, *[node2_468722, leafnode_468724], **kwargs_468725)
            
            # Testing the type of an if condition (line 684)
            if_condition_468727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 684, 16), isinstance_call_result_468726)
            # Assigning a type to the variable 'if_condition_468727' (line 684)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 16), 'if_condition_468727', if_condition_468727)
            # SSA begins for if statement (line 684)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'node1' (line 685)
            node1_468728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 29), 'node1')
            # Obtaining the member 'idx' of a type (line 685)
            idx_468729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 29), node1_468728, 'idx')
            # Testing the type of a for loop iterable (line 685)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 685, 20), idx_468729)
            # Getting the type of the for loop variable (line 685)
            for_loop_var_468730 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 685, 20), idx_468729)
            # Assigning a type to the variable 'i' (line 685)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 20), 'i', for_loop_var_468730)
            # SSA begins for a for statement (line 685)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'results' (line 686)
            results_468731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 24), 'results')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 686)
            i_468732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 32), 'i')
            # Getting the type of 'results' (line 686)
            results_468733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 24), 'results')
            # Obtaining the member '__getitem__' of a type (line 686)
            getitem___468734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 24), results_468733, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 686)
            subscript_call_result_468735 = invoke(stypy.reporting.localization.Localization(__file__, 686, 24), getitem___468734, i_468732)
            
            
            # Call to tolist(...): (line 686)
            # Processing the call keyword arguments (line 686)
            kwargs_468739 = {}
            # Getting the type of 'node2' (line 686)
            node2_468736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 38), 'node2', False)
            # Obtaining the member 'idx' of a type (line 686)
            idx_468737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 38), node2_468736, 'idx')
            # Obtaining the member 'tolist' of a type (line 686)
            tolist_468738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 38), idx_468737, 'tolist')
            # Calling tolist(args, kwargs) (line 686)
            tolist_call_result_468740 = invoke(stypy.reporting.localization.Localization(__file__, 686, 38), tolist_468738, *[], **kwargs_468739)
            
            # Applying the binary operator '+=' (line 686)
            result_iadd_468741 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 24), '+=', subscript_call_result_468735, tolist_call_result_468740)
            # Getting the type of 'results' (line 686)
            results_468742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 24), 'results')
            # Getting the type of 'i' (line 686)
            i_468743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 32), 'i')
            # Storing an element on a container (line 686)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 24), results_468742, (i_468743, result_iadd_468741))
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 684)
            module_type_store.open_ssa_branch('else')
            
            # Call to traverse_no_checking(...): (line 688)
            # Processing the call arguments (line 688)
            # Getting the type of 'node1' (line 688)
            node1_468745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 41), 'node1', False)
            # Getting the type of 'node2' (line 688)
            node2_468746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 48), 'node2', False)
            # Obtaining the member 'less' of a type (line 688)
            less_468747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 48), node2_468746, 'less')
            # Processing the call keyword arguments (line 688)
            kwargs_468748 = {}
            # Getting the type of 'traverse_no_checking' (line 688)
            traverse_no_checking_468744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 20), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 688)
            traverse_no_checking_call_result_468749 = invoke(stypy.reporting.localization.Localization(__file__, 688, 20), traverse_no_checking_468744, *[node1_468745, less_468747], **kwargs_468748)
            
            
            # Call to traverse_no_checking(...): (line 689)
            # Processing the call arguments (line 689)
            # Getting the type of 'node1' (line 689)
            node1_468751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 41), 'node1', False)
            # Getting the type of 'node2' (line 689)
            node2_468752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 48), 'node2', False)
            # Obtaining the member 'greater' of a type (line 689)
            greater_468753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 48), node2_468752, 'greater')
            # Processing the call keyword arguments (line 689)
            kwargs_468754 = {}
            # Getting the type of 'traverse_no_checking' (line 689)
            traverse_no_checking_468750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 20), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 689)
            traverse_no_checking_call_result_468755 = invoke(stypy.reporting.localization.Localization(__file__, 689, 20), traverse_no_checking_468750, *[node1_468751, greater_468753], **kwargs_468754)
            
            # SSA join for if statement (line 684)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 683)
            module_type_store.open_ssa_branch('else')
            
            # Call to traverse_no_checking(...): (line 691)
            # Processing the call arguments (line 691)
            # Getting the type of 'node1' (line 691)
            node1_468757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 37), 'node1', False)
            # Obtaining the member 'less' of a type (line 691)
            less_468758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 37), node1_468757, 'less')
            # Getting the type of 'node2' (line 691)
            node2_468759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 49), 'node2', False)
            # Processing the call keyword arguments (line 691)
            kwargs_468760 = {}
            # Getting the type of 'traverse_no_checking' (line 691)
            traverse_no_checking_468756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 691)
            traverse_no_checking_call_result_468761 = invoke(stypy.reporting.localization.Localization(__file__, 691, 16), traverse_no_checking_468756, *[less_468758, node2_468759], **kwargs_468760)
            
            
            # Call to traverse_no_checking(...): (line 692)
            # Processing the call arguments (line 692)
            # Getting the type of 'node1' (line 692)
            node1_468763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 37), 'node1', False)
            # Obtaining the member 'greater' of a type (line 692)
            greater_468764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 37), node1_468763, 'greater')
            # Getting the type of 'node2' (line 692)
            node2_468765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 52), 'node2', False)
            # Processing the call keyword arguments (line 692)
            kwargs_468766 = {}
            # Getting the type of 'traverse_no_checking' (line 692)
            traverse_no_checking_468762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 692)
            traverse_no_checking_call_result_468767 = invoke(stypy.reporting.localization.Localization(__file__, 692, 16), traverse_no_checking_468762, *[greater_468764, node2_468765], **kwargs_468766)
            
            # SSA join for if statement (line 683)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'traverse_no_checking(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'traverse_no_checking' in the type store
            # Getting the type of 'stypy_return_type' (line 682)
            stypy_return_type_468768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_468768)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'traverse_no_checking'
            return stypy_return_type_468768

        # Assigning a type to the variable 'traverse_no_checking' (line 682)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'traverse_no_checking', traverse_no_checking)
        
        # Call to traverse_checking(...): (line 694)
        # Processing the call arguments (line 694)
        # Getting the type of 'self' (line 694)
        self_468770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 26), 'self', False)
        # Obtaining the member 'tree' of a type (line 694)
        tree_468771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 26), self_468770, 'tree')
        
        # Call to Rectangle(...): (line 694)
        # Processing the call arguments (line 694)
        # Getting the type of 'self' (line 694)
        self_468773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 47), 'self', False)
        # Obtaining the member 'maxes' of a type (line 694)
        maxes_468774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 47), self_468773, 'maxes')
        # Getting the type of 'self' (line 694)
        self_468775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 59), 'self', False)
        # Obtaining the member 'mins' of a type (line 694)
        mins_468776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 59), self_468775, 'mins')
        # Processing the call keyword arguments (line 694)
        kwargs_468777 = {}
        # Getting the type of 'Rectangle' (line 694)
        Rectangle_468772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 37), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 694)
        Rectangle_call_result_468778 = invoke(stypy.reporting.localization.Localization(__file__, 694, 37), Rectangle_468772, *[maxes_468774, mins_468776], **kwargs_468777)
        
        # Getting the type of 'other' (line 695)
        other_468779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 26), 'other', False)
        # Obtaining the member 'tree' of a type (line 695)
        tree_468780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 26), other_468779, 'tree')
        
        # Call to Rectangle(...): (line 695)
        # Processing the call arguments (line 695)
        # Getting the type of 'other' (line 695)
        other_468782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 48), 'other', False)
        # Obtaining the member 'maxes' of a type (line 695)
        maxes_468783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 48), other_468782, 'maxes')
        # Getting the type of 'other' (line 695)
        other_468784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 61), 'other', False)
        # Obtaining the member 'mins' of a type (line 695)
        mins_468785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 61), other_468784, 'mins')
        # Processing the call keyword arguments (line 695)
        kwargs_468786 = {}
        # Getting the type of 'Rectangle' (line 695)
        Rectangle_468781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 38), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 695)
        Rectangle_call_result_468787 = invoke(stypy.reporting.localization.Localization(__file__, 695, 38), Rectangle_468781, *[maxes_468783, mins_468785], **kwargs_468786)
        
        # Processing the call keyword arguments (line 694)
        kwargs_468788 = {}
        # Getting the type of 'traverse_checking' (line 694)
        traverse_checking_468769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'traverse_checking', False)
        # Calling traverse_checking(args, kwargs) (line 694)
        traverse_checking_call_result_468789 = invoke(stypy.reporting.localization.Localization(__file__, 694, 8), traverse_checking_468769, *[tree_468771, Rectangle_call_result_468778, tree_468780, Rectangle_call_result_468787], **kwargs_468788)
        
        # Getting the type of 'results' (line 696)
        results_468790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 15), 'results')
        # Assigning a type to the variable 'stypy_return_type' (line 696)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'stypy_return_type', results_468790)
        
        # ################# End of 'query_ball_tree(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'query_ball_tree' in the type store
        # Getting the type of 'stypy_return_type' (line 629)
        stypy_return_type_468791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_468791)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'query_ball_tree'
        return stypy_return_type_468791


    @norecursion
    def query_pairs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_468792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 31), 'float')
        int_468793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 39), 'int')
        defaults = [float_468792, int_468793]
        # Create a new context for function 'query_pairs'
        module_type_store = module_type_store.open_function_context('query_pairs', 698, 4, False)
        # Assigning a type to the variable 'self' (line 699)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KDTree.query_pairs.__dict__.__setitem__('stypy_localization', localization)
        KDTree.query_pairs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KDTree.query_pairs.__dict__.__setitem__('stypy_type_store', module_type_store)
        KDTree.query_pairs.__dict__.__setitem__('stypy_function_name', 'KDTree.query_pairs')
        KDTree.query_pairs.__dict__.__setitem__('stypy_param_names_list', ['r', 'p', 'eps'])
        KDTree.query_pairs.__dict__.__setitem__('stypy_varargs_param_name', None)
        KDTree.query_pairs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KDTree.query_pairs.__dict__.__setitem__('stypy_call_defaults', defaults)
        KDTree.query_pairs.__dict__.__setitem__('stypy_call_varargs', varargs)
        KDTree.query_pairs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KDTree.query_pairs.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KDTree.query_pairs', ['r', 'p', 'eps'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'query_pairs', localization, ['r', 'p', 'eps'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'query_pairs(...)' code ##################

        str_468794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, (-1)), 'str', '\n        Find all pairs of points within a distance.\n\n        Parameters\n        ----------\n        r : positive float\n            The maximum distance.\n        p : float, optional\n            Which Minkowski norm to use.  `p` has to meet the condition\n            ``1 <= p <= infinity``.\n        eps : float, optional\n            Approximate search.  Branches of the tree are not explored\n            if their nearest points are further than ``r/(1+eps)``, and\n            branches are added in bulk if their furthest points are nearer\n            than ``r * (1+eps)``.  `eps` has to be non-negative.\n\n        Returns\n        -------\n        results : set\n            Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding\n            positions are close.\n\n        ')
        
        # Assigning a Call to a Name (line 722):
        
        # Assigning a Call to a Name (line 722):
        
        # Call to set(...): (line 722)
        # Processing the call keyword arguments (line 722)
        kwargs_468796 = {}
        # Getting the type of 'set' (line 722)
        set_468795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 18), 'set', False)
        # Calling set(args, kwargs) (line 722)
        set_call_result_468797 = invoke(stypy.reporting.localization.Localization(__file__, 722, 18), set_468795, *[], **kwargs_468796)
        
        # Assigning a type to the variable 'results' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'results', set_call_result_468797)

        @norecursion
        def traverse_checking(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'traverse_checking'
            module_type_store = module_type_store.open_function_context('traverse_checking', 724, 8, False)
            
            # Passed parameters checking function
            traverse_checking.stypy_localization = localization
            traverse_checking.stypy_type_of_self = None
            traverse_checking.stypy_type_store = module_type_store
            traverse_checking.stypy_function_name = 'traverse_checking'
            traverse_checking.stypy_param_names_list = ['node1', 'rect1', 'node2', 'rect2']
            traverse_checking.stypy_varargs_param_name = None
            traverse_checking.stypy_kwargs_param_name = None
            traverse_checking.stypy_call_defaults = defaults
            traverse_checking.stypy_call_varargs = varargs
            traverse_checking.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'traverse_checking', ['node1', 'rect1', 'node2', 'rect2'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'traverse_checking', localization, ['node1', 'rect1', 'node2', 'rect2'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'traverse_checking(...)' code ##################

            
            
            
            # Call to min_distance_rectangle(...): (line 725)
            # Processing the call arguments (line 725)
            # Getting the type of 'rect2' (line 725)
            rect2_468800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 44), 'rect2', False)
            # Getting the type of 'p' (line 725)
            p_468801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 51), 'p', False)
            # Processing the call keyword arguments (line 725)
            kwargs_468802 = {}
            # Getting the type of 'rect1' (line 725)
            rect1_468798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 15), 'rect1', False)
            # Obtaining the member 'min_distance_rectangle' of a type (line 725)
            min_distance_rectangle_468799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 15), rect1_468798, 'min_distance_rectangle')
            # Calling min_distance_rectangle(args, kwargs) (line 725)
            min_distance_rectangle_call_result_468803 = invoke(stypy.reporting.localization.Localization(__file__, 725, 15), min_distance_rectangle_468799, *[rect2_468800, p_468801], **kwargs_468802)
            
            # Getting the type of 'r' (line 725)
            r_468804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 56), 'r')
            float_468805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 59), 'float')
            # Getting the type of 'eps' (line 725)
            eps_468806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 62), 'eps')
            # Applying the binary operator '+' (line 725)
            result_add_468807 = python_operator(stypy.reporting.localization.Localization(__file__, 725, 59), '+', float_468805, eps_468806)
            
            # Applying the binary operator 'div' (line 725)
            result_div_468808 = python_operator(stypy.reporting.localization.Localization(__file__, 725, 56), 'div', r_468804, result_add_468807)
            
            # Applying the binary operator '>' (line 725)
            result_gt_468809 = python_operator(stypy.reporting.localization.Localization(__file__, 725, 15), '>', min_distance_rectangle_call_result_468803, result_div_468808)
            
            # Testing the type of an if condition (line 725)
            if_condition_468810 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 725, 12), result_gt_468809)
            # Assigning a type to the variable 'if_condition_468810' (line 725)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 12), 'if_condition_468810', if_condition_468810)
            # SSA begins for if statement (line 725)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 726)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 16), 'stypy_return_type', types.NoneType)
            # SSA branch for the else part of an if statement (line 725)
            module_type_store.open_ssa_branch('else')
            
            
            
            # Call to max_distance_rectangle(...): (line 727)
            # Processing the call arguments (line 727)
            # Getting the type of 'rect2' (line 727)
            rect2_468813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 46), 'rect2', False)
            # Getting the type of 'p' (line 727)
            p_468814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 53), 'p', False)
            # Processing the call keyword arguments (line 727)
            kwargs_468815 = {}
            # Getting the type of 'rect1' (line 727)
            rect1_468811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 17), 'rect1', False)
            # Obtaining the member 'max_distance_rectangle' of a type (line 727)
            max_distance_rectangle_468812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 17), rect1_468811, 'max_distance_rectangle')
            # Calling max_distance_rectangle(args, kwargs) (line 727)
            max_distance_rectangle_call_result_468816 = invoke(stypy.reporting.localization.Localization(__file__, 727, 17), max_distance_rectangle_468812, *[rect2_468813, p_468814], **kwargs_468815)
            
            # Getting the type of 'r' (line 727)
            r_468817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 58), 'r')
            float_468818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 61), 'float')
            # Getting the type of 'eps' (line 727)
            eps_468819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 64), 'eps')
            # Applying the binary operator '+' (line 727)
            result_add_468820 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 61), '+', float_468818, eps_468819)
            
            # Applying the binary operator '*' (line 727)
            result_mul_468821 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 58), '*', r_468817, result_add_468820)
            
            # Applying the binary operator '<' (line 727)
            result_lt_468822 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 17), '<', max_distance_rectangle_call_result_468816, result_mul_468821)
            
            # Testing the type of an if condition (line 727)
            if_condition_468823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 727, 17), result_lt_468822)
            # Assigning a type to the variable 'if_condition_468823' (line 727)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 17), 'if_condition_468823', if_condition_468823)
            # SSA begins for if statement (line 727)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to traverse_no_checking(...): (line 728)
            # Processing the call arguments (line 728)
            # Getting the type of 'node1' (line 728)
            node1_468825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 37), 'node1', False)
            # Getting the type of 'node2' (line 728)
            node2_468826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 44), 'node2', False)
            # Processing the call keyword arguments (line 728)
            kwargs_468827 = {}
            # Getting the type of 'traverse_no_checking' (line 728)
            traverse_no_checking_468824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 16), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 728)
            traverse_no_checking_call_result_468828 = invoke(stypy.reporting.localization.Localization(__file__, 728, 16), traverse_no_checking_468824, *[node1_468825, node2_468826], **kwargs_468827)
            
            # SSA branch for the else part of an if statement (line 727)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 729)
            # Processing the call arguments (line 729)
            # Getting the type of 'node1' (line 729)
            node1_468830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 28), 'node1', False)
            # Getting the type of 'KDTree' (line 729)
            KDTree_468831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 35), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 729)
            leafnode_468832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 35), KDTree_468831, 'leafnode')
            # Processing the call keyword arguments (line 729)
            kwargs_468833 = {}
            # Getting the type of 'isinstance' (line 729)
            isinstance_468829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 729)
            isinstance_call_result_468834 = invoke(stypy.reporting.localization.Localization(__file__, 729, 17), isinstance_468829, *[node1_468830, leafnode_468832], **kwargs_468833)
            
            # Testing the type of an if condition (line 729)
            if_condition_468835 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 17), isinstance_call_result_468834)
            # Assigning a type to the variable 'if_condition_468835' (line 729)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 17), 'if_condition_468835', if_condition_468835)
            # SSA begins for if statement (line 729)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to isinstance(...): (line 730)
            # Processing the call arguments (line 730)
            # Getting the type of 'node2' (line 730)
            node2_468837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 30), 'node2', False)
            # Getting the type of 'KDTree' (line 730)
            KDTree_468838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 37), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 730)
            leafnode_468839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 37), KDTree_468838, 'leafnode')
            # Processing the call keyword arguments (line 730)
            kwargs_468840 = {}
            # Getting the type of 'isinstance' (line 730)
            isinstance_468836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 730)
            isinstance_call_result_468841 = invoke(stypy.reporting.localization.Localization(__file__, 730, 19), isinstance_468836, *[node2_468837, leafnode_468839], **kwargs_468840)
            
            # Testing the type of an if condition (line 730)
            if_condition_468842 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 730, 16), isinstance_call_result_468841)
            # Assigning a type to the variable 'if_condition_468842' (line 730)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 16), 'if_condition_468842', if_condition_468842)
            # SSA begins for if statement (line 730)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            
            # Call to id(...): (line 732)
            # Processing the call arguments (line 732)
            # Getting the type of 'node1' (line 732)
            node1_468844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 26), 'node1', False)
            # Processing the call keyword arguments (line 732)
            kwargs_468845 = {}
            # Getting the type of 'id' (line 732)
            id_468843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 23), 'id', False)
            # Calling id(args, kwargs) (line 732)
            id_call_result_468846 = invoke(stypy.reporting.localization.Localization(__file__, 732, 23), id_468843, *[node1_468844], **kwargs_468845)
            
            
            # Call to id(...): (line 732)
            # Processing the call arguments (line 732)
            # Getting the type of 'node2' (line 732)
            node2_468848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 39), 'node2', False)
            # Processing the call keyword arguments (line 732)
            kwargs_468849 = {}
            # Getting the type of 'id' (line 732)
            id_468847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 36), 'id', False)
            # Calling id(args, kwargs) (line 732)
            id_call_result_468850 = invoke(stypy.reporting.localization.Localization(__file__, 732, 36), id_468847, *[node2_468848], **kwargs_468849)
            
            # Applying the binary operator '==' (line 732)
            result_eq_468851 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 23), '==', id_call_result_468846, id_call_result_468850)
            
            # Testing the type of an if condition (line 732)
            if_condition_468852 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 732, 20), result_eq_468851)
            # Assigning a type to the variable 'if_condition_468852' (line 732)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 20), 'if_condition_468852', if_condition_468852)
            # SSA begins for if statement (line 732)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 733):
            
            # Assigning a Subscript to a Name (line 733):
            
            # Obtaining the type of the subscript
            # Getting the type of 'node2' (line 733)
            node2_468853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 38), 'node2')
            # Obtaining the member 'idx' of a type (line 733)
            idx_468854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 38), node2_468853, 'idx')
            # Getting the type of 'self' (line 733)
            self_468855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 28), 'self')
            # Obtaining the member 'data' of a type (line 733)
            data_468856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 28), self_468855, 'data')
            # Obtaining the member '__getitem__' of a type (line 733)
            getitem___468857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 28), data_468856, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 733)
            subscript_call_result_468858 = invoke(stypy.reporting.localization.Localization(__file__, 733, 28), getitem___468857, idx_468854)
            
            # Assigning a type to the variable 'd' (line 733)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 24), 'd', subscript_call_result_468858)
            
            # Getting the type of 'node1' (line 734)
            node1_468859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 33), 'node1')
            # Obtaining the member 'idx' of a type (line 734)
            idx_468860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 33), node1_468859, 'idx')
            # Testing the type of a for loop iterable (line 734)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 734, 24), idx_468860)
            # Getting the type of the for loop variable (line 734)
            for_loop_var_468861 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 734, 24), idx_468860)
            # Assigning a type to the variable 'i' (line 734)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 24), 'i', for_loop_var_468861)
            # SSA begins for a for statement (line 734)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            
            
            # Call to minkowski_distance(...): (line 735)
            # Processing the call arguments (line 735)
            # Getting the type of 'd' (line 735)
            d_468863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 66), 'd', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 735)
            i_468864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 78), 'i', False)
            # Getting the type of 'self' (line 735)
            self_468865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 68), 'self', False)
            # Obtaining the member 'data' of a type (line 735)
            data_468866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 68), self_468865, 'data')
            # Obtaining the member '__getitem__' of a type (line 735)
            getitem___468867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 68), data_468866, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 735)
            subscript_call_result_468868 = invoke(stypy.reporting.localization.Localization(__file__, 735, 68), getitem___468867, i_468864)
            
            # Getting the type of 'p' (line 735)
            p_468869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 81), 'p', False)
            # Processing the call keyword arguments (line 735)
            kwargs_468870 = {}
            # Getting the type of 'minkowski_distance' (line 735)
            minkowski_distance_468862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 47), 'minkowski_distance', False)
            # Calling minkowski_distance(args, kwargs) (line 735)
            minkowski_distance_call_result_468871 = invoke(stypy.reporting.localization.Localization(__file__, 735, 47), minkowski_distance_468862, *[d_468863, subscript_call_result_468868, p_468869], **kwargs_468870)
            
            # Getting the type of 'r' (line 735)
            r_468872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 87), 'r')
            # Applying the binary operator '<=' (line 735)
            result_le_468873 = python_operator(stypy.reporting.localization.Localization(__file__, 735, 47), '<=', minkowski_distance_call_result_468871, r_468872)
            
            # Getting the type of 'node2' (line 735)
            node2_468874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 37), 'node2')
            # Obtaining the member 'idx' of a type (line 735)
            idx_468875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 37), node2_468874, 'idx')
            # Obtaining the member '__getitem__' of a type (line 735)
            getitem___468876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 37), idx_468875, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 735)
            subscript_call_result_468877 = invoke(stypy.reporting.localization.Localization(__file__, 735, 37), getitem___468876, result_le_468873)
            
            # Testing the type of a for loop iterable (line 735)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 735, 28), subscript_call_result_468877)
            # Getting the type of the for loop variable (line 735)
            for_loop_var_468878 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 735, 28), subscript_call_result_468877)
            # Assigning a type to the variable 'j' (line 735)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 28), 'j', for_loop_var_468878)
            # SSA begins for a for statement (line 735)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'i' (line 736)
            i_468879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 35), 'i')
            # Getting the type of 'j' (line 736)
            j_468880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 39), 'j')
            # Applying the binary operator '<' (line 736)
            result_lt_468881 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 35), '<', i_468879, j_468880)
            
            # Testing the type of an if condition (line 736)
            if_condition_468882 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 736, 32), result_lt_468881)
            # Assigning a type to the variable 'if_condition_468882' (line 736)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 32), 'if_condition_468882', if_condition_468882)
            # SSA begins for if statement (line 736)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to add(...): (line 737)
            # Processing the call arguments (line 737)
            
            # Obtaining an instance of the builtin type 'tuple' (line 737)
            tuple_468885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 49), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 737)
            # Adding element type (line 737)
            # Getting the type of 'i' (line 737)
            i_468886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 49), 'i', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 737, 49), tuple_468885, i_468886)
            # Adding element type (line 737)
            # Getting the type of 'j' (line 737)
            j_468887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 51), 'j', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 737, 49), tuple_468885, j_468887)
            
            # Processing the call keyword arguments (line 737)
            kwargs_468888 = {}
            # Getting the type of 'results' (line 737)
            results_468883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 36), 'results', False)
            # Obtaining the member 'add' of a type (line 737)
            add_468884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 36), results_468883, 'add')
            # Calling add(args, kwargs) (line 737)
            add_call_result_468889 = invoke(stypy.reporting.localization.Localization(__file__, 737, 36), add_468884, *[tuple_468885], **kwargs_468888)
            
            # SSA join for if statement (line 736)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 732)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Subscript to a Name (line 739):
            
            # Assigning a Subscript to a Name (line 739):
            
            # Obtaining the type of the subscript
            # Getting the type of 'node2' (line 739)
            node2_468890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 38), 'node2')
            # Obtaining the member 'idx' of a type (line 739)
            idx_468891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 38), node2_468890, 'idx')
            # Getting the type of 'self' (line 739)
            self_468892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 28), 'self')
            # Obtaining the member 'data' of a type (line 739)
            data_468893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 28), self_468892, 'data')
            # Obtaining the member '__getitem__' of a type (line 739)
            getitem___468894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 28), data_468893, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 739)
            subscript_call_result_468895 = invoke(stypy.reporting.localization.Localization(__file__, 739, 28), getitem___468894, idx_468891)
            
            # Assigning a type to the variable 'd' (line 739)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 24), 'd', subscript_call_result_468895)
            
            # Getting the type of 'node1' (line 740)
            node1_468896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 33), 'node1')
            # Obtaining the member 'idx' of a type (line 740)
            idx_468897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 33), node1_468896, 'idx')
            # Testing the type of a for loop iterable (line 740)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 740, 24), idx_468897)
            # Getting the type of the for loop variable (line 740)
            for_loop_var_468898 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 740, 24), idx_468897)
            # Assigning a type to the variable 'i' (line 740)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 24), 'i', for_loop_var_468898)
            # SSA begins for a for statement (line 740)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            
            
            # Call to minkowski_distance(...): (line 741)
            # Processing the call arguments (line 741)
            # Getting the type of 'd' (line 741)
            d_468900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 66), 'd', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 741)
            i_468901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 78), 'i', False)
            # Getting the type of 'self' (line 741)
            self_468902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 68), 'self', False)
            # Obtaining the member 'data' of a type (line 741)
            data_468903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 68), self_468902, 'data')
            # Obtaining the member '__getitem__' of a type (line 741)
            getitem___468904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 68), data_468903, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 741)
            subscript_call_result_468905 = invoke(stypy.reporting.localization.Localization(__file__, 741, 68), getitem___468904, i_468901)
            
            # Getting the type of 'p' (line 741)
            p_468906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 81), 'p', False)
            # Processing the call keyword arguments (line 741)
            kwargs_468907 = {}
            # Getting the type of 'minkowski_distance' (line 741)
            minkowski_distance_468899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 47), 'minkowski_distance', False)
            # Calling minkowski_distance(args, kwargs) (line 741)
            minkowski_distance_call_result_468908 = invoke(stypy.reporting.localization.Localization(__file__, 741, 47), minkowski_distance_468899, *[d_468900, subscript_call_result_468905, p_468906], **kwargs_468907)
            
            # Getting the type of 'r' (line 741)
            r_468909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 87), 'r')
            # Applying the binary operator '<=' (line 741)
            result_le_468910 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 47), '<=', minkowski_distance_call_result_468908, r_468909)
            
            # Getting the type of 'node2' (line 741)
            node2_468911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 37), 'node2')
            # Obtaining the member 'idx' of a type (line 741)
            idx_468912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 37), node2_468911, 'idx')
            # Obtaining the member '__getitem__' of a type (line 741)
            getitem___468913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 37), idx_468912, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 741)
            subscript_call_result_468914 = invoke(stypy.reporting.localization.Localization(__file__, 741, 37), getitem___468913, result_le_468910)
            
            # Testing the type of a for loop iterable (line 741)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 741, 28), subscript_call_result_468914)
            # Getting the type of the for loop variable (line 741)
            for_loop_var_468915 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 741, 28), subscript_call_result_468914)
            # Assigning a type to the variable 'j' (line 741)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 28), 'j', for_loop_var_468915)
            # SSA begins for a for statement (line 741)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'i' (line 742)
            i_468916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 35), 'i')
            # Getting the type of 'j' (line 742)
            j_468917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 39), 'j')
            # Applying the binary operator '<' (line 742)
            result_lt_468918 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 35), '<', i_468916, j_468917)
            
            # Testing the type of an if condition (line 742)
            if_condition_468919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 742, 32), result_lt_468918)
            # Assigning a type to the variable 'if_condition_468919' (line 742)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 32), 'if_condition_468919', if_condition_468919)
            # SSA begins for if statement (line 742)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to add(...): (line 743)
            # Processing the call arguments (line 743)
            
            # Obtaining an instance of the builtin type 'tuple' (line 743)
            tuple_468922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 49), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 743)
            # Adding element type (line 743)
            # Getting the type of 'i' (line 743)
            i_468923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 49), 'i', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 49), tuple_468922, i_468923)
            # Adding element type (line 743)
            # Getting the type of 'j' (line 743)
            j_468924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 51), 'j', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 743, 49), tuple_468922, j_468924)
            
            # Processing the call keyword arguments (line 743)
            kwargs_468925 = {}
            # Getting the type of 'results' (line 743)
            results_468920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 36), 'results', False)
            # Obtaining the member 'add' of a type (line 743)
            add_468921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 36), results_468920, 'add')
            # Calling add(args, kwargs) (line 743)
            add_call_result_468926 = invoke(stypy.reporting.localization.Localization(__file__, 743, 36), add_468921, *[tuple_468922], **kwargs_468925)
            
            # SSA branch for the else part of an if statement (line 742)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'j' (line 744)
            j_468927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 37), 'j')
            # Getting the type of 'i' (line 744)
            i_468928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 41), 'i')
            # Applying the binary operator '<' (line 744)
            result_lt_468929 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 37), '<', j_468927, i_468928)
            
            # Testing the type of an if condition (line 744)
            if_condition_468930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 744, 37), result_lt_468929)
            # Assigning a type to the variable 'if_condition_468930' (line 744)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 37), 'if_condition_468930', if_condition_468930)
            # SSA begins for if statement (line 744)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to add(...): (line 745)
            # Processing the call arguments (line 745)
            
            # Obtaining an instance of the builtin type 'tuple' (line 745)
            tuple_468933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 49), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 745)
            # Adding element type (line 745)
            # Getting the type of 'j' (line 745)
            j_468934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 49), 'j', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 49), tuple_468933, j_468934)
            # Adding element type (line 745)
            # Getting the type of 'i' (line 745)
            i_468935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 51), 'i', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 49), tuple_468933, i_468935)
            
            # Processing the call keyword arguments (line 745)
            kwargs_468936 = {}
            # Getting the type of 'results' (line 745)
            results_468931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 36), 'results', False)
            # Obtaining the member 'add' of a type (line 745)
            add_468932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 36), results_468931, 'add')
            # Calling add(args, kwargs) (line 745)
            add_call_result_468937 = invoke(stypy.reporting.localization.Localization(__file__, 745, 36), add_468932, *[tuple_468933], **kwargs_468936)
            
            # SSA join for if statement (line 744)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 742)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 732)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 730)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 747):
            
            # Assigning a Subscript to a Name (line 747):
            
            # Obtaining the type of the subscript
            int_468938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 20), 'int')
            
            # Call to split(...): (line 747)
            # Processing the call arguments (line 747)
            # Getting the type of 'node2' (line 747)
            node2_468941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 48), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 747)
            split_dim_468942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 48), node2_468941, 'split_dim')
            # Getting the type of 'node2' (line 747)
            node2_468943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 65), 'node2', False)
            # Obtaining the member 'split' of a type (line 747)
            split_468944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 65), node2_468943, 'split')
            # Processing the call keyword arguments (line 747)
            kwargs_468945 = {}
            # Getting the type of 'rect2' (line 747)
            rect2_468939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 36), 'rect2', False)
            # Obtaining the member 'split' of a type (line 747)
            split_468940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 36), rect2_468939, 'split')
            # Calling split(args, kwargs) (line 747)
            split_call_result_468946 = invoke(stypy.reporting.localization.Localization(__file__, 747, 36), split_468940, *[split_dim_468942, split_468944], **kwargs_468945)
            
            # Obtaining the member '__getitem__' of a type (line 747)
            getitem___468947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 20), split_call_result_468946, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 747)
            subscript_call_result_468948 = invoke(stypy.reporting.localization.Localization(__file__, 747, 20), getitem___468947, int_468938)
            
            # Assigning a type to the variable 'tuple_var_assignment_466786' (line 747)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 20), 'tuple_var_assignment_466786', subscript_call_result_468948)
            
            # Assigning a Subscript to a Name (line 747):
            
            # Obtaining the type of the subscript
            int_468949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 20), 'int')
            
            # Call to split(...): (line 747)
            # Processing the call arguments (line 747)
            # Getting the type of 'node2' (line 747)
            node2_468952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 48), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 747)
            split_dim_468953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 48), node2_468952, 'split_dim')
            # Getting the type of 'node2' (line 747)
            node2_468954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 65), 'node2', False)
            # Obtaining the member 'split' of a type (line 747)
            split_468955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 65), node2_468954, 'split')
            # Processing the call keyword arguments (line 747)
            kwargs_468956 = {}
            # Getting the type of 'rect2' (line 747)
            rect2_468950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 36), 'rect2', False)
            # Obtaining the member 'split' of a type (line 747)
            split_468951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 36), rect2_468950, 'split')
            # Calling split(args, kwargs) (line 747)
            split_call_result_468957 = invoke(stypy.reporting.localization.Localization(__file__, 747, 36), split_468951, *[split_dim_468953, split_468955], **kwargs_468956)
            
            # Obtaining the member '__getitem__' of a type (line 747)
            getitem___468958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 20), split_call_result_468957, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 747)
            subscript_call_result_468959 = invoke(stypy.reporting.localization.Localization(__file__, 747, 20), getitem___468958, int_468949)
            
            # Assigning a type to the variable 'tuple_var_assignment_466787' (line 747)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 20), 'tuple_var_assignment_466787', subscript_call_result_468959)
            
            # Assigning a Name to a Name (line 747):
            # Getting the type of 'tuple_var_assignment_466786' (line 747)
            tuple_var_assignment_466786_468960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 20), 'tuple_var_assignment_466786')
            # Assigning a type to the variable 'less' (line 747)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 20), 'less', tuple_var_assignment_466786_468960)
            
            # Assigning a Name to a Name (line 747):
            # Getting the type of 'tuple_var_assignment_466787' (line 747)
            tuple_var_assignment_466787_468961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 20), 'tuple_var_assignment_466787')
            # Assigning a type to the variable 'greater' (line 747)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 26), 'greater', tuple_var_assignment_466787_468961)
            
            # Call to traverse_checking(...): (line 748)
            # Processing the call arguments (line 748)
            # Getting the type of 'node1' (line 748)
            node1_468963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 38), 'node1', False)
            # Getting the type of 'rect1' (line 748)
            rect1_468964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 44), 'rect1', False)
            # Getting the type of 'node2' (line 748)
            node2_468965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 50), 'node2', False)
            # Obtaining the member 'less' of a type (line 748)
            less_468966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 50), node2_468965, 'less')
            # Getting the type of 'less' (line 748)
            less_468967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 61), 'less', False)
            # Processing the call keyword arguments (line 748)
            kwargs_468968 = {}
            # Getting the type of 'traverse_checking' (line 748)
            traverse_checking_468962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 20), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 748)
            traverse_checking_call_result_468969 = invoke(stypy.reporting.localization.Localization(__file__, 748, 20), traverse_checking_468962, *[node1_468963, rect1_468964, less_468966, less_468967], **kwargs_468968)
            
            
            # Call to traverse_checking(...): (line 749)
            # Processing the call arguments (line 749)
            # Getting the type of 'node1' (line 749)
            node1_468971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 38), 'node1', False)
            # Getting the type of 'rect1' (line 749)
            rect1_468972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 44), 'rect1', False)
            # Getting the type of 'node2' (line 749)
            node2_468973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 50), 'node2', False)
            # Obtaining the member 'greater' of a type (line 749)
            greater_468974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 50), node2_468973, 'greater')
            # Getting the type of 'greater' (line 749)
            greater_468975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 64), 'greater', False)
            # Processing the call keyword arguments (line 749)
            kwargs_468976 = {}
            # Getting the type of 'traverse_checking' (line 749)
            traverse_checking_468970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 20), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 749)
            traverse_checking_call_result_468977 = invoke(stypy.reporting.localization.Localization(__file__, 749, 20), traverse_checking_468970, *[node1_468971, rect1_468972, greater_468974, greater_468975], **kwargs_468976)
            
            # SSA join for if statement (line 730)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 729)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 750)
            # Processing the call arguments (line 750)
            # Getting the type of 'node2' (line 750)
            node2_468979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 28), 'node2', False)
            # Getting the type of 'KDTree' (line 750)
            KDTree_468980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 35), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 750)
            leafnode_468981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 35), KDTree_468980, 'leafnode')
            # Processing the call keyword arguments (line 750)
            kwargs_468982 = {}
            # Getting the type of 'isinstance' (line 750)
            isinstance_468978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 750)
            isinstance_call_result_468983 = invoke(stypy.reporting.localization.Localization(__file__, 750, 17), isinstance_468978, *[node2_468979, leafnode_468981], **kwargs_468982)
            
            # Testing the type of an if condition (line 750)
            if_condition_468984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 750, 17), isinstance_call_result_468983)
            # Assigning a type to the variable 'if_condition_468984' (line 750)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 17), 'if_condition_468984', if_condition_468984)
            # SSA begins for if statement (line 750)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 751):
            
            # Assigning a Subscript to a Name (line 751):
            
            # Obtaining the type of the subscript
            int_468985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 16), 'int')
            
            # Call to split(...): (line 751)
            # Processing the call arguments (line 751)
            # Getting the type of 'node1' (line 751)
            node1_468988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 44), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 751)
            split_dim_468989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 44), node1_468988, 'split_dim')
            # Getting the type of 'node1' (line 751)
            node1_468990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 61), 'node1', False)
            # Obtaining the member 'split' of a type (line 751)
            split_468991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 61), node1_468990, 'split')
            # Processing the call keyword arguments (line 751)
            kwargs_468992 = {}
            # Getting the type of 'rect1' (line 751)
            rect1_468986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 32), 'rect1', False)
            # Obtaining the member 'split' of a type (line 751)
            split_468987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 32), rect1_468986, 'split')
            # Calling split(args, kwargs) (line 751)
            split_call_result_468993 = invoke(stypy.reporting.localization.Localization(__file__, 751, 32), split_468987, *[split_dim_468989, split_468991], **kwargs_468992)
            
            # Obtaining the member '__getitem__' of a type (line 751)
            getitem___468994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 16), split_call_result_468993, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 751)
            subscript_call_result_468995 = invoke(stypy.reporting.localization.Localization(__file__, 751, 16), getitem___468994, int_468985)
            
            # Assigning a type to the variable 'tuple_var_assignment_466788' (line 751)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 16), 'tuple_var_assignment_466788', subscript_call_result_468995)
            
            # Assigning a Subscript to a Name (line 751):
            
            # Obtaining the type of the subscript
            int_468996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 16), 'int')
            
            # Call to split(...): (line 751)
            # Processing the call arguments (line 751)
            # Getting the type of 'node1' (line 751)
            node1_468999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 44), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 751)
            split_dim_469000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 44), node1_468999, 'split_dim')
            # Getting the type of 'node1' (line 751)
            node1_469001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 61), 'node1', False)
            # Obtaining the member 'split' of a type (line 751)
            split_469002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 61), node1_469001, 'split')
            # Processing the call keyword arguments (line 751)
            kwargs_469003 = {}
            # Getting the type of 'rect1' (line 751)
            rect1_468997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 32), 'rect1', False)
            # Obtaining the member 'split' of a type (line 751)
            split_468998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 32), rect1_468997, 'split')
            # Calling split(args, kwargs) (line 751)
            split_call_result_469004 = invoke(stypy.reporting.localization.Localization(__file__, 751, 32), split_468998, *[split_dim_469000, split_469002], **kwargs_469003)
            
            # Obtaining the member '__getitem__' of a type (line 751)
            getitem___469005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 16), split_call_result_469004, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 751)
            subscript_call_result_469006 = invoke(stypy.reporting.localization.Localization(__file__, 751, 16), getitem___469005, int_468996)
            
            # Assigning a type to the variable 'tuple_var_assignment_466789' (line 751)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 16), 'tuple_var_assignment_466789', subscript_call_result_469006)
            
            # Assigning a Name to a Name (line 751):
            # Getting the type of 'tuple_var_assignment_466788' (line 751)
            tuple_var_assignment_466788_469007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 16), 'tuple_var_assignment_466788')
            # Assigning a type to the variable 'less' (line 751)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 16), 'less', tuple_var_assignment_466788_469007)
            
            # Assigning a Name to a Name (line 751):
            # Getting the type of 'tuple_var_assignment_466789' (line 751)
            tuple_var_assignment_466789_469008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 16), 'tuple_var_assignment_466789')
            # Assigning a type to the variable 'greater' (line 751)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 22), 'greater', tuple_var_assignment_466789_469008)
            
            # Call to traverse_checking(...): (line 752)
            # Processing the call arguments (line 752)
            # Getting the type of 'node1' (line 752)
            node1_469010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 34), 'node1', False)
            # Obtaining the member 'less' of a type (line 752)
            less_469011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 34), node1_469010, 'less')
            # Getting the type of 'less' (line 752)
            less_469012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 45), 'less', False)
            # Getting the type of 'node2' (line 752)
            node2_469013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 50), 'node2', False)
            # Getting the type of 'rect2' (line 752)
            rect2_469014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 56), 'rect2', False)
            # Processing the call keyword arguments (line 752)
            kwargs_469015 = {}
            # Getting the type of 'traverse_checking' (line 752)
            traverse_checking_469009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 16), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 752)
            traverse_checking_call_result_469016 = invoke(stypy.reporting.localization.Localization(__file__, 752, 16), traverse_checking_469009, *[less_469011, less_469012, node2_469013, rect2_469014], **kwargs_469015)
            
            
            # Call to traverse_checking(...): (line 753)
            # Processing the call arguments (line 753)
            # Getting the type of 'node1' (line 753)
            node1_469018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 34), 'node1', False)
            # Obtaining the member 'greater' of a type (line 753)
            greater_469019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 34), node1_469018, 'greater')
            # Getting the type of 'greater' (line 753)
            greater_469020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 48), 'greater', False)
            # Getting the type of 'node2' (line 753)
            node2_469021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 56), 'node2', False)
            # Getting the type of 'rect2' (line 753)
            rect2_469022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 62), 'rect2', False)
            # Processing the call keyword arguments (line 753)
            kwargs_469023 = {}
            # Getting the type of 'traverse_checking' (line 753)
            traverse_checking_469017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 16), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 753)
            traverse_checking_call_result_469024 = invoke(stypy.reporting.localization.Localization(__file__, 753, 16), traverse_checking_469017, *[greater_469019, greater_469020, node2_469021, rect2_469022], **kwargs_469023)
            
            # SSA branch for the else part of an if statement (line 750)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 755):
            
            # Assigning a Subscript to a Name (line 755):
            
            # Obtaining the type of the subscript
            int_469025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 16), 'int')
            
            # Call to split(...): (line 755)
            # Processing the call arguments (line 755)
            # Getting the type of 'node1' (line 755)
            node1_469028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 46), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 755)
            split_dim_469029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 46), node1_469028, 'split_dim')
            # Getting the type of 'node1' (line 755)
            node1_469030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 63), 'node1', False)
            # Obtaining the member 'split' of a type (line 755)
            split_469031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 63), node1_469030, 'split')
            # Processing the call keyword arguments (line 755)
            kwargs_469032 = {}
            # Getting the type of 'rect1' (line 755)
            rect1_469026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 34), 'rect1', False)
            # Obtaining the member 'split' of a type (line 755)
            split_469027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 34), rect1_469026, 'split')
            # Calling split(args, kwargs) (line 755)
            split_call_result_469033 = invoke(stypy.reporting.localization.Localization(__file__, 755, 34), split_469027, *[split_dim_469029, split_469031], **kwargs_469032)
            
            # Obtaining the member '__getitem__' of a type (line 755)
            getitem___469034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 16), split_call_result_469033, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 755)
            subscript_call_result_469035 = invoke(stypy.reporting.localization.Localization(__file__, 755, 16), getitem___469034, int_469025)
            
            # Assigning a type to the variable 'tuple_var_assignment_466790' (line 755)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 16), 'tuple_var_assignment_466790', subscript_call_result_469035)
            
            # Assigning a Subscript to a Name (line 755):
            
            # Obtaining the type of the subscript
            int_469036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 16), 'int')
            
            # Call to split(...): (line 755)
            # Processing the call arguments (line 755)
            # Getting the type of 'node1' (line 755)
            node1_469039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 46), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 755)
            split_dim_469040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 46), node1_469039, 'split_dim')
            # Getting the type of 'node1' (line 755)
            node1_469041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 63), 'node1', False)
            # Obtaining the member 'split' of a type (line 755)
            split_469042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 63), node1_469041, 'split')
            # Processing the call keyword arguments (line 755)
            kwargs_469043 = {}
            # Getting the type of 'rect1' (line 755)
            rect1_469037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 34), 'rect1', False)
            # Obtaining the member 'split' of a type (line 755)
            split_469038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 34), rect1_469037, 'split')
            # Calling split(args, kwargs) (line 755)
            split_call_result_469044 = invoke(stypy.reporting.localization.Localization(__file__, 755, 34), split_469038, *[split_dim_469040, split_469042], **kwargs_469043)
            
            # Obtaining the member '__getitem__' of a type (line 755)
            getitem___469045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 16), split_call_result_469044, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 755)
            subscript_call_result_469046 = invoke(stypy.reporting.localization.Localization(__file__, 755, 16), getitem___469045, int_469036)
            
            # Assigning a type to the variable 'tuple_var_assignment_466791' (line 755)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 16), 'tuple_var_assignment_466791', subscript_call_result_469046)
            
            # Assigning a Name to a Name (line 755):
            # Getting the type of 'tuple_var_assignment_466790' (line 755)
            tuple_var_assignment_466790_469047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 16), 'tuple_var_assignment_466790')
            # Assigning a type to the variable 'less1' (line 755)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 16), 'less1', tuple_var_assignment_466790_469047)
            
            # Assigning a Name to a Name (line 755):
            # Getting the type of 'tuple_var_assignment_466791' (line 755)
            tuple_var_assignment_466791_469048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 16), 'tuple_var_assignment_466791')
            # Assigning a type to the variable 'greater1' (line 755)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 23), 'greater1', tuple_var_assignment_466791_469048)
            
            # Assigning a Call to a Tuple (line 756):
            
            # Assigning a Subscript to a Name (line 756):
            
            # Obtaining the type of the subscript
            int_469049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 16), 'int')
            
            # Call to split(...): (line 756)
            # Processing the call arguments (line 756)
            # Getting the type of 'node2' (line 756)
            node2_469052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 46), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 756)
            split_dim_469053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 46), node2_469052, 'split_dim')
            # Getting the type of 'node2' (line 756)
            node2_469054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 63), 'node2', False)
            # Obtaining the member 'split' of a type (line 756)
            split_469055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 63), node2_469054, 'split')
            # Processing the call keyword arguments (line 756)
            kwargs_469056 = {}
            # Getting the type of 'rect2' (line 756)
            rect2_469050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 34), 'rect2', False)
            # Obtaining the member 'split' of a type (line 756)
            split_469051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 34), rect2_469050, 'split')
            # Calling split(args, kwargs) (line 756)
            split_call_result_469057 = invoke(stypy.reporting.localization.Localization(__file__, 756, 34), split_469051, *[split_dim_469053, split_469055], **kwargs_469056)
            
            # Obtaining the member '__getitem__' of a type (line 756)
            getitem___469058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 16), split_call_result_469057, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 756)
            subscript_call_result_469059 = invoke(stypy.reporting.localization.Localization(__file__, 756, 16), getitem___469058, int_469049)
            
            # Assigning a type to the variable 'tuple_var_assignment_466792' (line 756)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 16), 'tuple_var_assignment_466792', subscript_call_result_469059)
            
            # Assigning a Subscript to a Name (line 756):
            
            # Obtaining the type of the subscript
            int_469060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 16), 'int')
            
            # Call to split(...): (line 756)
            # Processing the call arguments (line 756)
            # Getting the type of 'node2' (line 756)
            node2_469063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 46), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 756)
            split_dim_469064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 46), node2_469063, 'split_dim')
            # Getting the type of 'node2' (line 756)
            node2_469065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 63), 'node2', False)
            # Obtaining the member 'split' of a type (line 756)
            split_469066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 63), node2_469065, 'split')
            # Processing the call keyword arguments (line 756)
            kwargs_469067 = {}
            # Getting the type of 'rect2' (line 756)
            rect2_469061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 34), 'rect2', False)
            # Obtaining the member 'split' of a type (line 756)
            split_469062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 34), rect2_469061, 'split')
            # Calling split(args, kwargs) (line 756)
            split_call_result_469068 = invoke(stypy.reporting.localization.Localization(__file__, 756, 34), split_469062, *[split_dim_469064, split_469066], **kwargs_469067)
            
            # Obtaining the member '__getitem__' of a type (line 756)
            getitem___469069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 16), split_call_result_469068, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 756)
            subscript_call_result_469070 = invoke(stypy.reporting.localization.Localization(__file__, 756, 16), getitem___469069, int_469060)
            
            # Assigning a type to the variable 'tuple_var_assignment_466793' (line 756)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 16), 'tuple_var_assignment_466793', subscript_call_result_469070)
            
            # Assigning a Name to a Name (line 756):
            # Getting the type of 'tuple_var_assignment_466792' (line 756)
            tuple_var_assignment_466792_469071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 16), 'tuple_var_assignment_466792')
            # Assigning a type to the variable 'less2' (line 756)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 16), 'less2', tuple_var_assignment_466792_469071)
            
            # Assigning a Name to a Name (line 756):
            # Getting the type of 'tuple_var_assignment_466793' (line 756)
            tuple_var_assignment_466793_469072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 16), 'tuple_var_assignment_466793')
            # Assigning a type to the variable 'greater2' (line 756)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 23), 'greater2', tuple_var_assignment_466793_469072)
            
            # Call to traverse_checking(...): (line 757)
            # Processing the call arguments (line 757)
            # Getting the type of 'node1' (line 757)
            node1_469074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 34), 'node1', False)
            # Obtaining the member 'less' of a type (line 757)
            less_469075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 34), node1_469074, 'less')
            # Getting the type of 'less1' (line 757)
            less1_469076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 45), 'less1', False)
            # Getting the type of 'node2' (line 757)
            node2_469077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 51), 'node2', False)
            # Obtaining the member 'less' of a type (line 757)
            less_469078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 51), node2_469077, 'less')
            # Getting the type of 'less2' (line 757)
            less2_469079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 62), 'less2', False)
            # Processing the call keyword arguments (line 757)
            kwargs_469080 = {}
            # Getting the type of 'traverse_checking' (line 757)
            traverse_checking_469073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 16), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 757)
            traverse_checking_call_result_469081 = invoke(stypy.reporting.localization.Localization(__file__, 757, 16), traverse_checking_469073, *[less_469075, less1_469076, less_469078, less2_469079], **kwargs_469080)
            
            
            # Call to traverse_checking(...): (line 758)
            # Processing the call arguments (line 758)
            # Getting the type of 'node1' (line 758)
            node1_469083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 34), 'node1', False)
            # Obtaining the member 'less' of a type (line 758)
            less_469084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 34), node1_469083, 'less')
            # Getting the type of 'less1' (line 758)
            less1_469085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 45), 'less1', False)
            # Getting the type of 'node2' (line 758)
            node2_469086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 51), 'node2', False)
            # Obtaining the member 'greater' of a type (line 758)
            greater_469087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 51), node2_469086, 'greater')
            # Getting the type of 'greater2' (line 758)
            greater2_469088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 65), 'greater2', False)
            # Processing the call keyword arguments (line 758)
            kwargs_469089 = {}
            # Getting the type of 'traverse_checking' (line 758)
            traverse_checking_469082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 16), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 758)
            traverse_checking_call_result_469090 = invoke(stypy.reporting.localization.Localization(__file__, 758, 16), traverse_checking_469082, *[less_469084, less1_469085, greater_469087, greater2_469088], **kwargs_469089)
            
            
            
            
            # Call to id(...): (line 764)
            # Processing the call arguments (line 764)
            # Getting the type of 'node1' (line 764)
            node1_469092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 22), 'node1', False)
            # Processing the call keyword arguments (line 764)
            kwargs_469093 = {}
            # Getting the type of 'id' (line 764)
            id_469091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 19), 'id', False)
            # Calling id(args, kwargs) (line 764)
            id_call_result_469094 = invoke(stypy.reporting.localization.Localization(__file__, 764, 19), id_469091, *[node1_469092], **kwargs_469093)
            
            
            # Call to id(...): (line 764)
            # Processing the call arguments (line 764)
            # Getting the type of 'node2' (line 764)
            node2_469096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 35), 'node2', False)
            # Processing the call keyword arguments (line 764)
            kwargs_469097 = {}
            # Getting the type of 'id' (line 764)
            id_469095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 32), 'id', False)
            # Calling id(args, kwargs) (line 764)
            id_call_result_469098 = invoke(stypy.reporting.localization.Localization(__file__, 764, 32), id_469095, *[node2_469096], **kwargs_469097)
            
            # Applying the binary operator '!=' (line 764)
            result_ne_469099 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 19), '!=', id_call_result_469094, id_call_result_469098)
            
            # Testing the type of an if condition (line 764)
            if_condition_469100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 764, 16), result_ne_469099)
            # Assigning a type to the variable 'if_condition_469100' (line 764)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 16), 'if_condition_469100', if_condition_469100)
            # SSA begins for if statement (line 764)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to traverse_checking(...): (line 765)
            # Processing the call arguments (line 765)
            # Getting the type of 'node1' (line 765)
            node1_469102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 38), 'node1', False)
            # Obtaining the member 'greater' of a type (line 765)
            greater_469103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 38), node1_469102, 'greater')
            # Getting the type of 'greater1' (line 765)
            greater1_469104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 52), 'greater1', False)
            # Getting the type of 'node2' (line 765)
            node2_469105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 61), 'node2', False)
            # Obtaining the member 'less' of a type (line 765)
            less_469106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 61), node2_469105, 'less')
            # Getting the type of 'less2' (line 765)
            less2_469107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 72), 'less2', False)
            # Processing the call keyword arguments (line 765)
            kwargs_469108 = {}
            # Getting the type of 'traverse_checking' (line 765)
            traverse_checking_469101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 20), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 765)
            traverse_checking_call_result_469109 = invoke(stypy.reporting.localization.Localization(__file__, 765, 20), traverse_checking_469101, *[greater_469103, greater1_469104, less_469106, less2_469107], **kwargs_469108)
            
            # SSA join for if statement (line 764)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to traverse_checking(...): (line 767)
            # Processing the call arguments (line 767)
            # Getting the type of 'node1' (line 767)
            node1_469111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 34), 'node1', False)
            # Obtaining the member 'greater' of a type (line 767)
            greater_469112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 34), node1_469111, 'greater')
            # Getting the type of 'greater1' (line 767)
            greater1_469113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 48), 'greater1', False)
            # Getting the type of 'node2' (line 767)
            node2_469114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 57), 'node2', False)
            # Obtaining the member 'greater' of a type (line 767)
            greater_469115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 57), node2_469114, 'greater')
            # Getting the type of 'greater2' (line 767)
            greater2_469116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 71), 'greater2', False)
            # Processing the call keyword arguments (line 767)
            kwargs_469117 = {}
            # Getting the type of 'traverse_checking' (line 767)
            traverse_checking_469110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 16), 'traverse_checking', False)
            # Calling traverse_checking(args, kwargs) (line 767)
            traverse_checking_call_result_469118 = invoke(stypy.reporting.localization.Localization(__file__, 767, 16), traverse_checking_469110, *[greater_469112, greater1_469113, greater_469115, greater2_469116], **kwargs_469117)
            
            # SSA join for if statement (line 750)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 729)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 727)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 725)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'traverse_checking(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'traverse_checking' in the type store
            # Getting the type of 'stypy_return_type' (line 724)
            stypy_return_type_469119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_469119)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'traverse_checking'
            return stypy_return_type_469119

        # Assigning a type to the variable 'traverse_checking' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'traverse_checking', traverse_checking)

        @norecursion
        def traverse_no_checking(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'traverse_no_checking'
            module_type_store = module_type_store.open_function_context('traverse_no_checking', 769, 8, False)
            
            # Passed parameters checking function
            traverse_no_checking.stypy_localization = localization
            traverse_no_checking.stypy_type_of_self = None
            traverse_no_checking.stypy_type_store = module_type_store
            traverse_no_checking.stypy_function_name = 'traverse_no_checking'
            traverse_no_checking.stypy_param_names_list = ['node1', 'node2']
            traverse_no_checking.stypy_varargs_param_name = None
            traverse_no_checking.stypy_kwargs_param_name = None
            traverse_no_checking.stypy_call_defaults = defaults
            traverse_no_checking.stypy_call_varargs = varargs
            traverse_no_checking.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'traverse_no_checking', ['node1', 'node2'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'traverse_no_checking', localization, ['node1', 'node2'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'traverse_no_checking(...)' code ##################

            
            
            # Call to isinstance(...): (line 770)
            # Processing the call arguments (line 770)
            # Getting the type of 'node1' (line 770)
            node1_469121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 26), 'node1', False)
            # Getting the type of 'KDTree' (line 770)
            KDTree_469122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 33), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 770)
            leafnode_469123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 33), KDTree_469122, 'leafnode')
            # Processing the call keyword arguments (line 770)
            kwargs_469124 = {}
            # Getting the type of 'isinstance' (line 770)
            isinstance_469120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 770)
            isinstance_call_result_469125 = invoke(stypy.reporting.localization.Localization(__file__, 770, 15), isinstance_469120, *[node1_469121, leafnode_469123], **kwargs_469124)
            
            # Testing the type of an if condition (line 770)
            if_condition_469126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 770, 12), isinstance_call_result_469125)
            # Assigning a type to the variable 'if_condition_469126' (line 770)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'if_condition_469126', if_condition_469126)
            # SSA begins for if statement (line 770)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to isinstance(...): (line 771)
            # Processing the call arguments (line 771)
            # Getting the type of 'node2' (line 771)
            node2_469128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 30), 'node2', False)
            # Getting the type of 'KDTree' (line 771)
            KDTree_469129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 37), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 771)
            leafnode_469130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 37), KDTree_469129, 'leafnode')
            # Processing the call keyword arguments (line 771)
            kwargs_469131 = {}
            # Getting the type of 'isinstance' (line 771)
            isinstance_469127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 771)
            isinstance_call_result_469132 = invoke(stypy.reporting.localization.Localization(__file__, 771, 19), isinstance_469127, *[node2_469128, leafnode_469130], **kwargs_469131)
            
            # Testing the type of an if condition (line 771)
            if_condition_469133 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 771, 16), isinstance_call_result_469132)
            # Assigning a type to the variable 'if_condition_469133' (line 771)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 16), 'if_condition_469133', if_condition_469133)
            # SSA begins for if statement (line 771)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            
            # Call to id(...): (line 773)
            # Processing the call arguments (line 773)
            # Getting the type of 'node1' (line 773)
            node1_469135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 26), 'node1', False)
            # Processing the call keyword arguments (line 773)
            kwargs_469136 = {}
            # Getting the type of 'id' (line 773)
            id_469134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 23), 'id', False)
            # Calling id(args, kwargs) (line 773)
            id_call_result_469137 = invoke(stypy.reporting.localization.Localization(__file__, 773, 23), id_469134, *[node1_469135], **kwargs_469136)
            
            
            # Call to id(...): (line 773)
            # Processing the call arguments (line 773)
            # Getting the type of 'node2' (line 773)
            node2_469139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 39), 'node2', False)
            # Processing the call keyword arguments (line 773)
            kwargs_469140 = {}
            # Getting the type of 'id' (line 773)
            id_469138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 36), 'id', False)
            # Calling id(args, kwargs) (line 773)
            id_call_result_469141 = invoke(stypy.reporting.localization.Localization(__file__, 773, 36), id_469138, *[node2_469139], **kwargs_469140)
            
            # Applying the binary operator '==' (line 773)
            result_eq_469142 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 23), '==', id_call_result_469137, id_call_result_469141)
            
            # Testing the type of an if condition (line 773)
            if_condition_469143 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 773, 20), result_eq_469142)
            # Assigning a type to the variable 'if_condition_469143' (line 773)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 20), 'if_condition_469143', if_condition_469143)
            # SSA begins for if statement (line 773)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'node1' (line 774)
            node1_469144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 33), 'node1')
            # Obtaining the member 'idx' of a type (line 774)
            idx_469145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 33), node1_469144, 'idx')
            # Testing the type of a for loop iterable (line 774)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 774, 24), idx_469145)
            # Getting the type of the for loop variable (line 774)
            for_loop_var_469146 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 774, 24), idx_469145)
            # Assigning a type to the variable 'i' (line 774)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 24), 'i', for_loop_var_469146)
            # SSA begins for a for statement (line 774)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'node2' (line 775)
            node2_469147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 37), 'node2')
            # Obtaining the member 'idx' of a type (line 775)
            idx_469148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 37), node2_469147, 'idx')
            # Testing the type of a for loop iterable (line 775)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 775, 28), idx_469148)
            # Getting the type of the for loop variable (line 775)
            for_loop_var_469149 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 775, 28), idx_469148)
            # Assigning a type to the variable 'j' (line 775)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 28), 'j', for_loop_var_469149)
            # SSA begins for a for statement (line 775)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'i' (line 776)
            i_469150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 35), 'i')
            # Getting the type of 'j' (line 776)
            j_469151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 39), 'j')
            # Applying the binary operator '<' (line 776)
            result_lt_469152 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 35), '<', i_469150, j_469151)
            
            # Testing the type of an if condition (line 776)
            if_condition_469153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 776, 32), result_lt_469152)
            # Assigning a type to the variable 'if_condition_469153' (line 776)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 32), 'if_condition_469153', if_condition_469153)
            # SSA begins for if statement (line 776)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to add(...): (line 777)
            # Processing the call arguments (line 777)
            
            # Obtaining an instance of the builtin type 'tuple' (line 777)
            tuple_469156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 49), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 777)
            # Adding element type (line 777)
            # Getting the type of 'i' (line 777)
            i_469157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 49), 'i', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 49), tuple_469156, i_469157)
            # Adding element type (line 777)
            # Getting the type of 'j' (line 777)
            j_469158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 51), 'j', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 49), tuple_469156, j_469158)
            
            # Processing the call keyword arguments (line 777)
            kwargs_469159 = {}
            # Getting the type of 'results' (line 777)
            results_469154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 36), 'results', False)
            # Obtaining the member 'add' of a type (line 777)
            add_469155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 36), results_469154, 'add')
            # Calling add(args, kwargs) (line 777)
            add_call_result_469160 = invoke(stypy.reporting.localization.Localization(__file__, 777, 36), add_469155, *[tuple_469156], **kwargs_469159)
            
            # SSA join for if statement (line 776)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 773)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'node1' (line 779)
            node1_469161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 33), 'node1')
            # Obtaining the member 'idx' of a type (line 779)
            idx_469162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 33), node1_469161, 'idx')
            # Testing the type of a for loop iterable (line 779)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 779, 24), idx_469162)
            # Getting the type of the for loop variable (line 779)
            for_loop_var_469163 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 779, 24), idx_469162)
            # Assigning a type to the variable 'i' (line 779)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 24), 'i', for_loop_var_469163)
            # SSA begins for a for statement (line 779)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'node2' (line 780)
            node2_469164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 37), 'node2')
            # Obtaining the member 'idx' of a type (line 780)
            idx_469165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 37), node2_469164, 'idx')
            # Testing the type of a for loop iterable (line 780)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 780, 28), idx_469165)
            # Getting the type of the for loop variable (line 780)
            for_loop_var_469166 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 780, 28), idx_469165)
            # Assigning a type to the variable 'j' (line 780)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 28), 'j', for_loop_var_469166)
            # SSA begins for a for statement (line 780)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'i' (line 781)
            i_469167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 35), 'i')
            # Getting the type of 'j' (line 781)
            j_469168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 39), 'j')
            # Applying the binary operator '<' (line 781)
            result_lt_469169 = python_operator(stypy.reporting.localization.Localization(__file__, 781, 35), '<', i_469167, j_469168)
            
            # Testing the type of an if condition (line 781)
            if_condition_469170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 781, 32), result_lt_469169)
            # Assigning a type to the variable 'if_condition_469170' (line 781)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 32), 'if_condition_469170', if_condition_469170)
            # SSA begins for if statement (line 781)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to add(...): (line 782)
            # Processing the call arguments (line 782)
            
            # Obtaining an instance of the builtin type 'tuple' (line 782)
            tuple_469173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 49), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 782)
            # Adding element type (line 782)
            # Getting the type of 'i' (line 782)
            i_469174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 49), 'i', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 782, 49), tuple_469173, i_469174)
            # Adding element type (line 782)
            # Getting the type of 'j' (line 782)
            j_469175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 51), 'j', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 782, 49), tuple_469173, j_469175)
            
            # Processing the call keyword arguments (line 782)
            kwargs_469176 = {}
            # Getting the type of 'results' (line 782)
            results_469171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 36), 'results', False)
            # Obtaining the member 'add' of a type (line 782)
            add_469172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 36), results_469171, 'add')
            # Calling add(args, kwargs) (line 782)
            add_call_result_469177 = invoke(stypy.reporting.localization.Localization(__file__, 782, 36), add_469172, *[tuple_469173], **kwargs_469176)
            
            # SSA branch for the else part of an if statement (line 781)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'j' (line 783)
            j_469178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 37), 'j')
            # Getting the type of 'i' (line 783)
            i_469179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 41), 'i')
            # Applying the binary operator '<' (line 783)
            result_lt_469180 = python_operator(stypy.reporting.localization.Localization(__file__, 783, 37), '<', j_469178, i_469179)
            
            # Testing the type of an if condition (line 783)
            if_condition_469181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 783, 37), result_lt_469180)
            # Assigning a type to the variable 'if_condition_469181' (line 783)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 37), 'if_condition_469181', if_condition_469181)
            # SSA begins for if statement (line 783)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to add(...): (line 784)
            # Processing the call arguments (line 784)
            
            # Obtaining an instance of the builtin type 'tuple' (line 784)
            tuple_469184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 49), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 784)
            # Adding element type (line 784)
            # Getting the type of 'j' (line 784)
            j_469185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 49), 'j', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 784, 49), tuple_469184, j_469185)
            # Adding element type (line 784)
            # Getting the type of 'i' (line 784)
            i_469186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 51), 'i', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 784, 49), tuple_469184, i_469186)
            
            # Processing the call keyword arguments (line 784)
            kwargs_469187 = {}
            # Getting the type of 'results' (line 784)
            results_469182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 36), 'results', False)
            # Obtaining the member 'add' of a type (line 784)
            add_469183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 36), results_469182, 'add')
            # Calling add(args, kwargs) (line 784)
            add_call_result_469188 = invoke(stypy.reporting.localization.Localization(__file__, 784, 36), add_469183, *[tuple_469184], **kwargs_469187)
            
            # SSA join for if statement (line 783)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 781)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 773)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 771)
            module_type_store.open_ssa_branch('else')
            
            # Call to traverse_no_checking(...): (line 786)
            # Processing the call arguments (line 786)
            # Getting the type of 'node1' (line 786)
            node1_469190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 41), 'node1', False)
            # Getting the type of 'node2' (line 786)
            node2_469191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 48), 'node2', False)
            # Obtaining the member 'less' of a type (line 786)
            less_469192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 48), node2_469191, 'less')
            # Processing the call keyword arguments (line 786)
            kwargs_469193 = {}
            # Getting the type of 'traverse_no_checking' (line 786)
            traverse_no_checking_469189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 20), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 786)
            traverse_no_checking_call_result_469194 = invoke(stypy.reporting.localization.Localization(__file__, 786, 20), traverse_no_checking_469189, *[node1_469190, less_469192], **kwargs_469193)
            
            
            # Call to traverse_no_checking(...): (line 787)
            # Processing the call arguments (line 787)
            # Getting the type of 'node1' (line 787)
            node1_469196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 41), 'node1', False)
            # Getting the type of 'node2' (line 787)
            node2_469197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 48), 'node2', False)
            # Obtaining the member 'greater' of a type (line 787)
            greater_469198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 48), node2_469197, 'greater')
            # Processing the call keyword arguments (line 787)
            kwargs_469199 = {}
            # Getting the type of 'traverse_no_checking' (line 787)
            traverse_no_checking_469195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 20), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 787)
            traverse_no_checking_call_result_469200 = invoke(stypy.reporting.localization.Localization(__file__, 787, 20), traverse_no_checking_469195, *[node1_469196, greater_469198], **kwargs_469199)
            
            # SSA join for if statement (line 771)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 770)
            module_type_store.open_ssa_branch('else')
            
            
            
            # Call to id(...): (line 793)
            # Processing the call arguments (line 793)
            # Getting the type of 'node1' (line 793)
            node1_469202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 22), 'node1', False)
            # Processing the call keyword arguments (line 793)
            kwargs_469203 = {}
            # Getting the type of 'id' (line 793)
            id_469201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 19), 'id', False)
            # Calling id(args, kwargs) (line 793)
            id_call_result_469204 = invoke(stypy.reporting.localization.Localization(__file__, 793, 19), id_469201, *[node1_469202], **kwargs_469203)
            
            
            # Call to id(...): (line 793)
            # Processing the call arguments (line 793)
            # Getting the type of 'node2' (line 793)
            node2_469206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 35), 'node2', False)
            # Processing the call keyword arguments (line 793)
            kwargs_469207 = {}
            # Getting the type of 'id' (line 793)
            id_469205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 32), 'id', False)
            # Calling id(args, kwargs) (line 793)
            id_call_result_469208 = invoke(stypy.reporting.localization.Localization(__file__, 793, 32), id_469205, *[node2_469206], **kwargs_469207)
            
            # Applying the binary operator '==' (line 793)
            result_eq_469209 = python_operator(stypy.reporting.localization.Localization(__file__, 793, 19), '==', id_call_result_469204, id_call_result_469208)
            
            # Testing the type of an if condition (line 793)
            if_condition_469210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 793, 16), result_eq_469209)
            # Assigning a type to the variable 'if_condition_469210' (line 793)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 16), 'if_condition_469210', if_condition_469210)
            # SSA begins for if statement (line 793)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to traverse_no_checking(...): (line 794)
            # Processing the call arguments (line 794)
            # Getting the type of 'node1' (line 794)
            node1_469212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 41), 'node1', False)
            # Obtaining the member 'less' of a type (line 794)
            less_469213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 41), node1_469212, 'less')
            # Getting the type of 'node2' (line 794)
            node2_469214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 53), 'node2', False)
            # Obtaining the member 'less' of a type (line 794)
            less_469215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 53), node2_469214, 'less')
            # Processing the call keyword arguments (line 794)
            kwargs_469216 = {}
            # Getting the type of 'traverse_no_checking' (line 794)
            traverse_no_checking_469211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 20), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 794)
            traverse_no_checking_call_result_469217 = invoke(stypy.reporting.localization.Localization(__file__, 794, 20), traverse_no_checking_469211, *[less_469213, less_469215], **kwargs_469216)
            
            
            # Call to traverse_no_checking(...): (line 795)
            # Processing the call arguments (line 795)
            # Getting the type of 'node1' (line 795)
            node1_469219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 41), 'node1', False)
            # Obtaining the member 'less' of a type (line 795)
            less_469220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 41), node1_469219, 'less')
            # Getting the type of 'node2' (line 795)
            node2_469221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 53), 'node2', False)
            # Obtaining the member 'greater' of a type (line 795)
            greater_469222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 53), node2_469221, 'greater')
            # Processing the call keyword arguments (line 795)
            kwargs_469223 = {}
            # Getting the type of 'traverse_no_checking' (line 795)
            traverse_no_checking_469218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 20), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 795)
            traverse_no_checking_call_result_469224 = invoke(stypy.reporting.localization.Localization(__file__, 795, 20), traverse_no_checking_469218, *[less_469220, greater_469222], **kwargs_469223)
            
            
            # Call to traverse_no_checking(...): (line 796)
            # Processing the call arguments (line 796)
            # Getting the type of 'node1' (line 796)
            node1_469226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 41), 'node1', False)
            # Obtaining the member 'greater' of a type (line 796)
            greater_469227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 41), node1_469226, 'greater')
            # Getting the type of 'node2' (line 796)
            node2_469228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 56), 'node2', False)
            # Obtaining the member 'greater' of a type (line 796)
            greater_469229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 56), node2_469228, 'greater')
            # Processing the call keyword arguments (line 796)
            kwargs_469230 = {}
            # Getting the type of 'traverse_no_checking' (line 796)
            traverse_no_checking_469225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 20), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 796)
            traverse_no_checking_call_result_469231 = invoke(stypy.reporting.localization.Localization(__file__, 796, 20), traverse_no_checking_469225, *[greater_469227, greater_469229], **kwargs_469230)
            
            # SSA branch for the else part of an if statement (line 793)
            module_type_store.open_ssa_branch('else')
            
            # Call to traverse_no_checking(...): (line 798)
            # Processing the call arguments (line 798)
            # Getting the type of 'node1' (line 798)
            node1_469233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 41), 'node1', False)
            # Obtaining the member 'less' of a type (line 798)
            less_469234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 41), node1_469233, 'less')
            # Getting the type of 'node2' (line 798)
            node2_469235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 53), 'node2', False)
            # Processing the call keyword arguments (line 798)
            kwargs_469236 = {}
            # Getting the type of 'traverse_no_checking' (line 798)
            traverse_no_checking_469232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 20), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 798)
            traverse_no_checking_call_result_469237 = invoke(stypy.reporting.localization.Localization(__file__, 798, 20), traverse_no_checking_469232, *[less_469234, node2_469235], **kwargs_469236)
            
            
            # Call to traverse_no_checking(...): (line 799)
            # Processing the call arguments (line 799)
            # Getting the type of 'node1' (line 799)
            node1_469239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 41), 'node1', False)
            # Obtaining the member 'greater' of a type (line 799)
            greater_469240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 41), node1_469239, 'greater')
            # Getting the type of 'node2' (line 799)
            node2_469241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 56), 'node2', False)
            # Processing the call keyword arguments (line 799)
            kwargs_469242 = {}
            # Getting the type of 'traverse_no_checking' (line 799)
            traverse_no_checking_469238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 20), 'traverse_no_checking', False)
            # Calling traverse_no_checking(args, kwargs) (line 799)
            traverse_no_checking_call_result_469243 = invoke(stypy.reporting.localization.Localization(__file__, 799, 20), traverse_no_checking_469238, *[greater_469240, node2_469241], **kwargs_469242)
            
            # SSA join for if statement (line 793)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 770)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'traverse_no_checking(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'traverse_no_checking' in the type store
            # Getting the type of 'stypy_return_type' (line 769)
            stypy_return_type_469244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_469244)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'traverse_no_checking'
            return stypy_return_type_469244

        # Assigning a type to the variable 'traverse_no_checking' (line 769)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 8), 'traverse_no_checking', traverse_no_checking)
        
        # Call to traverse_checking(...): (line 801)
        # Processing the call arguments (line 801)
        # Getting the type of 'self' (line 801)
        self_469246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 26), 'self', False)
        # Obtaining the member 'tree' of a type (line 801)
        tree_469247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 26), self_469246, 'tree')
        
        # Call to Rectangle(...): (line 801)
        # Processing the call arguments (line 801)
        # Getting the type of 'self' (line 801)
        self_469249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 47), 'self', False)
        # Obtaining the member 'maxes' of a type (line 801)
        maxes_469250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 47), self_469249, 'maxes')
        # Getting the type of 'self' (line 801)
        self_469251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 59), 'self', False)
        # Obtaining the member 'mins' of a type (line 801)
        mins_469252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 59), self_469251, 'mins')
        # Processing the call keyword arguments (line 801)
        kwargs_469253 = {}
        # Getting the type of 'Rectangle' (line 801)
        Rectangle_469248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 37), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 801)
        Rectangle_call_result_469254 = invoke(stypy.reporting.localization.Localization(__file__, 801, 37), Rectangle_469248, *[maxes_469250, mins_469252], **kwargs_469253)
        
        # Getting the type of 'self' (line 802)
        self_469255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 26), 'self', False)
        # Obtaining the member 'tree' of a type (line 802)
        tree_469256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 26), self_469255, 'tree')
        
        # Call to Rectangle(...): (line 802)
        # Processing the call arguments (line 802)
        # Getting the type of 'self' (line 802)
        self_469258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 47), 'self', False)
        # Obtaining the member 'maxes' of a type (line 802)
        maxes_469259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 47), self_469258, 'maxes')
        # Getting the type of 'self' (line 802)
        self_469260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 59), 'self', False)
        # Obtaining the member 'mins' of a type (line 802)
        mins_469261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 59), self_469260, 'mins')
        # Processing the call keyword arguments (line 802)
        kwargs_469262 = {}
        # Getting the type of 'Rectangle' (line 802)
        Rectangle_469257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 37), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 802)
        Rectangle_call_result_469263 = invoke(stypy.reporting.localization.Localization(__file__, 802, 37), Rectangle_469257, *[maxes_469259, mins_469261], **kwargs_469262)
        
        # Processing the call keyword arguments (line 801)
        kwargs_469264 = {}
        # Getting the type of 'traverse_checking' (line 801)
        traverse_checking_469245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 8), 'traverse_checking', False)
        # Calling traverse_checking(args, kwargs) (line 801)
        traverse_checking_call_result_469265 = invoke(stypy.reporting.localization.Localization(__file__, 801, 8), traverse_checking_469245, *[tree_469247, Rectangle_call_result_469254, tree_469256, Rectangle_call_result_469263], **kwargs_469264)
        
        # Getting the type of 'results' (line 803)
        results_469266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 15), 'results')
        # Assigning a type to the variable 'stypy_return_type' (line 803)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 8), 'stypy_return_type', results_469266)
        
        # ################# End of 'query_pairs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'query_pairs' in the type store
        # Getting the type of 'stypy_return_type' (line 698)
        stypy_return_type_469267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_469267)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'query_pairs'
        return stypy_return_type_469267


    @norecursion
    def count_neighbors(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_469268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 42), 'float')
        defaults = [float_469268]
        # Create a new context for function 'count_neighbors'
        module_type_store = module_type_store.open_function_context('count_neighbors', 805, 4, False)
        # Assigning a type to the variable 'self' (line 806)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KDTree.count_neighbors.__dict__.__setitem__('stypy_localization', localization)
        KDTree.count_neighbors.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KDTree.count_neighbors.__dict__.__setitem__('stypy_type_store', module_type_store)
        KDTree.count_neighbors.__dict__.__setitem__('stypy_function_name', 'KDTree.count_neighbors')
        KDTree.count_neighbors.__dict__.__setitem__('stypy_param_names_list', ['other', 'r', 'p'])
        KDTree.count_neighbors.__dict__.__setitem__('stypy_varargs_param_name', None)
        KDTree.count_neighbors.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KDTree.count_neighbors.__dict__.__setitem__('stypy_call_defaults', defaults)
        KDTree.count_neighbors.__dict__.__setitem__('stypy_call_varargs', varargs)
        KDTree.count_neighbors.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KDTree.count_neighbors.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KDTree.count_neighbors', ['other', 'r', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'count_neighbors', localization, ['other', 'r', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'count_neighbors(...)' code ##################

        str_469269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, (-1)), 'str', '\n        Count how many nearby pairs can be formed.\n\n        Count the number of pairs (x1,x2) can be formed, with x1 drawn\n        from self and x2 drawn from `other`, and where\n        ``distance(x1, x2, p) <= r``.\n        This is the "two-point correlation" described in Gray and Moore 2000,\n        "N-body problems in statistical learning", and the code here is based\n        on their algorithm.\n\n        Parameters\n        ----------\n        other : KDTree instance\n            The other tree to draw points from.\n        r : float or one-dimensional array of floats\n            The radius to produce a count for. Multiple radii are searched with\n            a single tree traversal.\n        p : float, 1<=p<=infinity, optional\n            Which Minkowski p-norm to use\n\n        Returns\n        -------\n        result : int or 1-D array of ints\n            The number of pairs. Note that this is internally stored in a numpy\n            int, and so may overflow if very large (2e9).\n\n        ')

        @norecursion
        def traverse(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'traverse'
            module_type_store = module_type_store.open_function_context('traverse', 833, 8, False)
            
            # Passed parameters checking function
            traverse.stypy_localization = localization
            traverse.stypy_type_of_self = None
            traverse.stypy_type_store = module_type_store
            traverse.stypy_function_name = 'traverse'
            traverse.stypy_param_names_list = ['node1', 'rect1', 'node2', 'rect2', 'idx']
            traverse.stypy_varargs_param_name = None
            traverse.stypy_kwargs_param_name = None
            traverse.stypy_call_defaults = defaults
            traverse.stypy_call_varargs = varargs
            traverse.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'traverse', ['node1', 'rect1', 'node2', 'rect2', 'idx'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'traverse', localization, ['node1', 'rect1', 'node2', 'rect2', 'idx'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'traverse(...)' code ##################

            
            # Assigning a Call to a Name (line 834):
            
            # Assigning a Call to a Name (line 834):
            
            # Call to min_distance_rectangle(...): (line 834)
            # Processing the call arguments (line 834)
            # Getting the type of 'rect2' (line 834)
            rect2_469272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 49), 'rect2', False)
            # Getting the type of 'p' (line 834)
            p_469273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 55), 'p', False)
            # Processing the call keyword arguments (line 834)
            kwargs_469274 = {}
            # Getting the type of 'rect1' (line 834)
            rect1_469270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 20), 'rect1', False)
            # Obtaining the member 'min_distance_rectangle' of a type (line 834)
            min_distance_rectangle_469271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 20), rect1_469270, 'min_distance_rectangle')
            # Calling min_distance_rectangle(args, kwargs) (line 834)
            min_distance_rectangle_call_result_469275 = invoke(stypy.reporting.localization.Localization(__file__, 834, 20), min_distance_rectangle_469271, *[rect2_469272, p_469273], **kwargs_469274)
            
            # Assigning a type to the variable 'min_r' (line 834)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 12), 'min_r', min_distance_rectangle_call_result_469275)
            
            # Assigning a Call to a Name (line 835):
            
            # Assigning a Call to a Name (line 835):
            
            # Call to max_distance_rectangle(...): (line 835)
            # Processing the call arguments (line 835)
            # Getting the type of 'rect2' (line 835)
            rect2_469278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 49), 'rect2', False)
            # Getting the type of 'p' (line 835)
            p_469279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 55), 'p', False)
            # Processing the call keyword arguments (line 835)
            kwargs_469280 = {}
            # Getting the type of 'rect1' (line 835)
            rect1_469276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 20), 'rect1', False)
            # Obtaining the member 'max_distance_rectangle' of a type (line 835)
            max_distance_rectangle_469277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 20), rect1_469276, 'max_distance_rectangle')
            # Calling max_distance_rectangle(args, kwargs) (line 835)
            max_distance_rectangle_call_result_469281 = invoke(stypy.reporting.localization.Localization(__file__, 835, 20), max_distance_rectangle_469277, *[rect2_469278, p_469279], **kwargs_469280)
            
            # Assigning a type to the variable 'max_r' (line 835)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 12), 'max_r', max_distance_rectangle_call_result_469281)
            
            # Assigning a Compare to a Name (line 836):
            
            # Assigning a Compare to a Name (line 836):
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'idx' (line 836)
            idx_469282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 26), 'idx')
            # Getting the type of 'r' (line 836)
            r_469283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 24), 'r')
            # Obtaining the member '__getitem__' of a type (line 836)
            getitem___469284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 836, 24), r_469283, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 836)
            subscript_call_result_469285 = invoke(stypy.reporting.localization.Localization(__file__, 836, 24), getitem___469284, idx_469282)
            
            # Getting the type of 'max_r' (line 836)
            max_r_469286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 33), 'max_r')
            # Applying the binary operator '>' (line 836)
            result_gt_469287 = python_operator(stypy.reporting.localization.Localization(__file__, 836, 24), '>', subscript_call_result_469285, max_r_469286)
            
            # Assigning a type to the variable 'c_greater' (line 836)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 12), 'c_greater', result_gt_469287)
            
            # Getting the type of 'result' (line 837)
            result_469288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 12), 'result')
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            # Getting the type of 'c_greater' (line 837)
            c_greater_469289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 23), 'c_greater')
            # Getting the type of 'idx' (line 837)
            idx_469290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 19), 'idx')
            # Obtaining the member '__getitem__' of a type (line 837)
            getitem___469291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 19), idx_469290, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 837)
            subscript_call_result_469292 = invoke(stypy.reporting.localization.Localization(__file__, 837, 19), getitem___469291, c_greater_469289)
            
            # Getting the type of 'result' (line 837)
            result_469293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 12), 'result')
            # Obtaining the member '__getitem__' of a type (line 837)
            getitem___469294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 12), result_469293, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 837)
            subscript_call_result_469295 = invoke(stypy.reporting.localization.Localization(__file__, 837, 12), getitem___469294, subscript_call_result_469292)
            
            # Getting the type of 'node1' (line 837)
            node1_469296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 38), 'node1')
            # Obtaining the member 'children' of a type (line 837)
            children_469297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 38), node1_469296, 'children')
            # Getting the type of 'node2' (line 837)
            node2_469298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 53), 'node2')
            # Obtaining the member 'children' of a type (line 837)
            children_469299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 53), node2_469298, 'children')
            # Applying the binary operator '*' (line 837)
            result_mul_469300 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 38), '*', children_469297, children_469299)
            
            # Applying the binary operator '+=' (line 837)
            result_iadd_469301 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 12), '+=', subscript_call_result_469295, result_mul_469300)
            # Getting the type of 'result' (line 837)
            result_469302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 12), 'result')
            
            # Obtaining the type of the subscript
            # Getting the type of 'c_greater' (line 837)
            c_greater_469303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 23), 'c_greater')
            # Getting the type of 'idx' (line 837)
            idx_469304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 19), 'idx')
            # Obtaining the member '__getitem__' of a type (line 837)
            getitem___469305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 19), idx_469304, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 837)
            subscript_call_result_469306 = invoke(stypy.reporting.localization.Localization(__file__, 837, 19), getitem___469305, c_greater_469303)
            
            # Storing an element on a container (line 837)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 837, 12), result_469302, (subscript_call_result_469306, result_iadd_469301))
            
            
            # Assigning a Subscript to a Name (line 838):
            
            # Assigning a Subscript to a Name (line 838):
            
            # Obtaining the type of the subscript
            
            # Getting the type of 'min_r' (line 838)
            min_r_469307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 23), 'min_r')
            
            # Obtaining the type of the subscript
            # Getting the type of 'idx' (line 838)
            idx_469308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 34), 'idx')
            # Getting the type of 'r' (line 838)
            r_469309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 32), 'r')
            # Obtaining the member '__getitem__' of a type (line 838)
            getitem___469310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 32), r_469309, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 838)
            subscript_call_result_469311 = invoke(stypy.reporting.localization.Localization(__file__, 838, 32), getitem___469310, idx_469308)
            
            # Applying the binary operator '<=' (line 838)
            result_le_469312 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 23), '<=', min_r_469307, subscript_call_result_469311)
            
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'idx' (line 838)
            idx_469313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 45), 'idx')
            # Getting the type of 'r' (line 838)
            r_469314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 43), 'r')
            # Obtaining the member '__getitem__' of a type (line 838)
            getitem___469315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 43), r_469314, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 838)
            subscript_call_result_469316 = invoke(stypy.reporting.localization.Localization(__file__, 838, 43), getitem___469315, idx_469313)
            
            # Getting the type of 'max_r' (line 838)
            max_r_469317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 53), 'max_r')
            # Applying the binary operator '<=' (line 838)
            result_le_469318 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 43), '<=', subscript_call_result_469316, max_r_469317)
            
            # Applying the binary operator '&' (line 838)
            result_and__469319 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 22), '&', result_le_469312, result_le_469318)
            
            # Getting the type of 'idx' (line 838)
            idx_469320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 18), 'idx')
            # Obtaining the member '__getitem__' of a type (line 838)
            getitem___469321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 18), idx_469320, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 838)
            subscript_call_result_469322 = invoke(stypy.reporting.localization.Localization(__file__, 838, 18), getitem___469321, result_and__469319)
            
            # Assigning a type to the variable 'idx' (line 838)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 12), 'idx', subscript_call_result_469322)
            
            
            
            # Call to len(...): (line 839)
            # Processing the call arguments (line 839)
            # Getting the type of 'idx' (line 839)
            idx_469324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 19), 'idx', False)
            # Processing the call keyword arguments (line 839)
            kwargs_469325 = {}
            # Getting the type of 'len' (line 839)
            len_469323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 15), 'len', False)
            # Calling len(args, kwargs) (line 839)
            len_call_result_469326 = invoke(stypy.reporting.localization.Localization(__file__, 839, 15), len_469323, *[idx_469324], **kwargs_469325)
            
            int_469327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 27), 'int')
            # Applying the binary operator '==' (line 839)
            result_eq_469328 = python_operator(stypy.reporting.localization.Localization(__file__, 839, 15), '==', len_call_result_469326, int_469327)
            
            # Testing the type of an if condition (line 839)
            if_condition_469329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 839, 12), result_eq_469328)
            # Assigning a type to the variable 'if_condition_469329' (line 839)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 12), 'if_condition_469329', if_condition_469329)
            # SSA begins for if statement (line 839)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 840)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 16), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 839)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Call to isinstance(...): (line 842)
            # Processing the call arguments (line 842)
            # Getting the type of 'node1' (line 842)
            node1_469331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 26), 'node1', False)
            # Getting the type of 'KDTree' (line 842)
            KDTree_469332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 32), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 842)
            leafnode_469333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 32), KDTree_469332, 'leafnode')
            # Processing the call keyword arguments (line 842)
            kwargs_469334 = {}
            # Getting the type of 'isinstance' (line 842)
            isinstance_469330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 842)
            isinstance_call_result_469335 = invoke(stypy.reporting.localization.Localization(__file__, 842, 15), isinstance_469330, *[node1_469331, leafnode_469333], **kwargs_469334)
            
            # Testing the type of an if condition (line 842)
            if_condition_469336 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 842, 12), isinstance_call_result_469335)
            # Assigning a type to the variable 'if_condition_469336' (line 842)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 12), 'if_condition_469336', if_condition_469336)
            # SSA begins for if statement (line 842)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to isinstance(...): (line 843)
            # Processing the call arguments (line 843)
            # Getting the type of 'node2' (line 843)
            node2_469338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 30), 'node2', False)
            # Getting the type of 'KDTree' (line 843)
            KDTree_469339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 36), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 843)
            leafnode_469340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 36), KDTree_469339, 'leafnode')
            # Processing the call keyword arguments (line 843)
            kwargs_469341 = {}
            # Getting the type of 'isinstance' (line 843)
            isinstance_469337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 843)
            isinstance_call_result_469342 = invoke(stypy.reporting.localization.Localization(__file__, 843, 19), isinstance_469337, *[node2_469338, leafnode_469340], **kwargs_469341)
            
            # Testing the type of an if condition (line 843)
            if_condition_469343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 843, 16), isinstance_call_result_469342)
            # Assigning a type to the variable 'if_condition_469343' (line 843)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 16), 'if_condition_469343', if_condition_469343)
            # SSA begins for if statement (line 843)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 844):
            
            # Assigning a Call to a Name (line 844):
            
            # Call to ravel(...): (line 844)
            # Processing the call keyword arguments (line 844)
            kwargs_469373 = {}
            
            # Call to minkowski_distance(...): (line 844)
            # Processing the call arguments (line 844)
            
            # Obtaining the type of the subscript
            slice_469345 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 844, 44), None, None, None)
            # Getting the type of 'np' (line 844)
            np_469346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 67), 'np', False)
            # Obtaining the member 'newaxis' of a type (line 844)
            newaxis_469347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 67), np_469346, 'newaxis')
            slice_469348 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 844, 44), None, None, None)
            
            # Obtaining the type of the subscript
            # Getting the type of 'node1' (line 844)
            node1_469349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 54), 'node1', False)
            # Obtaining the member 'idx' of a type (line 844)
            idx_469350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 54), node1_469349, 'idx')
            # Getting the type of 'self' (line 844)
            self_469351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 44), 'self', False)
            # Obtaining the member 'data' of a type (line 844)
            data_469352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 44), self_469351, 'data')
            # Obtaining the member '__getitem__' of a type (line 844)
            getitem___469353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 44), data_469352, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 844)
            subscript_call_result_469354 = invoke(stypy.reporting.localization.Localization(__file__, 844, 44), getitem___469353, idx_469350)
            
            # Obtaining the member '__getitem__' of a type (line 844)
            getitem___469355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 44), subscript_call_result_469354, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 844)
            subscript_call_result_469356 = invoke(stypy.reporting.localization.Localization(__file__, 844, 44), getitem___469355, (slice_469345, newaxis_469347, slice_469348))
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'np' (line 845)
            np_469357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 56), 'np', False)
            # Obtaining the member 'newaxis' of a type (line 845)
            newaxis_469358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 56), np_469357, 'newaxis')
            slice_469359 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 845, 34), None, None, None)
            slice_469360 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 845, 34), None, None, None)
            
            # Obtaining the type of the subscript
            # Getting the type of 'node2' (line 845)
            node2_469361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 45), 'node2', False)
            # Obtaining the member 'idx' of a type (line 845)
            idx_469362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 45), node2_469361, 'idx')
            # Getting the type of 'other' (line 845)
            other_469363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 34), 'other', False)
            # Obtaining the member 'data' of a type (line 845)
            data_469364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 34), other_469363, 'data')
            # Obtaining the member '__getitem__' of a type (line 845)
            getitem___469365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 34), data_469364, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 845)
            subscript_call_result_469366 = invoke(stypy.reporting.localization.Localization(__file__, 845, 34), getitem___469365, idx_469362)
            
            # Obtaining the member '__getitem__' of a type (line 845)
            getitem___469367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 34), subscript_call_result_469366, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 845)
            subscript_call_result_469368 = invoke(stypy.reporting.localization.Localization(__file__, 845, 34), getitem___469367, (newaxis_469358, slice_469359, slice_469360))
            
            # Getting the type of 'p' (line 846)
            p_469369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 34), 'p', False)
            # Processing the call keyword arguments (line 844)
            kwargs_469370 = {}
            # Getting the type of 'minkowski_distance' (line 844)
            minkowski_distance_469344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 25), 'minkowski_distance', False)
            # Calling minkowski_distance(args, kwargs) (line 844)
            minkowski_distance_call_result_469371 = invoke(stypy.reporting.localization.Localization(__file__, 844, 25), minkowski_distance_469344, *[subscript_call_result_469356, subscript_call_result_469368, p_469369], **kwargs_469370)
            
            # Obtaining the member 'ravel' of a type (line 844)
            ravel_469372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 25), minkowski_distance_call_result_469371, 'ravel')
            # Calling ravel(args, kwargs) (line 844)
            ravel_call_result_469374 = invoke(stypy.reporting.localization.Localization(__file__, 844, 25), ravel_469372, *[], **kwargs_469373)
            
            # Assigning a type to the variable 'ds' (line 844)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 20), 'ds', ravel_call_result_469374)
            
            # Call to sort(...): (line 847)
            # Processing the call keyword arguments (line 847)
            kwargs_469377 = {}
            # Getting the type of 'ds' (line 847)
            ds_469375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 20), 'ds', False)
            # Obtaining the member 'sort' of a type (line 847)
            sort_469376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 20), ds_469375, 'sort')
            # Calling sort(args, kwargs) (line 847)
            sort_call_result_469378 = invoke(stypy.reporting.localization.Localization(__file__, 847, 20), sort_469376, *[], **kwargs_469377)
            
            
            # Getting the type of 'result' (line 848)
            result_469379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 20), 'result')
            
            # Obtaining the type of the subscript
            # Getting the type of 'idx' (line 848)
            idx_469380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 27), 'idx')
            # Getting the type of 'result' (line 848)
            result_469381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 20), 'result')
            # Obtaining the member '__getitem__' of a type (line 848)
            getitem___469382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 20), result_469381, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 848)
            subscript_call_result_469383 = invoke(stypy.reporting.localization.Localization(__file__, 848, 20), getitem___469382, idx_469380)
            
            
            # Call to searchsorted(...): (line 848)
            # Processing the call arguments (line 848)
            # Getting the type of 'ds' (line 848)
            ds_469386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 51), 'ds', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'idx' (line 848)
            idx_469387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 56), 'idx', False)
            # Getting the type of 'r' (line 848)
            r_469388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 54), 'r', False)
            # Obtaining the member '__getitem__' of a type (line 848)
            getitem___469389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 54), r_469388, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 848)
            subscript_call_result_469390 = invoke(stypy.reporting.localization.Localization(__file__, 848, 54), getitem___469389, idx_469387)
            
            # Processing the call keyword arguments (line 848)
            str_469391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 66), 'str', 'right')
            keyword_469392 = str_469391
            kwargs_469393 = {'side': keyword_469392}
            # Getting the type of 'np' (line 848)
            np_469384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 35), 'np', False)
            # Obtaining the member 'searchsorted' of a type (line 848)
            searchsorted_469385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 35), np_469384, 'searchsorted')
            # Calling searchsorted(args, kwargs) (line 848)
            searchsorted_call_result_469394 = invoke(stypy.reporting.localization.Localization(__file__, 848, 35), searchsorted_469385, *[ds_469386, subscript_call_result_469390], **kwargs_469393)
            
            # Applying the binary operator '+=' (line 848)
            result_iadd_469395 = python_operator(stypy.reporting.localization.Localization(__file__, 848, 20), '+=', subscript_call_result_469383, searchsorted_call_result_469394)
            # Getting the type of 'result' (line 848)
            result_469396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 20), 'result')
            # Getting the type of 'idx' (line 848)
            idx_469397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 27), 'idx')
            # Storing an element on a container (line 848)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 848, 20), result_469396, (idx_469397, result_iadd_469395))
            
            # SSA branch for the else part of an if statement (line 843)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 850):
            
            # Assigning a Subscript to a Name (line 850):
            
            # Obtaining the type of the subscript
            int_469398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 20), 'int')
            
            # Call to split(...): (line 850)
            # Processing the call arguments (line 850)
            # Getting the type of 'node2' (line 850)
            node2_469401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 48), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 850)
            split_dim_469402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 48), node2_469401, 'split_dim')
            # Getting the type of 'node2' (line 850)
            node2_469403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 65), 'node2', False)
            # Obtaining the member 'split' of a type (line 850)
            split_469404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 65), node2_469403, 'split')
            # Processing the call keyword arguments (line 850)
            kwargs_469405 = {}
            # Getting the type of 'rect2' (line 850)
            rect2_469399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 36), 'rect2', False)
            # Obtaining the member 'split' of a type (line 850)
            split_469400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 36), rect2_469399, 'split')
            # Calling split(args, kwargs) (line 850)
            split_call_result_469406 = invoke(stypy.reporting.localization.Localization(__file__, 850, 36), split_469400, *[split_dim_469402, split_469404], **kwargs_469405)
            
            # Obtaining the member '__getitem__' of a type (line 850)
            getitem___469407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 20), split_call_result_469406, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 850)
            subscript_call_result_469408 = invoke(stypy.reporting.localization.Localization(__file__, 850, 20), getitem___469407, int_469398)
            
            # Assigning a type to the variable 'tuple_var_assignment_466794' (line 850)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 20), 'tuple_var_assignment_466794', subscript_call_result_469408)
            
            # Assigning a Subscript to a Name (line 850):
            
            # Obtaining the type of the subscript
            int_469409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 20), 'int')
            
            # Call to split(...): (line 850)
            # Processing the call arguments (line 850)
            # Getting the type of 'node2' (line 850)
            node2_469412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 48), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 850)
            split_dim_469413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 48), node2_469412, 'split_dim')
            # Getting the type of 'node2' (line 850)
            node2_469414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 65), 'node2', False)
            # Obtaining the member 'split' of a type (line 850)
            split_469415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 65), node2_469414, 'split')
            # Processing the call keyword arguments (line 850)
            kwargs_469416 = {}
            # Getting the type of 'rect2' (line 850)
            rect2_469410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 36), 'rect2', False)
            # Obtaining the member 'split' of a type (line 850)
            split_469411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 36), rect2_469410, 'split')
            # Calling split(args, kwargs) (line 850)
            split_call_result_469417 = invoke(stypy.reporting.localization.Localization(__file__, 850, 36), split_469411, *[split_dim_469413, split_469415], **kwargs_469416)
            
            # Obtaining the member '__getitem__' of a type (line 850)
            getitem___469418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 20), split_call_result_469417, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 850)
            subscript_call_result_469419 = invoke(stypy.reporting.localization.Localization(__file__, 850, 20), getitem___469418, int_469409)
            
            # Assigning a type to the variable 'tuple_var_assignment_466795' (line 850)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 20), 'tuple_var_assignment_466795', subscript_call_result_469419)
            
            # Assigning a Name to a Name (line 850):
            # Getting the type of 'tuple_var_assignment_466794' (line 850)
            tuple_var_assignment_466794_469420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 20), 'tuple_var_assignment_466794')
            # Assigning a type to the variable 'less' (line 850)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 20), 'less', tuple_var_assignment_466794_469420)
            
            # Assigning a Name to a Name (line 850):
            # Getting the type of 'tuple_var_assignment_466795' (line 850)
            tuple_var_assignment_466795_469421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 20), 'tuple_var_assignment_466795')
            # Assigning a type to the variable 'greater' (line 850)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 26), 'greater', tuple_var_assignment_466795_469421)
            
            # Call to traverse(...): (line 851)
            # Processing the call arguments (line 851)
            # Getting the type of 'node1' (line 851)
            node1_469423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 29), 'node1', False)
            # Getting the type of 'rect1' (line 851)
            rect1_469424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 36), 'rect1', False)
            # Getting the type of 'node2' (line 851)
            node2_469425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 43), 'node2', False)
            # Obtaining the member 'less' of a type (line 851)
            less_469426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 43), node2_469425, 'less')
            # Getting the type of 'less' (line 851)
            less_469427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 55), 'less', False)
            # Getting the type of 'idx' (line 851)
            idx_469428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 61), 'idx', False)
            # Processing the call keyword arguments (line 851)
            kwargs_469429 = {}
            # Getting the type of 'traverse' (line 851)
            traverse_469422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 20), 'traverse', False)
            # Calling traverse(args, kwargs) (line 851)
            traverse_call_result_469430 = invoke(stypy.reporting.localization.Localization(__file__, 851, 20), traverse_469422, *[node1_469423, rect1_469424, less_469426, less_469427, idx_469428], **kwargs_469429)
            
            
            # Call to traverse(...): (line 852)
            # Processing the call arguments (line 852)
            # Getting the type of 'node1' (line 852)
            node1_469432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 29), 'node1', False)
            # Getting the type of 'rect1' (line 852)
            rect1_469433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 36), 'rect1', False)
            # Getting the type of 'node2' (line 852)
            node2_469434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 43), 'node2', False)
            # Obtaining the member 'greater' of a type (line 852)
            greater_469435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 43), node2_469434, 'greater')
            # Getting the type of 'greater' (line 852)
            greater_469436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 58), 'greater', False)
            # Getting the type of 'idx' (line 852)
            idx_469437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 67), 'idx', False)
            # Processing the call keyword arguments (line 852)
            kwargs_469438 = {}
            # Getting the type of 'traverse' (line 852)
            traverse_469431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 20), 'traverse', False)
            # Calling traverse(args, kwargs) (line 852)
            traverse_call_result_469439 = invoke(stypy.reporting.localization.Localization(__file__, 852, 20), traverse_469431, *[node1_469432, rect1_469433, greater_469435, greater_469436, idx_469437], **kwargs_469438)
            
            # SSA join for if statement (line 843)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 842)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 854)
            # Processing the call arguments (line 854)
            # Getting the type of 'node2' (line 854)
            node2_469441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 30), 'node2', False)
            # Getting the type of 'KDTree' (line 854)
            KDTree_469442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 36), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 854)
            leafnode_469443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 854, 36), KDTree_469442, 'leafnode')
            # Processing the call keyword arguments (line 854)
            kwargs_469444 = {}
            # Getting the type of 'isinstance' (line 854)
            isinstance_469440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 854)
            isinstance_call_result_469445 = invoke(stypy.reporting.localization.Localization(__file__, 854, 19), isinstance_469440, *[node2_469441, leafnode_469443], **kwargs_469444)
            
            # Testing the type of an if condition (line 854)
            if_condition_469446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 854, 16), isinstance_call_result_469445)
            # Assigning a type to the variable 'if_condition_469446' (line 854)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 16), 'if_condition_469446', if_condition_469446)
            # SSA begins for if statement (line 854)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 855):
            
            # Assigning a Subscript to a Name (line 855):
            
            # Obtaining the type of the subscript
            int_469447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, 20), 'int')
            
            # Call to split(...): (line 855)
            # Processing the call arguments (line 855)
            # Getting the type of 'node1' (line 855)
            node1_469450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 48), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 855)
            split_dim_469451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 48), node1_469450, 'split_dim')
            # Getting the type of 'node1' (line 855)
            node1_469452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 65), 'node1', False)
            # Obtaining the member 'split' of a type (line 855)
            split_469453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 65), node1_469452, 'split')
            # Processing the call keyword arguments (line 855)
            kwargs_469454 = {}
            # Getting the type of 'rect1' (line 855)
            rect1_469448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 36), 'rect1', False)
            # Obtaining the member 'split' of a type (line 855)
            split_469449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 36), rect1_469448, 'split')
            # Calling split(args, kwargs) (line 855)
            split_call_result_469455 = invoke(stypy.reporting.localization.Localization(__file__, 855, 36), split_469449, *[split_dim_469451, split_469453], **kwargs_469454)
            
            # Obtaining the member '__getitem__' of a type (line 855)
            getitem___469456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 20), split_call_result_469455, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 855)
            subscript_call_result_469457 = invoke(stypy.reporting.localization.Localization(__file__, 855, 20), getitem___469456, int_469447)
            
            # Assigning a type to the variable 'tuple_var_assignment_466796' (line 855)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 20), 'tuple_var_assignment_466796', subscript_call_result_469457)
            
            # Assigning a Subscript to a Name (line 855):
            
            # Obtaining the type of the subscript
            int_469458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, 20), 'int')
            
            # Call to split(...): (line 855)
            # Processing the call arguments (line 855)
            # Getting the type of 'node1' (line 855)
            node1_469461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 48), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 855)
            split_dim_469462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 48), node1_469461, 'split_dim')
            # Getting the type of 'node1' (line 855)
            node1_469463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 65), 'node1', False)
            # Obtaining the member 'split' of a type (line 855)
            split_469464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 65), node1_469463, 'split')
            # Processing the call keyword arguments (line 855)
            kwargs_469465 = {}
            # Getting the type of 'rect1' (line 855)
            rect1_469459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 36), 'rect1', False)
            # Obtaining the member 'split' of a type (line 855)
            split_469460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 36), rect1_469459, 'split')
            # Calling split(args, kwargs) (line 855)
            split_call_result_469466 = invoke(stypy.reporting.localization.Localization(__file__, 855, 36), split_469460, *[split_dim_469462, split_469464], **kwargs_469465)
            
            # Obtaining the member '__getitem__' of a type (line 855)
            getitem___469467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 20), split_call_result_469466, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 855)
            subscript_call_result_469468 = invoke(stypy.reporting.localization.Localization(__file__, 855, 20), getitem___469467, int_469458)
            
            # Assigning a type to the variable 'tuple_var_assignment_466797' (line 855)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 20), 'tuple_var_assignment_466797', subscript_call_result_469468)
            
            # Assigning a Name to a Name (line 855):
            # Getting the type of 'tuple_var_assignment_466796' (line 855)
            tuple_var_assignment_466796_469469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 20), 'tuple_var_assignment_466796')
            # Assigning a type to the variable 'less' (line 855)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 20), 'less', tuple_var_assignment_466796_469469)
            
            # Assigning a Name to a Name (line 855):
            # Getting the type of 'tuple_var_assignment_466797' (line 855)
            tuple_var_assignment_466797_469470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 20), 'tuple_var_assignment_466797')
            # Assigning a type to the variable 'greater' (line 855)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 26), 'greater', tuple_var_assignment_466797_469470)
            
            # Call to traverse(...): (line 856)
            # Processing the call arguments (line 856)
            # Getting the type of 'node1' (line 856)
            node1_469472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 29), 'node1', False)
            # Obtaining the member 'less' of a type (line 856)
            less_469473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 29), node1_469472, 'less')
            # Getting the type of 'less' (line 856)
            less_469474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 41), 'less', False)
            # Getting the type of 'node2' (line 856)
            node2_469475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 47), 'node2', False)
            # Getting the type of 'rect2' (line 856)
            rect2_469476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 54), 'rect2', False)
            # Getting the type of 'idx' (line 856)
            idx_469477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 61), 'idx', False)
            # Processing the call keyword arguments (line 856)
            kwargs_469478 = {}
            # Getting the type of 'traverse' (line 856)
            traverse_469471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 20), 'traverse', False)
            # Calling traverse(args, kwargs) (line 856)
            traverse_call_result_469479 = invoke(stypy.reporting.localization.Localization(__file__, 856, 20), traverse_469471, *[less_469473, less_469474, node2_469475, rect2_469476, idx_469477], **kwargs_469478)
            
            
            # Call to traverse(...): (line 857)
            # Processing the call arguments (line 857)
            # Getting the type of 'node1' (line 857)
            node1_469481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 29), 'node1', False)
            # Obtaining the member 'greater' of a type (line 857)
            greater_469482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 857, 29), node1_469481, 'greater')
            # Getting the type of 'greater' (line 857)
            greater_469483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 44), 'greater', False)
            # Getting the type of 'node2' (line 857)
            node2_469484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 53), 'node2', False)
            # Getting the type of 'rect2' (line 857)
            rect2_469485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 60), 'rect2', False)
            # Getting the type of 'idx' (line 857)
            idx_469486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 67), 'idx', False)
            # Processing the call keyword arguments (line 857)
            kwargs_469487 = {}
            # Getting the type of 'traverse' (line 857)
            traverse_469480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 20), 'traverse', False)
            # Calling traverse(args, kwargs) (line 857)
            traverse_call_result_469488 = invoke(stypy.reporting.localization.Localization(__file__, 857, 20), traverse_469480, *[greater_469482, greater_469483, node2_469484, rect2_469485, idx_469486], **kwargs_469487)
            
            # SSA branch for the else part of an if statement (line 854)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 859):
            
            # Assigning a Subscript to a Name (line 859):
            
            # Obtaining the type of the subscript
            int_469489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 20), 'int')
            
            # Call to split(...): (line 859)
            # Processing the call arguments (line 859)
            # Getting the type of 'node1' (line 859)
            node1_469492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 50), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 859)
            split_dim_469493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 50), node1_469492, 'split_dim')
            # Getting the type of 'node1' (line 859)
            node1_469494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 67), 'node1', False)
            # Obtaining the member 'split' of a type (line 859)
            split_469495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 67), node1_469494, 'split')
            # Processing the call keyword arguments (line 859)
            kwargs_469496 = {}
            # Getting the type of 'rect1' (line 859)
            rect1_469490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 38), 'rect1', False)
            # Obtaining the member 'split' of a type (line 859)
            split_469491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 38), rect1_469490, 'split')
            # Calling split(args, kwargs) (line 859)
            split_call_result_469497 = invoke(stypy.reporting.localization.Localization(__file__, 859, 38), split_469491, *[split_dim_469493, split_469495], **kwargs_469496)
            
            # Obtaining the member '__getitem__' of a type (line 859)
            getitem___469498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 20), split_call_result_469497, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 859)
            subscript_call_result_469499 = invoke(stypy.reporting.localization.Localization(__file__, 859, 20), getitem___469498, int_469489)
            
            # Assigning a type to the variable 'tuple_var_assignment_466798' (line 859)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 20), 'tuple_var_assignment_466798', subscript_call_result_469499)
            
            # Assigning a Subscript to a Name (line 859):
            
            # Obtaining the type of the subscript
            int_469500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 20), 'int')
            
            # Call to split(...): (line 859)
            # Processing the call arguments (line 859)
            # Getting the type of 'node1' (line 859)
            node1_469503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 50), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 859)
            split_dim_469504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 50), node1_469503, 'split_dim')
            # Getting the type of 'node1' (line 859)
            node1_469505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 67), 'node1', False)
            # Obtaining the member 'split' of a type (line 859)
            split_469506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 67), node1_469505, 'split')
            # Processing the call keyword arguments (line 859)
            kwargs_469507 = {}
            # Getting the type of 'rect1' (line 859)
            rect1_469501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 38), 'rect1', False)
            # Obtaining the member 'split' of a type (line 859)
            split_469502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 38), rect1_469501, 'split')
            # Calling split(args, kwargs) (line 859)
            split_call_result_469508 = invoke(stypy.reporting.localization.Localization(__file__, 859, 38), split_469502, *[split_dim_469504, split_469506], **kwargs_469507)
            
            # Obtaining the member '__getitem__' of a type (line 859)
            getitem___469509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 20), split_call_result_469508, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 859)
            subscript_call_result_469510 = invoke(stypy.reporting.localization.Localization(__file__, 859, 20), getitem___469509, int_469500)
            
            # Assigning a type to the variable 'tuple_var_assignment_466799' (line 859)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 20), 'tuple_var_assignment_466799', subscript_call_result_469510)
            
            # Assigning a Name to a Name (line 859):
            # Getting the type of 'tuple_var_assignment_466798' (line 859)
            tuple_var_assignment_466798_469511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 20), 'tuple_var_assignment_466798')
            # Assigning a type to the variable 'less1' (line 859)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 20), 'less1', tuple_var_assignment_466798_469511)
            
            # Assigning a Name to a Name (line 859):
            # Getting the type of 'tuple_var_assignment_466799' (line 859)
            tuple_var_assignment_466799_469512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 20), 'tuple_var_assignment_466799')
            # Assigning a type to the variable 'greater1' (line 859)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 27), 'greater1', tuple_var_assignment_466799_469512)
            
            # Assigning a Call to a Tuple (line 860):
            
            # Assigning a Subscript to a Name (line 860):
            
            # Obtaining the type of the subscript
            int_469513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 20), 'int')
            
            # Call to split(...): (line 860)
            # Processing the call arguments (line 860)
            # Getting the type of 'node2' (line 860)
            node2_469516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 50), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 860)
            split_dim_469517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 50), node2_469516, 'split_dim')
            # Getting the type of 'node2' (line 860)
            node2_469518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 67), 'node2', False)
            # Obtaining the member 'split' of a type (line 860)
            split_469519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 67), node2_469518, 'split')
            # Processing the call keyword arguments (line 860)
            kwargs_469520 = {}
            # Getting the type of 'rect2' (line 860)
            rect2_469514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 38), 'rect2', False)
            # Obtaining the member 'split' of a type (line 860)
            split_469515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 38), rect2_469514, 'split')
            # Calling split(args, kwargs) (line 860)
            split_call_result_469521 = invoke(stypy.reporting.localization.Localization(__file__, 860, 38), split_469515, *[split_dim_469517, split_469519], **kwargs_469520)
            
            # Obtaining the member '__getitem__' of a type (line 860)
            getitem___469522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 20), split_call_result_469521, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 860)
            subscript_call_result_469523 = invoke(stypy.reporting.localization.Localization(__file__, 860, 20), getitem___469522, int_469513)
            
            # Assigning a type to the variable 'tuple_var_assignment_466800' (line 860)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 20), 'tuple_var_assignment_466800', subscript_call_result_469523)
            
            # Assigning a Subscript to a Name (line 860):
            
            # Obtaining the type of the subscript
            int_469524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 20), 'int')
            
            # Call to split(...): (line 860)
            # Processing the call arguments (line 860)
            # Getting the type of 'node2' (line 860)
            node2_469527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 50), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 860)
            split_dim_469528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 50), node2_469527, 'split_dim')
            # Getting the type of 'node2' (line 860)
            node2_469529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 67), 'node2', False)
            # Obtaining the member 'split' of a type (line 860)
            split_469530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 67), node2_469529, 'split')
            # Processing the call keyword arguments (line 860)
            kwargs_469531 = {}
            # Getting the type of 'rect2' (line 860)
            rect2_469525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 38), 'rect2', False)
            # Obtaining the member 'split' of a type (line 860)
            split_469526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 38), rect2_469525, 'split')
            # Calling split(args, kwargs) (line 860)
            split_call_result_469532 = invoke(stypy.reporting.localization.Localization(__file__, 860, 38), split_469526, *[split_dim_469528, split_469530], **kwargs_469531)
            
            # Obtaining the member '__getitem__' of a type (line 860)
            getitem___469533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 20), split_call_result_469532, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 860)
            subscript_call_result_469534 = invoke(stypy.reporting.localization.Localization(__file__, 860, 20), getitem___469533, int_469524)
            
            # Assigning a type to the variable 'tuple_var_assignment_466801' (line 860)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 20), 'tuple_var_assignment_466801', subscript_call_result_469534)
            
            # Assigning a Name to a Name (line 860):
            # Getting the type of 'tuple_var_assignment_466800' (line 860)
            tuple_var_assignment_466800_469535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 20), 'tuple_var_assignment_466800')
            # Assigning a type to the variable 'less2' (line 860)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 20), 'less2', tuple_var_assignment_466800_469535)
            
            # Assigning a Name to a Name (line 860):
            # Getting the type of 'tuple_var_assignment_466801' (line 860)
            tuple_var_assignment_466801_469536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 20), 'tuple_var_assignment_466801')
            # Assigning a type to the variable 'greater2' (line 860)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 27), 'greater2', tuple_var_assignment_466801_469536)
            
            # Call to traverse(...): (line 861)
            # Processing the call arguments (line 861)
            # Getting the type of 'node1' (line 861)
            node1_469538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 29), 'node1', False)
            # Obtaining the member 'less' of a type (line 861)
            less_469539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 29), node1_469538, 'less')
            # Getting the type of 'less1' (line 861)
            less1_469540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 40), 'less1', False)
            # Getting the type of 'node2' (line 861)
            node2_469541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 46), 'node2', False)
            # Obtaining the member 'less' of a type (line 861)
            less_469542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 46), node2_469541, 'less')
            # Getting the type of 'less2' (line 861)
            less2_469543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 57), 'less2', False)
            # Getting the type of 'idx' (line 861)
            idx_469544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 63), 'idx', False)
            # Processing the call keyword arguments (line 861)
            kwargs_469545 = {}
            # Getting the type of 'traverse' (line 861)
            traverse_469537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 20), 'traverse', False)
            # Calling traverse(args, kwargs) (line 861)
            traverse_call_result_469546 = invoke(stypy.reporting.localization.Localization(__file__, 861, 20), traverse_469537, *[less_469539, less1_469540, less_469542, less2_469543, idx_469544], **kwargs_469545)
            
            
            # Call to traverse(...): (line 862)
            # Processing the call arguments (line 862)
            # Getting the type of 'node1' (line 862)
            node1_469548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 29), 'node1', False)
            # Obtaining the member 'less' of a type (line 862)
            less_469549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 29), node1_469548, 'less')
            # Getting the type of 'less1' (line 862)
            less1_469550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 40), 'less1', False)
            # Getting the type of 'node2' (line 862)
            node2_469551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 46), 'node2', False)
            # Obtaining the member 'greater' of a type (line 862)
            greater_469552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 46), node2_469551, 'greater')
            # Getting the type of 'greater2' (line 862)
            greater2_469553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 60), 'greater2', False)
            # Getting the type of 'idx' (line 862)
            idx_469554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 69), 'idx', False)
            # Processing the call keyword arguments (line 862)
            kwargs_469555 = {}
            # Getting the type of 'traverse' (line 862)
            traverse_469547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 20), 'traverse', False)
            # Calling traverse(args, kwargs) (line 862)
            traverse_call_result_469556 = invoke(stypy.reporting.localization.Localization(__file__, 862, 20), traverse_469547, *[less_469549, less1_469550, greater_469552, greater2_469553, idx_469554], **kwargs_469555)
            
            
            # Call to traverse(...): (line 863)
            # Processing the call arguments (line 863)
            # Getting the type of 'node1' (line 863)
            node1_469558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 29), 'node1', False)
            # Obtaining the member 'greater' of a type (line 863)
            greater_469559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 863, 29), node1_469558, 'greater')
            # Getting the type of 'greater1' (line 863)
            greater1_469560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 43), 'greater1', False)
            # Getting the type of 'node2' (line 863)
            node2_469561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 52), 'node2', False)
            # Obtaining the member 'less' of a type (line 863)
            less_469562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 863, 52), node2_469561, 'less')
            # Getting the type of 'less2' (line 863)
            less2_469563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 63), 'less2', False)
            # Getting the type of 'idx' (line 863)
            idx_469564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 69), 'idx', False)
            # Processing the call keyword arguments (line 863)
            kwargs_469565 = {}
            # Getting the type of 'traverse' (line 863)
            traverse_469557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 20), 'traverse', False)
            # Calling traverse(args, kwargs) (line 863)
            traverse_call_result_469566 = invoke(stypy.reporting.localization.Localization(__file__, 863, 20), traverse_469557, *[greater_469559, greater1_469560, less_469562, less2_469563, idx_469564], **kwargs_469565)
            
            
            # Call to traverse(...): (line 864)
            # Processing the call arguments (line 864)
            # Getting the type of 'node1' (line 864)
            node1_469568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 29), 'node1', False)
            # Obtaining the member 'greater' of a type (line 864)
            greater_469569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 29), node1_469568, 'greater')
            # Getting the type of 'greater1' (line 864)
            greater1_469570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 43), 'greater1', False)
            # Getting the type of 'node2' (line 864)
            node2_469571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 52), 'node2', False)
            # Obtaining the member 'greater' of a type (line 864)
            greater_469572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 52), node2_469571, 'greater')
            # Getting the type of 'greater2' (line 864)
            greater2_469573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 66), 'greater2', False)
            # Getting the type of 'idx' (line 864)
            idx_469574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 75), 'idx', False)
            # Processing the call keyword arguments (line 864)
            kwargs_469575 = {}
            # Getting the type of 'traverse' (line 864)
            traverse_469567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 20), 'traverse', False)
            # Calling traverse(args, kwargs) (line 864)
            traverse_call_result_469576 = invoke(stypy.reporting.localization.Localization(__file__, 864, 20), traverse_469567, *[greater_469569, greater1_469570, greater_469572, greater2_469573, idx_469574], **kwargs_469575)
            
            # SSA join for if statement (line 854)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 842)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'traverse(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'traverse' in the type store
            # Getting the type of 'stypy_return_type' (line 833)
            stypy_return_type_469577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_469577)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'traverse'
            return stypy_return_type_469577

        # Assigning a type to the variable 'traverse' (line 833)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'traverse', traverse)
        
        # Assigning a Call to a Name (line 866):
        
        # Assigning a Call to a Name (line 866):
        
        # Call to Rectangle(...): (line 866)
        # Processing the call arguments (line 866)
        # Getting the type of 'self' (line 866)
        self_469579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 23), 'self', False)
        # Obtaining the member 'maxes' of a type (line 866)
        maxes_469580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 23), self_469579, 'maxes')
        # Getting the type of 'self' (line 866)
        self_469581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 35), 'self', False)
        # Obtaining the member 'mins' of a type (line 866)
        mins_469582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 35), self_469581, 'mins')
        # Processing the call keyword arguments (line 866)
        kwargs_469583 = {}
        # Getting the type of 'Rectangle' (line 866)
        Rectangle_469578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 13), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 866)
        Rectangle_call_result_469584 = invoke(stypy.reporting.localization.Localization(__file__, 866, 13), Rectangle_469578, *[maxes_469580, mins_469582], **kwargs_469583)
        
        # Assigning a type to the variable 'R1' (line 866)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 8), 'R1', Rectangle_call_result_469584)
        
        # Assigning a Call to a Name (line 867):
        
        # Assigning a Call to a Name (line 867):
        
        # Call to Rectangle(...): (line 867)
        # Processing the call arguments (line 867)
        # Getting the type of 'other' (line 867)
        other_469586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 23), 'other', False)
        # Obtaining the member 'maxes' of a type (line 867)
        maxes_469587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 23), other_469586, 'maxes')
        # Getting the type of 'other' (line 867)
        other_469588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 36), 'other', False)
        # Obtaining the member 'mins' of a type (line 867)
        mins_469589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 36), other_469588, 'mins')
        # Processing the call keyword arguments (line 867)
        kwargs_469590 = {}
        # Getting the type of 'Rectangle' (line 867)
        Rectangle_469585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 13), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 867)
        Rectangle_call_result_469591 = invoke(stypy.reporting.localization.Localization(__file__, 867, 13), Rectangle_469585, *[maxes_469587, mins_469589], **kwargs_469590)
        
        # Assigning a type to the variable 'R2' (line 867)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 8), 'R2', Rectangle_call_result_469591)
        
        
        
        # Call to shape(...): (line 868)
        # Processing the call arguments (line 868)
        # Getting the type of 'r' (line 868)
        r_469594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 20), 'r', False)
        # Processing the call keyword arguments (line 868)
        kwargs_469595 = {}
        # Getting the type of 'np' (line 868)
        np_469592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 11), 'np', False)
        # Obtaining the member 'shape' of a type (line 868)
        shape_469593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 11), np_469592, 'shape')
        # Calling shape(args, kwargs) (line 868)
        shape_call_result_469596 = invoke(stypy.reporting.localization.Localization(__file__, 868, 11), shape_469593, *[r_469594], **kwargs_469595)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 868)
        tuple_469597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 868)
        
        # Applying the binary operator '==' (line 868)
        result_eq_469598 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 11), '==', shape_call_result_469596, tuple_469597)
        
        # Testing the type of an if condition (line 868)
        if_condition_469599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 868, 8), result_eq_469598)
        # Assigning a type to the variable 'if_condition_469599' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'if_condition_469599', if_condition_469599)
        # SSA begins for if statement (line 868)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 869):
        
        # Assigning a Call to a Name (line 869):
        
        # Call to array(...): (line 869)
        # Processing the call arguments (line 869)
        
        # Obtaining an instance of the builtin type 'list' (line 869)
        list_469602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 869)
        # Adding element type (line 869)
        # Getting the type of 'r' (line 869)
        r_469603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 26), 'r', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 869, 25), list_469602, r_469603)
        
        # Processing the call keyword arguments (line 869)
        kwargs_469604 = {}
        # Getting the type of 'np' (line 869)
        np_469600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 869)
        array_469601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 16), np_469600, 'array')
        # Calling array(args, kwargs) (line 869)
        array_call_result_469605 = invoke(stypy.reporting.localization.Localization(__file__, 869, 16), array_469601, *[list_469602], **kwargs_469604)
        
        # Assigning a type to the variable 'r' (line 869)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 12), 'r', array_call_result_469605)
        
        # Assigning a Call to a Name (line 870):
        
        # Assigning a Call to a Name (line 870):
        
        # Call to zeros(...): (line 870)
        # Processing the call arguments (line 870)
        int_469608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 30), 'int')
        # Processing the call keyword arguments (line 870)
        # Getting the type of 'int' (line 870)
        int_469609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 38), 'int', False)
        keyword_469610 = int_469609
        kwargs_469611 = {'dtype': keyword_469610}
        # Getting the type of 'np' (line 870)
        np_469606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 21), 'np', False)
        # Obtaining the member 'zeros' of a type (line 870)
        zeros_469607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 21), np_469606, 'zeros')
        # Calling zeros(args, kwargs) (line 870)
        zeros_call_result_469612 = invoke(stypy.reporting.localization.Localization(__file__, 870, 21), zeros_469607, *[int_469608], **kwargs_469611)
        
        # Assigning a type to the variable 'result' (line 870)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 12), 'result', zeros_call_result_469612)
        
        # Call to traverse(...): (line 871)
        # Processing the call arguments (line 871)
        # Getting the type of 'self' (line 871)
        self_469614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 21), 'self', False)
        # Obtaining the member 'tree' of a type (line 871)
        tree_469615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 21), self_469614, 'tree')
        # Getting the type of 'R1' (line 871)
        R1_469616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 32), 'R1', False)
        # Getting the type of 'other' (line 871)
        other_469617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 36), 'other', False)
        # Obtaining the member 'tree' of a type (line 871)
        tree_469618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 36), other_469617, 'tree')
        # Getting the type of 'R2' (line 871)
        R2_469619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 48), 'R2', False)
        
        # Call to arange(...): (line 871)
        # Processing the call arguments (line 871)
        int_469622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 62), 'int')
        # Processing the call keyword arguments (line 871)
        kwargs_469623 = {}
        # Getting the type of 'np' (line 871)
        np_469620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 52), 'np', False)
        # Obtaining the member 'arange' of a type (line 871)
        arange_469621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 52), np_469620, 'arange')
        # Calling arange(args, kwargs) (line 871)
        arange_call_result_469624 = invoke(stypy.reporting.localization.Localization(__file__, 871, 52), arange_469621, *[int_469622], **kwargs_469623)
        
        # Processing the call keyword arguments (line 871)
        kwargs_469625 = {}
        # Getting the type of 'traverse' (line 871)
        traverse_469613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 12), 'traverse', False)
        # Calling traverse(args, kwargs) (line 871)
        traverse_call_result_469626 = invoke(stypy.reporting.localization.Localization(__file__, 871, 12), traverse_469613, *[tree_469615, R1_469616, tree_469618, R2_469619, arange_call_result_469624], **kwargs_469625)
        
        
        # Obtaining the type of the subscript
        int_469627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 26), 'int')
        # Getting the type of 'result' (line 872)
        result_469628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 19), 'result')
        # Obtaining the member '__getitem__' of a type (line 872)
        getitem___469629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 19), result_469628, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 872)
        subscript_call_result_469630 = invoke(stypy.reporting.localization.Localization(__file__, 872, 19), getitem___469629, int_469627)
        
        # Assigning a type to the variable 'stypy_return_type' (line 872)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'stypy_return_type', subscript_call_result_469630)
        # SSA branch for the else part of an if statement (line 868)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 873)
        # Processing the call arguments (line 873)
        
        # Call to shape(...): (line 873)
        # Processing the call arguments (line 873)
        # Getting the type of 'r' (line 873)
        r_469634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 26), 'r', False)
        # Processing the call keyword arguments (line 873)
        kwargs_469635 = {}
        # Getting the type of 'np' (line 873)
        np_469632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 17), 'np', False)
        # Obtaining the member 'shape' of a type (line 873)
        shape_469633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 873, 17), np_469632, 'shape')
        # Calling shape(args, kwargs) (line 873)
        shape_call_result_469636 = invoke(stypy.reporting.localization.Localization(__file__, 873, 17), shape_469633, *[r_469634], **kwargs_469635)
        
        # Processing the call keyword arguments (line 873)
        kwargs_469637 = {}
        # Getting the type of 'len' (line 873)
        len_469631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 13), 'len', False)
        # Calling len(args, kwargs) (line 873)
        len_call_result_469638 = invoke(stypy.reporting.localization.Localization(__file__, 873, 13), len_469631, *[shape_call_result_469636], **kwargs_469637)
        
        int_469639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 33), 'int')
        # Applying the binary operator '==' (line 873)
        result_eq_469640 = python_operator(stypy.reporting.localization.Localization(__file__, 873, 13), '==', len_call_result_469638, int_469639)
        
        # Testing the type of an if condition (line 873)
        if_condition_469641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 873, 13), result_eq_469640)
        # Assigning a type to the variable 'if_condition_469641' (line 873)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 13), 'if_condition_469641', if_condition_469641)
        # SSA begins for if statement (line 873)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 874):
        
        # Assigning a Call to a Name (line 874):
        
        # Call to asarray(...): (line 874)
        # Processing the call arguments (line 874)
        # Getting the type of 'r' (line 874)
        r_469644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 27), 'r', False)
        # Processing the call keyword arguments (line 874)
        kwargs_469645 = {}
        # Getting the type of 'np' (line 874)
        np_469642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 16), 'np', False)
        # Obtaining the member 'asarray' of a type (line 874)
        asarray_469643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 16), np_469642, 'asarray')
        # Calling asarray(args, kwargs) (line 874)
        asarray_call_result_469646 = invoke(stypy.reporting.localization.Localization(__file__, 874, 16), asarray_469643, *[r_469644], **kwargs_469645)
        
        # Assigning a type to the variable 'r' (line 874)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 874, 12), 'r', asarray_call_result_469646)
        
        # Assigning a Attribute to a Tuple (line 875):
        
        # Assigning a Subscript to a Name (line 875):
        
        # Obtaining the type of the subscript
        int_469647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 12), 'int')
        # Getting the type of 'r' (line 875)
        r_469648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 17), 'r')
        # Obtaining the member 'shape' of a type (line 875)
        shape_469649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 17), r_469648, 'shape')
        # Obtaining the member '__getitem__' of a type (line 875)
        getitem___469650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 12), shape_469649, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 875)
        subscript_call_result_469651 = invoke(stypy.reporting.localization.Localization(__file__, 875, 12), getitem___469650, int_469647)
        
        # Assigning a type to the variable 'tuple_var_assignment_466802' (line 875)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 12), 'tuple_var_assignment_466802', subscript_call_result_469651)
        
        # Assigning a Name to a Name (line 875):
        # Getting the type of 'tuple_var_assignment_466802' (line 875)
        tuple_var_assignment_466802_469652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 12), 'tuple_var_assignment_466802')
        # Assigning a type to the variable 'n' (line 875)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 12), 'n', tuple_var_assignment_466802_469652)
        
        # Assigning a Call to a Name (line 876):
        
        # Assigning a Call to a Name (line 876):
        
        # Call to zeros(...): (line 876)
        # Processing the call arguments (line 876)
        # Getting the type of 'n' (line 876)
        n_469655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 30), 'n', False)
        # Processing the call keyword arguments (line 876)
        # Getting the type of 'int' (line 876)
        int_469656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 38), 'int', False)
        keyword_469657 = int_469656
        kwargs_469658 = {'dtype': keyword_469657}
        # Getting the type of 'np' (line 876)
        np_469653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 21), 'np', False)
        # Obtaining the member 'zeros' of a type (line 876)
        zeros_469654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 21), np_469653, 'zeros')
        # Calling zeros(args, kwargs) (line 876)
        zeros_call_result_469659 = invoke(stypy.reporting.localization.Localization(__file__, 876, 21), zeros_469654, *[n_469655], **kwargs_469658)
        
        # Assigning a type to the variable 'result' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 12), 'result', zeros_call_result_469659)
        
        # Call to traverse(...): (line 877)
        # Processing the call arguments (line 877)
        # Getting the type of 'self' (line 877)
        self_469661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 21), 'self', False)
        # Obtaining the member 'tree' of a type (line 877)
        tree_469662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 21), self_469661, 'tree')
        # Getting the type of 'R1' (line 877)
        R1_469663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 32), 'R1', False)
        # Getting the type of 'other' (line 877)
        other_469664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 36), 'other', False)
        # Obtaining the member 'tree' of a type (line 877)
        tree_469665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 36), other_469664, 'tree')
        # Getting the type of 'R2' (line 877)
        R2_469666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 48), 'R2', False)
        
        # Call to arange(...): (line 877)
        # Processing the call arguments (line 877)
        # Getting the type of 'n' (line 877)
        n_469669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 62), 'n', False)
        # Processing the call keyword arguments (line 877)
        kwargs_469670 = {}
        # Getting the type of 'np' (line 877)
        np_469667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 52), 'np', False)
        # Obtaining the member 'arange' of a type (line 877)
        arange_469668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 52), np_469667, 'arange')
        # Calling arange(args, kwargs) (line 877)
        arange_call_result_469671 = invoke(stypy.reporting.localization.Localization(__file__, 877, 52), arange_469668, *[n_469669], **kwargs_469670)
        
        # Processing the call keyword arguments (line 877)
        kwargs_469672 = {}
        # Getting the type of 'traverse' (line 877)
        traverse_469660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 12), 'traverse', False)
        # Calling traverse(args, kwargs) (line 877)
        traverse_call_result_469673 = invoke(stypy.reporting.localization.Localization(__file__, 877, 12), traverse_469660, *[tree_469662, R1_469663, tree_469665, R2_469666, arange_call_result_469671], **kwargs_469672)
        
        # Getting the type of 'result' (line 878)
        result_469674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 19), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 878)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 12), 'stypy_return_type', result_469674)
        # SSA branch for the else part of an if statement (line 873)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 880)
        # Processing the call arguments (line 880)
        str_469676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 29), 'str', 'r must be either a single value or a one-dimensional array of values')
        # Processing the call keyword arguments (line 880)
        kwargs_469677 = {}
        # Getting the type of 'ValueError' (line 880)
        ValueError_469675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 880)
        ValueError_call_result_469678 = invoke(stypy.reporting.localization.Localization(__file__, 880, 18), ValueError_469675, *[str_469676], **kwargs_469677)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 880, 12), ValueError_call_result_469678, 'raise parameter', BaseException)
        # SSA join for if statement (line 873)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 868)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'count_neighbors(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'count_neighbors' in the type store
        # Getting the type of 'stypy_return_type' (line 805)
        stypy_return_type_469679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_469679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'count_neighbors'
        return stypy_return_type_469679


    @norecursion
    def sparse_distance_matrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_469680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 60), 'float')
        defaults = [float_469680]
        # Create a new context for function 'sparse_distance_matrix'
        module_type_store = module_type_store.open_function_context('sparse_distance_matrix', 882, 4, False)
        # Assigning a type to the variable 'self' (line 883)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 883, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        KDTree.sparse_distance_matrix.__dict__.__setitem__('stypy_localization', localization)
        KDTree.sparse_distance_matrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        KDTree.sparse_distance_matrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        KDTree.sparse_distance_matrix.__dict__.__setitem__('stypy_function_name', 'KDTree.sparse_distance_matrix')
        KDTree.sparse_distance_matrix.__dict__.__setitem__('stypy_param_names_list', ['other', 'max_distance', 'p'])
        KDTree.sparse_distance_matrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        KDTree.sparse_distance_matrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        KDTree.sparse_distance_matrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        KDTree.sparse_distance_matrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        KDTree.sparse_distance_matrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        KDTree.sparse_distance_matrix.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'KDTree.sparse_distance_matrix', ['other', 'max_distance', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sparse_distance_matrix', localization, ['other', 'max_distance', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sparse_distance_matrix(...)' code ##################

        str_469681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, (-1)), 'str', '\n        Compute a sparse distance matrix\n\n        Computes a distance matrix between two KDTrees, leaving as zero\n        any distance greater than max_distance.\n\n        Parameters\n        ----------\n        other : KDTree\n\n        max_distance : positive float\n\n        p : float, optional\n\n        Returns\n        -------\n        result : dok_matrix\n            Sparse matrix representing the results in "dictionary of keys" format.\n\n        ')
        
        # Assigning a Call to a Name (line 903):
        
        # Assigning a Call to a Name (line 903):
        
        # Call to dok_matrix(...): (line 903)
        # Processing the call arguments (line 903)
        
        # Obtaining an instance of the builtin type 'tuple' (line 903)
        tuple_469685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 903)
        # Adding element type (line 903)
        # Getting the type of 'self' (line 903)
        self_469686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 42), 'self', False)
        # Obtaining the member 'n' of a type (line 903)
        n_469687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 42), self_469686, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 903, 42), tuple_469685, n_469687)
        # Adding element type (line 903)
        # Getting the type of 'other' (line 903)
        other_469688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 49), 'other', False)
        # Obtaining the member 'n' of a type (line 903)
        n_469689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 49), other_469688, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 903, 42), tuple_469685, n_469689)
        
        # Processing the call keyword arguments (line 903)
        kwargs_469690 = {}
        # Getting the type of 'scipy' (line 903)
        scipy_469682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 17), 'scipy', False)
        # Obtaining the member 'sparse' of a type (line 903)
        sparse_469683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 17), scipy_469682, 'sparse')
        # Obtaining the member 'dok_matrix' of a type (line 903)
        dok_matrix_469684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 17), sparse_469683, 'dok_matrix')
        # Calling dok_matrix(args, kwargs) (line 903)
        dok_matrix_call_result_469691 = invoke(stypy.reporting.localization.Localization(__file__, 903, 17), dok_matrix_469684, *[tuple_469685], **kwargs_469690)
        
        # Assigning a type to the variable 'result' (line 903)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 8), 'result', dok_matrix_call_result_469691)

        @norecursion
        def traverse(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'traverse'
            module_type_store = module_type_store.open_function_context('traverse', 905, 8, False)
            
            # Passed parameters checking function
            traverse.stypy_localization = localization
            traverse.stypy_type_of_self = None
            traverse.stypy_type_store = module_type_store
            traverse.stypy_function_name = 'traverse'
            traverse.stypy_param_names_list = ['node1', 'rect1', 'node2', 'rect2']
            traverse.stypy_varargs_param_name = None
            traverse.stypy_kwargs_param_name = None
            traverse.stypy_call_defaults = defaults
            traverse.stypy_call_varargs = varargs
            traverse.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'traverse', ['node1', 'rect1', 'node2', 'rect2'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'traverse', localization, ['node1', 'rect1', 'node2', 'rect2'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'traverse(...)' code ##################

            
            
            
            # Call to min_distance_rectangle(...): (line 906)
            # Processing the call arguments (line 906)
            # Getting the type of 'rect2' (line 906)
            rect2_469694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 44), 'rect2', False)
            # Getting the type of 'p' (line 906)
            p_469695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 51), 'p', False)
            # Processing the call keyword arguments (line 906)
            kwargs_469696 = {}
            # Getting the type of 'rect1' (line 906)
            rect1_469692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 15), 'rect1', False)
            # Obtaining the member 'min_distance_rectangle' of a type (line 906)
            min_distance_rectangle_469693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 15), rect1_469692, 'min_distance_rectangle')
            # Calling min_distance_rectangle(args, kwargs) (line 906)
            min_distance_rectangle_call_result_469697 = invoke(stypy.reporting.localization.Localization(__file__, 906, 15), min_distance_rectangle_469693, *[rect2_469694, p_469695], **kwargs_469696)
            
            # Getting the type of 'max_distance' (line 906)
            max_distance_469698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 56), 'max_distance')
            # Applying the binary operator '>' (line 906)
            result_gt_469699 = python_operator(stypy.reporting.localization.Localization(__file__, 906, 15), '>', min_distance_rectangle_call_result_469697, max_distance_469698)
            
            # Testing the type of an if condition (line 906)
            if_condition_469700 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 906, 12), result_gt_469699)
            # Assigning a type to the variable 'if_condition_469700' (line 906)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 906, 12), 'if_condition_469700', if_condition_469700)
            # SSA begins for if statement (line 906)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 907)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 16), 'stypy_return_type', types.NoneType)
            # SSA branch for the else part of an if statement (line 906)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 908)
            # Processing the call arguments (line 908)
            # Getting the type of 'node1' (line 908)
            node1_469702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 28), 'node1', False)
            # Getting the type of 'KDTree' (line 908)
            KDTree_469703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 35), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 908)
            leafnode_469704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 35), KDTree_469703, 'leafnode')
            # Processing the call keyword arguments (line 908)
            kwargs_469705 = {}
            # Getting the type of 'isinstance' (line 908)
            isinstance_469701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 908)
            isinstance_call_result_469706 = invoke(stypy.reporting.localization.Localization(__file__, 908, 17), isinstance_469701, *[node1_469702, leafnode_469704], **kwargs_469705)
            
            # Testing the type of an if condition (line 908)
            if_condition_469707 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 908, 17), isinstance_call_result_469706)
            # Assigning a type to the variable 'if_condition_469707' (line 908)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 17), 'if_condition_469707', if_condition_469707)
            # SSA begins for if statement (line 908)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to isinstance(...): (line 909)
            # Processing the call arguments (line 909)
            # Getting the type of 'node2' (line 909)
            node2_469709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 30), 'node2', False)
            # Getting the type of 'KDTree' (line 909)
            KDTree_469710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 37), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 909)
            leafnode_469711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 37), KDTree_469710, 'leafnode')
            # Processing the call keyword arguments (line 909)
            kwargs_469712 = {}
            # Getting the type of 'isinstance' (line 909)
            isinstance_469708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 909)
            isinstance_call_result_469713 = invoke(stypy.reporting.localization.Localization(__file__, 909, 19), isinstance_469708, *[node2_469709, leafnode_469711], **kwargs_469712)
            
            # Testing the type of an if condition (line 909)
            if_condition_469714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 909, 16), isinstance_call_result_469713)
            # Assigning a type to the variable 'if_condition_469714' (line 909)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 16), 'if_condition_469714', if_condition_469714)
            # SSA begins for if statement (line 909)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'node1' (line 910)
            node1_469715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 29), 'node1')
            # Obtaining the member 'idx' of a type (line 910)
            idx_469716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 910, 29), node1_469715, 'idx')
            # Testing the type of a for loop iterable (line 910)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 910, 20), idx_469716)
            # Getting the type of the for loop variable (line 910)
            for_loop_var_469717 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 910, 20), idx_469716)
            # Assigning a type to the variable 'i' (line 910)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 20), 'i', for_loop_var_469717)
            # SSA begins for a for statement (line 910)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'node2' (line 911)
            node2_469718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 33), 'node2')
            # Obtaining the member 'idx' of a type (line 911)
            idx_469719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 911, 33), node2_469718, 'idx')
            # Testing the type of a for loop iterable (line 911)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 911, 24), idx_469719)
            # Getting the type of the for loop variable (line 911)
            for_loop_var_469720 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 911, 24), idx_469719)
            # Assigning a type to the variable 'j' (line 911)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 24), 'j', for_loop_var_469720)
            # SSA begins for a for statement (line 911)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 912):
            
            # Assigning a Call to a Name (line 912):
            
            # Call to minkowski_distance(...): (line 912)
            # Processing the call arguments (line 912)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 912)
            i_469722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 61), 'i', False)
            # Getting the type of 'self' (line 912)
            self_469723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 51), 'self', False)
            # Obtaining the member 'data' of a type (line 912)
            data_469724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 51), self_469723, 'data')
            # Obtaining the member '__getitem__' of a type (line 912)
            getitem___469725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 51), data_469724, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 912)
            subscript_call_result_469726 = invoke(stypy.reporting.localization.Localization(__file__, 912, 51), getitem___469725, i_469722)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 912)
            j_469727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 75), 'j', False)
            # Getting the type of 'other' (line 912)
            other_469728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 64), 'other', False)
            # Obtaining the member 'data' of a type (line 912)
            data_469729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 64), other_469728, 'data')
            # Obtaining the member '__getitem__' of a type (line 912)
            getitem___469730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 64), data_469729, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 912)
            subscript_call_result_469731 = invoke(stypy.reporting.localization.Localization(__file__, 912, 64), getitem___469730, j_469727)
            
            # Getting the type of 'p' (line 912)
            p_469732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 78), 'p', False)
            # Processing the call keyword arguments (line 912)
            kwargs_469733 = {}
            # Getting the type of 'minkowski_distance' (line 912)
            minkowski_distance_469721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 32), 'minkowski_distance', False)
            # Calling minkowski_distance(args, kwargs) (line 912)
            minkowski_distance_call_result_469734 = invoke(stypy.reporting.localization.Localization(__file__, 912, 32), minkowski_distance_469721, *[subscript_call_result_469726, subscript_call_result_469731, p_469732], **kwargs_469733)
            
            # Assigning a type to the variable 'd' (line 912)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 912, 28), 'd', minkowski_distance_call_result_469734)
            
            
            # Getting the type of 'd' (line 913)
            d_469735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 31), 'd')
            # Getting the type of 'max_distance' (line 913)
            max_distance_469736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 36), 'max_distance')
            # Applying the binary operator '<=' (line 913)
            result_le_469737 = python_operator(stypy.reporting.localization.Localization(__file__, 913, 31), '<=', d_469735, max_distance_469736)
            
            # Testing the type of an if condition (line 913)
            if_condition_469738 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 913, 28), result_le_469737)
            # Assigning a type to the variable 'if_condition_469738' (line 913)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 28), 'if_condition_469738', if_condition_469738)
            # SSA begins for if statement (line 913)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Subscript (line 914):
            
            # Assigning a Name to a Subscript (line 914):
            # Getting the type of 'd' (line 914)
            d_469739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 46), 'd')
            # Getting the type of 'result' (line 914)
            result_469740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 32), 'result')
            
            # Obtaining an instance of the builtin type 'tuple' (line 914)
            tuple_469741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 39), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 914)
            # Adding element type (line 914)
            # Getting the type of 'i' (line 914)
            i_469742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 39), 'i')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 914, 39), tuple_469741, i_469742)
            # Adding element type (line 914)
            # Getting the type of 'j' (line 914)
            j_469743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 41), 'j')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 914, 39), tuple_469741, j_469743)
            
            # Storing an element on a container (line 914)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 914, 32), result_469740, (tuple_469741, d_469739))
            # SSA join for if statement (line 913)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 909)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 916):
            
            # Assigning a Subscript to a Name (line 916):
            
            # Obtaining the type of the subscript
            int_469744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 20), 'int')
            
            # Call to split(...): (line 916)
            # Processing the call arguments (line 916)
            # Getting the type of 'node2' (line 916)
            node2_469747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 48), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 916)
            split_dim_469748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 48), node2_469747, 'split_dim')
            # Getting the type of 'node2' (line 916)
            node2_469749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 65), 'node2', False)
            # Obtaining the member 'split' of a type (line 916)
            split_469750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 65), node2_469749, 'split')
            # Processing the call keyword arguments (line 916)
            kwargs_469751 = {}
            # Getting the type of 'rect2' (line 916)
            rect2_469745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 36), 'rect2', False)
            # Obtaining the member 'split' of a type (line 916)
            split_469746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 36), rect2_469745, 'split')
            # Calling split(args, kwargs) (line 916)
            split_call_result_469752 = invoke(stypy.reporting.localization.Localization(__file__, 916, 36), split_469746, *[split_dim_469748, split_469750], **kwargs_469751)
            
            # Obtaining the member '__getitem__' of a type (line 916)
            getitem___469753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 20), split_call_result_469752, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 916)
            subscript_call_result_469754 = invoke(stypy.reporting.localization.Localization(__file__, 916, 20), getitem___469753, int_469744)
            
            # Assigning a type to the variable 'tuple_var_assignment_466803' (line 916)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 20), 'tuple_var_assignment_466803', subscript_call_result_469754)
            
            # Assigning a Subscript to a Name (line 916):
            
            # Obtaining the type of the subscript
            int_469755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 20), 'int')
            
            # Call to split(...): (line 916)
            # Processing the call arguments (line 916)
            # Getting the type of 'node2' (line 916)
            node2_469758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 48), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 916)
            split_dim_469759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 48), node2_469758, 'split_dim')
            # Getting the type of 'node2' (line 916)
            node2_469760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 65), 'node2', False)
            # Obtaining the member 'split' of a type (line 916)
            split_469761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 65), node2_469760, 'split')
            # Processing the call keyword arguments (line 916)
            kwargs_469762 = {}
            # Getting the type of 'rect2' (line 916)
            rect2_469756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 36), 'rect2', False)
            # Obtaining the member 'split' of a type (line 916)
            split_469757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 36), rect2_469756, 'split')
            # Calling split(args, kwargs) (line 916)
            split_call_result_469763 = invoke(stypy.reporting.localization.Localization(__file__, 916, 36), split_469757, *[split_dim_469759, split_469761], **kwargs_469762)
            
            # Obtaining the member '__getitem__' of a type (line 916)
            getitem___469764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 20), split_call_result_469763, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 916)
            subscript_call_result_469765 = invoke(stypy.reporting.localization.Localization(__file__, 916, 20), getitem___469764, int_469755)
            
            # Assigning a type to the variable 'tuple_var_assignment_466804' (line 916)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 20), 'tuple_var_assignment_466804', subscript_call_result_469765)
            
            # Assigning a Name to a Name (line 916):
            # Getting the type of 'tuple_var_assignment_466803' (line 916)
            tuple_var_assignment_466803_469766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 20), 'tuple_var_assignment_466803')
            # Assigning a type to the variable 'less' (line 916)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 20), 'less', tuple_var_assignment_466803_469766)
            
            # Assigning a Name to a Name (line 916):
            # Getting the type of 'tuple_var_assignment_466804' (line 916)
            tuple_var_assignment_466804_469767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 20), 'tuple_var_assignment_466804')
            # Assigning a type to the variable 'greater' (line 916)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 26), 'greater', tuple_var_assignment_466804_469767)
            
            # Call to traverse(...): (line 917)
            # Processing the call arguments (line 917)
            # Getting the type of 'node1' (line 917)
            node1_469769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 29), 'node1', False)
            # Getting the type of 'rect1' (line 917)
            rect1_469770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 35), 'rect1', False)
            # Getting the type of 'node2' (line 917)
            node2_469771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 41), 'node2', False)
            # Obtaining the member 'less' of a type (line 917)
            less_469772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 41), node2_469771, 'less')
            # Getting the type of 'less' (line 917)
            less_469773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 52), 'less', False)
            # Processing the call keyword arguments (line 917)
            kwargs_469774 = {}
            # Getting the type of 'traverse' (line 917)
            traverse_469768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 20), 'traverse', False)
            # Calling traverse(args, kwargs) (line 917)
            traverse_call_result_469775 = invoke(stypy.reporting.localization.Localization(__file__, 917, 20), traverse_469768, *[node1_469769, rect1_469770, less_469772, less_469773], **kwargs_469774)
            
            
            # Call to traverse(...): (line 918)
            # Processing the call arguments (line 918)
            # Getting the type of 'node1' (line 918)
            node1_469777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 29), 'node1', False)
            # Getting the type of 'rect1' (line 918)
            rect1_469778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 35), 'rect1', False)
            # Getting the type of 'node2' (line 918)
            node2_469779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 41), 'node2', False)
            # Obtaining the member 'greater' of a type (line 918)
            greater_469780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 41), node2_469779, 'greater')
            # Getting the type of 'greater' (line 918)
            greater_469781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 55), 'greater', False)
            # Processing the call keyword arguments (line 918)
            kwargs_469782 = {}
            # Getting the type of 'traverse' (line 918)
            traverse_469776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 20), 'traverse', False)
            # Calling traverse(args, kwargs) (line 918)
            traverse_call_result_469783 = invoke(stypy.reporting.localization.Localization(__file__, 918, 20), traverse_469776, *[node1_469777, rect1_469778, greater_469780, greater_469781], **kwargs_469782)
            
            # SSA join for if statement (line 909)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 908)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 919)
            # Processing the call arguments (line 919)
            # Getting the type of 'node2' (line 919)
            node2_469785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 28), 'node2', False)
            # Getting the type of 'KDTree' (line 919)
            KDTree_469786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 35), 'KDTree', False)
            # Obtaining the member 'leafnode' of a type (line 919)
            leafnode_469787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 35), KDTree_469786, 'leafnode')
            # Processing the call keyword arguments (line 919)
            kwargs_469788 = {}
            # Getting the type of 'isinstance' (line 919)
            isinstance_469784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 919)
            isinstance_call_result_469789 = invoke(stypy.reporting.localization.Localization(__file__, 919, 17), isinstance_469784, *[node2_469785, leafnode_469787], **kwargs_469788)
            
            # Testing the type of an if condition (line 919)
            if_condition_469790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 919, 17), isinstance_call_result_469789)
            # Assigning a type to the variable 'if_condition_469790' (line 919)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 17), 'if_condition_469790', if_condition_469790)
            # SSA begins for if statement (line 919)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 920):
            
            # Assigning a Subscript to a Name (line 920):
            
            # Obtaining the type of the subscript
            int_469791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 16), 'int')
            
            # Call to split(...): (line 920)
            # Processing the call arguments (line 920)
            # Getting the type of 'node1' (line 920)
            node1_469794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 44), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 920)
            split_dim_469795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 44), node1_469794, 'split_dim')
            # Getting the type of 'node1' (line 920)
            node1_469796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 61), 'node1', False)
            # Obtaining the member 'split' of a type (line 920)
            split_469797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 61), node1_469796, 'split')
            # Processing the call keyword arguments (line 920)
            kwargs_469798 = {}
            # Getting the type of 'rect1' (line 920)
            rect1_469792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 32), 'rect1', False)
            # Obtaining the member 'split' of a type (line 920)
            split_469793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 32), rect1_469792, 'split')
            # Calling split(args, kwargs) (line 920)
            split_call_result_469799 = invoke(stypy.reporting.localization.Localization(__file__, 920, 32), split_469793, *[split_dim_469795, split_469797], **kwargs_469798)
            
            # Obtaining the member '__getitem__' of a type (line 920)
            getitem___469800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 16), split_call_result_469799, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 920)
            subscript_call_result_469801 = invoke(stypy.reporting.localization.Localization(__file__, 920, 16), getitem___469800, int_469791)
            
            # Assigning a type to the variable 'tuple_var_assignment_466805' (line 920)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 16), 'tuple_var_assignment_466805', subscript_call_result_469801)
            
            # Assigning a Subscript to a Name (line 920):
            
            # Obtaining the type of the subscript
            int_469802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 16), 'int')
            
            # Call to split(...): (line 920)
            # Processing the call arguments (line 920)
            # Getting the type of 'node1' (line 920)
            node1_469805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 44), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 920)
            split_dim_469806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 44), node1_469805, 'split_dim')
            # Getting the type of 'node1' (line 920)
            node1_469807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 61), 'node1', False)
            # Obtaining the member 'split' of a type (line 920)
            split_469808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 61), node1_469807, 'split')
            # Processing the call keyword arguments (line 920)
            kwargs_469809 = {}
            # Getting the type of 'rect1' (line 920)
            rect1_469803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 32), 'rect1', False)
            # Obtaining the member 'split' of a type (line 920)
            split_469804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 32), rect1_469803, 'split')
            # Calling split(args, kwargs) (line 920)
            split_call_result_469810 = invoke(stypy.reporting.localization.Localization(__file__, 920, 32), split_469804, *[split_dim_469806, split_469808], **kwargs_469809)
            
            # Obtaining the member '__getitem__' of a type (line 920)
            getitem___469811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 16), split_call_result_469810, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 920)
            subscript_call_result_469812 = invoke(stypy.reporting.localization.Localization(__file__, 920, 16), getitem___469811, int_469802)
            
            # Assigning a type to the variable 'tuple_var_assignment_466806' (line 920)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 16), 'tuple_var_assignment_466806', subscript_call_result_469812)
            
            # Assigning a Name to a Name (line 920):
            # Getting the type of 'tuple_var_assignment_466805' (line 920)
            tuple_var_assignment_466805_469813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 16), 'tuple_var_assignment_466805')
            # Assigning a type to the variable 'less' (line 920)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 16), 'less', tuple_var_assignment_466805_469813)
            
            # Assigning a Name to a Name (line 920):
            # Getting the type of 'tuple_var_assignment_466806' (line 920)
            tuple_var_assignment_466806_469814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 16), 'tuple_var_assignment_466806')
            # Assigning a type to the variable 'greater' (line 920)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 22), 'greater', tuple_var_assignment_466806_469814)
            
            # Call to traverse(...): (line 921)
            # Processing the call arguments (line 921)
            # Getting the type of 'node1' (line 921)
            node1_469816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 25), 'node1', False)
            # Obtaining the member 'less' of a type (line 921)
            less_469817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 25), node1_469816, 'less')
            # Getting the type of 'less' (line 921)
            less_469818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 36), 'less', False)
            # Getting the type of 'node2' (line 921)
            node2_469819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 41), 'node2', False)
            # Getting the type of 'rect2' (line 921)
            rect2_469820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 47), 'rect2', False)
            # Processing the call keyword arguments (line 921)
            kwargs_469821 = {}
            # Getting the type of 'traverse' (line 921)
            traverse_469815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 16), 'traverse', False)
            # Calling traverse(args, kwargs) (line 921)
            traverse_call_result_469822 = invoke(stypy.reporting.localization.Localization(__file__, 921, 16), traverse_469815, *[less_469817, less_469818, node2_469819, rect2_469820], **kwargs_469821)
            
            
            # Call to traverse(...): (line 922)
            # Processing the call arguments (line 922)
            # Getting the type of 'node1' (line 922)
            node1_469824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 25), 'node1', False)
            # Obtaining the member 'greater' of a type (line 922)
            greater_469825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 25), node1_469824, 'greater')
            # Getting the type of 'greater' (line 922)
            greater_469826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 39), 'greater', False)
            # Getting the type of 'node2' (line 922)
            node2_469827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 47), 'node2', False)
            # Getting the type of 'rect2' (line 922)
            rect2_469828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 53), 'rect2', False)
            # Processing the call keyword arguments (line 922)
            kwargs_469829 = {}
            # Getting the type of 'traverse' (line 922)
            traverse_469823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 16), 'traverse', False)
            # Calling traverse(args, kwargs) (line 922)
            traverse_call_result_469830 = invoke(stypy.reporting.localization.Localization(__file__, 922, 16), traverse_469823, *[greater_469825, greater_469826, node2_469827, rect2_469828], **kwargs_469829)
            
            # SSA branch for the else part of an if statement (line 919)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 924):
            
            # Assigning a Subscript to a Name (line 924):
            
            # Obtaining the type of the subscript
            int_469831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 16), 'int')
            
            # Call to split(...): (line 924)
            # Processing the call arguments (line 924)
            # Getting the type of 'node1' (line 924)
            node1_469834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 46), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 924)
            split_dim_469835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 46), node1_469834, 'split_dim')
            # Getting the type of 'node1' (line 924)
            node1_469836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 63), 'node1', False)
            # Obtaining the member 'split' of a type (line 924)
            split_469837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 63), node1_469836, 'split')
            # Processing the call keyword arguments (line 924)
            kwargs_469838 = {}
            # Getting the type of 'rect1' (line 924)
            rect1_469832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 34), 'rect1', False)
            # Obtaining the member 'split' of a type (line 924)
            split_469833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 34), rect1_469832, 'split')
            # Calling split(args, kwargs) (line 924)
            split_call_result_469839 = invoke(stypy.reporting.localization.Localization(__file__, 924, 34), split_469833, *[split_dim_469835, split_469837], **kwargs_469838)
            
            # Obtaining the member '__getitem__' of a type (line 924)
            getitem___469840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 16), split_call_result_469839, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 924)
            subscript_call_result_469841 = invoke(stypy.reporting.localization.Localization(__file__, 924, 16), getitem___469840, int_469831)
            
            # Assigning a type to the variable 'tuple_var_assignment_466807' (line 924)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 16), 'tuple_var_assignment_466807', subscript_call_result_469841)
            
            # Assigning a Subscript to a Name (line 924):
            
            # Obtaining the type of the subscript
            int_469842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 16), 'int')
            
            # Call to split(...): (line 924)
            # Processing the call arguments (line 924)
            # Getting the type of 'node1' (line 924)
            node1_469845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 46), 'node1', False)
            # Obtaining the member 'split_dim' of a type (line 924)
            split_dim_469846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 46), node1_469845, 'split_dim')
            # Getting the type of 'node1' (line 924)
            node1_469847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 63), 'node1', False)
            # Obtaining the member 'split' of a type (line 924)
            split_469848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 63), node1_469847, 'split')
            # Processing the call keyword arguments (line 924)
            kwargs_469849 = {}
            # Getting the type of 'rect1' (line 924)
            rect1_469843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 34), 'rect1', False)
            # Obtaining the member 'split' of a type (line 924)
            split_469844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 34), rect1_469843, 'split')
            # Calling split(args, kwargs) (line 924)
            split_call_result_469850 = invoke(stypy.reporting.localization.Localization(__file__, 924, 34), split_469844, *[split_dim_469846, split_469848], **kwargs_469849)
            
            # Obtaining the member '__getitem__' of a type (line 924)
            getitem___469851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 16), split_call_result_469850, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 924)
            subscript_call_result_469852 = invoke(stypy.reporting.localization.Localization(__file__, 924, 16), getitem___469851, int_469842)
            
            # Assigning a type to the variable 'tuple_var_assignment_466808' (line 924)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 16), 'tuple_var_assignment_466808', subscript_call_result_469852)
            
            # Assigning a Name to a Name (line 924):
            # Getting the type of 'tuple_var_assignment_466807' (line 924)
            tuple_var_assignment_466807_469853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 16), 'tuple_var_assignment_466807')
            # Assigning a type to the variable 'less1' (line 924)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 16), 'less1', tuple_var_assignment_466807_469853)
            
            # Assigning a Name to a Name (line 924):
            # Getting the type of 'tuple_var_assignment_466808' (line 924)
            tuple_var_assignment_466808_469854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 16), 'tuple_var_assignment_466808')
            # Assigning a type to the variable 'greater1' (line 924)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 23), 'greater1', tuple_var_assignment_466808_469854)
            
            # Assigning a Call to a Tuple (line 925):
            
            # Assigning a Subscript to a Name (line 925):
            
            # Obtaining the type of the subscript
            int_469855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 16), 'int')
            
            # Call to split(...): (line 925)
            # Processing the call arguments (line 925)
            # Getting the type of 'node2' (line 925)
            node2_469858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 46), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 925)
            split_dim_469859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 46), node2_469858, 'split_dim')
            # Getting the type of 'node2' (line 925)
            node2_469860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 63), 'node2', False)
            # Obtaining the member 'split' of a type (line 925)
            split_469861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 63), node2_469860, 'split')
            # Processing the call keyword arguments (line 925)
            kwargs_469862 = {}
            # Getting the type of 'rect2' (line 925)
            rect2_469856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 34), 'rect2', False)
            # Obtaining the member 'split' of a type (line 925)
            split_469857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 34), rect2_469856, 'split')
            # Calling split(args, kwargs) (line 925)
            split_call_result_469863 = invoke(stypy.reporting.localization.Localization(__file__, 925, 34), split_469857, *[split_dim_469859, split_469861], **kwargs_469862)
            
            # Obtaining the member '__getitem__' of a type (line 925)
            getitem___469864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 16), split_call_result_469863, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 925)
            subscript_call_result_469865 = invoke(stypy.reporting.localization.Localization(__file__, 925, 16), getitem___469864, int_469855)
            
            # Assigning a type to the variable 'tuple_var_assignment_466809' (line 925)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 16), 'tuple_var_assignment_466809', subscript_call_result_469865)
            
            # Assigning a Subscript to a Name (line 925):
            
            # Obtaining the type of the subscript
            int_469866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 16), 'int')
            
            # Call to split(...): (line 925)
            # Processing the call arguments (line 925)
            # Getting the type of 'node2' (line 925)
            node2_469869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 46), 'node2', False)
            # Obtaining the member 'split_dim' of a type (line 925)
            split_dim_469870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 46), node2_469869, 'split_dim')
            # Getting the type of 'node2' (line 925)
            node2_469871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 63), 'node2', False)
            # Obtaining the member 'split' of a type (line 925)
            split_469872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 63), node2_469871, 'split')
            # Processing the call keyword arguments (line 925)
            kwargs_469873 = {}
            # Getting the type of 'rect2' (line 925)
            rect2_469867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 34), 'rect2', False)
            # Obtaining the member 'split' of a type (line 925)
            split_469868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 34), rect2_469867, 'split')
            # Calling split(args, kwargs) (line 925)
            split_call_result_469874 = invoke(stypy.reporting.localization.Localization(__file__, 925, 34), split_469868, *[split_dim_469870, split_469872], **kwargs_469873)
            
            # Obtaining the member '__getitem__' of a type (line 925)
            getitem___469875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 16), split_call_result_469874, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 925)
            subscript_call_result_469876 = invoke(stypy.reporting.localization.Localization(__file__, 925, 16), getitem___469875, int_469866)
            
            # Assigning a type to the variable 'tuple_var_assignment_466810' (line 925)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 16), 'tuple_var_assignment_466810', subscript_call_result_469876)
            
            # Assigning a Name to a Name (line 925):
            # Getting the type of 'tuple_var_assignment_466809' (line 925)
            tuple_var_assignment_466809_469877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 16), 'tuple_var_assignment_466809')
            # Assigning a type to the variable 'less2' (line 925)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 16), 'less2', tuple_var_assignment_466809_469877)
            
            # Assigning a Name to a Name (line 925):
            # Getting the type of 'tuple_var_assignment_466810' (line 925)
            tuple_var_assignment_466810_469878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 16), 'tuple_var_assignment_466810')
            # Assigning a type to the variable 'greater2' (line 925)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 23), 'greater2', tuple_var_assignment_466810_469878)
            
            # Call to traverse(...): (line 926)
            # Processing the call arguments (line 926)
            # Getting the type of 'node1' (line 926)
            node1_469880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 25), 'node1', False)
            # Obtaining the member 'less' of a type (line 926)
            less_469881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 25), node1_469880, 'less')
            # Getting the type of 'less1' (line 926)
            less1_469882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 36), 'less1', False)
            # Getting the type of 'node2' (line 926)
            node2_469883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 42), 'node2', False)
            # Obtaining the member 'less' of a type (line 926)
            less_469884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 42), node2_469883, 'less')
            # Getting the type of 'less2' (line 926)
            less2_469885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 53), 'less2', False)
            # Processing the call keyword arguments (line 926)
            kwargs_469886 = {}
            # Getting the type of 'traverse' (line 926)
            traverse_469879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 16), 'traverse', False)
            # Calling traverse(args, kwargs) (line 926)
            traverse_call_result_469887 = invoke(stypy.reporting.localization.Localization(__file__, 926, 16), traverse_469879, *[less_469881, less1_469882, less_469884, less2_469885], **kwargs_469886)
            
            
            # Call to traverse(...): (line 927)
            # Processing the call arguments (line 927)
            # Getting the type of 'node1' (line 927)
            node1_469889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 25), 'node1', False)
            # Obtaining the member 'less' of a type (line 927)
            less_469890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 25), node1_469889, 'less')
            # Getting the type of 'less1' (line 927)
            less1_469891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 36), 'less1', False)
            # Getting the type of 'node2' (line 927)
            node2_469892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 42), 'node2', False)
            # Obtaining the member 'greater' of a type (line 927)
            greater_469893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 42), node2_469892, 'greater')
            # Getting the type of 'greater2' (line 927)
            greater2_469894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 56), 'greater2', False)
            # Processing the call keyword arguments (line 927)
            kwargs_469895 = {}
            # Getting the type of 'traverse' (line 927)
            traverse_469888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 16), 'traverse', False)
            # Calling traverse(args, kwargs) (line 927)
            traverse_call_result_469896 = invoke(stypy.reporting.localization.Localization(__file__, 927, 16), traverse_469888, *[less_469890, less1_469891, greater_469893, greater2_469894], **kwargs_469895)
            
            
            # Call to traverse(...): (line 928)
            # Processing the call arguments (line 928)
            # Getting the type of 'node1' (line 928)
            node1_469898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 25), 'node1', False)
            # Obtaining the member 'greater' of a type (line 928)
            greater_469899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 25), node1_469898, 'greater')
            # Getting the type of 'greater1' (line 928)
            greater1_469900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 39), 'greater1', False)
            # Getting the type of 'node2' (line 928)
            node2_469901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 48), 'node2', False)
            # Obtaining the member 'less' of a type (line 928)
            less_469902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 48), node2_469901, 'less')
            # Getting the type of 'less2' (line 928)
            less2_469903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 59), 'less2', False)
            # Processing the call keyword arguments (line 928)
            kwargs_469904 = {}
            # Getting the type of 'traverse' (line 928)
            traverse_469897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 16), 'traverse', False)
            # Calling traverse(args, kwargs) (line 928)
            traverse_call_result_469905 = invoke(stypy.reporting.localization.Localization(__file__, 928, 16), traverse_469897, *[greater_469899, greater1_469900, less_469902, less2_469903], **kwargs_469904)
            
            
            # Call to traverse(...): (line 929)
            # Processing the call arguments (line 929)
            # Getting the type of 'node1' (line 929)
            node1_469907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 25), 'node1', False)
            # Obtaining the member 'greater' of a type (line 929)
            greater_469908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 25), node1_469907, 'greater')
            # Getting the type of 'greater1' (line 929)
            greater1_469909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 39), 'greater1', False)
            # Getting the type of 'node2' (line 929)
            node2_469910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 48), 'node2', False)
            # Obtaining the member 'greater' of a type (line 929)
            greater_469911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 48), node2_469910, 'greater')
            # Getting the type of 'greater2' (line 929)
            greater2_469912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 62), 'greater2', False)
            # Processing the call keyword arguments (line 929)
            kwargs_469913 = {}
            # Getting the type of 'traverse' (line 929)
            traverse_469906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 16), 'traverse', False)
            # Calling traverse(args, kwargs) (line 929)
            traverse_call_result_469914 = invoke(stypy.reporting.localization.Localization(__file__, 929, 16), traverse_469906, *[greater_469908, greater1_469909, greater_469911, greater2_469912], **kwargs_469913)
            
            # SSA join for if statement (line 919)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 908)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 906)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'traverse(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'traverse' in the type store
            # Getting the type of 'stypy_return_type' (line 905)
            stypy_return_type_469915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_469915)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'traverse'
            return stypy_return_type_469915

        # Assigning a type to the variable 'traverse' (line 905)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 8), 'traverse', traverse)
        
        # Call to traverse(...): (line 930)
        # Processing the call arguments (line 930)
        # Getting the type of 'self' (line 930)
        self_469917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 17), 'self', False)
        # Obtaining the member 'tree' of a type (line 930)
        tree_469918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 17), self_469917, 'tree')
        
        # Call to Rectangle(...): (line 930)
        # Processing the call arguments (line 930)
        # Getting the type of 'self' (line 930)
        self_469920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 38), 'self', False)
        # Obtaining the member 'maxes' of a type (line 930)
        maxes_469921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 38), self_469920, 'maxes')
        # Getting the type of 'self' (line 930)
        self_469922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 50), 'self', False)
        # Obtaining the member 'mins' of a type (line 930)
        mins_469923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 50), self_469922, 'mins')
        # Processing the call keyword arguments (line 930)
        kwargs_469924 = {}
        # Getting the type of 'Rectangle' (line 930)
        Rectangle_469919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 28), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 930)
        Rectangle_call_result_469925 = invoke(stypy.reporting.localization.Localization(__file__, 930, 28), Rectangle_469919, *[maxes_469921, mins_469923], **kwargs_469924)
        
        # Getting the type of 'other' (line 931)
        other_469926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 17), 'other', False)
        # Obtaining the member 'tree' of a type (line 931)
        tree_469927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 17), other_469926, 'tree')
        
        # Call to Rectangle(...): (line 931)
        # Processing the call arguments (line 931)
        # Getting the type of 'other' (line 931)
        other_469929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 39), 'other', False)
        # Obtaining the member 'maxes' of a type (line 931)
        maxes_469930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 39), other_469929, 'maxes')
        # Getting the type of 'other' (line 931)
        other_469931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 52), 'other', False)
        # Obtaining the member 'mins' of a type (line 931)
        mins_469932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 52), other_469931, 'mins')
        # Processing the call keyword arguments (line 931)
        kwargs_469933 = {}
        # Getting the type of 'Rectangle' (line 931)
        Rectangle_469928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 29), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 931)
        Rectangle_call_result_469934 = invoke(stypy.reporting.localization.Localization(__file__, 931, 29), Rectangle_469928, *[maxes_469930, mins_469932], **kwargs_469933)
        
        # Processing the call keyword arguments (line 930)
        kwargs_469935 = {}
        # Getting the type of 'traverse' (line 930)
        traverse_469916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 8), 'traverse', False)
        # Calling traverse(args, kwargs) (line 930)
        traverse_call_result_469936 = invoke(stypy.reporting.localization.Localization(__file__, 930, 8), traverse_469916, *[tree_469918, Rectangle_call_result_469925, tree_469927, Rectangle_call_result_469934], **kwargs_469935)
        
        # Getting the type of 'result' (line 933)
        result_469937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 933)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 8), 'stypy_return_type', result_469937)
        
        # ################# End of 'sparse_distance_matrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sparse_distance_matrix' in the type store
        # Getting the type of 'stypy_return_type' (line 882)
        stypy_return_type_469938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_469938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sparse_distance_matrix'
        return stypy_return_type_469938


# Assigning a type to the variable 'KDTree' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'KDTree', KDTree)

@norecursion
def distance_matrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_469939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 28), 'int')
    int_469940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 41), 'int')
    defaults = [int_469939, int_469940]
    # Create a new context for function 'distance_matrix'
    module_type_store = module_type_store.open_function_context('distance_matrix', 936, 0, False)
    
    # Passed parameters checking function
    distance_matrix.stypy_localization = localization
    distance_matrix.stypy_type_of_self = None
    distance_matrix.stypy_type_store = module_type_store
    distance_matrix.stypy_function_name = 'distance_matrix'
    distance_matrix.stypy_param_names_list = ['x', 'y', 'p', 'threshold']
    distance_matrix.stypy_varargs_param_name = None
    distance_matrix.stypy_kwargs_param_name = None
    distance_matrix.stypy_call_defaults = defaults
    distance_matrix.stypy_call_varargs = varargs
    distance_matrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'distance_matrix', ['x', 'y', 'p', 'threshold'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'distance_matrix', localization, ['x', 'y', 'p', 'threshold'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'distance_matrix(...)' code ##################

    str_469941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, (-1)), 'str', '\n    Compute the distance matrix.\n\n    Returns the matrix of all pair-wise distances.\n\n    Parameters\n    ----------\n    x : (M, K) array_like\n        Matrix of M vectors in K dimensions.\n    y : (N, K) array_like\n        Matrix of N vectors in K dimensions.\n    p : float, 1 <= p <= infinity\n        Which Minkowski p-norm to use.\n    threshold : positive int\n        If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead\n        of large temporary arrays.\n\n    Returns\n    -------\n    result : (M, N) ndarray\n        Matrix containing the distance from every vector in `x` to every vector\n        in `y`.\n\n    Examples\n    --------\n    >>> from scipy.spatial import distance_matrix\n    >>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])\n    array([[ 1.        ,  1.41421356],\n           [ 1.41421356,  1.        ]])\n\n    ')
    
    # Assigning a Call to a Name (line 969):
    
    # Assigning a Call to a Name (line 969):
    
    # Call to asarray(...): (line 969)
    # Processing the call arguments (line 969)
    # Getting the type of 'x' (line 969)
    x_469944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 19), 'x', False)
    # Processing the call keyword arguments (line 969)
    kwargs_469945 = {}
    # Getting the type of 'np' (line 969)
    np_469942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 969)
    asarray_469943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 969, 8), np_469942, 'asarray')
    # Calling asarray(args, kwargs) (line 969)
    asarray_call_result_469946 = invoke(stypy.reporting.localization.Localization(__file__, 969, 8), asarray_469943, *[x_469944], **kwargs_469945)
    
    # Assigning a type to the variable 'x' (line 969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 4), 'x', asarray_call_result_469946)
    
    # Assigning a Attribute to a Tuple (line 970):
    
    # Assigning a Subscript to a Name (line 970):
    
    # Obtaining the type of the subscript
    int_469947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 4), 'int')
    # Getting the type of 'x' (line 970)
    x_469948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 11), 'x')
    # Obtaining the member 'shape' of a type (line 970)
    shape_469949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 11), x_469948, 'shape')
    # Obtaining the member '__getitem__' of a type (line 970)
    getitem___469950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 4), shape_469949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 970)
    subscript_call_result_469951 = invoke(stypy.reporting.localization.Localization(__file__, 970, 4), getitem___469950, int_469947)
    
    # Assigning a type to the variable 'tuple_var_assignment_466811' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'tuple_var_assignment_466811', subscript_call_result_469951)
    
    # Assigning a Subscript to a Name (line 970):
    
    # Obtaining the type of the subscript
    int_469952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 4), 'int')
    # Getting the type of 'x' (line 970)
    x_469953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 11), 'x')
    # Obtaining the member 'shape' of a type (line 970)
    shape_469954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 11), x_469953, 'shape')
    # Obtaining the member '__getitem__' of a type (line 970)
    getitem___469955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 4), shape_469954, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 970)
    subscript_call_result_469956 = invoke(stypy.reporting.localization.Localization(__file__, 970, 4), getitem___469955, int_469952)
    
    # Assigning a type to the variable 'tuple_var_assignment_466812' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'tuple_var_assignment_466812', subscript_call_result_469956)
    
    # Assigning a Name to a Name (line 970):
    # Getting the type of 'tuple_var_assignment_466811' (line 970)
    tuple_var_assignment_466811_469957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'tuple_var_assignment_466811')
    # Assigning a type to the variable 'm' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'm', tuple_var_assignment_466811_469957)
    
    # Assigning a Name to a Name (line 970):
    # Getting the type of 'tuple_var_assignment_466812' (line 970)
    tuple_var_assignment_466812_469958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'tuple_var_assignment_466812')
    # Assigning a type to the variable 'k' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 7), 'k', tuple_var_assignment_466812_469958)
    
    # Assigning a Call to a Name (line 971):
    
    # Assigning a Call to a Name (line 971):
    
    # Call to asarray(...): (line 971)
    # Processing the call arguments (line 971)
    # Getting the type of 'y' (line 971)
    y_469961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 19), 'y', False)
    # Processing the call keyword arguments (line 971)
    kwargs_469962 = {}
    # Getting the type of 'np' (line 971)
    np_469959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 971)
    asarray_469960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 8), np_469959, 'asarray')
    # Calling asarray(args, kwargs) (line 971)
    asarray_call_result_469963 = invoke(stypy.reporting.localization.Localization(__file__, 971, 8), asarray_469960, *[y_469961], **kwargs_469962)
    
    # Assigning a type to the variable 'y' (line 971)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 4), 'y', asarray_call_result_469963)
    
    # Assigning a Attribute to a Tuple (line 972):
    
    # Assigning a Subscript to a Name (line 972):
    
    # Obtaining the type of the subscript
    int_469964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 4), 'int')
    # Getting the type of 'y' (line 972)
    y_469965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 12), 'y')
    # Obtaining the member 'shape' of a type (line 972)
    shape_469966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 12), y_469965, 'shape')
    # Obtaining the member '__getitem__' of a type (line 972)
    getitem___469967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 4), shape_469966, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 972)
    subscript_call_result_469968 = invoke(stypy.reporting.localization.Localization(__file__, 972, 4), getitem___469967, int_469964)
    
    # Assigning a type to the variable 'tuple_var_assignment_466813' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 4), 'tuple_var_assignment_466813', subscript_call_result_469968)
    
    # Assigning a Subscript to a Name (line 972):
    
    # Obtaining the type of the subscript
    int_469969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 4), 'int')
    # Getting the type of 'y' (line 972)
    y_469970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 12), 'y')
    # Obtaining the member 'shape' of a type (line 972)
    shape_469971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 12), y_469970, 'shape')
    # Obtaining the member '__getitem__' of a type (line 972)
    getitem___469972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 4), shape_469971, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 972)
    subscript_call_result_469973 = invoke(stypy.reporting.localization.Localization(__file__, 972, 4), getitem___469972, int_469969)
    
    # Assigning a type to the variable 'tuple_var_assignment_466814' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 4), 'tuple_var_assignment_466814', subscript_call_result_469973)
    
    # Assigning a Name to a Name (line 972):
    # Getting the type of 'tuple_var_assignment_466813' (line 972)
    tuple_var_assignment_466813_469974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 4), 'tuple_var_assignment_466813')
    # Assigning a type to the variable 'n' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 4), 'n', tuple_var_assignment_466813_469974)
    
    # Assigning a Name to a Name (line 972):
    # Getting the type of 'tuple_var_assignment_466814' (line 972)
    tuple_var_assignment_466814_469975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 4), 'tuple_var_assignment_466814')
    # Assigning a type to the variable 'kk' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 7), 'kk', tuple_var_assignment_466814_469975)
    
    
    # Getting the type of 'k' (line 974)
    k_469976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 7), 'k')
    # Getting the type of 'kk' (line 974)
    kk_469977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 12), 'kk')
    # Applying the binary operator '!=' (line 974)
    result_ne_469978 = python_operator(stypy.reporting.localization.Localization(__file__, 974, 7), '!=', k_469976, kk_469977)
    
    # Testing the type of an if condition (line 974)
    if_condition_469979 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 974, 4), result_ne_469978)
    # Assigning a type to the variable 'if_condition_469979' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 4), 'if_condition_469979', if_condition_469979)
    # SSA begins for if statement (line 974)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 975)
    # Processing the call arguments (line 975)
    str_469981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 25), 'str', 'x contains %d-dimensional vectors but y contains %d-dimensional vectors')
    
    # Obtaining an instance of the builtin type 'tuple' (line 975)
    tuple_469982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 102), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 975)
    # Adding element type (line 975)
    # Getting the type of 'k' (line 975)
    k_469983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 102), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 975, 102), tuple_469982, k_469983)
    # Adding element type (line 975)
    # Getting the type of 'kk' (line 975)
    kk_469984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 105), 'kk', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 975, 102), tuple_469982, kk_469984)
    
    # Applying the binary operator '%' (line 975)
    result_mod_469985 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 25), '%', str_469981, tuple_469982)
    
    # Processing the call keyword arguments (line 975)
    kwargs_469986 = {}
    # Getting the type of 'ValueError' (line 975)
    ValueError_469980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 975)
    ValueError_call_result_469987 = invoke(stypy.reporting.localization.Localization(__file__, 975, 14), ValueError_469980, *[result_mod_469985], **kwargs_469986)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 975, 8), ValueError_call_result_469987, 'raise parameter', BaseException)
    # SSA join for if statement (line 974)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'm' (line 977)
    m_469988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 7), 'm')
    # Getting the type of 'n' (line 977)
    n_469989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 9), 'n')
    # Applying the binary operator '*' (line 977)
    result_mul_469990 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 7), '*', m_469988, n_469989)
    
    # Getting the type of 'k' (line 977)
    k_469991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 11), 'k')
    # Applying the binary operator '*' (line 977)
    result_mul_469992 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 10), '*', result_mul_469990, k_469991)
    
    # Getting the type of 'threshold' (line 977)
    threshold_469993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 16), 'threshold')
    # Applying the binary operator '<=' (line 977)
    result_le_469994 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 7), '<=', result_mul_469992, threshold_469993)
    
    # Testing the type of an if condition (line 977)
    if_condition_469995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 977, 4), result_le_469994)
    # Assigning a type to the variable 'if_condition_469995' (line 977)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 4), 'if_condition_469995', if_condition_469995)
    # SSA begins for if statement (line 977)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to minkowski_distance(...): (line 978)
    # Processing the call arguments (line 978)
    
    # Obtaining the type of the subscript
    slice_469997 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 978, 34), None, None, None)
    # Getting the type of 'np' (line 978)
    np_469998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 38), 'np', False)
    # Obtaining the member 'newaxis' of a type (line 978)
    newaxis_469999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 978, 38), np_469998, 'newaxis')
    slice_470000 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 978, 34), None, None, None)
    # Getting the type of 'x' (line 978)
    x_470001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 34), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 978)
    getitem___470002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 978, 34), x_470001, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 978)
    subscript_call_result_470003 = invoke(stypy.reporting.localization.Localization(__file__, 978, 34), getitem___470002, (slice_469997, newaxis_469999, slice_470000))
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'np' (line 978)
    np_470004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 54), 'np', False)
    # Obtaining the member 'newaxis' of a type (line 978)
    newaxis_470005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 978, 54), np_470004, 'newaxis')
    slice_470006 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 978, 52), None, None, None)
    slice_470007 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 978, 52), None, None, None)
    # Getting the type of 'y' (line 978)
    y_470008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 52), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 978)
    getitem___470009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 978, 52), y_470008, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 978)
    subscript_call_result_470010 = invoke(stypy.reporting.localization.Localization(__file__, 978, 52), getitem___470009, (newaxis_470005, slice_470006, slice_470007))
    
    # Getting the type of 'p' (line 978)
    p_470011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 70), 'p', False)
    # Processing the call keyword arguments (line 978)
    kwargs_470012 = {}
    # Getting the type of 'minkowski_distance' (line 978)
    minkowski_distance_469996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 15), 'minkowski_distance', False)
    # Calling minkowski_distance(args, kwargs) (line 978)
    minkowski_distance_call_result_470013 = invoke(stypy.reporting.localization.Localization(__file__, 978, 15), minkowski_distance_469996, *[subscript_call_result_470003, subscript_call_result_470010, p_470011], **kwargs_470012)
    
    # Assigning a type to the variable 'stypy_return_type' (line 978)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 8), 'stypy_return_type', minkowski_distance_call_result_470013)
    # SSA branch for the else part of an if statement (line 977)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 980):
    
    # Assigning a Call to a Name (line 980):
    
    # Call to empty(...): (line 980)
    # Processing the call arguments (line 980)
    
    # Obtaining an instance of the builtin type 'tuple' (line 980)
    tuple_470016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 980)
    # Adding element type (line 980)
    # Getting the type of 'm' (line 980)
    m_470017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 27), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 980, 27), tuple_470016, m_470017)
    # Adding element type (line 980)
    # Getting the type of 'n' (line 980)
    n_470018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 29), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 980, 27), tuple_470016, n_470018)
    
    # Processing the call keyword arguments (line 980)
    # Getting the type of 'float' (line 980)
    float_470019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 38), 'float', False)
    keyword_470020 = float_470019
    kwargs_470021 = {'dtype': keyword_470020}
    # Getting the type of 'np' (line 980)
    np_470014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 17), 'np', False)
    # Obtaining the member 'empty' of a type (line 980)
    empty_470015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 980, 17), np_470014, 'empty')
    # Calling empty(args, kwargs) (line 980)
    empty_call_result_470022 = invoke(stypy.reporting.localization.Localization(__file__, 980, 17), empty_470015, *[tuple_470016], **kwargs_470021)
    
    # Assigning a type to the variable 'result' (line 980)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 8), 'result', empty_call_result_470022)
    
    
    # Getting the type of 'm' (line 981)
    m_470023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 11), 'm')
    # Getting the type of 'n' (line 981)
    n_470024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 15), 'n')
    # Applying the binary operator '<' (line 981)
    result_lt_470025 = python_operator(stypy.reporting.localization.Localization(__file__, 981, 11), '<', m_470023, n_470024)
    
    # Testing the type of an if condition (line 981)
    if_condition_470026 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 981, 8), result_lt_470025)
    # Assigning a type to the variable 'if_condition_470026' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 8), 'if_condition_470026', if_condition_470026)
    # SSA begins for if statement (line 981)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to range(...): (line 982)
    # Processing the call arguments (line 982)
    # Getting the type of 'm' (line 982)
    m_470028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 27), 'm', False)
    # Processing the call keyword arguments (line 982)
    kwargs_470029 = {}
    # Getting the type of 'range' (line 982)
    range_470027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 21), 'range', False)
    # Calling range(args, kwargs) (line 982)
    range_call_result_470030 = invoke(stypy.reporting.localization.Localization(__file__, 982, 21), range_470027, *[m_470028], **kwargs_470029)
    
    # Testing the type of a for loop iterable (line 982)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 982, 12), range_call_result_470030)
    # Getting the type of the for loop variable (line 982)
    for_loop_var_470031 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 982, 12), range_call_result_470030)
    # Assigning a type to the variable 'i' (line 982)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 12), 'i', for_loop_var_470031)
    # SSA begins for a for statement (line 982)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 983):
    
    # Assigning a Call to a Subscript (line 983):
    
    # Call to minkowski_distance(...): (line 983)
    # Processing the call arguments (line 983)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 983)
    i_470033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 51), 'i', False)
    # Getting the type of 'x' (line 983)
    x_470034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 49), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 983)
    getitem___470035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 983, 49), x_470034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 983)
    subscript_call_result_470036 = invoke(stypy.reporting.localization.Localization(__file__, 983, 49), getitem___470035, i_470033)
    
    # Getting the type of 'y' (line 983)
    y_470037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 54), 'y', False)
    # Getting the type of 'p' (line 983)
    p_470038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 56), 'p', False)
    # Processing the call keyword arguments (line 983)
    kwargs_470039 = {}
    # Getting the type of 'minkowski_distance' (line 983)
    minkowski_distance_470032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 30), 'minkowski_distance', False)
    # Calling minkowski_distance(args, kwargs) (line 983)
    minkowski_distance_call_result_470040 = invoke(stypy.reporting.localization.Localization(__file__, 983, 30), minkowski_distance_470032, *[subscript_call_result_470036, y_470037, p_470038], **kwargs_470039)
    
    # Getting the type of 'result' (line 983)
    result_470041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 16), 'result')
    # Getting the type of 'i' (line 983)
    i_470042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 23), 'i')
    slice_470043 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 983, 16), None, None, None)
    # Storing an element on a container (line 983)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 983, 16), result_470041, ((i_470042, slice_470043), minkowski_distance_call_result_470040))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 981)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to range(...): (line 985)
    # Processing the call arguments (line 985)
    # Getting the type of 'n' (line 985)
    n_470045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 27), 'n', False)
    # Processing the call keyword arguments (line 985)
    kwargs_470046 = {}
    # Getting the type of 'range' (line 985)
    range_470044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 21), 'range', False)
    # Calling range(args, kwargs) (line 985)
    range_call_result_470047 = invoke(stypy.reporting.localization.Localization(__file__, 985, 21), range_470044, *[n_470045], **kwargs_470046)
    
    # Testing the type of a for loop iterable (line 985)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 985, 12), range_call_result_470047)
    # Getting the type of the for loop variable (line 985)
    for_loop_var_470048 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 985, 12), range_call_result_470047)
    # Assigning a type to the variable 'j' (line 985)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 12), 'j', for_loop_var_470048)
    # SSA begins for a for statement (line 985)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 986):
    
    # Assigning a Call to a Subscript (line 986):
    
    # Call to minkowski_distance(...): (line 986)
    # Processing the call arguments (line 986)
    # Getting the type of 'x' (line 986)
    x_470050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 49), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 986)
    j_470051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 53), 'j', False)
    # Getting the type of 'y' (line 986)
    y_470052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 51), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 986)
    getitem___470053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 986, 51), y_470052, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 986)
    subscript_call_result_470054 = invoke(stypy.reporting.localization.Localization(__file__, 986, 51), getitem___470053, j_470051)
    
    # Getting the type of 'p' (line 986)
    p_470055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 56), 'p', False)
    # Processing the call keyword arguments (line 986)
    kwargs_470056 = {}
    # Getting the type of 'minkowski_distance' (line 986)
    minkowski_distance_470049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 30), 'minkowski_distance', False)
    # Calling minkowski_distance(args, kwargs) (line 986)
    minkowski_distance_call_result_470057 = invoke(stypy.reporting.localization.Localization(__file__, 986, 30), minkowski_distance_470049, *[x_470050, subscript_call_result_470054, p_470055], **kwargs_470056)
    
    # Getting the type of 'result' (line 986)
    result_470058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 16), 'result')
    slice_470059 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 986, 16), None, None, None)
    # Getting the type of 'j' (line 986)
    j_470060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 25), 'j')
    # Storing an element on a container (line 986)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 986, 16), result_470058, ((slice_470059, j_470060), minkowski_distance_call_result_470057))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 981)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 987)
    result_470061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 987)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 987, 8), 'stypy_return_type', result_470061)
    # SSA join for if statement (line 977)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'distance_matrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'distance_matrix' in the type store
    # Getting the type of 'stypy_return_type' (line 936)
    stypy_return_type_470062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_470062)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'distance_matrix'
    return stypy_return_type_470062

# Assigning a type to the variable 'distance_matrix' (line 936)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 0), 'distance_matrix', distance_matrix)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
