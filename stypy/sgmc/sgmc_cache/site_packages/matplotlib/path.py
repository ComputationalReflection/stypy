
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A module for dealing with the polylines used throughout matplotlib.
3: 
4: The primary class for polyline handling in matplotlib is :class:`Path`.
5: Almost all vector drawing makes use of Paths somewhere in the drawing
6: pipeline.
7: 
8: Whilst a :class:`Path` instance itself cannot be drawn, there exists
9: :class:`~matplotlib.artist.Artist` subclasses which can be used for
10: convenient Path visualisation - the two most frequently used of these are
11: :class:`~matplotlib.patches.PathPatch` and
12: :class:`~matplotlib.collections.PathCollection`.
13: '''
14: 
15: from __future__ import (absolute_import, division, print_function,
16:                         unicode_literals)
17: 
18: import six
19: 
20: import math
21: from weakref import WeakValueDictionary
22: 
23: import numpy as np
24: 
25: from . import _path, rcParams
26: from .cbook import (_to_unmasked_float_array, simple_linear_interpolation,
27:                     maxdict)
28: 
29: 
30: class Path(object):
31:     '''
32:     :class:`Path` represents a series of possibly disconnected,
33:     possibly closed, line and curve segments.
34: 
35:     The underlying storage is made up of two parallel numpy arrays:
36:       - *vertices*: an Nx2 float array of vertices
37:       - *codes*: an N-length uint8 array of vertex types
38: 
39:     These two arrays always have the same length in the first
40:     dimension.  For example, to represent a cubic curve, you must
41:     provide three vertices as well as three codes ``CURVE3``.
42: 
43:     The code types are:
44: 
45:        - ``STOP``   :  1 vertex (ignored)
46:            A marker for the end of the entire path (currently not
47:            required and ignored)
48: 
49:        - ``MOVETO`` :  1 vertex
50:             Pick up the pen and move to the given vertex.
51: 
52:        - ``LINETO`` :  1 vertex
53:             Draw a line from the current position to the given vertex.
54: 
55:        - ``CURVE3`` :  1 control point, 1 endpoint
56:           Draw a quadratic Bezier curve from the current position,
57:           with the given control point, to the given end point.
58: 
59:        - ``CURVE4`` :  2 control points, 1 endpoint
60:           Draw a cubic Bezier curve from the current position, with
61:           the given control points, to the given end point.
62: 
63:        - ``CLOSEPOLY`` : 1 vertex (ignored)
64:           Draw a line segment to the start point of the current
65:           polyline.
66: 
67:     Users of Path objects should not access the vertices and codes
68:     arrays directly.  Instead, they should use :meth:`iter_segments`
69:     or :meth:`cleaned` to get the vertex/code pairs.  This is important,
70:     since many :class:`Path` objects, as an optimization, do not store a
71:     *codes* at all, but have a default one provided for them by
72:     :meth:`iter_segments`.
73: 
74:     Some behavior of Path objects can be controlled by rcParams. See
75:     the rcParams whose keys contain 'path.'.
76: 
77:     .. note::
78: 
79:         The vertices and codes arrays should be treated as
80:         immutable -- there are a number of optimizations and assumptions
81:         made up front in the constructor that will not change when the
82:         data changes.
83: 
84:     '''
85: 
86:     # Path codes
87:     STOP = 0         # 1 vertex
88:     MOVETO = 1       # 1 vertex
89:     LINETO = 2       # 1 vertex
90:     CURVE3 = 3       # 2 vertices
91:     CURVE4 = 4       # 3 vertices
92:     CLOSEPOLY = 79   # 1 vertex
93: 
94:     #: A dictionary mapping Path codes to the number of vertices that the
95:     #: code expects.
96:     NUM_VERTICES_FOR_CODE = {STOP: 1,
97:                              MOVETO: 1,
98:                              LINETO: 1,
99:                              CURVE3: 2,
100:                              CURVE4: 3,
101:                              CLOSEPOLY: 1}
102: 
103:     code_type = np.uint8
104: 
105:     def __init__(self, vertices, codes=None, _interpolation_steps=1,
106:                  closed=False, readonly=False):
107:         '''
108:         Create a new path with the given vertices and codes.
109: 
110:         Parameters
111:         ----------
112:         vertices : array_like
113:             The ``(n, 2)`` float array, masked array or sequence of pairs
114:             representing the vertices of the path.
115: 
116:             If *vertices* contains masked values, they will be converted
117:             to NaNs which are then handled correctly by the Agg
118:             PathIterator and other consumers of path data, such as
119:             :meth:`iter_segments`.
120:         codes : {None, array_like}, optional
121:             n-length array integers representing the codes of the path.
122:             If not None, codes must be the same length as vertices.
123:             If None, *vertices* will be treated as a series of line segments.
124:         _interpolation_steps : int, optional
125:             Used as a hint to certain projections, such as Polar, that this
126:             path should be linearly interpolated immediately before drawing.
127:             This attribute is primarily an implementation detail and is not
128:             intended for public use.
129:         closed : bool, optional
130:             If *codes* is None and closed is True, vertices will be treated as
131:             line segments of a closed polygon.
132:         readonly : bool, optional
133:             Makes the path behave in an immutable way and sets the vertices
134:             and codes as read-only arrays.
135:         '''
136:         vertices = _to_unmasked_float_array(vertices)
137:         if (vertices.ndim != 2) or (vertices.shape[1] != 2):
138:             msg = "'vertices' must be a 2D list or array with shape Nx2"
139:             raise ValueError(msg)
140: 
141:         if codes is not None:
142:             codes = np.asarray(codes, self.code_type)
143:             if (codes.ndim != 1) or len(codes) != len(vertices):
144:                 msg = ("'codes' must be a 1D list or array with the same"
145:                        " length of 'vertices'")
146:                 raise ValueError(msg)
147:             if len(codes) and codes[0] != self.MOVETO:
148:                 msg = ("The first element of 'code' must be equal to 'MOVETO':"
149:                        " {0}")
150:                 raise ValueError(msg.format(self.MOVETO))
151:         elif closed:
152:             codes = np.empty(len(vertices), dtype=self.code_type)
153:             codes[0] = self.MOVETO
154:             codes[1:-1] = self.LINETO
155:             codes[-1] = self.CLOSEPOLY
156: 
157:         self._vertices = vertices
158:         self._codes = codes
159:         self._interpolation_steps = _interpolation_steps
160:         self._update_values()
161: 
162:         if readonly:
163:             self._vertices.flags.writeable = False
164:             if self._codes is not None:
165:                 self._codes.flags.writeable = False
166:             self._readonly = True
167:         else:
168:             self._readonly = False
169: 
170:     @classmethod
171:     def _fast_from_codes_and_verts(cls, verts, codes, internals=None):
172:         '''
173:         Creates a Path instance without the expense of calling the constructor
174: 
175:         Parameters
176:         ----------
177:         verts : numpy array
178:         codes : numpy array
179:         internals : dict or None
180:             The attributes that the resulting path should have.
181:             Allowed keys are ``readonly``, ``should_simplify``,
182:             ``simplify_threshold``, ``has_nonfinite`` and
183:             ``interpolation_steps``.
184: 
185:         '''
186:         internals = internals or {}
187:         pth = cls.__new__(cls)
188:         pth._vertices = _to_unmasked_float_array(verts)
189:         pth._codes = codes
190:         pth._readonly = internals.pop('readonly', False)
191:         pth.should_simplify = internals.pop('should_simplify', True)
192:         pth.simplify_threshold = (
193:             internals.pop('simplify_threshold',
194:                           rcParams['path.simplify_threshold'])
195:         )
196:         pth._has_nonfinite = internals.pop('has_nonfinite', False)
197:         pth._interpolation_steps = internals.pop('interpolation_steps', 1)
198:         if internals:
199:             raise ValueError('Unexpected internals provided to '
200:                              '_fast_from_codes_and_verts: '
201:                              '{0}'.format('\n *'.join(internals)))
202:         return pth
203: 
204:     def _update_values(self):
205:         self._simplify_threshold = rcParams['path.simplify_threshold']
206:         self._should_simplify = (
207:             self._simplify_threshold > 0 and
208:             rcParams['path.simplify'] and
209:             len(self._vertices) >= 128 and
210:             (self._codes is None or np.all(self._codes <= Path.LINETO))
211:         )
212:         self._has_nonfinite = not np.isfinite(self._vertices).all()
213: 
214:     @property
215:     def vertices(self):
216:         '''
217:         The list of vertices in the `Path` as an Nx2 numpy array.
218:         '''
219:         return self._vertices
220: 
221:     @vertices.setter
222:     def vertices(self, vertices):
223:         if self._readonly:
224:             raise AttributeError("Can't set vertices on a readonly Path")
225:         self._vertices = vertices
226:         self._update_values()
227: 
228:     @property
229:     def codes(self):
230:         '''
231:         The list of codes in the `Path` as a 1-D numpy array.  Each
232:         code is one of `STOP`, `MOVETO`, `LINETO`, `CURVE3`, `CURVE4`
233:         or `CLOSEPOLY`.  For codes that correspond to more than one
234:         vertex (`CURVE3` and `CURVE4`), that code will be repeated so
235:         that the length of `self.vertices` and `self.codes` is always
236:         the same.
237:         '''
238:         return self._codes
239: 
240:     @codes.setter
241:     def codes(self, codes):
242:         if self._readonly:
243:             raise AttributeError("Can't set codes on a readonly Path")
244:         self._codes = codes
245:         self._update_values()
246: 
247:     @property
248:     def simplify_threshold(self):
249:         '''
250:         The fraction of a pixel difference below which vertices will
251:         be simplified out.
252:         '''
253:         return self._simplify_threshold
254: 
255:     @simplify_threshold.setter
256:     def simplify_threshold(self, threshold):
257:         self._simplify_threshold = threshold
258: 
259:     @property
260:     def has_nonfinite(self):
261:         '''
262:         `True` if the vertices array has nonfinite values.
263:         '''
264:         return self._has_nonfinite
265: 
266:     @property
267:     def should_simplify(self):
268:         '''
269:         `True` if the vertices array should be simplified.
270:         '''
271:         return self._should_simplify
272: 
273:     @should_simplify.setter
274:     def should_simplify(self, should_simplify):
275:         self._should_simplify = should_simplify
276: 
277:     @property
278:     def readonly(self):
279:         '''
280:         `True` if the `Path` is read-only.
281:         '''
282:         return self._readonly
283: 
284:     def __copy__(self):
285:         '''
286:         Returns a shallow copy of the `Path`, which will share the
287:         vertices and codes with the source `Path`.
288:         '''
289:         import copy
290:         return copy.copy(self)
291: 
292:     copy = __copy__
293: 
294:     def __deepcopy__(self, memo=None):
295:         '''
296:         Returns a deepcopy of the `Path`.  The `Path` will not be
297:         readonly, even if the source `Path` is.
298:         '''
299:         try:
300:             codes = self.codes.copy()
301:         except AttributeError:
302:             codes = None
303:         return self.__class__(
304:             self.vertices.copy(), codes,
305:             _interpolation_steps=self._interpolation_steps)
306: 
307:     deepcopy = __deepcopy__
308: 
309:     @classmethod
310:     def make_compound_path_from_polys(cls, XY):
311:         '''
312:         Make a compound path object to draw a number
313:         of polygons with equal numbers of sides XY is a (numpolys x
314:         numsides x 2) numpy array of vertices.  Return object is a
315:         :class:`Path`
316: 
317:         .. plot:: gallery/api/histogram_path.py
318: 
319:         '''
320: 
321:         # for each poly: 1 for the MOVETO, (numsides-1) for the LINETO, 1 for
322:         # the CLOSEPOLY; the vert for the closepoly is ignored but we still
323:         # need it to keep the codes aligned with the vertices
324:         numpolys, numsides, two = XY.shape
325:         if two != 2:
326:             raise ValueError("The third dimension of 'XY' must be 2")
327:         stride = numsides + 1
328:         nverts = numpolys * stride
329:         verts = np.zeros((nverts, 2))
330:         codes = np.ones(nverts, int) * cls.LINETO
331:         codes[0::stride] = cls.MOVETO
332:         codes[numsides::stride] = cls.CLOSEPOLY
333:         for i in range(numsides):
334:             verts[i::stride] = XY[:, i]
335: 
336:         return cls(verts, codes)
337: 
338:     @classmethod
339:     def make_compound_path(cls, *args):
340:         '''Make a compound path from a list of Path objects.'''
341:         # Handle an empty list in args (i.e. no args).
342:         if not args:
343:             return Path(np.empty([0, 2], dtype=np.float32))
344: 
345:         lengths = [len(x) for x in args]
346:         total_length = sum(lengths)
347: 
348:         vertices = np.vstack([x.vertices for x in args])
349:         vertices.reshape((total_length, 2))
350: 
351:         codes = np.empty(total_length, dtype=cls.code_type)
352:         i = 0
353:         for path in args:
354:             if path.codes is None:
355:                 codes[i] = cls.MOVETO
356:                 codes[i + 1:i + len(path.vertices)] = cls.LINETO
357:             else:
358:                 codes[i:i + len(path.codes)] = path.codes
359:             i += len(path.vertices)
360: 
361:         return cls(vertices, codes)
362: 
363:     def __repr__(self):
364:         return "Path(%r, %r)" % (self.vertices, self.codes)
365: 
366:     def __len__(self):
367:         return len(self.vertices)
368: 
369:     def iter_segments(self, transform=None, remove_nans=True, clip=None,
370:                       snap=False, stroke_width=1.0, simplify=None,
371:                       curves=True, sketch=None):
372:         '''
373:         Iterates over all of the curve segments in the path.  Each
374:         iteration returns a 2-tuple (*vertices*, *code*), where
375:         *vertices* is a sequence of 1 - 3 coordinate pairs, and *code* is
376:         one of the :class:`Path` codes.
377: 
378:         Additionally, this method can provide a number of standard
379:         cleanups and conversions to the path.
380: 
381:         Parameters
382:         ----------
383:         transform : None or :class:`~matplotlib.transforms.Transform` instance
384:             If not None, the given affine transformation will
385:             be applied to the path.
386:         remove_nans : {False, True}, optional
387:             If True, will remove all NaNs from the path and
388:             insert MOVETO commands to skip over them.
389:         clip : None or sequence, optional
390:             If not None, must be a four-tuple (x1, y1, x2, y2)
391:             defining a rectangle in which to clip the path.
392:         snap : None or bool, optional
393:             If None, auto-snap to pixels, to reduce
394:             fuzziness of rectilinear lines.  If True, force snapping, and
395:             if False, don't snap.
396:         stroke_width : float, optional
397:             The width of the stroke being drawn.  Needed
398:              as a hint for the snapping algorithm.
399:         simplify : None or bool, optional
400:             If True, perform simplification, to remove
401:              vertices that do not affect the appearance of the path.  If
402:              False, perform no simplification.  If None, use the
403:              should_simplify member variable.  See also the rcParams
404:              path.simplify and path.simplify_threshold.
405:         curves : {True, False}, optional
406:             If True, curve segments will be returned as curve
407:             segments.  If False, all curves will be converted to line
408:             segments.
409:         sketch : None or sequence, optional
410:             If not None, must be a 3-tuple of the form
411:             (scale, length, randomness), representing the sketch
412:             parameters.
413:         '''
414:         if not len(self):
415:             return
416: 
417:         cleaned = self.cleaned(transform=transform,
418:                                remove_nans=remove_nans, clip=clip,
419:                                snap=snap, stroke_width=stroke_width,
420:                                simplify=simplify, curves=curves,
421:                                sketch=sketch)
422:         vertices = cleaned.vertices
423:         codes = cleaned.codes
424:         len_vertices = vertices.shape[0]
425: 
426:         # Cache these object lookups for performance in the loop.
427:         NUM_VERTICES_FOR_CODE = self.NUM_VERTICES_FOR_CODE
428:         STOP = self.STOP
429: 
430:         i = 0
431:         while i < len_vertices:
432:             code = codes[i]
433:             if code == STOP:
434:                 return
435:             else:
436:                 num_vertices = NUM_VERTICES_FOR_CODE[code]
437:                 curr_vertices = vertices[i:i+num_vertices].flatten()
438:                 yield curr_vertices, code
439:                 i += num_vertices
440: 
441:     def cleaned(self, transform=None, remove_nans=False, clip=None,
442:                 quantize=False, simplify=False, curves=False,
443:                 stroke_width=1.0, snap=False, sketch=None):
444:         '''
445:         Cleans up the path according to the parameters returning a new
446:         Path instance.
447: 
448:         .. seealso::
449: 
450:             See :meth:`iter_segments` for details of the keyword arguments.
451: 
452:         Returns
453:         -------
454:         Path instance with cleaned up vertices and codes.
455: 
456:         '''
457:         vertices, codes = _path.cleanup_path(self, transform,
458:                                              remove_nans, clip,
459:                                              snap, stroke_width,
460:                                              simplify, curves, sketch)
461:         internals = {'should_simplify': self.should_simplify and not simplify,
462:                      'has_nonfinite': self.has_nonfinite and not remove_nans,
463:                      'simplify_threshold': self.simplify_threshold,
464:                      'interpolation_steps': self._interpolation_steps}
465:         return Path._fast_from_codes_and_verts(vertices, codes, internals)
466: 
467:     def transformed(self, transform):
468:         '''
469:         Return a transformed copy of the path.
470: 
471:         .. seealso::
472: 
473:             :class:`matplotlib.transforms.TransformedPath`
474:                 A specialized path class that will cache the
475:                 transformed result and automatically update when the
476:                 transform changes.
477:         '''
478:         return Path(transform.transform(self.vertices), self.codes,
479:                     self._interpolation_steps)
480: 
481:     def contains_point(self, point, transform=None, radius=0.0):
482:         '''
483:         Returns whether the (closed) path contains the given point.
484: 
485:         If *transform* is not ``None``, the path will be transformed before
486:         performing the test.
487: 
488:         *radius* allows the path to be made slightly larger or smaller.
489:         '''
490:         if transform is not None:
491:             transform = transform.frozen()
492:         # `point_in_path` does not handle nonlinear transforms, so we
493:         # transform the path ourselves.  If `transform` is affine, letting
494:         # `point_in_path` handle the transform avoids allocating an extra
495:         # buffer.
496:         if transform and not transform.is_affine:
497:             self = transform.transform_path(self)
498:             transform = None
499:         return _path.point_in_path(point[0], point[1], radius, self, transform)
500: 
501:     def contains_points(self, points, transform=None, radius=0.0):
502:         '''
503:         Returns a bool array which is ``True`` if the (closed) path contains
504:         the corresponding point.
505: 
506:         If *transform* is not ``None``, the path will be transformed before
507:         performing the test.
508: 
509:         *radius* allows the path to be made slightly larger or smaller.
510:         '''
511:         if transform is not None:
512:             transform = transform.frozen()
513:         result = _path.points_in_path(points, radius, self, transform)
514:         return result.astype('bool')
515: 
516:     def contains_path(self, path, transform=None):
517:         '''
518:         Returns whether this (closed) path completely contains the given path.
519: 
520:         If *transform* is not ``None``, the path will be transformed before
521:         performing the test.
522:         '''
523:         if transform is not None:
524:             transform = transform.frozen()
525:         return _path.path_in_path(self, None, path, transform)
526: 
527:     def get_extents(self, transform=None):
528:         '''
529:         Returns the extents (*xmin*, *ymin*, *xmax*, *ymax*) of the
530:         path.
531: 
532:         Unlike computing the extents on the *vertices* alone, this
533:         algorithm will take into account the curves and deal with
534:         control points appropriately.
535:         '''
536:         from .transforms import Bbox
537:         path = self
538:         if transform is not None:
539:             transform = transform.frozen()
540:             if not transform.is_affine:
541:                 path = self.transformed(transform)
542:                 transform = None
543:         return Bbox(_path.get_path_extents(path, transform))
544: 
545:     def intersects_path(self, other, filled=True):
546:         '''
547:         Returns *True* if this path intersects another given path.
548: 
549:         *filled*, when True, treats the paths as if they were filled.
550:         That is, if one path completely encloses the other,
551:         :meth:`intersects_path` will return True.
552:         '''
553:         return _path.path_intersects_path(self, other, filled)
554: 
555:     def intersects_bbox(self, bbox, filled=True):
556:         '''
557:         Returns *True* if this path intersects a given
558:         :class:`~matplotlib.transforms.Bbox`.
559: 
560:         *filled*, when True, treats the path as if it was filled.
561:         That is, if the path completely encloses the bounding box,
562:         :meth:`intersects_bbox` will return True.
563: 
564:         The bounding box is always considered filled.
565:         '''
566:         return _path.path_intersects_rectangle(self,
567:             bbox.x0, bbox.y0, bbox.x1, bbox.y1, filled)
568: 
569:     def interpolated(self, steps):
570:         '''
571:         Returns a new path resampled to length N x steps.  Does not
572:         currently handle interpolating curves.
573:         '''
574:         if steps == 1:
575:             return self
576: 
577:         vertices = simple_linear_interpolation(self.vertices, steps)
578:         codes = self.codes
579:         if codes is not None:
580:             new_codes = Path.LINETO * np.ones(((len(codes) - 1) * steps + 1, ))
581:             new_codes[0::steps] = codes
582:         else:
583:             new_codes = None
584:         return Path(vertices, new_codes)
585: 
586:     def to_polygons(self, transform=None, width=0, height=0, closed_only=True):
587:         '''
588:         Convert this path to a list of polygons or polylines.  Each
589:         polygon/polyline is an Nx2 array of vertices.  In other words,
590:         each polygon has no ``MOVETO`` instructions or curves.  This
591:         is useful for displaying in backends that do not support
592:         compound paths or Bezier curves, such as GDK.
593: 
594:         If *width* and *height* are both non-zero then the lines will
595:         be simplified so that vertices outside of (0, 0), (width,
596:         height) will be clipped.
597: 
598:         If *closed_only* is `True` (default), only closed polygons,
599:         with the last point being the same as the first point, will be
600:         returned.  Any unclosed polylines in the path will be
601:         explicitly closed.  If *closed_only* is `False`, any unclosed
602:         polygons in the path will be returned as unclosed polygons,
603:         and the closed polygons will be returned explicitly closed by
604:         setting the last point to the same as the first point.
605:         '''
606:         if len(self.vertices) == 0:
607:             return []
608: 
609:         if transform is not None:
610:             transform = transform.frozen()
611: 
612:         if self.codes is None and (width == 0 or height == 0):
613:             vertices = self.vertices
614:             if closed_only:
615:                 if len(vertices) < 3:
616:                     return []
617:                 elif np.any(vertices[0] != vertices[-1]):
618:                     vertices = list(vertices) + [vertices[0]]
619: 
620:             if transform is None:
621:                 return [vertices]
622:             else:
623:                 return [transform.transform(vertices)]
624: 
625:         # Deal with the case where there are curves and/or multiple
626:         # subpaths (using extension code)
627:         return _path.convert_path_to_polygons(
628:             self, transform, width, height, closed_only)
629: 
630:     _unit_rectangle = None
631: 
632:     @classmethod
633:     def unit_rectangle(cls):
634:         '''
635:         Return a :class:`Path` instance of the unit rectangle
636:         from (0, 0) to (1, 1).
637:         '''
638:         if cls._unit_rectangle is None:
639:             cls._unit_rectangle = \
640:                 cls([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
641:                      [0.0, 0.0]],
642:                     [cls.MOVETO, cls.LINETO, cls.LINETO, cls.LINETO,
643:                      cls.CLOSEPOLY],
644:                     readonly=True)
645:         return cls._unit_rectangle
646: 
647:     _unit_regular_polygons = WeakValueDictionary()
648: 
649:     @classmethod
650:     def unit_regular_polygon(cls, numVertices):
651:         '''
652:         Return a :class:`Path` instance for a unit regular
653:         polygon with the given *numVertices* and radius of 1.0,
654:         centered at (0, 0).
655:         '''
656:         if numVertices <= 16:
657:             path = cls._unit_regular_polygons.get(numVertices)
658:         else:
659:             path = None
660:         if path is None:
661:             theta = (2*np.pi/numVertices *
662:                      np.arange(numVertices + 1).reshape((numVertices + 1, 1)))
663:             # This initial rotation is to make sure the polygon always
664:             # "points-up"
665:             theta += np.pi / 2.0
666:             verts = np.concatenate((np.cos(theta), np.sin(theta)), 1)
667:             codes = np.empty((numVertices + 1,))
668:             codes[0] = cls.MOVETO
669:             codes[1:-1] = cls.LINETO
670:             codes[-1] = cls.CLOSEPOLY
671:             path = cls(verts, codes, readonly=True)
672:             if numVertices <= 16:
673:                 cls._unit_regular_polygons[numVertices] = path
674:         return path
675: 
676:     _unit_regular_stars = WeakValueDictionary()
677: 
678:     @classmethod
679:     def unit_regular_star(cls, numVertices, innerCircle=0.5):
680:         '''
681:         Return a :class:`Path` for a unit regular star
682:         with the given numVertices and radius of 1.0, centered at (0,
683:         0).
684:         '''
685:         if numVertices <= 16:
686:             path = cls._unit_regular_stars.get((numVertices, innerCircle))
687:         else:
688:             path = None
689:         if path is None:
690:             ns2 = numVertices * 2
691:             theta = (2*np.pi/ns2 * np.arange(ns2 + 1))
692:             # This initial rotation is to make sure the polygon always
693:             # "points-up"
694:             theta += np.pi / 2.0
695:             r = np.ones(ns2 + 1)
696:             r[1::2] = innerCircle
697:             verts = np.vstack((r*np.cos(theta), r*np.sin(theta))).transpose()
698:             codes = np.empty((ns2 + 1,))
699:             codes[0] = cls.MOVETO
700:             codes[1:-1] = cls.LINETO
701:             codes[-1] = cls.CLOSEPOLY
702:             path = cls(verts, codes, readonly=True)
703:             if numVertices <= 16:
704:                 cls._unit_regular_stars[(numVertices, innerCircle)] = path
705:         return path
706: 
707:     @classmethod
708:     def unit_regular_asterisk(cls, numVertices):
709:         '''
710:         Return a :class:`Path` for a unit regular
711:         asterisk with the given numVertices and radius of 1.0,
712:         centered at (0, 0).
713:         '''
714:         return cls.unit_regular_star(numVertices, 0.0)
715: 
716:     _unit_circle = None
717: 
718:     @classmethod
719:     def unit_circle(cls):
720:         '''
721:         Return the readonly :class:`Path` of the unit circle.
722: 
723:         For most cases, :func:`Path.circle` will be what you want.
724: 
725:         '''
726:         if cls._unit_circle is None:
727:             cls._unit_circle = cls.circle(center=(0, 0), radius=1,
728:                                           readonly=True)
729:         return cls._unit_circle
730: 
731:     @classmethod
732:     def circle(cls, center=(0., 0.), radius=1., readonly=False):
733:         '''
734:         Return a Path representing a circle of a given radius and center.
735: 
736:         Parameters
737:         ----------
738:         center : pair of floats
739:             The center of the circle. Default ``(0, 0)``.
740:         radius : float
741:             The radius of the circle. Default is 1.
742:         readonly : bool
743:             Whether the created path should have the "readonly" argument
744:             set when creating the Path instance.
745: 
746:         Notes
747:         -----
748:         The circle is approximated using cubic Bezier curves.  This
749:         uses 8 splines around the circle using the approach presented
750:         here:
751: 
752:           Lancaster, Don.  `Approximating a Circle or an Ellipse Using Four
753:           Bezier Cubic Splines <http://www.tinaja.com/glib/ellipse4.pdf>`_.
754: 
755:         '''
756:         MAGIC = 0.2652031
757:         SQRTHALF = np.sqrt(0.5)
758:         MAGIC45 = SQRTHALF * MAGIC
759: 
760:         vertices = np.array([[0.0, -1.0],
761: 
762:                              [MAGIC, -1.0],
763:                              [SQRTHALF-MAGIC45, -SQRTHALF-MAGIC45],
764:                              [SQRTHALF, -SQRTHALF],
765: 
766:                              [SQRTHALF+MAGIC45, -SQRTHALF+MAGIC45],
767:                              [1.0, -MAGIC],
768:                              [1.0, 0.0],
769: 
770:                              [1.0, MAGIC],
771:                              [SQRTHALF+MAGIC45, SQRTHALF-MAGIC45],
772:                              [SQRTHALF, SQRTHALF],
773: 
774:                              [SQRTHALF-MAGIC45, SQRTHALF+MAGIC45],
775:                              [MAGIC, 1.0],
776:                              [0.0, 1.0],
777: 
778:                              [-MAGIC, 1.0],
779:                              [-SQRTHALF+MAGIC45, SQRTHALF+MAGIC45],
780:                              [-SQRTHALF, SQRTHALF],
781: 
782:                              [-SQRTHALF-MAGIC45, SQRTHALF-MAGIC45],
783:                              [-1.0, MAGIC],
784:                              [-1.0, 0.0],
785: 
786:                              [-1.0, -MAGIC],
787:                              [-SQRTHALF-MAGIC45, -SQRTHALF+MAGIC45],
788:                              [-SQRTHALF, -SQRTHALF],
789: 
790:                              [-SQRTHALF+MAGIC45, -SQRTHALF-MAGIC45],
791:                              [-MAGIC, -1.0],
792:                              [0.0, -1.0],
793: 
794:                              [0.0, -1.0]],
795:                             dtype=float)
796: 
797:         codes = [cls.CURVE4] * 26
798:         codes[0] = cls.MOVETO
799:         codes[-1] = cls.CLOSEPOLY
800:         return Path(vertices * radius + center, codes, readonly=readonly)
801: 
802:     _unit_circle_righthalf = None
803: 
804:     @classmethod
805:     def unit_circle_righthalf(cls):
806:         '''
807:         Return a :class:`Path` of the right half
808:         of a unit circle. The circle is approximated using cubic Bezier
809:         curves.  This uses 4 splines around the circle using the approach
810:         presented here:
811: 
812:           Lancaster, Don.  `Approximating a Circle or an Ellipse Using Four
813:           Bezier Cubic Splines <http://www.tinaja.com/glib/ellipse4.pdf>`_.
814:         '''
815:         if cls._unit_circle_righthalf is None:
816:             MAGIC = 0.2652031
817:             SQRTHALF = np.sqrt(0.5)
818:             MAGIC45 = SQRTHALF * MAGIC
819: 
820:             vertices = np.array(
821:                 [[0.0, -1.0],
822: 
823:                  [MAGIC, -1.0],
824:                  [SQRTHALF-MAGIC45, -SQRTHALF-MAGIC45],
825:                  [SQRTHALF, -SQRTHALF],
826: 
827:                  [SQRTHALF+MAGIC45, -SQRTHALF+MAGIC45],
828:                  [1.0, -MAGIC],
829:                  [1.0, 0.0],
830: 
831:                  [1.0, MAGIC],
832:                  [SQRTHALF+MAGIC45, SQRTHALF-MAGIC45],
833:                  [SQRTHALF, SQRTHALF],
834: 
835:                  [SQRTHALF-MAGIC45, SQRTHALF+MAGIC45],
836:                  [MAGIC, 1.0],
837:                  [0.0, 1.0],
838: 
839:                  [0.0, -1.0]],
840: 
841:                 float)
842: 
843:             codes = cls.CURVE4 * np.ones(14)
844:             codes[0] = cls.MOVETO
845:             codes[-1] = cls.CLOSEPOLY
846: 
847:             cls._unit_circle_righthalf = cls(vertices, codes, readonly=True)
848:         return cls._unit_circle_righthalf
849: 
850:     @classmethod
851:     def arc(cls, theta1, theta2, n=None, is_wedge=False):
852:         '''
853:         Return an arc on the unit circle from angle
854:         *theta1* to angle *theta2* (in degrees).
855: 
856:         *theta2* is unwrapped to produce the shortest arc within 360 degrees.
857:         That is, if *theta2* > *theta1* + 360, the arc will be from *theta1* to
858:         *theta2* - 360 and not a full circle plus some extra overlap.
859: 
860:         If *n* is provided, it is the number of spline segments to make.
861:         If *n* is not provided, the number of spline segments is
862:         determined based on the delta between *theta1* and *theta2*.
863: 
864:            Masionobe, L.  2003.  `Drawing an elliptical arc using
865:            polylines, quadratic or cubic Bezier curves
866:            <http://www.spaceroots.org/documents/ellipse/index.html>`_.
867:         '''
868:         halfpi = np.pi * 0.5
869: 
870:         eta1 = theta1
871:         eta2 = theta2 - 360 * np.floor((theta2 - theta1) / 360)
872:         # Ensure 2pi range is not flattened to 0 due to floating-point errors,
873:         # but don't try to expand existing 0 range.
874:         if theta2 != theta1 and eta2 <= eta1:
875:             eta2 += 360
876:         eta1, eta2 = np.deg2rad([eta1, eta2])
877: 
878:         # number of curve segments to make
879:         if n is None:
880:             n = int(2 ** np.ceil((eta2 - eta1) / halfpi))
881:         if n < 1:
882:             raise ValueError("n must be >= 1 or None")
883: 
884:         deta = (eta2 - eta1) / n
885:         t = np.tan(0.5 * deta)
886:         alpha = np.sin(deta) * (np.sqrt(4.0 + 3.0 * t * t) - 1) / 3.0
887: 
888:         steps = np.linspace(eta1, eta2, n + 1, True)
889:         cos_eta = np.cos(steps)
890:         sin_eta = np.sin(steps)
891: 
892:         xA = cos_eta[:-1]
893:         yA = sin_eta[:-1]
894:         xA_dot = -yA
895:         yA_dot = xA
896: 
897:         xB = cos_eta[1:]
898:         yB = sin_eta[1:]
899:         xB_dot = -yB
900:         yB_dot = xB
901: 
902:         if is_wedge:
903:             length = n * 3 + 4
904:             vertices = np.zeros((length, 2), float)
905:             codes = cls.CURVE4 * np.ones((length, ), cls.code_type)
906:             vertices[1] = [xA[0], yA[0]]
907:             codes[0:2] = [cls.MOVETO, cls.LINETO]
908:             codes[-2:] = [cls.LINETO, cls.CLOSEPOLY]
909:             vertex_offset = 2
910:             end = length - 2
911:         else:
912:             length = n * 3 + 1
913:             vertices = np.empty((length, 2), float)
914:             codes = cls.CURVE4 * np.ones((length, ), cls.code_type)
915:             vertices[0] = [xA[0], yA[0]]
916:             codes[0] = cls.MOVETO
917:             vertex_offset = 1
918:             end = length
919: 
920:         vertices[vertex_offset:end:3, 0] = xA + alpha * xA_dot
921:         vertices[vertex_offset:end:3, 1] = yA + alpha * yA_dot
922:         vertices[vertex_offset+1:end:3, 0] = xB - alpha * xB_dot
923:         vertices[vertex_offset+1:end:3, 1] = yB - alpha * yB_dot
924:         vertices[vertex_offset+2:end:3, 0] = xB
925:         vertices[vertex_offset+2:end:3, 1] = yB
926: 
927:         return cls(vertices, codes, readonly=True)
928: 
929:     @classmethod
930:     def wedge(cls, theta1, theta2, n=None):
931:         '''
932:         Return a wedge of the unit circle from angle
933:         *theta1* to angle *theta2* (in degrees).
934: 
935:         *theta2* is unwrapped to produce the shortest wedge within 360 degrees.
936:         That is, if *theta2* > *theta1* + 360, the wedge will be from *theta1*
937:         to *theta2* - 360 and not a full circle plus some extra overlap.
938: 
939:         If *n* is provided, it is the number of spline segments to make.
940:         If *n* is not provided, the number of spline segments is
941:         determined based on the delta between *theta1* and *theta2*.
942:         '''
943:         return cls.arc(theta1, theta2, n, True)
944: 
945:     _hatch_dict = maxdict(8)
946: 
947:     @classmethod
948:     def hatch(cls, hatchpattern, density=6):
949:         '''
950:         Given a hatch specifier, *hatchpattern*, generates a Path that
951:         can be used in a repeated hatching pattern.  *density* is the
952:         number of lines per unit square.
953:         '''
954:         from matplotlib.hatch import get_path
955: 
956:         if hatchpattern is None:
957:             return None
958: 
959:         hatch_path = cls._hatch_dict.get((hatchpattern, density))
960:         if hatch_path is not None:
961:             return hatch_path
962: 
963:         hatch_path = get_path(hatchpattern, density)
964:         cls._hatch_dict[(hatchpattern, density)] = hatch_path
965:         return hatch_path
966: 
967:     def clip_to_bbox(self, bbox, inside=True):
968:         '''
969:         Clip the path to the given bounding box.
970: 
971:         The path must be made up of one or more closed polygons.  This
972:         algorithm will not behave correctly for unclosed paths.
973: 
974:         If *inside* is `True`, clip to the inside of the box, otherwise
975:         to the outside of the box.
976:         '''
977:         # Use make_compound_path_from_polys
978:         verts = _path.clip_path_to_rect(self, bbox, inside)
979:         paths = [Path(poly) for poly in verts]
980:         return self.make_compound_path(*paths)
981: 
982: 
983: def get_path_collection_extents(
984:         master_transform, paths, transforms, offsets, offset_transform):
985:     '''
986:     Given a sequence of :class:`Path` objects,
987:     :class:`~matplotlib.transforms.Transform` objects and offsets, as
988:     found in a :class:`~matplotlib.collections.PathCollection`,
989:     returns the bounding box that encapsulates all of them.
990: 
991:     *master_transform* is a global transformation to apply to all paths
992: 
993:     *paths* is a sequence of :class:`Path` instances.
994: 
995:     *transforms* is a sequence of
996:     :class:`~matplotlib.transforms.Affine2D` instances.
997: 
998:     *offsets* is a sequence of (x, y) offsets (or an Nx2 array)
999: 
1000:     *offset_transform* is a :class:`~matplotlib.transforms.Affine2D`
1001:     to apply to the offsets before applying the offset to the path.
1002: 
1003:     The way that *paths*, *transforms* and *offsets* are combined
1004:     follows the same method as for collections.  Each is iterated over
1005:     independently, so if you have 3 paths, 2 transforms and 1 offset,
1006:     their combinations are as follows:
1007: 
1008:         (A, A, A), (B, B, A), (C, A, A)
1009:     '''
1010:     from .transforms import Bbox
1011:     if len(paths) == 0:
1012:         raise ValueError("No paths provided")
1013:     return Bbox.from_extents(*_path.get_path_collection_extents(
1014:         master_transform, paths, np.atleast_3d(transforms),
1015:         offsets, offset_transform))
1016: 
1017: 
1018: def get_paths_extents(paths, transforms=[]):
1019:     '''
1020:     Given a sequence of :class:`Path` objects and optional
1021:     :class:`~matplotlib.transforms.Transform` objects, returns the
1022:     bounding box that encapsulates all of them.
1023: 
1024:     *paths* is a sequence of :class:`Path` instances.
1025: 
1026:     *transforms* is an optional sequence of
1027:     :class:`~matplotlib.transforms.Affine2D` instances to apply to
1028:     each path.
1029:     '''
1030:     from .transforms import Bbox, Affine2D
1031:     if len(paths) == 0:
1032:         raise ValueError("No paths provided")
1033:     return Bbox.from_extents(*_path.get_path_collection_extents(
1034:         Affine2D(), paths, transforms, [], Affine2D()))
1035: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_111656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'unicode', u'\nA module for dealing with the polylines used throughout matplotlib.\n\nThe primary class for polyline handling in matplotlib is :class:`Path`.\nAlmost all vector drawing makes use of Paths somewhere in the drawing\npipeline.\n\nWhilst a :class:`Path` instance itself cannot be drawn, there exists\n:class:`~matplotlib.artist.Artist` subclasses which can be used for\nconvenient Path visualisation - the two most frequently used of these are\n:class:`~matplotlib.patches.PathPatch` and\n:class:`~matplotlib.collections.PathCollection`.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import six' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_111657 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'six')

if (type(import_111657) is not StypyTypeError):

    if (import_111657 != 'pyd_module'):
        __import__(import_111657)
        sys_modules_111658 = sys.modules[import_111657]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'six', sys_modules_111658.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'six', import_111657)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import math' statement (line 20)
import math

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from weakref import WeakValueDictionary' statement (line 21)
try:
    from weakref import WeakValueDictionary

except:
    WeakValueDictionary = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'weakref', None, module_type_store, ['WeakValueDictionary'], [WeakValueDictionary])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import numpy' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_111659 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy')

if (type(import_111659) is not StypyTypeError):

    if (import_111659 != 'pyd_module'):
        __import__(import_111659)
        sys_modules_111660 = sys.modules[import_111659]
        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'np', sys_modules_111660.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy', import_111659)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from matplotlib import _path, rcParams' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_111661 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib')

if (type(import_111661) is not StypyTypeError):

    if (import_111661 != 'pyd_module'):
        __import__(import_111661)
        sys_modules_111662 = sys.modules[import_111661]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', sys_modules_111662.module_type_store, module_type_store, ['_path', 'rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_111662, sys_modules_111662.module_type_store, module_type_store)
    else:
        from matplotlib import _path, rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', None, module_type_store, ['_path', 'rcParams'], [_path, rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib', import_111661)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from matplotlib.cbook import _to_unmasked_float_array, simple_linear_interpolation, maxdict' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_111663 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.cbook')

if (type(import_111663) is not StypyTypeError):

    if (import_111663 != 'pyd_module'):
        __import__(import_111663)
        sys_modules_111664 = sys.modules[import_111663]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.cbook', sys_modules_111664.module_type_store, module_type_store, ['_to_unmasked_float_array', 'simple_linear_interpolation', 'maxdict'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_111664, sys_modules_111664.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import _to_unmasked_float_array, simple_linear_interpolation, maxdict

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.cbook', None, module_type_store, ['_to_unmasked_float_array', 'simple_linear_interpolation', 'maxdict'], [_to_unmasked_float_array, simple_linear_interpolation, maxdict])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.cbook', import_111663)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'Path' class

class Path(object, ):
    unicode_111665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, (-1)), 'unicode', u"\n    :class:`Path` represents a series of possibly disconnected,\n    possibly closed, line and curve segments.\n\n    The underlying storage is made up of two parallel numpy arrays:\n      - *vertices*: an Nx2 float array of vertices\n      - *codes*: an N-length uint8 array of vertex types\n\n    These two arrays always have the same length in the first\n    dimension.  For example, to represent a cubic curve, you must\n    provide three vertices as well as three codes ``CURVE3``.\n\n    The code types are:\n\n       - ``STOP``   :  1 vertex (ignored)\n           A marker for the end of the entire path (currently not\n           required and ignored)\n\n       - ``MOVETO`` :  1 vertex\n            Pick up the pen and move to the given vertex.\n\n       - ``LINETO`` :  1 vertex\n            Draw a line from the current position to the given vertex.\n\n       - ``CURVE3`` :  1 control point, 1 endpoint\n          Draw a quadratic Bezier curve from the current position,\n          with the given control point, to the given end point.\n\n       - ``CURVE4`` :  2 control points, 1 endpoint\n          Draw a cubic Bezier curve from the current position, with\n          the given control points, to the given end point.\n\n       - ``CLOSEPOLY`` : 1 vertex (ignored)\n          Draw a line segment to the start point of the current\n          polyline.\n\n    Users of Path objects should not access the vertices and codes\n    arrays directly.  Instead, they should use :meth:`iter_segments`\n    or :meth:`cleaned` to get the vertex/code pairs.  This is important,\n    since many :class:`Path` objects, as an optimization, do not store a\n    *codes* at all, but have a default one provided for them by\n    :meth:`iter_segments`.\n\n    Some behavior of Path objects can be controlled by rcParams. See\n    the rcParams whose keys contain 'path.'.\n\n    .. note::\n\n        The vertices and codes arrays should be treated as\n        immutable -- there are a number of optimizations and assumptions\n        made up front in the constructor that will not change when the\n        data changes.\n\n    ")
    
    # Assigning a Num to a Name (line 87):
    
    # Assigning a Num to a Name (line 88):
    
    # Assigning a Num to a Name (line 89):
    
    # Assigning a Num to a Name (line 90):
    
    # Assigning a Num to a Name (line 91):
    
    # Assigning a Num to a Name (line 92):
    
    # Assigning a Dict to a Name (line 96):
    
    # Assigning a Attribute to a Name (line 103):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 105)
        None_111666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 39), 'None')
        int_111667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 66), 'int')
        # Getting the type of 'False' (line 106)
        False_111668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'False')
        # Getting the type of 'False' (line 106)
        False_111669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'False')
        defaults = [None_111666, int_111667, False_111668, False_111669]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.__init__', ['vertices', 'codes', '_interpolation_steps', 'closed', 'readonly'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['vertices', 'codes', '_interpolation_steps', 'closed', 'readonly'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_111670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, (-1)), 'unicode', u'\n        Create a new path with the given vertices and codes.\n\n        Parameters\n        ----------\n        vertices : array_like\n            The ``(n, 2)`` float array, masked array or sequence of pairs\n            representing the vertices of the path.\n\n            If *vertices* contains masked values, they will be converted\n            to NaNs which are then handled correctly by the Agg\n            PathIterator and other consumers of path data, such as\n            :meth:`iter_segments`.\n        codes : {None, array_like}, optional\n            n-length array integers representing the codes of the path.\n            If not None, codes must be the same length as vertices.\n            If None, *vertices* will be treated as a series of line segments.\n        _interpolation_steps : int, optional\n            Used as a hint to certain projections, such as Polar, that this\n            path should be linearly interpolated immediately before drawing.\n            This attribute is primarily an implementation detail and is not\n            intended for public use.\n        closed : bool, optional\n            If *codes* is None and closed is True, vertices will be treated as\n            line segments of a closed polygon.\n        readonly : bool, optional\n            Makes the path behave in an immutable way and sets the vertices\n            and codes as read-only arrays.\n        ')
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to _to_unmasked_float_array(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'vertices' (line 136)
        vertices_111672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 44), 'vertices', False)
        # Processing the call keyword arguments (line 136)
        kwargs_111673 = {}
        # Getting the type of '_to_unmasked_float_array' (line 136)
        _to_unmasked_float_array_111671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), '_to_unmasked_float_array', False)
        # Calling _to_unmasked_float_array(args, kwargs) (line 136)
        _to_unmasked_float_array_call_result_111674 = invoke(stypy.reporting.localization.Localization(__file__, 136, 19), _to_unmasked_float_array_111671, *[vertices_111672], **kwargs_111673)
        
        # Assigning a type to the variable 'vertices' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'vertices', _to_unmasked_float_array_call_result_111674)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'vertices' (line 137)
        vertices_111675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'vertices')
        # Obtaining the member 'ndim' of a type (line 137)
        ndim_111676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), vertices_111675, 'ndim')
        int_111677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 29), 'int')
        # Applying the binary operator '!=' (line 137)
        result_ne_111678 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 12), '!=', ndim_111676, int_111677)
        
        
        
        # Obtaining the type of the subscript
        int_111679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 51), 'int')
        # Getting the type of 'vertices' (line 137)
        vertices_111680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'vertices')
        # Obtaining the member 'shape' of a type (line 137)
        shape_111681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 36), vertices_111680, 'shape')
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___111682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 36), shape_111681, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_111683 = invoke(stypy.reporting.localization.Localization(__file__, 137, 36), getitem___111682, int_111679)
        
        int_111684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 57), 'int')
        # Applying the binary operator '!=' (line 137)
        result_ne_111685 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 36), '!=', subscript_call_result_111683, int_111684)
        
        # Applying the binary operator 'or' (line 137)
        result_or_keyword_111686 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 11), 'or', result_ne_111678, result_ne_111685)
        
        # Testing the type of an if condition (line 137)
        if_condition_111687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), result_or_keyword_111686)
        # Assigning a type to the variable 'if_condition_111687' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_111687', if_condition_111687)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 138):
        
        # Assigning a Str to a Name (line 138):
        unicode_111688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 18), 'unicode', u"'vertices' must be a 2D list or array with shape Nx2")
        # Assigning a type to the variable 'msg' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'msg', unicode_111688)
        
        # Call to ValueError(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'msg' (line 139)
        msg_111690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 29), 'msg', False)
        # Processing the call keyword arguments (line 139)
        kwargs_111691 = {}
        # Getting the type of 'ValueError' (line 139)
        ValueError_111689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 139)
        ValueError_call_result_111692 = invoke(stypy.reporting.localization.Localization(__file__, 139, 18), ValueError_111689, *[msg_111690], **kwargs_111691)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 139, 12), ValueError_call_result_111692, 'raise parameter', BaseException)
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 141)
        # Getting the type of 'codes' (line 141)
        codes_111693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'codes')
        # Getting the type of 'None' (line 141)
        None_111694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'None')
        
        (may_be_111695, more_types_in_union_111696) = may_not_be_none(codes_111693, None_111694)

        if may_be_111695:

            if more_types_in_union_111696:
                # Runtime conditional SSA (line 141)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 142):
            
            # Assigning a Call to a Name (line 142):
            
            # Call to asarray(...): (line 142)
            # Processing the call arguments (line 142)
            # Getting the type of 'codes' (line 142)
            codes_111699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 'codes', False)
            # Getting the type of 'self' (line 142)
            self_111700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 38), 'self', False)
            # Obtaining the member 'code_type' of a type (line 142)
            code_type_111701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 38), self_111700, 'code_type')
            # Processing the call keyword arguments (line 142)
            kwargs_111702 = {}
            # Getting the type of 'np' (line 142)
            np_111697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'np', False)
            # Obtaining the member 'asarray' of a type (line 142)
            asarray_111698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 20), np_111697, 'asarray')
            # Calling asarray(args, kwargs) (line 142)
            asarray_call_result_111703 = invoke(stypy.reporting.localization.Localization(__file__, 142, 20), asarray_111698, *[codes_111699, code_type_111701], **kwargs_111702)
            
            # Assigning a type to the variable 'codes' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'codes', asarray_call_result_111703)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'codes' (line 143)
            codes_111704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'codes')
            # Obtaining the member 'ndim' of a type (line 143)
            ndim_111705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), codes_111704, 'ndim')
            int_111706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 30), 'int')
            # Applying the binary operator '!=' (line 143)
            result_ne_111707 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 16), '!=', ndim_111705, int_111706)
            
            
            
            # Call to len(...): (line 143)
            # Processing the call arguments (line 143)
            # Getting the type of 'codes' (line 143)
            codes_111709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 40), 'codes', False)
            # Processing the call keyword arguments (line 143)
            kwargs_111710 = {}
            # Getting the type of 'len' (line 143)
            len_111708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 36), 'len', False)
            # Calling len(args, kwargs) (line 143)
            len_call_result_111711 = invoke(stypy.reporting.localization.Localization(__file__, 143, 36), len_111708, *[codes_111709], **kwargs_111710)
            
            
            # Call to len(...): (line 143)
            # Processing the call arguments (line 143)
            # Getting the type of 'vertices' (line 143)
            vertices_111713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 54), 'vertices', False)
            # Processing the call keyword arguments (line 143)
            kwargs_111714 = {}
            # Getting the type of 'len' (line 143)
            len_111712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 50), 'len', False)
            # Calling len(args, kwargs) (line 143)
            len_call_result_111715 = invoke(stypy.reporting.localization.Localization(__file__, 143, 50), len_111712, *[vertices_111713], **kwargs_111714)
            
            # Applying the binary operator '!=' (line 143)
            result_ne_111716 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 36), '!=', len_call_result_111711, len_call_result_111715)
            
            # Applying the binary operator 'or' (line 143)
            result_or_keyword_111717 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 15), 'or', result_ne_111707, result_ne_111716)
            
            # Testing the type of an if condition (line 143)
            if_condition_111718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 12), result_or_keyword_111717)
            # Assigning a type to the variable 'if_condition_111718' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'if_condition_111718', if_condition_111718)
            # SSA begins for if statement (line 143)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 144):
            
            # Assigning a Str to a Name (line 144):
            unicode_111719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'unicode', u"'codes' must be a 1D list or array with the same length of 'vertices'")
            # Assigning a type to the variable 'msg' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'msg', unicode_111719)
            
            # Call to ValueError(...): (line 146)
            # Processing the call arguments (line 146)
            # Getting the type of 'msg' (line 146)
            msg_111721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 33), 'msg', False)
            # Processing the call keyword arguments (line 146)
            kwargs_111722 = {}
            # Getting the type of 'ValueError' (line 146)
            ValueError_111720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 146)
            ValueError_call_result_111723 = invoke(stypy.reporting.localization.Localization(__file__, 146, 22), ValueError_111720, *[msg_111721], **kwargs_111722)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 146, 16), ValueError_call_result_111723, 'raise parameter', BaseException)
            # SSA join for if statement (line 143)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Evaluating a boolean operation
            
            # Call to len(...): (line 147)
            # Processing the call arguments (line 147)
            # Getting the type of 'codes' (line 147)
            codes_111725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'codes', False)
            # Processing the call keyword arguments (line 147)
            kwargs_111726 = {}
            # Getting the type of 'len' (line 147)
            len_111724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'len', False)
            # Calling len(args, kwargs) (line 147)
            len_call_result_111727 = invoke(stypy.reporting.localization.Localization(__file__, 147, 15), len_111724, *[codes_111725], **kwargs_111726)
            
            
            
            # Obtaining the type of the subscript
            int_111728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 36), 'int')
            # Getting the type of 'codes' (line 147)
            codes_111729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 30), 'codes')
            # Obtaining the member '__getitem__' of a type (line 147)
            getitem___111730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 30), codes_111729, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 147)
            subscript_call_result_111731 = invoke(stypy.reporting.localization.Localization(__file__, 147, 30), getitem___111730, int_111728)
            
            # Getting the type of 'self' (line 147)
            self_111732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 42), 'self')
            # Obtaining the member 'MOVETO' of a type (line 147)
            MOVETO_111733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 42), self_111732, 'MOVETO')
            # Applying the binary operator '!=' (line 147)
            result_ne_111734 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 30), '!=', subscript_call_result_111731, MOVETO_111733)
            
            # Applying the binary operator 'and' (line 147)
            result_and_keyword_111735 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 15), 'and', len_call_result_111727, result_ne_111734)
            
            # Testing the type of an if condition (line 147)
            if_condition_111736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 12), result_and_keyword_111735)
            # Assigning a type to the variable 'if_condition_111736' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'if_condition_111736', if_condition_111736)
            # SSA begins for if statement (line 147)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 148):
            
            # Assigning a Str to a Name (line 148):
            unicode_111737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 23), 'unicode', u"The first element of 'code' must be equal to 'MOVETO': {0}")
            # Assigning a type to the variable 'msg' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'msg', unicode_111737)
            
            # Call to ValueError(...): (line 150)
            # Processing the call arguments (line 150)
            
            # Call to format(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'self' (line 150)
            self_111741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 44), 'self', False)
            # Obtaining the member 'MOVETO' of a type (line 150)
            MOVETO_111742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 44), self_111741, 'MOVETO')
            # Processing the call keyword arguments (line 150)
            kwargs_111743 = {}
            # Getting the type of 'msg' (line 150)
            msg_111739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 33), 'msg', False)
            # Obtaining the member 'format' of a type (line 150)
            format_111740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 33), msg_111739, 'format')
            # Calling format(args, kwargs) (line 150)
            format_call_result_111744 = invoke(stypy.reporting.localization.Localization(__file__, 150, 33), format_111740, *[MOVETO_111742], **kwargs_111743)
            
            # Processing the call keyword arguments (line 150)
            kwargs_111745 = {}
            # Getting the type of 'ValueError' (line 150)
            ValueError_111738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 150)
            ValueError_call_result_111746 = invoke(stypy.reporting.localization.Localization(__file__, 150, 22), ValueError_111738, *[format_call_result_111744], **kwargs_111745)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 150, 16), ValueError_call_result_111746, 'raise parameter', BaseException)
            # SSA join for if statement (line 147)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_111696:
                # Runtime conditional SSA for else branch (line 141)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_111695) or more_types_in_union_111696):
            
            # Getting the type of 'closed' (line 151)
            closed_111747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'closed')
            # Testing the type of an if condition (line 151)
            if_condition_111748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 13), closed_111747)
            # Assigning a type to the variable 'if_condition_111748' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'if_condition_111748', if_condition_111748)
            # SSA begins for if statement (line 151)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 152):
            
            # Assigning a Call to a Name (line 152):
            
            # Call to empty(...): (line 152)
            # Processing the call arguments (line 152)
            
            # Call to len(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'vertices' (line 152)
            vertices_111752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 33), 'vertices', False)
            # Processing the call keyword arguments (line 152)
            kwargs_111753 = {}
            # Getting the type of 'len' (line 152)
            len_111751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 29), 'len', False)
            # Calling len(args, kwargs) (line 152)
            len_call_result_111754 = invoke(stypy.reporting.localization.Localization(__file__, 152, 29), len_111751, *[vertices_111752], **kwargs_111753)
            
            # Processing the call keyword arguments (line 152)
            # Getting the type of 'self' (line 152)
            self_111755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 50), 'self', False)
            # Obtaining the member 'code_type' of a type (line 152)
            code_type_111756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 50), self_111755, 'code_type')
            keyword_111757 = code_type_111756
            kwargs_111758 = {'dtype': keyword_111757}
            # Getting the type of 'np' (line 152)
            np_111749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'np', False)
            # Obtaining the member 'empty' of a type (line 152)
            empty_111750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 20), np_111749, 'empty')
            # Calling empty(args, kwargs) (line 152)
            empty_call_result_111759 = invoke(stypy.reporting.localization.Localization(__file__, 152, 20), empty_111750, *[len_call_result_111754], **kwargs_111758)
            
            # Assigning a type to the variable 'codes' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'codes', empty_call_result_111759)
            
            # Assigning a Attribute to a Subscript (line 153):
            
            # Assigning a Attribute to a Subscript (line 153):
            # Getting the type of 'self' (line 153)
            self_111760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'self')
            # Obtaining the member 'MOVETO' of a type (line 153)
            MOVETO_111761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 23), self_111760, 'MOVETO')
            # Getting the type of 'codes' (line 153)
            codes_111762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'codes')
            int_111763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 18), 'int')
            # Storing an element on a container (line 153)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 12), codes_111762, (int_111763, MOVETO_111761))
            
            # Assigning a Attribute to a Subscript (line 154):
            
            # Assigning a Attribute to a Subscript (line 154):
            # Getting the type of 'self' (line 154)
            self_111764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'self')
            # Obtaining the member 'LINETO' of a type (line 154)
            LINETO_111765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 26), self_111764, 'LINETO')
            # Getting the type of 'codes' (line 154)
            codes_111766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'codes')
            int_111767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 18), 'int')
            int_111768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'int')
            slice_111769 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 154, 12), int_111767, int_111768, None)
            # Storing an element on a container (line 154)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 12), codes_111766, (slice_111769, LINETO_111765))
            
            # Assigning a Attribute to a Subscript (line 155):
            
            # Assigning a Attribute to a Subscript (line 155):
            # Getting the type of 'self' (line 155)
            self_111770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'self')
            # Obtaining the member 'CLOSEPOLY' of a type (line 155)
            CLOSEPOLY_111771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 24), self_111770, 'CLOSEPOLY')
            # Getting the type of 'codes' (line 155)
            codes_111772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'codes')
            int_111773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 18), 'int')
            # Storing an element on a container (line 155)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 12), codes_111772, (int_111773, CLOSEPOLY_111771))
            # SSA join for if statement (line 151)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_111695 and more_types_in_union_111696):
                # SSA join for if statement (line 141)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 157):
        
        # Assigning a Name to a Attribute (line 157):
        # Getting the type of 'vertices' (line 157)
        vertices_111774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 25), 'vertices')
        # Getting the type of 'self' (line 157)
        self_111775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member '_vertices' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_111775, '_vertices', vertices_111774)
        
        # Assigning a Name to a Attribute (line 158):
        
        # Assigning a Name to a Attribute (line 158):
        # Getting the type of 'codes' (line 158)
        codes_111776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'codes')
        # Getting the type of 'self' (line 158)
        self_111777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self')
        # Setting the type of the member '_codes' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_111777, '_codes', codes_111776)
        
        # Assigning a Name to a Attribute (line 159):
        
        # Assigning a Name to a Attribute (line 159):
        # Getting the type of '_interpolation_steps' (line 159)
        _interpolation_steps_111778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 36), '_interpolation_steps')
        # Getting the type of 'self' (line 159)
        self_111779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self')
        # Setting the type of the member '_interpolation_steps' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_111779, '_interpolation_steps', _interpolation_steps_111778)
        
        # Call to _update_values(...): (line 160)
        # Processing the call keyword arguments (line 160)
        kwargs_111782 = {}
        # Getting the type of 'self' (line 160)
        self_111780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self', False)
        # Obtaining the member '_update_values' of a type (line 160)
        _update_values_111781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_111780, '_update_values')
        # Calling _update_values(args, kwargs) (line 160)
        _update_values_call_result_111783 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), _update_values_111781, *[], **kwargs_111782)
        
        
        # Getting the type of 'readonly' (line 162)
        readonly_111784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'readonly')
        # Testing the type of an if condition (line 162)
        if_condition_111785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 8), readonly_111784)
        # Assigning a type to the variable 'if_condition_111785' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'if_condition_111785', if_condition_111785)
        # SSA begins for if statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 163):
        
        # Assigning a Name to a Attribute (line 163):
        # Getting the type of 'False' (line 163)
        False_111786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 45), 'False')
        # Getting the type of 'self' (line 163)
        self_111787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'self')
        # Obtaining the member '_vertices' of a type (line 163)
        _vertices_111788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), self_111787, '_vertices')
        # Obtaining the member 'flags' of a type (line 163)
        flags_111789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), _vertices_111788, 'flags')
        # Setting the type of the member 'writeable' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), flags_111789, 'writeable', False_111786)
        
        
        # Getting the type of 'self' (line 164)
        self_111790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'self')
        # Obtaining the member '_codes' of a type (line 164)
        _codes_111791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 15), self_111790, '_codes')
        # Getting the type of 'None' (line 164)
        None_111792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'None')
        # Applying the binary operator 'isnot' (line 164)
        result_is_not_111793 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 15), 'isnot', _codes_111791, None_111792)
        
        # Testing the type of an if condition (line 164)
        if_condition_111794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 12), result_is_not_111793)
        # Assigning a type to the variable 'if_condition_111794' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'if_condition_111794', if_condition_111794)
        # SSA begins for if statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 165):
        
        # Assigning a Name to a Attribute (line 165):
        # Getting the type of 'False' (line 165)
        False_111795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 46), 'False')
        # Getting the type of 'self' (line 165)
        self_111796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'self')
        # Obtaining the member '_codes' of a type (line 165)
        _codes_111797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), self_111796, '_codes')
        # Obtaining the member 'flags' of a type (line 165)
        flags_111798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), _codes_111797, 'flags')
        # Setting the type of the member 'writeable' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), flags_111798, 'writeable', False_111795)
        # SSA join for if statement (line 164)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 166):
        
        # Assigning a Name to a Attribute (line 166):
        # Getting the type of 'True' (line 166)
        True_111799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 29), 'True')
        # Getting the type of 'self' (line 166)
        self_111800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'self')
        # Setting the type of the member '_readonly' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), self_111800, '_readonly', True_111799)
        # SSA branch for the else part of an if statement (line 162)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 168):
        
        # Assigning a Name to a Attribute (line 168):
        # Getting the type of 'False' (line 168)
        False_111801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 29), 'False')
        # Getting the type of 'self' (line 168)
        self_111802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'self')
        # Setting the type of the member '_readonly' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), self_111802, '_readonly', False_111801)
        # SSA join for if statement (line 162)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _fast_from_codes_and_verts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 171)
        None_111803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 64), 'None')
        defaults = [None_111803]
        # Create a new context for function '_fast_from_codes_and_verts'
        module_type_store = module_type_store.open_function_context('_fast_from_codes_and_verts', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path._fast_from_codes_and_verts.__dict__.__setitem__('stypy_localization', localization)
        Path._fast_from_codes_and_verts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path._fast_from_codes_and_verts.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path._fast_from_codes_and_verts.__dict__.__setitem__('stypy_function_name', 'Path._fast_from_codes_and_verts')
        Path._fast_from_codes_and_verts.__dict__.__setitem__('stypy_param_names_list', ['verts', 'codes', 'internals'])
        Path._fast_from_codes_and_verts.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path._fast_from_codes_and_verts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path._fast_from_codes_and_verts.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path._fast_from_codes_and_verts.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path._fast_from_codes_and_verts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path._fast_from_codes_and_verts.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path._fast_from_codes_and_verts', ['verts', 'codes', 'internals'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fast_from_codes_and_verts', localization, ['verts', 'codes', 'internals'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fast_from_codes_and_verts(...)' code ##################

        unicode_111804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, (-1)), 'unicode', u'\n        Creates a Path instance without the expense of calling the constructor\n\n        Parameters\n        ----------\n        verts : numpy array\n        codes : numpy array\n        internals : dict or None\n            The attributes that the resulting path should have.\n            Allowed keys are ``readonly``, ``should_simplify``,\n            ``simplify_threshold``, ``has_nonfinite`` and\n            ``interpolation_steps``.\n\n        ')
        
        # Assigning a BoolOp to a Name (line 186):
        
        # Assigning a BoolOp to a Name (line 186):
        
        # Evaluating a boolean operation
        # Getting the type of 'internals' (line 186)
        internals_111805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'internals')
        
        # Obtaining an instance of the builtin type 'dict' (line 186)
        dict_111806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 186)
        
        # Applying the binary operator 'or' (line 186)
        result_or_keyword_111807 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 20), 'or', internals_111805, dict_111806)
        
        # Assigning a type to the variable 'internals' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'internals', result_or_keyword_111807)
        
        # Assigning a Call to a Name (line 187):
        
        # Assigning a Call to a Name (line 187):
        
        # Call to __new__(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'cls' (line 187)
        cls_111810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 26), 'cls', False)
        # Processing the call keyword arguments (line 187)
        kwargs_111811 = {}
        # Getting the type of 'cls' (line 187)
        cls_111808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'cls', False)
        # Obtaining the member '__new__' of a type (line 187)
        new___111809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 14), cls_111808, '__new__')
        # Calling __new__(args, kwargs) (line 187)
        new___call_result_111812 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), new___111809, *[cls_111810], **kwargs_111811)
        
        # Assigning a type to the variable 'pth' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'pth', new___call_result_111812)
        
        # Assigning a Call to a Attribute (line 188):
        
        # Assigning a Call to a Attribute (line 188):
        
        # Call to _to_unmasked_float_array(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'verts' (line 188)
        verts_111814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 49), 'verts', False)
        # Processing the call keyword arguments (line 188)
        kwargs_111815 = {}
        # Getting the type of '_to_unmasked_float_array' (line 188)
        _to_unmasked_float_array_111813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), '_to_unmasked_float_array', False)
        # Calling _to_unmasked_float_array(args, kwargs) (line 188)
        _to_unmasked_float_array_call_result_111816 = invoke(stypy.reporting.localization.Localization(__file__, 188, 24), _to_unmasked_float_array_111813, *[verts_111814], **kwargs_111815)
        
        # Getting the type of 'pth' (line 188)
        pth_111817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'pth')
        # Setting the type of the member '_vertices' of a type (line 188)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), pth_111817, '_vertices', _to_unmasked_float_array_call_result_111816)
        
        # Assigning a Name to a Attribute (line 189):
        
        # Assigning a Name to a Attribute (line 189):
        # Getting the type of 'codes' (line 189)
        codes_111818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), 'codes')
        # Getting the type of 'pth' (line 189)
        pth_111819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'pth')
        # Setting the type of the member '_codes' of a type (line 189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), pth_111819, '_codes', codes_111818)
        
        # Assigning a Call to a Attribute (line 190):
        
        # Assigning a Call to a Attribute (line 190):
        
        # Call to pop(...): (line 190)
        # Processing the call arguments (line 190)
        unicode_111822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 38), 'unicode', u'readonly')
        # Getting the type of 'False' (line 190)
        False_111823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 50), 'False', False)
        # Processing the call keyword arguments (line 190)
        kwargs_111824 = {}
        # Getting the type of 'internals' (line 190)
        internals_111820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 24), 'internals', False)
        # Obtaining the member 'pop' of a type (line 190)
        pop_111821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 24), internals_111820, 'pop')
        # Calling pop(args, kwargs) (line 190)
        pop_call_result_111825 = invoke(stypy.reporting.localization.Localization(__file__, 190, 24), pop_111821, *[unicode_111822, False_111823], **kwargs_111824)
        
        # Getting the type of 'pth' (line 190)
        pth_111826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'pth')
        # Setting the type of the member '_readonly' of a type (line 190)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), pth_111826, '_readonly', pop_call_result_111825)
        
        # Assigning a Call to a Attribute (line 191):
        
        # Assigning a Call to a Attribute (line 191):
        
        # Call to pop(...): (line 191)
        # Processing the call arguments (line 191)
        unicode_111829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 44), 'unicode', u'should_simplify')
        # Getting the type of 'True' (line 191)
        True_111830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 63), 'True', False)
        # Processing the call keyword arguments (line 191)
        kwargs_111831 = {}
        # Getting the type of 'internals' (line 191)
        internals_111827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'internals', False)
        # Obtaining the member 'pop' of a type (line 191)
        pop_111828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 30), internals_111827, 'pop')
        # Calling pop(args, kwargs) (line 191)
        pop_call_result_111832 = invoke(stypy.reporting.localization.Localization(__file__, 191, 30), pop_111828, *[unicode_111829, True_111830], **kwargs_111831)
        
        # Getting the type of 'pth' (line 191)
        pth_111833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'pth')
        # Setting the type of the member 'should_simplify' of a type (line 191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), pth_111833, 'should_simplify', pop_call_result_111832)
        
        # Assigning a Call to a Attribute (line 192):
        
        # Assigning a Call to a Attribute (line 192):
        
        # Call to pop(...): (line 193)
        # Processing the call arguments (line 193)
        unicode_111836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 26), 'unicode', u'simplify_threshold')
        
        # Obtaining the type of the subscript
        unicode_111837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 35), 'unicode', u'path.simplify_threshold')
        # Getting the type of 'rcParams' (line 194)
        rcParams_111838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 26), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 194)
        getitem___111839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 26), rcParams_111838, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 194)
        subscript_call_result_111840 = invoke(stypy.reporting.localization.Localization(__file__, 194, 26), getitem___111839, unicode_111837)
        
        # Processing the call keyword arguments (line 193)
        kwargs_111841 = {}
        # Getting the type of 'internals' (line 193)
        internals_111834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'internals', False)
        # Obtaining the member 'pop' of a type (line 193)
        pop_111835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), internals_111834, 'pop')
        # Calling pop(args, kwargs) (line 193)
        pop_call_result_111842 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), pop_111835, *[unicode_111836, subscript_call_result_111840], **kwargs_111841)
        
        # Getting the type of 'pth' (line 192)
        pth_111843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'pth')
        # Setting the type of the member 'simplify_threshold' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), pth_111843, 'simplify_threshold', pop_call_result_111842)
        
        # Assigning a Call to a Attribute (line 196):
        
        # Assigning a Call to a Attribute (line 196):
        
        # Call to pop(...): (line 196)
        # Processing the call arguments (line 196)
        unicode_111846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 43), 'unicode', u'has_nonfinite')
        # Getting the type of 'False' (line 196)
        False_111847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 60), 'False', False)
        # Processing the call keyword arguments (line 196)
        kwargs_111848 = {}
        # Getting the type of 'internals' (line 196)
        internals_111844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 29), 'internals', False)
        # Obtaining the member 'pop' of a type (line 196)
        pop_111845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 29), internals_111844, 'pop')
        # Calling pop(args, kwargs) (line 196)
        pop_call_result_111849 = invoke(stypy.reporting.localization.Localization(__file__, 196, 29), pop_111845, *[unicode_111846, False_111847], **kwargs_111848)
        
        # Getting the type of 'pth' (line 196)
        pth_111850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'pth')
        # Setting the type of the member '_has_nonfinite' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), pth_111850, '_has_nonfinite', pop_call_result_111849)
        
        # Assigning a Call to a Attribute (line 197):
        
        # Assigning a Call to a Attribute (line 197):
        
        # Call to pop(...): (line 197)
        # Processing the call arguments (line 197)
        unicode_111853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 49), 'unicode', u'interpolation_steps')
        int_111854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 72), 'int')
        # Processing the call keyword arguments (line 197)
        kwargs_111855 = {}
        # Getting the type of 'internals' (line 197)
        internals_111851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 35), 'internals', False)
        # Obtaining the member 'pop' of a type (line 197)
        pop_111852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 35), internals_111851, 'pop')
        # Calling pop(args, kwargs) (line 197)
        pop_call_result_111856 = invoke(stypy.reporting.localization.Localization(__file__, 197, 35), pop_111852, *[unicode_111853, int_111854], **kwargs_111855)
        
        # Getting the type of 'pth' (line 197)
        pth_111857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'pth')
        # Setting the type of the member '_interpolation_steps' of a type (line 197)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), pth_111857, '_interpolation_steps', pop_call_result_111856)
        
        # Getting the type of 'internals' (line 198)
        internals_111858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'internals')
        # Testing the type of an if condition (line 198)
        if_condition_111859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), internals_111858)
        # Assigning a type to the variable 'if_condition_111859' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_111859', if_condition_111859)
        # SSA begins for if statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Call to format(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Call to join(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'internals' (line 201)
        internals_111865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 54), 'internals', False)
        # Processing the call keyword arguments (line 201)
        kwargs_111866 = {}
        unicode_111863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 42), 'unicode', u'\n *')
        # Obtaining the member 'join' of a type (line 201)
        join_111864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 42), unicode_111863, 'join')
        # Calling join(args, kwargs) (line 201)
        join_call_result_111867 = invoke(stypy.reporting.localization.Localization(__file__, 201, 42), join_111864, *[internals_111865], **kwargs_111866)
        
        # Processing the call keyword arguments (line 199)
        kwargs_111868 = {}
        unicode_111861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'unicode', u'Unexpected internals provided to _fast_from_codes_and_verts: {0}')
        # Obtaining the member 'format' of a type (line 199)
        format_111862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 29), unicode_111861, 'format')
        # Calling format(args, kwargs) (line 199)
        format_call_result_111869 = invoke(stypy.reporting.localization.Localization(__file__, 199, 29), format_111862, *[join_call_result_111867], **kwargs_111868)
        
        # Processing the call keyword arguments (line 199)
        kwargs_111870 = {}
        # Getting the type of 'ValueError' (line 199)
        ValueError_111860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 199)
        ValueError_call_result_111871 = invoke(stypy.reporting.localization.Localization(__file__, 199, 18), ValueError_111860, *[format_call_result_111869], **kwargs_111870)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 199, 12), ValueError_call_result_111871, 'raise parameter', BaseException)
        # SSA join for if statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'pth' (line 202)
        pth_111872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 15), 'pth')
        # Assigning a type to the variable 'stypy_return_type' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'stypy_return_type', pth_111872)
        
        # ################# End of '_fast_from_codes_and_verts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fast_from_codes_and_verts' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_111873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111873)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fast_from_codes_and_verts'
        return stypy_return_type_111873


    @norecursion
    def _update_values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update_values'
        module_type_store = module_type_store.open_function_context('_update_values', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path._update_values.__dict__.__setitem__('stypy_localization', localization)
        Path._update_values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path._update_values.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path._update_values.__dict__.__setitem__('stypy_function_name', 'Path._update_values')
        Path._update_values.__dict__.__setitem__('stypy_param_names_list', [])
        Path._update_values.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path._update_values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path._update_values.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path._update_values.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path._update_values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path._update_values.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path._update_values', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update_values', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update_values(...)' code ##################

        
        # Assigning a Subscript to a Attribute (line 205):
        
        # Assigning a Subscript to a Attribute (line 205):
        
        # Obtaining the type of the subscript
        unicode_111874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 44), 'unicode', u'path.simplify_threshold')
        # Getting the type of 'rcParams' (line 205)
        rcParams_111875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 35), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___111876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 35), rcParams_111875, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_111877 = invoke(stypy.reporting.localization.Localization(__file__, 205, 35), getitem___111876, unicode_111874)
        
        # Getting the type of 'self' (line 205)
        self_111878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self')
        # Setting the type of the member '_simplify_threshold' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_111878, '_simplify_threshold', subscript_call_result_111877)
        
        # Assigning a BoolOp to a Attribute (line 206):
        
        # Assigning a BoolOp to a Attribute (line 206):
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 207)
        self_111879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'self')
        # Obtaining the member '_simplify_threshold' of a type (line 207)
        _simplify_threshold_111880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), self_111879, '_simplify_threshold')
        int_111881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 39), 'int')
        # Applying the binary operator '>' (line 207)
        result_gt_111882 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 12), '>', _simplify_threshold_111880, int_111881)
        
        
        # Obtaining the type of the subscript
        unicode_111883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 21), 'unicode', u'path.simplify')
        # Getting the type of 'rcParams' (line 208)
        rcParams_111884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___111885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), rcParams_111884, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_111886 = invoke(stypy.reporting.localization.Localization(__file__, 208, 12), getitem___111885, unicode_111883)
        
        # Applying the binary operator 'and' (line 207)
        result_and_keyword_111887 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 12), 'and', result_gt_111882, subscript_call_result_111886)
        
        
        # Call to len(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'self' (line 209)
        self_111889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'self', False)
        # Obtaining the member '_vertices' of a type (line 209)
        _vertices_111890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 16), self_111889, '_vertices')
        # Processing the call keyword arguments (line 209)
        kwargs_111891 = {}
        # Getting the type of 'len' (line 209)
        len_111888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'len', False)
        # Calling len(args, kwargs) (line 209)
        len_call_result_111892 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), len_111888, *[_vertices_111890], **kwargs_111891)
        
        int_111893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 35), 'int')
        # Applying the binary operator '>=' (line 209)
        result_ge_111894 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 12), '>=', len_call_result_111892, int_111893)
        
        # Applying the binary operator 'and' (line 207)
        result_and_keyword_111895 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 12), 'and', result_and_keyword_111887, result_ge_111894)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 210)
        self_111896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), 'self')
        # Obtaining the member '_codes' of a type (line 210)
        _codes_111897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 13), self_111896, '_codes')
        # Getting the type of 'None' (line 210)
        None_111898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 28), 'None')
        # Applying the binary operator 'is' (line 210)
        result_is__111899 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 13), 'is', _codes_111897, None_111898)
        
        
        # Call to all(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Getting the type of 'self' (line 210)
        self_111902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 43), 'self', False)
        # Obtaining the member '_codes' of a type (line 210)
        _codes_111903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 43), self_111902, '_codes')
        # Getting the type of 'Path' (line 210)
        Path_111904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 58), 'Path', False)
        # Obtaining the member 'LINETO' of a type (line 210)
        LINETO_111905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 58), Path_111904, 'LINETO')
        # Applying the binary operator '<=' (line 210)
        result_le_111906 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 43), '<=', _codes_111903, LINETO_111905)
        
        # Processing the call keyword arguments (line 210)
        kwargs_111907 = {}
        # Getting the type of 'np' (line 210)
        np_111900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 36), 'np', False)
        # Obtaining the member 'all' of a type (line 210)
        all_111901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 36), np_111900, 'all')
        # Calling all(args, kwargs) (line 210)
        all_call_result_111908 = invoke(stypy.reporting.localization.Localization(__file__, 210, 36), all_111901, *[result_le_111906], **kwargs_111907)
        
        # Applying the binary operator 'or' (line 210)
        result_or_keyword_111909 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 13), 'or', result_is__111899, all_call_result_111908)
        
        # Applying the binary operator 'and' (line 207)
        result_and_keyword_111910 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 12), 'and', result_and_keyword_111895, result_or_keyword_111909)
        
        # Getting the type of 'self' (line 206)
        self_111911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self')
        # Setting the type of the member '_should_simplify' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_111911, '_should_simplify', result_and_keyword_111910)
        
        # Assigning a UnaryOp to a Attribute (line 212):
        
        # Assigning a UnaryOp to a Attribute (line 212):
        
        
        # Call to all(...): (line 212)
        # Processing the call keyword arguments (line 212)
        kwargs_111919 = {}
        
        # Call to isfinite(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'self' (line 212)
        self_111914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 46), 'self', False)
        # Obtaining the member '_vertices' of a type (line 212)
        _vertices_111915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 46), self_111914, '_vertices')
        # Processing the call keyword arguments (line 212)
        kwargs_111916 = {}
        # Getting the type of 'np' (line 212)
        np_111912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 34), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 212)
        isfinite_111913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 34), np_111912, 'isfinite')
        # Calling isfinite(args, kwargs) (line 212)
        isfinite_call_result_111917 = invoke(stypy.reporting.localization.Localization(__file__, 212, 34), isfinite_111913, *[_vertices_111915], **kwargs_111916)
        
        # Obtaining the member 'all' of a type (line 212)
        all_111918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 34), isfinite_call_result_111917, 'all')
        # Calling all(args, kwargs) (line 212)
        all_call_result_111920 = invoke(stypy.reporting.localization.Localization(__file__, 212, 34), all_111918, *[], **kwargs_111919)
        
        # Applying the 'not' unary operator (line 212)
        result_not__111921 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 30), 'not', all_call_result_111920)
        
        # Getting the type of 'self' (line 212)
        self_111922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'self')
        # Setting the type of the member '_has_nonfinite' of a type (line 212)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), self_111922, '_has_nonfinite', result_not__111921)
        
        # ################# End of '_update_values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update_values' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_111923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111923)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update_values'
        return stypy_return_type_111923


    @norecursion
    def vertices(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'vertices'
        module_type_store = module_type_store.open_function_context('vertices', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.vertices.__dict__.__setitem__('stypy_localization', localization)
        Path.vertices.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.vertices.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.vertices.__dict__.__setitem__('stypy_function_name', 'Path.vertices')
        Path.vertices.__dict__.__setitem__('stypy_param_names_list', [])
        Path.vertices.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.vertices.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.vertices.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.vertices.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.vertices.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.vertices.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.vertices', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'vertices', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'vertices(...)' code ##################

        unicode_111924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, (-1)), 'unicode', u'\n        The list of vertices in the `Path` as an Nx2 numpy array.\n        ')
        # Getting the type of 'self' (line 219)
        self_111925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'self')
        # Obtaining the member '_vertices' of a type (line 219)
        _vertices_111926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 15), self_111925, '_vertices')
        # Assigning a type to the variable 'stypy_return_type' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'stypy_return_type', _vertices_111926)
        
        # ################# End of 'vertices(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'vertices' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_111927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111927)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'vertices'
        return stypy_return_type_111927


    @norecursion
    def vertices(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'vertices'
        module_type_store = module_type_store.open_function_context('vertices', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.vertices.__dict__.__setitem__('stypy_localization', localization)
        Path.vertices.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.vertices.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.vertices.__dict__.__setitem__('stypy_function_name', 'Path.vertices')
        Path.vertices.__dict__.__setitem__('stypy_param_names_list', ['vertices'])
        Path.vertices.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.vertices.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.vertices.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.vertices.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.vertices.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.vertices.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.vertices', ['vertices'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'vertices', localization, ['vertices'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'vertices(...)' code ##################

        
        # Getting the type of 'self' (line 223)
        self_111928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'self')
        # Obtaining the member '_readonly' of a type (line 223)
        _readonly_111929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 11), self_111928, '_readonly')
        # Testing the type of an if condition (line 223)
        if_condition_111930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 8), _readonly_111929)
        # Assigning a type to the variable 'if_condition_111930' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'if_condition_111930', if_condition_111930)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AttributeError(...): (line 224)
        # Processing the call arguments (line 224)
        unicode_111932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 33), 'unicode', u"Can't set vertices on a readonly Path")
        # Processing the call keyword arguments (line 224)
        kwargs_111933 = {}
        # Getting the type of 'AttributeError' (line 224)
        AttributeError_111931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 224)
        AttributeError_call_result_111934 = invoke(stypy.reporting.localization.Localization(__file__, 224, 18), AttributeError_111931, *[unicode_111932], **kwargs_111933)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 224, 12), AttributeError_call_result_111934, 'raise parameter', BaseException)
        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 225):
        
        # Assigning a Name to a Attribute (line 225):
        # Getting the type of 'vertices' (line 225)
        vertices_111935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 25), 'vertices')
        # Getting the type of 'self' (line 225)
        self_111936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'self')
        # Setting the type of the member '_vertices' of a type (line 225)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), self_111936, '_vertices', vertices_111935)
        
        # Call to _update_values(...): (line 226)
        # Processing the call keyword arguments (line 226)
        kwargs_111939 = {}
        # Getting the type of 'self' (line 226)
        self_111937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'self', False)
        # Obtaining the member '_update_values' of a type (line 226)
        _update_values_111938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), self_111937, '_update_values')
        # Calling _update_values(args, kwargs) (line 226)
        _update_values_call_result_111940 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), _update_values_111938, *[], **kwargs_111939)
        
        
        # ################# End of 'vertices(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'vertices' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_111941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111941)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'vertices'
        return stypy_return_type_111941


    @norecursion
    def codes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'codes'
        module_type_store = module_type_store.open_function_context('codes', 228, 4, False)
        # Assigning a type to the variable 'self' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.codes.__dict__.__setitem__('stypy_localization', localization)
        Path.codes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.codes.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.codes.__dict__.__setitem__('stypy_function_name', 'Path.codes')
        Path.codes.__dict__.__setitem__('stypy_param_names_list', [])
        Path.codes.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.codes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.codes.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.codes.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.codes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.codes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.codes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'codes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'codes(...)' code ##################

        unicode_111942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, (-1)), 'unicode', u'\n        The list of codes in the `Path` as a 1-D numpy array.  Each\n        code is one of `STOP`, `MOVETO`, `LINETO`, `CURVE3`, `CURVE4`\n        or `CLOSEPOLY`.  For codes that correspond to more than one\n        vertex (`CURVE3` and `CURVE4`), that code will be repeated so\n        that the length of `self.vertices` and `self.codes` is always\n        the same.\n        ')
        # Getting the type of 'self' (line 238)
        self_111943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'self')
        # Obtaining the member '_codes' of a type (line 238)
        _codes_111944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 15), self_111943, '_codes')
        # Assigning a type to the variable 'stypy_return_type' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'stypy_return_type', _codes_111944)
        
        # ################# End of 'codes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'codes' in the type store
        # Getting the type of 'stypy_return_type' (line 228)
        stypy_return_type_111945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111945)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'codes'
        return stypy_return_type_111945


    @norecursion
    def codes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'codes'
        module_type_store = module_type_store.open_function_context('codes', 240, 4, False)
        # Assigning a type to the variable 'self' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.codes.__dict__.__setitem__('stypy_localization', localization)
        Path.codes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.codes.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.codes.__dict__.__setitem__('stypy_function_name', 'Path.codes')
        Path.codes.__dict__.__setitem__('stypy_param_names_list', ['codes'])
        Path.codes.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.codes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.codes.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.codes.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.codes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.codes.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.codes', ['codes'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'codes', localization, ['codes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'codes(...)' code ##################

        
        # Getting the type of 'self' (line 242)
        self_111946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'self')
        # Obtaining the member '_readonly' of a type (line 242)
        _readonly_111947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 11), self_111946, '_readonly')
        # Testing the type of an if condition (line 242)
        if_condition_111948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 8), _readonly_111947)
        # Assigning a type to the variable 'if_condition_111948' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'if_condition_111948', if_condition_111948)
        # SSA begins for if statement (line 242)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AttributeError(...): (line 243)
        # Processing the call arguments (line 243)
        unicode_111950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 33), 'unicode', u"Can't set codes on a readonly Path")
        # Processing the call keyword arguments (line 243)
        kwargs_111951 = {}
        # Getting the type of 'AttributeError' (line 243)
        AttributeError_111949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 243)
        AttributeError_call_result_111952 = invoke(stypy.reporting.localization.Localization(__file__, 243, 18), AttributeError_111949, *[unicode_111950], **kwargs_111951)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 243, 12), AttributeError_call_result_111952, 'raise parameter', BaseException)
        # SSA join for if statement (line 242)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 244):
        
        # Assigning a Name to a Attribute (line 244):
        # Getting the type of 'codes' (line 244)
        codes_111953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 22), 'codes')
        # Getting the type of 'self' (line 244)
        self_111954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self')
        # Setting the type of the member '_codes' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_111954, '_codes', codes_111953)
        
        # Call to _update_values(...): (line 245)
        # Processing the call keyword arguments (line 245)
        kwargs_111957 = {}
        # Getting the type of 'self' (line 245)
        self_111955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self', False)
        # Obtaining the member '_update_values' of a type (line 245)
        _update_values_111956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_111955, '_update_values')
        # Calling _update_values(args, kwargs) (line 245)
        _update_values_call_result_111958 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), _update_values_111956, *[], **kwargs_111957)
        
        
        # ################# End of 'codes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'codes' in the type store
        # Getting the type of 'stypy_return_type' (line 240)
        stypy_return_type_111959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111959)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'codes'
        return stypy_return_type_111959


    @norecursion
    def simplify_threshold(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'simplify_threshold'
        module_type_store = module_type_store.open_function_context('simplify_threshold', 247, 4, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.simplify_threshold.__dict__.__setitem__('stypy_localization', localization)
        Path.simplify_threshold.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.simplify_threshold.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.simplify_threshold.__dict__.__setitem__('stypy_function_name', 'Path.simplify_threshold')
        Path.simplify_threshold.__dict__.__setitem__('stypy_param_names_list', [])
        Path.simplify_threshold.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.simplify_threshold.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.simplify_threshold.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.simplify_threshold.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.simplify_threshold.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.simplify_threshold.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.simplify_threshold', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'simplify_threshold', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'simplify_threshold(...)' code ##################

        unicode_111960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, (-1)), 'unicode', u'\n        The fraction of a pixel difference below which vertices will\n        be simplified out.\n        ')
        # Getting the type of 'self' (line 253)
        self_111961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 15), 'self')
        # Obtaining the member '_simplify_threshold' of a type (line 253)
        _simplify_threshold_111962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 15), self_111961, '_simplify_threshold')
        # Assigning a type to the variable 'stypy_return_type' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'stypy_return_type', _simplify_threshold_111962)
        
        # ################# End of 'simplify_threshold(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'simplify_threshold' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_111963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111963)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'simplify_threshold'
        return stypy_return_type_111963


    @norecursion
    def simplify_threshold(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'simplify_threshold'
        module_type_store = module_type_store.open_function_context('simplify_threshold', 255, 4, False)
        # Assigning a type to the variable 'self' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.simplify_threshold.__dict__.__setitem__('stypy_localization', localization)
        Path.simplify_threshold.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.simplify_threshold.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.simplify_threshold.__dict__.__setitem__('stypy_function_name', 'Path.simplify_threshold')
        Path.simplify_threshold.__dict__.__setitem__('stypy_param_names_list', ['threshold'])
        Path.simplify_threshold.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.simplify_threshold.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.simplify_threshold.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.simplify_threshold.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.simplify_threshold.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.simplify_threshold.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.simplify_threshold', ['threshold'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'simplify_threshold', localization, ['threshold'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'simplify_threshold(...)' code ##################

        
        # Assigning a Name to a Attribute (line 257):
        
        # Assigning a Name to a Attribute (line 257):
        # Getting the type of 'threshold' (line 257)
        threshold_111964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 35), 'threshold')
        # Getting the type of 'self' (line 257)
        self_111965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self')
        # Setting the type of the member '_simplify_threshold' of a type (line 257)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_111965, '_simplify_threshold', threshold_111964)
        
        # ################# End of 'simplify_threshold(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'simplify_threshold' in the type store
        # Getting the type of 'stypy_return_type' (line 255)
        stypy_return_type_111966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111966)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'simplify_threshold'
        return stypy_return_type_111966


    @norecursion
    def has_nonfinite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_nonfinite'
        module_type_store = module_type_store.open_function_context('has_nonfinite', 259, 4, False)
        # Assigning a type to the variable 'self' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.has_nonfinite.__dict__.__setitem__('stypy_localization', localization)
        Path.has_nonfinite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.has_nonfinite.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.has_nonfinite.__dict__.__setitem__('stypy_function_name', 'Path.has_nonfinite')
        Path.has_nonfinite.__dict__.__setitem__('stypy_param_names_list', [])
        Path.has_nonfinite.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.has_nonfinite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.has_nonfinite.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.has_nonfinite.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.has_nonfinite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.has_nonfinite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.has_nonfinite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_nonfinite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_nonfinite(...)' code ##################

        unicode_111967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, (-1)), 'unicode', u'\n        `True` if the vertices array has nonfinite values.\n        ')
        # Getting the type of 'self' (line 264)
        self_111968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 15), 'self')
        # Obtaining the member '_has_nonfinite' of a type (line 264)
        _has_nonfinite_111969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 15), self_111968, '_has_nonfinite')
        # Assigning a type to the variable 'stypy_return_type' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'stypy_return_type', _has_nonfinite_111969)
        
        # ################# End of 'has_nonfinite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_nonfinite' in the type store
        # Getting the type of 'stypy_return_type' (line 259)
        stypy_return_type_111970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111970)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_nonfinite'
        return stypy_return_type_111970


    @norecursion
    def should_simplify(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'should_simplify'
        module_type_store = module_type_store.open_function_context('should_simplify', 266, 4, False)
        # Assigning a type to the variable 'self' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.should_simplify.__dict__.__setitem__('stypy_localization', localization)
        Path.should_simplify.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.should_simplify.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.should_simplify.__dict__.__setitem__('stypy_function_name', 'Path.should_simplify')
        Path.should_simplify.__dict__.__setitem__('stypy_param_names_list', [])
        Path.should_simplify.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.should_simplify.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.should_simplify.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.should_simplify.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.should_simplify.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.should_simplify.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.should_simplify', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'should_simplify', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'should_simplify(...)' code ##################

        unicode_111971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, (-1)), 'unicode', u'\n        `True` if the vertices array should be simplified.\n        ')
        # Getting the type of 'self' (line 271)
        self_111972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'self')
        # Obtaining the member '_should_simplify' of a type (line 271)
        _should_simplify_111973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 15), self_111972, '_should_simplify')
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'stypy_return_type', _should_simplify_111973)
        
        # ################# End of 'should_simplify(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'should_simplify' in the type store
        # Getting the type of 'stypy_return_type' (line 266)
        stypy_return_type_111974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111974)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'should_simplify'
        return stypy_return_type_111974


    @norecursion
    def should_simplify(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'should_simplify'
        module_type_store = module_type_store.open_function_context('should_simplify', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.should_simplify.__dict__.__setitem__('stypy_localization', localization)
        Path.should_simplify.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.should_simplify.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.should_simplify.__dict__.__setitem__('stypy_function_name', 'Path.should_simplify')
        Path.should_simplify.__dict__.__setitem__('stypy_param_names_list', ['should_simplify'])
        Path.should_simplify.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.should_simplify.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.should_simplify.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.should_simplify.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.should_simplify.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.should_simplify.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.should_simplify', ['should_simplify'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'should_simplify', localization, ['should_simplify'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'should_simplify(...)' code ##################

        
        # Assigning a Name to a Attribute (line 275):
        
        # Assigning a Name to a Attribute (line 275):
        # Getting the type of 'should_simplify' (line 275)
        should_simplify_111975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 32), 'should_simplify')
        # Getting the type of 'self' (line 275)
        self_111976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self')
        # Setting the type of the member '_should_simplify' of a type (line 275)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_111976, '_should_simplify', should_simplify_111975)
        
        # ################# End of 'should_simplify(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'should_simplify' in the type store
        # Getting the type of 'stypy_return_type' (line 273)
        stypy_return_type_111977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111977)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'should_simplify'
        return stypy_return_type_111977


    @norecursion
    def readonly(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'readonly'
        module_type_store = module_type_store.open_function_context('readonly', 277, 4, False)
        # Assigning a type to the variable 'self' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.readonly.__dict__.__setitem__('stypy_localization', localization)
        Path.readonly.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.readonly.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.readonly.__dict__.__setitem__('stypy_function_name', 'Path.readonly')
        Path.readonly.__dict__.__setitem__('stypy_param_names_list', [])
        Path.readonly.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.readonly.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.readonly.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.readonly.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.readonly.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.readonly.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.readonly', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'readonly', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'readonly(...)' code ##################

        unicode_111978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, (-1)), 'unicode', u'\n        `True` if the `Path` is read-only.\n        ')
        # Getting the type of 'self' (line 282)
        self_111979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'self')
        # Obtaining the member '_readonly' of a type (line 282)
        _readonly_111980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 15), self_111979, '_readonly')
        # Assigning a type to the variable 'stypy_return_type' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'stypy_return_type', _readonly_111980)
        
        # ################# End of 'readonly(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'readonly' in the type store
        # Getting the type of 'stypy_return_type' (line 277)
        stypy_return_type_111981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111981)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'readonly'
        return stypy_return_type_111981


    @norecursion
    def __copy__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__copy__'
        module_type_store = module_type_store.open_function_context('__copy__', 284, 4, False)
        # Assigning a type to the variable 'self' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.__copy__.__dict__.__setitem__('stypy_localization', localization)
        Path.__copy__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.__copy__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.__copy__.__dict__.__setitem__('stypy_function_name', 'Path.__copy__')
        Path.__copy__.__dict__.__setitem__('stypy_param_names_list', [])
        Path.__copy__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.__copy__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.__copy__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.__copy__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.__copy__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.__copy__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.__copy__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__copy__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__copy__(...)' code ##################

        unicode_111982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, (-1)), 'unicode', u'\n        Returns a shallow copy of the `Path`, which will share the\n        vertices and codes with the source `Path`.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 289, 8))
        
        # 'import copy' statement (line 289)
        import copy

        import_module(stypy.reporting.localization.Localization(__file__, 289, 8), 'copy', copy, module_type_store)
        
        
        # Call to copy(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'self' (line 290)
        self_111985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 25), 'self', False)
        # Processing the call keyword arguments (line 290)
        kwargs_111986 = {}
        # Getting the type of 'copy' (line 290)
        copy_111983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'copy', False)
        # Obtaining the member 'copy' of a type (line 290)
        copy_111984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 15), copy_111983, 'copy')
        # Calling copy(args, kwargs) (line 290)
        copy_call_result_111987 = invoke(stypy.reporting.localization.Localization(__file__, 290, 15), copy_111984, *[self_111985], **kwargs_111986)
        
        # Assigning a type to the variable 'stypy_return_type' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'stypy_return_type', copy_call_result_111987)
        
        # ################# End of '__copy__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__copy__' in the type store
        # Getting the type of 'stypy_return_type' (line 284)
        stypy_return_type_111988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_111988)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__copy__'
        return stypy_return_type_111988

    
    # Assigning a Name to a Name (line 292):

    @norecursion
    def __deepcopy__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 294)
        None_111989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 32), 'None')
        defaults = [None_111989]
        # Create a new context for function '__deepcopy__'
        module_type_store = module_type_store.open_function_context('__deepcopy__', 294, 4, False)
        # Assigning a type to the variable 'self' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.__deepcopy__.__dict__.__setitem__('stypy_localization', localization)
        Path.__deepcopy__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.__deepcopy__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.__deepcopy__.__dict__.__setitem__('stypy_function_name', 'Path.__deepcopy__')
        Path.__deepcopy__.__dict__.__setitem__('stypy_param_names_list', ['memo'])
        Path.__deepcopy__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.__deepcopy__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.__deepcopy__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.__deepcopy__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.__deepcopy__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.__deepcopy__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.__deepcopy__', ['memo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__deepcopy__', localization, ['memo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__deepcopy__(...)' code ##################

        unicode_111990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, (-1)), 'unicode', u'\n        Returns a deepcopy of the `Path`.  The `Path` will not be\n        readonly, even if the source `Path` is.\n        ')
        
        
        # SSA begins for try-except statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to copy(...): (line 300)
        # Processing the call keyword arguments (line 300)
        kwargs_111994 = {}
        # Getting the type of 'self' (line 300)
        self_111991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'self', False)
        # Obtaining the member 'codes' of a type (line 300)
        codes_111992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 20), self_111991, 'codes')
        # Obtaining the member 'copy' of a type (line 300)
        copy_111993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 20), codes_111992, 'copy')
        # Calling copy(args, kwargs) (line 300)
        copy_call_result_111995 = invoke(stypy.reporting.localization.Localization(__file__, 300, 20), copy_111993, *[], **kwargs_111994)
        
        # Assigning a type to the variable 'codes' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'codes', copy_call_result_111995)
        # SSA branch for the except part of a try statement (line 299)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 299)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 302):
        
        # Assigning a Name to a Name (line 302):
        # Getting the type of 'None' (line 302)
        None_111996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 20), 'None')
        # Assigning a type to the variable 'codes' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'codes', None_111996)
        # SSA join for try-except statement (line 299)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __class__(...): (line 303)
        # Processing the call arguments (line 303)
        
        # Call to copy(...): (line 304)
        # Processing the call keyword arguments (line 304)
        kwargs_112002 = {}
        # Getting the type of 'self' (line 304)
        self_111999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'self', False)
        # Obtaining the member 'vertices' of a type (line 304)
        vertices_112000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), self_111999, 'vertices')
        # Obtaining the member 'copy' of a type (line 304)
        copy_112001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), vertices_112000, 'copy')
        # Calling copy(args, kwargs) (line 304)
        copy_call_result_112003 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), copy_112001, *[], **kwargs_112002)
        
        # Getting the type of 'codes' (line 304)
        codes_112004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 34), 'codes', False)
        # Processing the call keyword arguments (line 303)
        # Getting the type of 'self' (line 305)
        self_112005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 33), 'self', False)
        # Obtaining the member '_interpolation_steps' of a type (line 305)
        _interpolation_steps_112006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 33), self_112005, '_interpolation_steps')
        keyword_112007 = _interpolation_steps_112006
        kwargs_112008 = {'_interpolation_steps': keyword_112007}
        # Getting the type of 'self' (line 303)
        self_111997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 303)
        class___111998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 15), self_111997, '__class__')
        # Calling __class__(args, kwargs) (line 303)
        class___call_result_112009 = invoke(stypy.reporting.localization.Localization(__file__, 303, 15), class___111998, *[copy_call_result_112003, codes_112004], **kwargs_112008)
        
        # Assigning a type to the variable 'stypy_return_type' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'stypy_return_type', class___call_result_112009)
        
        # ################# End of '__deepcopy__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__deepcopy__' in the type store
        # Getting the type of 'stypy_return_type' (line 294)
        stypy_return_type_112010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112010)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__deepcopy__'
        return stypy_return_type_112010

    
    # Assigning a Name to a Name (line 307):

    @norecursion
    def make_compound_path_from_polys(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_compound_path_from_polys'
        module_type_store = module_type_store.open_function_context('make_compound_path_from_polys', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.make_compound_path_from_polys.__dict__.__setitem__('stypy_localization', localization)
        Path.make_compound_path_from_polys.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.make_compound_path_from_polys.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.make_compound_path_from_polys.__dict__.__setitem__('stypy_function_name', 'Path.make_compound_path_from_polys')
        Path.make_compound_path_from_polys.__dict__.__setitem__('stypy_param_names_list', ['XY'])
        Path.make_compound_path_from_polys.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.make_compound_path_from_polys.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.make_compound_path_from_polys.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.make_compound_path_from_polys.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.make_compound_path_from_polys.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.make_compound_path_from_polys.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.make_compound_path_from_polys', ['XY'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_compound_path_from_polys', localization, ['XY'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_compound_path_from_polys(...)' code ##################

        unicode_112011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, (-1)), 'unicode', u'\n        Make a compound path object to draw a number\n        of polygons with equal numbers of sides XY is a (numpolys x\n        numsides x 2) numpy array of vertices.  Return object is a\n        :class:`Path`\n\n        .. plot:: gallery/api/histogram_path.py\n\n        ')
        
        # Assigning a Attribute to a Tuple (line 324):
        
        # Assigning a Subscript to a Name (line 324):
        
        # Obtaining the type of the subscript
        int_112012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 8), 'int')
        # Getting the type of 'XY' (line 324)
        XY_112013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'XY')
        # Obtaining the member 'shape' of a type (line 324)
        shape_112014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 34), XY_112013, 'shape')
        # Obtaining the member '__getitem__' of a type (line 324)
        getitem___112015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), shape_112014, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 324)
        subscript_call_result_112016 = invoke(stypy.reporting.localization.Localization(__file__, 324, 8), getitem___112015, int_112012)
        
        # Assigning a type to the variable 'tuple_var_assignment_111647' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'tuple_var_assignment_111647', subscript_call_result_112016)
        
        # Assigning a Subscript to a Name (line 324):
        
        # Obtaining the type of the subscript
        int_112017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 8), 'int')
        # Getting the type of 'XY' (line 324)
        XY_112018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'XY')
        # Obtaining the member 'shape' of a type (line 324)
        shape_112019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 34), XY_112018, 'shape')
        # Obtaining the member '__getitem__' of a type (line 324)
        getitem___112020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), shape_112019, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 324)
        subscript_call_result_112021 = invoke(stypy.reporting.localization.Localization(__file__, 324, 8), getitem___112020, int_112017)
        
        # Assigning a type to the variable 'tuple_var_assignment_111648' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'tuple_var_assignment_111648', subscript_call_result_112021)
        
        # Assigning a Subscript to a Name (line 324):
        
        # Obtaining the type of the subscript
        int_112022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 8), 'int')
        # Getting the type of 'XY' (line 324)
        XY_112023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'XY')
        # Obtaining the member 'shape' of a type (line 324)
        shape_112024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 34), XY_112023, 'shape')
        # Obtaining the member '__getitem__' of a type (line 324)
        getitem___112025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), shape_112024, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 324)
        subscript_call_result_112026 = invoke(stypy.reporting.localization.Localization(__file__, 324, 8), getitem___112025, int_112022)
        
        # Assigning a type to the variable 'tuple_var_assignment_111649' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'tuple_var_assignment_111649', subscript_call_result_112026)
        
        # Assigning a Name to a Name (line 324):
        # Getting the type of 'tuple_var_assignment_111647' (line 324)
        tuple_var_assignment_111647_112027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'tuple_var_assignment_111647')
        # Assigning a type to the variable 'numpolys' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'numpolys', tuple_var_assignment_111647_112027)
        
        # Assigning a Name to a Name (line 324):
        # Getting the type of 'tuple_var_assignment_111648' (line 324)
        tuple_var_assignment_111648_112028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'tuple_var_assignment_111648')
        # Assigning a type to the variable 'numsides' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 18), 'numsides', tuple_var_assignment_111648_112028)
        
        # Assigning a Name to a Name (line 324):
        # Getting the type of 'tuple_var_assignment_111649' (line 324)
        tuple_var_assignment_111649_112029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'tuple_var_assignment_111649')
        # Assigning a type to the variable 'two' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 28), 'two', tuple_var_assignment_111649_112029)
        
        
        # Getting the type of 'two' (line 325)
        two_112030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'two')
        int_112031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 18), 'int')
        # Applying the binary operator '!=' (line 325)
        result_ne_112032 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 11), '!=', two_112030, int_112031)
        
        # Testing the type of an if condition (line 325)
        if_condition_112033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 8), result_ne_112032)
        # Assigning a type to the variable 'if_condition_112033' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'if_condition_112033', if_condition_112033)
        # SSA begins for if statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 326)
        # Processing the call arguments (line 326)
        unicode_112035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 29), 'unicode', u"The third dimension of 'XY' must be 2")
        # Processing the call keyword arguments (line 326)
        kwargs_112036 = {}
        # Getting the type of 'ValueError' (line 326)
        ValueError_112034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 326)
        ValueError_call_result_112037 = invoke(stypy.reporting.localization.Localization(__file__, 326, 18), ValueError_112034, *[unicode_112035], **kwargs_112036)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 326, 12), ValueError_call_result_112037, 'raise parameter', BaseException)
        # SSA join for if statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 327):
        
        # Assigning a BinOp to a Name (line 327):
        # Getting the type of 'numsides' (line 327)
        numsides_112038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 17), 'numsides')
        int_112039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 28), 'int')
        # Applying the binary operator '+' (line 327)
        result_add_112040 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 17), '+', numsides_112038, int_112039)
        
        # Assigning a type to the variable 'stride' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'stride', result_add_112040)
        
        # Assigning a BinOp to a Name (line 328):
        
        # Assigning a BinOp to a Name (line 328):
        # Getting the type of 'numpolys' (line 328)
        numpolys_112041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 17), 'numpolys')
        # Getting the type of 'stride' (line 328)
        stride_112042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 28), 'stride')
        # Applying the binary operator '*' (line 328)
        result_mul_112043 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 17), '*', numpolys_112041, stride_112042)
        
        # Assigning a type to the variable 'nverts' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'nverts', result_mul_112043)
        
        # Assigning a Call to a Name (line 329):
        
        # Assigning a Call to a Name (line 329):
        
        # Call to zeros(...): (line 329)
        # Processing the call arguments (line 329)
        
        # Obtaining an instance of the builtin type 'tuple' (line 329)
        tuple_112046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 329)
        # Adding element type (line 329)
        # Getting the type of 'nverts' (line 329)
        nverts_112047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 26), 'nverts', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 26), tuple_112046, nverts_112047)
        # Adding element type (line 329)
        int_112048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 26), tuple_112046, int_112048)
        
        # Processing the call keyword arguments (line 329)
        kwargs_112049 = {}
        # Getting the type of 'np' (line 329)
        np_112044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'np', False)
        # Obtaining the member 'zeros' of a type (line 329)
        zeros_112045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 16), np_112044, 'zeros')
        # Calling zeros(args, kwargs) (line 329)
        zeros_call_result_112050 = invoke(stypy.reporting.localization.Localization(__file__, 329, 16), zeros_112045, *[tuple_112046], **kwargs_112049)
        
        # Assigning a type to the variable 'verts' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'verts', zeros_call_result_112050)
        
        # Assigning a BinOp to a Name (line 330):
        
        # Assigning a BinOp to a Name (line 330):
        
        # Call to ones(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'nverts' (line 330)
        nverts_112053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 24), 'nverts', False)
        # Getting the type of 'int' (line 330)
        int_112054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 32), 'int', False)
        # Processing the call keyword arguments (line 330)
        kwargs_112055 = {}
        # Getting the type of 'np' (line 330)
        np_112051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'np', False)
        # Obtaining the member 'ones' of a type (line 330)
        ones_112052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 16), np_112051, 'ones')
        # Calling ones(args, kwargs) (line 330)
        ones_call_result_112056 = invoke(stypy.reporting.localization.Localization(__file__, 330, 16), ones_112052, *[nverts_112053, int_112054], **kwargs_112055)
        
        # Getting the type of 'cls' (line 330)
        cls_112057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 39), 'cls')
        # Obtaining the member 'LINETO' of a type (line 330)
        LINETO_112058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 39), cls_112057, 'LINETO')
        # Applying the binary operator '*' (line 330)
        result_mul_112059 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 16), '*', ones_call_result_112056, LINETO_112058)
        
        # Assigning a type to the variable 'codes' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'codes', result_mul_112059)
        
        # Assigning a Attribute to a Subscript (line 331):
        
        # Assigning a Attribute to a Subscript (line 331):
        # Getting the type of 'cls' (line 331)
        cls_112060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 27), 'cls')
        # Obtaining the member 'MOVETO' of a type (line 331)
        MOVETO_112061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 27), cls_112060, 'MOVETO')
        # Getting the type of 'codes' (line 331)
        codes_112062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'codes')
        int_112063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 14), 'int')
        # Getting the type of 'stride' (line 331)
        stride_112064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 17), 'stride')
        slice_112065 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 331, 8), int_112063, None, stride_112064)
        # Storing an element on a container (line 331)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 8), codes_112062, (slice_112065, MOVETO_112061))
        
        # Assigning a Attribute to a Subscript (line 332):
        
        # Assigning a Attribute to a Subscript (line 332):
        # Getting the type of 'cls' (line 332)
        cls_112066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'cls')
        # Obtaining the member 'CLOSEPOLY' of a type (line 332)
        CLOSEPOLY_112067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 34), cls_112066, 'CLOSEPOLY')
        # Getting the type of 'codes' (line 332)
        codes_112068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'codes')
        # Getting the type of 'numsides' (line 332)
        numsides_112069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), 'numsides')
        # Getting the type of 'stride' (line 332)
        stride_112070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 24), 'stride')
        slice_112071 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 332, 8), numsides_112069, None, stride_112070)
        # Storing an element on a container (line 332)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 8), codes_112068, (slice_112071, CLOSEPOLY_112067))
        
        
        # Call to range(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'numsides' (line 333)
        numsides_112073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 23), 'numsides', False)
        # Processing the call keyword arguments (line 333)
        kwargs_112074 = {}
        # Getting the type of 'range' (line 333)
        range_112072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 17), 'range', False)
        # Calling range(args, kwargs) (line 333)
        range_call_result_112075 = invoke(stypy.reporting.localization.Localization(__file__, 333, 17), range_112072, *[numsides_112073], **kwargs_112074)
        
        # Testing the type of a for loop iterable (line 333)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 333, 8), range_call_result_112075)
        # Getting the type of the for loop variable (line 333)
        for_loop_var_112076 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 333, 8), range_call_result_112075)
        # Assigning a type to the variable 'i' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'i', for_loop_var_112076)
        # SSA begins for a for statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 334):
        
        # Assigning a Subscript to a Subscript (line 334):
        
        # Obtaining the type of the subscript
        slice_112077 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 334, 31), None, None, None)
        # Getting the type of 'i' (line 334)
        i_112078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 37), 'i')
        # Getting the type of 'XY' (line 334)
        XY_112079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 31), 'XY')
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___112080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 31), XY_112079, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 334)
        subscript_call_result_112081 = invoke(stypy.reporting.localization.Localization(__file__, 334, 31), getitem___112080, (slice_112077, i_112078))
        
        # Getting the type of 'verts' (line 334)
        verts_112082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'verts')
        # Getting the type of 'i' (line 334)
        i_112083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 18), 'i')
        # Getting the type of 'stride' (line 334)
        stride_112084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 21), 'stride')
        slice_112085 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 334, 12), i_112083, None, stride_112084)
        # Storing an element on a container (line 334)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 12), verts_112082, (slice_112085, subscript_call_result_112081))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to cls(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'verts' (line 336)
        verts_112087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 'verts', False)
        # Getting the type of 'codes' (line 336)
        codes_112088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 26), 'codes', False)
        # Processing the call keyword arguments (line 336)
        kwargs_112089 = {}
        # Getting the type of 'cls' (line 336)
        cls_112086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 336)
        cls_call_result_112090 = invoke(stypy.reporting.localization.Localization(__file__, 336, 15), cls_112086, *[verts_112087, codes_112088], **kwargs_112089)
        
        # Assigning a type to the variable 'stypy_return_type' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'stypy_return_type', cls_call_result_112090)
        
        # ################# End of 'make_compound_path_from_polys(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_compound_path_from_polys' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_112091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112091)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_compound_path_from_polys'
        return stypy_return_type_112091


    @norecursion
    def make_compound_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_compound_path'
        module_type_store = module_type_store.open_function_context('make_compound_path', 338, 4, False)
        # Assigning a type to the variable 'self' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.make_compound_path.__dict__.__setitem__('stypy_localization', localization)
        Path.make_compound_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.make_compound_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.make_compound_path.__dict__.__setitem__('stypy_function_name', 'Path.make_compound_path')
        Path.make_compound_path.__dict__.__setitem__('stypy_param_names_list', [])
        Path.make_compound_path.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Path.make_compound_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.make_compound_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.make_compound_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.make_compound_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.make_compound_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.make_compound_path', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_compound_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_compound_path(...)' code ##################

        unicode_112092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 8), 'unicode', u'Make a compound path from a list of Path objects.')
        
        
        # Getting the type of 'args' (line 342)
        args_112093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'args')
        # Applying the 'not' unary operator (line 342)
        result_not__112094 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 11), 'not', args_112093)
        
        # Testing the type of an if condition (line 342)
        if_condition_112095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 8), result_not__112094)
        # Assigning a type to the variable 'if_condition_112095' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'if_condition_112095', if_condition_112095)
        # SSA begins for if statement (line 342)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to Path(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Call to empty(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_112099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        int_112100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 33), list_112099, int_112100)
        # Adding element type (line 343)
        int_112101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 33), list_112099, int_112101)
        
        # Processing the call keyword arguments (line 343)
        # Getting the type of 'np' (line 343)
        np_112102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 47), 'np', False)
        # Obtaining the member 'float32' of a type (line 343)
        float32_112103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 47), np_112102, 'float32')
        keyword_112104 = float32_112103
        kwargs_112105 = {'dtype': keyword_112104}
        # Getting the type of 'np' (line 343)
        np_112097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'np', False)
        # Obtaining the member 'empty' of a type (line 343)
        empty_112098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 24), np_112097, 'empty')
        # Calling empty(args, kwargs) (line 343)
        empty_call_result_112106 = invoke(stypy.reporting.localization.Localization(__file__, 343, 24), empty_112098, *[list_112099], **kwargs_112105)
        
        # Processing the call keyword arguments (line 343)
        kwargs_112107 = {}
        # Getting the type of 'Path' (line 343)
        Path_112096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'Path', False)
        # Calling Path(args, kwargs) (line 343)
        Path_call_result_112108 = invoke(stypy.reporting.localization.Localization(__file__, 343, 19), Path_112096, *[empty_call_result_112106], **kwargs_112107)
        
        # Assigning a type to the variable 'stypy_return_type' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'stypy_return_type', Path_call_result_112108)
        # SSA join for if statement (line 342)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a ListComp to a Name (line 345):
        
        # Assigning a ListComp to a Name (line 345):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'args' (line 345)
        args_112113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 35), 'args')
        comprehension_112114 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 19), args_112113)
        # Assigning a type to the variable 'x' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'x', comprehension_112114)
        
        # Call to len(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'x' (line 345)
        x_112110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 'x', False)
        # Processing the call keyword arguments (line 345)
        kwargs_112111 = {}
        # Getting the type of 'len' (line 345)
        len_112109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'len', False)
        # Calling len(args, kwargs) (line 345)
        len_call_result_112112 = invoke(stypy.reporting.localization.Localization(__file__, 345, 19), len_112109, *[x_112110], **kwargs_112111)
        
        list_112115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 19), list_112115, len_call_result_112112)
        # Assigning a type to the variable 'lengths' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'lengths', list_112115)
        
        # Assigning a Call to a Name (line 346):
        
        # Assigning a Call to a Name (line 346):
        
        # Call to sum(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'lengths' (line 346)
        lengths_112117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 27), 'lengths', False)
        # Processing the call keyword arguments (line 346)
        kwargs_112118 = {}
        # Getting the type of 'sum' (line 346)
        sum_112116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 23), 'sum', False)
        # Calling sum(args, kwargs) (line 346)
        sum_call_result_112119 = invoke(stypy.reporting.localization.Localization(__file__, 346, 23), sum_112116, *[lengths_112117], **kwargs_112118)
        
        # Assigning a type to the variable 'total_length' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'total_length', sum_call_result_112119)
        
        # Assigning a Call to a Name (line 348):
        
        # Assigning a Call to a Name (line 348):
        
        # Call to vstack(...): (line 348)
        # Processing the call arguments (line 348)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'args' (line 348)
        args_112124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 50), 'args', False)
        comprehension_112125 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 30), args_112124)
        # Assigning a type to the variable 'x' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 30), 'x', comprehension_112125)
        # Getting the type of 'x' (line 348)
        x_112122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 30), 'x', False)
        # Obtaining the member 'vertices' of a type (line 348)
        vertices_112123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 30), x_112122, 'vertices')
        list_112126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 30), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 30), list_112126, vertices_112123)
        # Processing the call keyword arguments (line 348)
        kwargs_112127 = {}
        # Getting the type of 'np' (line 348)
        np_112120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 19), 'np', False)
        # Obtaining the member 'vstack' of a type (line 348)
        vstack_112121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 19), np_112120, 'vstack')
        # Calling vstack(args, kwargs) (line 348)
        vstack_call_result_112128 = invoke(stypy.reporting.localization.Localization(__file__, 348, 19), vstack_112121, *[list_112126], **kwargs_112127)
        
        # Assigning a type to the variable 'vertices' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'vertices', vstack_call_result_112128)
        
        # Call to reshape(...): (line 349)
        # Processing the call arguments (line 349)
        
        # Obtaining an instance of the builtin type 'tuple' (line 349)
        tuple_112131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 349)
        # Adding element type (line 349)
        # Getting the type of 'total_length' (line 349)
        total_length_112132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 26), 'total_length', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 26), tuple_112131, total_length_112132)
        # Adding element type (line 349)
        int_112133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 26), tuple_112131, int_112133)
        
        # Processing the call keyword arguments (line 349)
        kwargs_112134 = {}
        # Getting the type of 'vertices' (line 349)
        vertices_112129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'vertices', False)
        # Obtaining the member 'reshape' of a type (line 349)
        reshape_112130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), vertices_112129, 'reshape')
        # Calling reshape(args, kwargs) (line 349)
        reshape_call_result_112135 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), reshape_112130, *[tuple_112131], **kwargs_112134)
        
        
        # Assigning a Call to a Name (line 351):
        
        # Assigning a Call to a Name (line 351):
        
        # Call to empty(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'total_length' (line 351)
        total_length_112138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 25), 'total_length', False)
        # Processing the call keyword arguments (line 351)
        # Getting the type of 'cls' (line 351)
        cls_112139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 45), 'cls', False)
        # Obtaining the member 'code_type' of a type (line 351)
        code_type_112140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 45), cls_112139, 'code_type')
        keyword_112141 = code_type_112140
        kwargs_112142 = {'dtype': keyword_112141}
        # Getting the type of 'np' (line 351)
        np_112136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 16), 'np', False)
        # Obtaining the member 'empty' of a type (line 351)
        empty_112137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 16), np_112136, 'empty')
        # Calling empty(args, kwargs) (line 351)
        empty_call_result_112143 = invoke(stypy.reporting.localization.Localization(__file__, 351, 16), empty_112137, *[total_length_112138], **kwargs_112142)
        
        # Assigning a type to the variable 'codes' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'codes', empty_call_result_112143)
        
        # Assigning a Num to a Name (line 352):
        
        # Assigning a Num to a Name (line 352):
        int_112144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 12), 'int')
        # Assigning a type to the variable 'i' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'i', int_112144)
        
        # Getting the type of 'args' (line 353)
        args_112145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 20), 'args')
        # Testing the type of a for loop iterable (line 353)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 353, 8), args_112145)
        # Getting the type of the for loop variable (line 353)
        for_loop_var_112146 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 353, 8), args_112145)
        # Assigning a type to the variable 'path' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'path', for_loop_var_112146)
        # SSA begins for a for statement (line 353)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 354)
        # Getting the type of 'path' (line 354)
        path_112147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 15), 'path')
        # Obtaining the member 'codes' of a type (line 354)
        codes_112148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 15), path_112147, 'codes')
        # Getting the type of 'None' (line 354)
        None_112149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 29), 'None')
        
        (may_be_112150, more_types_in_union_112151) = may_be_none(codes_112148, None_112149)

        if may_be_112150:

            if more_types_in_union_112151:
                # Runtime conditional SSA (line 354)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Subscript (line 355):
            
            # Assigning a Attribute to a Subscript (line 355):
            # Getting the type of 'cls' (line 355)
            cls_112152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 27), 'cls')
            # Obtaining the member 'MOVETO' of a type (line 355)
            MOVETO_112153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 27), cls_112152, 'MOVETO')
            # Getting the type of 'codes' (line 355)
            codes_112154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'codes')
            # Getting the type of 'i' (line 355)
            i_112155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 22), 'i')
            # Storing an element on a container (line 355)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 16), codes_112154, (i_112155, MOVETO_112153))
            
            # Assigning a Attribute to a Subscript (line 356):
            
            # Assigning a Attribute to a Subscript (line 356):
            # Getting the type of 'cls' (line 356)
            cls_112156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 54), 'cls')
            # Obtaining the member 'LINETO' of a type (line 356)
            LINETO_112157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 54), cls_112156, 'LINETO')
            # Getting the type of 'codes' (line 356)
            codes_112158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 16), 'codes')
            # Getting the type of 'i' (line 356)
            i_112159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'i')
            int_112160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 26), 'int')
            # Applying the binary operator '+' (line 356)
            result_add_112161 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 22), '+', i_112159, int_112160)
            
            # Getting the type of 'i' (line 356)
            i_112162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 28), 'i')
            
            # Call to len(...): (line 356)
            # Processing the call arguments (line 356)
            # Getting the type of 'path' (line 356)
            path_112164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'path', False)
            # Obtaining the member 'vertices' of a type (line 356)
            vertices_112165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 36), path_112164, 'vertices')
            # Processing the call keyword arguments (line 356)
            kwargs_112166 = {}
            # Getting the type of 'len' (line 356)
            len_112163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 32), 'len', False)
            # Calling len(args, kwargs) (line 356)
            len_call_result_112167 = invoke(stypy.reporting.localization.Localization(__file__, 356, 32), len_112163, *[vertices_112165], **kwargs_112166)
            
            # Applying the binary operator '+' (line 356)
            result_add_112168 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 28), '+', i_112162, len_call_result_112167)
            
            slice_112169 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 356, 16), result_add_112161, result_add_112168, None)
            # Storing an element on a container (line 356)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 16), codes_112158, (slice_112169, LINETO_112157))

            if more_types_in_union_112151:
                # Runtime conditional SSA for else branch (line 354)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_112150) or more_types_in_union_112151):
            
            # Assigning a Attribute to a Subscript (line 358):
            
            # Assigning a Attribute to a Subscript (line 358):
            # Getting the type of 'path' (line 358)
            path_112170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 47), 'path')
            # Obtaining the member 'codes' of a type (line 358)
            codes_112171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 47), path_112170, 'codes')
            # Getting the type of 'codes' (line 358)
            codes_112172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'codes')
            # Getting the type of 'i' (line 358)
            i_112173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 22), 'i')
            # Getting the type of 'i' (line 358)
            i_112174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 24), 'i')
            
            # Call to len(...): (line 358)
            # Processing the call arguments (line 358)
            # Getting the type of 'path' (line 358)
            path_112176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 32), 'path', False)
            # Obtaining the member 'codes' of a type (line 358)
            codes_112177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 32), path_112176, 'codes')
            # Processing the call keyword arguments (line 358)
            kwargs_112178 = {}
            # Getting the type of 'len' (line 358)
            len_112175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 28), 'len', False)
            # Calling len(args, kwargs) (line 358)
            len_call_result_112179 = invoke(stypy.reporting.localization.Localization(__file__, 358, 28), len_112175, *[codes_112177], **kwargs_112178)
            
            # Applying the binary operator '+' (line 358)
            result_add_112180 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 24), '+', i_112174, len_call_result_112179)
            
            slice_112181 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 358, 16), i_112173, result_add_112180, None)
            # Storing an element on a container (line 358)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 16), codes_112172, (slice_112181, codes_112171))

            if (may_be_112150 and more_types_in_union_112151):
                # SSA join for if statement (line 354)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'i' (line 359)
        i_112182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'i')
        
        # Call to len(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'path' (line 359)
        path_112184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 21), 'path', False)
        # Obtaining the member 'vertices' of a type (line 359)
        vertices_112185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 21), path_112184, 'vertices')
        # Processing the call keyword arguments (line 359)
        kwargs_112186 = {}
        # Getting the type of 'len' (line 359)
        len_112183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 17), 'len', False)
        # Calling len(args, kwargs) (line 359)
        len_call_result_112187 = invoke(stypy.reporting.localization.Localization(__file__, 359, 17), len_112183, *[vertices_112185], **kwargs_112186)
        
        # Applying the binary operator '+=' (line 359)
        result_iadd_112188 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 12), '+=', i_112182, len_call_result_112187)
        # Assigning a type to the variable 'i' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'i', result_iadd_112188)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to cls(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'vertices' (line 361)
        vertices_112190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 19), 'vertices', False)
        # Getting the type of 'codes' (line 361)
        codes_112191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 29), 'codes', False)
        # Processing the call keyword arguments (line 361)
        kwargs_112192 = {}
        # Getting the type of 'cls' (line 361)
        cls_112189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 361)
        cls_call_result_112193 = invoke(stypy.reporting.localization.Localization(__file__, 361, 15), cls_112189, *[vertices_112190, codes_112191], **kwargs_112192)
        
        # Assigning a type to the variable 'stypy_return_type' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'stypy_return_type', cls_call_result_112193)
        
        # ################# End of 'make_compound_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_compound_path' in the type store
        # Getting the type of 'stypy_return_type' (line 338)
        stypy_return_type_112194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112194)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_compound_path'
        return stypy_return_type_112194


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 363, 4, False)
        # Assigning a type to the variable 'self' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Path.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Path.stypy__repr__')
        Path.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Path.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        unicode_112195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 15), 'unicode', u'Path(%r, %r)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 364)
        tuple_112196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 364)
        # Adding element type (line 364)
        # Getting the type of 'self' (line 364)
        self_112197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 33), 'self')
        # Obtaining the member 'vertices' of a type (line 364)
        vertices_112198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 33), self_112197, 'vertices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 33), tuple_112196, vertices_112198)
        # Adding element type (line 364)
        # Getting the type of 'self' (line 364)
        self_112199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 48), 'self')
        # Obtaining the member 'codes' of a type (line 364)
        codes_112200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 48), self_112199, 'codes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 33), tuple_112196, codes_112200)
        
        # Applying the binary operator '%' (line 364)
        result_mod_112201 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 15), '%', unicode_112195, tuple_112196)
        
        # Assigning a type to the variable 'stypy_return_type' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'stypy_return_type', result_mod_112201)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 363)
        stypy_return_type_112202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112202)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_112202


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 366, 4, False)
        # Assigning a type to the variable 'self' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.__len__.__dict__.__setitem__('stypy_localization', localization)
        Path.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.__len__.__dict__.__setitem__('stypy_function_name', 'Path.__len__')
        Path.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        Path.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        
        # Call to len(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'self' (line 367)
        self_112204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 19), 'self', False)
        # Obtaining the member 'vertices' of a type (line 367)
        vertices_112205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 19), self_112204, 'vertices')
        # Processing the call keyword arguments (line 367)
        kwargs_112206 = {}
        # Getting the type of 'len' (line 367)
        len_112203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 15), 'len', False)
        # Calling len(args, kwargs) (line 367)
        len_call_result_112207 = invoke(stypy.reporting.localization.Localization(__file__, 367, 15), len_112203, *[vertices_112205], **kwargs_112206)
        
        # Assigning a type to the variable 'stypy_return_type' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'stypy_return_type', len_call_result_112207)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 366)
        stypy_return_type_112208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112208)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_112208


    @norecursion
    def iter_segments(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 369)
        None_112209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 38), 'None')
        # Getting the type of 'True' (line 369)
        True_112210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 56), 'True')
        # Getting the type of 'None' (line 369)
        None_112211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 67), 'None')
        # Getting the type of 'False' (line 370)
        False_112212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 27), 'False')
        float_112213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 47), 'float')
        # Getting the type of 'None' (line 370)
        None_112214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 61), 'None')
        # Getting the type of 'True' (line 371)
        True_112215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 29), 'True')
        # Getting the type of 'None' (line 371)
        None_112216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 42), 'None')
        defaults = [None_112209, True_112210, None_112211, False_112212, float_112213, None_112214, True_112215, None_112216]
        # Create a new context for function 'iter_segments'
        module_type_store = module_type_store.open_function_context('iter_segments', 369, 4, False)
        # Assigning a type to the variable 'self' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.iter_segments.__dict__.__setitem__('stypy_localization', localization)
        Path.iter_segments.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.iter_segments.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.iter_segments.__dict__.__setitem__('stypy_function_name', 'Path.iter_segments')
        Path.iter_segments.__dict__.__setitem__('stypy_param_names_list', ['transform', 'remove_nans', 'clip', 'snap', 'stroke_width', 'simplify', 'curves', 'sketch'])
        Path.iter_segments.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.iter_segments.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.iter_segments.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.iter_segments.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.iter_segments.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.iter_segments.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.iter_segments', ['transform', 'remove_nans', 'clip', 'snap', 'stroke_width', 'simplify', 'curves', 'sketch'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'iter_segments', localization, ['transform', 'remove_nans', 'clip', 'snap', 'stroke_width', 'simplify', 'curves', 'sketch'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'iter_segments(...)' code ##################

        unicode_112217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, (-1)), 'unicode', u"\n        Iterates over all of the curve segments in the path.  Each\n        iteration returns a 2-tuple (*vertices*, *code*), where\n        *vertices* is a sequence of 1 - 3 coordinate pairs, and *code* is\n        one of the :class:`Path` codes.\n\n        Additionally, this method can provide a number of standard\n        cleanups and conversions to the path.\n\n        Parameters\n        ----------\n        transform : None or :class:`~matplotlib.transforms.Transform` instance\n            If not None, the given affine transformation will\n            be applied to the path.\n        remove_nans : {False, True}, optional\n            If True, will remove all NaNs from the path and\n            insert MOVETO commands to skip over them.\n        clip : None or sequence, optional\n            If not None, must be a four-tuple (x1, y1, x2, y2)\n            defining a rectangle in which to clip the path.\n        snap : None or bool, optional\n            If None, auto-snap to pixels, to reduce\n            fuzziness of rectilinear lines.  If True, force snapping, and\n            if False, don't snap.\n        stroke_width : float, optional\n            The width of the stroke being drawn.  Needed\n             as a hint for the snapping algorithm.\n        simplify : None or bool, optional\n            If True, perform simplification, to remove\n             vertices that do not affect the appearance of the path.  If\n             False, perform no simplification.  If None, use the\n             should_simplify member variable.  See also the rcParams\n             path.simplify and path.simplify_threshold.\n        curves : {True, False}, optional\n            If True, curve segments will be returned as curve\n            segments.  If False, all curves will be converted to line\n            segments.\n        sketch : None or sequence, optional\n            If not None, must be a 3-tuple of the form\n            (scale, length, randomness), representing the sketch\n            parameters.\n        ")
        
        
        
        # Call to len(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'self' (line 414)
        self_112219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 19), 'self', False)
        # Processing the call keyword arguments (line 414)
        kwargs_112220 = {}
        # Getting the type of 'len' (line 414)
        len_112218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 15), 'len', False)
        # Calling len(args, kwargs) (line 414)
        len_call_result_112221 = invoke(stypy.reporting.localization.Localization(__file__, 414, 15), len_112218, *[self_112219], **kwargs_112220)
        
        # Applying the 'not' unary operator (line 414)
        result_not__112222 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 11), 'not', len_call_result_112221)
        
        # Testing the type of an if condition (line 414)
        if_condition_112223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 414, 8), result_not__112222)
        # Assigning a type to the variable 'if_condition_112223' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'if_condition_112223', if_condition_112223)
        # SSA begins for if statement (line 414)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 414)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 417):
        
        # Assigning a Call to a Name (line 417):
        
        # Call to cleaned(...): (line 417)
        # Processing the call keyword arguments (line 417)
        # Getting the type of 'transform' (line 417)
        transform_112226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 41), 'transform', False)
        keyword_112227 = transform_112226
        # Getting the type of 'remove_nans' (line 418)
        remove_nans_112228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 43), 'remove_nans', False)
        keyword_112229 = remove_nans_112228
        # Getting the type of 'clip' (line 418)
        clip_112230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 61), 'clip', False)
        keyword_112231 = clip_112230
        # Getting the type of 'snap' (line 419)
        snap_112232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 36), 'snap', False)
        keyword_112233 = snap_112232
        # Getting the type of 'stroke_width' (line 419)
        stroke_width_112234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 55), 'stroke_width', False)
        keyword_112235 = stroke_width_112234
        # Getting the type of 'simplify' (line 420)
        simplify_112236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 40), 'simplify', False)
        keyword_112237 = simplify_112236
        # Getting the type of 'curves' (line 420)
        curves_112238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 57), 'curves', False)
        keyword_112239 = curves_112238
        # Getting the type of 'sketch' (line 421)
        sketch_112240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 38), 'sketch', False)
        keyword_112241 = sketch_112240
        kwargs_112242 = {'remove_nans': keyword_112229, 'clip': keyword_112231, 'transform': keyword_112227, 'curves': keyword_112239, 'stroke_width': keyword_112235, 'simplify': keyword_112237, 'sketch': keyword_112241, 'snap': keyword_112233}
        # Getting the type of 'self' (line 417)
        self_112224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'self', False)
        # Obtaining the member 'cleaned' of a type (line 417)
        cleaned_112225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 18), self_112224, 'cleaned')
        # Calling cleaned(args, kwargs) (line 417)
        cleaned_call_result_112243 = invoke(stypy.reporting.localization.Localization(__file__, 417, 18), cleaned_112225, *[], **kwargs_112242)
        
        # Assigning a type to the variable 'cleaned' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'cleaned', cleaned_call_result_112243)
        
        # Assigning a Attribute to a Name (line 422):
        
        # Assigning a Attribute to a Name (line 422):
        # Getting the type of 'cleaned' (line 422)
        cleaned_112244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 19), 'cleaned')
        # Obtaining the member 'vertices' of a type (line 422)
        vertices_112245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 19), cleaned_112244, 'vertices')
        # Assigning a type to the variable 'vertices' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'vertices', vertices_112245)
        
        # Assigning a Attribute to a Name (line 423):
        
        # Assigning a Attribute to a Name (line 423):
        # Getting the type of 'cleaned' (line 423)
        cleaned_112246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'cleaned')
        # Obtaining the member 'codes' of a type (line 423)
        codes_112247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), cleaned_112246, 'codes')
        # Assigning a type to the variable 'codes' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'codes', codes_112247)
        
        # Assigning a Subscript to a Name (line 424):
        
        # Assigning a Subscript to a Name (line 424):
        
        # Obtaining the type of the subscript
        int_112248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 38), 'int')
        # Getting the type of 'vertices' (line 424)
        vertices_112249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 23), 'vertices')
        # Obtaining the member 'shape' of a type (line 424)
        shape_112250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 23), vertices_112249, 'shape')
        # Obtaining the member '__getitem__' of a type (line 424)
        getitem___112251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 23), shape_112250, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 424)
        subscript_call_result_112252 = invoke(stypy.reporting.localization.Localization(__file__, 424, 23), getitem___112251, int_112248)
        
        # Assigning a type to the variable 'len_vertices' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'len_vertices', subscript_call_result_112252)
        
        # Assigning a Attribute to a Name (line 427):
        
        # Assigning a Attribute to a Name (line 427):
        # Getting the type of 'self' (line 427)
        self_112253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 32), 'self')
        # Obtaining the member 'NUM_VERTICES_FOR_CODE' of a type (line 427)
        NUM_VERTICES_FOR_CODE_112254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 32), self_112253, 'NUM_VERTICES_FOR_CODE')
        # Assigning a type to the variable 'NUM_VERTICES_FOR_CODE' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'NUM_VERTICES_FOR_CODE', NUM_VERTICES_FOR_CODE_112254)
        
        # Assigning a Attribute to a Name (line 428):
        
        # Assigning a Attribute to a Name (line 428):
        # Getting the type of 'self' (line 428)
        self_112255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 15), 'self')
        # Obtaining the member 'STOP' of a type (line 428)
        STOP_112256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 15), self_112255, 'STOP')
        # Assigning a type to the variable 'STOP' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'STOP', STOP_112256)
        
        # Assigning a Num to a Name (line 430):
        
        # Assigning a Num to a Name (line 430):
        int_112257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 12), 'int')
        # Assigning a type to the variable 'i' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'i', int_112257)
        
        
        # Getting the type of 'i' (line 431)
        i_112258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 14), 'i')
        # Getting the type of 'len_vertices' (line 431)
        len_vertices_112259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 18), 'len_vertices')
        # Applying the binary operator '<' (line 431)
        result_lt_112260 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 14), '<', i_112258, len_vertices_112259)
        
        # Testing the type of an if condition (line 431)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 431, 8), result_lt_112260)
        # SSA begins for while statement (line 431)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Subscript to a Name (line 432):
        
        # Assigning a Subscript to a Name (line 432):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 432)
        i_112261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 25), 'i')
        # Getting the type of 'codes' (line 432)
        codes_112262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 19), 'codes')
        # Obtaining the member '__getitem__' of a type (line 432)
        getitem___112263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 19), codes_112262, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 432)
        subscript_call_result_112264 = invoke(stypy.reporting.localization.Localization(__file__, 432, 19), getitem___112263, i_112261)
        
        # Assigning a type to the variable 'code' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'code', subscript_call_result_112264)
        
        
        # Getting the type of 'code' (line 433)
        code_112265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'code')
        # Getting the type of 'STOP' (line 433)
        STOP_112266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 23), 'STOP')
        # Applying the binary operator '==' (line 433)
        result_eq_112267 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 15), '==', code_112265, STOP_112266)
        
        # Testing the type of an if condition (line 433)
        if_condition_112268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 12), result_eq_112267)
        # Assigning a type to the variable 'if_condition_112268' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 12), 'if_condition_112268', if_condition_112268)
        # SSA begins for if statement (line 433)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'stypy_return_type', types.NoneType)
        # SSA branch for the else part of an if statement (line 433)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 436):
        
        # Assigning a Subscript to a Name (line 436):
        
        # Obtaining the type of the subscript
        # Getting the type of 'code' (line 436)
        code_112269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 53), 'code')
        # Getting the type of 'NUM_VERTICES_FOR_CODE' (line 436)
        NUM_VERTICES_FOR_CODE_112270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 31), 'NUM_VERTICES_FOR_CODE')
        # Obtaining the member '__getitem__' of a type (line 436)
        getitem___112271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 31), NUM_VERTICES_FOR_CODE_112270, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 436)
        subscript_call_result_112272 = invoke(stypy.reporting.localization.Localization(__file__, 436, 31), getitem___112271, code_112269)
        
        # Assigning a type to the variable 'num_vertices' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'num_vertices', subscript_call_result_112272)
        
        # Assigning a Call to a Name (line 437):
        
        # Assigning a Call to a Name (line 437):
        
        # Call to flatten(...): (line 437)
        # Processing the call keyword arguments (line 437)
        kwargs_112282 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 437)
        i_112273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 41), 'i', False)
        # Getting the type of 'i' (line 437)
        i_112274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 43), 'i', False)
        # Getting the type of 'num_vertices' (line 437)
        num_vertices_112275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 45), 'num_vertices', False)
        # Applying the binary operator '+' (line 437)
        result_add_112276 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 43), '+', i_112274, num_vertices_112275)
        
        slice_112277 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 437, 32), i_112273, result_add_112276, None)
        # Getting the type of 'vertices' (line 437)
        vertices_112278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 32), 'vertices', False)
        # Obtaining the member '__getitem__' of a type (line 437)
        getitem___112279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 32), vertices_112278, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 437)
        subscript_call_result_112280 = invoke(stypy.reporting.localization.Localization(__file__, 437, 32), getitem___112279, slice_112277)
        
        # Obtaining the member 'flatten' of a type (line 437)
        flatten_112281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 32), subscript_call_result_112280, 'flatten')
        # Calling flatten(args, kwargs) (line 437)
        flatten_call_result_112283 = invoke(stypy.reporting.localization.Localization(__file__, 437, 32), flatten_112281, *[], **kwargs_112282)
        
        # Assigning a type to the variable 'curr_vertices' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 16), 'curr_vertices', flatten_call_result_112283)
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 438)
        tuple_112284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 438)
        # Adding element type (line 438)
        # Getting the type of 'curr_vertices' (line 438)
        curr_vertices_112285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), 'curr_vertices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 22), tuple_112284, curr_vertices_112285)
        # Adding element type (line 438)
        # Getting the type of 'code' (line 438)
        code_112286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 37), 'code')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 22), tuple_112284, code_112286)
        
        GeneratorType_112287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 16), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 16), GeneratorType_112287, tuple_112284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'stypy_return_type', GeneratorType_112287)
        
        # Getting the type of 'i' (line 439)
        i_112288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'i')
        # Getting the type of 'num_vertices' (line 439)
        num_vertices_112289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), 'num_vertices')
        # Applying the binary operator '+=' (line 439)
        result_iadd_112290 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 16), '+=', i_112288, num_vertices_112289)
        # Assigning a type to the variable 'i' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'i', result_iadd_112290)
        
        # SSA join for if statement (line 433)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 431)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'iter_segments(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'iter_segments' in the type store
        # Getting the type of 'stypy_return_type' (line 369)
        stypy_return_type_112291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112291)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'iter_segments'
        return stypy_return_type_112291


    @norecursion
    def cleaned(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 441)
        None_112292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 32), 'None')
        # Getting the type of 'False' (line 441)
        False_112293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 50), 'False')
        # Getting the type of 'None' (line 441)
        None_112294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 62), 'None')
        # Getting the type of 'False' (line 442)
        False_112295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 25), 'False')
        # Getting the type of 'False' (line 442)
        False_112296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 41), 'False')
        # Getting the type of 'False' (line 442)
        False_112297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 55), 'False')
        float_112298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 29), 'float')
        # Getting the type of 'False' (line 443)
        False_112299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 39), 'False')
        # Getting the type of 'None' (line 443)
        None_112300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 53), 'None')
        defaults = [None_112292, False_112293, None_112294, False_112295, False_112296, False_112297, float_112298, False_112299, None_112300]
        # Create a new context for function 'cleaned'
        module_type_store = module_type_store.open_function_context('cleaned', 441, 4, False)
        # Assigning a type to the variable 'self' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.cleaned.__dict__.__setitem__('stypy_localization', localization)
        Path.cleaned.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.cleaned.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.cleaned.__dict__.__setitem__('stypy_function_name', 'Path.cleaned')
        Path.cleaned.__dict__.__setitem__('stypy_param_names_list', ['transform', 'remove_nans', 'clip', 'quantize', 'simplify', 'curves', 'stroke_width', 'snap', 'sketch'])
        Path.cleaned.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.cleaned.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.cleaned.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.cleaned.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.cleaned.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.cleaned.__dict__.__setitem__('stypy_declared_arg_number', 10)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.cleaned', ['transform', 'remove_nans', 'clip', 'quantize', 'simplify', 'curves', 'stroke_width', 'snap', 'sketch'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cleaned', localization, ['transform', 'remove_nans', 'clip', 'quantize', 'simplify', 'curves', 'stroke_width', 'snap', 'sketch'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cleaned(...)' code ##################

        unicode_112301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, (-1)), 'unicode', u'\n        Cleans up the path according to the parameters returning a new\n        Path instance.\n\n        .. seealso::\n\n            See :meth:`iter_segments` for details of the keyword arguments.\n\n        Returns\n        -------\n        Path instance with cleaned up vertices and codes.\n\n        ')
        
        # Assigning a Call to a Tuple (line 457):
        
        # Assigning a Call to a Name:
        
        # Call to cleanup_path(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'self' (line 457)
        self_112304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 45), 'self', False)
        # Getting the type of 'transform' (line 457)
        transform_112305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 51), 'transform', False)
        # Getting the type of 'remove_nans' (line 458)
        remove_nans_112306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 45), 'remove_nans', False)
        # Getting the type of 'clip' (line 458)
        clip_112307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 58), 'clip', False)
        # Getting the type of 'snap' (line 459)
        snap_112308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 45), 'snap', False)
        # Getting the type of 'stroke_width' (line 459)
        stroke_width_112309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 51), 'stroke_width', False)
        # Getting the type of 'simplify' (line 460)
        simplify_112310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 45), 'simplify', False)
        # Getting the type of 'curves' (line 460)
        curves_112311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 55), 'curves', False)
        # Getting the type of 'sketch' (line 460)
        sketch_112312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 63), 'sketch', False)
        # Processing the call keyword arguments (line 457)
        kwargs_112313 = {}
        # Getting the type of '_path' (line 457)
        _path_112302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 26), '_path', False)
        # Obtaining the member 'cleanup_path' of a type (line 457)
        cleanup_path_112303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 26), _path_112302, 'cleanup_path')
        # Calling cleanup_path(args, kwargs) (line 457)
        cleanup_path_call_result_112314 = invoke(stypy.reporting.localization.Localization(__file__, 457, 26), cleanup_path_112303, *[self_112304, transform_112305, remove_nans_112306, clip_112307, snap_112308, stroke_width_112309, simplify_112310, curves_112311, sketch_112312], **kwargs_112313)
        
        # Assigning a type to the variable 'call_assignment_111650' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'call_assignment_111650', cleanup_path_call_result_112314)
        
        # Assigning a Call to a Name (line 457):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_112317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 8), 'int')
        # Processing the call keyword arguments
        kwargs_112318 = {}
        # Getting the type of 'call_assignment_111650' (line 457)
        call_assignment_111650_112315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'call_assignment_111650', False)
        # Obtaining the member '__getitem__' of a type (line 457)
        getitem___112316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), call_assignment_111650_112315, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_112319 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___112316, *[int_112317], **kwargs_112318)
        
        # Assigning a type to the variable 'call_assignment_111651' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'call_assignment_111651', getitem___call_result_112319)
        
        # Assigning a Name to a Name (line 457):
        # Getting the type of 'call_assignment_111651' (line 457)
        call_assignment_111651_112320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'call_assignment_111651')
        # Assigning a type to the variable 'vertices' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'vertices', call_assignment_111651_112320)
        
        # Assigning a Call to a Name (line 457):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_112323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 8), 'int')
        # Processing the call keyword arguments
        kwargs_112324 = {}
        # Getting the type of 'call_assignment_111650' (line 457)
        call_assignment_111650_112321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'call_assignment_111650', False)
        # Obtaining the member '__getitem__' of a type (line 457)
        getitem___112322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), call_assignment_111650_112321, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_112325 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___112322, *[int_112323], **kwargs_112324)
        
        # Assigning a type to the variable 'call_assignment_111652' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'call_assignment_111652', getitem___call_result_112325)
        
        # Assigning a Name to a Name (line 457):
        # Getting the type of 'call_assignment_111652' (line 457)
        call_assignment_111652_112326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'call_assignment_111652')
        # Assigning a type to the variable 'codes' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 18), 'codes', call_assignment_111652_112326)
        
        # Assigning a Dict to a Name (line 461):
        
        # Assigning a Dict to a Name (line 461):
        
        # Obtaining an instance of the builtin type 'dict' (line 461)
        dict_112327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 20), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 461)
        # Adding element type (key, value) (line 461)
        unicode_112328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 21), 'unicode', u'should_simplify')
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 461)
        self_112329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 40), 'self')
        # Obtaining the member 'should_simplify' of a type (line 461)
        should_simplify_112330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 40), self_112329, 'should_simplify')
        
        # Getting the type of 'simplify' (line 461)
        simplify_112331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 69), 'simplify')
        # Applying the 'not' unary operator (line 461)
        result_not__112332 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 65), 'not', simplify_112331)
        
        # Applying the binary operator 'and' (line 461)
        result_and_keyword_112333 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 40), 'and', should_simplify_112330, result_not__112332)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 20), dict_112327, (unicode_112328, result_and_keyword_112333))
        # Adding element type (key, value) (line 461)
        unicode_112334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 21), 'unicode', u'has_nonfinite')
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 462)
        self_112335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 38), 'self')
        # Obtaining the member 'has_nonfinite' of a type (line 462)
        has_nonfinite_112336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 38), self_112335, 'has_nonfinite')
        
        # Getting the type of 'remove_nans' (line 462)
        remove_nans_112337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 65), 'remove_nans')
        # Applying the 'not' unary operator (line 462)
        result_not__112338 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 61), 'not', remove_nans_112337)
        
        # Applying the binary operator 'and' (line 462)
        result_and_keyword_112339 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 38), 'and', has_nonfinite_112336, result_not__112338)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 20), dict_112327, (unicode_112334, result_and_keyword_112339))
        # Adding element type (key, value) (line 461)
        unicode_112340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 21), 'unicode', u'simplify_threshold')
        # Getting the type of 'self' (line 463)
        self_112341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 43), 'self')
        # Obtaining the member 'simplify_threshold' of a type (line 463)
        simplify_threshold_112342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 43), self_112341, 'simplify_threshold')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 20), dict_112327, (unicode_112340, simplify_threshold_112342))
        # Adding element type (key, value) (line 461)
        unicode_112343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 21), 'unicode', u'interpolation_steps')
        # Getting the type of 'self' (line 464)
        self_112344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 44), 'self')
        # Obtaining the member '_interpolation_steps' of a type (line 464)
        _interpolation_steps_112345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 44), self_112344, '_interpolation_steps')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 20), dict_112327, (unicode_112343, _interpolation_steps_112345))
        
        # Assigning a type to the variable 'internals' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'internals', dict_112327)
        
        # Call to _fast_from_codes_and_verts(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'vertices' (line 465)
        vertices_112348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 47), 'vertices', False)
        # Getting the type of 'codes' (line 465)
        codes_112349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 57), 'codes', False)
        # Getting the type of 'internals' (line 465)
        internals_112350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 64), 'internals', False)
        # Processing the call keyword arguments (line 465)
        kwargs_112351 = {}
        # Getting the type of 'Path' (line 465)
        Path_112346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'Path', False)
        # Obtaining the member '_fast_from_codes_and_verts' of a type (line 465)
        _fast_from_codes_and_verts_112347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 15), Path_112346, '_fast_from_codes_and_verts')
        # Calling _fast_from_codes_and_verts(args, kwargs) (line 465)
        _fast_from_codes_and_verts_call_result_112352 = invoke(stypy.reporting.localization.Localization(__file__, 465, 15), _fast_from_codes_and_verts_112347, *[vertices_112348, codes_112349, internals_112350], **kwargs_112351)
        
        # Assigning a type to the variable 'stypy_return_type' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'stypy_return_type', _fast_from_codes_and_verts_call_result_112352)
        
        # ################# End of 'cleaned(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cleaned' in the type store
        # Getting the type of 'stypy_return_type' (line 441)
        stypy_return_type_112353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cleaned'
        return stypy_return_type_112353


    @norecursion
    def transformed(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'transformed'
        module_type_store = module_type_store.open_function_context('transformed', 467, 4, False)
        # Assigning a type to the variable 'self' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.transformed.__dict__.__setitem__('stypy_localization', localization)
        Path.transformed.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.transformed.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.transformed.__dict__.__setitem__('stypy_function_name', 'Path.transformed')
        Path.transformed.__dict__.__setitem__('stypy_param_names_list', ['transform'])
        Path.transformed.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.transformed.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.transformed.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.transformed.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.transformed.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.transformed.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.transformed', ['transform'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transformed', localization, ['transform'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transformed(...)' code ##################

        unicode_112354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, (-1)), 'unicode', u'\n        Return a transformed copy of the path.\n\n        .. seealso::\n\n            :class:`matplotlib.transforms.TransformedPath`\n                A specialized path class that will cache the\n                transformed result and automatically update when the\n                transform changes.\n        ')
        
        # Call to Path(...): (line 478)
        # Processing the call arguments (line 478)
        
        # Call to transform(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'self' (line 478)
        self_112358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 40), 'self', False)
        # Obtaining the member 'vertices' of a type (line 478)
        vertices_112359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 40), self_112358, 'vertices')
        # Processing the call keyword arguments (line 478)
        kwargs_112360 = {}
        # Getting the type of 'transform' (line 478)
        transform_112356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 20), 'transform', False)
        # Obtaining the member 'transform' of a type (line 478)
        transform_112357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 20), transform_112356, 'transform')
        # Calling transform(args, kwargs) (line 478)
        transform_call_result_112361 = invoke(stypy.reporting.localization.Localization(__file__, 478, 20), transform_112357, *[vertices_112359], **kwargs_112360)
        
        # Getting the type of 'self' (line 478)
        self_112362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 56), 'self', False)
        # Obtaining the member 'codes' of a type (line 478)
        codes_112363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 56), self_112362, 'codes')
        # Getting the type of 'self' (line 479)
        self_112364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 20), 'self', False)
        # Obtaining the member '_interpolation_steps' of a type (line 479)
        _interpolation_steps_112365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 20), self_112364, '_interpolation_steps')
        # Processing the call keyword arguments (line 478)
        kwargs_112366 = {}
        # Getting the type of 'Path' (line 478)
        Path_112355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 15), 'Path', False)
        # Calling Path(args, kwargs) (line 478)
        Path_call_result_112367 = invoke(stypy.reporting.localization.Localization(__file__, 478, 15), Path_112355, *[transform_call_result_112361, codes_112363, _interpolation_steps_112365], **kwargs_112366)
        
        # Assigning a type to the variable 'stypy_return_type' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'stypy_return_type', Path_call_result_112367)
        
        # ################# End of 'transformed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transformed' in the type store
        # Getting the type of 'stypy_return_type' (line 467)
        stypy_return_type_112368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112368)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transformed'
        return stypy_return_type_112368


    @norecursion
    def contains_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 481)
        None_112369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 46), 'None')
        float_112370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 59), 'float')
        defaults = [None_112369, float_112370]
        # Create a new context for function 'contains_point'
        module_type_store = module_type_store.open_function_context('contains_point', 481, 4, False)
        # Assigning a type to the variable 'self' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.contains_point.__dict__.__setitem__('stypy_localization', localization)
        Path.contains_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.contains_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.contains_point.__dict__.__setitem__('stypy_function_name', 'Path.contains_point')
        Path.contains_point.__dict__.__setitem__('stypy_param_names_list', ['point', 'transform', 'radius'])
        Path.contains_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.contains_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.contains_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.contains_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.contains_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.contains_point.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.contains_point', ['point', 'transform', 'radius'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'contains_point', localization, ['point', 'transform', 'radius'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'contains_point(...)' code ##################

        unicode_112371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, (-1)), 'unicode', u'\n        Returns whether the (closed) path contains the given point.\n\n        If *transform* is not ``None``, the path will be transformed before\n        performing the test.\n\n        *radius* allows the path to be made slightly larger or smaller.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 490)
        # Getting the type of 'transform' (line 490)
        transform_112372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'transform')
        # Getting the type of 'None' (line 490)
        None_112373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 28), 'None')
        
        (may_be_112374, more_types_in_union_112375) = may_not_be_none(transform_112372, None_112373)

        if may_be_112374:

            if more_types_in_union_112375:
                # Runtime conditional SSA (line 490)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 491):
            
            # Assigning a Call to a Name (line 491):
            
            # Call to frozen(...): (line 491)
            # Processing the call keyword arguments (line 491)
            kwargs_112378 = {}
            # Getting the type of 'transform' (line 491)
            transform_112376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 24), 'transform', False)
            # Obtaining the member 'frozen' of a type (line 491)
            frozen_112377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 24), transform_112376, 'frozen')
            # Calling frozen(args, kwargs) (line 491)
            frozen_call_result_112379 = invoke(stypy.reporting.localization.Localization(__file__, 491, 24), frozen_112377, *[], **kwargs_112378)
            
            # Assigning a type to the variable 'transform' (line 491)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'transform', frozen_call_result_112379)

            if more_types_in_union_112375:
                # SSA join for if statement (line 490)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'transform' (line 496)
        transform_112380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 11), 'transform')
        
        # Getting the type of 'transform' (line 496)
        transform_112381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 29), 'transform')
        # Obtaining the member 'is_affine' of a type (line 496)
        is_affine_112382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 29), transform_112381, 'is_affine')
        # Applying the 'not' unary operator (line 496)
        result_not__112383 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 25), 'not', is_affine_112382)
        
        # Applying the binary operator 'and' (line 496)
        result_and_keyword_112384 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 11), 'and', transform_112380, result_not__112383)
        
        # Testing the type of an if condition (line 496)
        if_condition_112385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 496, 8), result_and_keyword_112384)
        # Assigning a type to the variable 'if_condition_112385' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'if_condition_112385', if_condition_112385)
        # SSA begins for if statement (line 496)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 497):
        
        # Assigning a Call to a Name (line 497):
        
        # Call to transform_path(...): (line 497)
        # Processing the call arguments (line 497)
        # Getting the type of 'self' (line 497)
        self_112388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 44), 'self', False)
        # Processing the call keyword arguments (line 497)
        kwargs_112389 = {}
        # Getting the type of 'transform' (line 497)
        transform_112386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 19), 'transform', False)
        # Obtaining the member 'transform_path' of a type (line 497)
        transform_path_112387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 19), transform_112386, 'transform_path')
        # Calling transform_path(args, kwargs) (line 497)
        transform_path_call_result_112390 = invoke(stypy.reporting.localization.Localization(__file__, 497, 19), transform_path_112387, *[self_112388], **kwargs_112389)
        
        # Assigning a type to the variable 'self' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'self', transform_path_call_result_112390)
        
        # Assigning a Name to a Name (line 498):
        
        # Assigning a Name to a Name (line 498):
        # Getting the type of 'None' (line 498)
        None_112391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 24), 'None')
        # Assigning a type to the variable 'transform' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'transform', None_112391)
        # SSA join for if statement (line 496)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to point_in_path(...): (line 499)
        # Processing the call arguments (line 499)
        
        # Obtaining the type of the subscript
        int_112394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 41), 'int')
        # Getting the type of 'point' (line 499)
        point_112395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 35), 'point', False)
        # Obtaining the member '__getitem__' of a type (line 499)
        getitem___112396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 35), point_112395, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 499)
        subscript_call_result_112397 = invoke(stypy.reporting.localization.Localization(__file__, 499, 35), getitem___112396, int_112394)
        
        
        # Obtaining the type of the subscript
        int_112398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 51), 'int')
        # Getting the type of 'point' (line 499)
        point_112399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 45), 'point', False)
        # Obtaining the member '__getitem__' of a type (line 499)
        getitem___112400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 45), point_112399, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 499)
        subscript_call_result_112401 = invoke(stypy.reporting.localization.Localization(__file__, 499, 45), getitem___112400, int_112398)
        
        # Getting the type of 'radius' (line 499)
        radius_112402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 55), 'radius', False)
        # Getting the type of 'self' (line 499)
        self_112403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 63), 'self', False)
        # Getting the type of 'transform' (line 499)
        transform_112404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 69), 'transform', False)
        # Processing the call keyword arguments (line 499)
        kwargs_112405 = {}
        # Getting the type of '_path' (line 499)
        _path_112392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 15), '_path', False)
        # Obtaining the member 'point_in_path' of a type (line 499)
        point_in_path_112393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 15), _path_112392, 'point_in_path')
        # Calling point_in_path(args, kwargs) (line 499)
        point_in_path_call_result_112406 = invoke(stypy.reporting.localization.Localization(__file__, 499, 15), point_in_path_112393, *[subscript_call_result_112397, subscript_call_result_112401, radius_112402, self_112403, transform_112404], **kwargs_112405)
        
        # Assigning a type to the variable 'stypy_return_type' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'stypy_return_type', point_in_path_call_result_112406)
        
        # ################# End of 'contains_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'contains_point' in the type store
        # Getting the type of 'stypy_return_type' (line 481)
        stypy_return_type_112407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112407)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'contains_point'
        return stypy_return_type_112407


    @norecursion
    def contains_points(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 501)
        None_112408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 48), 'None')
        float_112409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 61), 'float')
        defaults = [None_112408, float_112409]
        # Create a new context for function 'contains_points'
        module_type_store = module_type_store.open_function_context('contains_points', 501, 4, False)
        # Assigning a type to the variable 'self' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.contains_points.__dict__.__setitem__('stypy_localization', localization)
        Path.contains_points.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.contains_points.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.contains_points.__dict__.__setitem__('stypy_function_name', 'Path.contains_points')
        Path.contains_points.__dict__.__setitem__('stypy_param_names_list', ['points', 'transform', 'radius'])
        Path.contains_points.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.contains_points.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.contains_points.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.contains_points.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.contains_points.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.contains_points.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.contains_points', ['points', 'transform', 'radius'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'contains_points', localization, ['points', 'transform', 'radius'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'contains_points(...)' code ##################

        unicode_112410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, (-1)), 'unicode', u'\n        Returns a bool array which is ``True`` if the (closed) path contains\n        the corresponding point.\n\n        If *transform* is not ``None``, the path will be transformed before\n        performing the test.\n\n        *radius* allows the path to be made slightly larger or smaller.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 511)
        # Getting the type of 'transform' (line 511)
        transform_112411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'transform')
        # Getting the type of 'None' (line 511)
        None_112412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 28), 'None')
        
        (may_be_112413, more_types_in_union_112414) = may_not_be_none(transform_112411, None_112412)

        if may_be_112413:

            if more_types_in_union_112414:
                # Runtime conditional SSA (line 511)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 512):
            
            # Assigning a Call to a Name (line 512):
            
            # Call to frozen(...): (line 512)
            # Processing the call keyword arguments (line 512)
            kwargs_112417 = {}
            # Getting the type of 'transform' (line 512)
            transform_112415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 24), 'transform', False)
            # Obtaining the member 'frozen' of a type (line 512)
            frozen_112416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 24), transform_112415, 'frozen')
            # Calling frozen(args, kwargs) (line 512)
            frozen_call_result_112418 = invoke(stypy.reporting.localization.Localization(__file__, 512, 24), frozen_112416, *[], **kwargs_112417)
            
            # Assigning a type to the variable 'transform' (line 512)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'transform', frozen_call_result_112418)

            if more_types_in_union_112414:
                # SSA join for if statement (line 511)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 513):
        
        # Assigning a Call to a Name (line 513):
        
        # Call to points_in_path(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'points' (line 513)
        points_112421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 38), 'points', False)
        # Getting the type of 'radius' (line 513)
        radius_112422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 46), 'radius', False)
        # Getting the type of 'self' (line 513)
        self_112423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 54), 'self', False)
        # Getting the type of 'transform' (line 513)
        transform_112424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 60), 'transform', False)
        # Processing the call keyword arguments (line 513)
        kwargs_112425 = {}
        # Getting the type of '_path' (line 513)
        _path_112419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 17), '_path', False)
        # Obtaining the member 'points_in_path' of a type (line 513)
        points_in_path_112420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 17), _path_112419, 'points_in_path')
        # Calling points_in_path(args, kwargs) (line 513)
        points_in_path_call_result_112426 = invoke(stypy.reporting.localization.Localization(__file__, 513, 17), points_in_path_112420, *[points_112421, radius_112422, self_112423, transform_112424], **kwargs_112425)
        
        # Assigning a type to the variable 'result' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'result', points_in_path_call_result_112426)
        
        # Call to astype(...): (line 514)
        # Processing the call arguments (line 514)
        unicode_112429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 29), 'unicode', u'bool')
        # Processing the call keyword arguments (line 514)
        kwargs_112430 = {}
        # Getting the type of 'result' (line 514)
        result_112427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 15), 'result', False)
        # Obtaining the member 'astype' of a type (line 514)
        astype_112428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 15), result_112427, 'astype')
        # Calling astype(args, kwargs) (line 514)
        astype_call_result_112431 = invoke(stypy.reporting.localization.Localization(__file__, 514, 15), astype_112428, *[unicode_112429], **kwargs_112430)
        
        # Assigning a type to the variable 'stypy_return_type' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'stypy_return_type', astype_call_result_112431)
        
        # ################# End of 'contains_points(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'contains_points' in the type store
        # Getting the type of 'stypy_return_type' (line 501)
        stypy_return_type_112432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112432)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'contains_points'
        return stypy_return_type_112432


    @norecursion
    def contains_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 516)
        None_112433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 44), 'None')
        defaults = [None_112433]
        # Create a new context for function 'contains_path'
        module_type_store = module_type_store.open_function_context('contains_path', 516, 4, False)
        # Assigning a type to the variable 'self' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.contains_path.__dict__.__setitem__('stypy_localization', localization)
        Path.contains_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.contains_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.contains_path.__dict__.__setitem__('stypy_function_name', 'Path.contains_path')
        Path.contains_path.__dict__.__setitem__('stypy_param_names_list', ['path', 'transform'])
        Path.contains_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.contains_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.contains_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.contains_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.contains_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.contains_path.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.contains_path', ['path', 'transform'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'contains_path', localization, ['path', 'transform'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'contains_path(...)' code ##################

        unicode_112434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, (-1)), 'unicode', u'\n        Returns whether this (closed) path completely contains the given path.\n\n        If *transform* is not ``None``, the path will be transformed before\n        performing the test.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 523)
        # Getting the type of 'transform' (line 523)
        transform_112435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'transform')
        # Getting the type of 'None' (line 523)
        None_112436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 28), 'None')
        
        (may_be_112437, more_types_in_union_112438) = may_not_be_none(transform_112435, None_112436)

        if may_be_112437:

            if more_types_in_union_112438:
                # Runtime conditional SSA (line 523)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 524):
            
            # Assigning a Call to a Name (line 524):
            
            # Call to frozen(...): (line 524)
            # Processing the call keyword arguments (line 524)
            kwargs_112441 = {}
            # Getting the type of 'transform' (line 524)
            transform_112439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 24), 'transform', False)
            # Obtaining the member 'frozen' of a type (line 524)
            frozen_112440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 24), transform_112439, 'frozen')
            # Calling frozen(args, kwargs) (line 524)
            frozen_call_result_112442 = invoke(stypy.reporting.localization.Localization(__file__, 524, 24), frozen_112440, *[], **kwargs_112441)
            
            # Assigning a type to the variable 'transform' (line 524)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'transform', frozen_call_result_112442)

            if more_types_in_union_112438:
                # SSA join for if statement (line 523)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to path_in_path(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'self' (line 525)
        self_112445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 34), 'self', False)
        # Getting the type of 'None' (line 525)
        None_112446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 40), 'None', False)
        # Getting the type of 'path' (line 525)
        path_112447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 46), 'path', False)
        # Getting the type of 'transform' (line 525)
        transform_112448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 52), 'transform', False)
        # Processing the call keyword arguments (line 525)
        kwargs_112449 = {}
        # Getting the type of '_path' (line 525)
        _path_112443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 15), '_path', False)
        # Obtaining the member 'path_in_path' of a type (line 525)
        path_in_path_112444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 15), _path_112443, 'path_in_path')
        # Calling path_in_path(args, kwargs) (line 525)
        path_in_path_call_result_112450 = invoke(stypy.reporting.localization.Localization(__file__, 525, 15), path_in_path_112444, *[self_112445, None_112446, path_112447, transform_112448], **kwargs_112449)
        
        # Assigning a type to the variable 'stypy_return_type' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'stypy_return_type', path_in_path_call_result_112450)
        
        # ################# End of 'contains_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'contains_path' in the type store
        # Getting the type of 'stypy_return_type' (line 516)
        stypy_return_type_112451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112451)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'contains_path'
        return stypy_return_type_112451


    @norecursion
    def get_extents(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 527)
        None_112452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 36), 'None')
        defaults = [None_112452]
        # Create a new context for function 'get_extents'
        module_type_store = module_type_store.open_function_context('get_extents', 527, 4, False)
        # Assigning a type to the variable 'self' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.get_extents.__dict__.__setitem__('stypy_localization', localization)
        Path.get_extents.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.get_extents.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.get_extents.__dict__.__setitem__('stypy_function_name', 'Path.get_extents')
        Path.get_extents.__dict__.__setitem__('stypy_param_names_list', ['transform'])
        Path.get_extents.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.get_extents.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.get_extents.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.get_extents.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.get_extents.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.get_extents.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.get_extents', ['transform'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_extents', localization, ['transform'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_extents(...)' code ##################

        unicode_112453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, (-1)), 'unicode', u'\n        Returns the extents (*xmin*, *ymin*, *xmax*, *ymax*) of the\n        path.\n\n        Unlike computing the extents on the *vertices* alone, this\n        algorithm will take into account the curves and deal with\n        control points appropriately.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 536, 8))
        
        # 'from matplotlib.transforms import Bbox' statement (line 536)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
        import_112454 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 536, 8), 'matplotlib.transforms')

        if (type(import_112454) is not StypyTypeError):

            if (import_112454 != 'pyd_module'):
                __import__(import_112454)
                sys_modules_112455 = sys.modules[import_112454]
                import_from_module(stypy.reporting.localization.Localization(__file__, 536, 8), 'matplotlib.transforms', sys_modules_112455.module_type_store, module_type_store, ['Bbox'])
                nest_module(stypy.reporting.localization.Localization(__file__, 536, 8), __file__, sys_modules_112455, sys_modules_112455.module_type_store, module_type_store)
            else:
                from matplotlib.transforms import Bbox

                import_from_module(stypy.reporting.localization.Localization(__file__, 536, 8), 'matplotlib.transforms', None, module_type_store, ['Bbox'], [Bbox])

        else:
            # Assigning a type to the variable 'matplotlib.transforms' (line 536)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'matplotlib.transforms', import_112454)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
        
        
        # Assigning a Name to a Name (line 537):
        
        # Assigning a Name to a Name (line 537):
        # Getting the type of 'self' (line 537)
        self_112456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 15), 'self')
        # Assigning a type to the variable 'path' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'path', self_112456)
        
        # Type idiom detected: calculating its left and rigth part (line 538)
        # Getting the type of 'transform' (line 538)
        transform_112457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'transform')
        # Getting the type of 'None' (line 538)
        None_112458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 28), 'None')
        
        (may_be_112459, more_types_in_union_112460) = may_not_be_none(transform_112457, None_112458)

        if may_be_112459:

            if more_types_in_union_112460:
                # Runtime conditional SSA (line 538)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 539):
            
            # Assigning a Call to a Name (line 539):
            
            # Call to frozen(...): (line 539)
            # Processing the call keyword arguments (line 539)
            kwargs_112463 = {}
            # Getting the type of 'transform' (line 539)
            transform_112461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 24), 'transform', False)
            # Obtaining the member 'frozen' of a type (line 539)
            frozen_112462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 24), transform_112461, 'frozen')
            # Calling frozen(args, kwargs) (line 539)
            frozen_call_result_112464 = invoke(stypy.reporting.localization.Localization(__file__, 539, 24), frozen_112462, *[], **kwargs_112463)
            
            # Assigning a type to the variable 'transform' (line 539)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'transform', frozen_call_result_112464)
            
            
            # Getting the type of 'transform' (line 540)
            transform_112465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 19), 'transform')
            # Obtaining the member 'is_affine' of a type (line 540)
            is_affine_112466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 19), transform_112465, 'is_affine')
            # Applying the 'not' unary operator (line 540)
            result_not__112467 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 15), 'not', is_affine_112466)
            
            # Testing the type of an if condition (line 540)
            if_condition_112468 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 540, 12), result_not__112467)
            # Assigning a type to the variable 'if_condition_112468' (line 540)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'if_condition_112468', if_condition_112468)
            # SSA begins for if statement (line 540)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 541):
            
            # Assigning a Call to a Name (line 541):
            
            # Call to transformed(...): (line 541)
            # Processing the call arguments (line 541)
            # Getting the type of 'transform' (line 541)
            transform_112471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 40), 'transform', False)
            # Processing the call keyword arguments (line 541)
            kwargs_112472 = {}
            # Getting the type of 'self' (line 541)
            self_112469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 23), 'self', False)
            # Obtaining the member 'transformed' of a type (line 541)
            transformed_112470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 23), self_112469, 'transformed')
            # Calling transformed(args, kwargs) (line 541)
            transformed_call_result_112473 = invoke(stypy.reporting.localization.Localization(__file__, 541, 23), transformed_112470, *[transform_112471], **kwargs_112472)
            
            # Assigning a type to the variable 'path' (line 541)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'path', transformed_call_result_112473)
            
            # Assigning a Name to a Name (line 542):
            
            # Assigning a Name to a Name (line 542):
            # Getting the type of 'None' (line 542)
            None_112474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 28), 'None')
            # Assigning a type to the variable 'transform' (line 542)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'transform', None_112474)
            # SSA join for if statement (line 540)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_112460:
                # SSA join for if statement (line 538)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to Bbox(...): (line 543)
        # Processing the call arguments (line 543)
        
        # Call to get_path_extents(...): (line 543)
        # Processing the call arguments (line 543)
        # Getting the type of 'path' (line 543)
        path_112478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 43), 'path', False)
        # Getting the type of 'transform' (line 543)
        transform_112479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 49), 'transform', False)
        # Processing the call keyword arguments (line 543)
        kwargs_112480 = {}
        # Getting the type of '_path' (line 543)
        _path_112476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 20), '_path', False)
        # Obtaining the member 'get_path_extents' of a type (line 543)
        get_path_extents_112477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 20), _path_112476, 'get_path_extents')
        # Calling get_path_extents(args, kwargs) (line 543)
        get_path_extents_call_result_112481 = invoke(stypy.reporting.localization.Localization(__file__, 543, 20), get_path_extents_112477, *[path_112478, transform_112479], **kwargs_112480)
        
        # Processing the call keyword arguments (line 543)
        kwargs_112482 = {}
        # Getting the type of 'Bbox' (line 543)
        Bbox_112475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 15), 'Bbox', False)
        # Calling Bbox(args, kwargs) (line 543)
        Bbox_call_result_112483 = invoke(stypy.reporting.localization.Localization(__file__, 543, 15), Bbox_112475, *[get_path_extents_call_result_112481], **kwargs_112482)
        
        # Assigning a type to the variable 'stypy_return_type' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'stypy_return_type', Bbox_call_result_112483)
        
        # ################# End of 'get_extents(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_extents' in the type store
        # Getting the type of 'stypy_return_type' (line 527)
        stypy_return_type_112484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112484)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_extents'
        return stypy_return_type_112484


    @norecursion
    def intersects_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 545)
        True_112485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 44), 'True')
        defaults = [True_112485]
        # Create a new context for function 'intersects_path'
        module_type_store = module_type_store.open_function_context('intersects_path', 545, 4, False)
        # Assigning a type to the variable 'self' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.intersects_path.__dict__.__setitem__('stypy_localization', localization)
        Path.intersects_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.intersects_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.intersects_path.__dict__.__setitem__('stypy_function_name', 'Path.intersects_path')
        Path.intersects_path.__dict__.__setitem__('stypy_param_names_list', ['other', 'filled'])
        Path.intersects_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.intersects_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.intersects_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.intersects_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.intersects_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.intersects_path.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.intersects_path', ['other', 'filled'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'intersects_path', localization, ['other', 'filled'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'intersects_path(...)' code ##################

        unicode_112486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, (-1)), 'unicode', u'\n        Returns *True* if this path intersects another given path.\n\n        *filled*, when True, treats the paths as if they were filled.\n        That is, if one path completely encloses the other,\n        :meth:`intersects_path` will return True.\n        ')
        
        # Call to path_intersects_path(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'self' (line 553)
        self_112489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 42), 'self', False)
        # Getting the type of 'other' (line 553)
        other_112490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 48), 'other', False)
        # Getting the type of 'filled' (line 553)
        filled_112491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 55), 'filled', False)
        # Processing the call keyword arguments (line 553)
        kwargs_112492 = {}
        # Getting the type of '_path' (line 553)
        _path_112487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 15), '_path', False)
        # Obtaining the member 'path_intersects_path' of a type (line 553)
        path_intersects_path_112488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 15), _path_112487, 'path_intersects_path')
        # Calling path_intersects_path(args, kwargs) (line 553)
        path_intersects_path_call_result_112493 = invoke(stypy.reporting.localization.Localization(__file__, 553, 15), path_intersects_path_112488, *[self_112489, other_112490, filled_112491], **kwargs_112492)
        
        # Assigning a type to the variable 'stypy_return_type' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'stypy_return_type', path_intersects_path_call_result_112493)
        
        # ################# End of 'intersects_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'intersects_path' in the type store
        # Getting the type of 'stypy_return_type' (line 545)
        stypy_return_type_112494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112494)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'intersects_path'
        return stypy_return_type_112494


    @norecursion
    def intersects_bbox(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 555)
        True_112495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 43), 'True')
        defaults = [True_112495]
        # Create a new context for function 'intersects_bbox'
        module_type_store = module_type_store.open_function_context('intersects_bbox', 555, 4, False)
        # Assigning a type to the variable 'self' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.intersects_bbox.__dict__.__setitem__('stypy_localization', localization)
        Path.intersects_bbox.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.intersects_bbox.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.intersects_bbox.__dict__.__setitem__('stypy_function_name', 'Path.intersects_bbox')
        Path.intersects_bbox.__dict__.__setitem__('stypy_param_names_list', ['bbox', 'filled'])
        Path.intersects_bbox.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.intersects_bbox.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.intersects_bbox.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.intersects_bbox.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.intersects_bbox.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.intersects_bbox.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.intersects_bbox', ['bbox', 'filled'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'intersects_bbox', localization, ['bbox', 'filled'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'intersects_bbox(...)' code ##################

        unicode_112496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, (-1)), 'unicode', u'\n        Returns *True* if this path intersects a given\n        :class:`~matplotlib.transforms.Bbox`.\n\n        *filled*, when True, treats the path as if it was filled.\n        That is, if the path completely encloses the bounding box,\n        :meth:`intersects_bbox` will return True.\n\n        The bounding box is always considered filled.\n        ')
        
        # Call to path_intersects_rectangle(...): (line 566)
        # Processing the call arguments (line 566)
        # Getting the type of 'self' (line 566)
        self_112499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 47), 'self', False)
        # Getting the type of 'bbox' (line 567)
        bbox_112500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'bbox', False)
        # Obtaining the member 'x0' of a type (line 567)
        x0_112501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 12), bbox_112500, 'x0')
        # Getting the type of 'bbox' (line 567)
        bbox_112502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 21), 'bbox', False)
        # Obtaining the member 'y0' of a type (line 567)
        y0_112503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 21), bbox_112502, 'y0')
        # Getting the type of 'bbox' (line 567)
        bbox_112504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 30), 'bbox', False)
        # Obtaining the member 'x1' of a type (line 567)
        x1_112505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 30), bbox_112504, 'x1')
        # Getting the type of 'bbox' (line 567)
        bbox_112506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 39), 'bbox', False)
        # Obtaining the member 'y1' of a type (line 567)
        y1_112507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 39), bbox_112506, 'y1')
        # Getting the type of 'filled' (line 567)
        filled_112508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 48), 'filled', False)
        # Processing the call keyword arguments (line 566)
        kwargs_112509 = {}
        # Getting the type of '_path' (line 566)
        _path_112497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 15), '_path', False)
        # Obtaining the member 'path_intersects_rectangle' of a type (line 566)
        path_intersects_rectangle_112498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 15), _path_112497, 'path_intersects_rectangle')
        # Calling path_intersects_rectangle(args, kwargs) (line 566)
        path_intersects_rectangle_call_result_112510 = invoke(stypy.reporting.localization.Localization(__file__, 566, 15), path_intersects_rectangle_112498, *[self_112499, x0_112501, y0_112503, x1_112505, y1_112507, filled_112508], **kwargs_112509)
        
        # Assigning a type to the variable 'stypy_return_type' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'stypy_return_type', path_intersects_rectangle_call_result_112510)
        
        # ################# End of 'intersects_bbox(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'intersects_bbox' in the type store
        # Getting the type of 'stypy_return_type' (line 555)
        stypy_return_type_112511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112511)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'intersects_bbox'
        return stypy_return_type_112511


    @norecursion
    def interpolated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'interpolated'
        module_type_store = module_type_store.open_function_context('interpolated', 569, 4, False)
        # Assigning a type to the variable 'self' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.interpolated.__dict__.__setitem__('stypy_localization', localization)
        Path.interpolated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.interpolated.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.interpolated.__dict__.__setitem__('stypy_function_name', 'Path.interpolated')
        Path.interpolated.__dict__.__setitem__('stypy_param_names_list', ['steps'])
        Path.interpolated.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.interpolated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.interpolated.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.interpolated.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.interpolated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.interpolated.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.interpolated', ['steps'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'interpolated', localization, ['steps'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'interpolated(...)' code ##################

        unicode_112512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, (-1)), 'unicode', u'\n        Returns a new path resampled to length N x steps.  Does not\n        currently handle interpolating curves.\n        ')
        
        
        # Getting the type of 'steps' (line 574)
        steps_112513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 11), 'steps')
        int_112514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 20), 'int')
        # Applying the binary operator '==' (line 574)
        result_eq_112515 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 11), '==', steps_112513, int_112514)
        
        # Testing the type of an if condition (line 574)
        if_condition_112516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 574, 8), result_eq_112515)
        # Assigning a type to the variable 'if_condition_112516' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'if_condition_112516', if_condition_112516)
        # SSA begins for if statement (line 574)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 575)
        self_112517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'stypy_return_type', self_112517)
        # SSA join for if statement (line 574)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 577):
        
        # Assigning a Call to a Name (line 577):
        
        # Call to simple_linear_interpolation(...): (line 577)
        # Processing the call arguments (line 577)
        # Getting the type of 'self' (line 577)
        self_112519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 47), 'self', False)
        # Obtaining the member 'vertices' of a type (line 577)
        vertices_112520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 47), self_112519, 'vertices')
        # Getting the type of 'steps' (line 577)
        steps_112521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 62), 'steps', False)
        # Processing the call keyword arguments (line 577)
        kwargs_112522 = {}
        # Getting the type of 'simple_linear_interpolation' (line 577)
        simple_linear_interpolation_112518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 19), 'simple_linear_interpolation', False)
        # Calling simple_linear_interpolation(args, kwargs) (line 577)
        simple_linear_interpolation_call_result_112523 = invoke(stypy.reporting.localization.Localization(__file__, 577, 19), simple_linear_interpolation_112518, *[vertices_112520, steps_112521], **kwargs_112522)
        
        # Assigning a type to the variable 'vertices' (line 577)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'vertices', simple_linear_interpolation_call_result_112523)
        
        # Assigning a Attribute to a Name (line 578):
        
        # Assigning a Attribute to a Name (line 578):
        # Getting the type of 'self' (line 578)
        self_112524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 16), 'self')
        # Obtaining the member 'codes' of a type (line 578)
        codes_112525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 16), self_112524, 'codes')
        # Assigning a type to the variable 'codes' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'codes', codes_112525)
        
        # Type idiom detected: calculating its left and rigth part (line 579)
        # Getting the type of 'codes' (line 579)
        codes_112526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'codes')
        # Getting the type of 'None' (line 579)
        None_112527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 24), 'None')
        
        (may_be_112528, more_types_in_union_112529) = may_not_be_none(codes_112526, None_112527)

        if may_be_112528:

            if more_types_in_union_112529:
                # Runtime conditional SSA (line 579)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 580):
            
            # Assigning a BinOp to a Name (line 580):
            # Getting the type of 'Path' (line 580)
            Path_112530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 24), 'Path')
            # Obtaining the member 'LINETO' of a type (line 580)
            LINETO_112531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 24), Path_112530, 'LINETO')
            
            # Call to ones(...): (line 580)
            # Processing the call arguments (line 580)
            
            # Obtaining an instance of the builtin type 'tuple' (line 580)
            tuple_112534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 47), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 580)
            # Adding element type (line 580)
            
            # Call to len(...): (line 580)
            # Processing the call arguments (line 580)
            # Getting the type of 'codes' (line 580)
            codes_112536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 52), 'codes', False)
            # Processing the call keyword arguments (line 580)
            kwargs_112537 = {}
            # Getting the type of 'len' (line 580)
            len_112535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 48), 'len', False)
            # Calling len(args, kwargs) (line 580)
            len_call_result_112538 = invoke(stypy.reporting.localization.Localization(__file__, 580, 48), len_112535, *[codes_112536], **kwargs_112537)
            
            int_112539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 61), 'int')
            # Applying the binary operator '-' (line 580)
            result_sub_112540 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 48), '-', len_call_result_112538, int_112539)
            
            # Getting the type of 'steps' (line 580)
            steps_112541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 66), 'steps', False)
            # Applying the binary operator '*' (line 580)
            result_mul_112542 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 47), '*', result_sub_112540, steps_112541)
            
            int_112543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 74), 'int')
            # Applying the binary operator '+' (line 580)
            result_add_112544 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 47), '+', result_mul_112542, int_112543)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 47), tuple_112534, result_add_112544)
            
            # Processing the call keyword arguments (line 580)
            kwargs_112545 = {}
            # Getting the type of 'np' (line 580)
            np_112532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 38), 'np', False)
            # Obtaining the member 'ones' of a type (line 580)
            ones_112533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 38), np_112532, 'ones')
            # Calling ones(args, kwargs) (line 580)
            ones_call_result_112546 = invoke(stypy.reporting.localization.Localization(__file__, 580, 38), ones_112533, *[tuple_112534], **kwargs_112545)
            
            # Applying the binary operator '*' (line 580)
            result_mul_112547 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 24), '*', LINETO_112531, ones_call_result_112546)
            
            # Assigning a type to the variable 'new_codes' (line 580)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'new_codes', result_mul_112547)
            
            # Assigning a Name to a Subscript (line 581):
            
            # Assigning a Name to a Subscript (line 581):
            # Getting the type of 'codes' (line 581)
            codes_112548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 34), 'codes')
            # Getting the type of 'new_codes' (line 581)
            new_codes_112549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 12), 'new_codes')
            int_112550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 22), 'int')
            # Getting the type of 'steps' (line 581)
            steps_112551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 25), 'steps')
            slice_112552 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 581, 12), int_112550, None, steps_112551)
            # Storing an element on a container (line 581)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 581, 12), new_codes_112549, (slice_112552, codes_112548))

            if more_types_in_union_112529:
                # Runtime conditional SSA for else branch (line 579)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_112528) or more_types_in_union_112529):
            
            # Assigning a Name to a Name (line 583):
            
            # Assigning a Name to a Name (line 583):
            # Getting the type of 'None' (line 583)
            None_112553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 24), 'None')
            # Assigning a type to the variable 'new_codes' (line 583)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'new_codes', None_112553)

            if (may_be_112528 and more_types_in_union_112529):
                # SSA join for if statement (line 579)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to Path(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 'vertices' (line 584)
        vertices_112555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'vertices', False)
        # Getting the type of 'new_codes' (line 584)
        new_codes_112556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 30), 'new_codes', False)
        # Processing the call keyword arguments (line 584)
        kwargs_112557 = {}
        # Getting the type of 'Path' (line 584)
        Path_112554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 15), 'Path', False)
        # Calling Path(args, kwargs) (line 584)
        Path_call_result_112558 = invoke(stypy.reporting.localization.Localization(__file__, 584, 15), Path_112554, *[vertices_112555, new_codes_112556], **kwargs_112557)
        
        # Assigning a type to the variable 'stypy_return_type' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'stypy_return_type', Path_call_result_112558)
        
        # ################# End of 'interpolated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'interpolated' in the type store
        # Getting the type of 'stypy_return_type' (line 569)
        stypy_return_type_112559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112559)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'interpolated'
        return stypy_return_type_112559


    @norecursion
    def to_polygons(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 586)
        None_112560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 36), 'None')
        int_112561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 48), 'int')
        int_112562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 58), 'int')
        # Getting the type of 'True' (line 586)
        True_112563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 73), 'True')
        defaults = [None_112560, int_112561, int_112562, True_112563]
        # Create a new context for function 'to_polygons'
        module_type_store = module_type_store.open_function_context('to_polygons', 586, 4, False)
        # Assigning a type to the variable 'self' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.to_polygons.__dict__.__setitem__('stypy_localization', localization)
        Path.to_polygons.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.to_polygons.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.to_polygons.__dict__.__setitem__('stypy_function_name', 'Path.to_polygons')
        Path.to_polygons.__dict__.__setitem__('stypy_param_names_list', ['transform', 'width', 'height', 'closed_only'])
        Path.to_polygons.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.to_polygons.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.to_polygons.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.to_polygons.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.to_polygons.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.to_polygons.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.to_polygons', ['transform', 'width', 'height', 'closed_only'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'to_polygons', localization, ['transform', 'width', 'height', 'closed_only'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'to_polygons(...)' code ##################

        unicode_112564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, (-1)), 'unicode', u'\n        Convert this path to a list of polygons or polylines.  Each\n        polygon/polyline is an Nx2 array of vertices.  In other words,\n        each polygon has no ``MOVETO`` instructions or curves.  This\n        is useful for displaying in backends that do not support\n        compound paths or Bezier curves, such as GDK.\n\n        If *width* and *height* are both non-zero then the lines will\n        be simplified so that vertices outside of (0, 0), (width,\n        height) will be clipped.\n\n        If *closed_only* is `True` (default), only closed polygons,\n        with the last point being the same as the first point, will be\n        returned.  Any unclosed polylines in the path will be\n        explicitly closed.  If *closed_only* is `False`, any unclosed\n        polygons in the path will be returned as unclosed polygons,\n        and the closed polygons will be returned explicitly closed by\n        setting the last point to the same as the first point.\n        ')
        
        
        
        # Call to len(...): (line 606)
        # Processing the call arguments (line 606)
        # Getting the type of 'self' (line 606)
        self_112566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 15), 'self', False)
        # Obtaining the member 'vertices' of a type (line 606)
        vertices_112567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 15), self_112566, 'vertices')
        # Processing the call keyword arguments (line 606)
        kwargs_112568 = {}
        # Getting the type of 'len' (line 606)
        len_112565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 11), 'len', False)
        # Calling len(args, kwargs) (line 606)
        len_call_result_112569 = invoke(stypy.reporting.localization.Localization(__file__, 606, 11), len_112565, *[vertices_112567], **kwargs_112568)
        
        int_112570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 33), 'int')
        # Applying the binary operator '==' (line 606)
        result_eq_112571 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 11), '==', len_call_result_112569, int_112570)
        
        # Testing the type of an if condition (line 606)
        if_condition_112572 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 606, 8), result_eq_112571)
        # Assigning a type to the variable 'if_condition_112572' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'if_condition_112572', if_condition_112572)
        # SSA begins for if statement (line 606)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 607)
        list_112573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 607)
        
        # Assigning a type to the variable 'stypy_return_type' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'stypy_return_type', list_112573)
        # SSA join for if statement (line 606)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 609)
        # Getting the type of 'transform' (line 609)
        transform_112574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'transform')
        # Getting the type of 'None' (line 609)
        None_112575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 28), 'None')
        
        (may_be_112576, more_types_in_union_112577) = may_not_be_none(transform_112574, None_112575)

        if may_be_112576:

            if more_types_in_union_112577:
                # Runtime conditional SSA (line 609)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 610):
            
            # Assigning a Call to a Name (line 610):
            
            # Call to frozen(...): (line 610)
            # Processing the call keyword arguments (line 610)
            kwargs_112580 = {}
            # Getting the type of 'transform' (line 610)
            transform_112578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 24), 'transform', False)
            # Obtaining the member 'frozen' of a type (line 610)
            frozen_112579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 24), transform_112578, 'frozen')
            # Calling frozen(args, kwargs) (line 610)
            frozen_call_result_112581 = invoke(stypy.reporting.localization.Localization(__file__, 610, 24), frozen_112579, *[], **kwargs_112580)
            
            # Assigning a type to the variable 'transform' (line 610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'transform', frozen_call_result_112581)

            if more_types_in_union_112577:
                # SSA join for if statement (line 609)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 612)
        self_112582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 11), 'self')
        # Obtaining the member 'codes' of a type (line 612)
        codes_112583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 11), self_112582, 'codes')
        # Getting the type of 'None' (line 612)
        None_112584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 25), 'None')
        # Applying the binary operator 'is' (line 612)
        result_is__112585 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 11), 'is', codes_112583, None_112584)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'width' (line 612)
        width_112586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 35), 'width')
        int_112587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 44), 'int')
        # Applying the binary operator '==' (line 612)
        result_eq_112588 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 35), '==', width_112586, int_112587)
        
        
        # Getting the type of 'height' (line 612)
        height_112589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 49), 'height')
        int_112590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 59), 'int')
        # Applying the binary operator '==' (line 612)
        result_eq_112591 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 49), '==', height_112589, int_112590)
        
        # Applying the binary operator 'or' (line 612)
        result_or_keyword_112592 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 35), 'or', result_eq_112588, result_eq_112591)
        
        # Applying the binary operator 'and' (line 612)
        result_and_keyword_112593 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 11), 'and', result_is__112585, result_or_keyword_112592)
        
        # Testing the type of an if condition (line 612)
        if_condition_112594 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 8), result_and_keyword_112593)
        # Assigning a type to the variable 'if_condition_112594' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'if_condition_112594', if_condition_112594)
        # SSA begins for if statement (line 612)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 613):
        
        # Assigning a Attribute to a Name (line 613):
        # Getting the type of 'self' (line 613)
        self_112595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 23), 'self')
        # Obtaining the member 'vertices' of a type (line 613)
        vertices_112596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 23), self_112595, 'vertices')
        # Assigning a type to the variable 'vertices' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'vertices', vertices_112596)
        
        # Getting the type of 'closed_only' (line 614)
        closed_only_112597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 15), 'closed_only')
        # Testing the type of an if condition (line 614)
        if_condition_112598 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 614, 12), closed_only_112597)
        # Assigning a type to the variable 'if_condition_112598' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'if_condition_112598', if_condition_112598)
        # SSA begins for if statement (line 614)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to len(...): (line 615)
        # Processing the call arguments (line 615)
        # Getting the type of 'vertices' (line 615)
        vertices_112600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 23), 'vertices', False)
        # Processing the call keyword arguments (line 615)
        kwargs_112601 = {}
        # Getting the type of 'len' (line 615)
        len_112599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 19), 'len', False)
        # Calling len(args, kwargs) (line 615)
        len_call_result_112602 = invoke(stypy.reporting.localization.Localization(__file__, 615, 19), len_112599, *[vertices_112600], **kwargs_112601)
        
        int_112603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 35), 'int')
        # Applying the binary operator '<' (line 615)
        result_lt_112604 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 19), '<', len_call_result_112602, int_112603)
        
        # Testing the type of an if condition (line 615)
        if_condition_112605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 615, 16), result_lt_112604)
        # Assigning a type to the variable 'if_condition_112605' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 16), 'if_condition_112605', if_condition_112605)
        # SSA begins for if statement (line 615)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 616)
        list_112606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 616)
        
        # Assigning a type to the variable 'stypy_return_type' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 20), 'stypy_return_type', list_112606)
        # SSA branch for the else part of an if statement (line 615)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to any(...): (line 617)
        # Processing the call arguments (line 617)
        
        
        # Obtaining the type of the subscript
        int_112609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 37), 'int')
        # Getting the type of 'vertices' (line 617)
        vertices_112610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 28), 'vertices', False)
        # Obtaining the member '__getitem__' of a type (line 617)
        getitem___112611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 28), vertices_112610, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 617)
        subscript_call_result_112612 = invoke(stypy.reporting.localization.Localization(__file__, 617, 28), getitem___112611, int_112609)
        
        
        # Obtaining the type of the subscript
        int_112613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 52), 'int')
        # Getting the type of 'vertices' (line 617)
        vertices_112614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 43), 'vertices', False)
        # Obtaining the member '__getitem__' of a type (line 617)
        getitem___112615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 43), vertices_112614, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 617)
        subscript_call_result_112616 = invoke(stypy.reporting.localization.Localization(__file__, 617, 43), getitem___112615, int_112613)
        
        # Applying the binary operator '!=' (line 617)
        result_ne_112617 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 28), '!=', subscript_call_result_112612, subscript_call_result_112616)
        
        # Processing the call keyword arguments (line 617)
        kwargs_112618 = {}
        # Getting the type of 'np' (line 617)
        np_112607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 21), 'np', False)
        # Obtaining the member 'any' of a type (line 617)
        any_112608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 21), np_112607, 'any')
        # Calling any(args, kwargs) (line 617)
        any_call_result_112619 = invoke(stypy.reporting.localization.Localization(__file__, 617, 21), any_112608, *[result_ne_112617], **kwargs_112618)
        
        # Testing the type of an if condition (line 617)
        if_condition_112620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 617, 21), any_call_result_112619)
        # Assigning a type to the variable 'if_condition_112620' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 21), 'if_condition_112620', if_condition_112620)
        # SSA begins for if statement (line 617)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 618):
        
        # Assigning a BinOp to a Name (line 618):
        
        # Call to list(...): (line 618)
        # Processing the call arguments (line 618)
        # Getting the type of 'vertices' (line 618)
        vertices_112622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 36), 'vertices', False)
        # Processing the call keyword arguments (line 618)
        kwargs_112623 = {}
        # Getting the type of 'list' (line 618)
        list_112621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 31), 'list', False)
        # Calling list(args, kwargs) (line 618)
        list_call_result_112624 = invoke(stypy.reporting.localization.Localization(__file__, 618, 31), list_112621, *[vertices_112622], **kwargs_112623)
        
        
        # Obtaining an instance of the builtin type 'list' (line 618)
        list_112625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 618)
        # Adding element type (line 618)
        
        # Obtaining the type of the subscript
        int_112626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 58), 'int')
        # Getting the type of 'vertices' (line 618)
        vertices_112627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 49), 'vertices')
        # Obtaining the member '__getitem__' of a type (line 618)
        getitem___112628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 49), vertices_112627, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 618)
        subscript_call_result_112629 = invoke(stypy.reporting.localization.Localization(__file__, 618, 49), getitem___112628, int_112626)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 48), list_112625, subscript_call_result_112629)
        
        # Applying the binary operator '+' (line 618)
        result_add_112630 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 31), '+', list_call_result_112624, list_112625)
        
        # Assigning a type to the variable 'vertices' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 20), 'vertices', result_add_112630)
        # SSA join for if statement (line 617)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 615)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 614)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 620)
        # Getting the type of 'transform' (line 620)
        transform_112631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'transform')
        # Getting the type of 'None' (line 620)
        None_112632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 28), 'None')
        
        (may_be_112633, more_types_in_union_112634) = may_be_none(transform_112631, None_112632)

        if may_be_112633:

            if more_types_in_union_112634:
                # Runtime conditional SSA (line 620)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Obtaining an instance of the builtin type 'list' (line 621)
            list_112635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 621)
            # Adding element type (line 621)
            # Getting the type of 'vertices' (line 621)
            vertices_112636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 24), 'vertices')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 23), list_112635, vertices_112636)
            
            # Assigning a type to the variable 'stypy_return_type' (line 621)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 16), 'stypy_return_type', list_112635)

            if more_types_in_union_112634:
                # Runtime conditional SSA for else branch (line 620)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_112633) or more_types_in_union_112634):
            
            # Obtaining an instance of the builtin type 'list' (line 623)
            list_112637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 623)
            # Adding element type (line 623)
            
            # Call to transform(...): (line 623)
            # Processing the call arguments (line 623)
            # Getting the type of 'vertices' (line 623)
            vertices_112640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 44), 'vertices', False)
            # Processing the call keyword arguments (line 623)
            kwargs_112641 = {}
            # Getting the type of 'transform' (line 623)
            transform_112638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 24), 'transform', False)
            # Obtaining the member 'transform' of a type (line 623)
            transform_112639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 24), transform_112638, 'transform')
            # Calling transform(args, kwargs) (line 623)
            transform_call_result_112642 = invoke(stypy.reporting.localization.Localization(__file__, 623, 24), transform_112639, *[vertices_112640], **kwargs_112641)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 23), list_112637, transform_call_result_112642)
            
            # Assigning a type to the variable 'stypy_return_type' (line 623)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'stypy_return_type', list_112637)

            if (may_be_112633 and more_types_in_union_112634):
                # SSA join for if statement (line 620)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 612)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to convert_path_to_polygons(...): (line 627)
        # Processing the call arguments (line 627)
        # Getting the type of 'self' (line 628)
        self_112645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 12), 'self', False)
        # Getting the type of 'transform' (line 628)
        transform_112646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 18), 'transform', False)
        # Getting the type of 'width' (line 628)
        width_112647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 29), 'width', False)
        # Getting the type of 'height' (line 628)
        height_112648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 36), 'height', False)
        # Getting the type of 'closed_only' (line 628)
        closed_only_112649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 44), 'closed_only', False)
        # Processing the call keyword arguments (line 627)
        kwargs_112650 = {}
        # Getting the type of '_path' (line 627)
        _path_112643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 15), '_path', False)
        # Obtaining the member 'convert_path_to_polygons' of a type (line 627)
        convert_path_to_polygons_112644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 15), _path_112643, 'convert_path_to_polygons')
        # Calling convert_path_to_polygons(args, kwargs) (line 627)
        convert_path_to_polygons_call_result_112651 = invoke(stypy.reporting.localization.Localization(__file__, 627, 15), convert_path_to_polygons_112644, *[self_112645, transform_112646, width_112647, height_112648, closed_only_112649], **kwargs_112650)
        
        # Assigning a type to the variable 'stypy_return_type' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'stypy_return_type', convert_path_to_polygons_call_result_112651)
        
        # ################# End of 'to_polygons(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'to_polygons' in the type store
        # Getting the type of 'stypy_return_type' (line 586)
        stypy_return_type_112652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112652)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'to_polygons'
        return stypy_return_type_112652

    
    # Assigning a Name to a Name (line 630):

    @norecursion
    def unit_rectangle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unit_rectangle'
        module_type_store = module_type_store.open_function_context('unit_rectangle', 632, 4, False)
        # Assigning a type to the variable 'self' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.unit_rectangle.__dict__.__setitem__('stypy_localization', localization)
        Path.unit_rectangle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.unit_rectangle.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.unit_rectangle.__dict__.__setitem__('stypy_function_name', 'Path.unit_rectangle')
        Path.unit_rectangle.__dict__.__setitem__('stypy_param_names_list', [])
        Path.unit_rectangle.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.unit_rectangle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.unit_rectangle.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.unit_rectangle.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.unit_rectangle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.unit_rectangle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.unit_rectangle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unit_rectangle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unit_rectangle(...)' code ##################

        unicode_112653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, (-1)), 'unicode', u'\n        Return a :class:`Path` instance of the unit rectangle\n        from (0, 0) to (1, 1).\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 638)
        # Getting the type of 'cls' (line 638)
        cls_112654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 11), 'cls')
        # Obtaining the member '_unit_rectangle' of a type (line 638)
        _unit_rectangle_112655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 11), cls_112654, '_unit_rectangle')
        # Getting the type of 'None' (line 638)
        None_112656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 34), 'None')
        
        (may_be_112657, more_types_in_union_112658) = may_be_none(_unit_rectangle_112655, None_112656)

        if may_be_112657:

            if more_types_in_union_112658:
                # Runtime conditional SSA (line 638)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 639):
            
            # Assigning a Call to a Attribute (line 639):
            
            # Call to cls(...): (line 640)
            # Processing the call arguments (line 640)
            
            # Obtaining an instance of the builtin type 'list' (line 640)
            list_112660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 640)
            # Adding element type (line 640)
            
            # Obtaining an instance of the builtin type 'list' (line 640)
            list_112661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 21), 'list')
            # Adding type elements to the builtin type 'list' instance (line 640)
            # Adding element type (line 640)
            float_112662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 22), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 21), list_112661, float_112662)
            # Adding element type (line 640)
            float_112663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 27), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 21), list_112661, float_112663)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 20), list_112660, list_112661)
            # Adding element type (line 640)
            
            # Obtaining an instance of the builtin type 'list' (line 640)
            list_112664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 640)
            # Adding element type (line 640)
            float_112665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 34), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 33), list_112664, float_112665)
            # Adding element type (line 640)
            float_112666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 39), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 33), list_112664, float_112666)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 20), list_112660, list_112664)
            # Adding element type (line 640)
            
            # Obtaining an instance of the builtin type 'list' (line 640)
            list_112667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 45), 'list')
            # Adding type elements to the builtin type 'list' instance (line 640)
            # Adding element type (line 640)
            float_112668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 46), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 45), list_112667, float_112668)
            # Adding element type (line 640)
            float_112669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 51), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 45), list_112667, float_112669)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 20), list_112660, list_112667)
            # Adding element type (line 640)
            
            # Obtaining an instance of the builtin type 'list' (line 640)
            list_112670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 57), 'list')
            # Adding type elements to the builtin type 'list' instance (line 640)
            # Adding element type (line 640)
            float_112671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 58), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 57), list_112670, float_112671)
            # Adding element type (line 640)
            float_112672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 63), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 57), list_112670, float_112672)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 20), list_112660, list_112670)
            # Adding element type (line 640)
            
            # Obtaining an instance of the builtin type 'list' (line 641)
            list_112673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 21), 'list')
            # Adding type elements to the builtin type 'list' instance (line 641)
            # Adding element type (line 641)
            float_112674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 22), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 21), list_112673, float_112674)
            # Adding element type (line 641)
            float_112675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 27), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 21), list_112673, float_112675)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 20), list_112660, list_112673)
            
            
            # Obtaining an instance of the builtin type 'list' (line 642)
            list_112676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 642)
            # Adding element type (line 642)
            # Getting the type of 'cls' (line 642)
            cls_112677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 21), 'cls', False)
            # Obtaining the member 'MOVETO' of a type (line 642)
            MOVETO_112678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 21), cls_112677, 'MOVETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 20), list_112676, MOVETO_112678)
            # Adding element type (line 642)
            # Getting the type of 'cls' (line 642)
            cls_112679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 33), 'cls', False)
            # Obtaining the member 'LINETO' of a type (line 642)
            LINETO_112680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 33), cls_112679, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 20), list_112676, LINETO_112680)
            # Adding element type (line 642)
            # Getting the type of 'cls' (line 642)
            cls_112681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 45), 'cls', False)
            # Obtaining the member 'LINETO' of a type (line 642)
            LINETO_112682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 45), cls_112681, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 20), list_112676, LINETO_112682)
            # Adding element type (line 642)
            # Getting the type of 'cls' (line 642)
            cls_112683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 57), 'cls', False)
            # Obtaining the member 'LINETO' of a type (line 642)
            LINETO_112684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 57), cls_112683, 'LINETO')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 20), list_112676, LINETO_112684)
            # Adding element type (line 642)
            # Getting the type of 'cls' (line 643)
            cls_112685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 21), 'cls', False)
            # Obtaining the member 'CLOSEPOLY' of a type (line 643)
            CLOSEPOLY_112686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 21), cls_112685, 'CLOSEPOLY')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 20), list_112676, CLOSEPOLY_112686)
            
            # Processing the call keyword arguments (line 640)
            # Getting the type of 'True' (line 644)
            True_112687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 29), 'True', False)
            keyword_112688 = True_112687
            kwargs_112689 = {'readonly': keyword_112688}
            # Getting the type of 'cls' (line 640)
            cls_112659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 16), 'cls', False)
            # Calling cls(args, kwargs) (line 640)
            cls_call_result_112690 = invoke(stypy.reporting.localization.Localization(__file__, 640, 16), cls_112659, *[list_112660, list_112676], **kwargs_112689)
            
            # Getting the type of 'cls' (line 639)
            cls_112691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'cls')
            # Setting the type of the member '_unit_rectangle' of a type (line 639)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 12), cls_112691, '_unit_rectangle', cls_call_result_112690)

            if more_types_in_union_112658:
                # SSA join for if statement (line 638)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'cls' (line 645)
        cls_112692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 15), 'cls')
        # Obtaining the member '_unit_rectangle' of a type (line 645)
        _unit_rectangle_112693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 15), cls_112692, '_unit_rectangle')
        # Assigning a type to the variable 'stypy_return_type' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'stypy_return_type', _unit_rectangle_112693)
        
        # ################# End of 'unit_rectangle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unit_rectangle' in the type store
        # Getting the type of 'stypy_return_type' (line 632)
        stypy_return_type_112694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112694)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unit_rectangle'
        return stypy_return_type_112694

    
    # Assigning a Call to a Name (line 647):

    @norecursion
    def unit_regular_polygon(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unit_regular_polygon'
        module_type_store = module_type_store.open_function_context('unit_regular_polygon', 649, 4, False)
        # Assigning a type to the variable 'self' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.unit_regular_polygon.__dict__.__setitem__('stypy_localization', localization)
        Path.unit_regular_polygon.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.unit_regular_polygon.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.unit_regular_polygon.__dict__.__setitem__('stypy_function_name', 'Path.unit_regular_polygon')
        Path.unit_regular_polygon.__dict__.__setitem__('stypy_param_names_list', ['numVertices'])
        Path.unit_regular_polygon.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.unit_regular_polygon.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.unit_regular_polygon.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.unit_regular_polygon.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.unit_regular_polygon.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.unit_regular_polygon.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.unit_regular_polygon', ['numVertices'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unit_regular_polygon', localization, ['numVertices'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unit_regular_polygon(...)' code ##################

        unicode_112695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, (-1)), 'unicode', u'\n        Return a :class:`Path` instance for a unit regular\n        polygon with the given *numVertices* and radius of 1.0,\n        centered at (0, 0).\n        ')
        
        
        # Getting the type of 'numVertices' (line 656)
        numVertices_112696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 11), 'numVertices')
        int_112697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 26), 'int')
        # Applying the binary operator '<=' (line 656)
        result_le_112698 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 11), '<=', numVertices_112696, int_112697)
        
        # Testing the type of an if condition (line 656)
        if_condition_112699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 656, 8), result_le_112698)
        # Assigning a type to the variable 'if_condition_112699' (line 656)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'if_condition_112699', if_condition_112699)
        # SSA begins for if statement (line 656)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 657):
        
        # Assigning a Call to a Name (line 657):
        
        # Call to get(...): (line 657)
        # Processing the call arguments (line 657)
        # Getting the type of 'numVertices' (line 657)
        numVertices_112703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 50), 'numVertices', False)
        # Processing the call keyword arguments (line 657)
        kwargs_112704 = {}
        # Getting the type of 'cls' (line 657)
        cls_112700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 19), 'cls', False)
        # Obtaining the member '_unit_regular_polygons' of a type (line 657)
        _unit_regular_polygons_112701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 19), cls_112700, '_unit_regular_polygons')
        # Obtaining the member 'get' of a type (line 657)
        get_112702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 19), _unit_regular_polygons_112701, 'get')
        # Calling get(args, kwargs) (line 657)
        get_call_result_112705 = invoke(stypy.reporting.localization.Localization(__file__, 657, 19), get_112702, *[numVertices_112703], **kwargs_112704)
        
        # Assigning a type to the variable 'path' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'path', get_call_result_112705)
        # SSA branch for the else part of an if statement (line 656)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 659):
        
        # Assigning a Name to a Name (line 659):
        # Getting the type of 'None' (line 659)
        None_112706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 19), 'None')
        # Assigning a type to the variable 'path' (line 659)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'path', None_112706)
        # SSA join for if statement (line 656)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 660)
        # Getting the type of 'path' (line 660)
        path_112707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 11), 'path')
        # Getting the type of 'None' (line 660)
        None_112708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 19), 'None')
        
        (may_be_112709, more_types_in_union_112710) = may_be_none(path_112707, None_112708)

        if may_be_112709:

            if more_types_in_union_112710:
                # Runtime conditional SSA (line 660)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 661):
            
            # Assigning a BinOp to a Name (line 661):
            int_112711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 21), 'int')
            # Getting the type of 'np' (line 661)
            np_112712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 23), 'np')
            # Obtaining the member 'pi' of a type (line 661)
            pi_112713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 23), np_112712, 'pi')
            # Applying the binary operator '*' (line 661)
            result_mul_112714 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 21), '*', int_112711, pi_112713)
            
            # Getting the type of 'numVertices' (line 661)
            numVertices_112715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 29), 'numVertices')
            # Applying the binary operator 'div' (line 661)
            result_div_112716 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 28), 'div', result_mul_112714, numVertices_112715)
            
            
            # Call to reshape(...): (line 662)
            # Processing the call arguments (line 662)
            
            # Obtaining an instance of the builtin type 'tuple' (line 662)
            tuple_112725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 57), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 662)
            # Adding element type (line 662)
            # Getting the type of 'numVertices' (line 662)
            numVertices_112726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 57), 'numVertices', False)
            int_112727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 71), 'int')
            # Applying the binary operator '+' (line 662)
            result_add_112728 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 57), '+', numVertices_112726, int_112727)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 57), tuple_112725, result_add_112728)
            # Adding element type (line 662)
            int_112729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 74), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 57), tuple_112725, int_112729)
            
            # Processing the call keyword arguments (line 662)
            kwargs_112730 = {}
            
            # Call to arange(...): (line 662)
            # Processing the call arguments (line 662)
            # Getting the type of 'numVertices' (line 662)
            numVertices_112719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 31), 'numVertices', False)
            int_112720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 45), 'int')
            # Applying the binary operator '+' (line 662)
            result_add_112721 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 31), '+', numVertices_112719, int_112720)
            
            # Processing the call keyword arguments (line 662)
            kwargs_112722 = {}
            # Getting the type of 'np' (line 662)
            np_112717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 21), 'np', False)
            # Obtaining the member 'arange' of a type (line 662)
            arange_112718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 21), np_112717, 'arange')
            # Calling arange(args, kwargs) (line 662)
            arange_call_result_112723 = invoke(stypy.reporting.localization.Localization(__file__, 662, 21), arange_112718, *[result_add_112721], **kwargs_112722)
            
            # Obtaining the member 'reshape' of a type (line 662)
            reshape_112724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 21), arange_call_result_112723, 'reshape')
            # Calling reshape(args, kwargs) (line 662)
            reshape_call_result_112731 = invoke(stypy.reporting.localization.Localization(__file__, 662, 21), reshape_112724, *[tuple_112725], **kwargs_112730)
            
            # Applying the binary operator '*' (line 661)
            result_mul_112732 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 41), '*', result_div_112716, reshape_call_result_112731)
            
            # Assigning a type to the variable 'theta' (line 661)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'theta', result_mul_112732)
            
            # Getting the type of 'theta' (line 665)
            theta_112733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'theta')
            # Getting the type of 'np' (line 665)
            np_112734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 21), 'np')
            # Obtaining the member 'pi' of a type (line 665)
            pi_112735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 21), np_112734, 'pi')
            float_112736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 29), 'float')
            # Applying the binary operator 'div' (line 665)
            result_div_112737 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 21), 'div', pi_112735, float_112736)
            
            # Applying the binary operator '+=' (line 665)
            result_iadd_112738 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 12), '+=', theta_112733, result_div_112737)
            # Assigning a type to the variable 'theta' (line 665)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'theta', result_iadd_112738)
            
            
            # Assigning a Call to a Name (line 666):
            
            # Assigning a Call to a Name (line 666):
            
            # Call to concatenate(...): (line 666)
            # Processing the call arguments (line 666)
            
            # Obtaining an instance of the builtin type 'tuple' (line 666)
            tuple_112741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 36), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 666)
            # Adding element type (line 666)
            
            # Call to cos(...): (line 666)
            # Processing the call arguments (line 666)
            # Getting the type of 'theta' (line 666)
            theta_112744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 43), 'theta', False)
            # Processing the call keyword arguments (line 666)
            kwargs_112745 = {}
            # Getting the type of 'np' (line 666)
            np_112742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 36), 'np', False)
            # Obtaining the member 'cos' of a type (line 666)
            cos_112743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 36), np_112742, 'cos')
            # Calling cos(args, kwargs) (line 666)
            cos_call_result_112746 = invoke(stypy.reporting.localization.Localization(__file__, 666, 36), cos_112743, *[theta_112744], **kwargs_112745)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 36), tuple_112741, cos_call_result_112746)
            # Adding element type (line 666)
            
            # Call to sin(...): (line 666)
            # Processing the call arguments (line 666)
            # Getting the type of 'theta' (line 666)
            theta_112749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 58), 'theta', False)
            # Processing the call keyword arguments (line 666)
            kwargs_112750 = {}
            # Getting the type of 'np' (line 666)
            np_112747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 51), 'np', False)
            # Obtaining the member 'sin' of a type (line 666)
            sin_112748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 51), np_112747, 'sin')
            # Calling sin(args, kwargs) (line 666)
            sin_call_result_112751 = invoke(stypy.reporting.localization.Localization(__file__, 666, 51), sin_112748, *[theta_112749], **kwargs_112750)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 36), tuple_112741, sin_call_result_112751)
            
            int_112752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 67), 'int')
            # Processing the call keyword arguments (line 666)
            kwargs_112753 = {}
            # Getting the type of 'np' (line 666)
            np_112739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 20), 'np', False)
            # Obtaining the member 'concatenate' of a type (line 666)
            concatenate_112740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 20), np_112739, 'concatenate')
            # Calling concatenate(args, kwargs) (line 666)
            concatenate_call_result_112754 = invoke(stypy.reporting.localization.Localization(__file__, 666, 20), concatenate_112740, *[tuple_112741, int_112752], **kwargs_112753)
            
            # Assigning a type to the variable 'verts' (line 666)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'verts', concatenate_call_result_112754)
            
            # Assigning a Call to a Name (line 667):
            
            # Assigning a Call to a Name (line 667):
            
            # Call to empty(...): (line 667)
            # Processing the call arguments (line 667)
            
            # Obtaining an instance of the builtin type 'tuple' (line 667)
            tuple_112757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 667)
            # Adding element type (line 667)
            # Getting the type of 'numVertices' (line 667)
            numVertices_112758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 30), 'numVertices', False)
            int_112759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 44), 'int')
            # Applying the binary operator '+' (line 667)
            result_add_112760 = python_operator(stypy.reporting.localization.Localization(__file__, 667, 30), '+', numVertices_112758, int_112759)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 30), tuple_112757, result_add_112760)
            
            # Processing the call keyword arguments (line 667)
            kwargs_112761 = {}
            # Getting the type of 'np' (line 667)
            np_112755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 20), 'np', False)
            # Obtaining the member 'empty' of a type (line 667)
            empty_112756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 20), np_112755, 'empty')
            # Calling empty(args, kwargs) (line 667)
            empty_call_result_112762 = invoke(stypy.reporting.localization.Localization(__file__, 667, 20), empty_112756, *[tuple_112757], **kwargs_112761)
            
            # Assigning a type to the variable 'codes' (line 667)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'codes', empty_call_result_112762)
            
            # Assigning a Attribute to a Subscript (line 668):
            
            # Assigning a Attribute to a Subscript (line 668):
            # Getting the type of 'cls' (line 668)
            cls_112763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 23), 'cls')
            # Obtaining the member 'MOVETO' of a type (line 668)
            MOVETO_112764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 23), cls_112763, 'MOVETO')
            # Getting the type of 'codes' (line 668)
            codes_112765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), 'codes')
            int_112766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 18), 'int')
            # Storing an element on a container (line 668)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 668, 12), codes_112765, (int_112766, MOVETO_112764))
            
            # Assigning a Attribute to a Subscript (line 669):
            
            # Assigning a Attribute to a Subscript (line 669):
            # Getting the type of 'cls' (line 669)
            cls_112767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 26), 'cls')
            # Obtaining the member 'LINETO' of a type (line 669)
            LINETO_112768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 26), cls_112767, 'LINETO')
            # Getting the type of 'codes' (line 669)
            codes_112769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'codes')
            int_112770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 18), 'int')
            int_112771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 20), 'int')
            slice_112772 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 669, 12), int_112770, int_112771, None)
            # Storing an element on a container (line 669)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 12), codes_112769, (slice_112772, LINETO_112768))
            
            # Assigning a Attribute to a Subscript (line 670):
            
            # Assigning a Attribute to a Subscript (line 670):
            # Getting the type of 'cls' (line 670)
            cls_112773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 24), 'cls')
            # Obtaining the member 'CLOSEPOLY' of a type (line 670)
            CLOSEPOLY_112774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 24), cls_112773, 'CLOSEPOLY')
            # Getting the type of 'codes' (line 670)
            codes_112775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), 'codes')
            int_112776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 18), 'int')
            # Storing an element on a container (line 670)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 670, 12), codes_112775, (int_112776, CLOSEPOLY_112774))
            
            # Assigning a Call to a Name (line 671):
            
            # Assigning a Call to a Name (line 671):
            
            # Call to cls(...): (line 671)
            # Processing the call arguments (line 671)
            # Getting the type of 'verts' (line 671)
            verts_112778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 23), 'verts', False)
            # Getting the type of 'codes' (line 671)
            codes_112779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 30), 'codes', False)
            # Processing the call keyword arguments (line 671)
            # Getting the type of 'True' (line 671)
            True_112780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 46), 'True', False)
            keyword_112781 = True_112780
            kwargs_112782 = {'readonly': keyword_112781}
            # Getting the type of 'cls' (line 671)
            cls_112777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 19), 'cls', False)
            # Calling cls(args, kwargs) (line 671)
            cls_call_result_112783 = invoke(stypy.reporting.localization.Localization(__file__, 671, 19), cls_112777, *[verts_112778, codes_112779], **kwargs_112782)
            
            # Assigning a type to the variable 'path' (line 671)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 12), 'path', cls_call_result_112783)
            
            
            # Getting the type of 'numVertices' (line 672)
            numVertices_112784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 15), 'numVertices')
            int_112785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 30), 'int')
            # Applying the binary operator '<=' (line 672)
            result_le_112786 = python_operator(stypy.reporting.localization.Localization(__file__, 672, 15), '<=', numVertices_112784, int_112785)
            
            # Testing the type of an if condition (line 672)
            if_condition_112787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 672, 12), result_le_112786)
            # Assigning a type to the variable 'if_condition_112787' (line 672)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'if_condition_112787', if_condition_112787)
            # SSA begins for if statement (line 672)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Subscript (line 673):
            
            # Assigning a Name to a Subscript (line 673):
            # Getting the type of 'path' (line 673)
            path_112788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 58), 'path')
            # Getting the type of 'cls' (line 673)
            cls_112789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 16), 'cls')
            # Obtaining the member '_unit_regular_polygons' of a type (line 673)
            _unit_regular_polygons_112790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 16), cls_112789, '_unit_regular_polygons')
            # Getting the type of 'numVertices' (line 673)
            numVertices_112791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 43), 'numVertices')
            # Storing an element on a container (line 673)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 673, 16), _unit_regular_polygons_112790, (numVertices_112791, path_112788))
            # SSA join for if statement (line 672)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_112710:
                # SSA join for if statement (line 660)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'path' (line 674)
        path_112792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 15), 'path')
        # Assigning a type to the variable 'stypy_return_type' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'stypy_return_type', path_112792)
        
        # ################# End of 'unit_regular_polygon(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unit_regular_polygon' in the type store
        # Getting the type of 'stypy_return_type' (line 649)
        stypy_return_type_112793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112793)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unit_regular_polygon'
        return stypy_return_type_112793

    
    # Assigning a Call to a Name (line 676):

    @norecursion
    def unit_regular_star(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_112794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 56), 'float')
        defaults = [float_112794]
        # Create a new context for function 'unit_regular_star'
        module_type_store = module_type_store.open_function_context('unit_regular_star', 678, 4, False)
        # Assigning a type to the variable 'self' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.unit_regular_star.__dict__.__setitem__('stypy_localization', localization)
        Path.unit_regular_star.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.unit_regular_star.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.unit_regular_star.__dict__.__setitem__('stypy_function_name', 'Path.unit_regular_star')
        Path.unit_regular_star.__dict__.__setitem__('stypy_param_names_list', ['numVertices', 'innerCircle'])
        Path.unit_regular_star.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.unit_regular_star.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.unit_regular_star.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.unit_regular_star.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.unit_regular_star.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.unit_regular_star.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.unit_regular_star', ['numVertices', 'innerCircle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unit_regular_star', localization, ['numVertices', 'innerCircle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unit_regular_star(...)' code ##################

        unicode_112795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, (-1)), 'unicode', u'\n        Return a :class:`Path` for a unit regular star\n        with the given numVertices and radius of 1.0, centered at (0,\n        0).\n        ')
        
        
        # Getting the type of 'numVertices' (line 685)
        numVertices_112796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 11), 'numVertices')
        int_112797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 26), 'int')
        # Applying the binary operator '<=' (line 685)
        result_le_112798 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 11), '<=', numVertices_112796, int_112797)
        
        # Testing the type of an if condition (line 685)
        if_condition_112799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 685, 8), result_le_112798)
        # Assigning a type to the variable 'if_condition_112799' (line 685)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 8), 'if_condition_112799', if_condition_112799)
        # SSA begins for if statement (line 685)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 686):
        
        # Assigning a Call to a Name (line 686):
        
        # Call to get(...): (line 686)
        # Processing the call arguments (line 686)
        
        # Obtaining an instance of the builtin type 'tuple' (line 686)
        tuple_112803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 686)
        # Adding element type (line 686)
        # Getting the type of 'numVertices' (line 686)
        numVertices_112804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 48), 'numVertices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 48), tuple_112803, numVertices_112804)
        # Adding element type (line 686)
        # Getting the type of 'innerCircle' (line 686)
        innerCircle_112805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 61), 'innerCircle', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 686, 48), tuple_112803, innerCircle_112805)
        
        # Processing the call keyword arguments (line 686)
        kwargs_112806 = {}
        # Getting the type of 'cls' (line 686)
        cls_112800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 19), 'cls', False)
        # Obtaining the member '_unit_regular_stars' of a type (line 686)
        _unit_regular_stars_112801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 19), cls_112800, '_unit_regular_stars')
        # Obtaining the member 'get' of a type (line 686)
        get_112802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 19), _unit_regular_stars_112801, 'get')
        # Calling get(args, kwargs) (line 686)
        get_call_result_112807 = invoke(stypy.reporting.localization.Localization(__file__, 686, 19), get_112802, *[tuple_112803], **kwargs_112806)
        
        # Assigning a type to the variable 'path' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 12), 'path', get_call_result_112807)
        # SSA branch for the else part of an if statement (line 685)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 688):
        
        # Assigning a Name to a Name (line 688):
        # Getting the type of 'None' (line 688)
        None_112808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 19), 'None')
        # Assigning a type to the variable 'path' (line 688)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 12), 'path', None_112808)
        # SSA join for if statement (line 685)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 689)
        # Getting the type of 'path' (line 689)
        path_112809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 11), 'path')
        # Getting the type of 'None' (line 689)
        None_112810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 19), 'None')
        
        (may_be_112811, more_types_in_union_112812) = may_be_none(path_112809, None_112810)

        if may_be_112811:

            if more_types_in_union_112812:
                # Runtime conditional SSA (line 689)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 690):
            
            # Assigning a BinOp to a Name (line 690):
            # Getting the type of 'numVertices' (line 690)
            numVertices_112813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 18), 'numVertices')
            int_112814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 32), 'int')
            # Applying the binary operator '*' (line 690)
            result_mul_112815 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 18), '*', numVertices_112813, int_112814)
            
            # Assigning a type to the variable 'ns2' (line 690)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 12), 'ns2', result_mul_112815)
            
            # Assigning a BinOp to a Name (line 691):
            
            # Assigning a BinOp to a Name (line 691):
            int_112816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 21), 'int')
            # Getting the type of 'np' (line 691)
            np_112817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 23), 'np')
            # Obtaining the member 'pi' of a type (line 691)
            pi_112818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 23), np_112817, 'pi')
            # Applying the binary operator '*' (line 691)
            result_mul_112819 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 21), '*', int_112816, pi_112818)
            
            # Getting the type of 'ns2' (line 691)
            ns2_112820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 29), 'ns2')
            # Applying the binary operator 'div' (line 691)
            result_div_112821 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 28), 'div', result_mul_112819, ns2_112820)
            
            
            # Call to arange(...): (line 691)
            # Processing the call arguments (line 691)
            # Getting the type of 'ns2' (line 691)
            ns2_112824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 45), 'ns2', False)
            int_112825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 51), 'int')
            # Applying the binary operator '+' (line 691)
            result_add_112826 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 45), '+', ns2_112824, int_112825)
            
            # Processing the call keyword arguments (line 691)
            kwargs_112827 = {}
            # Getting the type of 'np' (line 691)
            np_112822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 35), 'np', False)
            # Obtaining the member 'arange' of a type (line 691)
            arange_112823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 35), np_112822, 'arange')
            # Calling arange(args, kwargs) (line 691)
            arange_call_result_112828 = invoke(stypy.reporting.localization.Localization(__file__, 691, 35), arange_112823, *[result_add_112826], **kwargs_112827)
            
            # Applying the binary operator '*' (line 691)
            result_mul_112829 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 33), '*', result_div_112821, arange_call_result_112828)
            
            # Assigning a type to the variable 'theta' (line 691)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 12), 'theta', result_mul_112829)
            
            # Getting the type of 'theta' (line 694)
            theta_112830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 12), 'theta')
            # Getting the type of 'np' (line 694)
            np_112831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 21), 'np')
            # Obtaining the member 'pi' of a type (line 694)
            pi_112832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 21), np_112831, 'pi')
            float_112833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 29), 'float')
            # Applying the binary operator 'div' (line 694)
            result_div_112834 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 21), 'div', pi_112832, float_112833)
            
            # Applying the binary operator '+=' (line 694)
            result_iadd_112835 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 12), '+=', theta_112830, result_div_112834)
            # Assigning a type to the variable 'theta' (line 694)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 12), 'theta', result_iadd_112835)
            
            
            # Assigning a Call to a Name (line 695):
            
            # Assigning a Call to a Name (line 695):
            
            # Call to ones(...): (line 695)
            # Processing the call arguments (line 695)
            # Getting the type of 'ns2' (line 695)
            ns2_112838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 24), 'ns2', False)
            int_112839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 30), 'int')
            # Applying the binary operator '+' (line 695)
            result_add_112840 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 24), '+', ns2_112838, int_112839)
            
            # Processing the call keyword arguments (line 695)
            kwargs_112841 = {}
            # Getting the type of 'np' (line 695)
            np_112836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'np', False)
            # Obtaining the member 'ones' of a type (line 695)
            ones_112837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 16), np_112836, 'ones')
            # Calling ones(args, kwargs) (line 695)
            ones_call_result_112842 = invoke(stypy.reporting.localization.Localization(__file__, 695, 16), ones_112837, *[result_add_112840], **kwargs_112841)
            
            # Assigning a type to the variable 'r' (line 695)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 12), 'r', ones_call_result_112842)
            
            # Assigning a Name to a Subscript (line 696):
            
            # Assigning a Name to a Subscript (line 696):
            # Getting the type of 'innerCircle' (line 696)
            innerCircle_112843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 22), 'innerCircle')
            # Getting the type of 'r' (line 696)
            r_112844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 12), 'r')
            int_112845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 14), 'int')
            int_112846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 17), 'int')
            slice_112847 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 696, 12), int_112845, None, int_112846)
            # Storing an element on a container (line 696)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 696, 12), r_112844, (slice_112847, innerCircle_112843))
            
            # Assigning a Call to a Name (line 697):
            
            # Assigning a Call to a Name (line 697):
            
            # Call to transpose(...): (line 697)
            # Processing the call keyword arguments (line 697)
            kwargs_112868 = {}
            
            # Call to vstack(...): (line 697)
            # Processing the call arguments (line 697)
            
            # Obtaining an instance of the builtin type 'tuple' (line 697)
            tuple_112850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 697)
            # Adding element type (line 697)
            # Getting the type of 'r' (line 697)
            r_112851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 31), 'r', False)
            
            # Call to cos(...): (line 697)
            # Processing the call arguments (line 697)
            # Getting the type of 'theta' (line 697)
            theta_112854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 40), 'theta', False)
            # Processing the call keyword arguments (line 697)
            kwargs_112855 = {}
            # Getting the type of 'np' (line 697)
            np_112852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 33), 'np', False)
            # Obtaining the member 'cos' of a type (line 697)
            cos_112853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 33), np_112852, 'cos')
            # Calling cos(args, kwargs) (line 697)
            cos_call_result_112856 = invoke(stypy.reporting.localization.Localization(__file__, 697, 33), cos_112853, *[theta_112854], **kwargs_112855)
            
            # Applying the binary operator '*' (line 697)
            result_mul_112857 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 31), '*', r_112851, cos_call_result_112856)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 31), tuple_112850, result_mul_112857)
            # Adding element type (line 697)
            # Getting the type of 'r' (line 697)
            r_112858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 48), 'r', False)
            
            # Call to sin(...): (line 697)
            # Processing the call arguments (line 697)
            # Getting the type of 'theta' (line 697)
            theta_112861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 57), 'theta', False)
            # Processing the call keyword arguments (line 697)
            kwargs_112862 = {}
            # Getting the type of 'np' (line 697)
            np_112859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 50), 'np', False)
            # Obtaining the member 'sin' of a type (line 697)
            sin_112860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 50), np_112859, 'sin')
            # Calling sin(args, kwargs) (line 697)
            sin_call_result_112863 = invoke(stypy.reporting.localization.Localization(__file__, 697, 50), sin_112860, *[theta_112861], **kwargs_112862)
            
            # Applying the binary operator '*' (line 697)
            result_mul_112864 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 48), '*', r_112858, sin_call_result_112863)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 31), tuple_112850, result_mul_112864)
            
            # Processing the call keyword arguments (line 697)
            kwargs_112865 = {}
            # Getting the type of 'np' (line 697)
            np_112848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 20), 'np', False)
            # Obtaining the member 'vstack' of a type (line 697)
            vstack_112849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 20), np_112848, 'vstack')
            # Calling vstack(args, kwargs) (line 697)
            vstack_call_result_112866 = invoke(stypy.reporting.localization.Localization(__file__, 697, 20), vstack_112849, *[tuple_112850], **kwargs_112865)
            
            # Obtaining the member 'transpose' of a type (line 697)
            transpose_112867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 20), vstack_call_result_112866, 'transpose')
            # Calling transpose(args, kwargs) (line 697)
            transpose_call_result_112869 = invoke(stypy.reporting.localization.Localization(__file__, 697, 20), transpose_112867, *[], **kwargs_112868)
            
            # Assigning a type to the variable 'verts' (line 697)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 12), 'verts', transpose_call_result_112869)
            
            # Assigning a Call to a Name (line 698):
            
            # Assigning a Call to a Name (line 698):
            
            # Call to empty(...): (line 698)
            # Processing the call arguments (line 698)
            
            # Obtaining an instance of the builtin type 'tuple' (line 698)
            tuple_112872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 698)
            # Adding element type (line 698)
            # Getting the type of 'ns2' (line 698)
            ns2_112873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 30), 'ns2', False)
            int_112874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 36), 'int')
            # Applying the binary operator '+' (line 698)
            result_add_112875 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 30), '+', ns2_112873, int_112874)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 698, 30), tuple_112872, result_add_112875)
            
            # Processing the call keyword arguments (line 698)
            kwargs_112876 = {}
            # Getting the type of 'np' (line 698)
            np_112870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 20), 'np', False)
            # Obtaining the member 'empty' of a type (line 698)
            empty_112871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 20), np_112870, 'empty')
            # Calling empty(args, kwargs) (line 698)
            empty_call_result_112877 = invoke(stypy.reporting.localization.Localization(__file__, 698, 20), empty_112871, *[tuple_112872], **kwargs_112876)
            
            # Assigning a type to the variable 'codes' (line 698)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 12), 'codes', empty_call_result_112877)
            
            # Assigning a Attribute to a Subscript (line 699):
            
            # Assigning a Attribute to a Subscript (line 699):
            # Getting the type of 'cls' (line 699)
            cls_112878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 23), 'cls')
            # Obtaining the member 'MOVETO' of a type (line 699)
            MOVETO_112879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 23), cls_112878, 'MOVETO')
            # Getting the type of 'codes' (line 699)
            codes_112880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'codes')
            int_112881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 18), 'int')
            # Storing an element on a container (line 699)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 699, 12), codes_112880, (int_112881, MOVETO_112879))
            
            # Assigning a Attribute to a Subscript (line 700):
            
            # Assigning a Attribute to a Subscript (line 700):
            # Getting the type of 'cls' (line 700)
            cls_112882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 26), 'cls')
            # Obtaining the member 'LINETO' of a type (line 700)
            LINETO_112883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 26), cls_112882, 'LINETO')
            # Getting the type of 'codes' (line 700)
            codes_112884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'codes')
            int_112885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 18), 'int')
            int_112886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 20), 'int')
            slice_112887 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 700, 12), int_112885, int_112886, None)
            # Storing an element on a container (line 700)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 700, 12), codes_112884, (slice_112887, LINETO_112883))
            
            # Assigning a Attribute to a Subscript (line 701):
            
            # Assigning a Attribute to a Subscript (line 701):
            # Getting the type of 'cls' (line 701)
            cls_112888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 24), 'cls')
            # Obtaining the member 'CLOSEPOLY' of a type (line 701)
            CLOSEPOLY_112889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 24), cls_112888, 'CLOSEPOLY')
            # Getting the type of 'codes' (line 701)
            codes_112890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 12), 'codes')
            int_112891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 18), 'int')
            # Storing an element on a container (line 701)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 701, 12), codes_112890, (int_112891, CLOSEPOLY_112889))
            
            # Assigning a Call to a Name (line 702):
            
            # Assigning a Call to a Name (line 702):
            
            # Call to cls(...): (line 702)
            # Processing the call arguments (line 702)
            # Getting the type of 'verts' (line 702)
            verts_112893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 23), 'verts', False)
            # Getting the type of 'codes' (line 702)
            codes_112894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 30), 'codes', False)
            # Processing the call keyword arguments (line 702)
            # Getting the type of 'True' (line 702)
            True_112895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 46), 'True', False)
            keyword_112896 = True_112895
            kwargs_112897 = {'readonly': keyword_112896}
            # Getting the type of 'cls' (line 702)
            cls_112892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 19), 'cls', False)
            # Calling cls(args, kwargs) (line 702)
            cls_call_result_112898 = invoke(stypy.reporting.localization.Localization(__file__, 702, 19), cls_112892, *[verts_112893, codes_112894], **kwargs_112897)
            
            # Assigning a type to the variable 'path' (line 702)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'path', cls_call_result_112898)
            
            
            # Getting the type of 'numVertices' (line 703)
            numVertices_112899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 15), 'numVertices')
            int_112900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 30), 'int')
            # Applying the binary operator '<=' (line 703)
            result_le_112901 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 15), '<=', numVertices_112899, int_112900)
            
            # Testing the type of an if condition (line 703)
            if_condition_112902 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 703, 12), result_le_112901)
            # Assigning a type to the variable 'if_condition_112902' (line 703)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 12), 'if_condition_112902', if_condition_112902)
            # SSA begins for if statement (line 703)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Subscript (line 704):
            
            # Assigning a Name to a Subscript (line 704):
            # Getting the type of 'path' (line 704)
            path_112903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 70), 'path')
            # Getting the type of 'cls' (line 704)
            cls_112904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 16), 'cls')
            # Obtaining the member '_unit_regular_stars' of a type (line 704)
            _unit_regular_stars_112905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 16), cls_112904, '_unit_regular_stars')
            
            # Obtaining an instance of the builtin type 'tuple' (line 704)
            tuple_112906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 41), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 704)
            # Adding element type (line 704)
            # Getting the type of 'numVertices' (line 704)
            numVertices_112907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 41), 'numVertices')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 704, 41), tuple_112906, numVertices_112907)
            # Adding element type (line 704)
            # Getting the type of 'innerCircle' (line 704)
            innerCircle_112908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 54), 'innerCircle')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 704, 41), tuple_112906, innerCircle_112908)
            
            # Storing an element on a container (line 704)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 704, 16), _unit_regular_stars_112905, (tuple_112906, path_112903))
            # SSA join for if statement (line 703)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_112812:
                # SSA join for if statement (line 689)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'path' (line 705)
        path_112909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 15), 'path')
        # Assigning a type to the variable 'stypy_return_type' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'stypy_return_type', path_112909)
        
        # ################# End of 'unit_regular_star(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unit_regular_star' in the type store
        # Getting the type of 'stypy_return_type' (line 678)
        stypy_return_type_112910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112910)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unit_regular_star'
        return stypy_return_type_112910


    @norecursion
    def unit_regular_asterisk(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unit_regular_asterisk'
        module_type_store = module_type_store.open_function_context('unit_regular_asterisk', 707, 4, False)
        # Assigning a type to the variable 'self' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.unit_regular_asterisk.__dict__.__setitem__('stypy_localization', localization)
        Path.unit_regular_asterisk.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.unit_regular_asterisk.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.unit_regular_asterisk.__dict__.__setitem__('stypy_function_name', 'Path.unit_regular_asterisk')
        Path.unit_regular_asterisk.__dict__.__setitem__('stypy_param_names_list', ['numVertices'])
        Path.unit_regular_asterisk.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.unit_regular_asterisk.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.unit_regular_asterisk.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.unit_regular_asterisk.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.unit_regular_asterisk.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.unit_regular_asterisk.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.unit_regular_asterisk', ['numVertices'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unit_regular_asterisk', localization, ['numVertices'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unit_regular_asterisk(...)' code ##################

        unicode_112911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, (-1)), 'unicode', u'\n        Return a :class:`Path` for a unit regular\n        asterisk with the given numVertices and radius of 1.0,\n        centered at (0, 0).\n        ')
        
        # Call to unit_regular_star(...): (line 714)
        # Processing the call arguments (line 714)
        # Getting the type of 'numVertices' (line 714)
        numVertices_112914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 37), 'numVertices', False)
        float_112915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 50), 'float')
        # Processing the call keyword arguments (line 714)
        kwargs_112916 = {}
        # Getting the type of 'cls' (line 714)
        cls_112912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 15), 'cls', False)
        # Obtaining the member 'unit_regular_star' of a type (line 714)
        unit_regular_star_112913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 15), cls_112912, 'unit_regular_star')
        # Calling unit_regular_star(args, kwargs) (line 714)
        unit_regular_star_call_result_112917 = invoke(stypy.reporting.localization.Localization(__file__, 714, 15), unit_regular_star_112913, *[numVertices_112914, float_112915], **kwargs_112916)
        
        # Assigning a type to the variable 'stypy_return_type' (line 714)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'stypy_return_type', unit_regular_star_call_result_112917)
        
        # ################# End of 'unit_regular_asterisk(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unit_regular_asterisk' in the type store
        # Getting the type of 'stypy_return_type' (line 707)
        stypy_return_type_112918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112918)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unit_regular_asterisk'
        return stypy_return_type_112918

    
    # Assigning a Name to a Name (line 716):

    @norecursion
    def unit_circle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unit_circle'
        module_type_store = module_type_store.open_function_context('unit_circle', 718, 4, False)
        # Assigning a type to the variable 'self' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.unit_circle.__dict__.__setitem__('stypy_localization', localization)
        Path.unit_circle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.unit_circle.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.unit_circle.__dict__.__setitem__('stypy_function_name', 'Path.unit_circle')
        Path.unit_circle.__dict__.__setitem__('stypy_param_names_list', [])
        Path.unit_circle.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.unit_circle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.unit_circle.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.unit_circle.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.unit_circle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.unit_circle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.unit_circle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unit_circle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unit_circle(...)' code ##################

        unicode_112919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, (-1)), 'unicode', u'\n        Return the readonly :class:`Path` of the unit circle.\n\n        For most cases, :func:`Path.circle` will be what you want.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 726)
        # Getting the type of 'cls' (line 726)
        cls_112920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 11), 'cls')
        # Obtaining the member '_unit_circle' of a type (line 726)
        _unit_circle_112921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 11), cls_112920, '_unit_circle')
        # Getting the type of 'None' (line 726)
        None_112922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 31), 'None')
        
        (may_be_112923, more_types_in_union_112924) = may_be_none(_unit_circle_112921, None_112922)

        if may_be_112923:

            if more_types_in_union_112924:
                # Runtime conditional SSA (line 726)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 727):
            
            # Assigning a Call to a Attribute (line 727):
            
            # Call to circle(...): (line 727)
            # Processing the call keyword arguments (line 727)
            
            # Obtaining an instance of the builtin type 'tuple' (line 727)
            tuple_112927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 50), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 727)
            # Adding element type (line 727)
            int_112928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 50), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 727, 50), tuple_112927, int_112928)
            # Adding element type (line 727)
            int_112929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 53), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 727, 50), tuple_112927, int_112929)
            
            keyword_112930 = tuple_112927
            int_112931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 64), 'int')
            keyword_112932 = int_112931
            # Getting the type of 'True' (line 728)
            True_112933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 51), 'True', False)
            keyword_112934 = True_112933
            kwargs_112935 = {'readonly': keyword_112934, 'radius': keyword_112932, 'center': keyword_112930}
            # Getting the type of 'cls' (line 727)
            cls_112925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 31), 'cls', False)
            # Obtaining the member 'circle' of a type (line 727)
            circle_112926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 31), cls_112925, 'circle')
            # Calling circle(args, kwargs) (line 727)
            circle_call_result_112936 = invoke(stypy.reporting.localization.Localization(__file__, 727, 31), circle_112926, *[], **kwargs_112935)
            
            # Getting the type of 'cls' (line 727)
            cls_112937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 12), 'cls')
            # Setting the type of the member '_unit_circle' of a type (line 727)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 12), cls_112937, '_unit_circle', circle_call_result_112936)

            if more_types_in_union_112924:
                # SSA join for if statement (line 726)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'cls' (line 729)
        cls_112938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 15), 'cls')
        # Obtaining the member '_unit_circle' of a type (line 729)
        _unit_circle_112939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 15), cls_112938, '_unit_circle')
        # Assigning a type to the variable 'stypy_return_type' (line 729)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 8), 'stypy_return_type', _unit_circle_112939)
        
        # ################# End of 'unit_circle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unit_circle' in the type store
        # Getting the type of 'stypy_return_type' (line 718)
        stypy_return_type_112940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112940)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unit_circle'
        return stypy_return_type_112940


    @norecursion
    def circle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'tuple' (line 732)
        tuple_112941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 732)
        # Adding element type (line 732)
        float_112942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 28), tuple_112941, float_112942)
        # Adding element type (line 732)
        float_112943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 28), tuple_112941, float_112943)
        
        float_112944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 44), 'float')
        # Getting the type of 'False' (line 732)
        False_112945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 57), 'False')
        defaults = [tuple_112941, float_112944, False_112945]
        # Create a new context for function 'circle'
        module_type_store = module_type_store.open_function_context('circle', 731, 4, False)
        # Assigning a type to the variable 'self' (line 732)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.circle.__dict__.__setitem__('stypy_localization', localization)
        Path.circle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.circle.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.circle.__dict__.__setitem__('stypy_function_name', 'Path.circle')
        Path.circle.__dict__.__setitem__('stypy_param_names_list', ['center', 'radius', 'readonly'])
        Path.circle.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.circle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.circle.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.circle.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.circle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.circle.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.circle', ['center', 'radius', 'readonly'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'circle', localization, ['center', 'radius', 'readonly'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'circle(...)' code ##################

        unicode_112946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, (-1)), 'unicode', u'\n        Return a Path representing a circle of a given radius and center.\n\n        Parameters\n        ----------\n        center : pair of floats\n            The center of the circle. Default ``(0, 0)``.\n        radius : float\n            The radius of the circle. Default is 1.\n        readonly : bool\n            Whether the created path should have the "readonly" argument\n            set when creating the Path instance.\n\n        Notes\n        -----\n        The circle is approximated using cubic Bezier curves.  This\n        uses 8 splines around the circle using the approach presented\n        here:\n\n          Lancaster, Don.  `Approximating a Circle or an Ellipse Using Four\n          Bezier Cubic Splines <http://www.tinaja.com/glib/ellipse4.pdf>`_.\n\n        ')
        
        # Assigning a Num to a Name (line 756):
        
        # Assigning a Num to a Name (line 756):
        float_112947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 16), 'float')
        # Assigning a type to the variable 'MAGIC' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'MAGIC', float_112947)
        
        # Assigning a Call to a Name (line 757):
        
        # Assigning a Call to a Name (line 757):
        
        # Call to sqrt(...): (line 757)
        # Processing the call arguments (line 757)
        float_112950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 27), 'float')
        # Processing the call keyword arguments (line 757)
        kwargs_112951 = {}
        # Getting the type of 'np' (line 757)
        np_112948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 19), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 757)
        sqrt_112949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 19), np_112948, 'sqrt')
        # Calling sqrt(args, kwargs) (line 757)
        sqrt_call_result_112952 = invoke(stypy.reporting.localization.Localization(__file__, 757, 19), sqrt_112949, *[float_112950], **kwargs_112951)
        
        # Assigning a type to the variable 'SQRTHALF' (line 757)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 8), 'SQRTHALF', sqrt_call_result_112952)
        
        # Assigning a BinOp to a Name (line 758):
        
        # Assigning a BinOp to a Name (line 758):
        # Getting the type of 'SQRTHALF' (line 758)
        SQRTHALF_112953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 18), 'SQRTHALF')
        # Getting the type of 'MAGIC' (line 758)
        MAGIC_112954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 29), 'MAGIC')
        # Applying the binary operator '*' (line 758)
        result_mul_112955 = python_operator(stypy.reporting.localization.Localization(__file__, 758, 18), '*', SQRTHALF_112953, MAGIC_112954)
        
        # Assigning a type to the variable 'MAGIC45' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'MAGIC45', result_mul_112955)
        
        # Assigning a Call to a Name (line 760):
        
        # Assigning a Call to a Name (line 760):
        
        # Call to array(...): (line 760)
        # Processing the call arguments (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 760)
        list_112958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 760)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 760)
        list_112959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 760)
        # Adding element type (line 760)
        float_112960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 29), list_112959, float_112960)
        # Adding element type (line 760)
        float_112961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 29), list_112959, float_112961)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_112959)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 762)
        list_112962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 762)
        # Adding element type (line 762)
        # Getting the type of 'MAGIC' (line 762)
        MAGIC_112963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 30), 'MAGIC', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 762, 29), list_112962, MAGIC_112963)
        # Adding element type (line 762)
        float_112964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 762, 29), list_112962, float_112964)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_112962)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 763)
        list_112965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 763)
        # Adding element type (line 763)
        # Getting the type of 'SQRTHALF' (line 763)
        SQRTHALF_112966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 30), 'SQRTHALF', False)
        # Getting the type of 'MAGIC45' (line 763)
        MAGIC45_112967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 39), 'MAGIC45', False)
        # Applying the binary operator '-' (line 763)
        result_sub_112968 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 30), '-', SQRTHALF_112966, MAGIC45_112967)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 29), list_112965, result_sub_112968)
        # Adding element type (line 763)
        
        # Getting the type of 'SQRTHALF' (line 763)
        SQRTHALF_112969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 49), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 763)
        result___neg___112970 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 48), 'usub', SQRTHALF_112969)
        
        # Getting the type of 'MAGIC45' (line 763)
        MAGIC45_112971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 58), 'MAGIC45', False)
        # Applying the binary operator '-' (line 763)
        result_sub_112972 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 48), '-', result___neg___112970, MAGIC45_112971)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 29), list_112965, result_sub_112972)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_112965)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 764)
        list_112973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 764)
        # Adding element type (line 764)
        # Getting the type of 'SQRTHALF' (line 764)
        SQRTHALF_112974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 30), 'SQRTHALF', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 764, 29), list_112973, SQRTHALF_112974)
        # Adding element type (line 764)
        
        # Getting the type of 'SQRTHALF' (line 764)
        SQRTHALF_112975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 41), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 764)
        result___neg___112976 = python_operator(stypy.reporting.localization.Localization(__file__, 764, 40), 'usub', SQRTHALF_112975)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 764, 29), list_112973, result___neg___112976)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_112973)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 766)
        list_112977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 766)
        # Adding element type (line 766)
        # Getting the type of 'SQRTHALF' (line 766)
        SQRTHALF_112978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 30), 'SQRTHALF', False)
        # Getting the type of 'MAGIC45' (line 766)
        MAGIC45_112979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 39), 'MAGIC45', False)
        # Applying the binary operator '+' (line 766)
        result_add_112980 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 30), '+', SQRTHALF_112978, MAGIC45_112979)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 766, 29), list_112977, result_add_112980)
        # Adding element type (line 766)
        
        # Getting the type of 'SQRTHALF' (line 766)
        SQRTHALF_112981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 49), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 766)
        result___neg___112982 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 48), 'usub', SQRTHALF_112981)
        
        # Getting the type of 'MAGIC45' (line 766)
        MAGIC45_112983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 58), 'MAGIC45', False)
        # Applying the binary operator '+' (line 766)
        result_add_112984 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 48), '+', result___neg___112982, MAGIC45_112983)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 766, 29), list_112977, result_add_112984)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_112977)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 767)
        list_112985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 767)
        # Adding element type (line 767)
        float_112986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 29), list_112985, float_112986)
        # Adding element type (line 767)
        
        # Getting the type of 'MAGIC' (line 767)
        MAGIC_112987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 36), 'MAGIC', False)
        # Applying the 'usub' unary operator (line 767)
        result___neg___112988 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 35), 'usub', MAGIC_112987)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 29), list_112985, result___neg___112988)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_112985)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 768)
        list_112989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 768)
        # Adding element type (line 768)
        float_112990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 29), list_112989, float_112990)
        # Adding element type (line 768)
        float_112991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 29), list_112989, float_112991)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_112989)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 770)
        list_112992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 770)
        # Adding element type (line 770)
        float_112993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 770, 29), list_112992, float_112993)
        # Adding element type (line 770)
        # Getting the type of 'MAGIC' (line 770)
        MAGIC_112994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 35), 'MAGIC', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 770, 29), list_112992, MAGIC_112994)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_112992)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 771)
        list_112995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 771)
        # Adding element type (line 771)
        # Getting the type of 'SQRTHALF' (line 771)
        SQRTHALF_112996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 30), 'SQRTHALF', False)
        # Getting the type of 'MAGIC45' (line 771)
        MAGIC45_112997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 39), 'MAGIC45', False)
        # Applying the binary operator '+' (line 771)
        result_add_112998 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 30), '+', SQRTHALF_112996, MAGIC45_112997)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 771, 29), list_112995, result_add_112998)
        # Adding element type (line 771)
        # Getting the type of 'SQRTHALF' (line 771)
        SQRTHALF_112999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 48), 'SQRTHALF', False)
        # Getting the type of 'MAGIC45' (line 771)
        MAGIC45_113000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 57), 'MAGIC45', False)
        # Applying the binary operator '-' (line 771)
        result_sub_113001 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 48), '-', SQRTHALF_112999, MAGIC45_113000)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 771, 29), list_112995, result_sub_113001)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_112995)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 772)
        list_113002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 772)
        # Adding element type (line 772)
        # Getting the type of 'SQRTHALF' (line 772)
        SQRTHALF_113003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 30), 'SQRTHALF', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 29), list_113002, SQRTHALF_113003)
        # Adding element type (line 772)
        # Getting the type of 'SQRTHALF' (line 772)
        SQRTHALF_113004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 40), 'SQRTHALF', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 29), list_113002, SQRTHALF_113004)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113002)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 774)
        list_113005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 774)
        # Adding element type (line 774)
        # Getting the type of 'SQRTHALF' (line 774)
        SQRTHALF_113006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 30), 'SQRTHALF', False)
        # Getting the type of 'MAGIC45' (line 774)
        MAGIC45_113007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 39), 'MAGIC45', False)
        # Applying the binary operator '-' (line 774)
        result_sub_113008 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 30), '-', SQRTHALF_113006, MAGIC45_113007)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 774, 29), list_113005, result_sub_113008)
        # Adding element type (line 774)
        # Getting the type of 'SQRTHALF' (line 774)
        SQRTHALF_113009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 48), 'SQRTHALF', False)
        # Getting the type of 'MAGIC45' (line 774)
        MAGIC45_113010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 57), 'MAGIC45', False)
        # Applying the binary operator '+' (line 774)
        result_add_113011 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 48), '+', SQRTHALF_113009, MAGIC45_113010)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 774, 29), list_113005, result_add_113011)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113005)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 775)
        list_113012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 775)
        # Adding element type (line 775)
        # Getting the type of 'MAGIC' (line 775)
        MAGIC_113013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 30), 'MAGIC', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 29), list_113012, MAGIC_113013)
        # Adding element type (line 775)
        float_113014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 29), list_113012, float_113014)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113012)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 776)
        list_113015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 776)
        # Adding element type (line 776)
        float_113016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 776, 29), list_113015, float_113016)
        # Adding element type (line 776)
        float_113017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 776, 29), list_113015, float_113017)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113015)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 778)
        list_113018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 778)
        # Adding element type (line 778)
        
        # Getting the type of 'MAGIC' (line 778)
        MAGIC_113019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 31), 'MAGIC', False)
        # Applying the 'usub' unary operator (line 778)
        result___neg___113020 = python_operator(stypy.reporting.localization.Localization(__file__, 778, 30), 'usub', MAGIC_113019)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 29), list_113018, result___neg___113020)
        # Adding element type (line 778)
        float_113021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 29), list_113018, float_113021)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113018)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 779)
        list_113022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 779)
        # Adding element type (line 779)
        
        # Getting the type of 'SQRTHALF' (line 779)
        SQRTHALF_113023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 31), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 779)
        result___neg___113024 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 30), 'usub', SQRTHALF_113023)
        
        # Getting the type of 'MAGIC45' (line 779)
        MAGIC45_113025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 40), 'MAGIC45', False)
        # Applying the binary operator '+' (line 779)
        result_add_113026 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 30), '+', result___neg___113024, MAGIC45_113025)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 779, 29), list_113022, result_add_113026)
        # Adding element type (line 779)
        # Getting the type of 'SQRTHALF' (line 779)
        SQRTHALF_113027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 49), 'SQRTHALF', False)
        # Getting the type of 'MAGIC45' (line 779)
        MAGIC45_113028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 58), 'MAGIC45', False)
        # Applying the binary operator '+' (line 779)
        result_add_113029 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 49), '+', SQRTHALF_113027, MAGIC45_113028)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 779, 29), list_113022, result_add_113029)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113022)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 780)
        list_113030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 780)
        # Adding element type (line 780)
        
        # Getting the type of 'SQRTHALF' (line 780)
        SQRTHALF_113031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 31), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 780)
        result___neg___113032 = python_operator(stypy.reporting.localization.Localization(__file__, 780, 30), 'usub', SQRTHALF_113031)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 780, 29), list_113030, result___neg___113032)
        # Adding element type (line 780)
        # Getting the type of 'SQRTHALF' (line 780)
        SQRTHALF_113033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 41), 'SQRTHALF', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 780, 29), list_113030, SQRTHALF_113033)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113030)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 782)
        list_113034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 782)
        # Adding element type (line 782)
        
        # Getting the type of 'SQRTHALF' (line 782)
        SQRTHALF_113035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 31), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 782)
        result___neg___113036 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 30), 'usub', SQRTHALF_113035)
        
        # Getting the type of 'MAGIC45' (line 782)
        MAGIC45_113037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 40), 'MAGIC45', False)
        # Applying the binary operator '-' (line 782)
        result_sub_113038 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 30), '-', result___neg___113036, MAGIC45_113037)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 782, 29), list_113034, result_sub_113038)
        # Adding element type (line 782)
        # Getting the type of 'SQRTHALF' (line 782)
        SQRTHALF_113039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 49), 'SQRTHALF', False)
        # Getting the type of 'MAGIC45' (line 782)
        MAGIC45_113040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 58), 'MAGIC45', False)
        # Applying the binary operator '-' (line 782)
        result_sub_113041 = python_operator(stypy.reporting.localization.Localization(__file__, 782, 49), '-', SQRTHALF_113039, MAGIC45_113040)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 782, 29), list_113034, result_sub_113041)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113034)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 783)
        list_113042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 783)
        # Adding element type (line 783)
        float_113043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 783, 29), list_113042, float_113043)
        # Adding element type (line 783)
        # Getting the type of 'MAGIC' (line 783)
        MAGIC_113044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 36), 'MAGIC', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 783, 29), list_113042, MAGIC_113044)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113042)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 784)
        list_113045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 784)
        # Adding element type (line 784)
        float_113046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 784, 29), list_113045, float_113046)
        # Adding element type (line 784)
        float_113047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 784, 29), list_113045, float_113047)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113045)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 786)
        list_113048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 786)
        # Adding element type (line 786)
        float_113049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 786, 29), list_113048, float_113049)
        # Adding element type (line 786)
        
        # Getting the type of 'MAGIC' (line 786)
        MAGIC_113050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 37), 'MAGIC', False)
        # Applying the 'usub' unary operator (line 786)
        result___neg___113051 = python_operator(stypy.reporting.localization.Localization(__file__, 786, 36), 'usub', MAGIC_113050)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 786, 29), list_113048, result___neg___113051)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113048)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 787)
        list_113052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 787)
        # Adding element type (line 787)
        
        # Getting the type of 'SQRTHALF' (line 787)
        SQRTHALF_113053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 31), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 787)
        result___neg___113054 = python_operator(stypy.reporting.localization.Localization(__file__, 787, 30), 'usub', SQRTHALF_113053)
        
        # Getting the type of 'MAGIC45' (line 787)
        MAGIC45_113055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 40), 'MAGIC45', False)
        # Applying the binary operator '-' (line 787)
        result_sub_113056 = python_operator(stypy.reporting.localization.Localization(__file__, 787, 30), '-', result___neg___113054, MAGIC45_113055)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 29), list_113052, result_sub_113056)
        # Adding element type (line 787)
        
        # Getting the type of 'SQRTHALF' (line 787)
        SQRTHALF_113057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 50), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 787)
        result___neg___113058 = python_operator(stypy.reporting.localization.Localization(__file__, 787, 49), 'usub', SQRTHALF_113057)
        
        # Getting the type of 'MAGIC45' (line 787)
        MAGIC45_113059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 59), 'MAGIC45', False)
        # Applying the binary operator '+' (line 787)
        result_add_113060 = python_operator(stypy.reporting.localization.Localization(__file__, 787, 49), '+', result___neg___113058, MAGIC45_113059)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 29), list_113052, result_add_113060)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113052)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 788)
        list_113061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 788)
        # Adding element type (line 788)
        
        # Getting the type of 'SQRTHALF' (line 788)
        SQRTHALF_113062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 31), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 788)
        result___neg___113063 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 30), 'usub', SQRTHALF_113062)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 29), list_113061, result___neg___113063)
        # Adding element type (line 788)
        
        # Getting the type of 'SQRTHALF' (line 788)
        SQRTHALF_113064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 42), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 788)
        result___neg___113065 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 41), 'usub', SQRTHALF_113064)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 29), list_113061, result___neg___113065)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113061)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 790)
        list_113066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 790)
        # Adding element type (line 790)
        
        # Getting the type of 'SQRTHALF' (line 790)
        SQRTHALF_113067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 31), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 790)
        result___neg___113068 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 30), 'usub', SQRTHALF_113067)
        
        # Getting the type of 'MAGIC45' (line 790)
        MAGIC45_113069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 40), 'MAGIC45', False)
        # Applying the binary operator '+' (line 790)
        result_add_113070 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 30), '+', result___neg___113068, MAGIC45_113069)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 790, 29), list_113066, result_add_113070)
        # Adding element type (line 790)
        
        # Getting the type of 'SQRTHALF' (line 790)
        SQRTHALF_113071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 50), 'SQRTHALF', False)
        # Applying the 'usub' unary operator (line 790)
        result___neg___113072 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 49), 'usub', SQRTHALF_113071)
        
        # Getting the type of 'MAGIC45' (line 790)
        MAGIC45_113073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 59), 'MAGIC45', False)
        # Applying the binary operator '-' (line 790)
        result_sub_113074 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 49), '-', result___neg___113072, MAGIC45_113073)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 790, 29), list_113066, result_sub_113074)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113066)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 791)
        list_113075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 791)
        # Adding element type (line 791)
        
        # Getting the type of 'MAGIC' (line 791)
        MAGIC_113076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 31), 'MAGIC', False)
        # Applying the 'usub' unary operator (line 791)
        result___neg___113077 = python_operator(stypy.reporting.localization.Localization(__file__, 791, 30), 'usub', MAGIC_113076)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 791, 29), list_113075, result___neg___113077)
        # Adding element type (line 791)
        float_113078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 791, 29), list_113075, float_113078)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113075)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 792)
        list_113079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 792)
        # Adding element type (line 792)
        float_113080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 792, 29), list_113079, float_113080)
        # Adding element type (line 792)
        float_113081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 792, 29), list_113079, float_113081)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113079)
        # Adding element type (line 760)
        
        # Obtaining an instance of the builtin type 'list' (line 794)
        list_113082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 794)
        # Adding element type (line 794)
        float_113083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 794, 29), list_113082, float_113083)
        # Adding element type (line 794)
        float_113084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 794, 29), list_113082, float_113084)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 760, 28), list_112958, list_113082)
        
        # Processing the call keyword arguments (line 760)
        # Getting the type of 'float' (line 795)
        float_113085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 34), 'float', False)
        keyword_113086 = float_113085
        kwargs_113087 = {'dtype': keyword_113086}
        # Getting the type of 'np' (line 760)
        np_112956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 760)
        array_112957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 19), np_112956, 'array')
        # Calling array(args, kwargs) (line 760)
        array_call_result_113088 = invoke(stypy.reporting.localization.Localization(__file__, 760, 19), array_112957, *[list_112958], **kwargs_113087)
        
        # Assigning a type to the variable 'vertices' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'vertices', array_call_result_113088)
        
        # Assigning a BinOp to a Name (line 797):
        
        # Assigning a BinOp to a Name (line 797):
        
        # Obtaining an instance of the builtin type 'list' (line 797)
        list_113089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 797)
        # Adding element type (line 797)
        # Getting the type of 'cls' (line 797)
        cls_113090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 17), 'cls')
        # Obtaining the member 'CURVE4' of a type (line 797)
        CURVE4_113091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 17), cls_113090, 'CURVE4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 16), list_113089, CURVE4_113091)
        
        int_113092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 31), 'int')
        # Applying the binary operator '*' (line 797)
        result_mul_113093 = python_operator(stypy.reporting.localization.Localization(__file__, 797, 16), '*', list_113089, int_113092)
        
        # Assigning a type to the variable 'codes' (line 797)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'codes', result_mul_113093)
        
        # Assigning a Attribute to a Subscript (line 798):
        
        # Assigning a Attribute to a Subscript (line 798):
        # Getting the type of 'cls' (line 798)
        cls_113094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 19), 'cls')
        # Obtaining the member 'MOVETO' of a type (line 798)
        MOVETO_113095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 19), cls_113094, 'MOVETO')
        # Getting the type of 'codes' (line 798)
        codes_113096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 8), 'codes')
        int_113097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 14), 'int')
        # Storing an element on a container (line 798)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 798, 8), codes_113096, (int_113097, MOVETO_113095))
        
        # Assigning a Attribute to a Subscript (line 799):
        
        # Assigning a Attribute to a Subscript (line 799):
        # Getting the type of 'cls' (line 799)
        cls_113098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 20), 'cls')
        # Obtaining the member 'CLOSEPOLY' of a type (line 799)
        CLOSEPOLY_113099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 20), cls_113098, 'CLOSEPOLY')
        # Getting the type of 'codes' (line 799)
        codes_113100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 8), 'codes')
        int_113101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 14), 'int')
        # Storing an element on a container (line 799)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 799, 8), codes_113100, (int_113101, CLOSEPOLY_113099))
        
        # Call to Path(...): (line 800)
        # Processing the call arguments (line 800)
        # Getting the type of 'vertices' (line 800)
        vertices_113103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 20), 'vertices', False)
        # Getting the type of 'radius' (line 800)
        radius_113104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 31), 'radius', False)
        # Applying the binary operator '*' (line 800)
        result_mul_113105 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 20), '*', vertices_113103, radius_113104)
        
        # Getting the type of 'center' (line 800)
        center_113106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 40), 'center', False)
        # Applying the binary operator '+' (line 800)
        result_add_113107 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 20), '+', result_mul_113105, center_113106)
        
        # Getting the type of 'codes' (line 800)
        codes_113108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 48), 'codes', False)
        # Processing the call keyword arguments (line 800)
        # Getting the type of 'readonly' (line 800)
        readonly_113109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 64), 'readonly', False)
        keyword_113110 = readonly_113109
        kwargs_113111 = {'readonly': keyword_113110}
        # Getting the type of 'Path' (line 800)
        Path_113102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 15), 'Path', False)
        # Calling Path(args, kwargs) (line 800)
        Path_call_result_113112 = invoke(stypy.reporting.localization.Localization(__file__, 800, 15), Path_113102, *[result_add_113107, codes_113108], **kwargs_113111)
        
        # Assigning a type to the variable 'stypy_return_type' (line 800)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 8), 'stypy_return_type', Path_call_result_113112)
        
        # ################# End of 'circle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'circle' in the type store
        # Getting the type of 'stypy_return_type' (line 731)
        stypy_return_type_113113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113113)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'circle'
        return stypy_return_type_113113

    
    # Assigning a Name to a Name (line 802):

    @norecursion
    def unit_circle_righthalf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unit_circle_righthalf'
        module_type_store = module_type_store.open_function_context('unit_circle_righthalf', 804, 4, False)
        # Assigning a type to the variable 'self' (line 805)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.unit_circle_righthalf.__dict__.__setitem__('stypy_localization', localization)
        Path.unit_circle_righthalf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.unit_circle_righthalf.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.unit_circle_righthalf.__dict__.__setitem__('stypy_function_name', 'Path.unit_circle_righthalf')
        Path.unit_circle_righthalf.__dict__.__setitem__('stypy_param_names_list', [])
        Path.unit_circle_righthalf.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.unit_circle_righthalf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.unit_circle_righthalf.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.unit_circle_righthalf.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.unit_circle_righthalf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.unit_circle_righthalf.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.unit_circle_righthalf', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unit_circle_righthalf', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unit_circle_righthalf(...)' code ##################

        unicode_113114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, (-1)), 'unicode', u'\n        Return a :class:`Path` of the right half\n        of a unit circle. The circle is approximated using cubic Bezier\n        curves.  This uses 4 splines around the circle using the approach\n        presented here:\n\n          Lancaster, Don.  `Approximating a Circle or an Ellipse Using Four\n          Bezier Cubic Splines <http://www.tinaja.com/glib/ellipse4.pdf>`_.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 815)
        # Getting the type of 'cls' (line 815)
        cls_113115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 11), 'cls')
        # Obtaining the member '_unit_circle_righthalf' of a type (line 815)
        _unit_circle_righthalf_113116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 11), cls_113115, '_unit_circle_righthalf')
        # Getting the type of 'None' (line 815)
        None_113117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 41), 'None')
        
        (may_be_113118, more_types_in_union_113119) = may_be_none(_unit_circle_righthalf_113116, None_113117)

        if may_be_113118:

            if more_types_in_union_113119:
                # Runtime conditional SSA (line 815)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 816):
            
            # Assigning a Num to a Name (line 816):
            float_113120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 20), 'float')
            # Assigning a type to the variable 'MAGIC' (line 816)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 12), 'MAGIC', float_113120)
            
            # Assigning a Call to a Name (line 817):
            
            # Assigning a Call to a Name (line 817):
            
            # Call to sqrt(...): (line 817)
            # Processing the call arguments (line 817)
            float_113123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 31), 'float')
            # Processing the call keyword arguments (line 817)
            kwargs_113124 = {}
            # Getting the type of 'np' (line 817)
            np_113121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 23), 'np', False)
            # Obtaining the member 'sqrt' of a type (line 817)
            sqrt_113122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 23), np_113121, 'sqrt')
            # Calling sqrt(args, kwargs) (line 817)
            sqrt_call_result_113125 = invoke(stypy.reporting.localization.Localization(__file__, 817, 23), sqrt_113122, *[float_113123], **kwargs_113124)
            
            # Assigning a type to the variable 'SQRTHALF' (line 817)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'SQRTHALF', sqrt_call_result_113125)
            
            # Assigning a BinOp to a Name (line 818):
            
            # Assigning a BinOp to a Name (line 818):
            # Getting the type of 'SQRTHALF' (line 818)
            SQRTHALF_113126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 22), 'SQRTHALF')
            # Getting the type of 'MAGIC' (line 818)
            MAGIC_113127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 33), 'MAGIC')
            # Applying the binary operator '*' (line 818)
            result_mul_113128 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 22), '*', SQRTHALF_113126, MAGIC_113127)
            
            # Assigning a type to the variable 'MAGIC45' (line 818)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'MAGIC45', result_mul_113128)
            
            # Assigning a Call to a Name (line 820):
            
            # Assigning a Call to a Name (line 820):
            
            # Call to array(...): (line 820)
            # Processing the call arguments (line 820)
            
            # Obtaining an instance of the builtin type 'list' (line 821)
            list_113131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 16), 'list')
            # Adding type elements to the builtin type 'list' instance (line 821)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 821)
            list_113132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 821)
            # Adding element type (line 821)
            float_113133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 18), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 17), list_113132, float_113133)
            # Adding element type (line 821)
            float_113134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 23), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 17), list_113132, float_113134)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113132)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 823)
            list_113135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 823)
            # Adding element type (line 823)
            # Getting the type of 'MAGIC' (line 823)
            MAGIC_113136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 18), 'MAGIC', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 823, 17), list_113135, MAGIC_113136)
            # Adding element type (line 823)
            float_113137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 25), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 823, 17), list_113135, float_113137)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113135)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 824)
            list_113138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 824)
            # Adding element type (line 824)
            # Getting the type of 'SQRTHALF' (line 824)
            SQRTHALF_113139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 18), 'SQRTHALF', False)
            # Getting the type of 'MAGIC45' (line 824)
            MAGIC45_113140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 27), 'MAGIC45', False)
            # Applying the binary operator '-' (line 824)
            result_sub_113141 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 18), '-', SQRTHALF_113139, MAGIC45_113140)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 824, 17), list_113138, result_sub_113141)
            # Adding element type (line 824)
            
            # Getting the type of 'SQRTHALF' (line 824)
            SQRTHALF_113142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 37), 'SQRTHALF', False)
            # Applying the 'usub' unary operator (line 824)
            result___neg___113143 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 36), 'usub', SQRTHALF_113142)
            
            # Getting the type of 'MAGIC45' (line 824)
            MAGIC45_113144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 46), 'MAGIC45', False)
            # Applying the binary operator '-' (line 824)
            result_sub_113145 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 36), '-', result___neg___113143, MAGIC45_113144)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 824, 17), list_113138, result_sub_113145)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113138)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 825)
            list_113146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 825)
            # Adding element type (line 825)
            # Getting the type of 'SQRTHALF' (line 825)
            SQRTHALF_113147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 18), 'SQRTHALF', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 825, 17), list_113146, SQRTHALF_113147)
            # Adding element type (line 825)
            
            # Getting the type of 'SQRTHALF' (line 825)
            SQRTHALF_113148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 29), 'SQRTHALF', False)
            # Applying the 'usub' unary operator (line 825)
            result___neg___113149 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 28), 'usub', SQRTHALF_113148)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 825, 17), list_113146, result___neg___113149)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113146)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 827)
            list_113150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 827)
            # Adding element type (line 827)
            # Getting the type of 'SQRTHALF' (line 827)
            SQRTHALF_113151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 18), 'SQRTHALF', False)
            # Getting the type of 'MAGIC45' (line 827)
            MAGIC45_113152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 27), 'MAGIC45', False)
            # Applying the binary operator '+' (line 827)
            result_add_113153 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 18), '+', SQRTHALF_113151, MAGIC45_113152)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 827, 17), list_113150, result_add_113153)
            # Adding element type (line 827)
            
            # Getting the type of 'SQRTHALF' (line 827)
            SQRTHALF_113154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 37), 'SQRTHALF', False)
            # Applying the 'usub' unary operator (line 827)
            result___neg___113155 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 36), 'usub', SQRTHALF_113154)
            
            # Getting the type of 'MAGIC45' (line 827)
            MAGIC45_113156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 46), 'MAGIC45', False)
            # Applying the binary operator '+' (line 827)
            result_add_113157 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 36), '+', result___neg___113155, MAGIC45_113156)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 827, 17), list_113150, result_add_113157)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113150)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 828)
            list_113158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 828)
            # Adding element type (line 828)
            float_113159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 18), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 828, 17), list_113158, float_113159)
            # Adding element type (line 828)
            
            # Getting the type of 'MAGIC' (line 828)
            MAGIC_113160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 24), 'MAGIC', False)
            # Applying the 'usub' unary operator (line 828)
            result___neg___113161 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 23), 'usub', MAGIC_113160)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 828, 17), list_113158, result___neg___113161)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113158)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 829)
            list_113162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 829)
            # Adding element type (line 829)
            float_113163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 18), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 829, 17), list_113162, float_113163)
            # Adding element type (line 829)
            float_113164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 23), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 829, 17), list_113162, float_113164)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113162)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 831)
            list_113165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 831)
            # Adding element type (line 831)
            float_113166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 18), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 17), list_113165, float_113166)
            # Adding element type (line 831)
            # Getting the type of 'MAGIC' (line 831)
            MAGIC_113167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 23), 'MAGIC', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 17), list_113165, MAGIC_113167)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113165)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 832)
            list_113168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 832)
            # Adding element type (line 832)
            # Getting the type of 'SQRTHALF' (line 832)
            SQRTHALF_113169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 18), 'SQRTHALF', False)
            # Getting the type of 'MAGIC45' (line 832)
            MAGIC45_113170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 27), 'MAGIC45', False)
            # Applying the binary operator '+' (line 832)
            result_add_113171 = python_operator(stypy.reporting.localization.Localization(__file__, 832, 18), '+', SQRTHALF_113169, MAGIC45_113170)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 832, 17), list_113168, result_add_113171)
            # Adding element type (line 832)
            # Getting the type of 'SQRTHALF' (line 832)
            SQRTHALF_113172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 36), 'SQRTHALF', False)
            # Getting the type of 'MAGIC45' (line 832)
            MAGIC45_113173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 45), 'MAGIC45', False)
            # Applying the binary operator '-' (line 832)
            result_sub_113174 = python_operator(stypy.reporting.localization.Localization(__file__, 832, 36), '-', SQRTHALF_113172, MAGIC45_113173)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 832, 17), list_113168, result_sub_113174)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113168)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 833)
            list_113175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 833)
            # Adding element type (line 833)
            # Getting the type of 'SQRTHALF' (line 833)
            SQRTHALF_113176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 18), 'SQRTHALF', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 833, 17), list_113175, SQRTHALF_113176)
            # Adding element type (line 833)
            # Getting the type of 'SQRTHALF' (line 833)
            SQRTHALF_113177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 28), 'SQRTHALF', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 833, 17), list_113175, SQRTHALF_113177)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113175)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 835)
            list_113178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 835)
            # Adding element type (line 835)
            # Getting the type of 'SQRTHALF' (line 835)
            SQRTHALF_113179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 18), 'SQRTHALF', False)
            # Getting the type of 'MAGIC45' (line 835)
            MAGIC45_113180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 27), 'MAGIC45', False)
            # Applying the binary operator '-' (line 835)
            result_sub_113181 = python_operator(stypy.reporting.localization.Localization(__file__, 835, 18), '-', SQRTHALF_113179, MAGIC45_113180)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 835, 17), list_113178, result_sub_113181)
            # Adding element type (line 835)
            # Getting the type of 'SQRTHALF' (line 835)
            SQRTHALF_113182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 36), 'SQRTHALF', False)
            # Getting the type of 'MAGIC45' (line 835)
            MAGIC45_113183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 45), 'MAGIC45', False)
            # Applying the binary operator '+' (line 835)
            result_add_113184 = python_operator(stypy.reporting.localization.Localization(__file__, 835, 36), '+', SQRTHALF_113182, MAGIC45_113183)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 835, 17), list_113178, result_add_113184)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113178)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 836)
            list_113185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 836)
            # Adding element type (line 836)
            # Getting the type of 'MAGIC' (line 836)
            MAGIC_113186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 18), 'MAGIC', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 836, 17), list_113185, MAGIC_113186)
            # Adding element type (line 836)
            float_113187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 25), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 836, 17), list_113185, float_113187)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113185)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 837)
            list_113188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 837)
            # Adding element type (line 837)
            float_113189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 18), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 837, 17), list_113188, float_113189)
            # Adding element type (line 837)
            float_113190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 23), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 837, 17), list_113188, float_113190)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113188)
            # Adding element type (line 821)
            
            # Obtaining an instance of the builtin type 'list' (line 839)
            list_113191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 17), 'list')
            # Adding type elements to the builtin type 'list' instance (line 839)
            # Adding element type (line 839)
            float_113192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 18), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 839, 17), list_113191, float_113192)
            # Adding element type (line 839)
            float_113193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 23), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 839, 17), list_113191, float_113193)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), list_113131, list_113191)
            
            # Getting the type of 'float' (line 841)
            float_113194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 16), 'float', False)
            # Processing the call keyword arguments (line 820)
            kwargs_113195 = {}
            # Getting the type of 'np' (line 820)
            np_113129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 23), 'np', False)
            # Obtaining the member 'array' of a type (line 820)
            array_113130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 23), np_113129, 'array')
            # Calling array(args, kwargs) (line 820)
            array_call_result_113196 = invoke(stypy.reporting.localization.Localization(__file__, 820, 23), array_113130, *[list_113131, float_113194], **kwargs_113195)
            
            # Assigning a type to the variable 'vertices' (line 820)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 12), 'vertices', array_call_result_113196)
            
            # Assigning a BinOp to a Name (line 843):
            
            # Assigning a BinOp to a Name (line 843):
            # Getting the type of 'cls' (line 843)
            cls_113197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 20), 'cls')
            # Obtaining the member 'CURVE4' of a type (line 843)
            CURVE4_113198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 20), cls_113197, 'CURVE4')
            
            # Call to ones(...): (line 843)
            # Processing the call arguments (line 843)
            int_113201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 41), 'int')
            # Processing the call keyword arguments (line 843)
            kwargs_113202 = {}
            # Getting the type of 'np' (line 843)
            np_113199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 33), 'np', False)
            # Obtaining the member 'ones' of a type (line 843)
            ones_113200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 33), np_113199, 'ones')
            # Calling ones(args, kwargs) (line 843)
            ones_call_result_113203 = invoke(stypy.reporting.localization.Localization(__file__, 843, 33), ones_113200, *[int_113201], **kwargs_113202)
            
            # Applying the binary operator '*' (line 843)
            result_mul_113204 = python_operator(stypy.reporting.localization.Localization(__file__, 843, 20), '*', CURVE4_113198, ones_call_result_113203)
            
            # Assigning a type to the variable 'codes' (line 843)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'codes', result_mul_113204)
            
            # Assigning a Attribute to a Subscript (line 844):
            
            # Assigning a Attribute to a Subscript (line 844):
            # Getting the type of 'cls' (line 844)
            cls_113205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 23), 'cls')
            # Obtaining the member 'MOVETO' of a type (line 844)
            MOVETO_113206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 23), cls_113205, 'MOVETO')
            # Getting the type of 'codes' (line 844)
            codes_113207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 12), 'codes')
            int_113208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 18), 'int')
            # Storing an element on a container (line 844)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 844, 12), codes_113207, (int_113208, MOVETO_113206))
            
            # Assigning a Attribute to a Subscript (line 845):
            
            # Assigning a Attribute to a Subscript (line 845):
            # Getting the type of 'cls' (line 845)
            cls_113209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 24), 'cls')
            # Obtaining the member 'CLOSEPOLY' of a type (line 845)
            CLOSEPOLY_113210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 24), cls_113209, 'CLOSEPOLY')
            # Getting the type of 'codes' (line 845)
            codes_113211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 12), 'codes')
            int_113212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 18), 'int')
            # Storing an element on a container (line 845)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 845, 12), codes_113211, (int_113212, CLOSEPOLY_113210))
            
            # Assigning a Call to a Attribute (line 847):
            
            # Assigning a Call to a Attribute (line 847):
            
            # Call to cls(...): (line 847)
            # Processing the call arguments (line 847)
            # Getting the type of 'vertices' (line 847)
            vertices_113214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 45), 'vertices', False)
            # Getting the type of 'codes' (line 847)
            codes_113215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 55), 'codes', False)
            # Processing the call keyword arguments (line 847)
            # Getting the type of 'True' (line 847)
            True_113216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 71), 'True', False)
            keyword_113217 = True_113216
            kwargs_113218 = {'readonly': keyword_113217}
            # Getting the type of 'cls' (line 847)
            cls_113213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 41), 'cls', False)
            # Calling cls(args, kwargs) (line 847)
            cls_call_result_113219 = invoke(stypy.reporting.localization.Localization(__file__, 847, 41), cls_113213, *[vertices_113214, codes_113215], **kwargs_113218)
            
            # Getting the type of 'cls' (line 847)
            cls_113220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 12), 'cls')
            # Setting the type of the member '_unit_circle_righthalf' of a type (line 847)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 12), cls_113220, '_unit_circle_righthalf', cls_call_result_113219)

            if more_types_in_union_113119:
                # SSA join for if statement (line 815)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'cls' (line 848)
        cls_113221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 15), 'cls')
        # Obtaining the member '_unit_circle_righthalf' of a type (line 848)
        _unit_circle_righthalf_113222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 15), cls_113221, '_unit_circle_righthalf')
        # Assigning a type to the variable 'stypy_return_type' (line 848)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'stypy_return_type', _unit_circle_righthalf_113222)
        
        # ################# End of 'unit_circle_righthalf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unit_circle_righthalf' in the type store
        # Getting the type of 'stypy_return_type' (line 804)
        stypy_return_type_113223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113223)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unit_circle_righthalf'
        return stypy_return_type_113223


    @norecursion
    def arc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 851)
        None_113224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 35), 'None')
        # Getting the type of 'False' (line 851)
        False_113225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 50), 'False')
        defaults = [None_113224, False_113225]
        # Create a new context for function 'arc'
        module_type_store = module_type_store.open_function_context('arc', 850, 4, False)
        # Assigning a type to the variable 'self' (line 851)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.arc.__dict__.__setitem__('stypy_localization', localization)
        Path.arc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.arc.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.arc.__dict__.__setitem__('stypy_function_name', 'Path.arc')
        Path.arc.__dict__.__setitem__('stypy_param_names_list', ['theta1', 'theta2', 'n', 'is_wedge'])
        Path.arc.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.arc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.arc.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.arc.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.arc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.arc.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.arc', ['theta1', 'theta2', 'n', 'is_wedge'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'arc', localization, ['theta1', 'theta2', 'n', 'is_wedge'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'arc(...)' code ##################

        unicode_113226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, (-1)), 'unicode', u'\n        Return an arc on the unit circle from angle\n        *theta1* to angle *theta2* (in degrees).\n\n        *theta2* is unwrapped to produce the shortest arc within 360 degrees.\n        That is, if *theta2* > *theta1* + 360, the arc will be from *theta1* to\n        *theta2* - 360 and not a full circle plus some extra overlap.\n\n        If *n* is provided, it is the number of spline segments to make.\n        If *n* is not provided, the number of spline segments is\n        determined based on the delta between *theta1* and *theta2*.\n\n           Masionobe, L.  2003.  `Drawing an elliptical arc using\n           polylines, quadratic or cubic Bezier curves\n           <http://www.spaceroots.org/documents/ellipse/index.html>`_.\n        ')
        
        # Assigning a BinOp to a Name (line 868):
        
        # Assigning a BinOp to a Name (line 868):
        # Getting the type of 'np' (line 868)
        np_113227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 17), 'np')
        # Obtaining the member 'pi' of a type (line 868)
        pi_113228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 17), np_113227, 'pi')
        float_113229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 25), 'float')
        # Applying the binary operator '*' (line 868)
        result_mul_113230 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 17), '*', pi_113228, float_113229)
        
        # Assigning a type to the variable 'halfpi' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'halfpi', result_mul_113230)
        
        # Assigning a Name to a Name (line 870):
        
        # Assigning a Name to a Name (line 870):
        # Getting the type of 'theta1' (line 870)
        theta1_113231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 15), 'theta1')
        # Assigning a type to the variable 'eta1' (line 870)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'eta1', theta1_113231)
        
        # Assigning a BinOp to a Name (line 871):
        
        # Assigning a BinOp to a Name (line 871):
        # Getting the type of 'theta2' (line 871)
        theta2_113232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 15), 'theta2')
        int_113233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 24), 'int')
        
        # Call to floor(...): (line 871)
        # Processing the call arguments (line 871)
        # Getting the type of 'theta2' (line 871)
        theta2_113236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 40), 'theta2', False)
        # Getting the type of 'theta1' (line 871)
        theta1_113237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 49), 'theta1', False)
        # Applying the binary operator '-' (line 871)
        result_sub_113238 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 40), '-', theta2_113236, theta1_113237)
        
        int_113239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 59), 'int')
        # Applying the binary operator 'div' (line 871)
        result_div_113240 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 39), 'div', result_sub_113238, int_113239)
        
        # Processing the call keyword arguments (line 871)
        kwargs_113241 = {}
        # Getting the type of 'np' (line 871)
        np_113234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 30), 'np', False)
        # Obtaining the member 'floor' of a type (line 871)
        floor_113235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 30), np_113234, 'floor')
        # Calling floor(args, kwargs) (line 871)
        floor_call_result_113242 = invoke(stypy.reporting.localization.Localization(__file__, 871, 30), floor_113235, *[result_div_113240], **kwargs_113241)
        
        # Applying the binary operator '*' (line 871)
        result_mul_113243 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 24), '*', int_113233, floor_call_result_113242)
        
        # Applying the binary operator '-' (line 871)
        result_sub_113244 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 15), '-', theta2_113232, result_mul_113243)
        
        # Assigning a type to the variable 'eta2' (line 871)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 8), 'eta2', result_sub_113244)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'theta2' (line 874)
        theta2_113245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 11), 'theta2')
        # Getting the type of 'theta1' (line 874)
        theta1_113246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 21), 'theta1')
        # Applying the binary operator '!=' (line 874)
        result_ne_113247 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 11), '!=', theta2_113245, theta1_113246)
        
        
        # Getting the type of 'eta2' (line 874)
        eta2_113248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 32), 'eta2')
        # Getting the type of 'eta1' (line 874)
        eta1_113249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 40), 'eta1')
        # Applying the binary operator '<=' (line 874)
        result_le_113250 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 32), '<=', eta2_113248, eta1_113249)
        
        # Applying the binary operator 'and' (line 874)
        result_and_keyword_113251 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 11), 'and', result_ne_113247, result_le_113250)
        
        # Testing the type of an if condition (line 874)
        if_condition_113252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 874, 8), result_and_keyword_113251)
        # Assigning a type to the variable 'if_condition_113252' (line 874)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 874, 8), 'if_condition_113252', if_condition_113252)
        # SSA begins for if statement (line 874)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'eta2' (line 875)
        eta2_113253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 12), 'eta2')
        int_113254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 20), 'int')
        # Applying the binary operator '+=' (line 875)
        result_iadd_113255 = python_operator(stypy.reporting.localization.Localization(__file__, 875, 12), '+=', eta2_113253, int_113254)
        # Assigning a type to the variable 'eta2' (line 875)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 12), 'eta2', result_iadd_113255)
        
        # SSA join for if statement (line 874)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 876):
        
        # Assigning a Call to a Name:
        
        # Call to deg2rad(...): (line 876)
        # Processing the call arguments (line 876)
        
        # Obtaining an instance of the builtin type 'list' (line 876)
        list_113258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 876)
        # Adding element type (line 876)
        # Getting the type of 'eta1' (line 876)
        eta1_113259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 33), 'eta1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 876, 32), list_113258, eta1_113259)
        # Adding element type (line 876)
        # Getting the type of 'eta2' (line 876)
        eta2_113260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 39), 'eta2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 876, 32), list_113258, eta2_113260)
        
        # Processing the call keyword arguments (line 876)
        kwargs_113261 = {}
        # Getting the type of 'np' (line 876)
        np_113256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 21), 'np', False)
        # Obtaining the member 'deg2rad' of a type (line 876)
        deg2rad_113257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 21), np_113256, 'deg2rad')
        # Calling deg2rad(args, kwargs) (line 876)
        deg2rad_call_result_113262 = invoke(stypy.reporting.localization.Localization(__file__, 876, 21), deg2rad_113257, *[list_113258], **kwargs_113261)
        
        # Assigning a type to the variable 'call_assignment_111653' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'call_assignment_111653', deg2rad_call_result_113262)
        
        # Assigning a Call to a Name (line 876):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_113265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 8), 'int')
        # Processing the call keyword arguments
        kwargs_113266 = {}
        # Getting the type of 'call_assignment_111653' (line 876)
        call_assignment_111653_113263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'call_assignment_111653', False)
        # Obtaining the member '__getitem__' of a type (line 876)
        getitem___113264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 8), call_assignment_111653_113263, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_113267 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___113264, *[int_113265], **kwargs_113266)
        
        # Assigning a type to the variable 'call_assignment_111654' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'call_assignment_111654', getitem___call_result_113267)
        
        # Assigning a Name to a Name (line 876):
        # Getting the type of 'call_assignment_111654' (line 876)
        call_assignment_111654_113268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'call_assignment_111654')
        # Assigning a type to the variable 'eta1' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'eta1', call_assignment_111654_113268)
        
        # Assigning a Call to a Name (line 876):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_113271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 8), 'int')
        # Processing the call keyword arguments
        kwargs_113272 = {}
        # Getting the type of 'call_assignment_111653' (line 876)
        call_assignment_111653_113269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'call_assignment_111653', False)
        # Obtaining the member '__getitem__' of a type (line 876)
        getitem___113270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 8), call_assignment_111653_113269, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_113273 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___113270, *[int_113271], **kwargs_113272)
        
        # Assigning a type to the variable 'call_assignment_111655' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'call_assignment_111655', getitem___call_result_113273)
        
        # Assigning a Name to a Name (line 876):
        # Getting the type of 'call_assignment_111655' (line 876)
        call_assignment_111655_113274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'call_assignment_111655')
        # Assigning a type to the variable 'eta2' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 14), 'eta2', call_assignment_111655_113274)
        
        # Type idiom detected: calculating its left and rigth part (line 879)
        # Getting the type of 'n' (line 879)
        n_113275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 11), 'n')
        # Getting the type of 'None' (line 879)
        None_113276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 16), 'None')
        
        (may_be_113277, more_types_in_union_113278) = may_be_none(n_113275, None_113276)

        if may_be_113277:

            if more_types_in_union_113278:
                # Runtime conditional SSA (line 879)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 880):
            
            # Assigning a Call to a Name (line 880):
            
            # Call to int(...): (line 880)
            # Processing the call arguments (line 880)
            int_113280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 20), 'int')
            
            # Call to ceil(...): (line 880)
            # Processing the call arguments (line 880)
            # Getting the type of 'eta2' (line 880)
            eta2_113283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 34), 'eta2', False)
            # Getting the type of 'eta1' (line 880)
            eta1_113284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 41), 'eta1', False)
            # Applying the binary operator '-' (line 880)
            result_sub_113285 = python_operator(stypy.reporting.localization.Localization(__file__, 880, 34), '-', eta2_113283, eta1_113284)
            
            # Getting the type of 'halfpi' (line 880)
            halfpi_113286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 49), 'halfpi', False)
            # Applying the binary operator 'div' (line 880)
            result_div_113287 = python_operator(stypy.reporting.localization.Localization(__file__, 880, 33), 'div', result_sub_113285, halfpi_113286)
            
            # Processing the call keyword arguments (line 880)
            kwargs_113288 = {}
            # Getting the type of 'np' (line 880)
            np_113281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 25), 'np', False)
            # Obtaining the member 'ceil' of a type (line 880)
            ceil_113282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 25), np_113281, 'ceil')
            # Calling ceil(args, kwargs) (line 880)
            ceil_call_result_113289 = invoke(stypy.reporting.localization.Localization(__file__, 880, 25), ceil_113282, *[result_div_113287], **kwargs_113288)
            
            # Applying the binary operator '**' (line 880)
            result_pow_113290 = python_operator(stypy.reporting.localization.Localization(__file__, 880, 20), '**', int_113280, ceil_call_result_113289)
            
            # Processing the call keyword arguments (line 880)
            kwargs_113291 = {}
            # Getting the type of 'int' (line 880)
            int_113279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 16), 'int', False)
            # Calling int(args, kwargs) (line 880)
            int_call_result_113292 = invoke(stypy.reporting.localization.Localization(__file__, 880, 16), int_113279, *[result_pow_113290], **kwargs_113291)
            
            # Assigning a type to the variable 'n' (line 880)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 12), 'n', int_call_result_113292)

            if more_types_in_union_113278:
                # SSA join for if statement (line 879)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'n' (line 881)
        n_113293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 11), 'n')
        int_113294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 15), 'int')
        # Applying the binary operator '<' (line 881)
        result_lt_113295 = python_operator(stypy.reporting.localization.Localization(__file__, 881, 11), '<', n_113293, int_113294)
        
        # Testing the type of an if condition (line 881)
        if_condition_113296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 881, 8), result_lt_113295)
        # Assigning a type to the variable 'if_condition_113296' (line 881)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 8), 'if_condition_113296', if_condition_113296)
        # SSA begins for if statement (line 881)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 882)
        # Processing the call arguments (line 882)
        unicode_113298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 29), 'unicode', u'n must be >= 1 or None')
        # Processing the call keyword arguments (line 882)
        kwargs_113299 = {}
        # Getting the type of 'ValueError' (line 882)
        ValueError_113297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 882)
        ValueError_call_result_113300 = invoke(stypy.reporting.localization.Localization(__file__, 882, 18), ValueError_113297, *[unicode_113298], **kwargs_113299)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 882, 12), ValueError_call_result_113300, 'raise parameter', BaseException)
        # SSA join for if statement (line 881)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 884):
        
        # Assigning a BinOp to a Name (line 884):
        # Getting the type of 'eta2' (line 884)
        eta2_113301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 16), 'eta2')
        # Getting the type of 'eta1' (line 884)
        eta1_113302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 23), 'eta1')
        # Applying the binary operator '-' (line 884)
        result_sub_113303 = python_operator(stypy.reporting.localization.Localization(__file__, 884, 16), '-', eta2_113301, eta1_113302)
        
        # Getting the type of 'n' (line 884)
        n_113304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 31), 'n')
        # Applying the binary operator 'div' (line 884)
        result_div_113305 = python_operator(stypy.reporting.localization.Localization(__file__, 884, 15), 'div', result_sub_113303, n_113304)
        
        # Assigning a type to the variable 'deta' (line 884)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 8), 'deta', result_div_113305)
        
        # Assigning a Call to a Name (line 885):
        
        # Assigning a Call to a Name (line 885):
        
        # Call to tan(...): (line 885)
        # Processing the call arguments (line 885)
        float_113308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 19), 'float')
        # Getting the type of 'deta' (line 885)
        deta_113309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 25), 'deta', False)
        # Applying the binary operator '*' (line 885)
        result_mul_113310 = python_operator(stypy.reporting.localization.Localization(__file__, 885, 19), '*', float_113308, deta_113309)
        
        # Processing the call keyword arguments (line 885)
        kwargs_113311 = {}
        # Getting the type of 'np' (line 885)
        np_113306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 12), 'np', False)
        # Obtaining the member 'tan' of a type (line 885)
        tan_113307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 12), np_113306, 'tan')
        # Calling tan(args, kwargs) (line 885)
        tan_call_result_113312 = invoke(stypy.reporting.localization.Localization(__file__, 885, 12), tan_113307, *[result_mul_113310], **kwargs_113311)
        
        # Assigning a type to the variable 't' (line 885)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 8), 't', tan_call_result_113312)
        
        # Assigning a BinOp to a Name (line 886):
        
        # Assigning a BinOp to a Name (line 886):
        
        # Call to sin(...): (line 886)
        # Processing the call arguments (line 886)
        # Getting the type of 'deta' (line 886)
        deta_113315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 23), 'deta', False)
        # Processing the call keyword arguments (line 886)
        kwargs_113316 = {}
        # Getting the type of 'np' (line 886)
        np_113313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 16), 'np', False)
        # Obtaining the member 'sin' of a type (line 886)
        sin_113314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 16), np_113313, 'sin')
        # Calling sin(args, kwargs) (line 886)
        sin_call_result_113317 = invoke(stypy.reporting.localization.Localization(__file__, 886, 16), sin_113314, *[deta_113315], **kwargs_113316)
        
        
        # Call to sqrt(...): (line 886)
        # Processing the call arguments (line 886)
        float_113320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 40), 'float')
        float_113321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 46), 'float')
        # Getting the type of 't' (line 886)
        t_113322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 52), 't', False)
        # Applying the binary operator '*' (line 886)
        result_mul_113323 = python_operator(stypy.reporting.localization.Localization(__file__, 886, 46), '*', float_113321, t_113322)
        
        # Getting the type of 't' (line 886)
        t_113324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 56), 't', False)
        # Applying the binary operator '*' (line 886)
        result_mul_113325 = python_operator(stypy.reporting.localization.Localization(__file__, 886, 54), '*', result_mul_113323, t_113324)
        
        # Applying the binary operator '+' (line 886)
        result_add_113326 = python_operator(stypy.reporting.localization.Localization(__file__, 886, 40), '+', float_113320, result_mul_113325)
        
        # Processing the call keyword arguments (line 886)
        kwargs_113327 = {}
        # Getting the type of 'np' (line 886)
        np_113318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 32), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 886)
        sqrt_113319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 32), np_113318, 'sqrt')
        # Calling sqrt(args, kwargs) (line 886)
        sqrt_call_result_113328 = invoke(stypy.reporting.localization.Localization(__file__, 886, 32), sqrt_113319, *[result_add_113326], **kwargs_113327)
        
        int_113329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 61), 'int')
        # Applying the binary operator '-' (line 886)
        result_sub_113330 = python_operator(stypy.reporting.localization.Localization(__file__, 886, 32), '-', sqrt_call_result_113328, int_113329)
        
        # Applying the binary operator '*' (line 886)
        result_mul_113331 = python_operator(stypy.reporting.localization.Localization(__file__, 886, 16), '*', sin_call_result_113317, result_sub_113330)
        
        float_113332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 66), 'float')
        # Applying the binary operator 'div' (line 886)
        result_div_113333 = python_operator(stypy.reporting.localization.Localization(__file__, 886, 64), 'div', result_mul_113331, float_113332)
        
        # Assigning a type to the variable 'alpha' (line 886)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 8), 'alpha', result_div_113333)
        
        # Assigning a Call to a Name (line 888):
        
        # Assigning a Call to a Name (line 888):
        
        # Call to linspace(...): (line 888)
        # Processing the call arguments (line 888)
        # Getting the type of 'eta1' (line 888)
        eta1_113336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 28), 'eta1', False)
        # Getting the type of 'eta2' (line 888)
        eta2_113337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 34), 'eta2', False)
        # Getting the type of 'n' (line 888)
        n_113338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 40), 'n', False)
        int_113339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 44), 'int')
        # Applying the binary operator '+' (line 888)
        result_add_113340 = python_operator(stypy.reporting.localization.Localization(__file__, 888, 40), '+', n_113338, int_113339)
        
        # Getting the type of 'True' (line 888)
        True_113341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 47), 'True', False)
        # Processing the call keyword arguments (line 888)
        kwargs_113342 = {}
        # Getting the type of 'np' (line 888)
        np_113334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 16), 'np', False)
        # Obtaining the member 'linspace' of a type (line 888)
        linspace_113335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 16), np_113334, 'linspace')
        # Calling linspace(args, kwargs) (line 888)
        linspace_call_result_113343 = invoke(stypy.reporting.localization.Localization(__file__, 888, 16), linspace_113335, *[eta1_113336, eta2_113337, result_add_113340, True_113341], **kwargs_113342)
        
        # Assigning a type to the variable 'steps' (line 888)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'steps', linspace_call_result_113343)
        
        # Assigning a Call to a Name (line 889):
        
        # Assigning a Call to a Name (line 889):
        
        # Call to cos(...): (line 889)
        # Processing the call arguments (line 889)
        # Getting the type of 'steps' (line 889)
        steps_113346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 25), 'steps', False)
        # Processing the call keyword arguments (line 889)
        kwargs_113347 = {}
        # Getting the type of 'np' (line 889)
        np_113344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 18), 'np', False)
        # Obtaining the member 'cos' of a type (line 889)
        cos_113345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 18), np_113344, 'cos')
        # Calling cos(args, kwargs) (line 889)
        cos_call_result_113348 = invoke(stypy.reporting.localization.Localization(__file__, 889, 18), cos_113345, *[steps_113346], **kwargs_113347)
        
        # Assigning a type to the variable 'cos_eta' (line 889)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 889, 8), 'cos_eta', cos_call_result_113348)
        
        # Assigning a Call to a Name (line 890):
        
        # Assigning a Call to a Name (line 890):
        
        # Call to sin(...): (line 890)
        # Processing the call arguments (line 890)
        # Getting the type of 'steps' (line 890)
        steps_113351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 25), 'steps', False)
        # Processing the call keyword arguments (line 890)
        kwargs_113352 = {}
        # Getting the type of 'np' (line 890)
        np_113349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 18), 'np', False)
        # Obtaining the member 'sin' of a type (line 890)
        sin_113350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 890, 18), np_113349, 'sin')
        # Calling sin(args, kwargs) (line 890)
        sin_call_result_113353 = invoke(stypy.reporting.localization.Localization(__file__, 890, 18), sin_113350, *[steps_113351], **kwargs_113352)
        
        # Assigning a type to the variable 'sin_eta' (line 890)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 890, 8), 'sin_eta', sin_call_result_113353)
        
        # Assigning a Subscript to a Name (line 892):
        
        # Assigning a Subscript to a Name (line 892):
        
        # Obtaining the type of the subscript
        int_113354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 22), 'int')
        slice_113355 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 892, 13), None, int_113354, None)
        # Getting the type of 'cos_eta' (line 892)
        cos_eta_113356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 13), 'cos_eta')
        # Obtaining the member '__getitem__' of a type (line 892)
        getitem___113357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 13), cos_eta_113356, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 892)
        subscript_call_result_113358 = invoke(stypy.reporting.localization.Localization(__file__, 892, 13), getitem___113357, slice_113355)
        
        # Assigning a type to the variable 'xA' (line 892)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 8), 'xA', subscript_call_result_113358)
        
        # Assigning a Subscript to a Name (line 893):
        
        # Assigning a Subscript to a Name (line 893):
        
        # Obtaining the type of the subscript
        int_113359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 22), 'int')
        slice_113360 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 893, 13), None, int_113359, None)
        # Getting the type of 'sin_eta' (line 893)
        sin_eta_113361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 13), 'sin_eta')
        # Obtaining the member '__getitem__' of a type (line 893)
        getitem___113362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 893, 13), sin_eta_113361, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 893)
        subscript_call_result_113363 = invoke(stypy.reporting.localization.Localization(__file__, 893, 13), getitem___113362, slice_113360)
        
        # Assigning a type to the variable 'yA' (line 893)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 8), 'yA', subscript_call_result_113363)
        
        # Assigning a UnaryOp to a Name (line 894):
        
        # Assigning a UnaryOp to a Name (line 894):
        
        # Getting the type of 'yA' (line 894)
        yA_113364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 18), 'yA')
        # Applying the 'usub' unary operator (line 894)
        result___neg___113365 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 17), 'usub', yA_113364)
        
        # Assigning a type to the variable 'xA_dot' (line 894)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 8), 'xA_dot', result___neg___113365)
        
        # Assigning a Name to a Name (line 895):
        
        # Assigning a Name to a Name (line 895):
        # Getting the type of 'xA' (line 895)
        xA_113366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 17), 'xA')
        # Assigning a type to the variable 'yA_dot' (line 895)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 8), 'yA_dot', xA_113366)
        
        # Assigning a Subscript to a Name (line 897):
        
        # Assigning a Subscript to a Name (line 897):
        
        # Obtaining the type of the subscript
        int_113367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 21), 'int')
        slice_113368 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 897, 13), int_113367, None, None)
        # Getting the type of 'cos_eta' (line 897)
        cos_eta_113369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 13), 'cos_eta')
        # Obtaining the member '__getitem__' of a type (line 897)
        getitem___113370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 13), cos_eta_113369, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 897)
        subscript_call_result_113371 = invoke(stypy.reporting.localization.Localization(__file__, 897, 13), getitem___113370, slice_113368)
        
        # Assigning a type to the variable 'xB' (line 897)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 8), 'xB', subscript_call_result_113371)
        
        # Assigning a Subscript to a Name (line 898):
        
        # Assigning a Subscript to a Name (line 898):
        
        # Obtaining the type of the subscript
        int_113372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 21), 'int')
        slice_113373 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 898, 13), int_113372, None, None)
        # Getting the type of 'sin_eta' (line 898)
        sin_eta_113374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 13), 'sin_eta')
        # Obtaining the member '__getitem__' of a type (line 898)
        getitem___113375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 898, 13), sin_eta_113374, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 898)
        subscript_call_result_113376 = invoke(stypy.reporting.localization.Localization(__file__, 898, 13), getitem___113375, slice_113373)
        
        # Assigning a type to the variable 'yB' (line 898)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 898, 8), 'yB', subscript_call_result_113376)
        
        # Assigning a UnaryOp to a Name (line 899):
        
        # Assigning a UnaryOp to a Name (line 899):
        
        # Getting the type of 'yB' (line 899)
        yB_113377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 18), 'yB')
        # Applying the 'usub' unary operator (line 899)
        result___neg___113378 = python_operator(stypy.reporting.localization.Localization(__file__, 899, 17), 'usub', yB_113377)
        
        # Assigning a type to the variable 'xB_dot' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 8), 'xB_dot', result___neg___113378)
        
        # Assigning a Name to a Name (line 900):
        
        # Assigning a Name to a Name (line 900):
        # Getting the type of 'xB' (line 900)
        xB_113379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 17), 'xB')
        # Assigning a type to the variable 'yB_dot' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 8), 'yB_dot', xB_113379)
        
        # Getting the type of 'is_wedge' (line 902)
        is_wedge_113380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 11), 'is_wedge')
        # Testing the type of an if condition (line 902)
        if_condition_113381 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 902, 8), is_wedge_113380)
        # Assigning a type to the variable 'if_condition_113381' (line 902)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 8), 'if_condition_113381', if_condition_113381)
        # SSA begins for if statement (line 902)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 903):
        
        # Assigning a BinOp to a Name (line 903):
        # Getting the type of 'n' (line 903)
        n_113382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 21), 'n')
        int_113383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 25), 'int')
        # Applying the binary operator '*' (line 903)
        result_mul_113384 = python_operator(stypy.reporting.localization.Localization(__file__, 903, 21), '*', n_113382, int_113383)
        
        int_113385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 29), 'int')
        # Applying the binary operator '+' (line 903)
        result_add_113386 = python_operator(stypy.reporting.localization.Localization(__file__, 903, 21), '+', result_mul_113384, int_113385)
        
        # Assigning a type to the variable 'length' (line 903)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 12), 'length', result_add_113386)
        
        # Assigning a Call to a Name (line 904):
        
        # Assigning a Call to a Name (line 904):
        
        # Call to zeros(...): (line 904)
        # Processing the call arguments (line 904)
        
        # Obtaining an instance of the builtin type 'tuple' (line 904)
        tuple_113389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 904, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 904)
        # Adding element type (line 904)
        # Getting the type of 'length' (line 904)
        length_113390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 33), 'length', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 904, 33), tuple_113389, length_113390)
        # Adding element type (line 904)
        int_113391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 904, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 904, 33), tuple_113389, int_113391)
        
        # Getting the type of 'float' (line 904)
        float_113392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 45), 'float', False)
        # Processing the call keyword arguments (line 904)
        kwargs_113393 = {}
        # Getting the type of 'np' (line 904)
        np_113387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 23), 'np', False)
        # Obtaining the member 'zeros' of a type (line 904)
        zeros_113388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 904, 23), np_113387, 'zeros')
        # Calling zeros(args, kwargs) (line 904)
        zeros_call_result_113394 = invoke(stypy.reporting.localization.Localization(__file__, 904, 23), zeros_113388, *[tuple_113389, float_113392], **kwargs_113393)
        
        # Assigning a type to the variable 'vertices' (line 904)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 904, 12), 'vertices', zeros_call_result_113394)
        
        # Assigning a BinOp to a Name (line 905):
        
        # Assigning a BinOp to a Name (line 905):
        # Getting the type of 'cls' (line 905)
        cls_113395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 20), 'cls')
        # Obtaining the member 'CURVE4' of a type (line 905)
        CURVE4_113396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 20), cls_113395, 'CURVE4')
        
        # Call to ones(...): (line 905)
        # Processing the call arguments (line 905)
        
        # Obtaining an instance of the builtin type 'tuple' (line 905)
        tuple_113399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 905)
        # Adding element type (line 905)
        # Getting the type of 'length' (line 905)
        length_113400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 42), 'length', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 905, 42), tuple_113399, length_113400)
        
        # Getting the type of 'cls' (line 905)
        cls_113401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 53), 'cls', False)
        # Obtaining the member 'code_type' of a type (line 905)
        code_type_113402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 53), cls_113401, 'code_type')
        # Processing the call keyword arguments (line 905)
        kwargs_113403 = {}
        # Getting the type of 'np' (line 905)
        np_113397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 33), 'np', False)
        # Obtaining the member 'ones' of a type (line 905)
        ones_113398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 33), np_113397, 'ones')
        # Calling ones(args, kwargs) (line 905)
        ones_call_result_113404 = invoke(stypy.reporting.localization.Localization(__file__, 905, 33), ones_113398, *[tuple_113399, code_type_113402], **kwargs_113403)
        
        # Applying the binary operator '*' (line 905)
        result_mul_113405 = python_operator(stypy.reporting.localization.Localization(__file__, 905, 20), '*', CURVE4_113396, ones_call_result_113404)
        
        # Assigning a type to the variable 'codes' (line 905)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 12), 'codes', result_mul_113405)
        
        # Assigning a List to a Subscript (line 906):
        
        # Assigning a List to a Subscript (line 906):
        
        # Obtaining an instance of the builtin type 'list' (line 906)
        list_113406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 906)
        # Adding element type (line 906)
        
        # Obtaining the type of the subscript
        int_113407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 30), 'int')
        # Getting the type of 'xA' (line 906)
        xA_113408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 27), 'xA')
        # Obtaining the member '__getitem__' of a type (line 906)
        getitem___113409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 27), xA_113408, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 906)
        subscript_call_result_113410 = invoke(stypy.reporting.localization.Localization(__file__, 906, 27), getitem___113409, int_113407)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 906, 26), list_113406, subscript_call_result_113410)
        # Adding element type (line 906)
        
        # Obtaining the type of the subscript
        int_113411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 37), 'int')
        # Getting the type of 'yA' (line 906)
        yA_113412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 34), 'yA')
        # Obtaining the member '__getitem__' of a type (line 906)
        getitem___113413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 34), yA_113412, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 906)
        subscript_call_result_113414 = invoke(stypy.reporting.localization.Localization(__file__, 906, 34), getitem___113413, int_113411)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 906, 26), list_113406, subscript_call_result_113414)
        
        # Getting the type of 'vertices' (line 906)
        vertices_113415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 12), 'vertices')
        int_113416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 21), 'int')
        # Storing an element on a container (line 906)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 906, 12), vertices_113415, (int_113416, list_113406))
        
        # Assigning a List to a Subscript (line 907):
        
        # Assigning a List to a Subscript (line 907):
        
        # Obtaining an instance of the builtin type 'list' (line 907)
        list_113417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 907)
        # Adding element type (line 907)
        # Getting the type of 'cls' (line 907)
        cls_113418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 26), 'cls')
        # Obtaining the member 'MOVETO' of a type (line 907)
        MOVETO_113419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 26), cls_113418, 'MOVETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 25), list_113417, MOVETO_113419)
        # Adding element type (line 907)
        # Getting the type of 'cls' (line 907)
        cls_113420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 38), 'cls')
        # Obtaining the member 'LINETO' of a type (line 907)
        LINETO_113421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 38), cls_113420, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 25), list_113417, LINETO_113421)
        
        # Getting the type of 'codes' (line 907)
        codes_113422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 12), 'codes')
        int_113423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 18), 'int')
        int_113424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 20), 'int')
        slice_113425 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 907, 12), int_113423, int_113424, None)
        # Storing an element on a container (line 907)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 12), codes_113422, (slice_113425, list_113417))
        
        # Assigning a List to a Subscript (line 908):
        
        # Assigning a List to a Subscript (line 908):
        
        # Obtaining an instance of the builtin type 'list' (line 908)
        list_113426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 908, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 908)
        # Adding element type (line 908)
        # Getting the type of 'cls' (line 908)
        cls_113427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 26), 'cls')
        # Obtaining the member 'LINETO' of a type (line 908)
        LINETO_113428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 26), cls_113427, 'LINETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 908, 25), list_113426, LINETO_113428)
        # Adding element type (line 908)
        # Getting the type of 'cls' (line 908)
        cls_113429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 38), 'cls')
        # Obtaining the member 'CLOSEPOLY' of a type (line 908)
        CLOSEPOLY_113430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 38), cls_113429, 'CLOSEPOLY')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 908, 25), list_113426, CLOSEPOLY_113430)
        
        # Getting the type of 'codes' (line 908)
        codes_113431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 12), 'codes')
        int_113432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 908, 18), 'int')
        slice_113433 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 908, 12), int_113432, None, None)
        # Storing an element on a container (line 908)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 908, 12), codes_113431, (slice_113433, list_113426))
        
        # Assigning a Num to a Name (line 909):
        
        # Assigning a Num to a Name (line 909):
        int_113434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 28), 'int')
        # Assigning a type to the variable 'vertex_offset' (line 909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 12), 'vertex_offset', int_113434)
        
        # Assigning a BinOp to a Name (line 910):
        
        # Assigning a BinOp to a Name (line 910):
        # Getting the type of 'length' (line 910)
        length_113435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 18), 'length')
        int_113436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 27), 'int')
        # Applying the binary operator '-' (line 910)
        result_sub_113437 = python_operator(stypy.reporting.localization.Localization(__file__, 910, 18), '-', length_113435, int_113436)
        
        # Assigning a type to the variable 'end' (line 910)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 12), 'end', result_sub_113437)
        # SSA branch for the else part of an if statement (line 902)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 912):
        
        # Assigning a BinOp to a Name (line 912):
        # Getting the type of 'n' (line 912)
        n_113438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 21), 'n')
        int_113439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 25), 'int')
        # Applying the binary operator '*' (line 912)
        result_mul_113440 = python_operator(stypy.reporting.localization.Localization(__file__, 912, 21), '*', n_113438, int_113439)
        
        int_113441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 29), 'int')
        # Applying the binary operator '+' (line 912)
        result_add_113442 = python_operator(stypy.reporting.localization.Localization(__file__, 912, 21), '+', result_mul_113440, int_113441)
        
        # Assigning a type to the variable 'length' (line 912)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 912, 12), 'length', result_add_113442)
        
        # Assigning a Call to a Name (line 913):
        
        # Assigning a Call to a Name (line 913):
        
        # Call to empty(...): (line 913)
        # Processing the call arguments (line 913)
        
        # Obtaining an instance of the builtin type 'tuple' (line 913)
        tuple_113445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 913)
        # Adding element type (line 913)
        # Getting the type of 'length' (line 913)
        length_113446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 33), 'length', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 913, 33), tuple_113445, length_113446)
        # Adding element type (line 913)
        int_113447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 913, 33), tuple_113445, int_113447)
        
        # Getting the type of 'float' (line 913)
        float_113448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 45), 'float', False)
        # Processing the call keyword arguments (line 913)
        kwargs_113449 = {}
        # Getting the type of 'np' (line 913)
        np_113443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 23), 'np', False)
        # Obtaining the member 'empty' of a type (line 913)
        empty_113444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 23), np_113443, 'empty')
        # Calling empty(args, kwargs) (line 913)
        empty_call_result_113450 = invoke(stypy.reporting.localization.Localization(__file__, 913, 23), empty_113444, *[tuple_113445, float_113448], **kwargs_113449)
        
        # Assigning a type to the variable 'vertices' (line 913)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 12), 'vertices', empty_call_result_113450)
        
        # Assigning a BinOp to a Name (line 914):
        
        # Assigning a BinOp to a Name (line 914):
        # Getting the type of 'cls' (line 914)
        cls_113451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 20), 'cls')
        # Obtaining the member 'CURVE4' of a type (line 914)
        CURVE4_113452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 20), cls_113451, 'CURVE4')
        
        # Call to ones(...): (line 914)
        # Processing the call arguments (line 914)
        
        # Obtaining an instance of the builtin type 'tuple' (line 914)
        tuple_113455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 914)
        # Adding element type (line 914)
        # Getting the type of 'length' (line 914)
        length_113456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 42), 'length', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 914, 42), tuple_113455, length_113456)
        
        # Getting the type of 'cls' (line 914)
        cls_113457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 53), 'cls', False)
        # Obtaining the member 'code_type' of a type (line 914)
        code_type_113458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 53), cls_113457, 'code_type')
        # Processing the call keyword arguments (line 914)
        kwargs_113459 = {}
        # Getting the type of 'np' (line 914)
        np_113453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 33), 'np', False)
        # Obtaining the member 'ones' of a type (line 914)
        ones_113454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 33), np_113453, 'ones')
        # Calling ones(args, kwargs) (line 914)
        ones_call_result_113460 = invoke(stypy.reporting.localization.Localization(__file__, 914, 33), ones_113454, *[tuple_113455, code_type_113458], **kwargs_113459)
        
        # Applying the binary operator '*' (line 914)
        result_mul_113461 = python_operator(stypy.reporting.localization.Localization(__file__, 914, 20), '*', CURVE4_113452, ones_call_result_113460)
        
        # Assigning a type to the variable 'codes' (line 914)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 914, 12), 'codes', result_mul_113461)
        
        # Assigning a List to a Subscript (line 915):
        
        # Assigning a List to a Subscript (line 915):
        
        # Obtaining an instance of the builtin type 'list' (line 915)
        list_113462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 915)
        # Adding element type (line 915)
        
        # Obtaining the type of the subscript
        int_113463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 30), 'int')
        # Getting the type of 'xA' (line 915)
        xA_113464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 27), 'xA')
        # Obtaining the member '__getitem__' of a type (line 915)
        getitem___113465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 27), xA_113464, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 915)
        subscript_call_result_113466 = invoke(stypy.reporting.localization.Localization(__file__, 915, 27), getitem___113465, int_113463)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 915, 26), list_113462, subscript_call_result_113466)
        # Adding element type (line 915)
        
        # Obtaining the type of the subscript
        int_113467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 37), 'int')
        # Getting the type of 'yA' (line 915)
        yA_113468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 34), 'yA')
        # Obtaining the member '__getitem__' of a type (line 915)
        getitem___113469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 34), yA_113468, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 915)
        subscript_call_result_113470 = invoke(stypy.reporting.localization.Localization(__file__, 915, 34), getitem___113469, int_113467)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 915, 26), list_113462, subscript_call_result_113470)
        
        # Getting the type of 'vertices' (line 915)
        vertices_113471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 12), 'vertices')
        int_113472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 21), 'int')
        # Storing an element on a container (line 915)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 915, 12), vertices_113471, (int_113472, list_113462))
        
        # Assigning a Attribute to a Subscript (line 916):
        
        # Assigning a Attribute to a Subscript (line 916):
        # Getting the type of 'cls' (line 916)
        cls_113473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 23), 'cls')
        # Obtaining the member 'MOVETO' of a type (line 916)
        MOVETO_113474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 23), cls_113473, 'MOVETO')
        # Getting the type of 'codes' (line 916)
        codes_113475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 12), 'codes')
        int_113476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 18), 'int')
        # Storing an element on a container (line 916)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 916, 12), codes_113475, (int_113476, MOVETO_113474))
        
        # Assigning a Num to a Name (line 917):
        
        # Assigning a Num to a Name (line 917):
        int_113477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 917, 28), 'int')
        # Assigning a type to the variable 'vertex_offset' (line 917)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 12), 'vertex_offset', int_113477)
        
        # Assigning a Name to a Name (line 918):
        
        # Assigning a Name to a Name (line 918):
        # Getting the type of 'length' (line 918)
        length_113478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 18), 'length')
        # Assigning a type to the variable 'end' (line 918)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 12), 'end', length_113478)
        # SSA join for if statement (line 902)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Subscript (line 920):
        
        # Assigning a BinOp to a Subscript (line 920):
        # Getting the type of 'xA' (line 920)
        xA_113479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 43), 'xA')
        # Getting the type of 'alpha' (line 920)
        alpha_113480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 48), 'alpha')
        # Getting the type of 'xA_dot' (line 920)
        xA_dot_113481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 56), 'xA_dot')
        # Applying the binary operator '*' (line 920)
        result_mul_113482 = python_operator(stypy.reporting.localization.Localization(__file__, 920, 48), '*', alpha_113480, xA_dot_113481)
        
        # Applying the binary operator '+' (line 920)
        result_add_113483 = python_operator(stypy.reporting.localization.Localization(__file__, 920, 43), '+', xA_113479, result_mul_113482)
        
        # Getting the type of 'vertices' (line 920)
        vertices_113484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 8), 'vertices')
        # Getting the type of 'vertex_offset' (line 920)
        vertex_offset_113485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 17), 'vertex_offset')
        # Getting the type of 'end' (line 920)
        end_113486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 31), 'end')
        int_113487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 35), 'int')
        slice_113488 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 920, 8), vertex_offset_113485, end_113486, int_113487)
        int_113489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 38), 'int')
        # Storing an element on a container (line 920)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 920, 8), vertices_113484, ((slice_113488, int_113489), result_add_113483))
        
        # Assigning a BinOp to a Subscript (line 921):
        
        # Assigning a BinOp to a Subscript (line 921):
        # Getting the type of 'yA' (line 921)
        yA_113490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 43), 'yA')
        # Getting the type of 'alpha' (line 921)
        alpha_113491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 48), 'alpha')
        # Getting the type of 'yA_dot' (line 921)
        yA_dot_113492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 56), 'yA_dot')
        # Applying the binary operator '*' (line 921)
        result_mul_113493 = python_operator(stypy.reporting.localization.Localization(__file__, 921, 48), '*', alpha_113491, yA_dot_113492)
        
        # Applying the binary operator '+' (line 921)
        result_add_113494 = python_operator(stypy.reporting.localization.Localization(__file__, 921, 43), '+', yA_113490, result_mul_113493)
        
        # Getting the type of 'vertices' (line 921)
        vertices_113495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 8), 'vertices')
        # Getting the type of 'vertex_offset' (line 921)
        vertex_offset_113496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 17), 'vertex_offset')
        # Getting the type of 'end' (line 921)
        end_113497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 31), 'end')
        int_113498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 35), 'int')
        slice_113499 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 921, 8), vertex_offset_113496, end_113497, int_113498)
        int_113500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 38), 'int')
        # Storing an element on a container (line 921)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 921, 8), vertices_113495, ((slice_113499, int_113500), result_add_113494))
        
        # Assigning a BinOp to a Subscript (line 922):
        
        # Assigning a BinOp to a Subscript (line 922):
        # Getting the type of 'xB' (line 922)
        xB_113501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 45), 'xB')
        # Getting the type of 'alpha' (line 922)
        alpha_113502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 50), 'alpha')
        # Getting the type of 'xB_dot' (line 922)
        xB_dot_113503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 58), 'xB_dot')
        # Applying the binary operator '*' (line 922)
        result_mul_113504 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 50), '*', alpha_113502, xB_dot_113503)
        
        # Applying the binary operator '-' (line 922)
        result_sub_113505 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 45), '-', xB_113501, result_mul_113504)
        
        # Getting the type of 'vertices' (line 922)
        vertices_113506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'vertices')
        # Getting the type of 'vertex_offset' (line 922)
        vertex_offset_113507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 17), 'vertex_offset')
        int_113508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 31), 'int')
        # Applying the binary operator '+' (line 922)
        result_add_113509 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 17), '+', vertex_offset_113507, int_113508)
        
        # Getting the type of 'end' (line 922)
        end_113510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 33), 'end')
        int_113511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 37), 'int')
        slice_113512 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 922, 8), result_add_113509, end_113510, int_113511)
        int_113513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 40), 'int')
        # Storing an element on a container (line 922)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 922, 8), vertices_113506, ((slice_113512, int_113513), result_sub_113505))
        
        # Assigning a BinOp to a Subscript (line 923):
        
        # Assigning a BinOp to a Subscript (line 923):
        # Getting the type of 'yB' (line 923)
        yB_113514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 45), 'yB')
        # Getting the type of 'alpha' (line 923)
        alpha_113515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 50), 'alpha')
        # Getting the type of 'yB_dot' (line 923)
        yB_dot_113516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 58), 'yB_dot')
        # Applying the binary operator '*' (line 923)
        result_mul_113517 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 50), '*', alpha_113515, yB_dot_113516)
        
        # Applying the binary operator '-' (line 923)
        result_sub_113518 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 45), '-', yB_113514, result_mul_113517)
        
        # Getting the type of 'vertices' (line 923)
        vertices_113519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 8), 'vertices')
        # Getting the type of 'vertex_offset' (line 923)
        vertex_offset_113520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 17), 'vertex_offset')
        int_113521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 31), 'int')
        # Applying the binary operator '+' (line 923)
        result_add_113522 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 17), '+', vertex_offset_113520, int_113521)
        
        # Getting the type of 'end' (line 923)
        end_113523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 33), 'end')
        int_113524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 37), 'int')
        slice_113525 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 923, 8), result_add_113522, end_113523, int_113524)
        int_113526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 40), 'int')
        # Storing an element on a container (line 923)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 923, 8), vertices_113519, ((slice_113525, int_113526), result_sub_113518))
        
        # Assigning a Name to a Subscript (line 924):
        
        # Assigning a Name to a Subscript (line 924):
        # Getting the type of 'xB' (line 924)
        xB_113527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 45), 'xB')
        # Getting the type of 'vertices' (line 924)
        vertices_113528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'vertices')
        # Getting the type of 'vertex_offset' (line 924)
        vertex_offset_113529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 17), 'vertex_offset')
        int_113530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 31), 'int')
        # Applying the binary operator '+' (line 924)
        result_add_113531 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 17), '+', vertex_offset_113529, int_113530)
        
        # Getting the type of 'end' (line 924)
        end_113532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 33), 'end')
        int_113533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 37), 'int')
        slice_113534 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 924, 8), result_add_113531, end_113532, int_113533)
        int_113535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 40), 'int')
        # Storing an element on a container (line 924)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 924, 8), vertices_113528, ((slice_113534, int_113535), xB_113527))
        
        # Assigning a Name to a Subscript (line 925):
        
        # Assigning a Name to a Subscript (line 925):
        # Getting the type of 'yB' (line 925)
        yB_113536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 45), 'yB')
        # Getting the type of 'vertices' (line 925)
        vertices_113537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 8), 'vertices')
        # Getting the type of 'vertex_offset' (line 925)
        vertex_offset_113538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 17), 'vertex_offset')
        int_113539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 31), 'int')
        # Applying the binary operator '+' (line 925)
        result_add_113540 = python_operator(stypy.reporting.localization.Localization(__file__, 925, 17), '+', vertex_offset_113538, int_113539)
        
        # Getting the type of 'end' (line 925)
        end_113541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 33), 'end')
        int_113542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 37), 'int')
        slice_113543 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 925, 8), result_add_113540, end_113541, int_113542)
        int_113544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 40), 'int')
        # Storing an element on a container (line 925)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 925, 8), vertices_113537, ((slice_113543, int_113544), yB_113536))
        
        # Call to cls(...): (line 927)
        # Processing the call arguments (line 927)
        # Getting the type of 'vertices' (line 927)
        vertices_113546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 19), 'vertices', False)
        # Getting the type of 'codes' (line 927)
        codes_113547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 29), 'codes', False)
        # Processing the call keyword arguments (line 927)
        # Getting the type of 'True' (line 927)
        True_113548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 45), 'True', False)
        keyword_113549 = True_113548
        kwargs_113550 = {'readonly': keyword_113549}
        # Getting the type of 'cls' (line 927)
        cls_113545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 927)
        cls_call_result_113551 = invoke(stypy.reporting.localization.Localization(__file__, 927, 15), cls_113545, *[vertices_113546, codes_113547], **kwargs_113550)
        
        # Assigning a type to the variable 'stypy_return_type' (line 927)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 8), 'stypy_return_type', cls_call_result_113551)
        
        # ################# End of 'arc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'arc' in the type store
        # Getting the type of 'stypy_return_type' (line 850)
        stypy_return_type_113552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'arc'
        return stypy_return_type_113552


    @norecursion
    def wedge(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 930)
        None_113553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 37), 'None')
        defaults = [None_113553]
        # Create a new context for function 'wedge'
        module_type_store = module_type_store.open_function_context('wedge', 929, 4, False)
        # Assigning a type to the variable 'self' (line 930)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.wedge.__dict__.__setitem__('stypy_localization', localization)
        Path.wedge.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.wedge.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.wedge.__dict__.__setitem__('stypy_function_name', 'Path.wedge')
        Path.wedge.__dict__.__setitem__('stypy_param_names_list', ['theta1', 'theta2', 'n'])
        Path.wedge.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.wedge.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.wedge.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.wedge.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.wedge.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.wedge.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.wedge', ['theta1', 'theta2', 'n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wedge', localization, ['theta1', 'theta2', 'n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wedge(...)' code ##################

        unicode_113554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, (-1)), 'unicode', u'\n        Return a wedge of the unit circle from angle\n        *theta1* to angle *theta2* (in degrees).\n\n        *theta2* is unwrapped to produce the shortest wedge within 360 degrees.\n        That is, if *theta2* > *theta1* + 360, the wedge will be from *theta1*\n        to *theta2* - 360 and not a full circle plus some extra overlap.\n\n        If *n* is provided, it is the number of spline segments to make.\n        If *n* is not provided, the number of spline segments is\n        determined based on the delta between *theta1* and *theta2*.\n        ')
        
        # Call to arc(...): (line 943)
        # Processing the call arguments (line 943)
        # Getting the type of 'theta1' (line 943)
        theta1_113557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 23), 'theta1', False)
        # Getting the type of 'theta2' (line 943)
        theta2_113558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 31), 'theta2', False)
        # Getting the type of 'n' (line 943)
        n_113559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 39), 'n', False)
        # Getting the type of 'True' (line 943)
        True_113560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 42), 'True', False)
        # Processing the call keyword arguments (line 943)
        kwargs_113561 = {}
        # Getting the type of 'cls' (line 943)
        cls_113555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 15), 'cls', False)
        # Obtaining the member 'arc' of a type (line 943)
        arc_113556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 943, 15), cls_113555, 'arc')
        # Calling arc(args, kwargs) (line 943)
        arc_call_result_113562 = invoke(stypy.reporting.localization.Localization(__file__, 943, 15), arc_113556, *[theta1_113557, theta2_113558, n_113559, True_113560], **kwargs_113561)
        
        # Assigning a type to the variable 'stypy_return_type' (line 943)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 943, 8), 'stypy_return_type', arc_call_result_113562)
        
        # ################# End of 'wedge(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wedge' in the type store
        # Getting the type of 'stypy_return_type' (line 929)
        stypy_return_type_113563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113563)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wedge'
        return stypy_return_type_113563

    
    # Assigning a Call to a Name (line 945):

    @norecursion
    def hatch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_113564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 41), 'int')
        defaults = [int_113564]
        # Create a new context for function 'hatch'
        module_type_store = module_type_store.open_function_context('hatch', 947, 4, False)
        # Assigning a type to the variable 'self' (line 948)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.hatch.__dict__.__setitem__('stypy_localization', localization)
        Path.hatch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.hatch.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.hatch.__dict__.__setitem__('stypy_function_name', 'Path.hatch')
        Path.hatch.__dict__.__setitem__('stypy_param_names_list', ['hatchpattern', 'density'])
        Path.hatch.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.hatch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.hatch.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.hatch.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.hatch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.hatch.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.hatch', ['hatchpattern', 'density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'hatch', localization, ['hatchpattern', 'density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'hatch(...)' code ##################

        unicode_113565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, (-1)), 'unicode', u'\n        Given a hatch specifier, *hatchpattern*, generates a Path that\n        can be used in a repeated hatching pattern.  *density* is the\n        number of lines per unit square.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 954, 8))
        
        # 'from matplotlib.hatch import get_path' statement (line 954)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
        import_113566 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 954, 8), 'matplotlib.hatch')

        if (type(import_113566) is not StypyTypeError):

            if (import_113566 != 'pyd_module'):
                __import__(import_113566)
                sys_modules_113567 = sys.modules[import_113566]
                import_from_module(stypy.reporting.localization.Localization(__file__, 954, 8), 'matplotlib.hatch', sys_modules_113567.module_type_store, module_type_store, ['get_path'])
                nest_module(stypy.reporting.localization.Localization(__file__, 954, 8), __file__, sys_modules_113567, sys_modules_113567.module_type_store, module_type_store)
            else:
                from matplotlib.hatch import get_path

                import_from_module(stypy.reporting.localization.Localization(__file__, 954, 8), 'matplotlib.hatch', None, module_type_store, ['get_path'], [get_path])

        else:
            # Assigning a type to the variable 'matplotlib.hatch' (line 954)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 8), 'matplotlib.hatch', import_113566)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
        
        
        # Type idiom detected: calculating its left and rigth part (line 956)
        # Getting the type of 'hatchpattern' (line 956)
        hatchpattern_113568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 11), 'hatchpattern')
        # Getting the type of 'None' (line 956)
        None_113569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 27), 'None')
        
        (may_be_113570, more_types_in_union_113571) = may_be_none(hatchpattern_113568, None_113569)

        if may_be_113570:

            if more_types_in_union_113571:
                # Runtime conditional SSA (line 956)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 957)
            None_113572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 957)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 957, 12), 'stypy_return_type', None_113572)

            if more_types_in_union_113571:
                # SSA join for if statement (line 956)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 959):
        
        # Assigning a Call to a Name (line 959):
        
        # Call to get(...): (line 959)
        # Processing the call arguments (line 959)
        
        # Obtaining an instance of the builtin type 'tuple' (line 959)
        tuple_113576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 959)
        # Adding element type (line 959)
        # Getting the type of 'hatchpattern' (line 959)
        hatchpattern_113577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 42), 'hatchpattern', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 959, 42), tuple_113576, hatchpattern_113577)
        # Adding element type (line 959)
        # Getting the type of 'density' (line 959)
        density_113578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 56), 'density', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 959, 42), tuple_113576, density_113578)
        
        # Processing the call keyword arguments (line 959)
        kwargs_113579 = {}
        # Getting the type of 'cls' (line 959)
        cls_113573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 21), 'cls', False)
        # Obtaining the member '_hatch_dict' of a type (line 959)
        _hatch_dict_113574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 959, 21), cls_113573, '_hatch_dict')
        # Obtaining the member 'get' of a type (line 959)
        get_113575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 959, 21), _hatch_dict_113574, 'get')
        # Calling get(args, kwargs) (line 959)
        get_call_result_113580 = invoke(stypy.reporting.localization.Localization(__file__, 959, 21), get_113575, *[tuple_113576], **kwargs_113579)
        
        # Assigning a type to the variable 'hatch_path' (line 959)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 8), 'hatch_path', get_call_result_113580)
        
        # Type idiom detected: calculating its left and rigth part (line 960)
        # Getting the type of 'hatch_path' (line 960)
        hatch_path_113581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 8), 'hatch_path')
        # Getting the type of 'None' (line 960)
        None_113582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 29), 'None')
        
        (may_be_113583, more_types_in_union_113584) = may_not_be_none(hatch_path_113581, None_113582)

        if may_be_113583:

            if more_types_in_union_113584:
                # Runtime conditional SSA (line 960)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'hatch_path' (line 961)
            hatch_path_113585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 19), 'hatch_path')
            # Assigning a type to the variable 'stypy_return_type' (line 961)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 12), 'stypy_return_type', hatch_path_113585)

            if more_types_in_union_113584:
                # SSA join for if statement (line 960)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 963):
        
        # Assigning a Call to a Name (line 963):
        
        # Call to get_path(...): (line 963)
        # Processing the call arguments (line 963)
        # Getting the type of 'hatchpattern' (line 963)
        hatchpattern_113587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 30), 'hatchpattern', False)
        # Getting the type of 'density' (line 963)
        density_113588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 44), 'density', False)
        # Processing the call keyword arguments (line 963)
        kwargs_113589 = {}
        # Getting the type of 'get_path' (line 963)
        get_path_113586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 21), 'get_path', False)
        # Calling get_path(args, kwargs) (line 963)
        get_path_call_result_113590 = invoke(stypy.reporting.localization.Localization(__file__, 963, 21), get_path_113586, *[hatchpattern_113587, density_113588], **kwargs_113589)
        
        # Assigning a type to the variable 'hatch_path' (line 963)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 8), 'hatch_path', get_path_call_result_113590)
        
        # Assigning a Name to a Subscript (line 964):
        
        # Assigning a Name to a Subscript (line 964):
        # Getting the type of 'hatch_path' (line 964)
        hatch_path_113591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 51), 'hatch_path')
        # Getting the type of 'cls' (line 964)
        cls_113592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 8), 'cls')
        # Obtaining the member '_hatch_dict' of a type (line 964)
        _hatch_dict_113593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 964, 8), cls_113592, '_hatch_dict')
        
        # Obtaining an instance of the builtin type 'tuple' (line 964)
        tuple_113594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 964)
        # Adding element type (line 964)
        # Getting the type of 'hatchpattern' (line 964)
        hatchpattern_113595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 25), 'hatchpattern')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 25), tuple_113594, hatchpattern_113595)
        # Adding element type (line 964)
        # Getting the type of 'density' (line 964)
        density_113596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 39), 'density')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 25), tuple_113594, density_113596)
        
        # Storing an element on a container (line 964)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 8), _hatch_dict_113593, (tuple_113594, hatch_path_113591))
        # Getting the type of 'hatch_path' (line 965)
        hatch_path_113597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 15), 'hatch_path')
        # Assigning a type to the variable 'stypy_return_type' (line 965)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 8), 'stypy_return_type', hatch_path_113597)
        
        # ################# End of 'hatch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'hatch' in the type store
        # Getting the type of 'stypy_return_type' (line 947)
        stypy_return_type_113598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113598)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'hatch'
        return stypy_return_type_113598


    @norecursion
    def clip_to_bbox(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 967)
        True_113599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 40), 'True')
        defaults = [True_113599]
        # Create a new context for function 'clip_to_bbox'
        module_type_store = module_type_store.open_function_context('clip_to_bbox', 967, 4, False)
        # Assigning a type to the variable 'self' (line 968)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 968, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Path.clip_to_bbox.__dict__.__setitem__('stypy_localization', localization)
        Path.clip_to_bbox.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Path.clip_to_bbox.__dict__.__setitem__('stypy_type_store', module_type_store)
        Path.clip_to_bbox.__dict__.__setitem__('stypy_function_name', 'Path.clip_to_bbox')
        Path.clip_to_bbox.__dict__.__setitem__('stypy_param_names_list', ['bbox', 'inside'])
        Path.clip_to_bbox.__dict__.__setitem__('stypy_varargs_param_name', None)
        Path.clip_to_bbox.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Path.clip_to_bbox.__dict__.__setitem__('stypy_call_defaults', defaults)
        Path.clip_to_bbox.__dict__.__setitem__('stypy_call_varargs', varargs)
        Path.clip_to_bbox.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Path.clip_to_bbox.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Path.clip_to_bbox', ['bbox', 'inside'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clip_to_bbox', localization, ['bbox', 'inside'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clip_to_bbox(...)' code ##################

        unicode_113600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, (-1)), 'unicode', u'\n        Clip the path to the given bounding box.\n\n        The path must be made up of one or more closed polygons.  This\n        algorithm will not behave correctly for unclosed paths.\n\n        If *inside* is `True`, clip to the inside of the box, otherwise\n        to the outside of the box.\n        ')
        
        # Assigning a Call to a Name (line 978):
        
        # Assigning a Call to a Name (line 978):
        
        # Call to clip_path_to_rect(...): (line 978)
        # Processing the call arguments (line 978)
        # Getting the type of 'self' (line 978)
        self_113603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 40), 'self', False)
        # Getting the type of 'bbox' (line 978)
        bbox_113604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 46), 'bbox', False)
        # Getting the type of 'inside' (line 978)
        inside_113605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 52), 'inside', False)
        # Processing the call keyword arguments (line 978)
        kwargs_113606 = {}
        # Getting the type of '_path' (line 978)
        _path_113601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 16), '_path', False)
        # Obtaining the member 'clip_path_to_rect' of a type (line 978)
        clip_path_to_rect_113602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 978, 16), _path_113601, 'clip_path_to_rect')
        # Calling clip_path_to_rect(args, kwargs) (line 978)
        clip_path_to_rect_call_result_113607 = invoke(stypy.reporting.localization.Localization(__file__, 978, 16), clip_path_to_rect_113602, *[self_113603, bbox_113604, inside_113605], **kwargs_113606)
        
        # Assigning a type to the variable 'verts' (line 978)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 8), 'verts', clip_path_to_rect_call_result_113607)
        
        # Assigning a ListComp to a Name (line 979):
        
        # Assigning a ListComp to a Name (line 979):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'verts' (line 979)
        verts_113612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 40), 'verts')
        comprehension_113613 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 979, 17), verts_113612)
        # Assigning a type to the variable 'poly' (line 979)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 17), 'poly', comprehension_113613)
        
        # Call to Path(...): (line 979)
        # Processing the call arguments (line 979)
        # Getting the type of 'poly' (line 979)
        poly_113609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 22), 'poly', False)
        # Processing the call keyword arguments (line 979)
        kwargs_113610 = {}
        # Getting the type of 'Path' (line 979)
        Path_113608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 17), 'Path', False)
        # Calling Path(args, kwargs) (line 979)
        Path_call_result_113611 = invoke(stypy.reporting.localization.Localization(__file__, 979, 17), Path_113608, *[poly_113609], **kwargs_113610)
        
        list_113614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 979, 17), list_113614, Path_call_result_113611)
        # Assigning a type to the variable 'paths' (line 979)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 8), 'paths', list_113614)
        
        # Call to make_compound_path(...): (line 980)
        # Getting the type of 'paths' (line 980)
        paths_113617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 40), 'paths', False)
        # Processing the call keyword arguments (line 980)
        kwargs_113618 = {}
        # Getting the type of 'self' (line 980)
        self_113615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 15), 'self', False)
        # Obtaining the member 'make_compound_path' of a type (line 980)
        make_compound_path_113616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 980, 15), self_113615, 'make_compound_path')
        # Calling make_compound_path(args, kwargs) (line 980)
        make_compound_path_call_result_113619 = invoke(stypy.reporting.localization.Localization(__file__, 980, 15), make_compound_path_113616, *[paths_113617], **kwargs_113618)
        
        # Assigning a type to the variable 'stypy_return_type' (line 980)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 8), 'stypy_return_type', make_compound_path_call_result_113619)
        
        # ################# End of 'clip_to_bbox(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clip_to_bbox' in the type store
        # Getting the type of 'stypy_return_type' (line 967)
        stypy_return_type_113620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_113620)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clip_to_bbox'
        return stypy_return_type_113620


# Assigning a type to the variable 'Path' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'Path', Path)

# Assigning a Num to a Name (line 87):
int_113621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 11), 'int')
# Getting the type of 'Path'
Path_113622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member 'STOP' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113622, 'STOP', int_113621)

# Assigning a Num to a Name (line 88):
int_113623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'int')
# Getting the type of 'Path'
Path_113624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member 'MOVETO' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113624, 'MOVETO', int_113623)

# Assigning a Num to a Name (line 89):
int_113625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 13), 'int')
# Getting the type of 'Path'
Path_113626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member 'LINETO' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113626, 'LINETO', int_113625)

# Assigning a Num to a Name (line 90):
int_113627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 13), 'int')
# Getting the type of 'Path'
Path_113628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member 'CURVE3' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113628, 'CURVE3', int_113627)

# Assigning a Num to a Name (line 91):
int_113629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 13), 'int')
# Getting the type of 'Path'
Path_113630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member 'CURVE4' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113630, 'CURVE4', int_113629)

# Assigning a Num to a Name (line 92):
int_113631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 16), 'int')
# Getting the type of 'Path'
Path_113632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member 'CLOSEPOLY' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113632, 'CLOSEPOLY', int_113631)

# Assigning a Dict to a Name (line 96):

# Obtaining an instance of the builtin type 'dict' (line 96)
dict_113633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 28), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 96)
# Adding element type (key, value) (line 96)
# Getting the type of 'Path'
Path_113634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Obtaining the member 'STOP' of a type
STOP_113635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113634, 'STOP')
int_113636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 35), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), dict_113633, (STOP_113635, int_113636))
# Adding element type (key, value) (line 96)
# Getting the type of 'Path'
Path_113637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Obtaining the member 'MOVETO' of a type
MOVETO_113638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113637, 'MOVETO')
int_113639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 37), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), dict_113633, (MOVETO_113638, int_113639))
# Adding element type (key, value) (line 96)
# Getting the type of 'Path'
Path_113640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Obtaining the member 'LINETO' of a type
LINETO_113641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113640, 'LINETO')
int_113642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 37), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), dict_113633, (LINETO_113641, int_113642))
# Adding element type (key, value) (line 96)
# Getting the type of 'Path'
Path_113643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Obtaining the member 'CURVE3' of a type
CURVE3_113644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113643, 'CURVE3')
int_113645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 37), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), dict_113633, (CURVE3_113644, int_113645))
# Adding element type (key, value) (line 96)
# Getting the type of 'Path'
Path_113646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Obtaining the member 'CURVE4' of a type
CURVE4_113647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113646, 'CURVE4')
int_113648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 37), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), dict_113633, (CURVE4_113647, int_113648))
# Adding element type (key, value) (line 96)
# Getting the type of 'Path'
Path_113649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Obtaining the member 'CLOSEPOLY' of a type
CLOSEPOLY_113650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113649, 'CLOSEPOLY')
int_113651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 40), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), dict_113633, (CLOSEPOLY_113650, int_113651))

# Getting the type of 'Path'
Path_113652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member 'NUM_VERTICES_FOR_CODE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113652, 'NUM_VERTICES_FOR_CODE', dict_113633)

# Assigning a Attribute to a Name (line 103):
# Getting the type of 'np' (line 103)
np_113653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'np')
# Obtaining the member 'uint8' of a type (line 103)
uint8_113654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), np_113653, 'uint8')
# Getting the type of 'Path'
Path_113655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member 'code_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113655, 'code_type', uint8_113654)

# Assigning a Name to a Name (line 292):
# Getting the type of 'Path'
Path_113656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Obtaining the member '__copy__' of a type
copy___113657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113656, '__copy__')
# Getting the type of 'Path'
Path_113658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member 'copy' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113658, 'copy', copy___113657)

# Assigning a Name to a Name (line 307):
# Getting the type of 'Path'
Path_113659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Obtaining the member '__deepcopy__' of a type
deepcopy___113660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113659, '__deepcopy__')
# Getting the type of 'Path'
Path_113661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member 'deepcopy' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113661, 'deepcopy', deepcopy___113660)

# Assigning a Name to a Name (line 630):
# Getting the type of 'None' (line 630)
None_113662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 22), 'None')
# Getting the type of 'Path'
Path_113663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member '_unit_rectangle' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113663, '_unit_rectangle', None_113662)

# Assigning a Call to a Name (line 647):

# Call to WeakValueDictionary(...): (line 647)
# Processing the call keyword arguments (line 647)
kwargs_113665 = {}
# Getting the type of 'WeakValueDictionary' (line 647)
WeakValueDictionary_113664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 29), 'WeakValueDictionary', False)
# Calling WeakValueDictionary(args, kwargs) (line 647)
WeakValueDictionary_call_result_113666 = invoke(stypy.reporting.localization.Localization(__file__, 647, 29), WeakValueDictionary_113664, *[], **kwargs_113665)

# Getting the type of 'Path'
Path_113667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member '_unit_regular_polygons' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113667, '_unit_regular_polygons', WeakValueDictionary_call_result_113666)

# Assigning a Call to a Name (line 676):

# Call to WeakValueDictionary(...): (line 676)
# Processing the call keyword arguments (line 676)
kwargs_113669 = {}
# Getting the type of 'WeakValueDictionary' (line 676)
WeakValueDictionary_113668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 26), 'WeakValueDictionary', False)
# Calling WeakValueDictionary(args, kwargs) (line 676)
WeakValueDictionary_call_result_113670 = invoke(stypy.reporting.localization.Localization(__file__, 676, 26), WeakValueDictionary_113668, *[], **kwargs_113669)

# Getting the type of 'Path'
Path_113671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member '_unit_regular_stars' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113671, '_unit_regular_stars', WeakValueDictionary_call_result_113670)

# Assigning a Name to a Name (line 716):
# Getting the type of 'None' (line 716)
None_113672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 19), 'None')
# Getting the type of 'Path'
Path_113673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member '_unit_circle' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113673, '_unit_circle', None_113672)

# Assigning a Name to a Name (line 802):
# Getting the type of 'None' (line 802)
None_113674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 29), 'None')
# Getting the type of 'Path'
Path_113675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member '_unit_circle_righthalf' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113675, '_unit_circle_righthalf', None_113674)

# Assigning a Call to a Name (line 945):

# Call to maxdict(...): (line 945)
# Processing the call arguments (line 945)
int_113677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 26), 'int')
# Processing the call keyword arguments (line 945)
kwargs_113678 = {}
# Getting the type of 'maxdict' (line 945)
maxdict_113676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 18), 'maxdict', False)
# Calling maxdict(args, kwargs) (line 945)
maxdict_call_result_113679 = invoke(stypy.reporting.localization.Localization(__file__, 945, 18), maxdict_113676, *[int_113677], **kwargs_113678)

# Getting the type of 'Path'
Path_113680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Path')
# Setting the type of the member '_hatch_dict' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Path_113680, '_hatch_dict', maxdict_call_result_113679)

@norecursion
def get_path_collection_extents(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_path_collection_extents'
    module_type_store = module_type_store.open_function_context('get_path_collection_extents', 983, 0, False)
    
    # Passed parameters checking function
    get_path_collection_extents.stypy_localization = localization
    get_path_collection_extents.stypy_type_of_self = None
    get_path_collection_extents.stypy_type_store = module_type_store
    get_path_collection_extents.stypy_function_name = 'get_path_collection_extents'
    get_path_collection_extents.stypy_param_names_list = ['master_transform', 'paths', 'transforms', 'offsets', 'offset_transform']
    get_path_collection_extents.stypy_varargs_param_name = None
    get_path_collection_extents.stypy_kwargs_param_name = None
    get_path_collection_extents.stypy_call_defaults = defaults
    get_path_collection_extents.stypy_call_varargs = varargs
    get_path_collection_extents.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_path_collection_extents', ['master_transform', 'paths', 'transforms', 'offsets', 'offset_transform'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_path_collection_extents', localization, ['master_transform', 'paths', 'transforms', 'offsets', 'offset_transform'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_path_collection_extents(...)' code ##################

    unicode_113681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, (-1)), 'unicode', u'\n    Given a sequence of :class:`Path` objects,\n    :class:`~matplotlib.transforms.Transform` objects and offsets, as\n    found in a :class:`~matplotlib.collections.PathCollection`,\n    returns the bounding box that encapsulates all of them.\n\n    *master_transform* is a global transformation to apply to all paths\n\n    *paths* is a sequence of :class:`Path` instances.\n\n    *transforms* is a sequence of\n    :class:`~matplotlib.transforms.Affine2D` instances.\n\n    *offsets* is a sequence of (x, y) offsets (or an Nx2 array)\n\n    *offset_transform* is a :class:`~matplotlib.transforms.Affine2D`\n    to apply to the offsets before applying the offset to the path.\n\n    The way that *paths*, *transforms* and *offsets* are combined\n    follows the same method as for collections.  Each is iterated over\n    independently, so if you have 3 paths, 2 transforms and 1 offset,\n    their combinations are as follows:\n\n        (A, A, A), (B, B, A), (C, A, A)\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1010, 4))
    
    # 'from matplotlib.transforms import Bbox' statement (line 1010)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
    import_113682 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1010, 4), 'matplotlib.transforms')

    if (type(import_113682) is not StypyTypeError):

        if (import_113682 != 'pyd_module'):
            __import__(import_113682)
            sys_modules_113683 = sys.modules[import_113682]
            import_from_module(stypy.reporting.localization.Localization(__file__, 1010, 4), 'matplotlib.transforms', sys_modules_113683.module_type_store, module_type_store, ['Bbox'])
            nest_module(stypy.reporting.localization.Localization(__file__, 1010, 4), __file__, sys_modules_113683, sys_modules_113683.module_type_store, module_type_store)
        else:
            from matplotlib.transforms import Bbox

            import_from_module(stypy.reporting.localization.Localization(__file__, 1010, 4), 'matplotlib.transforms', None, module_type_store, ['Bbox'], [Bbox])

    else:
        # Assigning a type to the variable 'matplotlib.transforms' (line 1010)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1010, 4), 'matplotlib.transforms', import_113682)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
    
    
    
    
    # Call to len(...): (line 1011)
    # Processing the call arguments (line 1011)
    # Getting the type of 'paths' (line 1011)
    paths_113685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 11), 'paths', False)
    # Processing the call keyword arguments (line 1011)
    kwargs_113686 = {}
    # Getting the type of 'len' (line 1011)
    len_113684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 7), 'len', False)
    # Calling len(args, kwargs) (line 1011)
    len_call_result_113687 = invoke(stypy.reporting.localization.Localization(__file__, 1011, 7), len_113684, *[paths_113685], **kwargs_113686)
    
    int_113688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 21), 'int')
    # Applying the binary operator '==' (line 1011)
    result_eq_113689 = python_operator(stypy.reporting.localization.Localization(__file__, 1011, 7), '==', len_call_result_113687, int_113688)
    
    # Testing the type of an if condition (line 1011)
    if_condition_113690 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1011, 4), result_eq_113689)
    # Assigning a type to the variable 'if_condition_113690' (line 1011)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1011, 4), 'if_condition_113690', if_condition_113690)
    # SSA begins for if statement (line 1011)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1012)
    # Processing the call arguments (line 1012)
    unicode_113692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 25), 'unicode', u'No paths provided')
    # Processing the call keyword arguments (line 1012)
    kwargs_113693 = {}
    # Getting the type of 'ValueError' (line 1012)
    ValueError_113691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1012)
    ValueError_call_result_113694 = invoke(stypy.reporting.localization.Localization(__file__, 1012, 14), ValueError_113691, *[unicode_113692], **kwargs_113693)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1012, 8), ValueError_call_result_113694, 'raise parameter', BaseException)
    # SSA join for if statement (line 1011)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to from_extents(...): (line 1013)
    
    # Call to get_path_collection_extents(...): (line 1013)
    # Processing the call arguments (line 1013)
    # Getting the type of 'master_transform' (line 1014)
    master_transform_113699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 8), 'master_transform', False)
    # Getting the type of 'paths' (line 1014)
    paths_113700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 26), 'paths', False)
    
    # Call to atleast_3d(...): (line 1014)
    # Processing the call arguments (line 1014)
    # Getting the type of 'transforms' (line 1014)
    transforms_113703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 47), 'transforms', False)
    # Processing the call keyword arguments (line 1014)
    kwargs_113704 = {}
    # Getting the type of 'np' (line 1014)
    np_113701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 33), 'np', False)
    # Obtaining the member 'atleast_3d' of a type (line 1014)
    atleast_3d_113702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1014, 33), np_113701, 'atleast_3d')
    # Calling atleast_3d(args, kwargs) (line 1014)
    atleast_3d_call_result_113705 = invoke(stypy.reporting.localization.Localization(__file__, 1014, 33), atleast_3d_113702, *[transforms_113703], **kwargs_113704)
    
    # Getting the type of 'offsets' (line 1015)
    offsets_113706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 8), 'offsets', False)
    # Getting the type of 'offset_transform' (line 1015)
    offset_transform_113707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 17), 'offset_transform', False)
    # Processing the call keyword arguments (line 1013)
    kwargs_113708 = {}
    # Getting the type of '_path' (line 1013)
    _path_113697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 30), '_path', False)
    # Obtaining the member 'get_path_collection_extents' of a type (line 1013)
    get_path_collection_extents_113698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1013, 30), _path_113697, 'get_path_collection_extents')
    # Calling get_path_collection_extents(args, kwargs) (line 1013)
    get_path_collection_extents_call_result_113709 = invoke(stypy.reporting.localization.Localization(__file__, 1013, 30), get_path_collection_extents_113698, *[master_transform_113699, paths_113700, atleast_3d_call_result_113705, offsets_113706, offset_transform_113707], **kwargs_113708)
    
    # Processing the call keyword arguments (line 1013)
    kwargs_113710 = {}
    # Getting the type of 'Bbox' (line 1013)
    Bbox_113695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 11), 'Bbox', False)
    # Obtaining the member 'from_extents' of a type (line 1013)
    from_extents_113696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1013, 11), Bbox_113695, 'from_extents')
    # Calling from_extents(args, kwargs) (line 1013)
    from_extents_call_result_113711 = invoke(stypy.reporting.localization.Localization(__file__, 1013, 11), from_extents_113696, *[get_path_collection_extents_call_result_113709], **kwargs_113710)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1013)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1013, 4), 'stypy_return_type', from_extents_call_result_113711)
    
    # ################# End of 'get_path_collection_extents(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_path_collection_extents' in the type store
    # Getting the type of 'stypy_return_type' (line 983)
    stypy_return_type_113712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_113712)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_path_collection_extents'
    return stypy_return_type_113712

# Assigning a type to the variable 'get_path_collection_extents' (line 983)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 983, 0), 'get_path_collection_extents', get_path_collection_extents)

@norecursion
def get_paths_extents(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'list' (line 1018)
    list_113713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1018)
    
    defaults = [list_113713]
    # Create a new context for function 'get_paths_extents'
    module_type_store = module_type_store.open_function_context('get_paths_extents', 1018, 0, False)
    
    # Passed parameters checking function
    get_paths_extents.stypy_localization = localization
    get_paths_extents.stypy_type_of_self = None
    get_paths_extents.stypy_type_store = module_type_store
    get_paths_extents.stypy_function_name = 'get_paths_extents'
    get_paths_extents.stypy_param_names_list = ['paths', 'transforms']
    get_paths_extents.stypy_varargs_param_name = None
    get_paths_extents.stypy_kwargs_param_name = None
    get_paths_extents.stypy_call_defaults = defaults
    get_paths_extents.stypy_call_varargs = varargs
    get_paths_extents.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_paths_extents', ['paths', 'transforms'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_paths_extents', localization, ['paths', 'transforms'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_paths_extents(...)' code ##################

    unicode_113714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, (-1)), 'unicode', u'\n    Given a sequence of :class:`Path` objects and optional\n    :class:`~matplotlib.transforms.Transform` objects, returns the\n    bounding box that encapsulates all of them.\n\n    *paths* is a sequence of :class:`Path` instances.\n\n    *transforms* is an optional sequence of\n    :class:`~matplotlib.transforms.Affine2D` instances to apply to\n    each path.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1030, 4))
    
    # 'from matplotlib.transforms import Bbox, Affine2D' statement (line 1030)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
    import_113715 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1030, 4), 'matplotlib.transforms')

    if (type(import_113715) is not StypyTypeError):

        if (import_113715 != 'pyd_module'):
            __import__(import_113715)
            sys_modules_113716 = sys.modules[import_113715]
            import_from_module(stypy.reporting.localization.Localization(__file__, 1030, 4), 'matplotlib.transforms', sys_modules_113716.module_type_store, module_type_store, ['Bbox', 'Affine2D'])
            nest_module(stypy.reporting.localization.Localization(__file__, 1030, 4), __file__, sys_modules_113716, sys_modules_113716.module_type_store, module_type_store)
        else:
            from matplotlib.transforms import Bbox, Affine2D

            import_from_module(stypy.reporting.localization.Localization(__file__, 1030, 4), 'matplotlib.transforms', None, module_type_store, ['Bbox', 'Affine2D'], [Bbox, Affine2D])

    else:
        # Assigning a type to the variable 'matplotlib.transforms' (line 1030)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 4), 'matplotlib.transforms', import_113715)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
    
    
    
    
    # Call to len(...): (line 1031)
    # Processing the call arguments (line 1031)
    # Getting the type of 'paths' (line 1031)
    paths_113718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 11), 'paths', False)
    # Processing the call keyword arguments (line 1031)
    kwargs_113719 = {}
    # Getting the type of 'len' (line 1031)
    len_113717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 7), 'len', False)
    # Calling len(args, kwargs) (line 1031)
    len_call_result_113720 = invoke(stypy.reporting.localization.Localization(__file__, 1031, 7), len_113717, *[paths_113718], **kwargs_113719)
    
    int_113721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 21), 'int')
    # Applying the binary operator '==' (line 1031)
    result_eq_113722 = python_operator(stypy.reporting.localization.Localization(__file__, 1031, 7), '==', len_call_result_113720, int_113721)
    
    # Testing the type of an if condition (line 1031)
    if_condition_113723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1031, 4), result_eq_113722)
    # Assigning a type to the variable 'if_condition_113723' (line 1031)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 4), 'if_condition_113723', if_condition_113723)
    # SSA begins for if statement (line 1031)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1032)
    # Processing the call arguments (line 1032)
    unicode_113725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 25), 'unicode', u'No paths provided')
    # Processing the call keyword arguments (line 1032)
    kwargs_113726 = {}
    # Getting the type of 'ValueError' (line 1032)
    ValueError_113724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1032)
    ValueError_call_result_113727 = invoke(stypy.reporting.localization.Localization(__file__, 1032, 14), ValueError_113724, *[unicode_113725], **kwargs_113726)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1032, 8), ValueError_call_result_113727, 'raise parameter', BaseException)
    # SSA join for if statement (line 1031)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to from_extents(...): (line 1033)
    
    # Call to get_path_collection_extents(...): (line 1033)
    # Processing the call arguments (line 1033)
    
    # Call to Affine2D(...): (line 1034)
    # Processing the call keyword arguments (line 1034)
    kwargs_113733 = {}
    # Getting the type of 'Affine2D' (line 1034)
    Affine2D_113732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 8), 'Affine2D', False)
    # Calling Affine2D(args, kwargs) (line 1034)
    Affine2D_call_result_113734 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 8), Affine2D_113732, *[], **kwargs_113733)
    
    # Getting the type of 'paths' (line 1034)
    paths_113735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 20), 'paths', False)
    # Getting the type of 'transforms' (line 1034)
    transforms_113736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 27), 'transforms', False)
    
    # Obtaining an instance of the builtin type 'list' (line 1034)
    list_113737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1034)
    
    
    # Call to Affine2D(...): (line 1034)
    # Processing the call keyword arguments (line 1034)
    kwargs_113739 = {}
    # Getting the type of 'Affine2D' (line 1034)
    Affine2D_113738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 43), 'Affine2D', False)
    # Calling Affine2D(args, kwargs) (line 1034)
    Affine2D_call_result_113740 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 43), Affine2D_113738, *[], **kwargs_113739)
    
    # Processing the call keyword arguments (line 1033)
    kwargs_113741 = {}
    # Getting the type of '_path' (line 1033)
    _path_113730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 30), '_path', False)
    # Obtaining the member 'get_path_collection_extents' of a type (line 1033)
    get_path_collection_extents_113731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 30), _path_113730, 'get_path_collection_extents')
    # Calling get_path_collection_extents(args, kwargs) (line 1033)
    get_path_collection_extents_call_result_113742 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 30), get_path_collection_extents_113731, *[Affine2D_call_result_113734, paths_113735, transforms_113736, list_113737, Affine2D_call_result_113740], **kwargs_113741)
    
    # Processing the call keyword arguments (line 1033)
    kwargs_113743 = {}
    # Getting the type of 'Bbox' (line 1033)
    Bbox_113728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 11), 'Bbox', False)
    # Obtaining the member 'from_extents' of a type (line 1033)
    from_extents_113729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 11), Bbox_113728, 'from_extents')
    # Calling from_extents(args, kwargs) (line 1033)
    from_extents_call_result_113744 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 11), from_extents_113729, *[get_path_collection_extents_call_result_113742], **kwargs_113743)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 4), 'stypy_return_type', from_extents_call_result_113744)
    
    # ################# End of 'get_paths_extents(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_paths_extents' in the type store
    # Getting the type of 'stypy_return_type' (line 1018)
    stypy_return_type_113745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_113745)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_paths_extents'
    return stypy_return_type_113745

# Assigning a type to the variable 'get_paths_extents' (line 1018)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1018, 0), 'get_paths_extents', get_paths_extents)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
