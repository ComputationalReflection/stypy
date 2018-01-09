
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Support for plotting vector fields.
3: 
4: Presently this contains Quiver and Barb. Quiver plots an arrow in the
5: direction of the vector, with the size of the arrow related to the
6: magnitude of the vector.
7: 
8: Barbs are like quiver in that they point along a vector, but
9: the magnitude of the vector is given schematically by the presence of barbs
10: or flags on the barb.
11: 
12: This will also become a home for things such as standard
13: deviation ellipses, which can and will be derived very easily from
14: the Quiver code.
15: '''
16: 
17: from __future__ import (absolute_import, division, print_function,
18:                         unicode_literals)
19: 
20: import six
21: import weakref
22: 
23: import numpy as np
24: from numpy import ma
25: import matplotlib.collections as mcollections
26: import matplotlib.transforms as transforms
27: import matplotlib.text as mtext
28: import matplotlib.artist as martist
29: from matplotlib.artist import allow_rasterization
30: from matplotlib import docstring
31: import matplotlib.font_manager as font_manager
32: import matplotlib.cbook as cbook
33: from matplotlib.cbook import delete_masked_points
34: from matplotlib.patches import CirclePolygon
35: import math
36: 
37: 
38: _quiver_doc = '''
39: Plot a 2-D field of arrows.
40: 
41: Call signatures::
42: 
43:   quiver(U, V, **kw)
44:   quiver(U, V, C, **kw)
45:   quiver(X, Y, U, V, **kw)
46:   quiver(X, Y, U, V, C, **kw)
47: 
48: *U* and *V* are the arrow data, *X* and *Y* set the location of the
49: arrows, and *C* sets the color of the arrows. These arguments may be 1-D or
50: 2-D arrays or sequences.
51: 
52: If *X* and *Y* are absent, they will be generated as a uniform grid.
53: If *U* and *V* are 2-D arrays and *X* and *Y* are 1-D, and if ``len(X)`` and
54: ``len(Y)`` match the column and row dimensions of *U*, then *X* and *Y* will be
55: expanded with :func:`numpy.meshgrid`.
56: 
57: The default settings auto-scales the length of the arrows to a reasonable size.
58: To change this behavior see the *scale* and *scale_units* kwargs.
59: 
60: The defaults give a slightly swept-back arrow; to make the head a
61: triangle, make *headaxislength* the same as *headlength*. To make the
62: arrow more pointed, reduce *headwidth* or increase *headlength* and
63: *headaxislength*. To make the head smaller relative to the shaft,
64: scale down all the head parameters. You will probably do best to leave
65: minshaft alone.
66: 
67: *linewidths* and *edgecolors* can be used to customize the arrow
68: outlines.
69: 
70: Parameters
71: ----------
72: X : 1D or 2D array, sequence, optional
73:     The x coordinates of the arrow locations
74: Y : 1D or 2D array, sequence, optional
75:     The y coordinates of the arrow locations
76: U : 1D or 2D array or masked array, sequence
77:     The x components of the arrow vectors
78: V : 1D or 2D array or masked array, sequence
79:     The y components of the arrow vectors
80: C : 1D or 2D array, sequence, optional
81:     The arrow colors
82: units : [ 'width' | 'height' | 'dots' | 'inches' | 'x' | 'y' | 'xy' ]
83:     The arrow dimensions (except for *length*) are measured in multiples of
84:     this unit.
85: 
86:     'width' or 'height': the width or height of the axis
87: 
88:     'dots' or 'inches': pixels or inches, based on the figure dpi
89: 
90:     'x', 'y', or 'xy': respectively *X*, *Y*, or :math:`\\sqrt{X^2 + Y^2}`
91:     in data units
92: 
93:     The arrows scale differently depending on the units.  For
94:     'x' or 'y', the arrows get larger as one zooms in; for other
95:     units, the arrow size is independent of the zoom state.  For
96:     'width or 'height', the arrow size increases with the width and
97:     height of the axes, respectively, when the window is resized;
98:     for 'dots' or 'inches', resizing does not change the arrows.
99: angles : [ 'uv' | 'xy' ], array, optional
100:     Method for determining the angle of the arrows. Default is 'uv'.
101: 
102:     'uv': the arrow axis aspect ratio is 1 so that
103:     if *U*==*V* the orientation of the arrow on the plot is 45 degrees
104:     counter-clockwise from the horizontal axis (positive to the right).
105: 
106:     'xy': arrows point from (x,y) to (x+u, y+v).
107:     Use this for plotting a gradient field, for example.
108: 
109:     Alternatively, arbitrary angles may be specified as an array
110:     of values in degrees, counter-clockwise from the horizontal axis.
111: 
112:     Note: inverting a data axis will correspondingly invert the
113:     arrows only with ``angles='xy'``.
114: scale : None, float, optional
115:     Number of data units per arrow length unit, e.g., m/s per plot width; a
116:     smaller scale parameter makes the arrow longer. Default is *None*.
117: 
118:     If *None*, a simple autoscaling algorithm is used, based on the average
119:     vector length and the number of vectors. The arrow length unit is given by
120:     the *scale_units* parameter
121: scale_units : [ 'width' | 'height' | 'dots' | 'inches' | 'x' | 'y' | 'xy' ], \
122: None, optional
123:     If the *scale* kwarg is *None*, the arrow length unit. Default is *None*.
124: 
125:     e.g. *scale_units* is 'inches', *scale* is 2.0, and
126:     ``(u,v) = (1,0)``, then the vector will be 0.5 inches long.
127: 
128:     If *scale_units* is 'width'/'height', then the vector will be half the
129:     width/height of the axes.
130: 
131:     If *scale_units* is 'x' then the vector will be 0.5 x-axis
132:     units. To plot vectors in the x-y plane, with u and v having
133:     the same units as x and y, use
134:     ``angles='xy', scale_units='xy', scale=1``.
135: width : scalar, optional
136:     Shaft width in arrow units; default depends on choice of units,
137:     above, and number of vectors; a typical starting value is about
138:     0.005 times the width of the plot.
139: headwidth : scalar, optional
140:     Head width as multiple of shaft width, default is 3
141: headlength : scalar, optional
142:     Head length as multiple of shaft width, default is 5
143: headaxislength : scalar, optional
144:     Head length at shaft intersection, default is 4.5
145: minshaft : scalar, optional
146:     Length below which arrow scales, in units of head length. Do not
147:     set this to less than 1, or small arrows will look terrible!
148:     Default is 1
149: minlength : scalar, optional
150:     Minimum length as a multiple of shaft width; if an arrow length
151:     is less than this, plot a dot (hexagon) of this diameter instead.
152:     Default is 1.
153: pivot : [ 'tail' | 'mid' | 'middle' | 'tip' ], optional
154:     The part of the arrow that is at the grid point; the arrow rotates
155:     about this point, hence the name *pivot*.
156: color : [ color | color sequence ], optional
157:     This is a synonym for the
158:     :class:`~matplotlib.collections.PolyCollection` facecolor kwarg.
159:     If *C* has been set, *color* has no effect.
160: 
161: Notes
162: -----
163: Additional :class:`~matplotlib.collections.PolyCollection`
164: keyword arguments:
165: 
166: %(PolyCollection)s
167: 
168: See Also
169: --------
170: quiverkey : Add a key to a quiver plot
171: ''' % docstring.interpd.params
172: 
173: _quiverkey_doc = '''
174: Add a key to a quiver plot.
175: 
176: Call signature::
177: 
178:   quiverkey(Q, X, Y, U, label, **kw)
179: 
180: Arguments:
181: 
182:   *Q*:
183:     The Quiver instance returned by a call to quiver.
184: 
185:   *X*, *Y*:
186:     The location of the key; additional explanation follows.
187: 
188:   *U*:
189:     The length of the key
190: 
191:   *label*:
192:     A string with the length and units of the key
193: 
194: Keyword arguments:
195: 
196:   *angle* = 0
197:     The angle of the key arrow. Measured in degrees anti-clockwise from the
198:     x-axis.
199: 
200:   *coordinates* = [ 'axes' | 'figure' | 'data' | 'inches' ]
201:     Coordinate system and units for *X*, *Y*: 'axes' and 'figure' are
202:     normalized coordinate systems with 0,0 in the lower left and 1,1
203:     in the upper right; 'data' are the axes data coordinates (used for
204:     the locations of the vectors in the quiver plot itself); 'inches'
205:     is position in the figure in inches, with 0,0 at the lower left
206:     corner.
207: 
208:   *color*:
209:     overrides face and edge colors from *Q*.
210: 
211:   *labelpos* = [ 'N' | 'S' | 'E' | 'W' ]
212:     Position the label above, below, to the right, to the left of the
213:     arrow, respectively.
214: 
215:   *labelsep*:
216:     Distance in inches between the arrow and the label.  Default is
217:     0.1
218: 
219:   *labelcolor*:
220:     defaults to default :class:`~matplotlib.text.Text` color.
221: 
222:   *fontproperties*:
223:     A dictionary with keyword arguments accepted by the
224:     :class:`~matplotlib.font_manager.FontProperties` initializer:
225:     *family*, *style*, *variant*, *size*, *weight*
226: 
227: Any additional keyword arguments are used to override vector
228: properties taken from *Q*.
229: 
230: The positioning of the key depends on *X*, *Y*, *coordinates*, and
231: *labelpos*.  If *labelpos* is 'N' or 'S', *X*, *Y* give the position
232: of the middle of the key arrow.  If *labelpos* is 'E', *X*, *Y*
233: positions the head, and if *labelpos* is 'W', *X*, *Y* positions the
234: tail; in either of these two cases, *X*, *Y* is somewhere in the
235: middle of the arrow+label key object.
236: '''
237: 
238: 
239: class QuiverKey(martist.Artist):
240:     ''' Labelled arrow for use as a quiver plot scale key.'''
241:     halign = {'N': 'center', 'S': 'center', 'E': 'left', 'W': 'right'}
242:     valign = {'N': 'bottom', 'S': 'top', 'E': 'center', 'W': 'center'}
243:     pivot = {'N': 'middle', 'S': 'middle', 'E': 'tip', 'W': 'tail'}
244: 
245:     def __init__(self, Q, X, Y, U, label, **kw):
246:         martist.Artist.__init__(self)
247:         self.Q = Q
248:         self.X = X
249:         self.Y = Y
250:         self.U = U
251:         self.angle = kw.pop('angle', 0)
252:         self.coord = kw.pop('coordinates', 'axes')
253:         self.color = kw.pop('color', None)
254:         self.label = label
255:         self._labelsep_inches = kw.pop('labelsep', 0.1)
256:         self.labelsep = (self._labelsep_inches * Q.ax.figure.dpi)
257: 
258:         # try to prevent closure over the real self
259:         weak_self = weakref.ref(self)
260: 
261:         def on_dpi_change(fig):
262:             self_weakref = weak_self()
263:             if self_weakref is not None:
264:                 self_weakref.labelsep = (self_weakref._labelsep_inches*fig.dpi)
265:                 self_weakref._initialized = False  # simple brute force update
266:                                                    # works because _init is
267:                                                    # called at the start of
268:                                                    # draw.
269: 
270:         self._cid = Q.ax.figure.callbacks.connect('dpi_changed',
271:                                                   on_dpi_change)
272: 
273:         self.labelpos = kw.pop('labelpos', 'N')
274:         self.labelcolor = kw.pop('labelcolor', None)
275:         self.fontproperties = kw.pop('fontproperties', dict())
276:         self.kw = kw
277:         _fp = self.fontproperties
278:         # boxprops = dict(facecolor='red')
279:         self.text = mtext.Text(
280:                         text=label,  # bbox=boxprops,
281:                         horizontalalignment=self.halign[self.labelpos],
282:                         verticalalignment=self.valign[self.labelpos],
283:                         fontproperties=font_manager.FontProperties(**_fp))
284: 
285:         if self.labelcolor is not None:
286:             self.text.set_color(self.labelcolor)
287:         self._initialized = False
288:         self.zorder = Q.zorder + 0.1
289: 
290:     def remove(self):
291:         '''
292:         Overload the remove method
293:         '''
294:         self.Q.ax.figure.callbacks.disconnect(self._cid)
295:         self._cid = None
296:         # pass the remove call up the stack
297:         martist.Artist.remove(self)
298: 
299:     __init__.__doc__ = _quiverkey_doc
300: 
301:     def _init(self):
302:         if True:  # not self._initialized:
303:             if not self.Q._initialized:
304:                 self.Q._init()
305:             self._set_transform()
306:             _pivot = self.Q.pivot
307:             self.Q.pivot = self.pivot[self.labelpos]
308:             # Hack: save and restore the Umask
309:             _mask = self.Q.Umask
310:             self.Q.Umask = ma.nomask
311:             self.verts = self.Q._make_verts(np.array([self.U]),
312:                                             np.zeros((1,)),
313:                                             self.angle)
314:             self.Q.Umask = _mask
315:             self.Q.pivot = _pivot
316:             kw = self.Q.polykw
317:             kw.update(self.kw)
318:             self.vector = mcollections.PolyCollection(
319:                                         self.verts,
320:                                         offsets=[(self.X, self.Y)],
321:                                         transOffset=self.get_transform(),
322:                                         **kw)
323:             if self.color is not None:
324:                 self.vector.set_color(self.color)
325:             self.vector.set_transform(self.Q.get_transform())
326:             self.vector.set_figure(self.get_figure())
327:             self._initialized = True
328: 
329:     def _text_x(self, x):
330:         if self.labelpos == 'E':
331:             return x + self.labelsep
332:         elif self.labelpos == 'W':
333:             return x - self.labelsep
334:         else:
335:             return x
336: 
337:     def _text_y(self, y):
338:         if self.labelpos == 'N':
339:             return y + self.labelsep
340:         elif self.labelpos == 'S':
341:             return y - self.labelsep
342:         else:
343:             return y
344: 
345:     @allow_rasterization
346:     def draw(self, renderer):
347:         self._init()
348:         self.vector.draw(renderer)
349:         x, y = self.get_transform().transform_point((self.X, self.Y))
350:         self.text.set_x(self._text_x(x))
351:         self.text.set_y(self._text_y(y))
352:         self.text.draw(renderer)
353:         self.stale = False
354: 
355:     def _set_transform(self):
356:         if self.coord == 'data':
357:             self.set_transform(self.Q.ax.transData)
358:         elif self.coord == 'axes':
359:             self.set_transform(self.Q.ax.transAxes)
360:         elif self.coord == 'figure':
361:             self.set_transform(self.Q.ax.figure.transFigure)
362:         elif self.coord == 'inches':
363:             self.set_transform(self.Q.ax.figure.dpi_scale_trans)
364:         else:
365:             raise ValueError('unrecognized coordinates')
366: 
367:     def set_figure(self, fig):
368:         martist.Artist.set_figure(self, fig)
369:         self.text.set_figure(fig)
370: 
371:     def contains(self, mouseevent):
372:         # Maybe the dictionary should allow one to
373:         # distinguish between a text hit and a vector hit.
374:         if (self.text.contains(mouseevent)[0] or
375:                 self.vector.contains(mouseevent)[0]):
376:             return True, {}
377:         return False, {}
378: 
379:     quiverkey_doc = _quiverkey_doc
380: 
381: 
382: # This is a helper function that parses out the various combination of
383: # arguments for doing colored vector plots.  Pulling it out here
384: # allows both Quiver and Barbs to use it
385: def _parse_args(*args):
386:     X, Y, U, V, C = [None] * 5
387:     args = list(args)
388: 
389:     # The use of atleast_1d allows for handling scalar arguments while also
390:     # keeping masked arrays
391:     if len(args) == 3 or len(args) == 5:
392:         C = np.atleast_1d(args.pop(-1))
393:     V = np.atleast_1d(args.pop(-1))
394:     U = np.atleast_1d(args.pop(-1))
395:     if U.ndim == 1:
396:         nr, nc = 1, U.shape[0]
397:     else:
398:         nr, nc = U.shape
399:     if len(args) == 2:  # remaining after removing U,V,C
400:         X, Y = [np.array(a).ravel() for a in args]
401:         if len(X) == nc and len(Y) == nr:
402:             X, Y = [a.ravel() for a in np.meshgrid(X, Y)]
403:     else:
404:         indexgrid = np.meshgrid(np.arange(nc), np.arange(nr))
405:         X, Y = [np.ravel(a) for a in indexgrid]
406:     return X, Y, U, V, C
407: 
408: 
409: def _check_consistent_shapes(*arrays):
410:     all_shapes = set(a.shape for a in arrays)
411:     if len(all_shapes) != 1:
412:         raise ValueError('The shapes of the passed in arrays do not match.')
413: 
414: 
415: class Quiver(mcollections.PolyCollection):
416:     '''
417:     Specialized PolyCollection for arrows.
418: 
419:     The only API method is set_UVC(), which can be used
420:     to change the size, orientation, and color of the
421:     arrows; their locations are fixed when the class is
422:     instantiated.  Possibly this method will be useful
423:     in animations.
424: 
425:     Much of the work in this class is done in the draw()
426:     method so that as much information as possible is available
427:     about the plot.  In subsequent draw() calls, recalculation
428:     is limited to things that might have changed, so there
429:     should be no performance penalty from putting the calculations
430:     in the draw() method.
431:     '''
432: 
433:     _PIVOT_VALS = ('tail', 'mid', 'middle', 'tip')
434: 
435:     @docstring.Substitution(_quiver_doc)
436:     def __init__(self, ax, *args, **kw):
437:         '''
438:         The constructor takes one required argument, an Axes
439:         instance, followed by the args and kwargs described
440:         by the following pylab interface documentation:
441:         %s
442:         '''
443:         self.ax = ax
444:         X, Y, U, V, C = _parse_args(*args)
445:         self.X = X
446:         self.Y = Y
447:         self.XY = np.hstack((X[:, np.newaxis], Y[:, np.newaxis]))
448:         self.N = len(X)
449:         self.scale = kw.pop('scale', None)
450:         self.headwidth = kw.pop('headwidth', 3)
451:         self.headlength = float(kw.pop('headlength', 5))
452:         self.headaxislength = kw.pop('headaxislength', 4.5)
453:         self.minshaft = kw.pop('minshaft', 1)
454:         self.minlength = kw.pop('minlength', 1)
455:         self.units = kw.pop('units', 'width')
456:         self.scale_units = kw.pop('scale_units', None)
457:         self.angles = kw.pop('angles', 'uv')
458:         self.width = kw.pop('width', None)
459:         self.color = kw.pop('color', 'k')
460: 
461:         pivot = kw.pop('pivot', 'tail').lower()
462:         # validate pivot
463:         if pivot not in self._PIVOT_VALS:
464:             raise ValueError(
465:                 'pivot must be one of {keys}, you passed {inp}'.format(
466:                       keys=self._PIVOT_VALS, inp=pivot))
467:         # normalize to 'middle'
468:         if pivot == 'mid':
469:             pivot = 'middle'
470:         self.pivot = pivot
471: 
472:         self.transform = kw.pop('transform', ax.transData)
473:         kw.setdefault('facecolors', self.color)
474:         kw.setdefault('linewidths', (0,))
475:         mcollections.PolyCollection.__init__(self, [], offsets=self.XY,
476:                                              transOffset=self.transform,
477:                                              closed=False,
478:                                              **kw)
479:         self.polykw = kw
480:         self.set_UVC(U, V, C)
481:         self._initialized = False
482: 
483:         self.keyvec = None
484:         self.keytext = None
485: 
486:         # try to prevent closure over the real self
487:         weak_self = weakref.ref(self)
488: 
489:         def on_dpi_change(fig):
490:             self_weakref = weak_self()
491:             if self_weakref is not None:
492:                 self_weakref._new_UV = True  # vertices depend on width, span
493:                                              # which in turn depend on dpi
494:                 self_weakref._initialized = False  # simple brute force update
495:                                                    # works because _init is
496:                                                    # called at the start of
497:                                                    # draw.
498: 
499:         self._cid = self.ax.figure.callbacks.connect('dpi_changed',
500:                                                      on_dpi_change)
501: 
502:     def remove(self):
503:         '''
504:         Overload the remove method
505:         '''
506:         # disconnect the call back
507:         self.ax.figure.callbacks.disconnect(self._cid)
508:         self._cid = None
509:         # pass the remove call up the stack
510:         mcollections.PolyCollection.remove(self)
511: 
512:     def _init(self):
513:         '''
514:         Initialization delayed until first draw;
515:         allow time for axes setup.
516:         '''
517:         # It seems that there are not enough event notifications
518:         # available to have this work on an as-needed basis at present.
519:         if True:  # not self._initialized:
520:             trans = self._set_transform()
521:             ax = self.ax
522:             sx, sy = trans.inverted().transform_point(
523:                                             (ax.bbox.width, ax.bbox.height))
524:             self.span = sx
525:             if self.width is None:
526:                 sn = np.clip(math.sqrt(self.N), 8, 25)
527:                 self.width = 0.06 * self.span / sn
528: 
529:             # _make_verts sets self.scale if not already specified
530:             if not self._initialized and self.scale is None:
531:                 self._make_verts(self.U, self.V, self.angles)
532: 
533:             self._initialized = True
534: 
535:     def get_datalim(self, transData):
536:         trans = self.get_transform()
537:         transOffset = self.get_offset_transform()
538:         full_transform = (trans - transData) + (transOffset - transData)
539:         XY = full_transform.transform(self.XY)
540:         bbox = transforms.Bbox.null()
541:         bbox.update_from_data_xy(XY, ignore=True)
542:         return bbox
543: 
544:     @allow_rasterization
545:     def draw(self, renderer):
546:         self._init()
547:         verts = self._make_verts(self.U, self.V, self.angles)
548:         self.set_verts(verts, closed=False)
549:         self._new_UV = False
550:         mcollections.PolyCollection.draw(self, renderer)
551:         self.stale = False
552: 
553:     def set_UVC(self, U, V, C=None):
554:         # We need to ensure we have a copy, not a reference
555:         # to an array that might change before draw().
556:         U = ma.masked_invalid(U, copy=True).ravel()
557:         V = ma.masked_invalid(V, copy=True).ravel()
558:         mask = ma.mask_or(U.mask, V.mask, copy=False, shrink=True)
559:         if C is not None:
560:             C = ma.masked_invalid(C, copy=True).ravel()
561:             mask = ma.mask_or(mask, C.mask, copy=False, shrink=True)
562:             if mask is ma.nomask:
563:                 C = C.filled()
564:             else:
565:                 C = ma.array(C, mask=mask, copy=False)
566:         self.U = U.filled(1)
567:         self.V = V.filled(1)
568:         self.Umask = mask
569:         if C is not None:
570:             self.set_array(C)
571:         self._new_UV = True
572:         self.stale = True
573: 
574:     def _dots_per_unit(self, units):
575:         '''
576:         Return a scale factor for converting from units to pixels
577:         '''
578:         ax = self.ax
579:         if units in ('x', 'y', 'xy'):
580:             if units == 'x':
581:                 dx0 = ax.viewLim.width
582:                 dx1 = ax.bbox.width
583:             elif units == 'y':
584:                 dx0 = ax.viewLim.height
585:                 dx1 = ax.bbox.height
586:             else:  # 'xy' is assumed
587:                 dxx0 = ax.viewLim.width
588:                 dxx1 = ax.bbox.width
589:                 dyy0 = ax.viewLim.height
590:                 dyy1 = ax.bbox.height
591:                 dx1 = np.hypot(dxx1, dyy1)
592:                 dx0 = np.hypot(dxx0, dyy0)
593:             dx = dx1 / dx0
594:         else:
595:             if units == 'width':
596:                 dx = ax.bbox.width
597:             elif units == 'height':
598:                 dx = ax.bbox.height
599:             elif units == 'dots':
600:                 dx = 1.0
601:             elif units == 'inches':
602:                 dx = ax.figure.dpi
603:             else:
604:                 raise ValueError('unrecognized units')
605:         return dx
606: 
607:     def _set_transform(self):
608:         '''
609:         Sets the PolygonCollection transform to go
610:         from arrow width units to pixels.
611:         '''
612:         dx = self._dots_per_unit(self.units)
613:         self._trans_scale = dx  # pixels per arrow width unit
614:         trans = transforms.Affine2D().scale(dx)
615:         self.set_transform(trans)
616:         return trans
617: 
618:     def _angles_lengths(self, U, V, eps=1):
619:         xy = self.ax.transData.transform(self.XY)
620:         uv = np.hstack((U[:, np.newaxis], V[:, np.newaxis]))
621:         xyp = self.ax.transData.transform(self.XY + eps * uv)
622:         dxy = xyp - xy
623:         angles = np.arctan2(dxy[:, 1], dxy[:, 0])
624:         lengths = np.hypot(*dxy.T) / eps
625:         return angles, lengths
626: 
627:     def _make_verts(self, U, V, angles):
628:         uv = (U + V * 1j)
629:         str_angles = angles if isinstance(angles, six.string_types) else ''
630:         if str_angles == 'xy' and self.scale_units == 'xy':
631:             # Here eps is 1 so that if we get U, V by diffing
632:             # the X, Y arrays, the vectors will connect the
633:             # points, regardless of the axis scaling (including log).
634:             angles, lengths = self._angles_lengths(U, V, eps=1)
635:         elif str_angles == 'xy' or self.scale_units == 'xy':
636:             # Calculate eps based on the extents of the plot
637:             # so that we don't end up with roundoff error from
638:             # adding a small number to a large.
639:             eps = np.abs(self.ax.dataLim.extents).max() * 0.001
640:             angles, lengths = self._angles_lengths(U, V, eps=eps)
641:         if str_angles and self.scale_units == 'xy':
642:             a = lengths
643:         else:
644:             a = np.abs(uv)
645:         if self.scale is None:
646:             sn = max(10, math.sqrt(self.N))
647:             if self.Umask is not ma.nomask:
648:                 amean = a[~self.Umask].mean()
649:             else:
650:                 amean = a.mean()
651:             # crude auto-scaling
652:             # scale is typical arrow length as a multiple of the arrow width
653:             scale = 1.8 * amean * sn / self.span
654:         if self.scale_units is None:
655:             if self.scale is None:
656:                 self.scale = scale
657:             widthu_per_lenu = 1.0
658:         else:
659:             if self.scale_units == 'xy':
660:                 dx = 1
661:             else:
662:                 dx = self._dots_per_unit(self.scale_units)
663:             widthu_per_lenu = dx / self._trans_scale
664:             if self.scale is None:
665:                 self.scale = scale * widthu_per_lenu
666:         length = a * (widthu_per_lenu / (self.scale * self.width))
667:         X, Y = self._h_arrows(length)
668:         if str_angles == 'xy':
669:             theta = angles
670:         elif str_angles == 'uv':
671:             theta = np.angle(uv)
672:         else:
673:             theta = ma.masked_invalid(np.deg2rad(angles)).filled(0)
674:         theta = theta.reshape((-1, 1))  # for broadcasting
675:         xy = (X + Y * 1j) * np.exp(1j * theta) * self.width
676:         xy = xy[:, :, np.newaxis]
677:         XY = np.concatenate((xy.real, xy.imag), axis=2)
678:         if self.Umask is not ma.nomask:
679:             XY = ma.array(XY)
680:             XY[self.Umask] = ma.masked
681:             # This might be handled more efficiently with nans, given
682:             # that nans will end up in the paths anyway.
683: 
684:         return XY
685: 
686:     def _h_arrows(self, length):
687:         ''' length is in arrow width units '''
688:         # It might be possible to streamline the code
689:         # and speed it up a bit by using complex (x,y)
690:         # instead of separate arrays; but any gain would be slight.
691:         minsh = self.minshaft * self.headlength
692:         N = len(length)
693:         length = length.reshape(N, 1)
694:         # This number is chosen based on when pixel values overflow in Agg
695:         # causing rendering errors
696:         # length = np.minimum(length, 2 ** 16)
697:         np.clip(length, 0, 2 ** 16, out=length)
698:         # x, y: normal horizontal arrow
699:         x = np.array([0, -self.headaxislength,
700:                       -self.headlength, 0],
701:                      np.float64)
702:         x = x + np.array([0, 1, 1, 1]) * length
703:         y = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
704:         y = np.repeat(y[np.newaxis, :], N, axis=0)
705:         # x0, y0: arrow without shaft, for short vectors
706:         x0 = np.array([0, minsh - self.headaxislength,
707:                        minsh - self.headlength, minsh], np.float64)
708:         y0 = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
709:         ii = [0, 1, 2, 3, 2, 1, 0, 0]
710:         X = x.take(ii, 1)
711:         Y = y.take(ii, 1)
712:         Y[:, 3:-1] *= -1
713:         X0 = x0.take(ii)
714:         Y0 = y0.take(ii)
715:         Y0[3:-1] *= -1
716:         shrink = length / minsh if minsh != 0. else 0.
717:         X0 = shrink * X0[np.newaxis, :]
718:         Y0 = shrink * Y0[np.newaxis, :]
719:         short = np.repeat(length < minsh, 8, axis=1)
720:         # Now select X0, Y0 if short, otherwise X, Y
721:         np.copyto(X, X0, where=short)
722:         np.copyto(Y, Y0, where=short)
723:         if self.pivot == 'middle':
724:             X -= 0.5 * X[:, 3, np.newaxis]
725:         elif self.pivot == 'tip':
726:             X = X - X[:, 3, np.newaxis]   # numpy bug? using -= does not
727:                                           # work here unless we multiply
728:                                           # by a float first, as with 'mid'.
729:         elif self.pivot != 'tail':
730:             raise ValueError(("Quiver.pivot must have value in {{'middle', "
731:                               "'tip', 'tail'}} not {0}").format(self.pivot))
732: 
733:         tooshort = length < self.minlength
734:         if tooshort.any():
735:             # Use a heptagonal dot:
736:             th = np.arange(0, 8, 1, np.float64) * (np.pi / 3.0)
737:             x1 = np.cos(th) * self.minlength * 0.5
738:             y1 = np.sin(th) * self.minlength * 0.5
739:             X1 = np.repeat(x1[np.newaxis, :], N, axis=0)
740:             Y1 = np.repeat(y1[np.newaxis, :], N, axis=0)
741:             tooshort = np.repeat(tooshort, 8, 1)
742:             np.copyto(X, X1, where=tooshort)
743:             np.copyto(Y, Y1, where=tooshort)
744:         # Mask handling is deferred to the caller, _make_verts.
745:         return X, Y
746: 
747:     quiver_doc = _quiver_doc
748: 
749: 
750: _barbs_doc = r'''
751: Plot a 2-D field of barbs.
752: 
753: Call signatures::
754: 
755:   barb(U, V, **kw)
756:   barb(U, V, C, **kw)
757:   barb(X, Y, U, V, **kw)
758:   barb(X, Y, U, V, C, **kw)
759: 
760: Arguments:
761: 
762:   *X*, *Y*:
763:     The x and y coordinates of the barb locations
764:     (default is head of barb; see *pivot* kwarg)
765: 
766:   *U*, *V*:
767:     Give the x and y components of the barb shaft
768: 
769:   *C*:
770:     An optional array used to map colors to the barbs
771: 
772: All arguments may be 1-D or 2-D arrays or sequences. If *X* and *Y*
773: are absent, they will be generated as a uniform grid.  If *U* and *V*
774: are 2-D arrays but *X* and *Y* are 1-D, and if ``len(X)`` and ``len(Y)``
775: match the column and row dimensions of *U*, then *X* and *Y* will be
776: expanded with :func:`numpy.meshgrid`.
777: 
778: *U*, *V*, *C* may be masked arrays, but masked *X*, *Y* are not
779: supported at present.
780: 
781: Keyword arguments:
782: 
783:   *length*:
784:     Length of the barb in points; the other parts of the barb
785:     are scaled against this.
786:     Default is 7.
787: 
788:   *pivot*: [ 'tip' | 'middle' | float ]
789:     The part of the arrow that is at the grid point; the arrow rotates
790:     about this point, hence the name *pivot*.  Default is 'tip'. Can
791:     also be a number, which shifts the start of the barb that many
792:     points from the origin.
793: 
794:   *barbcolor*: [ color | color sequence ]
795:     Specifies the color all parts of the barb except any flags.  This
796:     parameter is analagous to the *edgecolor* parameter for polygons,
797:     which can be used instead. However this parameter will override
798:     facecolor.
799: 
800:   *flagcolor*: [ color | color sequence ]
801:     Specifies the color of any flags on the barb.  This parameter is
802:     analagous to the *facecolor* parameter for polygons, which can be
803:     used instead. However this parameter will override facecolor.  If
804:     this is not set (and *C* has not either) then *flagcolor* will be
805:     set to match *barbcolor* so that the barb has a uniform color. If
806:     *C* has been set, *flagcolor* has no effect.
807: 
808:   *sizes*:
809:     A dictionary of coefficients specifying the ratio of a given
810:     feature to the length of the barb. Only those values one wishes to
811:     override need to be included.  These features include:
812: 
813:         - 'spacing' - space between features (flags, full/half barbs)
814: 
815:         - 'height' - height (distance from shaft to top) of a flag or
816:           full barb
817: 
818:         - 'width' - width of a flag, twice the width of a full barb
819: 
820:         - 'emptybarb' - radius of the circle used for low magnitudes
821: 
822:   *fill_empty*:
823:     A flag on whether the empty barbs (circles) that are drawn should
824:     be filled with the flag color.  If they are not filled, they will
825:     be drawn such that no color is applied to the center.  Default is
826:     False
827: 
828:   *rounding*:
829:     A flag to indicate whether the vector magnitude should be rounded
830:     when allocating barb components.  If True, the magnitude is
831:     rounded to the nearest multiple of the half-barb increment.  If
832:     False, the magnitude is simply truncated to the next lowest
833:     multiple.  Default is True
834: 
835:   *barb_increments*:
836:     A dictionary of increments specifying values to associate with
837:     different parts of the barb. Only those values one wishes to
838:     override need to be included.
839: 
840:         - 'half' - half barbs (Default is 5)
841: 
842:         - 'full' - full barbs (Default is 10)
843: 
844:         - 'flag' - flags (default is 50)
845: 
846:   *flip_barb*:
847:     Either a single boolean flag or an array of booleans.  Single
848:     boolean indicates whether the lines and flags should point
849:     opposite to normal for all barbs.  An array (which should be the
850:     same size as the other data arrays) indicates whether to flip for
851:     each individual barb.  Normal behavior is for the barbs and lines
852:     to point right (comes from wind barbs having these features point
853:     towards low pressure in the Northern Hemisphere.)  Default is
854:     False
855: 
856: Barbs are traditionally used in meteorology as a way to plot the speed
857: and direction of wind observations, but can technically be used to
858: plot any two dimensional vector quantity.  As opposed to arrows, which
859: give vector magnitude by the length of the arrow, the barbs give more
860: quantitative information about the vector magnitude by putting slanted
861: lines or a triangle for various increments in magnitude, as show
862: schematically below::
863: 
864:  :     /\    \\
865:  :    /  \    \\
866:  :   /    \    \    \\
867:  :  /      \    \    \\
868:  : ------------------------------
869: 
870: .. note the double \\ at the end of each line to make the figure
871: .. render correctly
872: 
873: The largest increment is given by a triangle (or "flag"). After those
874: come full lines (barbs). The smallest increment is a half line.  There
875: is only, of course, ever at most 1 half line.  If the magnitude is
876: small and only needs a single half-line and no full lines or
877: triangles, the half-line is offset from the end of the barb so that it
878: can be easily distinguished from barbs with a single full line.  The
879: magnitude for the barb shown above would nominally be 65, using the
880: standard increments of 50, 10, and 5.
881: 
882: linewidths and edgecolors can be used to customize the barb.
883: Additional :class:`~matplotlib.collections.PolyCollection` keyword
884: arguments:
885: 
886: %(PolyCollection)s
887: ''' % docstring.interpd.params
888: 
889: docstring.interpd.update(barbs_doc=_barbs_doc)
890: 
891: 
892: class Barbs(mcollections.PolyCollection):
893:     '''
894:     Specialized PolyCollection for barbs.
895: 
896:     The only API method is :meth:`set_UVC`, which can be used to
897:     change the size, orientation, and color of the arrows.  Locations
898:     are changed using the :meth:`set_offsets` collection method.
899:     Possibly this method will be useful in animations.
900: 
901:     There is one internal function :meth:`_find_tails` which finds
902:     exactly what should be put on the barb given the vector magnitude.
903:     From there :meth:`_make_barbs` is used to find the vertices of the
904:     polygon to represent the barb based on this information.
905:     '''
906:     # This may be an abuse of polygons here to render what is essentially maybe
907:     # 1 triangle and a series of lines.  It works fine as far as I can tell
908:     # however.
909:     @docstring.interpd
910:     def __init__(self, ax, *args, **kw):
911:         '''
912:         The constructor takes one required argument, an Axes
913:         instance, followed by the args and kwargs described
914:         by the following pylab interface documentation:
915:         %(barbs_doc)s
916:         '''
917:         self._pivot = kw.pop('pivot', 'tip')
918:         self._length = kw.pop('length', 7)
919:         barbcolor = kw.pop('barbcolor', None)
920:         flagcolor = kw.pop('flagcolor', None)
921:         self.sizes = kw.pop('sizes', dict())
922:         self.fill_empty = kw.pop('fill_empty', False)
923:         self.barb_increments = kw.pop('barb_increments', dict())
924:         self.rounding = kw.pop('rounding', True)
925:         self.flip = kw.pop('flip_barb', False)
926:         transform = kw.pop('transform', ax.transData)
927: 
928:         # Flagcolor and barbcolor provide convenience parameters for
929:         # setting the facecolor and edgecolor, respectively, of the barb
930:         # polygon.  We also work here to make the flag the same color as the
931:         # rest of the barb by default
932: 
933:         if None in (barbcolor, flagcolor):
934:             kw['edgecolors'] = 'face'
935:             if flagcolor:
936:                 kw['facecolors'] = flagcolor
937:             elif barbcolor:
938:                 kw['facecolors'] = barbcolor
939:             else:
940:                 # Set to facecolor passed in or default to black
941:                 kw.setdefault('facecolors', 'k')
942:         else:
943:             kw['edgecolors'] = barbcolor
944:             kw['facecolors'] = flagcolor
945: 
946:         # Explicitly set a line width if we're not given one, otherwise
947:         # polygons are not outlined and we get no barbs
948:         if 'linewidth' not in kw and 'lw' not in kw:
949:             kw['linewidth'] = 1
950: 
951:         # Parse out the data arrays from the various configurations supported
952:         x, y, u, v, c = _parse_args(*args)
953:         self.x = x
954:         self.y = y
955:         xy = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
956: 
957:         # Make a collection
958:         barb_size = self._length ** 2 / 4  # Empirically determined
959:         mcollections.PolyCollection.__init__(self, [], (barb_size,),
960:                                              offsets=xy,
961:                                              transOffset=transform, **kw)
962:         self.set_transform(transforms.IdentityTransform())
963: 
964:         self.set_UVC(u, v, c)
965: 
966:     def _find_tails(self, mag, rounding=True, half=5, full=10, flag=50):
967:         '''
968:         Find how many of each of the tail pieces is necessary.  Flag
969:         specifies the increment for a flag, barb for a full barb, and half for
970:         half a barb. Mag should be the magnitude of a vector (i.e., >= 0).
971: 
972:         This returns a tuple of:
973: 
974:             (*number of flags*, *number of barbs*, *half_flag*, *empty_flag*)
975: 
976:         *half_flag* is a boolean whether half of a barb is needed,
977:         since there should only ever be one half on a given
978:         barb. *empty_flag* flag is an array of flags to easily tell if
979:         a barb is empty (too low to plot any barbs/flags.
980:         '''
981: 
982:         # If rounding, round to the nearest multiple of half, the smallest
983:         # increment
984:         if rounding:
985:             mag = half * (mag / half + 0.5).astype(int)
986: 
987:         num_flags = np.floor(mag / flag).astype(int)
988:         mag = np.mod(mag, flag)
989: 
990:         num_barb = np.floor(mag / full).astype(int)
991:         mag = np.mod(mag, full)
992: 
993:         half_flag = mag >= half
994:         empty_flag = ~(half_flag | (num_flags > 0) | (num_barb > 0))
995: 
996:         return num_flags, num_barb, half_flag, empty_flag
997: 
998:     def _make_barbs(self, u, v, nflags, nbarbs, half_barb, empty_flag, length,
999:                     pivot, sizes, fill_empty, flip):
1000:         '''
1001:         This function actually creates the wind barbs.  *u* and *v*
1002:         are components of the vector in the *x* and *y* directions,
1003:         respectively.
1004: 
1005:         *nflags*, *nbarbs*, and *half_barb*, empty_flag* are,
1006:         *respectively, the number of flags, number of barbs, flag for
1007:         *half a barb, and flag for empty barb, ostensibly obtained
1008:         *from :meth:`_find_tails`.
1009: 
1010:         *length* is the length of the barb staff in points.
1011: 
1012:         *pivot* specifies the point on the barb around which the
1013:         entire barb should be rotated.  Right now, valid options are
1014:         'tip' and 'middle'. Can also be a number, which shifts the start
1015:         of the barb that many points from the origin.
1016: 
1017:         *sizes* is a dictionary of coefficients specifying the ratio
1018:         of a given feature to the length of the barb. These features
1019:         include:
1020: 
1021:             - *spacing*: space between features (flags, full/half
1022:                barbs)
1023: 
1024:             - *height*: distance from shaft of top of a flag or full
1025:                barb
1026: 
1027:             - *width* - width of a flag, twice the width of a full barb
1028: 
1029:             - *emptybarb* - radius of the circle used for low
1030:                magnitudes
1031: 
1032:         *fill_empty* specifies whether the circle representing an
1033:         empty barb should be filled or not (this changes the drawing
1034:         of the polygon).
1035: 
1036:         *flip* is a flag indicating whether the features should be flipped to
1037:         the other side of the barb (useful for winds in the southern
1038:         hemisphere).
1039: 
1040:         This function returns list of arrays of vertices, defining a polygon
1041:         for each of the wind barbs.  These polygons have been rotated to
1042:         properly align with the vector direction.
1043:         '''
1044: 
1045:         # These control the spacing and size of barb elements relative to the
1046:         # length of the shaft
1047:         spacing = length * sizes.get('spacing', 0.125)
1048:         full_height = length * sizes.get('height', 0.4)
1049:         full_width = length * sizes.get('width', 0.25)
1050:         empty_rad = length * sizes.get('emptybarb', 0.15)
1051: 
1052:         # Controls y point where to pivot the barb.
1053:         pivot_points = dict(tip=0.0, middle=-length / 2.)
1054: 
1055:         # Check for flip
1056:         if flip:
1057:             full_height = -full_height
1058: 
1059:         endx = 0.0
1060:         try:
1061:             endy = float(pivot)
1062:         except ValueError:
1063:             endy = pivot_points[pivot.lower()]
1064: 
1065:         # Get the appropriate angle for the vector components.  The offset is
1066:         # due to the way the barb is initially drawn, going down the y-axis.
1067:         # This makes sense in a meteorological mode of thinking since there 0
1068:         # degrees corresponds to north (the y-axis traditionally)
1069:         angles = -(ma.arctan2(v, u) + np.pi / 2)
1070: 
1071:         # Used for low magnitude.  We just get the vertices, so if we make it
1072:         # out here, it can be reused.  The center set here should put the
1073:         # center of the circle at the location(offset), rather than at the
1074:         # same point as the barb pivot; this seems more sensible.
1075:         circ = CirclePolygon((0, 0), radius=empty_rad).get_verts()
1076:         if fill_empty:
1077:             empty_barb = circ
1078:         else:
1079:             # If we don't want the empty one filled, we make a degenerate
1080:             # polygon that wraps back over itself
1081:             empty_barb = np.concatenate((circ, circ[::-1]))
1082: 
1083:         barb_list = []
1084:         for index, angle in np.ndenumerate(angles):
1085:             # If the vector magnitude is too weak to draw anything, plot an
1086:             # empty circle instead
1087:             if empty_flag[index]:
1088:                 # We can skip the transform since the circle has no preferred
1089:                 # orientation
1090:                 barb_list.append(empty_barb)
1091:                 continue
1092: 
1093:             poly_verts = [(endx, endy)]
1094:             offset = length
1095: 
1096:             # Add vertices for each flag
1097:             for i in range(nflags[index]):
1098:                 # The spacing that works for the barbs is a little to much for
1099:                 # the flags, but this only occurs when we have more than 1
1100:                 # flag.
1101:                 if offset != length:
1102:                     offset += spacing / 2.
1103:                 poly_verts.extend(
1104:                     [[endx, endy + offset],
1105:                      [endx + full_height, endy - full_width / 2 + offset],
1106:                      [endx, endy - full_width + offset]])
1107: 
1108:                 offset -= full_width + spacing
1109: 
1110:             # Add vertices for each barb.  These really are lines, but works
1111:             # great adding 3 vertices that basically pull the polygon out and
1112:             # back down the line
1113:             for i in range(nbarbs[index]):
1114:                 poly_verts.extend(
1115:                     [(endx, endy + offset),
1116:                      (endx + full_height, endy + offset + full_width / 2),
1117:                      (endx, endy + offset)])
1118: 
1119:                 offset -= spacing
1120: 
1121:             # Add the vertices for half a barb, if needed
1122:             if half_barb[index]:
1123:                 # If the half barb is the first on the staff, traditionally it
1124:                 # is offset from the end to make it easy to distinguish from a
1125:                 # barb with a full one
1126:                 if offset == length:
1127:                     poly_verts.append((endx, endy + offset))
1128:                     offset -= 1.5 * spacing
1129:                 poly_verts.extend(
1130:                     [(endx, endy + offset),
1131:                      (endx + full_height / 2, endy + offset + full_width / 4),
1132:                      (endx, endy + offset)])
1133: 
1134:             # Rotate the barb according the angle. Making the barb first and
1135:             # then rotating it made the math for drawing the barb really easy.
1136:             # Also, the transform framework makes doing the rotation simple.
1137:             poly_verts = transforms.Affine2D().rotate(-angle).transform(
1138:                 poly_verts)
1139:             barb_list.append(poly_verts)
1140: 
1141:         return barb_list
1142: 
1143:     def set_UVC(self, U, V, C=None):
1144:         self.u = ma.masked_invalid(U, copy=False).ravel()
1145:         self.v = ma.masked_invalid(V, copy=False).ravel()
1146:         if C is not None:
1147:             c = ma.masked_invalid(C, copy=False).ravel()
1148:             x, y, u, v, c = delete_masked_points(self.x.ravel(),
1149:                                                  self.y.ravel(),
1150:                                                  self.u, self.v, c)
1151:             _check_consistent_shapes(x, y, u, v, c)
1152:         else:
1153:             x, y, u, v = delete_masked_points(self.x.ravel(), self.y.ravel(),
1154:                                               self.u, self.v)
1155:             _check_consistent_shapes(x, y, u, v)
1156: 
1157:         magnitude = np.hypot(u, v)
1158:         flags, barbs, halves, empty = self._find_tails(magnitude,
1159:                                                        self.rounding,
1160:                                                        **self.barb_increments)
1161: 
1162:         # Get the vertices for each of the barbs
1163: 
1164:         plot_barbs = self._make_barbs(u, v, flags, barbs, halves, empty,
1165:                                       self._length, self._pivot, self.sizes,
1166:                                       self.fill_empty, self.flip)
1167:         self.set_verts(plot_barbs)
1168: 
1169:         # Set the color array
1170:         if C is not None:
1171:             self.set_array(c)
1172: 
1173:         # Update the offsets in case the masked data changed
1174:         xy = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
1175:         self._offsets = xy
1176:         self.stale = True
1177: 
1178:     def set_offsets(self, xy):
1179:         '''
1180:         Set the offsets for the barb polygons.  This saves the offsets passed
1181:         in and actually sets version masked as appropriate for the existing
1182:         U/V data. *offsets* should be a sequence.
1183: 
1184:         ACCEPTS: sequence of pairs of floats
1185:         '''
1186:         self.x = xy[:, 0]
1187:         self.y = xy[:, 1]
1188:         x, y, u, v = delete_masked_points(self.x.ravel(), self.y.ravel(),
1189:                                           self.u, self.v)
1190:         _check_consistent_shapes(x, y, u, v)
1191:         xy = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
1192:         mcollections.PolyCollection.set_offsets(self, xy)
1193:         self.stale = True
1194: 
1195:     set_offsets.__doc__ = mcollections.PolyCollection.set_offsets.__doc__
1196: 
1197:     barbs_doc = _barbs_doc
1198: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_120719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'unicode', u'\nSupport for plotting vector fields.\n\nPresently this contains Quiver and Barb. Quiver plots an arrow in the\ndirection of the vector, with the size of the arrow related to the\nmagnitude of the vector.\n\nBarbs are like quiver in that they point along a vector, but\nthe magnitude of the vector is given schematically by the presence of barbs\nor flags on the barb.\n\nThis will also become a home for things such as standard\ndeviation ellipses, which can and will be derived very easily from\nthe Quiver code.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import six' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120720 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'six')

if (type(import_120720) is not StypyTypeError):

    if (import_120720 != 'pyd_module'):
        __import__(import_120720)
        sys_modules_120721 = sys.modules[import_120720]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'six', sys_modules_120721.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'six', import_120720)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import weakref' statement (line 21)
import weakref

import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'weakref', weakref, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import numpy' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120722 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy')

if (type(import_120722) is not StypyTypeError):

    if (import_120722 != 'pyd_module'):
        __import__(import_120722)
        sys_modules_120723 = sys.modules[import_120722]
        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'np', sys_modules_120723.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy', import_120722)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy import ma' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120724 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy')

if (type(import_120724) is not StypyTypeError):

    if (import_120724 != 'pyd_module'):
        __import__(import_120724)
        sys_modules_120725 = sys.modules[import_120724]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy', sys_modules_120725.module_type_store, module_type_store, ['ma'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_120725, sys_modules_120725.module_type_store, module_type_store)
    else:
        from numpy import ma

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy', None, module_type_store, ['ma'], [ma])

else:
    # Assigning a type to the variable 'numpy' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy', import_120724)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'import matplotlib.collections' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120726 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib.collections')

if (type(import_120726) is not StypyTypeError):

    if (import_120726 != 'pyd_module'):
        __import__(import_120726)
        sys_modules_120727 = sys.modules[import_120726]
        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'mcollections', sys_modules_120727.module_type_store, module_type_store)
    else:
        import matplotlib.collections as mcollections

        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'mcollections', matplotlib.collections, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.collections' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib.collections', import_120726)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'import matplotlib.transforms' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120728 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.transforms')

if (type(import_120728) is not StypyTypeError):

    if (import_120728 != 'pyd_module'):
        __import__(import_120728)
        sys_modules_120729 = sys.modules[import_120728]
        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'transforms', sys_modules_120729.module_type_store, module_type_store)
    else:
        import matplotlib.transforms as transforms

        import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'transforms', matplotlib.transforms, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.transforms', import_120728)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'import matplotlib.text' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120730 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib.text')

if (type(import_120730) is not StypyTypeError):

    if (import_120730 != 'pyd_module'):
        __import__(import_120730)
        sys_modules_120731 = sys.modules[import_120730]
        import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'mtext', sys_modules_120731.module_type_store, module_type_store)
    else:
        import matplotlib.text as mtext

        import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'mtext', matplotlib.text, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.text' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'matplotlib.text', import_120730)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import matplotlib.artist' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120732 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.artist')

if (type(import_120732) is not StypyTypeError):

    if (import_120732 != 'pyd_module'):
        __import__(import_120732)
        sys_modules_120733 = sys.modules[import_120732]
        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'martist', sys_modules_120733.module_type_store, module_type_store)
    else:
        import matplotlib.artist as martist

        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'martist', matplotlib.artist, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.artist' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.artist', import_120732)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from matplotlib.artist import allow_rasterization' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120734 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.artist')

if (type(import_120734) is not StypyTypeError):

    if (import_120734 != 'pyd_module'):
        __import__(import_120734)
        sys_modules_120735 = sys.modules[import_120734]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.artist', sys_modules_120735.module_type_store, module_type_store, ['allow_rasterization'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_120735, sys_modules_120735.module_type_store, module_type_store)
    else:
        from matplotlib.artist import allow_rasterization

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.artist', None, module_type_store, ['allow_rasterization'], [allow_rasterization])

else:
    # Assigning a type to the variable 'matplotlib.artist' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.artist', import_120734)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from matplotlib import docstring' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120736 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib')

if (type(import_120736) is not StypyTypeError):

    if (import_120736 != 'pyd_module'):
        __import__(import_120736)
        sys_modules_120737 = sys.modules[import_120736]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib', sys_modules_120737.module_type_store, module_type_store, ['docstring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_120737, sys_modules_120737.module_type_store, module_type_store)
    else:
        from matplotlib import docstring

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib', None, module_type_store, ['docstring'], [docstring])

else:
    # Assigning a type to the variable 'matplotlib' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib', import_120736)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'import matplotlib.font_manager' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120738 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.font_manager')

if (type(import_120738) is not StypyTypeError):

    if (import_120738 != 'pyd_module'):
        __import__(import_120738)
        sys_modules_120739 = sys.modules[import_120738]
        import_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'font_manager', sys_modules_120739.module_type_store, module_type_store)
    else:
        import matplotlib.font_manager as font_manager

        import_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'font_manager', matplotlib.font_manager, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.font_manager' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.font_manager', import_120738)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'import matplotlib.cbook' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120740 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.cbook')

if (type(import_120740) is not StypyTypeError):

    if (import_120740 != 'pyd_module'):
        __import__(import_120740)
        sys_modules_120741 = sys.modules[import_120740]
        import_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'cbook', sys_modules_120741.module_type_store, module_type_store)
    else:
        import matplotlib.cbook as cbook

        import_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'cbook', matplotlib.cbook, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.cbook', import_120740)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'from matplotlib.cbook import delete_masked_points' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120742 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.cbook')

if (type(import_120742) is not StypyTypeError):

    if (import_120742 != 'pyd_module'):
        __import__(import_120742)
        sys_modules_120743 = sys.modules[import_120742]
        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.cbook', sys_modules_120743.module_type_store, module_type_store, ['delete_masked_points'])
        nest_module(stypy.reporting.localization.Localization(__file__, 33, 0), __file__, sys_modules_120743, sys_modules_120743.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import delete_masked_points

        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.cbook', None, module_type_store, ['delete_masked_points'], [delete_masked_points])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.cbook', import_120742)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from matplotlib.patches import CirclePolygon' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_120744 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.patches')

if (type(import_120744) is not StypyTypeError):

    if (import_120744 != 'pyd_module'):
        __import__(import_120744)
        sys_modules_120745 = sys.modules[import_120744]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.patches', sys_modules_120745.module_type_store, module_type_store, ['CirclePolygon'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_120745, sys_modules_120745.module_type_store, module_type_store)
    else:
        from matplotlib.patches import CirclePolygon

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.patches', None, module_type_store, ['CirclePolygon'], [CirclePolygon])

else:
    # Assigning a type to the variable 'matplotlib.patches' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.patches', import_120744)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'import math' statement (line 35)
import math

import_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'math', math, module_type_store)


# Assigning a BinOp to a Name (line 38):

# Assigning a BinOp to a Name (line 38):
unicode_120746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, (-1)), 'unicode', u"\nPlot a 2-D field of arrows.\n\nCall signatures::\n\n  quiver(U, V, **kw)\n  quiver(U, V, C, **kw)\n  quiver(X, Y, U, V, **kw)\n  quiver(X, Y, U, V, C, **kw)\n\n*U* and *V* are the arrow data, *X* and *Y* set the location of the\narrows, and *C* sets the color of the arrows. These arguments may be 1-D or\n2-D arrays or sequences.\n\nIf *X* and *Y* are absent, they will be generated as a uniform grid.\nIf *U* and *V* are 2-D arrays and *X* and *Y* are 1-D, and if ``len(X)`` and\n``len(Y)`` match the column and row dimensions of *U*, then *X* and *Y* will be\nexpanded with :func:`numpy.meshgrid`.\n\nThe default settings auto-scales the length of the arrows to a reasonable size.\nTo change this behavior see the *scale* and *scale_units* kwargs.\n\nThe defaults give a slightly swept-back arrow; to make the head a\ntriangle, make *headaxislength* the same as *headlength*. To make the\narrow more pointed, reduce *headwidth* or increase *headlength* and\n*headaxislength*. To make the head smaller relative to the shaft,\nscale down all the head parameters. You will probably do best to leave\nminshaft alone.\n\n*linewidths* and *edgecolors* can be used to customize the arrow\noutlines.\n\nParameters\n----------\nX : 1D or 2D array, sequence, optional\n    The x coordinates of the arrow locations\nY : 1D or 2D array, sequence, optional\n    The y coordinates of the arrow locations\nU : 1D or 2D array or masked array, sequence\n    The x components of the arrow vectors\nV : 1D or 2D array or masked array, sequence\n    The y components of the arrow vectors\nC : 1D or 2D array, sequence, optional\n    The arrow colors\nunits : [ 'width' | 'height' | 'dots' | 'inches' | 'x' | 'y' | 'xy' ]\n    The arrow dimensions (except for *length*) are measured in multiples of\n    this unit.\n\n    'width' or 'height': the width or height of the axis\n\n    'dots' or 'inches': pixels or inches, based on the figure dpi\n\n    'x', 'y', or 'xy': respectively *X*, *Y*, or :math:`\\sqrt{X^2 + Y^2}`\n    in data units\n\n    The arrows scale differently depending on the units.  For\n    'x' or 'y', the arrows get larger as one zooms in; for other\n    units, the arrow size is independent of the zoom state.  For\n    'width or 'height', the arrow size increases with the width and\n    height of the axes, respectively, when the window is resized;\n    for 'dots' or 'inches', resizing does not change the arrows.\nangles : [ 'uv' | 'xy' ], array, optional\n    Method for determining the angle of the arrows. Default is 'uv'.\n\n    'uv': the arrow axis aspect ratio is 1 so that\n    if *U*==*V* the orientation of the arrow on the plot is 45 degrees\n    counter-clockwise from the horizontal axis (positive to the right).\n\n    'xy': arrows point from (x,y) to (x+u, y+v).\n    Use this for plotting a gradient field, for example.\n\n    Alternatively, arbitrary angles may be specified as an array\n    of values in degrees, counter-clockwise from the horizontal axis.\n\n    Note: inverting a data axis will correspondingly invert the\n    arrows only with ``angles='xy'``.\nscale : None, float, optional\n    Number of data units per arrow length unit, e.g., m/s per plot width; a\n    smaller scale parameter makes the arrow longer. Default is *None*.\n\n    If *None*, a simple autoscaling algorithm is used, based on the average\n    vector length and the number of vectors. The arrow length unit is given by\n    the *scale_units* parameter\nscale_units : [ 'width' | 'height' | 'dots' | 'inches' | 'x' | 'y' | 'xy' ], None, optional\n    If the *scale* kwarg is *None*, the arrow length unit. Default is *None*.\n\n    e.g. *scale_units* is 'inches', *scale* is 2.0, and\n    ``(u,v) = (1,0)``, then the vector will be 0.5 inches long.\n\n    If *scale_units* is 'width'/'height', then the vector will be half the\n    width/height of the axes.\n\n    If *scale_units* is 'x' then the vector will be 0.5 x-axis\n    units. To plot vectors in the x-y plane, with u and v having\n    the same units as x and y, use\n    ``angles='xy', scale_units='xy', scale=1``.\nwidth : scalar, optional\n    Shaft width in arrow units; default depends on choice of units,\n    above, and number of vectors; a typical starting value is about\n    0.005 times the width of the plot.\nheadwidth : scalar, optional\n    Head width as multiple of shaft width, default is 3\nheadlength : scalar, optional\n    Head length as multiple of shaft width, default is 5\nheadaxislength : scalar, optional\n    Head length at shaft intersection, default is 4.5\nminshaft : scalar, optional\n    Length below which arrow scales, in units of head length. Do not\n    set this to less than 1, or small arrows will look terrible!\n    Default is 1\nminlength : scalar, optional\n    Minimum length as a multiple of shaft width; if an arrow length\n    is less than this, plot a dot (hexagon) of this diameter instead.\n    Default is 1.\npivot : [ 'tail' | 'mid' | 'middle' | 'tip' ], optional\n    The part of the arrow that is at the grid point; the arrow rotates\n    about this point, hence the name *pivot*.\ncolor : [ color | color sequence ], optional\n    This is a synonym for the\n    :class:`~matplotlib.collections.PolyCollection` facecolor kwarg.\n    If *C* has been set, *color* has no effect.\n\nNotes\n-----\nAdditional :class:`~matplotlib.collections.PolyCollection`\nkeyword arguments:\n\n%(PolyCollection)s\n\nSee Also\n--------\nquiverkey : Add a key to a quiver plot\n")
# Getting the type of 'docstring' (line 171)
docstring_120747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 6), 'docstring')
# Obtaining the member 'interpd' of a type (line 171)
interpd_120748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 6), docstring_120747, 'interpd')
# Obtaining the member 'params' of a type (line 171)
params_120749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 6), interpd_120748, 'params')
# Applying the binary operator '%' (line 171)
result_mod_120750 = python_operator(stypy.reporting.localization.Localization(__file__, 171, (-1)), '%', unicode_120746, params_120749)

# Assigning a type to the variable '_quiver_doc' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), '_quiver_doc', result_mod_120750)

# Assigning a Str to a Name (line 173):

# Assigning a Str to a Name (line 173):
unicode_120751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, (-1)), 'unicode', u"\nAdd a key to a quiver plot.\n\nCall signature::\n\n  quiverkey(Q, X, Y, U, label, **kw)\n\nArguments:\n\n  *Q*:\n    The Quiver instance returned by a call to quiver.\n\n  *X*, *Y*:\n    The location of the key; additional explanation follows.\n\n  *U*:\n    The length of the key\n\n  *label*:\n    A string with the length and units of the key\n\nKeyword arguments:\n\n  *angle* = 0\n    The angle of the key arrow. Measured in degrees anti-clockwise from the\n    x-axis.\n\n  *coordinates* = [ 'axes' | 'figure' | 'data' | 'inches' ]\n    Coordinate system and units for *X*, *Y*: 'axes' and 'figure' are\n    normalized coordinate systems with 0,0 in the lower left and 1,1\n    in the upper right; 'data' are the axes data coordinates (used for\n    the locations of the vectors in the quiver plot itself); 'inches'\n    is position in the figure in inches, with 0,0 at the lower left\n    corner.\n\n  *color*:\n    overrides face and edge colors from *Q*.\n\n  *labelpos* = [ 'N' | 'S' | 'E' | 'W' ]\n    Position the label above, below, to the right, to the left of the\n    arrow, respectively.\n\n  *labelsep*:\n    Distance in inches between the arrow and the label.  Default is\n    0.1\n\n  *labelcolor*:\n    defaults to default :class:`~matplotlib.text.Text` color.\n\n  *fontproperties*:\n    A dictionary with keyword arguments accepted by the\n    :class:`~matplotlib.font_manager.FontProperties` initializer:\n    *family*, *style*, *variant*, *size*, *weight*\n\nAny additional keyword arguments are used to override vector\nproperties taken from *Q*.\n\nThe positioning of the key depends on *X*, *Y*, *coordinates*, and\n*labelpos*.  If *labelpos* is 'N' or 'S', *X*, *Y* give the position\nof the middle of the key arrow.  If *labelpos* is 'E', *X*, *Y*\npositions the head, and if *labelpos* is 'W', *X*, *Y* positions the\ntail; in either of these two cases, *X*, *Y* is somewhere in the\nmiddle of the arrow+label key object.\n")
# Assigning a type to the variable '_quiverkey_doc' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), '_quiverkey_doc', unicode_120751)
# Declaration of the 'QuiverKey' class
# Getting the type of 'martist' (line 239)
martist_120752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'martist')
# Obtaining the member 'Artist' of a type (line 239)
Artist_120753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 16), martist_120752, 'Artist')

class QuiverKey(Artist_120753, ):
    unicode_120754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 4), 'unicode', u' Labelled arrow for use as a quiver plot scale key.')
    
    # Assigning a Dict to a Name (line 241):
    
    # Assigning a Dict to a Name (line 242):
    
    # Assigning a Dict to a Name (line 243):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 245, 4, False)
        # Assigning a type to the variable 'self' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'QuiverKey.__init__', ['Q', 'X', 'Y', 'U', 'label'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['Q', 'X', 'Y', 'U', 'label'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'self' (line 246)
        self_120758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 32), 'self', False)
        # Processing the call keyword arguments (line 246)
        kwargs_120759 = {}
        # Getting the type of 'martist' (line 246)
        martist_120755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'martist', False)
        # Obtaining the member 'Artist' of a type (line 246)
        Artist_120756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), martist_120755, 'Artist')
        # Obtaining the member '__init__' of a type (line 246)
        init___120757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), Artist_120756, '__init__')
        # Calling __init__(args, kwargs) (line 246)
        init___call_result_120760 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), init___120757, *[self_120758], **kwargs_120759)
        
        
        # Assigning a Name to a Attribute (line 247):
        
        # Assigning a Name to a Attribute (line 247):
        # Getting the type of 'Q' (line 247)
        Q_120761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'Q')
        # Getting the type of 'self' (line 247)
        self_120762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'self')
        # Setting the type of the member 'Q' of a type (line 247)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), self_120762, 'Q', Q_120761)
        
        # Assigning a Name to a Attribute (line 248):
        
        # Assigning a Name to a Attribute (line 248):
        # Getting the type of 'X' (line 248)
        X_120763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 17), 'X')
        # Getting the type of 'self' (line 248)
        self_120764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self')
        # Setting the type of the member 'X' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_120764, 'X', X_120763)
        
        # Assigning a Name to a Attribute (line 249):
        
        # Assigning a Name to a Attribute (line 249):
        # Getting the type of 'Y' (line 249)
        Y_120765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 17), 'Y')
        # Getting the type of 'self' (line 249)
        self_120766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self')
        # Setting the type of the member 'Y' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_120766, 'Y', Y_120765)
        
        # Assigning a Name to a Attribute (line 250):
        
        # Assigning a Name to a Attribute (line 250):
        # Getting the type of 'U' (line 250)
        U_120767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 17), 'U')
        # Getting the type of 'self' (line 250)
        self_120768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member 'U' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_120768, 'U', U_120767)
        
        # Assigning a Call to a Attribute (line 251):
        
        # Assigning a Call to a Attribute (line 251):
        
        # Call to pop(...): (line 251)
        # Processing the call arguments (line 251)
        unicode_120771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 28), 'unicode', u'angle')
        int_120772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 37), 'int')
        # Processing the call keyword arguments (line 251)
        kwargs_120773 = {}
        # Getting the type of 'kw' (line 251)
        kw_120769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 21), 'kw', False)
        # Obtaining the member 'pop' of a type (line 251)
        pop_120770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 21), kw_120769, 'pop')
        # Calling pop(args, kwargs) (line 251)
        pop_call_result_120774 = invoke(stypy.reporting.localization.Localization(__file__, 251, 21), pop_120770, *[unicode_120771, int_120772], **kwargs_120773)
        
        # Getting the type of 'self' (line 251)
        self_120775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self')
        # Setting the type of the member 'angle' of a type (line 251)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_120775, 'angle', pop_call_result_120774)
        
        # Assigning a Call to a Attribute (line 252):
        
        # Assigning a Call to a Attribute (line 252):
        
        # Call to pop(...): (line 252)
        # Processing the call arguments (line 252)
        unicode_120778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 28), 'unicode', u'coordinates')
        unicode_120779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 43), 'unicode', u'axes')
        # Processing the call keyword arguments (line 252)
        kwargs_120780 = {}
        # Getting the type of 'kw' (line 252)
        kw_120776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 21), 'kw', False)
        # Obtaining the member 'pop' of a type (line 252)
        pop_120777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 21), kw_120776, 'pop')
        # Calling pop(args, kwargs) (line 252)
        pop_call_result_120781 = invoke(stypy.reporting.localization.Localization(__file__, 252, 21), pop_120777, *[unicode_120778, unicode_120779], **kwargs_120780)
        
        # Getting the type of 'self' (line 252)
        self_120782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'self')
        # Setting the type of the member 'coord' of a type (line 252)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), self_120782, 'coord', pop_call_result_120781)
        
        # Assigning a Call to a Attribute (line 253):
        
        # Assigning a Call to a Attribute (line 253):
        
        # Call to pop(...): (line 253)
        # Processing the call arguments (line 253)
        unicode_120785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 28), 'unicode', u'color')
        # Getting the type of 'None' (line 253)
        None_120786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 37), 'None', False)
        # Processing the call keyword arguments (line 253)
        kwargs_120787 = {}
        # Getting the type of 'kw' (line 253)
        kw_120783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 21), 'kw', False)
        # Obtaining the member 'pop' of a type (line 253)
        pop_120784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 21), kw_120783, 'pop')
        # Calling pop(args, kwargs) (line 253)
        pop_call_result_120788 = invoke(stypy.reporting.localization.Localization(__file__, 253, 21), pop_120784, *[unicode_120785, None_120786], **kwargs_120787)
        
        # Getting the type of 'self' (line 253)
        self_120789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'self')
        # Setting the type of the member 'color' of a type (line 253)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), self_120789, 'color', pop_call_result_120788)
        
        # Assigning a Name to a Attribute (line 254):
        
        # Assigning a Name to a Attribute (line 254):
        # Getting the type of 'label' (line 254)
        label_120790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 21), 'label')
        # Getting the type of 'self' (line 254)
        self_120791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'self')
        # Setting the type of the member 'label' of a type (line 254)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), self_120791, 'label', label_120790)
        
        # Assigning a Call to a Attribute (line 255):
        
        # Assigning a Call to a Attribute (line 255):
        
        # Call to pop(...): (line 255)
        # Processing the call arguments (line 255)
        unicode_120794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 39), 'unicode', u'labelsep')
        float_120795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 51), 'float')
        # Processing the call keyword arguments (line 255)
        kwargs_120796 = {}
        # Getting the type of 'kw' (line 255)
        kw_120792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 32), 'kw', False)
        # Obtaining the member 'pop' of a type (line 255)
        pop_120793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 32), kw_120792, 'pop')
        # Calling pop(args, kwargs) (line 255)
        pop_call_result_120797 = invoke(stypy.reporting.localization.Localization(__file__, 255, 32), pop_120793, *[unicode_120794, float_120795], **kwargs_120796)
        
        # Getting the type of 'self' (line 255)
        self_120798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'self')
        # Setting the type of the member '_labelsep_inches' of a type (line 255)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), self_120798, '_labelsep_inches', pop_call_result_120797)
        
        # Assigning a BinOp to a Attribute (line 256):
        
        # Assigning a BinOp to a Attribute (line 256):
        # Getting the type of 'self' (line 256)
        self_120799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 25), 'self')
        # Obtaining the member '_labelsep_inches' of a type (line 256)
        _labelsep_inches_120800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 25), self_120799, '_labelsep_inches')
        # Getting the type of 'Q' (line 256)
        Q_120801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 49), 'Q')
        # Obtaining the member 'ax' of a type (line 256)
        ax_120802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 49), Q_120801, 'ax')
        # Obtaining the member 'figure' of a type (line 256)
        figure_120803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 49), ax_120802, 'figure')
        # Obtaining the member 'dpi' of a type (line 256)
        dpi_120804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 49), figure_120803, 'dpi')
        # Applying the binary operator '*' (line 256)
        result_mul_120805 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 25), '*', _labelsep_inches_120800, dpi_120804)
        
        # Getting the type of 'self' (line 256)
        self_120806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'self')
        # Setting the type of the member 'labelsep' of a type (line 256)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), self_120806, 'labelsep', result_mul_120805)
        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Call to ref(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'self' (line 259)
        self_120809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 32), 'self', False)
        # Processing the call keyword arguments (line 259)
        kwargs_120810 = {}
        # Getting the type of 'weakref' (line 259)
        weakref_120807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'weakref', False)
        # Obtaining the member 'ref' of a type (line 259)
        ref_120808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 20), weakref_120807, 'ref')
        # Calling ref(args, kwargs) (line 259)
        ref_call_result_120811 = invoke(stypy.reporting.localization.Localization(__file__, 259, 20), ref_120808, *[self_120809], **kwargs_120810)
        
        # Assigning a type to the variable 'weak_self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'weak_self', ref_call_result_120811)

        @norecursion
        def on_dpi_change(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'on_dpi_change'
            module_type_store = module_type_store.open_function_context('on_dpi_change', 261, 8, False)
            
            # Passed parameters checking function
            on_dpi_change.stypy_localization = localization
            on_dpi_change.stypy_type_of_self = None
            on_dpi_change.stypy_type_store = module_type_store
            on_dpi_change.stypy_function_name = 'on_dpi_change'
            on_dpi_change.stypy_param_names_list = ['fig']
            on_dpi_change.stypy_varargs_param_name = None
            on_dpi_change.stypy_kwargs_param_name = None
            on_dpi_change.stypy_call_defaults = defaults
            on_dpi_change.stypy_call_varargs = varargs
            on_dpi_change.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'on_dpi_change', ['fig'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'on_dpi_change', localization, ['fig'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'on_dpi_change(...)' code ##################

            
            # Assigning a Call to a Name (line 262):
            
            # Assigning a Call to a Name (line 262):
            
            # Call to weak_self(...): (line 262)
            # Processing the call keyword arguments (line 262)
            kwargs_120813 = {}
            # Getting the type of 'weak_self' (line 262)
            weak_self_120812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 27), 'weak_self', False)
            # Calling weak_self(args, kwargs) (line 262)
            weak_self_call_result_120814 = invoke(stypy.reporting.localization.Localization(__file__, 262, 27), weak_self_120812, *[], **kwargs_120813)
            
            # Assigning a type to the variable 'self_weakref' (line 262)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'self_weakref', weak_self_call_result_120814)
            
            # Type idiom detected: calculating its left and rigth part (line 263)
            # Getting the type of 'self_weakref' (line 263)
            self_weakref_120815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'self_weakref')
            # Getting the type of 'None' (line 263)
            None_120816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 35), 'None')
            
            (may_be_120817, more_types_in_union_120818) = may_not_be_none(self_weakref_120815, None_120816)

            if may_be_120817:

                if more_types_in_union_120818:
                    # Runtime conditional SSA (line 263)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a BinOp to a Attribute (line 264):
                
                # Assigning a BinOp to a Attribute (line 264):
                # Getting the type of 'self_weakref' (line 264)
                self_weakref_120819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 41), 'self_weakref')
                # Obtaining the member '_labelsep_inches' of a type (line 264)
                _labelsep_inches_120820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 41), self_weakref_120819, '_labelsep_inches')
                # Getting the type of 'fig' (line 264)
                fig_120821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 71), 'fig')
                # Obtaining the member 'dpi' of a type (line 264)
                dpi_120822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 71), fig_120821, 'dpi')
                # Applying the binary operator '*' (line 264)
                result_mul_120823 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 41), '*', _labelsep_inches_120820, dpi_120822)
                
                # Getting the type of 'self_weakref' (line 264)
                self_weakref_120824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'self_weakref')
                # Setting the type of the member 'labelsep' of a type (line 264)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 16), self_weakref_120824, 'labelsep', result_mul_120823)
                
                # Assigning a Name to a Attribute (line 265):
                
                # Assigning a Name to a Attribute (line 265):
                # Getting the type of 'False' (line 265)
                False_120825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 44), 'False')
                # Getting the type of 'self_weakref' (line 265)
                self_weakref_120826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'self_weakref')
                # Setting the type of the member '_initialized' of a type (line 265)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 16), self_weakref_120826, '_initialized', False_120825)

                if more_types_in_union_120818:
                    # SSA join for if statement (line 263)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # ################# End of 'on_dpi_change(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'on_dpi_change' in the type store
            # Getting the type of 'stypy_return_type' (line 261)
            stypy_return_type_120827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_120827)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'on_dpi_change'
            return stypy_return_type_120827

        # Assigning a type to the variable 'on_dpi_change' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'on_dpi_change', on_dpi_change)
        
        # Assigning a Call to a Attribute (line 270):
        
        # Assigning a Call to a Attribute (line 270):
        
        # Call to connect(...): (line 270)
        # Processing the call arguments (line 270)
        unicode_120833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 50), 'unicode', u'dpi_changed')
        # Getting the type of 'on_dpi_change' (line 271)
        on_dpi_change_120834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 50), 'on_dpi_change', False)
        # Processing the call keyword arguments (line 270)
        kwargs_120835 = {}
        # Getting the type of 'Q' (line 270)
        Q_120828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'Q', False)
        # Obtaining the member 'ax' of a type (line 270)
        ax_120829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 20), Q_120828, 'ax')
        # Obtaining the member 'figure' of a type (line 270)
        figure_120830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 20), ax_120829, 'figure')
        # Obtaining the member 'callbacks' of a type (line 270)
        callbacks_120831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 20), figure_120830, 'callbacks')
        # Obtaining the member 'connect' of a type (line 270)
        connect_120832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 20), callbacks_120831, 'connect')
        # Calling connect(args, kwargs) (line 270)
        connect_call_result_120836 = invoke(stypy.reporting.localization.Localization(__file__, 270, 20), connect_120832, *[unicode_120833, on_dpi_change_120834], **kwargs_120835)
        
        # Getting the type of 'self' (line 270)
        self_120837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self')
        # Setting the type of the member '_cid' of a type (line 270)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_120837, '_cid', connect_call_result_120836)
        
        # Assigning a Call to a Attribute (line 273):
        
        # Assigning a Call to a Attribute (line 273):
        
        # Call to pop(...): (line 273)
        # Processing the call arguments (line 273)
        unicode_120840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 31), 'unicode', u'labelpos')
        unicode_120841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 43), 'unicode', u'N')
        # Processing the call keyword arguments (line 273)
        kwargs_120842 = {}
        # Getting the type of 'kw' (line 273)
        kw_120838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'kw', False)
        # Obtaining the member 'pop' of a type (line 273)
        pop_120839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 24), kw_120838, 'pop')
        # Calling pop(args, kwargs) (line 273)
        pop_call_result_120843 = invoke(stypy.reporting.localization.Localization(__file__, 273, 24), pop_120839, *[unicode_120840, unicode_120841], **kwargs_120842)
        
        # Getting the type of 'self' (line 273)
        self_120844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'self')
        # Setting the type of the member 'labelpos' of a type (line 273)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), self_120844, 'labelpos', pop_call_result_120843)
        
        # Assigning a Call to a Attribute (line 274):
        
        # Assigning a Call to a Attribute (line 274):
        
        # Call to pop(...): (line 274)
        # Processing the call arguments (line 274)
        unicode_120847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 33), 'unicode', u'labelcolor')
        # Getting the type of 'None' (line 274)
        None_120848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 47), 'None', False)
        # Processing the call keyword arguments (line 274)
        kwargs_120849 = {}
        # Getting the type of 'kw' (line 274)
        kw_120845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 26), 'kw', False)
        # Obtaining the member 'pop' of a type (line 274)
        pop_120846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 26), kw_120845, 'pop')
        # Calling pop(args, kwargs) (line 274)
        pop_call_result_120850 = invoke(stypy.reporting.localization.Localization(__file__, 274, 26), pop_120846, *[unicode_120847, None_120848], **kwargs_120849)
        
        # Getting the type of 'self' (line 274)
        self_120851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'self')
        # Setting the type of the member 'labelcolor' of a type (line 274)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), self_120851, 'labelcolor', pop_call_result_120850)
        
        # Assigning a Call to a Attribute (line 275):
        
        # Assigning a Call to a Attribute (line 275):
        
        # Call to pop(...): (line 275)
        # Processing the call arguments (line 275)
        unicode_120854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 37), 'unicode', u'fontproperties')
        
        # Call to dict(...): (line 275)
        # Processing the call keyword arguments (line 275)
        kwargs_120856 = {}
        # Getting the type of 'dict' (line 275)
        dict_120855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 55), 'dict', False)
        # Calling dict(args, kwargs) (line 275)
        dict_call_result_120857 = invoke(stypy.reporting.localization.Localization(__file__, 275, 55), dict_120855, *[], **kwargs_120856)
        
        # Processing the call keyword arguments (line 275)
        kwargs_120858 = {}
        # Getting the type of 'kw' (line 275)
        kw_120852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 30), 'kw', False)
        # Obtaining the member 'pop' of a type (line 275)
        pop_120853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 30), kw_120852, 'pop')
        # Calling pop(args, kwargs) (line 275)
        pop_call_result_120859 = invoke(stypy.reporting.localization.Localization(__file__, 275, 30), pop_120853, *[unicode_120854, dict_call_result_120857], **kwargs_120858)
        
        # Getting the type of 'self' (line 275)
        self_120860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self')
        # Setting the type of the member 'fontproperties' of a type (line 275)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_120860, 'fontproperties', pop_call_result_120859)
        
        # Assigning a Name to a Attribute (line 276):
        
        # Assigning a Name to a Attribute (line 276):
        # Getting the type of 'kw' (line 276)
        kw_120861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 18), 'kw')
        # Getting the type of 'self' (line 276)
        self_120862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'self')
        # Setting the type of the member 'kw' of a type (line 276)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), self_120862, 'kw', kw_120861)
        
        # Assigning a Attribute to a Name (line 277):
        
        # Assigning a Attribute to a Name (line 277):
        # Getting the type of 'self' (line 277)
        self_120863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 14), 'self')
        # Obtaining the member 'fontproperties' of a type (line 277)
        fontproperties_120864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 14), self_120863, 'fontproperties')
        # Assigning a type to the variable '_fp' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), '_fp', fontproperties_120864)
        
        # Assigning a Call to a Attribute (line 279):
        
        # Assigning a Call to a Attribute (line 279):
        
        # Call to Text(...): (line 279)
        # Processing the call keyword arguments (line 279)
        # Getting the type of 'label' (line 280)
        label_120867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 29), 'label', False)
        keyword_120868 = label_120867
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 281)
        self_120869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 56), 'self', False)
        # Obtaining the member 'labelpos' of a type (line 281)
        labelpos_120870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 56), self_120869, 'labelpos')
        # Getting the type of 'self' (line 281)
        self_120871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 44), 'self', False)
        # Obtaining the member 'halign' of a type (line 281)
        halign_120872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 44), self_120871, 'halign')
        # Obtaining the member '__getitem__' of a type (line 281)
        getitem___120873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 44), halign_120872, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 281)
        subscript_call_result_120874 = invoke(stypy.reporting.localization.Localization(__file__, 281, 44), getitem___120873, labelpos_120870)
        
        keyword_120875 = subscript_call_result_120874
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 282)
        self_120876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 54), 'self', False)
        # Obtaining the member 'labelpos' of a type (line 282)
        labelpos_120877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 54), self_120876, 'labelpos')
        # Getting the type of 'self' (line 282)
        self_120878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 42), 'self', False)
        # Obtaining the member 'valign' of a type (line 282)
        valign_120879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 42), self_120878, 'valign')
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___120880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 42), valign_120879, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_120881 = invoke(stypy.reporting.localization.Localization(__file__, 282, 42), getitem___120880, labelpos_120877)
        
        keyword_120882 = subscript_call_result_120881
        
        # Call to FontProperties(...): (line 283)
        # Processing the call keyword arguments (line 283)
        # Getting the type of '_fp' (line 283)
        _fp_120885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 69), '_fp', False)
        kwargs_120886 = {'_fp_120885': _fp_120885}
        # Getting the type of 'font_manager' (line 283)
        font_manager_120883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 39), 'font_manager', False)
        # Obtaining the member 'FontProperties' of a type (line 283)
        FontProperties_120884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 39), font_manager_120883, 'FontProperties')
        # Calling FontProperties(args, kwargs) (line 283)
        FontProperties_call_result_120887 = invoke(stypy.reporting.localization.Localization(__file__, 283, 39), FontProperties_120884, *[], **kwargs_120886)
        
        keyword_120888 = FontProperties_call_result_120887
        kwargs_120889 = {'text': keyword_120868, 'verticalalignment': keyword_120882, 'horizontalalignment': keyword_120875, 'fontproperties': keyword_120888}
        # Getting the type of 'mtext' (line 279)
        mtext_120865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'mtext', False)
        # Obtaining the member 'Text' of a type (line 279)
        Text_120866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 20), mtext_120865, 'Text')
        # Calling Text(args, kwargs) (line 279)
        Text_call_result_120890 = invoke(stypy.reporting.localization.Localization(__file__, 279, 20), Text_120866, *[], **kwargs_120889)
        
        # Getting the type of 'self' (line 279)
        self_120891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'self')
        # Setting the type of the member 'text' of a type (line 279)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), self_120891, 'text', Text_call_result_120890)
        
        
        # Getting the type of 'self' (line 285)
        self_120892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'self')
        # Obtaining the member 'labelcolor' of a type (line 285)
        labelcolor_120893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 11), self_120892, 'labelcolor')
        # Getting the type of 'None' (line 285)
        None_120894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 34), 'None')
        # Applying the binary operator 'isnot' (line 285)
        result_is_not_120895 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 11), 'isnot', labelcolor_120893, None_120894)
        
        # Testing the type of an if condition (line 285)
        if_condition_120896 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 8), result_is_not_120895)
        # Assigning a type to the variable 'if_condition_120896' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'if_condition_120896', if_condition_120896)
        # SSA begins for if statement (line 285)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_color(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'self' (line 286)
        self_120900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 32), 'self', False)
        # Obtaining the member 'labelcolor' of a type (line 286)
        labelcolor_120901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 32), self_120900, 'labelcolor')
        # Processing the call keyword arguments (line 286)
        kwargs_120902 = {}
        # Getting the type of 'self' (line 286)
        self_120897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'self', False)
        # Obtaining the member 'text' of a type (line 286)
        text_120898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 12), self_120897, 'text')
        # Obtaining the member 'set_color' of a type (line 286)
        set_color_120899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 12), text_120898, 'set_color')
        # Calling set_color(args, kwargs) (line 286)
        set_color_call_result_120903 = invoke(stypy.reporting.localization.Localization(__file__, 286, 12), set_color_120899, *[labelcolor_120901], **kwargs_120902)
        
        # SSA join for if statement (line 285)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 287):
        
        # Assigning a Name to a Attribute (line 287):
        # Getting the type of 'False' (line 287)
        False_120904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 28), 'False')
        # Getting the type of 'self' (line 287)
        self_120905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self')
        # Setting the type of the member '_initialized' of a type (line 287)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_120905, '_initialized', False_120904)
        
        # Assigning a BinOp to a Attribute (line 288):
        
        # Assigning a BinOp to a Attribute (line 288):
        # Getting the type of 'Q' (line 288)
        Q_120906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 22), 'Q')
        # Obtaining the member 'zorder' of a type (line 288)
        zorder_120907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 22), Q_120906, 'zorder')
        float_120908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 33), 'float')
        # Applying the binary operator '+' (line 288)
        result_add_120909 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 22), '+', zorder_120907, float_120908)
        
        # Getting the type of 'self' (line 288)
        self_120910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'self')
        # Setting the type of the member 'zorder' of a type (line 288)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), self_120910, 'zorder', result_add_120909)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def remove(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove'
        module_type_store = module_type_store.open_function_context('remove', 290, 4, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        QuiverKey.remove.__dict__.__setitem__('stypy_localization', localization)
        QuiverKey.remove.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        QuiverKey.remove.__dict__.__setitem__('stypy_type_store', module_type_store)
        QuiverKey.remove.__dict__.__setitem__('stypy_function_name', 'QuiverKey.remove')
        QuiverKey.remove.__dict__.__setitem__('stypy_param_names_list', [])
        QuiverKey.remove.__dict__.__setitem__('stypy_varargs_param_name', None)
        QuiverKey.remove.__dict__.__setitem__('stypy_kwargs_param_name', None)
        QuiverKey.remove.__dict__.__setitem__('stypy_call_defaults', defaults)
        QuiverKey.remove.__dict__.__setitem__('stypy_call_varargs', varargs)
        QuiverKey.remove.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        QuiverKey.remove.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'QuiverKey.remove', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove(...)' code ##################

        unicode_120911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, (-1)), 'unicode', u'\n        Overload the remove method\n        ')
        
        # Call to disconnect(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'self' (line 294)
        self_120918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 46), 'self', False)
        # Obtaining the member '_cid' of a type (line 294)
        _cid_120919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 46), self_120918, '_cid')
        # Processing the call keyword arguments (line 294)
        kwargs_120920 = {}
        # Getting the type of 'self' (line 294)
        self_120912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'self', False)
        # Obtaining the member 'Q' of a type (line 294)
        Q_120913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), self_120912, 'Q')
        # Obtaining the member 'ax' of a type (line 294)
        ax_120914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), Q_120913, 'ax')
        # Obtaining the member 'figure' of a type (line 294)
        figure_120915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), ax_120914, 'figure')
        # Obtaining the member 'callbacks' of a type (line 294)
        callbacks_120916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), figure_120915, 'callbacks')
        # Obtaining the member 'disconnect' of a type (line 294)
        disconnect_120917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), callbacks_120916, 'disconnect')
        # Calling disconnect(args, kwargs) (line 294)
        disconnect_call_result_120921 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), disconnect_120917, *[_cid_120919], **kwargs_120920)
        
        
        # Assigning a Name to a Attribute (line 295):
        
        # Assigning a Name to a Attribute (line 295):
        # Getting the type of 'None' (line 295)
        None_120922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'None')
        # Getting the type of 'self' (line 295)
        self_120923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'self')
        # Setting the type of the member '_cid' of a type (line 295)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), self_120923, '_cid', None_120922)
        
        # Call to remove(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'self' (line 297)
        self_120927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 30), 'self', False)
        # Processing the call keyword arguments (line 297)
        kwargs_120928 = {}
        # Getting the type of 'martist' (line 297)
        martist_120924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'martist', False)
        # Obtaining the member 'Artist' of a type (line 297)
        Artist_120925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), martist_120924, 'Artist')
        # Obtaining the member 'remove' of a type (line 297)
        remove_120926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), Artist_120925, 'remove')
        # Calling remove(args, kwargs) (line 297)
        remove_call_result_120929 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), remove_120926, *[self_120927], **kwargs_120928)
        
        
        # ################# End of 'remove(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_120930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_120930)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove'
        return stypy_return_type_120930

    
    # Assigning a Name to a Attribute (line 299):

    @norecursion
    def _init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init'
        module_type_store = module_type_store.open_function_context('_init', 301, 4, False)
        # Assigning a type to the variable 'self' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        QuiverKey._init.__dict__.__setitem__('stypy_localization', localization)
        QuiverKey._init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        QuiverKey._init.__dict__.__setitem__('stypy_type_store', module_type_store)
        QuiverKey._init.__dict__.__setitem__('stypy_function_name', 'QuiverKey._init')
        QuiverKey._init.__dict__.__setitem__('stypy_param_names_list', [])
        QuiverKey._init.__dict__.__setitem__('stypy_varargs_param_name', None)
        QuiverKey._init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        QuiverKey._init.__dict__.__setitem__('stypy_call_defaults', defaults)
        QuiverKey._init.__dict__.__setitem__('stypy_call_varargs', varargs)
        QuiverKey._init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        QuiverKey._init.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'QuiverKey._init', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init(...)' code ##################

        
        # Getting the type of 'True' (line 302)
        True_120931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 11), 'True')
        # Testing the type of an if condition (line 302)
        if_condition_120932 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 8), True_120931)
        # Assigning a type to the variable 'if_condition_120932' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'if_condition_120932', if_condition_120932)
        # SSA begins for if statement (line 302)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 303)
        self_120933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 19), 'self')
        # Obtaining the member 'Q' of a type (line 303)
        Q_120934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 19), self_120933, 'Q')
        # Obtaining the member '_initialized' of a type (line 303)
        _initialized_120935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 19), Q_120934, '_initialized')
        # Applying the 'not' unary operator (line 303)
        result_not__120936 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 15), 'not', _initialized_120935)
        
        # Testing the type of an if condition (line 303)
        if_condition_120937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 12), result_not__120936)
        # Assigning a type to the variable 'if_condition_120937' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'if_condition_120937', if_condition_120937)
        # SSA begins for if statement (line 303)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _init(...): (line 304)
        # Processing the call keyword arguments (line 304)
        kwargs_120941 = {}
        # Getting the type of 'self' (line 304)
        self_120938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'self', False)
        # Obtaining the member 'Q' of a type (line 304)
        Q_120939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 16), self_120938, 'Q')
        # Obtaining the member '_init' of a type (line 304)
        _init_120940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 16), Q_120939, '_init')
        # Calling _init(args, kwargs) (line 304)
        _init_call_result_120942 = invoke(stypy.reporting.localization.Localization(__file__, 304, 16), _init_120940, *[], **kwargs_120941)
        
        # SSA join for if statement (line 303)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _set_transform(...): (line 305)
        # Processing the call keyword arguments (line 305)
        kwargs_120945 = {}
        # Getting the type of 'self' (line 305)
        self_120943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'self', False)
        # Obtaining the member '_set_transform' of a type (line 305)
        _set_transform_120944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 12), self_120943, '_set_transform')
        # Calling _set_transform(args, kwargs) (line 305)
        _set_transform_call_result_120946 = invoke(stypy.reporting.localization.Localization(__file__, 305, 12), _set_transform_120944, *[], **kwargs_120945)
        
        
        # Assigning a Attribute to a Name (line 306):
        
        # Assigning a Attribute to a Name (line 306):
        # Getting the type of 'self' (line 306)
        self_120947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 21), 'self')
        # Obtaining the member 'Q' of a type (line 306)
        Q_120948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 21), self_120947, 'Q')
        # Obtaining the member 'pivot' of a type (line 306)
        pivot_120949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 21), Q_120948, 'pivot')
        # Assigning a type to the variable '_pivot' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), '_pivot', pivot_120949)
        
        # Assigning a Subscript to a Attribute (line 307):
        
        # Assigning a Subscript to a Attribute (line 307):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 307)
        self_120950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 38), 'self')
        # Obtaining the member 'labelpos' of a type (line 307)
        labelpos_120951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 38), self_120950, 'labelpos')
        # Getting the type of 'self' (line 307)
        self_120952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 27), 'self')
        # Obtaining the member 'pivot' of a type (line 307)
        pivot_120953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 27), self_120952, 'pivot')
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___120954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 27), pivot_120953, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_120955 = invoke(stypy.reporting.localization.Localization(__file__, 307, 27), getitem___120954, labelpos_120951)
        
        # Getting the type of 'self' (line 307)
        self_120956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'self')
        # Obtaining the member 'Q' of a type (line 307)
        Q_120957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), self_120956, 'Q')
        # Setting the type of the member 'pivot' of a type (line 307)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), Q_120957, 'pivot', subscript_call_result_120955)
        
        # Assigning a Attribute to a Name (line 309):
        
        # Assigning a Attribute to a Name (line 309):
        # Getting the type of 'self' (line 309)
        self_120958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 20), 'self')
        # Obtaining the member 'Q' of a type (line 309)
        Q_120959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 20), self_120958, 'Q')
        # Obtaining the member 'Umask' of a type (line 309)
        Umask_120960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 20), Q_120959, 'Umask')
        # Assigning a type to the variable '_mask' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), '_mask', Umask_120960)
        
        # Assigning a Attribute to a Attribute (line 310):
        
        # Assigning a Attribute to a Attribute (line 310):
        # Getting the type of 'ma' (line 310)
        ma_120961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 27), 'ma')
        # Obtaining the member 'nomask' of a type (line 310)
        nomask_120962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 27), ma_120961, 'nomask')
        # Getting the type of 'self' (line 310)
        self_120963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'self')
        # Obtaining the member 'Q' of a type (line 310)
        Q_120964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 12), self_120963, 'Q')
        # Setting the type of the member 'Umask' of a type (line 310)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 12), Q_120964, 'Umask', nomask_120962)
        
        # Assigning a Call to a Attribute (line 311):
        
        # Assigning a Call to a Attribute (line 311):
        
        # Call to _make_verts(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Call to array(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_120970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        # Getting the type of 'self' (line 311)
        self_120971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 54), 'self', False)
        # Obtaining the member 'U' of a type (line 311)
        U_120972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 54), self_120971, 'U')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 53), list_120970, U_120972)
        
        # Processing the call keyword arguments (line 311)
        kwargs_120973 = {}
        # Getting the type of 'np' (line 311)
        np_120968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 311)
        array_120969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 44), np_120968, 'array')
        # Calling array(args, kwargs) (line 311)
        array_call_result_120974 = invoke(stypy.reporting.localization.Localization(__file__, 311, 44), array_120969, *[list_120970], **kwargs_120973)
        
        
        # Call to zeros(...): (line 312)
        # Processing the call arguments (line 312)
        
        # Obtaining an instance of the builtin type 'tuple' (line 312)
        tuple_120977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 312)
        # Adding element type (line 312)
        int_120978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 54), tuple_120977, int_120978)
        
        # Processing the call keyword arguments (line 312)
        kwargs_120979 = {}
        # Getting the type of 'np' (line 312)
        np_120975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 44), 'np', False)
        # Obtaining the member 'zeros' of a type (line 312)
        zeros_120976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 44), np_120975, 'zeros')
        # Calling zeros(args, kwargs) (line 312)
        zeros_call_result_120980 = invoke(stypy.reporting.localization.Localization(__file__, 312, 44), zeros_120976, *[tuple_120977], **kwargs_120979)
        
        # Getting the type of 'self' (line 313)
        self_120981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 44), 'self', False)
        # Obtaining the member 'angle' of a type (line 313)
        angle_120982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 44), self_120981, 'angle')
        # Processing the call keyword arguments (line 311)
        kwargs_120983 = {}
        # Getting the type of 'self' (line 311)
        self_120965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 25), 'self', False)
        # Obtaining the member 'Q' of a type (line 311)
        Q_120966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 25), self_120965, 'Q')
        # Obtaining the member '_make_verts' of a type (line 311)
        _make_verts_120967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 25), Q_120966, '_make_verts')
        # Calling _make_verts(args, kwargs) (line 311)
        _make_verts_call_result_120984 = invoke(stypy.reporting.localization.Localization(__file__, 311, 25), _make_verts_120967, *[array_call_result_120974, zeros_call_result_120980, angle_120982], **kwargs_120983)
        
        # Getting the type of 'self' (line 311)
        self_120985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'self')
        # Setting the type of the member 'verts' of a type (line 311)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), self_120985, 'verts', _make_verts_call_result_120984)
        
        # Assigning a Name to a Attribute (line 314):
        
        # Assigning a Name to a Attribute (line 314):
        # Getting the type of '_mask' (line 314)
        _mask_120986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 27), '_mask')
        # Getting the type of 'self' (line 314)
        self_120987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'self')
        # Obtaining the member 'Q' of a type (line 314)
        Q_120988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 12), self_120987, 'Q')
        # Setting the type of the member 'Umask' of a type (line 314)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 12), Q_120988, 'Umask', _mask_120986)
        
        # Assigning a Name to a Attribute (line 315):
        
        # Assigning a Name to a Attribute (line 315):
        # Getting the type of '_pivot' (line 315)
        _pivot_120989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 27), '_pivot')
        # Getting the type of 'self' (line 315)
        self_120990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'self')
        # Obtaining the member 'Q' of a type (line 315)
        Q_120991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), self_120990, 'Q')
        # Setting the type of the member 'pivot' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), Q_120991, 'pivot', _pivot_120989)
        
        # Assigning a Attribute to a Name (line 316):
        
        # Assigning a Attribute to a Name (line 316):
        # Getting the type of 'self' (line 316)
        self_120992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 17), 'self')
        # Obtaining the member 'Q' of a type (line 316)
        Q_120993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 17), self_120992, 'Q')
        # Obtaining the member 'polykw' of a type (line 316)
        polykw_120994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 17), Q_120993, 'polykw')
        # Assigning a type to the variable 'kw' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'kw', polykw_120994)
        
        # Call to update(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'self' (line 317)
        self_120997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 22), 'self', False)
        # Obtaining the member 'kw' of a type (line 317)
        kw_120998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 22), self_120997, 'kw')
        # Processing the call keyword arguments (line 317)
        kwargs_120999 = {}
        # Getting the type of 'kw' (line 317)
        kw_120995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'kw', False)
        # Obtaining the member 'update' of a type (line 317)
        update_120996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), kw_120995, 'update')
        # Calling update(args, kwargs) (line 317)
        update_call_result_121000 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), update_120996, *[kw_120998], **kwargs_120999)
        
        
        # Assigning a Call to a Attribute (line 318):
        
        # Assigning a Call to a Attribute (line 318):
        
        # Call to PolyCollection(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'self' (line 319)
        self_121003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 40), 'self', False)
        # Obtaining the member 'verts' of a type (line 319)
        verts_121004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 40), self_121003, 'verts')
        # Processing the call keyword arguments (line 318)
        
        # Obtaining an instance of the builtin type 'list' (line 320)
        list_121005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 320)
        # Adding element type (line 320)
        
        # Obtaining an instance of the builtin type 'tuple' (line 320)
        tuple_121006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 320)
        # Adding element type (line 320)
        # Getting the type of 'self' (line 320)
        self_121007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 50), 'self', False)
        # Obtaining the member 'X' of a type (line 320)
        X_121008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 50), self_121007, 'X')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 50), tuple_121006, X_121008)
        # Adding element type (line 320)
        # Getting the type of 'self' (line 320)
        self_121009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 58), 'self', False)
        # Obtaining the member 'Y' of a type (line 320)
        Y_121010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 58), self_121009, 'Y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 50), tuple_121006, Y_121010)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 48), list_121005, tuple_121006)
        
        keyword_121011 = list_121005
        
        # Call to get_transform(...): (line 321)
        # Processing the call keyword arguments (line 321)
        kwargs_121014 = {}
        # Getting the type of 'self' (line 321)
        self_121012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 52), 'self', False)
        # Obtaining the member 'get_transform' of a type (line 321)
        get_transform_121013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 52), self_121012, 'get_transform')
        # Calling get_transform(args, kwargs) (line 321)
        get_transform_call_result_121015 = invoke(stypy.reporting.localization.Localization(__file__, 321, 52), get_transform_121013, *[], **kwargs_121014)
        
        keyword_121016 = get_transform_call_result_121015
        # Getting the type of 'kw' (line 322)
        kw_121017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 42), 'kw', False)
        kwargs_121018 = {'transOffset': keyword_121016, 'kw_121017': kw_121017, 'offsets': keyword_121011}
        # Getting the type of 'mcollections' (line 318)
        mcollections_121001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 26), 'mcollections', False)
        # Obtaining the member 'PolyCollection' of a type (line 318)
        PolyCollection_121002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 26), mcollections_121001, 'PolyCollection')
        # Calling PolyCollection(args, kwargs) (line 318)
        PolyCollection_call_result_121019 = invoke(stypy.reporting.localization.Localization(__file__, 318, 26), PolyCollection_121002, *[verts_121004], **kwargs_121018)
        
        # Getting the type of 'self' (line 318)
        self_121020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'self')
        # Setting the type of the member 'vector' of a type (line 318)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 12), self_121020, 'vector', PolyCollection_call_result_121019)
        
        
        # Getting the type of 'self' (line 323)
        self_121021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'self')
        # Obtaining the member 'color' of a type (line 323)
        color_121022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 15), self_121021, 'color')
        # Getting the type of 'None' (line 323)
        None_121023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 33), 'None')
        # Applying the binary operator 'isnot' (line 323)
        result_is_not_121024 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 15), 'isnot', color_121022, None_121023)
        
        # Testing the type of an if condition (line 323)
        if_condition_121025 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 12), result_is_not_121024)
        # Assigning a type to the variable 'if_condition_121025' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'if_condition_121025', if_condition_121025)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_color(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'self' (line 324)
        self_121029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 38), 'self', False)
        # Obtaining the member 'color' of a type (line 324)
        color_121030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 38), self_121029, 'color')
        # Processing the call keyword arguments (line 324)
        kwargs_121031 = {}
        # Getting the type of 'self' (line 324)
        self_121026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 16), 'self', False)
        # Obtaining the member 'vector' of a type (line 324)
        vector_121027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 16), self_121026, 'vector')
        # Obtaining the member 'set_color' of a type (line 324)
        set_color_121028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 16), vector_121027, 'set_color')
        # Calling set_color(args, kwargs) (line 324)
        set_color_call_result_121032 = invoke(stypy.reporting.localization.Localization(__file__, 324, 16), set_color_121028, *[color_121030], **kwargs_121031)
        
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_transform(...): (line 325)
        # Processing the call arguments (line 325)
        
        # Call to get_transform(...): (line 325)
        # Processing the call keyword arguments (line 325)
        kwargs_121039 = {}
        # Getting the type of 'self' (line 325)
        self_121036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 38), 'self', False)
        # Obtaining the member 'Q' of a type (line 325)
        Q_121037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 38), self_121036, 'Q')
        # Obtaining the member 'get_transform' of a type (line 325)
        get_transform_121038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 38), Q_121037, 'get_transform')
        # Calling get_transform(args, kwargs) (line 325)
        get_transform_call_result_121040 = invoke(stypy.reporting.localization.Localization(__file__, 325, 38), get_transform_121038, *[], **kwargs_121039)
        
        # Processing the call keyword arguments (line 325)
        kwargs_121041 = {}
        # Getting the type of 'self' (line 325)
        self_121033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'self', False)
        # Obtaining the member 'vector' of a type (line 325)
        vector_121034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), self_121033, 'vector')
        # Obtaining the member 'set_transform' of a type (line 325)
        set_transform_121035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), vector_121034, 'set_transform')
        # Calling set_transform(args, kwargs) (line 325)
        set_transform_call_result_121042 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), set_transform_121035, *[get_transform_call_result_121040], **kwargs_121041)
        
        
        # Call to set_figure(...): (line 326)
        # Processing the call arguments (line 326)
        
        # Call to get_figure(...): (line 326)
        # Processing the call keyword arguments (line 326)
        kwargs_121048 = {}
        # Getting the type of 'self' (line 326)
        self_121046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 35), 'self', False)
        # Obtaining the member 'get_figure' of a type (line 326)
        get_figure_121047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 35), self_121046, 'get_figure')
        # Calling get_figure(args, kwargs) (line 326)
        get_figure_call_result_121049 = invoke(stypy.reporting.localization.Localization(__file__, 326, 35), get_figure_121047, *[], **kwargs_121048)
        
        # Processing the call keyword arguments (line 326)
        kwargs_121050 = {}
        # Getting the type of 'self' (line 326)
        self_121043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'self', False)
        # Obtaining the member 'vector' of a type (line 326)
        vector_121044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 12), self_121043, 'vector')
        # Obtaining the member 'set_figure' of a type (line 326)
        set_figure_121045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 12), vector_121044, 'set_figure')
        # Calling set_figure(args, kwargs) (line 326)
        set_figure_call_result_121051 = invoke(stypy.reporting.localization.Localization(__file__, 326, 12), set_figure_121045, *[get_figure_call_result_121049], **kwargs_121050)
        
        
        # Assigning a Name to a Attribute (line 327):
        
        # Assigning a Name to a Attribute (line 327):
        # Getting the type of 'True' (line 327)
        True_121052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 32), 'True')
        # Getting the type of 'self' (line 327)
        self_121053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'self')
        # Setting the type of the member '_initialized' of a type (line 327)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 12), self_121053, '_initialized', True_121052)
        # SSA join for if statement (line 302)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init' in the type store
        # Getting the type of 'stypy_return_type' (line 301)
        stypy_return_type_121054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init'
        return stypy_return_type_121054


    @norecursion
    def _text_x(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_text_x'
        module_type_store = module_type_store.open_function_context('_text_x', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        QuiverKey._text_x.__dict__.__setitem__('stypy_localization', localization)
        QuiverKey._text_x.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        QuiverKey._text_x.__dict__.__setitem__('stypy_type_store', module_type_store)
        QuiverKey._text_x.__dict__.__setitem__('stypy_function_name', 'QuiverKey._text_x')
        QuiverKey._text_x.__dict__.__setitem__('stypy_param_names_list', ['x'])
        QuiverKey._text_x.__dict__.__setitem__('stypy_varargs_param_name', None)
        QuiverKey._text_x.__dict__.__setitem__('stypy_kwargs_param_name', None)
        QuiverKey._text_x.__dict__.__setitem__('stypy_call_defaults', defaults)
        QuiverKey._text_x.__dict__.__setitem__('stypy_call_varargs', varargs)
        QuiverKey._text_x.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        QuiverKey._text_x.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'QuiverKey._text_x', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_text_x', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_text_x(...)' code ##################

        
        
        # Getting the type of 'self' (line 330)
        self_121055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'self')
        # Obtaining the member 'labelpos' of a type (line 330)
        labelpos_121056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 11), self_121055, 'labelpos')
        unicode_121057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 28), 'unicode', u'E')
        # Applying the binary operator '==' (line 330)
        result_eq_121058 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 11), '==', labelpos_121056, unicode_121057)
        
        # Testing the type of an if condition (line 330)
        if_condition_121059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 8), result_eq_121058)
        # Assigning a type to the variable 'if_condition_121059' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'if_condition_121059', if_condition_121059)
        # SSA begins for if statement (line 330)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'x' (line 331)
        x_121060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 19), 'x')
        # Getting the type of 'self' (line 331)
        self_121061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 23), 'self')
        # Obtaining the member 'labelsep' of a type (line 331)
        labelsep_121062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 23), self_121061, 'labelsep')
        # Applying the binary operator '+' (line 331)
        result_add_121063 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 19), '+', x_121060, labelsep_121062)
        
        # Assigning a type to the variable 'stypy_return_type' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'stypy_return_type', result_add_121063)
        # SSA branch for the else part of an if statement (line 330)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 332)
        self_121064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 13), 'self')
        # Obtaining the member 'labelpos' of a type (line 332)
        labelpos_121065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 13), self_121064, 'labelpos')
        unicode_121066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 30), 'unicode', u'W')
        # Applying the binary operator '==' (line 332)
        result_eq_121067 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 13), '==', labelpos_121065, unicode_121066)
        
        # Testing the type of an if condition (line 332)
        if_condition_121068 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 13), result_eq_121067)
        # Assigning a type to the variable 'if_condition_121068' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 13), 'if_condition_121068', if_condition_121068)
        # SSA begins for if statement (line 332)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'x' (line 333)
        x_121069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 19), 'x')
        # Getting the type of 'self' (line 333)
        self_121070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 23), 'self')
        # Obtaining the member 'labelsep' of a type (line 333)
        labelsep_121071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 23), self_121070, 'labelsep')
        # Applying the binary operator '-' (line 333)
        result_sub_121072 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 19), '-', x_121069, labelsep_121071)
        
        # Assigning a type to the variable 'stypy_return_type' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'stypy_return_type', result_sub_121072)
        # SSA branch for the else part of an if statement (line 332)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'x' (line 335)
        x_121073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'stypy_return_type', x_121073)
        # SSA join for if statement (line 332)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 330)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_text_x(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_text_x' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_121074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121074)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_text_x'
        return stypy_return_type_121074


    @norecursion
    def _text_y(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_text_y'
        module_type_store = module_type_store.open_function_context('_text_y', 337, 4, False)
        # Assigning a type to the variable 'self' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        QuiverKey._text_y.__dict__.__setitem__('stypy_localization', localization)
        QuiverKey._text_y.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        QuiverKey._text_y.__dict__.__setitem__('stypy_type_store', module_type_store)
        QuiverKey._text_y.__dict__.__setitem__('stypy_function_name', 'QuiverKey._text_y')
        QuiverKey._text_y.__dict__.__setitem__('stypy_param_names_list', ['y'])
        QuiverKey._text_y.__dict__.__setitem__('stypy_varargs_param_name', None)
        QuiverKey._text_y.__dict__.__setitem__('stypy_kwargs_param_name', None)
        QuiverKey._text_y.__dict__.__setitem__('stypy_call_defaults', defaults)
        QuiverKey._text_y.__dict__.__setitem__('stypy_call_varargs', varargs)
        QuiverKey._text_y.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        QuiverKey._text_y.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'QuiverKey._text_y', ['y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_text_y', localization, ['y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_text_y(...)' code ##################

        
        
        # Getting the type of 'self' (line 338)
        self_121075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 11), 'self')
        # Obtaining the member 'labelpos' of a type (line 338)
        labelpos_121076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 11), self_121075, 'labelpos')
        unicode_121077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 28), 'unicode', u'N')
        # Applying the binary operator '==' (line 338)
        result_eq_121078 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 11), '==', labelpos_121076, unicode_121077)
        
        # Testing the type of an if condition (line 338)
        if_condition_121079 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 8), result_eq_121078)
        # Assigning a type to the variable 'if_condition_121079' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'if_condition_121079', if_condition_121079)
        # SSA begins for if statement (line 338)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'y' (line 339)
        y_121080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'y')
        # Getting the type of 'self' (line 339)
        self_121081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 23), 'self')
        # Obtaining the member 'labelsep' of a type (line 339)
        labelsep_121082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 23), self_121081, 'labelsep')
        # Applying the binary operator '+' (line 339)
        result_add_121083 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 19), '+', y_121080, labelsep_121082)
        
        # Assigning a type to the variable 'stypy_return_type' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'stypy_return_type', result_add_121083)
        # SSA branch for the else part of an if statement (line 338)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 340)
        self_121084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'self')
        # Obtaining the member 'labelpos' of a type (line 340)
        labelpos_121085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 13), self_121084, 'labelpos')
        unicode_121086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 30), 'unicode', u'S')
        # Applying the binary operator '==' (line 340)
        result_eq_121087 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 13), '==', labelpos_121085, unicode_121086)
        
        # Testing the type of an if condition (line 340)
        if_condition_121088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 13), result_eq_121087)
        # Assigning a type to the variable 'if_condition_121088' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'if_condition_121088', if_condition_121088)
        # SSA begins for if statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'y' (line 341)
        y_121089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 19), 'y')
        # Getting the type of 'self' (line 341)
        self_121090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 23), 'self')
        # Obtaining the member 'labelsep' of a type (line 341)
        labelsep_121091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 23), self_121090, 'labelsep')
        # Applying the binary operator '-' (line 341)
        result_sub_121092 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 19), '-', y_121089, labelsep_121091)
        
        # Assigning a type to the variable 'stypy_return_type' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'stypy_return_type', result_sub_121092)
        # SSA branch for the else part of an if statement (line 340)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'y' (line 343)
        y_121093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'stypy_return_type', y_121093)
        # SSA join for if statement (line 340)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 338)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_text_y(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_text_y' in the type store
        # Getting the type of 'stypy_return_type' (line 337)
        stypy_return_type_121094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121094)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_text_y'
        return stypy_return_type_121094


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 345, 4, False)
        # Assigning a type to the variable 'self' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        QuiverKey.draw.__dict__.__setitem__('stypy_localization', localization)
        QuiverKey.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        QuiverKey.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        QuiverKey.draw.__dict__.__setitem__('stypy_function_name', 'QuiverKey.draw')
        QuiverKey.draw.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        QuiverKey.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        QuiverKey.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        QuiverKey.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        QuiverKey.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        QuiverKey.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        QuiverKey.draw.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'QuiverKey.draw', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw(...)' code ##################

        
        # Call to _init(...): (line 347)
        # Processing the call keyword arguments (line 347)
        kwargs_121097 = {}
        # Getting the type of 'self' (line 347)
        self_121095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'self', False)
        # Obtaining the member '_init' of a type (line 347)
        _init_121096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), self_121095, '_init')
        # Calling _init(args, kwargs) (line 347)
        _init_call_result_121098 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), _init_121096, *[], **kwargs_121097)
        
        
        # Call to draw(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'renderer' (line 348)
        renderer_121102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 25), 'renderer', False)
        # Processing the call keyword arguments (line 348)
        kwargs_121103 = {}
        # Getting the type of 'self' (line 348)
        self_121099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'self', False)
        # Obtaining the member 'vector' of a type (line 348)
        vector_121100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), self_121099, 'vector')
        # Obtaining the member 'draw' of a type (line 348)
        draw_121101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), vector_121100, 'draw')
        # Calling draw(args, kwargs) (line 348)
        draw_call_result_121104 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), draw_121101, *[renderer_121102], **kwargs_121103)
        
        
        # Assigning a Call to a Tuple (line 349):
        
        # Assigning a Call to a Name:
        
        # Call to transform_point(...): (line 349)
        # Processing the call arguments (line 349)
        
        # Obtaining an instance of the builtin type 'tuple' (line 349)
        tuple_121110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 349)
        # Adding element type (line 349)
        # Getting the type of 'self' (line 349)
        self_121111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 53), 'self', False)
        # Obtaining the member 'X' of a type (line 349)
        X_121112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 53), self_121111, 'X')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 53), tuple_121110, X_121112)
        # Adding element type (line 349)
        # Getting the type of 'self' (line 349)
        self_121113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 61), 'self', False)
        # Obtaining the member 'Y' of a type (line 349)
        Y_121114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 61), self_121113, 'Y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 53), tuple_121110, Y_121114)
        
        # Processing the call keyword arguments (line 349)
        kwargs_121115 = {}
        
        # Call to get_transform(...): (line 349)
        # Processing the call keyword arguments (line 349)
        kwargs_121107 = {}
        # Getting the type of 'self' (line 349)
        self_121105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'self', False)
        # Obtaining the member 'get_transform' of a type (line 349)
        get_transform_121106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 15), self_121105, 'get_transform')
        # Calling get_transform(args, kwargs) (line 349)
        get_transform_call_result_121108 = invoke(stypy.reporting.localization.Localization(__file__, 349, 15), get_transform_121106, *[], **kwargs_121107)
        
        # Obtaining the member 'transform_point' of a type (line 349)
        transform_point_121109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 15), get_transform_call_result_121108, 'transform_point')
        # Calling transform_point(args, kwargs) (line 349)
        transform_point_call_result_121116 = invoke(stypy.reporting.localization.Localization(__file__, 349, 15), transform_point_121109, *[tuple_121110], **kwargs_121115)
        
        # Assigning a type to the variable 'call_assignment_120656' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'call_assignment_120656', transform_point_call_result_121116)
        
        # Assigning a Call to a Name (line 349):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_121119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 8), 'int')
        # Processing the call keyword arguments
        kwargs_121120 = {}
        # Getting the type of 'call_assignment_120656' (line 349)
        call_assignment_120656_121117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'call_assignment_120656', False)
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___121118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), call_assignment_120656_121117, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_121121 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121118, *[int_121119], **kwargs_121120)
        
        # Assigning a type to the variable 'call_assignment_120657' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'call_assignment_120657', getitem___call_result_121121)
        
        # Assigning a Name to a Name (line 349):
        # Getting the type of 'call_assignment_120657' (line 349)
        call_assignment_120657_121122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'call_assignment_120657')
        # Assigning a type to the variable 'x' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'x', call_assignment_120657_121122)
        
        # Assigning a Call to a Name (line 349):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_121125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 8), 'int')
        # Processing the call keyword arguments
        kwargs_121126 = {}
        # Getting the type of 'call_assignment_120656' (line 349)
        call_assignment_120656_121123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'call_assignment_120656', False)
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___121124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), call_assignment_120656_121123, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_121127 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121124, *[int_121125], **kwargs_121126)
        
        # Assigning a type to the variable 'call_assignment_120658' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'call_assignment_120658', getitem___call_result_121127)
        
        # Assigning a Name to a Name (line 349):
        # Getting the type of 'call_assignment_120658' (line 349)
        call_assignment_120658_121128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'call_assignment_120658')
        # Assigning a type to the variable 'y' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 11), 'y', call_assignment_120658_121128)
        
        # Call to set_x(...): (line 350)
        # Processing the call arguments (line 350)
        
        # Call to _text_x(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'x' (line 350)
        x_121134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 37), 'x', False)
        # Processing the call keyword arguments (line 350)
        kwargs_121135 = {}
        # Getting the type of 'self' (line 350)
        self_121132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 24), 'self', False)
        # Obtaining the member '_text_x' of a type (line 350)
        _text_x_121133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 24), self_121132, '_text_x')
        # Calling _text_x(args, kwargs) (line 350)
        _text_x_call_result_121136 = invoke(stypy.reporting.localization.Localization(__file__, 350, 24), _text_x_121133, *[x_121134], **kwargs_121135)
        
        # Processing the call keyword arguments (line 350)
        kwargs_121137 = {}
        # Getting the type of 'self' (line 350)
        self_121129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'self', False)
        # Obtaining the member 'text' of a type (line 350)
        text_121130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), self_121129, 'text')
        # Obtaining the member 'set_x' of a type (line 350)
        set_x_121131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), text_121130, 'set_x')
        # Calling set_x(args, kwargs) (line 350)
        set_x_call_result_121138 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), set_x_121131, *[_text_x_call_result_121136], **kwargs_121137)
        
        
        # Call to set_y(...): (line 351)
        # Processing the call arguments (line 351)
        
        # Call to _text_y(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'y' (line 351)
        y_121144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 37), 'y', False)
        # Processing the call keyword arguments (line 351)
        kwargs_121145 = {}
        # Getting the type of 'self' (line 351)
        self_121142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'self', False)
        # Obtaining the member '_text_y' of a type (line 351)
        _text_y_121143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 24), self_121142, '_text_y')
        # Calling _text_y(args, kwargs) (line 351)
        _text_y_call_result_121146 = invoke(stypy.reporting.localization.Localization(__file__, 351, 24), _text_y_121143, *[y_121144], **kwargs_121145)
        
        # Processing the call keyword arguments (line 351)
        kwargs_121147 = {}
        # Getting the type of 'self' (line 351)
        self_121139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'self', False)
        # Obtaining the member 'text' of a type (line 351)
        text_121140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), self_121139, 'text')
        # Obtaining the member 'set_y' of a type (line 351)
        set_y_121141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), text_121140, 'set_y')
        # Calling set_y(args, kwargs) (line 351)
        set_y_call_result_121148 = invoke(stypy.reporting.localization.Localization(__file__, 351, 8), set_y_121141, *[_text_y_call_result_121146], **kwargs_121147)
        
        
        # Call to draw(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'renderer' (line 352)
        renderer_121152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 23), 'renderer', False)
        # Processing the call keyword arguments (line 352)
        kwargs_121153 = {}
        # Getting the type of 'self' (line 352)
        self_121149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'self', False)
        # Obtaining the member 'text' of a type (line 352)
        text_121150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), self_121149, 'text')
        # Obtaining the member 'draw' of a type (line 352)
        draw_121151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), text_121150, 'draw')
        # Calling draw(args, kwargs) (line 352)
        draw_call_result_121154 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), draw_121151, *[renderer_121152], **kwargs_121153)
        
        
        # Assigning a Name to a Attribute (line 353):
        
        # Assigning a Name to a Attribute (line 353):
        # Getting the type of 'False' (line 353)
        False_121155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 21), 'False')
        # Getting the type of 'self' (line 353)
        self_121156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 353)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 8), self_121156, 'stale', False_121155)
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 345)
        stypy_return_type_121157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121157)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_121157


    @norecursion
    def _set_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_transform'
        module_type_store = module_type_store.open_function_context('_set_transform', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        QuiverKey._set_transform.__dict__.__setitem__('stypy_localization', localization)
        QuiverKey._set_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        QuiverKey._set_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        QuiverKey._set_transform.__dict__.__setitem__('stypy_function_name', 'QuiverKey._set_transform')
        QuiverKey._set_transform.__dict__.__setitem__('stypy_param_names_list', [])
        QuiverKey._set_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        QuiverKey._set_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        QuiverKey._set_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        QuiverKey._set_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        QuiverKey._set_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        QuiverKey._set_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'QuiverKey._set_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_transform(...)' code ##################

        
        
        # Getting the type of 'self' (line 356)
        self_121158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'self')
        # Obtaining the member 'coord' of a type (line 356)
        coord_121159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 11), self_121158, 'coord')
        unicode_121160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 25), 'unicode', u'data')
        # Applying the binary operator '==' (line 356)
        result_eq_121161 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 11), '==', coord_121159, unicode_121160)
        
        # Testing the type of an if condition (line 356)
        if_condition_121162 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 8), result_eq_121161)
        # Assigning a type to the variable 'if_condition_121162' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'if_condition_121162', if_condition_121162)
        # SSA begins for if statement (line 356)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_transform(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'self' (line 357)
        self_121165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 31), 'self', False)
        # Obtaining the member 'Q' of a type (line 357)
        Q_121166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 31), self_121165, 'Q')
        # Obtaining the member 'ax' of a type (line 357)
        ax_121167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 31), Q_121166, 'ax')
        # Obtaining the member 'transData' of a type (line 357)
        transData_121168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 31), ax_121167, 'transData')
        # Processing the call keyword arguments (line 357)
        kwargs_121169 = {}
        # Getting the type of 'self' (line 357)
        self_121163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'self', False)
        # Obtaining the member 'set_transform' of a type (line 357)
        set_transform_121164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 12), self_121163, 'set_transform')
        # Calling set_transform(args, kwargs) (line 357)
        set_transform_call_result_121170 = invoke(stypy.reporting.localization.Localization(__file__, 357, 12), set_transform_121164, *[transData_121168], **kwargs_121169)
        
        # SSA branch for the else part of an if statement (line 356)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 358)
        self_121171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 13), 'self')
        # Obtaining the member 'coord' of a type (line 358)
        coord_121172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 13), self_121171, 'coord')
        unicode_121173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 27), 'unicode', u'axes')
        # Applying the binary operator '==' (line 358)
        result_eq_121174 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 13), '==', coord_121172, unicode_121173)
        
        # Testing the type of an if condition (line 358)
        if_condition_121175 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 358, 13), result_eq_121174)
        # Assigning a type to the variable 'if_condition_121175' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 13), 'if_condition_121175', if_condition_121175)
        # SSA begins for if statement (line 358)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_transform(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'self' (line 359)
        self_121178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 31), 'self', False)
        # Obtaining the member 'Q' of a type (line 359)
        Q_121179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 31), self_121178, 'Q')
        # Obtaining the member 'ax' of a type (line 359)
        ax_121180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 31), Q_121179, 'ax')
        # Obtaining the member 'transAxes' of a type (line 359)
        transAxes_121181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 31), ax_121180, 'transAxes')
        # Processing the call keyword arguments (line 359)
        kwargs_121182 = {}
        # Getting the type of 'self' (line 359)
        self_121176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'self', False)
        # Obtaining the member 'set_transform' of a type (line 359)
        set_transform_121177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 12), self_121176, 'set_transform')
        # Calling set_transform(args, kwargs) (line 359)
        set_transform_call_result_121183 = invoke(stypy.reporting.localization.Localization(__file__, 359, 12), set_transform_121177, *[transAxes_121181], **kwargs_121182)
        
        # SSA branch for the else part of an if statement (line 358)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 360)
        self_121184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 13), 'self')
        # Obtaining the member 'coord' of a type (line 360)
        coord_121185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 13), self_121184, 'coord')
        unicode_121186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 27), 'unicode', u'figure')
        # Applying the binary operator '==' (line 360)
        result_eq_121187 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 13), '==', coord_121185, unicode_121186)
        
        # Testing the type of an if condition (line 360)
        if_condition_121188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 13), result_eq_121187)
        # Assigning a type to the variable 'if_condition_121188' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 13), 'if_condition_121188', if_condition_121188)
        # SSA begins for if statement (line 360)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_transform(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'self' (line 361)
        self_121191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 31), 'self', False)
        # Obtaining the member 'Q' of a type (line 361)
        Q_121192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 31), self_121191, 'Q')
        # Obtaining the member 'ax' of a type (line 361)
        ax_121193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 31), Q_121192, 'ax')
        # Obtaining the member 'figure' of a type (line 361)
        figure_121194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 31), ax_121193, 'figure')
        # Obtaining the member 'transFigure' of a type (line 361)
        transFigure_121195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 31), figure_121194, 'transFigure')
        # Processing the call keyword arguments (line 361)
        kwargs_121196 = {}
        # Getting the type of 'self' (line 361)
        self_121189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'self', False)
        # Obtaining the member 'set_transform' of a type (line 361)
        set_transform_121190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 12), self_121189, 'set_transform')
        # Calling set_transform(args, kwargs) (line 361)
        set_transform_call_result_121197 = invoke(stypy.reporting.localization.Localization(__file__, 361, 12), set_transform_121190, *[transFigure_121195], **kwargs_121196)
        
        # SSA branch for the else part of an if statement (line 360)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 362)
        self_121198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 13), 'self')
        # Obtaining the member 'coord' of a type (line 362)
        coord_121199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 13), self_121198, 'coord')
        unicode_121200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 27), 'unicode', u'inches')
        # Applying the binary operator '==' (line 362)
        result_eq_121201 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 13), '==', coord_121199, unicode_121200)
        
        # Testing the type of an if condition (line 362)
        if_condition_121202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 13), result_eq_121201)
        # Assigning a type to the variable 'if_condition_121202' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 13), 'if_condition_121202', if_condition_121202)
        # SSA begins for if statement (line 362)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_transform(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'self' (line 363)
        self_121205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 31), 'self', False)
        # Obtaining the member 'Q' of a type (line 363)
        Q_121206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 31), self_121205, 'Q')
        # Obtaining the member 'ax' of a type (line 363)
        ax_121207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 31), Q_121206, 'ax')
        # Obtaining the member 'figure' of a type (line 363)
        figure_121208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 31), ax_121207, 'figure')
        # Obtaining the member 'dpi_scale_trans' of a type (line 363)
        dpi_scale_trans_121209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 31), figure_121208, 'dpi_scale_trans')
        # Processing the call keyword arguments (line 363)
        kwargs_121210 = {}
        # Getting the type of 'self' (line 363)
        self_121203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'self', False)
        # Obtaining the member 'set_transform' of a type (line 363)
        set_transform_121204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 12), self_121203, 'set_transform')
        # Calling set_transform(args, kwargs) (line 363)
        set_transform_call_result_121211 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), set_transform_121204, *[dpi_scale_trans_121209], **kwargs_121210)
        
        # SSA branch for the else part of an if statement (line 362)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 365)
        # Processing the call arguments (line 365)
        unicode_121213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 29), 'unicode', u'unrecognized coordinates')
        # Processing the call keyword arguments (line 365)
        kwargs_121214 = {}
        # Getting the type of 'ValueError' (line 365)
        ValueError_121212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 365)
        ValueError_call_result_121215 = invoke(stypy.reporting.localization.Localization(__file__, 365, 18), ValueError_121212, *[unicode_121213], **kwargs_121214)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 365, 12), ValueError_call_result_121215, 'raise parameter', BaseException)
        # SSA join for if statement (line 362)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 360)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 358)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 356)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_set_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 355)
        stypy_return_type_121216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121216)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_transform'
        return stypy_return_type_121216


    @norecursion
    def set_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_figure'
        module_type_store = module_type_store.open_function_context('set_figure', 367, 4, False)
        # Assigning a type to the variable 'self' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        QuiverKey.set_figure.__dict__.__setitem__('stypy_localization', localization)
        QuiverKey.set_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        QuiverKey.set_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        QuiverKey.set_figure.__dict__.__setitem__('stypy_function_name', 'QuiverKey.set_figure')
        QuiverKey.set_figure.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        QuiverKey.set_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        QuiverKey.set_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        QuiverKey.set_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        QuiverKey.set_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        QuiverKey.set_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        QuiverKey.set_figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'QuiverKey.set_figure', ['fig'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_figure', localization, ['fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_figure(...)' code ##################

        
        # Call to set_figure(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'self' (line 368)
        self_121220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 34), 'self', False)
        # Getting the type of 'fig' (line 368)
        fig_121221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 40), 'fig', False)
        # Processing the call keyword arguments (line 368)
        kwargs_121222 = {}
        # Getting the type of 'martist' (line 368)
        martist_121217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'martist', False)
        # Obtaining the member 'Artist' of a type (line 368)
        Artist_121218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), martist_121217, 'Artist')
        # Obtaining the member 'set_figure' of a type (line 368)
        set_figure_121219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), Artist_121218, 'set_figure')
        # Calling set_figure(args, kwargs) (line 368)
        set_figure_call_result_121223 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), set_figure_121219, *[self_121220, fig_121221], **kwargs_121222)
        
        
        # Call to set_figure(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'fig' (line 369)
        fig_121227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 29), 'fig', False)
        # Processing the call keyword arguments (line 369)
        kwargs_121228 = {}
        # Getting the type of 'self' (line 369)
        self_121224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'self', False)
        # Obtaining the member 'text' of a type (line 369)
        text_121225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), self_121224, 'text')
        # Obtaining the member 'set_figure' of a type (line 369)
        set_figure_121226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), text_121225, 'set_figure')
        # Calling set_figure(args, kwargs) (line 369)
        set_figure_call_result_121229 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), set_figure_121226, *[fig_121227], **kwargs_121228)
        
        
        # ################# End of 'set_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 367)
        stypy_return_type_121230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121230)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_figure'
        return stypy_return_type_121230


    @norecursion
    def contains(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'contains'
        module_type_store = module_type_store.open_function_context('contains', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        QuiverKey.contains.__dict__.__setitem__('stypy_localization', localization)
        QuiverKey.contains.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        QuiverKey.contains.__dict__.__setitem__('stypy_type_store', module_type_store)
        QuiverKey.contains.__dict__.__setitem__('stypy_function_name', 'QuiverKey.contains')
        QuiverKey.contains.__dict__.__setitem__('stypy_param_names_list', ['mouseevent'])
        QuiverKey.contains.__dict__.__setitem__('stypy_varargs_param_name', None)
        QuiverKey.contains.__dict__.__setitem__('stypy_kwargs_param_name', None)
        QuiverKey.contains.__dict__.__setitem__('stypy_call_defaults', defaults)
        QuiverKey.contains.__dict__.__setitem__('stypy_call_varargs', varargs)
        QuiverKey.contains.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        QuiverKey.contains.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'QuiverKey.contains', ['mouseevent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'contains', localization, ['mouseevent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'contains(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Obtaining the type of the subscript
        int_121231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 43), 'int')
        
        # Call to contains(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'mouseevent' (line 374)
        mouseevent_121235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 31), 'mouseevent', False)
        # Processing the call keyword arguments (line 374)
        kwargs_121236 = {}
        # Getting the type of 'self' (line 374)
        self_121232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'self', False)
        # Obtaining the member 'text' of a type (line 374)
        text_121233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), self_121232, 'text')
        # Obtaining the member 'contains' of a type (line 374)
        contains_121234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), text_121233, 'contains')
        # Calling contains(args, kwargs) (line 374)
        contains_call_result_121237 = invoke(stypy.reporting.localization.Localization(__file__, 374, 12), contains_121234, *[mouseevent_121235], **kwargs_121236)
        
        # Obtaining the member '__getitem__' of a type (line 374)
        getitem___121238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), contains_call_result_121237, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 374)
        subscript_call_result_121239 = invoke(stypy.reporting.localization.Localization(__file__, 374, 12), getitem___121238, int_121231)
        
        
        # Obtaining the type of the subscript
        int_121240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 49), 'int')
        
        # Call to contains(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'mouseevent' (line 375)
        mouseevent_121244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 37), 'mouseevent', False)
        # Processing the call keyword arguments (line 375)
        kwargs_121245 = {}
        # Getting the type of 'self' (line 375)
        self_121241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'self', False)
        # Obtaining the member 'vector' of a type (line 375)
        vector_121242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 16), self_121241, 'vector')
        # Obtaining the member 'contains' of a type (line 375)
        contains_121243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 16), vector_121242, 'contains')
        # Calling contains(args, kwargs) (line 375)
        contains_call_result_121246 = invoke(stypy.reporting.localization.Localization(__file__, 375, 16), contains_121243, *[mouseevent_121244], **kwargs_121245)
        
        # Obtaining the member '__getitem__' of a type (line 375)
        getitem___121247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 16), contains_call_result_121246, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 375)
        subscript_call_result_121248 = invoke(stypy.reporting.localization.Localization(__file__, 375, 16), getitem___121247, int_121240)
        
        # Applying the binary operator 'or' (line 374)
        result_or_keyword_121249 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 12), 'or', subscript_call_result_121239, subscript_call_result_121248)
        
        # Testing the type of an if condition (line 374)
        if_condition_121250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 8), result_or_keyword_121249)
        # Assigning a type to the variable 'if_condition_121250' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'if_condition_121250', if_condition_121250)
        # SSA begins for if statement (line 374)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 376)
        tuple_121251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 376)
        # Adding element type (line 376)
        # Getting the type of 'True' (line 376)
        True_121252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 19), tuple_121251, True_121252)
        # Adding element type (line 376)
        
        # Obtaining an instance of the builtin type 'dict' (line 376)
        dict_121253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 376)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 19), tuple_121251, dict_121253)
        
        # Assigning a type to the variable 'stypy_return_type' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', tuple_121251)
        # SSA join for if statement (line 374)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 377)
        tuple_121254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 377)
        # Adding element type (line 377)
        # Getting the type of 'False' (line 377)
        False_121255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 15), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 15), tuple_121254, False_121255)
        # Adding element type (line 377)
        
        # Obtaining an instance of the builtin type 'dict' (line 377)
        dict_121256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 377)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 15), tuple_121254, dict_121256)
        
        # Assigning a type to the variable 'stypy_return_type' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'stypy_return_type', tuple_121254)
        
        # ################# End of 'contains(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'contains' in the type store
        # Getting the type of 'stypy_return_type' (line 371)
        stypy_return_type_121257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121257)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'contains'
        return stypy_return_type_121257

    
    # Assigning a Name to a Name (line 379):

# Assigning a type to the variable 'QuiverKey' (line 239)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'QuiverKey', QuiverKey)

# Assigning a Dict to a Name (line 241):

# Obtaining an instance of the builtin type 'dict' (line 241)
dict_121258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 241)
# Adding element type (key, value) (line 241)
unicode_121259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 14), 'unicode', u'N')
unicode_121260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 19), 'unicode', u'center')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 13), dict_121258, (unicode_121259, unicode_121260))
# Adding element type (key, value) (line 241)
unicode_121261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 29), 'unicode', u'S')
unicode_121262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 34), 'unicode', u'center')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 13), dict_121258, (unicode_121261, unicode_121262))
# Adding element type (key, value) (line 241)
unicode_121263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 44), 'unicode', u'E')
unicode_121264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 49), 'unicode', u'left')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 13), dict_121258, (unicode_121263, unicode_121264))
# Adding element type (key, value) (line 241)
unicode_121265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 57), 'unicode', u'W')
unicode_121266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 62), 'unicode', u'right')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 13), dict_121258, (unicode_121265, unicode_121266))

# Getting the type of 'QuiverKey'
QuiverKey_121267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'QuiverKey')
# Setting the type of the member 'halign' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), QuiverKey_121267, 'halign', dict_121258)

# Assigning a Dict to a Name (line 242):

# Obtaining an instance of the builtin type 'dict' (line 242)
dict_121268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 242)
# Adding element type (key, value) (line 242)
unicode_121269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 14), 'unicode', u'N')
unicode_121270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 19), 'unicode', u'bottom')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 13), dict_121268, (unicode_121269, unicode_121270))
# Adding element type (key, value) (line 242)
unicode_121271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 29), 'unicode', u'S')
unicode_121272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 34), 'unicode', u'top')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 13), dict_121268, (unicode_121271, unicode_121272))
# Adding element type (key, value) (line 242)
unicode_121273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 41), 'unicode', u'E')
unicode_121274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 46), 'unicode', u'center')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 13), dict_121268, (unicode_121273, unicode_121274))
# Adding element type (key, value) (line 242)
unicode_121275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 56), 'unicode', u'W')
unicode_121276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 61), 'unicode', u'center')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 13), dict_121268, (unicode_121275, unicode_121276))

# Getting the type of 'QuiverKey'
QuiverKey_121277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'QuiverKey')
# Setting the type of the member 'valign' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), QuiverKey_121277, 'valign', dict_121268)

# Assigning a Dict to a Name (line 243):

# Obtaining an instance of the builtin type 'dict' (line 243)
dict_121278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 243)
# Adding element type (key, value) (line 243)
unicode_121279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 13), 'unicode', u'N')
unicode_121280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 18), 'unicode', u'middle')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 12), dict_121278, (unicode_121279, unicode_121280))
# Adding element type (key, value) (line 243)
unicode_121281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 28), 'unicode', u'S')
unicode_121282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 33), 'unicode', u'middle')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 12), dict_121278, (unicode_121281, unicode_121282))
# Adding element type (key, value) (line 243)
unicode_121283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 43), 'unicode', u'E')
unicode_121284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 48), 'unicode', u'tip')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 12), dict_121278, (unicode_121283, unicode_121284))
# Adding element type (key, value) (line 243)
unicode_121285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 55), 'unicode', u'W')
unicode_121286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 60), 'unicode', u'tail')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 12), dict_121278, (unicode_121285, unicode_121286))

# Getting the type of 'QuiverKey'
QuiverKey_121287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'QuiverKey')
# Setting the type of the member 'pivot' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), QuiverKey_121287, 'pivot', dict_121278)

# Assigning a Name to a Attribute (line 299):
# Getting the type of '_quiverkey_doc' (line 299)
_quiverkey_doc_121288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 23), '_quiverkey_doc')
# Getting the type of 'QuiverKey'
QuiverKey_121289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'QuiverKey')
# Obtaining the member '__init__' of a type
init___121290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), QuiverKey_121289, '__init__')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), init___121290, '__doc__', _quiverkey_doc_121288)

# Assigning a Name to a Name (line 379):
# Getting the type of '_quiverkey_doc' (line 379)
_quiverkey_doc_121291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 20), '_quiverkey_doc')
# Getting the type of 'QuiverKey'
QuiverKey_121292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'QuiverKey')
# Setting the type of the member 'quiverkey_doc' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), QuiverKey_121292, 'quiverkey_doc', _quiverkey_doc_121291)

@norecursion
def _parse_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_parse_args'
    module_type_store = module_type_store.open_function_context('_parse_args', 385, 0, False)
    
    # Passed parameters checking function
    _parse_args.stypy_localization = localization
    _parse_args.stypy_type_of_self = None
    _parse_args.stypy_type_store = module_type_store
    _parse_args.stypy_function_name = '_parse_args'
    _parse_args.stypy_param_names_list = []
    _parse_args.stypy_varargs_param_name = 'args'
    _parse_args.stypy_kwargs_param_name = None
    _parse_args.stypy_call_defaults = defaults
    _parse_args.stypy_call_varargs = varargs
    _parse_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_parse_args', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_parse_args', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_parse_args(...)' code ##################

    
    # Assigning a BinOp to a Tuple (line 386):
    
    # Assigning a Subscript to a Name (line 386):
    
    # Obtaining the type of the subscript
    int_121293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 4), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 386)
    list_121294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 386)
    # Adding element type (line 386)
    # Getting the type of 'None' (line 386)
    None_121295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 21), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 20), list_121294, None_121295)
    
    int_121296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 29), 'int')
    # Applying the binary operator '*' (line 386)
    result_mul_121297 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 20), '*', list_121294, int_121296)
    
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___121298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 4), result_mul_121297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_121299 = invoke(stypy.reporting.localization.Localization(__file__, 386, 4), getitem___121298, int_121293)
    
    # Assigning a type to the variable 'tuple_var_assignment_120659' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_120659', subscript_call_result_121299)
    
    # Assigning a Subscript to a Name (line 386):
    
    # Obtaining the type of the subscript
    int_121300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 4), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 386)
    list_121301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 386)
    # Adding element type (line 386)
    # Getting the type of 'None' (line 386)
    None_121302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 21), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 20), list_121301, None_121302)
    
    int_121303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 29), 'int')
    # Applying the binary operator '*' (line 386)
    result_mul_121304 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 20), '*', list_121301, int_121303)
    
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___121305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 4), result_mul_121304, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_121306 = invoke(stypy.reporting.localization.Localization(__file__, 386, 4), getitem___121305, int_121300)
    
    # Assigning a type to the variable 'tuple_var_assignment_120660' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_120660', subscript_call_result_121306)
    
    # Assigning a Subscript to a Name (line 386):
    
    # Obtaining the type of the subscript
    int_121307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 4), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 386)
    list_121308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 386)
    # Adding element type (line 386)
    # Getting the type of 'None' (line 386)
    None_121309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 21), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 20), list_121308, None_121309)
    
    int_121310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 29), 'int')
    # Applying the binary operator '*' (line 386)
    result_mul_121311 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 20), '*', list_121308, int_121310)
    
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___121312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 4), result_mul_121311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_121313 = invoke(stypy.reporting.localization.Localization(__file__, 386, 4), getitem___121312, int_121307)
    
    # Assigning a type to the variable 'tuple_var_assignment_120661' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_120661', subscript_call_result_121313)
    
    # Assigning a Subscript to a Name (line 386):
    
    # Obtaining the type of the subscript
    int_121314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 4), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 386)
    list_121315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 386)
    # Adding element type (line 386)
    # Getting the type of 'None' (line 386)
    None_121316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 21), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 20), list_121315, None_121316)
    
    int_121317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 29), 'int')
    # Applying the binary operator '*' (line 386)
    result_mul_121318 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 20), '*', list_121315, int_121317)
    
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___121319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 4), result_mul_121318, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_121320 = invoke(stypy.reporting.localization.Localization(__file__, 386, 4), getitem___121319, int_121314)
    
    # Assigning a type to the variable 'tuple_var_assignment_120662' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_120662', subscript_call_result_121320)
    
    # Assigning a Subscript to a Name (line 386):
    
    # Obtaining the type of the subscript
    int_121321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 4), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 386)
    list_121322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 386)
    # Adding element type (line 386)
    # Getting the type of 'None' (line 386)
    None_121323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 21), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 20), list_121322, None_121323)
    
    int_121324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 29), 'int')
    # Applying the binary operator '*' (line 386)
    result_mul_121325 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 20), '*', list_121322, int_121324)
    
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___121326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 4), result_mul_121325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_121327 = invoke(stypy.reporting.localization.Localization(__file__, 386, 4), getitem___121326, int_121321)
    
    # Assigning a type to the variable 'tuple_var_assignment_120663' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_120663', subscript_call_result_121327)
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'tuple_var_assignment_120659' (line 386)
    tuple_var_assignment_120659_121328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_120659')
    # Assigning a type to the variable 'X' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'X', tuple_var_assignment_120659_121328)
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'tuple_var_assignment_120660' (line 386)
    tuple_var_assignment_120660_121329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_120660')
    # Assigning a type to the variable 'Y' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 7), 'Y', tuple_var_assignment_120660_121329)
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'tuple_var_assignment_120661' (line 386)
    tuple_var_assignment_120661_121330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_120661')
    # Assigning a type to the variable 'U' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 10), 'U', tuple_var_assignment_120661_121330)
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'tuple_var_assignment_120662' (line 386)
    tuple_var_assignment_120662_121331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_120662')
    # Assigning a type to the variable 'V' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 13), 'V', tuple_var_assignment_120662_121331)
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'tuple_var_assignment_120663' (line 386)
    tuple_var_assignment_120663_121332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_120663')
    # Assigning a type to the variable 'C' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 16), 'C', tuple_var_assignment_120663_121332)
    
    # Assigning a Call to a Name (line 387):
    
    # Assigning a Call to a Name (line 387):
    
    # Call to list(...): (line 387)
    # Processing the call arguments (line 387)
    # Getting the type of 'args' (line 387)
    args_121334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'args', False)
    # Processing the call keyword arguments (line 387)
    kwargs_121335 = {}
    # Getting the type of 'list' (line 387)
    list_121333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 11), 'list', False)
    # Calling list(args, kwargs) (line 387)
    list_call_result_121336 = invoke(stypy.reporting.localization.Localization(__file__, 387, 11), list_121333, *[args_121334], **kwargs_121335)
    
    # Assigning a type to the variable 'args' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'args', list_call_result_121336)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'args' (line 391)
    args_121338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 11), 'args', False)
    # Processing the call keyword arguments (line 391)
    kwargs_121339 = {}
    # Getting the type of 'len' (line 391)
    len_121337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 7), 'len', False)
    # Calling len(args, kwargs) (line 391)
    len_call_result_121340 = invoke(stypy.reporting.localization.Localization(__file__, 391, 7), len_121337, *[args_121338], **kwargs_121339)
    
    int_121341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 20), 'int')
    # Applying the binary operator '==' (line 391)
    result_eq_121342 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 7), '==', len_call_result_121340, int_121341)
    
    
    
    # Call to len(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'args' (line 391)
    args_121344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 29), 'args', False)
    # Processing the call keyword arguments (line 391)
    kwargs_121345 = {}
    # Getting the type of 'len' (line 391)
    len_121343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 25), 'len', False)
    # Calling len(args, kwargs) (line 391)
    len_call_result_121346 = invoke(stypy.reporting.localization.Localization(__file__, 391, 25), len_121343, *[args_121344], **kwargs_121345)
    
    int_121347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 38), 'int')
    # Applying the binary operator '==' (line 391)
    result_eq_121348 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 25), '==', len_call_result_121346, int_121347)
    
    # Applying the binary operator 'or' (line 391)
    result_or_keyword_121349 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 7), 'or', result_eq_121342, result_eq_121348)
    
    # Testing the type of an if condition (line 391)
    if_condition_121350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 4), result_or_keyword_121349)
    # Assigning a type to the variable 'if_condition_121350' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'if_condition_121350', if_condition_121350)
    # SSA begins for if statement (line 391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 392):
    
    # Assigning a Call to a Name (line 392):
    
    # Call to atleast_1d(...): (line 392)
    # Processing the call arguments (line 392)
    
    # Call to pop(...): (line 392)
    # Processing the call arguments (line 392)
    int_121355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 35), 'int')
    # Processing the call keyword arguments (line 392)
    kwargs_121356 = {}
    # Getting the type of 'args' (line 392)
    args_121353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 26), 'args', False)
    # Obtaining the member 'pop' of a type (line 392)
    pop_121354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 26), args_121353, 'pop')
    # Calling pop(args, kwargs) (line 392)
    pop_call_result_121357 = invoke(stypy.reporting.localization.Localization(__file__, 392, 26), pop_121354, *[int_121355], **kwargs_121356)
    
    # Processing the call keyword arguments (line 392)
    kwargs_121358 = {}
    # Getting the type of 'np' (line 392)
    np_121351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 392)
    atleast_1d_121352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), np_121351, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 392)
    atleast_1d_call_result_121359 = invoke(stypy.reporting.localization.Localization(__file__, 392, 12), atleast_1d_121352, *[pop_call_result_121357], **kwargs_121358)
    
    # Assigning a type to the variable 'C' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'C', atleast_1d_call_result_121359)
    # SSA join for if statement (line 391)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to atleast_1d(...): (line 393)
    # Processing the call arguments (line 393)
    
    # Call to pop(...): (line 393)
    # Processing the call arguments (line 393)
    int_121364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 31), 'int')
    # Processing the call keyword arguments (line 393)
    kwargs_121365 = {}
    # Getting the type of 'args' (line 393)
    args_121362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 22), 'args', False)
    # Obtaining the member 'pop' of a type (line 393)
    pop_121363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 22), args_121362, 'pop')
    # Calling pop(args, kwargs) (line 393)
    pop_call_result_121366 = invoke(stypy.reporting.localization.Localization(__file__, 393, 22), pop_121363, *[int_121364], **kwargs_121365)
    
    # Processing the call keyword arguments (line 393)
    kwargs_121367 = {}
    # Getting the type of 'np' (line 393)
    np_121360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 393)
    atleast_1d_121361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), np_121360, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 393)
    atleast_1d_call_result_121368 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), atleast_1d_121361, *[pop_call_result_121366], **kwargs_121367)
    
    # Assigning a type to the variable 'V' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'V', atleast_1d_call_result_121368)
    
    # Assigning a Call to a Name (line 394):
    
    # Assigning a Call to a Name (line 394):
    
    # Call to atleast_1d(...): (line 394)
    # Processing the call arguments (line 394)
    
    # Call to pop(...): (line 394)
    # Processing the call arguments (line 394)
    int_121373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 31), 'int')
    # Processing the call keyword arguments (line 394)
    kwargs_121374 = {}
    # Getting the type of 'args' (line 394)
    args_121371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 22), 'args', False)
    # Obtaining the member 'pop' of a type (line 394)
    pop_121372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 22), args_121371, 'pop')
    # Calling pop(args, kwargs) (line 394)
    pop_call_result_121375 = invoke(stypy.reporting.localization.Localization(__file__, 394, 22), pop_121372, *[int_121373], **kwargs_121374)
    
    # Processing the call keyword arguments (line 394)
    kwargs_121376 = {}
    # Getting the type of 'np' (line 394)
    np_121369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 394)
    atleast_1d_121370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), np_121369, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 394)
    atleast_1d_call_result_121377 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), atleast_1d_121370, *[pop_call_result_121375], **kwargs_121376)
    
    # Assigning a type to the variable 'U' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'U', atleast_1d_call_result_121377)
    
    
    # Getting the type of 'U' (line 395)
    U_121378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 7), 'U')
    # Obtaining the member 'ndim' of a type (line 395)
    ndim_121379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 7), U_121378, 'ndim')
    int_121380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 17), 'int')
    # Applying the binary operator '==' (line 395)
    result_eq_121381 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 7), '==', ndim_121379, int_121380)
    
    # Testing the type of an if condition (line 395)
    if_condition_121382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 4), result_eq_121381)
    # Assigning a type to the variable 'if_condition_121382' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'if_condition_121382', if_condition_121382)
    # SSA begins for if statement (line 395)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 396):
    
    # Assigning a Num to a Name (line 396):
    int_121383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 17), 'int')
    # Assigning a type to the variable 'tuple_assignment_120664' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'tuple_assignment_120664', int_121383)
    
    # Assigning a Subscript to a Name (line 396):
    
    # Obtaining the type of the subscript
    int_121384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 28), 'int')
    # Getting the type of 'U' (line 396)
    U_121385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 20), 'U')
    # Obtaining the member 'shape' of a type (line 396)
    shape_121386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 20), U_121385, 'shape')
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___121387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 20), shape_121386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_121388 = invoke(stypy.reporting.localization.Localization(__file__, 396, 20), getitem___121387, int_121384)
    
    # Assigning a type to the variable 'tuple_assignment_120665' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'tuple_assignment_120665', subscript_call_result_121388)
    
    # Assigning a Name to a Name (line 396):
    # Getting the type of 'tuple_assignment_120664' (line 396)
    tuple_assignment_120664_121389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'tuple_assignment_120664')
    # Assigning a type to the variable 'nr' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'nr', tuple_assignment_120664_121389)
    
    # Assigning a Name to a Name (line 396):
    # Getting the type of 'tuple_assignment_120665' (line 396)
    tuple_assignment_120665_121390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'tuple_assignment_120665')
    # Assigning a type to the variable 'nc' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'nc', tuple_assignment_120665_121390)
    # SSA branch for the else part of an if statement (line 395)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Tuple (line 398):
    
    # Assigning a Subscript to a Name (line 398):
    
    # Obtaining the type of the subscript
    int_121391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
    # Getting the type of 'U' (line 398)
    U_121392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 17), 'U')
    # Obtaining the member 'shape' of a type (line 398)
    shape_121393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 17), U_121392, 'shape')
    # Obtaining the member '__getitem__' of a type (line 398)
    getitem___121394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), shape_121393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 398)
    subscript_call_result_121395 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), getitem___121394, int_121391)
    
    # Assigning a type to the variable 'tuple_var_assignment_120666' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'tuple_var_assignment_120666', subscript_call_result_121395)
    
    # Assigning a Subscript to a Name (line 398):
    
    # Obtaining the type of the subscript
    int_121396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'int')
    # Getting the type of 'U' (line 398)
    U_121397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 17), 'U')
    # Obtaining the member 'shape' of a type (line 398)
    shape_121398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 17), U_121397, 'shape')
    # Obtaining the member '__getitem__' of a type (line 398)
    getitem___121399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), shape_121398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 398)
    subscript_call_result_121400 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), getitem___121399, int_121396)
    
    # Assigning a type to the variable 'tuple_var_assignment_120667' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'tuple_var_assignment_120667', subscript_call_result_121400)
    
    # Assigning a Name to a Name (line 398):
    # Getting the type of 'tuple_var_assignment_120666' (line 398)
    tuple_var_assignment_120666_121401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'tuple_var_assignment_120666')
    # Assigning a type to the variable 'nr' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'nr', tuple_var_assignment_120666_121401)
    
    # Assigning a Name to a Name (line 398):
    # Getting the type of 'tuple_var_assignment_120667' (line 398)
    tuple_var_assignment_120667_121402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'tuple_var_assignment_120667')
    # Assigning a type to the variable 'nc' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'nc', tuple_var_assignment_120667_121402)
    # SSA join for if statement (line 395)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'args' (line 399)
    args_121404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 11), 'args', False)
    # Processing the call keyword arguments (line 399)
    kwargs_121405 = {}
    # Getting the type of 'len' (line 399)
    len_121403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 7), 'len', False)
    # Calling len(args, kwargs) (line 399)
    len_call_result_121406 = invoke(stypy.reporting.localization.Localization(__file__, 399, 7), len_121403, *[args_121404], **kwargs_121405)
    
    int_121407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 20), 'int')
    # Applying the binary operator '==' (line 399)
    result_eq_121408 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 7), '==', len_call_result_121406, int_121407)
    
    # Testing the type of an if condition (line 399)
    if_condition_121409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 399, 4), result_eq_121408)
    # Assigning a type to the variable 'if_condition_121409' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'if_condition_121409', if_condition_121409)
    # SSA begins for if statement (line 399)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Tuple (line 400):
    
    # Assigning a Subscript to a Name (line 400):
    
    # Obtaining the type of the subscript
    int_121410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 8), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'args' (line 400)
    args_121419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 45), 'args')
    comprehension_121420 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 16), args_121419)
    # Assigning a type to the variable 'a' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 16), 'a', comprehension_121420)
    
    # Call to ravel(...): (line 400)
    # Processing the call keyword arguments (line 400)
    kwargs_121417 = {}
    
    # Call to array(...): (line 400)
    # Processing the call arguments (line 400)
    # Getting the type of 'a' (line 400)
    a_121413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 25), 'a', False)
    # Processing the call keyword arguments (line 400)
    kwargs_121414 = {}
    # Getting the type of 'np' (line 400)
    np_121411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 400)
    array_121412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 16), np_121411, 'array')
    # Calling array(args, kwargs) (line 400)
    array_call_result_121415 = invoke(stypy.reporting.localization.Localization(__file__, 400, 16), array_121412, *[a_121413], **kwargs_121414)
    
    # Obtaining the member 'ravel' of a type (line 400)
    ravel_121416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 16), array_call_result_121415, 'ravel')
    # Calling ravel(args, kwargs) (line 400)
    ravel_call_result_121418 = invoke(stypy.reporting.localization.Localization(__file__, 400, 16), ravel_121416, *[], **kwargs_121417)
    
    list_121421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 16), list_121421, ravel_call_result_121418)
    # Obtaining the member '__getitem__' of a type (line 400)
    getitem___121422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 8), list_121421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 400)
    subscript_call_result_121423 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), getitem___121422, int_121410)
    
    # Assigning a type to the variable 'tuple_var_assignment_120668' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'tuple_var_assignment_120668', subscript_call_result_121423)
    
    # Assigning a Subscript to a Name (line 400):
    
    # Obtaining the type of the subscript
    int_121424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 8), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'args' (line 400)
    args_121433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 45), 'args')
    comprehension_121434 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 16), args_121433)
    # Assigning a type to the variable 'a' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 16), 'a', comprehension_121434)
    
    # Call to ravel(...): (line 400)
    # Processing the call keyword arguments (line 400)
    kwargs_121431 = {}
    
    # Call to array(...): (line 400)
    # Processing the call arguments (line 400)
    # Getting the type of 'a' (line 400)
    a_121427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 25), 'a', False)
    # Processing the call keyword arguments (line 400)
    kwargs_121428 = {}
    # Getting the type of 'np' (line 400)
    np_121425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 400)
    array_121426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 16), np_121425, 'array')
    # Calling array(args, kwargs) (line 400)
    array_call_result_121429 = invoke(stypy.reporting.localization.Localization(__file__, 400, 16), array_121426, *[a_121427], **kwargs_121428)
    
    # Obtaining the member 'ravel' of a type (line 400)
    ravel_121430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 16), array_call_result_121429, 'ravel')
    # Calling ravel(args, kwargs) (line 400)
    ravel_call_result_121432 = invoke(stypy.reporting.localization.Localization(__file__, 400, 16), ravel_121430, *[], **kwargs_121431)
    
    list_121435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 16), list_121435, ravel_call_result_121432)
    # Obtaining the member '__getitem__' of a type (line 400)
    getitem___121436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 8), list_121435, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 400)
    subscript_call_result_121437 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), getitem___121436, int_121424)
    
    # Assigning a type to the variable 'tuple_var_assignment_120669' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'tuple_var_assignment_120669', subscript_call_result_121437)
    
    # Assigning a Name to a Name (line 400):
    # Getting the type of 'tuple_var_assignment_120668' (line 400)
    tuple_var_assignment_120668_121438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'tuple_var_assignment_120668')
    # Assigning a type to the variable 'X' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'X', tuple_var_assignment_120668_121438)
    
    # Assigning a Name to a Name (line 400):
    # Getting the type of 'tuple_var_assignment_120669' (line 400)
    tuple_var_assignment_120669_121439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'tuple_var_assignment_120669')
    # Assigning a type to the variable 'Y' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 11), 'Y', tuple_var_assignment_120669_121439)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 401)
    # Processing the call arguments (line 401)
    # Getting the type of 'X' (line 401)
    X_121441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'X', False)
    # Processing the call keyword arguments (line 401)
    kwargs_121442 = {}
    # Getting the type of 'len' (line 401)
    len_121440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 11), 'len', False)
    # Calling len(args, kwargs) (line 401)
    len_call_result_121443 = invoke(stypy.reporting.localization.Localization(__file__, 401, 11), len_121440, *[X_121441], **kwargs_121442)
    
    # Getting the type of 'nc' (line 401)
    nc_121444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 21), 'nc')
    # Applying the binary operator '==' (line 401)
    result_eq_121445 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 11), '==', len_call_result_121443, nc_121444)
    
    
    
    # Call to len(...): (line 401)
    # Processing the call arguments (line 401)
    # Getting the type of 'Y' (line 401)
    Y_121447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 32), 'Y', False)
    # Processing the call keyword arguments (line 401)
    kwargs_121448 = {}
    # Getting the type of 'len' (line 401)
    len_121446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 28), 'len', False)
    # Calling len(args, kwargs) (line 401)
    len_call_result_121449 = invoke(stypy.reporting.localization.Localization(__file__, 401, 28), len_121446, *[Y_121447], **kwargs_121448)
    
    # Getting the type of 'nr' (line 401)
    nr_121450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 38), 'nr')
    # Applying the binary operator '==' (line 401)
    result_eq_121451 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 28), '==', len_call_result_121449, nr_121450)
    
    # Applying the binary operator 'and' (line 401)
    result_and_keyword_121452 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 11), 'and', result_eq_121445, result_eq_121451)
    
    # Testing the type of an if condition (line 401)
    if_condition_121453 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 401, 8), result_and_keyword_121452)
    # Assigning a type to the variable 'if_condition_121453' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'if_condition_121453', if_condition_121453)
    # SSA begins for if statement (line 401)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Tuple (line 402):
    
    # Assigning a Subscript to a Name (line 402):
    
    # Obtaining the type of the subscript
    int_121454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 12), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to meshgrid(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'X' (line 402)
    X_121461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 51), 'X', False)
    # Getting the type of 'Y' (line 402)
    Y_121462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 54), 'Y', False)
    # Processing the call keyword arguments (line 402)
    kwargs_121463 = {}
    # Getting the type of 'np' (line 402)
    np_121459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 39), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 402)
    meshgrid_121460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 39), np_121459, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 402)
    meshgrid_call_result_121464 = invoke(stypy.reporting.localization.Localization(__file__, 402, 39), meshgrid_121460, *[X_121461, Y_121462], **kwargs_121463)
    
    comprehension_121465 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 20), meshgrid_call_result_121464)
    # Assigning a type to the variable 'a' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'a', comprehension_121465)
    
    # Call to ravel(...): (line 402)
    # Processing the call keyword arguments (line 402)
    kwargs_121457 = {}
    # Getting the type of 'a' (line 402)
    a_121455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'a', False)
    # Obtaining the member 'ravel' of a type (line 402)
    ravel_121456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 20), a_121455, 'ravel')
    # Calling ravel(args, kwargs) (line 402)
    ravel_call_result_121458 = invoke(stypy.reporting.localization.Localization(__file__, 402, 20), ravel_121456, *[], **kwargs_121457)
    
    list_121466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 20), list_121466, ravel_call_result_121458)
    # Obtaining the member '__getitem__' of a type (line 402)
    getitem___121467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 12), list_121466, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 402)
    subscript_call_result_121468 = invoke(stypy.reporting.localization.Localization(__file__, 402, 12), getitem___121467, int_121454)
    
    # Assigning a type to the variable 'tuple_var_assignment_120670' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'tuple_var_assignment_120670', subscript_call_result_121468)
    
    # Assigning a Subscript to a Name (line 402):
    
    # Obtaining the type of the subscript
    int_121469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 12), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to meshgrid(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'X' (line 402)
    X_121476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 51), 'X', False)
    # Getting the type of 'Y' (line 402)
    Y_121477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 54), 'Y', False)
    # Processing the call keyword arguments (line 402)
    kwargs_121478 = {}
    # Getting the type of 'np' (line 402)
    np_121474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 39), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 402)
    meshgrid_121475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 39), np_121474, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 402)
    meshgrid_call_result_121479 = invoke(stypy.reporting.localization.Localization(__file__, 402, 39), meshgrid_121475, *[X_121476, Y_121477], **kwargs_121478)
    
    comprehension_121480 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 20), meshgrid_call_result_121479)
    # Assigning a type to the variable 'a' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'a', comprehension_121480)
    
    # Call to ravel(...): (line 402)
    # Processing the call keyword arguments (line 402)
    kwargs_121472 = {}
    # Getting the type of 'a' (line 402)
    a_121470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'a', False)
    # Obtaining the member 'ravel' of a type (line 402)
    ravel_121471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 20), a_121470, 'ravel')
    # Calling ravel(args, kwargs) (line 402)
    ravel_call_result_121473 = invoke(stypy.reporting.localization.Localization(__file__, 402, 20), ravel_121471, *[], **kwargs_121472)
    
    list_121481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 20), list_121481, ravel_call_result_121473)
    # Obtaining the member '__getitem__' of a type (line 402)
    getitem___121482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 12), list_121481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 402)
    subscript_call_result_121483 = invoke(stypy.reporting.localization.Localization(__file__, 402, 12), getitem___121482, int_121469)
    
    # Assigning a type to the variable 'tuple_var_assignment_120671' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'tuple_var_assignment_120671', subscript_call_result_121483)
    
    # Assigning a Name to a Name (line 402):
    # Getting the type of 'tuple_var_assignment_120670' (line 402)
    tuple_var_assignment_120670_121484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'tuple_var_assignment_120670')
    # Assigning a type to the variable 'X' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'X', tuple_var_assignment_120670_121484)
    
    # Assigning a Name to a Name (line 402):
    # Getting the type of 'tuple_var_assignment_120671' (line 402)
    tuple_var_assignment_120671_121485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'tuple_var_assignment_120671')
    # Assigning a type to the variable 'Y' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'Y', tuple_var_assignment_120671_121485)
    # SSA join for if statement (line 401)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 399)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 404):
    
    # Assigning a Call to a Name (line 404):
    
    # Call to meshgrid(...): (line 404)
    # Processing the call arguments (line 404)
    
    # Call to arange(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'nc' (line 404)
    nc_121490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 42), 'nc', False)
    # Processing the call keyword arguments (line 404)
    kwargs_121491 = {}
    # Getting the type of 'np' (line 404)
    np_121488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 32), 'np', False)
    # Obtaining the member 'arange' of a type (line 404)
    arange_121489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 32), np_121488, 'arange')
    # Calling arange(args, kwargs) (line 404)
    arange_call_result_121492 = invoke(stypy.reporting.localization.Localization(__file__, 404, 32), arange_121489, *[nc_121490], **kwargs_121491)
    
    
    # Call to arange(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'nr' (line 404)
    nr_121495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 57), 'nr', False)
    # Processing the call keyword arguments (line 404)
    kwargs_121496 = {}
    # Getting the type of 'np' (line 404)
    np_121493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 47), 'np', False)
    # Obtaining the member 'arange' of a type (line 404)
    arange_121494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 47), np_121493, 'arange')
    # Calling arange(args, kwargs) (line 404)
    arange_call_result_121497 = invoke(stypy.reporting.localization.Localization(__file__, 404, 47), arange_121494, *[nr_121495], **kwargs_121496)
    
    # Processing the call keyword arguments (line 404)
    kwargs_121498 = {}
    # Getting the type of 'np' (line 404)
    np_121486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 20), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 404)
    meshgrid_121487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 20), np_121486, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 404)
    meshgrid_call_result_121499 = invoke(stypy.reporting.localization.Localization(__file__, 404, 20), meshgrid_121487, *[arange_call_result_121492, arange_call_result_121497], **kwargs_121498)
    
    # Assigning a type to the variable 'indexgrid' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'indexgrid', meshgrid_call_result_121499)
    
    # Assigning a ListComp to a Tuple (line 405):
    
    # Assigning a Subscript to a Name (line 405):
    
    # Obtaining the type of the subscript
    int_121500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 8), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'indexgrid' (line 405)
    indexgrid_121506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 37), 'indexgrid')
    comprehension_121507 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 16), indexgrid_121506)
    # Assigning a type to the variable 'a' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'a', comprehension_121507)
    
    # Call to ravel(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of 'a' (line 405)
    a_121503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'a', False)
    # Processing the call keyword arguments (line 405)
    kwargs_121504 = {}
    # Getting the type of 'np' (line 405)
    np_121501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'np', False)
    # Obtaining the member 'ravel' of a type (line 405)
    ravel_121502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 16), np_121501, 'ravel')
    # Calling ravel(args, kwargs) (line 405)
    ravel_call_result_121505 = invoke(stypy.reporting.localization.Localization(__file__, 405, 16), ravel_121502, *[a_121503], **kwargs_121504)
    
    list_121508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 16), list_121508, ravel_call_result_121505)
    # Obtaining the member '__getitem__' of a type (line 405)
    getitem___121509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), list_121508, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 405)
    subscript_call_result_121510 = invoke(stypy.reporting.localization.Localization(__file__, 405, 8), getitem___121509, int_121500)
    
    # Assigning a type to the variable 'tuple_var_assignment_120672' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_120672', subscript_call_result_121510)
    
    # Assigning a Subscript to a Name (line 405):
    
    # Obtaining the type of the subscript
    int_121511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 8), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'indexgrid' (line 405)
    indexgrid_121517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 37), 'indexgrid')
    comprehension_121518 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 16), indexgrid_121517)
    # Assigning a type to the variable 'a' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'a', comprehension_121518)
    
    # Call to ravel(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of 'a' (line 405)
    a_121514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'a', False)
    # Processing the call keyword arguments (line 405)
    kwargs_121515 = {}
    # Getting the type of 'np' (line 405)
    np_121512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'np', False)
    # Obtaining the member 'ravel' of a type (line 405)
    ravel_121513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 16), np_121512, 'ravel')
    # Calling ravel(args, kwargs) (line 405)
    ravel_call_result_121516 = invoke(stypy.reporting.localization.Localization(__file__, 405, 16), ravel_121513, *[a_121514], **kwargs_121515)
    
    list_121519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 16), list_121519, ravel_call_result_121516)
    # Obtaining the member '__getitem__' of a type (line 405)
    getitem___121520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), list_121519, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 405)
    subscript_call_result_121521 = invoke(stypy.reporting.localization.Localization(__file__, 405, 8), getitem___121520, int_121511)
    
    # Assigning a type to the variable 'tuple_var_assignment_120673' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_120673', subscript_call_result_121521)
    
    # Assigning a Name to a Name (line 405):
    # Getting the type of 'tuple_var_assignment_120672' (line 405)
    tuple_var_assignment_120672_121522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_120672')
    # Assigning a type to the variable 'X' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'X', tuple_var_assignment_120672_121522)
    
    # Assigning a Name to a Name (line 405):
    # Getting the type of 'tuple_var_assignment_120673' (line 405)
    tuple_var_assignment_120673_121523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tuple_var_assignment_120673')
    # Assigning a type to the variable 'Y' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 11), 'Y', tuple_var_assignment_120673_121523)
    # SSA join for if statement (line 399)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 406)
    tuple_121524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 406)
    # Adding element type (line 406)
    # Getting the type of 'X' (line 406)
    X_121525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 11), 'X')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 11), tuple_121524, X_121525)
    # Adding element type (line 406)
    # Getting the type of 'Y' (line 406)
    Y_121526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 14), 'Y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 11), tuple_121524, Y_121526)
    # Adding element type (line 406)
    # Getting the type of 'U' (line 406)
    U_121527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 17), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 11), tuple_121524, U_121527)
    # Adding element type (line 406)
    # Getting the type of 'V' (line 406)
    V_121528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 11), tuple_121524, V_121528)
    # Adding element type (line 406)
    # Getting the type of 'C' (line 406)
    C_121529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 23), 'C')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 11), tuple_121524, C_121529)
    
    # Assigning a type to the variable 'stypy_return_type' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'stypy_return_type', tuple_121524)
    
    # ################# End of '_parse_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_parse_args' in the type store
    # Getting the type of 'stypy_return_type' (line 385)
    stypy_return_type_121530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121530)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_parse_args'
    return stypy_return_type_121530

# Assigning a type to the variable '_parse_args' (line 385)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 0), '_parse_args', _parse_args)

@norecursion
def _check_consistent_shapes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_consistent_shapes'
    module_type_store = module_type_store.open_function_context('_check_consistent_shapes', 409, 0, False)
    
    # Passed parameters checking function
    _check_consistent_shapes.stypy_localization = localization
    _check_consistent_shapes.stypy_type_of_self = None
    _check_consistent_shapes.stypy_type_store = module_type_store
    _check_consistent_shapes.stypy_function_name = '_check_consistent_shapes'
    _check_consistent_shapes.stypy_param_names_list = []
    _check_consistent_shapes.stypy_varargs_param_name = 'arrays'
    _check_consistent_shapes.stypy_kwargs_param_name = None
    _check_consistent_shapes.stypy_call_defaults = defaults
    _check_consistent_shapes.stypy_call_varargs = varargs
    _check_consistent_shapes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_consistent_shapes', [], 'arrays', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_consistent_shapes', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_consistent_shapes(...)' code ##################

    
    # Assigning a Call to a Name (line 410):
    
    # Assigning a Call to a Name (line 410):
    
    # Call to set(...): (line 410)
    # Processing the call arguments (line 410)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 410, 21, True)
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 410)
    arrays_121534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 38), 'arrays', False)
    comprehension_121535 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 21), arrays_121534)
    # Assigning a type to the variable 'a' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 21), 'a', comprehension_121535)
    # Getting the type of 'a' (line 410)
    a_121532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 21), 'a', False)
    # Obtaining the member 'shape' of a type (line 410)
    shape_121533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 21), a_121532, 'shape')
    list_121536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 21), list_121536, shape_121533)
    # Processing the call keyword arguments (line 410)
    kwargs_121537 = {}
    # Getting the type of 'set' (line 410)
    set_121531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 17), 'set', False)
    # Calling set(args, kwargs) (line 410)
    set_call_result_121538 = invoke(stypy.reporting.localization.Localization(__file__, 410, 17), set_121531, *[list_121536], **kwargs_121537)
    
    # Assigning a type to the variable 'all_shapes' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'all_shapes', set_call_result_121538)
    
    
    
    # Call to len(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'all_shapes' (line 411)
    all_shapes_121540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 11), 'all_shapes', False)
    # Processing the call keyword arguments (line 411)
    kwargs_121541 = {}
    # Getting the type of 'len' (line 411)
    len_121539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 7), 'len', False)
    # Calling len(args, kwargs) (line 411)
    len_call_result_121542 = invoke(stypy.reporting.localization.Localization(__file__, 411, 7), len_121539, *[all_shapes_121540], **kwargs_121541)
    
    int_121543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 26), 'int')
    # Applying the binary operator '!=' (line 411)
    result_ne_121544 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 7), '!=', len_call_result_121542, int_121543)
    
    # Testing the type of an if condition (line 411)
    if_condition_121545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 4), result_ne_121544)
    # Assigning a type to the variable 'if_condition_121545' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'if_condition_121545', if_condition_121545)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 412)
    # Processing the call arguments (line 412)
    unicode_121547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 25), 'unicode', u'The shapes of the passed in arrays do not match.')
    # Processing the call keyword arguments (line 412)
    kwargs_121548 = {}
    # Getting the type of 'ValueError' (line 412)
    ValueError_121546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 412)
    ValueError_call_result_121549 = invoke(stypy.reporting.localization.Localization(__file__, 412, 14), ValueError_121546, *[unicode_121547], **kwargs_121548)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 412, 8), ValueError_call_result_121549, 'raise parameter', BaseException)
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_check_consistent_shapes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_consistent_shapes' in the type store
    # Getting the type of 'stypy_return_type' (line 409)
    stypy_return_type_121550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121550)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_consistent_shapes'
    return stypy_return_type_121550

# Assigning a type to the variable '_check_consistent_shapes' (line 409)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), '_check_consistent_shapes', _check_consistent_shapes)
# Declaration of the 'Quiver' class
# Getting the type of 'mcollections' (line 415)
mcollections_121551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 13), 'mcollections')
# Obtaining the member 'PolyCollection' of a type (line 415)
PolyCollection_121552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 13), mcollections_121551, 'PolyCollection')

class Quiver(PolyCollection_121552, ):
    unicode_121553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, (-1)), 'unicode', u'\n    Specialized PolyCollection for arrows.\n\n    The only API method is set_UVC(), which can be used\n    to change the size, orientation, and color of the\n    arrows; their locations are fixed when the class is\n    instantiated.  Possibly this method will be useful\n    in animations.\n\n    Much of the work in this class is done in the draw()\n    method so that as much information as possible is available\n    about the plot.  In subsequent draw() calls, recalculation\n    is limited to things that might have changed, so there\n    should be no performance penalty from putting the calculations\n    in the draw() method.\n    ')
    
    # Assigning a Tuple to a Name (line 433):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 435, 4, False)
        # Assigning a type to the variable 'self' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Quiver.__init__', ['ax'], 'args', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['ax'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_121554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, (-1)), 'unicode', u'\n        The constructor takes one required argument, an Axes\n        instance, followed by the args and kwargs described\n        by the following pylab interface documentation:\n        %s\n        ')
        
        # Assigning a Name to a Attribute (line 443):
        
        # Assigning a Name to a Attribute (line 443):
        # Getting the type of 'ax' (line 443)
        ax_121555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 18), 'ax')
        # Getting the type of 'self' (line 443)
        self_121556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'self')
        # Setting the type of the member 'ax' of a type (line 443)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), self_121556, 'ax', ax_121555)
        
        # Assigning a Call to a Tuple (line 444):
        
        # Assigning a Call to a Name:
        
        # Call to _parse_args(...): (line 444)
        # Getting the type of 'args' (line 444)
        args_121558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 37), 'args', False)
        # Processing the call keyword arguments (line 444)
        kwargs_121559 = {}
        # Getting the type of '_parse_args' (line 444)
        _parse_args_121557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 24), '_parse_args', False)
        # Calling _parse_args(args, kwargs) (line 444)
        _parse_args_call_result_121560 = invoke(stypy.reporting.localization.Localization(__file__, 444, 24), _parse_args_121557, *[args_121558], **kwargs_121559)
        
        # Assigning a type to the variable 'call_assignment_120674' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120674', _parse_args_call_result_121560)
        
        # Assigning a Call to a Name (line 444):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_121563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 8), 'int')
        # Processing the call keyword arguments
        kwargs_121564 = {}
        # Getting the type of 'call_assignment_120674' (line 444)
        call_assignment_120674_121561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120674', False)
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___121562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), call_assignment_120674_121561, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_121565 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121562, *[int_121563], **kwargs_121564)
        
        # Assigning a type to the variable 'call_assignment_120675' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120675', getitem___call_result_121565)
        
        # Assigning a Name to a Name (line 444):
        # Getting the type of 'call_assignment_120675' (line 444)
        call_assignment_120675_121566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120675')
        # Assigning a type to the variable 'X' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'X', call_assignment_120675_121566)
        
        # Assigning a Call to a Name (line 444):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_121569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 8), 'int')
        # Processing the call keyword arguments
        kwargs_121570 = {}
        # Getting the type of 'call_assignment_120674' (line 444)
        call_assignment_120674_121567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120674', False)
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___121568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), call_assignment_120674_121567, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_121571 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121568, *[int_121569], **kwargs_121570)
        
        # Assigning a type to the variable 'call_assignment_120676' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120676', getitem___call_result_121571)
        
        # Assigning a Name to a Name (line 444):
        # Getting the type of 'call_assignment_120676' (line 444)
        call_assignment_120676_121572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120676')
        # Assigning a type to the variable 'Y' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 11), 'Y', call_assignment_120676_121572)
        
        # Assigning a Call to a Name (line 444):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_121575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 8), 'int')
        # Processing the call keyword arguments
        kwargs_121576 = {}
        # Getting the type of 'call_assignment_120674' (line 444)
        call_assignment_120674_121573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120674', False)
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___121574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), call_assignment_120674_121573, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_121577 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121574, *[int_121575], **kwargs_121576)
        
        # Assigning a type to the variable 'call_assignment_120677' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120677', getitem___call_result_121577)
        
        # Assigning a Name to a Name (line 444):
        # Getting the type of 'call_assignment_120677' (line 444)
        call_assignment_120677_121578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120677')
        # Assigning a type to the variable 'U' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 14), 'U', call_assignment_120677_121578)
        
        # Assigning a Call to a Name (line 444):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_121581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 8), 'int')
        # Processing the call keyword arguments
        kwargs_121582 = {}
        # Getting the type of 'call_assignment_120674' (line 444)
        call_assignment_120674_121579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120674', False)
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___121580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), call_assignment_120674_121579, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_121583 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121580, *[int_121581], **kwargs_121582)
        
        # Assigning a type to the variable 'call_assignment_120678' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120678', getitem___call_result_121583)
        
        # Assigning a Name to a Name (line 444):
        # Getting the type of 'call_assignment_120678' (line 444)
        call_assignment_120678_121584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120678')
        # Assigning a type to the variable 'V' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 17), 'V', call_assignment_120678_121584)
        
        # Assigning a Call to a Name (line 444):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_121587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 8), 'int')
        # Processing the call keyword arguments
        kwargs_121588 = {}
        # Getting the type of 'call_assignment_120674' (line 444)
        call_assignment_120674_121585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120674', False)
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___121586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), call_assignment_120674_121585, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_121589 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121586, *[int_121587], **kwargs_121588)
        
        # Assigning a type to the variable 'call_assignment_120679' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120679', getitem___call_result_121589)
        
        # Assigning a Name to a Name (line 444):
        # Getting the type of 'call_assignment_120679' (line 444)
        call_assignment_120679_121590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'call_assignment_120679')
        # Assigning a type to the variable 'C' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'C', call_assignment_120679_121590)
        
        # Assigning a Name to a Attribute (line 445):
        
        # Assigning a Name to a Attribute (line 445):
        # Getting the type of 'X' (line 445)
        X_121591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 17), 'X')
        # Getting the type of 'self' (line 445)
        self_121592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'self')
        # Setting the type of the member 'X' of a type (line 445)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), self_121592, 'X', X_121591)
        
        # Assigning a Name to a Attribute (line 446):
        
        # Assigning a Name to a Attribute (line 446):
        # Getting the type of 'Y' (line 446)
        Y_121593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 17), 'Y')
        # Getting the type of 'self' (line 446)
        self_121594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'self')
        # Setting the type of the member 'Y' of a type (line 446)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), self_121594, 'Y', Y_121593)
        
        # Assigning a Call to a Attribute (line 447):
        
        # Assigning a Call to a Attribute (line 447):
        
        # Call to hstack(...): (line 447)
        # Processing the call arguments (line 447)
        
        # Obtaining an instance of the builtin type 'tuple' (line 447)
        tuple_121597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 447)
        # Adding element type (line 447)
        
        # Obtaining the type of the subscript
        slice_121598 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 447, 29), None, None, None)
        # Getting the type of 'np' (line 447)
        np_121599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 34), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 447)
        newaxis_121600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 34), np_121599, 'newaxis')
        # Getting the type of 'X' (line 447)
        X_121601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 29), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 447)
        getitem___121602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 29), X_121601, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 447)
        subscript_call_result_121603 = invoke(stypy.reporting.localization.Localization(__file__, 447, 29), getitem___121602, (slice_121598, newaxis_121600))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 29), tuple_121597, subscript_call_result_121603)
        # Adding element type (line 447)
        
        # Obtaining the type of the subscript
        slice_121604 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 447, 47), None, None, None)
        # Getting the type of 'np' (line 447)
        np_121605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 52), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 447)
        newaxis_121606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 52), np_121605, 'newaxis')
        # Getting the type of 'Y' (line 447)
        Y_121607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 47), 'Y', False)
        # Obtaining the member '__getitem__' of a type (line 447)
        getitem___121608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 47), Y_121607, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 447)
        subscript_call_result_121609 = invoke(stypy.reporting.localization.Localization(__file__, 447, 47), getitem___121608, (slice_121604, newaxis_121606))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 29), tuple_121597, subscript_call_result_121609)
        
        # Processing the call keyword arguments (line 447)
        kwargs_121610 = {}
        # Getting the type of 'np' (line 447)
        np_121595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 18), 'np', False)
        # Obtaining the member 'hstack' of a type (line 447)
        hstack_121596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 18), np_121595, 'hstack')
        # Calling hstack(args, kwargs) (line 447)
        hstack_call_result_121611 = invoke(stypy.reporting.localization.Localization(__file__, 447, 18), hstack_121596, *[tuple_121597], **kwargs_121610)
        
        # Getting the type of 'self' (line 447)
        self_121612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'self')
        # Setting the type of the member 'XY' of a type (line 447)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 8), self_121612, 'XY', hstack_call_result_121611)
        
        # Assigning a Call to a Attribute (line 448):
        
        # Assigning a Call to a Attribute (line 448):
        
        # Call to len(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 'X' (line 448)
        X_121614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 21), 'X', False)
        # Processing the call keyword arguments (line 448)
        kwargs_121615 = {}
        # Getting the type of 'len' (line 448)
        len_121613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 17), 'len', False)
        # Calling len(args, kwargs) (line 448)
        len_call_result_121616 = invoke(stypy.reporting.localization.Localization(__file__, 448, 17), len_121613, *[X_121614], **kwargs_121615)
        
        # Getting the type of 'self' (line 448)
        self_121617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'self')
        # Setting the type of the member 'N' of a type (line 448)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), self_121617, 'N', len_call_result_121616)
        
        # Assigning a Call to a Attribute (line 449):
        
        # Assigning a Call to a Attribute (line 449):
        
        # Call to pop(...): (line 449)
        # Processing the call arguments (line 449)
        unicode_121620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 28), 'unicode', u'scale')
        # Getting the type of 'None' (line 449)
        None_121621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 37), 'None', False)
        # Processing the call keyword arguments (line 449)
        kwargs_121622 = {}
        # Getting the type of 'kw' (line 449)
        kw_121618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 21), 'kw', False)
        # Obtaining the member 'pop' of a type (line 449)
        pop_121619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 21), kw_121618, 'pop')
        # Calling pop(args, kwargs) (line 449)
        pop_call_result_121623 = invoke(stypy.reporting.localization.Localization(__file__, 449, 21), pop_121619, *[unicode_121620, None_121621], **kwargs_121622)
        
        # Getting the type of 'self' (line 449)
        self_121624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'self')
        # Setting the type of the member 'scale' of a type (line 449)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), self_121624, 'scale', pop_call_result_121623)
        
        # Assigning a Call to a Attribute (line 450):
        
        # Assigning a Call to a Attribute (line 450):
        
        # Call to pop(...): (line 450)
        # Processing the call arguments (line 450)
        unicode_121627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 32), 'unicode', u'headwidth')
        int_121628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 45), 'int')
        # Processing the call keyword arguments (line 450)
        kwargs_121629 = {}
        # Getting the type of 'kw' (line 450)
        kw_121625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 25), 'kw', False)
        # Obtaining the member 'pop' of a type (line 450)
        pop_121626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 25), kw_121625, 'pop')
        # Calling pop(args, kwargs) (line 450)
        pop_call_result_121630 = invoke(stypy.reporting.localization.Localization(__file__, 450, 25), pop_121626, *[unicode_121627, int_121628], **kwargs_121629)
        
        # Getting the type of 'self' (line 450)
        self_121631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'self')
        # Setting the type of the member 'headwidth' of a type (line 450)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), self_121631, 'headwidth', pop_call_result_121630)
        
        # Assigning a Call to a Attribute (line 451):
        
        # Assigning a Call to a Attribute (line 451):
        
        # Call to float(...): (line 451)
        # Processing the call arguments (line 451)
        
        # Call to pop(...): (line 451)
        # Processing the call arguments (line 451)
        unicode_121635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 39), 'unicode', u'headlength')
        int_121636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 53), 'int')
        # Processing the call keyword arguments (line 451)
        kwargs_121637 = {}
        # Getting the type of 'kw' (line 451)
        kw_121633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 32), 'kw', False)
        # Obtaining the member 'pop' of a type (line 451)
        pop_121634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 32), kw_121633, 'pop')
        # Calling pop(args, kwargs) (line 451)
        pop_call_result_121638 = invoke(stypy.reporting.localization.Localization(__file__, 451, 32), pop_121634, *[unicode_121635, int_121636], **kwargs_121637)
        
        # Processing the call keyword arguments (line 451)
        kwargs_121639 = {}
        # Getting the type of 'float' (line 451)
        float_121632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 26), 'float', False)
        # Calling float(args, kwargs) (line 451)
        float_call_result_121640 = invoke(stypy.reporting.localization.Localization(__file__, 451, 26), float_121632, *[pop_call_result_121638], **kwargs_121639)
        
        # Getting the type of 'self' (line 451)
        self_121641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'self')
        # Setting the type of the member 'headlength' of a type (line 451)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), self_121641, 'headlength', float_call_result_121640)
        
        # Assigning a Call to a Attribute (line 452):
        
        # Assigning a Call to a Attribute (line 452):
        
        # Call to pop(...): (line 452)
        # Processing the call arguments (line 452)
        unicode_121644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 37), 'unicode', u'headaxislength')
        float_121645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 55), 'float')
        # Processing the call keyword arguments (line 452)
        kwargs_121646 = {}
        # Getting the type of 'kw' (line 452)
        kw_121642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 30), 'kw', False)
        # Obtaining the member 'pop' of a type (line 452)
        pop_121643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 30), kw_121642, 'pop')
        # Calling pop(args, kwargs) (line 452)
        pop_call_result_121647 = invoke(stypy.reporting.localization.Localization(__file__, 452, 30), pop_121643, *[unicode_121644, float_121645], **kwargs_121646)
        
        # Getting the type of 'self' (line 452)
        self_121648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'self')
        # Setting the type of the member 'headaxislength' of a type (line 452)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), self_121648, 'headaxislength', pop_call_result_121647)
        
        # Assigning a Call to a Attribute (line 453):
        
        # Assigning a Call to a Attribute (line 453):
        
        # Call to pop(...): (line 453)
        # Processing the call arguments (line 453)
        unicode_121651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 31), 'unicode', u'minshaft')
        int_121652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 43), 'int')
        # Processing the call keyword arguments (line 453)
        kwargs_121653 = {}
        # Getting the type of 'kw' (line 453)
        kw_121649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 24), 'kw', False)
        # Obtaining the member 'pop' of a type (line 453)
        pop_121650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 24), kw_121649, 'pop')
        # Calling pop(args, kwargs) (line 453)
        pop_call_result_121654 = invoke(stypy.reporting.localization.Localization(__file__, 453, 24), pop_121650, *[unicode_121651, int_121652], **kwargs_121653)
        
        # Getting the type of 'self' (line 453)
        self_121655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'self')
        # Setting the type of the member 'minshaft' of a type (line 453)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 8), self_121655, 'minshaft', pop_call_result_121654)
        
        # Assigning a Call to a Attribute (line 454):
        
        # Assigning a Call to a Attribute (line 454):
        
        # Call to pop(...): (line 454)
        # Processing the call arguments (line 454)
        unicode_121658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 32), 'unicode', u'minlength')
        int_121659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 45), 'int')
        # Processing the call keyword arguments (line 454)
        kwargs_121660 = {}
        # Getting the type of 'kw' (line 454)
        kw_121656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 25), 'kw', False)
        # Obtaining the member 'pop' of a type (line 454)
        pop_121657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 25), kw_121656, 'pop')
        # Calling pop(args, kwargs) (line 454)
        pop_call_result_121661 = invoke(stypy.reporting.localization.Localization(__file__, 454, 25), pop_121657, *[unicode_121658, int_121659], **kwargs_121660)
        
        # Getting the type of 'self' (line 454)
        self_121662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'self')
        # Setting the type of the member 'minlength' of a type (line 454)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), self_121662, 'minlength', pop_call_result_121661)
        
        # Assigning a Call to a Attribute (line 455):
        
        # Assigning a Call to a Attribute (line 455):
        
        # Call to pop(...): (line 455)
        # Processing the call arguments (line 455)
        unicode_121665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 28), 'unicode', u'units')
        unicode_121666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 37), 'unicode', u'width')
        # Processing the call keyword arguments (line 455)
        kwargs_121667 = {}
        # Getting the type of 'kw' (line 455)
        kw_121663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 21), 'kw', False)
        # Obtaining the member 'pop' of a type (line 455)
        pop_121664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 21), kw_121663, 'pop')
        # Calling pop(args, kwargs) (line 455)
        pop_call_result_121668 = invoke(stypy.reporting.localization.Localization(__file__, 455, 21), pop_121664, *[unicode_121665, unicode_121666], **kwargs_121667)
        
        # Getting the type of 'self' (line 455)
        self_121669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'self')
        # Setting the type of the member 'units' of a type (line 455)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), self_121669, 'units', pop_call_result_121668)
        
        # Assigning a Call to a Attribute (line 456):
        
        # Assigning a Call to a Attribute (line 456):
        
        # Call to pop(...): (line 456)
        # Processing the call arguments (line 456)
        unicode_121672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 34), 'unicode', u'scale_units')
        # Getting the type of 'None' (line 456)
        None_121673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 49), 'None', False)
        # Processing the call keyword arguments (line 456)
        kwargs_121674 = {}
        # Getting the type of 'kw' (line 456)
        kw_121670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 27), 'kw', False)
        # Obtaining the member 'pop' of a type (line 456)
        pop_121671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 27), kw_121670, 'pop')
        # Calling pop(args, kwargs) (line 456)
        pop_call_result_121675 = invoke(stypy.reporting.localization.Localization(__file__, 456, 27), pop_121671, *[unicode_121672, None_121673], **kwargs_121674)
        
        # Getting the type of 'self' (line 456)
        self_121676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'self')
        # Setting the type of the member 'scale_units' of a type (line 456)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), self_121676, 'scale_units', pop_call_result_121675)
        
        # Assigning a Call to a Attribute (line 457):
        
        # Assigning a Call to a Attribute (line 457):
        
        # Call to pop(...): (line 457)
        # Processing the call arguments (line 457)
        unicode_121679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 29), 'unicode', u'angles')
        unicode_121680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 39), 'unicode', u'uv')
        # Processing the call keyword arguments (line 457)
        kwargs_121681 = {}
        # Getting the type of 'kw' (line 457)
        kw_121677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 22), 'kw', False)
        # Obtaining the member 'pop' of a type (line 457)
        pop_121678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 22), kw_121677, 'pop')
        # Calling pop(args, kwargs) (line 457)
        pop_call_result_121682 = invoke(stypy.reporting.localization.Localization(__file__, 457, 22), pop_121678, *[unicode_121679, unicode_121680], **kwargs_121681)
        
        # Getting the type of 'self' (line 457)
        self_121683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'self')
        # Setting the type of the member 'angles' of a type (line 457)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), self_121683, 'angles', pop_call_result_121682)
        
        # Assigning a Call to a Attribute (line 458):
        
        # Assigning a Call to a Attribute (line 458):
        
        # Call to pop(...): (line 458)
        # Processing the call arguments (line 458)
        unicode_121686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 28), 'unicode', u'width')
        # Getting the type of 'None' (line 458)
        None_121687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 37), 'None', False)
        # Processing the call keyword arguments (line 458)
        kwargs_121688 = {}
        # Getting the type of 'kw' (line 458)
        kw_121684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 21), 'kw', False)
        # Obtaining the member 'pop' of a type (line 458)
        pop_121685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 21), kw_121684, 'pop')
        # Calling pop(args, kwargs) (line 458)
        pop_call_result_121689 = invoke(stypy.reporting.localization.Localization(__file__, 458, 21), pop_121685, *[unicode_121686, None_121687], **kwargs_121688)
        
        # Getting the type of 'self' (line 458)
        self_121690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'self')
        # Setting the type of the member 'width' of a type (line 458)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), self_121690, 'width', pop_call_result_121689)
        
        # Assigning a Call to a Attribute (line 459):
        
        # Assigning a Call to a Attribute (line 459):
        
        # Call to pop(...): (line 459)
        # Processing the call arguments (line 459)
        unicode_121693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 28), 'unicode', u'color')
        unicode_121694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 37), 'unicode', u'k')
        # Processing the call keyword arguments (line 459)
        kwargs_121695 = {}
        # Getting the type of 'kw' (line 459)
        kw_121691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 21), 'kw', False)
        # Obtaining the member 'pop' of a type (line 459)
        pop_121692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 21), kw_121691, 'pop')
        # Calling pop(args, kwargs) (line 459)
        pop_call_result_121696 = invoke(stypy.reporting.localization.Localization(__file__, 459, 21), pop_121692, *[unicode_121693, unicode_121694], **kwargs_121695)
        
        # Getting the type of 'self' (line 459)
        self_121697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'self')
        # Setting the type of the member 'color' of a type (line 459)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), self_121697, 'color', pop_call_result_121696)
        
        # Assigning a Call to a Name (line 461):
        
        # Assigning a Call to a Name (line 461):
        
        # Call to lower(...): (line 461)
        # Processing the call keyword arguments (line 461)
        kwargs_121705 = {}
        
        # Call to pop(...): (line 461)
        # Processing the call arguments (line 461)
        unicode_121700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 23), 'unicode', u'pivot')
        unicode_121701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 32), 'unicode', u'tail')
        # Processing the call keyword arguments (line 461)
        kwargs_121702 = {}
        # Getting the type of 'kw' (line 461)
        kw_121698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 16), 'kw', False)
        # Obtaining the member 'pop' of a type (line 461)
        pop_121699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 16), kw_121698, 'pop')
        # Calling pop(args, kwargs) (line 461)
        pop_call_result_121703 = invoke(stypy.reporting.localization.Localization(__file__, 461, 16), pop_121699, *[unicode_121700, unicode_121701], **kwargs_121702)
        
        # Obtaining the member 'lower' of a type (line 461)
        lower_121704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 16), pop_call_result_121703, 'lower')
        # Calling lower(args, kwargs) (line 461)
        lower_call_result_121706 = invoke(stypy.reporting.localization.Localization(__file__, 461, 16), lower_121704, *[], **kwargs_121705)
        
        # Assigning a type to the variable 'pivot' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'pivot', lower_call_result_121706)
        
        
        # Getting the type of 'pivot' (line 463)
        pivot_121707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'pivot')
        # Getting the type of 'self' (line 463)
        self_121708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 24), 'self')
        # Obtaining the member '_PIVOT_VALS' of a type (line 463)
        _PIVOT_VALS_121709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 24), self_121708, '_PIVOT_VALS')
        # Applying the binary operator 'notin' (line 463)
        result_contains_121710 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 11), 'notin', pivot_121707, _PIVOT_VALS_121709)
        
        # Testing the type of an if condition (line 463)
        if_condition_121711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 8), result_contains_121710)
        # Assigning a type to the variable 'if_condition_121711' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'if_condition_121711', if_condition_121711)
        # SSA begins for if statement (line 463)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 464)
        # Processing the call arguments (line 464)
        
        # Call to format(...): (line 465)
        # Processing the call keyword arguments (line 465)
        # Getting the type of 'self' (line 466)
        self_121715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 27), 'self', False)
        # Obtaining the member '_PIVOT_VALS' of a type (line 466)
        _PIVOT_VALS_121716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 27), self_121715, '_PIVOT_VALS')
        keyword_121717 = _PIVOT_VALS_121716
        # Getting the type of 'pivot' (line 466)
        pivot_121718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 49), 'pivot', False)
        keyword_121719 = pivot_121718
        kwargs_121720 = {'keys': keyword_121717, 'inp': keyword_121719}
        unicode_121713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 16), 'unicode', u'pivot must be one of {keys}, you passed {inp}')
        # Obtaining the member 'format' of a type (line 465)
        format_121714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 16), unicode_121713, 'format')
        # Calling format(args, kwargs) (line 465)
        format_call_result_121721 = invoke(stypy.reporting.localization.Localization(__file__, 465, 16), format_121714, *[], **kwargs_121720)
        
        # Processing the call keyword arguments (line 464)
        kwargs_121722 = {}
        # Getting the type of 'ValueError' (line 464)
        ValueError_121712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 464)
        ValueError_call_result_121723 = invoke(stypy.reporting.localization.Localization(__file__, 464, 18), ValueError_121712, *[format_call_result_121721], **kwargs_121722)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 464, 12), ValueError_call_result_121723, 'raise parameter', BaseException)
        # SSA join for if statement (line 463)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'pivot' (line 468)
        pivot_121724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 11), 'pivot')
        unicode_121725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 20), 'unicode', u'mid')
        # Applying the binary operator '==' (line 468)
        result_eq_121726 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 11), '==', pivot_121724, unicode_121725)
        
        # Testing the type of an if condition (line 468)
        if_condition_121727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 8), result_eq_121726)
        # Assigning a type to the variable 'if_condition_121727' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'if_condition_121727', if_condition_121727)
        # SSA begins for if statement (line 468)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 469):
        
        # Assigning a Str to a Name (line 469):
        unicode_121728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 20), 'unicode', u'middle')
        # Assigning a type to the variable 'pivot' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'pivot', unicode_121728)
        # SSA join for if statement (line 468)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 470):
        
        # Assigning a Name to a Attribute (line 470):
        # Getting the type of 'pivot' (line 470)
        pivot_121729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 21), 'pivot')
        # Getting the type of 'self' (line 470)
        self_121730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'self')
        # Setting the type of the member 'pivot' of a type (line 470)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 8), self_121730, 'pivot', pivot_121729)
        
        # Assigning a Call to a Attribute (line 472):
        
        # Assigning a Call to a Attribute (line 472):
        
        # Call to pop(...): (line 472)
        # Processing the call arguments (line 472)
        unicode_121733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 32), 'unicode', u'transform')
        # Getting the type of 'ax' (line 472)
        ax_121734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 45), 'ax', False)
        # Obtaining the member 'transData' of a type (line 472)
        transData_121735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 45), ax_121734, 'transData')
        # Processing the call keyword arguments (line 472)
        kwargs_121736 = {}
        # Getting the type of 'kw' (line 472)
        kw_121731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 25), 'kw', False)
        # Obtaining the member 'pop' of a type (line 472)
        pop_121732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 25), kw_121731, 'pop')
        # Calling pop(args, kwargs) (line 472)
        pop_call_result_121737 = invoke(stypy.reporting.localization.Localization(__file__, 472, 25), pop_121732, *[unicode_121733, transData_121735], **kwargs_121736)
        
        # Getting the type of 'self' (line 472)
        self_121738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'self')
        # Setting the type of the member 'transform' of a type (line 472)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), self_121738, 'transform', pop_call_result_121737)
        
        # Call to setdefault(...): (line 473)
        # Processing the call arguments (line 473)
        unicode_121741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 22), 'unicode', u'facecolors')
        # Getting the type of 'self' (line 473)
        self_121742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 36), 'self', False)
        # Obtaining the member 'color' of a type (line 473)
        color_121743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 36), self_121742, 'color')
        # Processing the call keyword arguments (line 473)
        kwargs_121744 = {}
        # Getting the type of 'kw' (line 473)
        kw_121739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'kw', False)
        # Obtaining the member 'setdefault' of a type (line 473)
        setdefault_121740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 8), kw_121739, 'setdefault')
        # Calling setdefault(args, kwargs) (line 473)
        setdefault_call_result_121745 = invoke(stypy.reporting.localization.Localization(__file__, 473, 8), setdefault_121740, *[unicode_121741, color_121743], **kwargs_121744)
        
        
        # Call to setdefault(...): (line 474)
        # Processing the call arguments (line 474)
        unicode_121748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 22), 'unicode', u'linewidths')
        
        # Obtaining an instance of the builtin type 'tuple' (line 474)
        tuple_121749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 474)
        # Adding element type (line 474)
        int_121750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 37), tuple_121749, int_121750)
        
        # Processing the call keyword arguments (line 474)
        kwargs_121751 = {}
        # Getting the type of 'kw' (line 474)
        kw_121746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'kw', False)
        # Obtaining the member 'setdefault' of a type (line 474)
        setdefault_121747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), kw_121746, 'setdefault')
        # Calling setdefault(args, kwargs) (line 474)
        setdefault_call_result_121752 = invoke(stypy.reporting.localization.Localization(__file__, 474, 8), setdefault_121747, *[unicode_121748, tuple_121749], **kwargs_121751)
        
        
        # Call to __init__(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'self' (line 475)
        self_121756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 45), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 475)
        list_121757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 475)
        
        # Processing the call keyword arguments (line 475)
        # Getting the type of 'self' (line 475)
        self_121758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 63), 'self', False)
        # Obtaining the member 'XY' of a type (line 475)
        XY_121759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 63), self_121758, 'XY')
        keyword_121760 = XY_121759
        # Getting the type of 'self' (line 476)
        self_121761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 57), 'self', False)
        # Obtaining the member 'transform' of a type (line 476)
        transform_121762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 57), self_121761, 'transform')
        keyword_121763 = transform_121762
        # Getting the type of 'False' (line 477)
        False_121764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 52), 'False', False)
        keyword_121765 = False_121764
        # Getting the type of 'kw' (line 478)
        kw_121766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 47), 'kw', False)
        kwargs_121767 = {'kw_121766': kw_121766, 'transOffset': keyword_121763, 'closed': keyword_121765, 'offsets': keyword_121760}
        # Getting the type of 'mcollections' (line 475)
        mcollections_121753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'mcollections', False)
        # Obtaining the member 'PolyCollection' of a type (line 475)
        PolyCollection_121754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), mcollections_121753, 'PolyCollection')
        # Obtaining the member '__init__' of a type (line 475)
        init___121755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), PolyCollection_121754, '__init__')
        # Calling __init__(args, kwargs) (line 475)
        init___call_result_121768 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), init___121755, *[self_121756, list_121757], **kwargs_121767)
        
        
        # Assigning a Name to a Attribute (line 479):
        
        # Assigning a Name to a Attribute (line 479):
        # Getting the type of 'kw' (line 479)
        kw_121769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 22), 'kw')
        # Getting the type of 'self' (line 479)
        self_121770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'self')
        # Setting the type of the member 'polykw' of a type (line 479)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 8), self_121770, 'polykw', kw_121769)
        
        # Call to set_UVC(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'U' (line 480)
        U_121773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 21), 'U', False)
        # Getting the type of 'V' (line 480)
        V_121774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 24), 'V', False)
        # Getting the type of 'C' (line 480)
        C_121775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 27), 'C', False)
        # Processing the call keyword arguments (line 480)
        kwargs_121776 = {}
        # Getting the type of 'self' (line 480)
        self_121771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'self', False)
        # Obtaining the member 'set_UVC' of a type (line 480)
        set_UVC_121772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), self_121771, 'set_UVC')
        # Calling set_UVC(args, kwargs) (line 480)
        set_UVC_call_result_121777 = invoke(stypy.reporting.localization.Localization(__file__, 480, 8), set_UVC_121772, *[U_121773, V_121774, C_121775], **kwargs_121776)
        
        
        # Assigning a Name to a Attribute (line 481):
        
        # Assigning a Name to a Attribute (line 481):
        # Getting the type of 'False' (line 481)
        False_121778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 28), 'False')
        # Getting the type of 'self' (line 481)
        self_121779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'self')
        # Setting the type of the member '_initialized' of a type (line 481)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 8), self_121779, '_initialized', False_121778)
        
        # Assigning a Name to a Attribute (line 483):
        
        # Assigning a Name to a Attribute (line 483):
        # Getting the type of 'None' (line 483)
        None_121780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 22), 'None')
        # Getting the type of 'self' (line 483)
        self_121781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'self')
        # Setting the type of the member 'keyvec' of a type (line 483)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), self_121781, 'keyvec', None_121780)
        
        # Assigning a Name to a Attribute (line 484):
        
        # Assigning a Name to a Attribute (line 484):
        # Getting the type of 'None' (line 484)
        None_121782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 23), 'None')
        # Getting the type of 'self' (line 484)
        self_121783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'self')
        # Setting the type of the member 'keytext' of a type (line 484)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), self_121783, 'keytext', None_121782)
        
        # Assigning a Call to a Name (line 487):
        
        # Assigning a Call to a Name (line 487):
        
        # Call to ref(...): (line 487)
        # Processing the call arguments (line 487)
        # Getting the type of 'self' (line 487)
        self_121786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 32), 'self', False)
        # Processing the call keyword arguments (line 487)
        kwargs_121787 = {}
        # Getting the type of 'weakref' (line 487)
        weakref_121784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 20), 'weakref', False)
        # Obtaining the member 'ref' of a type (line 487)
        ref_121785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 20), weakref_121784, 'ref')
        # Calling ref(args, kwargs) (line 487)
        ref_call_result_121788 = invoke(stypy.reporting.localization.Localization(__file__, 487, 20), ref_121785, *[self_121786], **kwargs_121787)
        
        # Assigning a type to the variable 'weak_self' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'weak_self', ref_call_result_121788)

        @norecursion
        def on_dpi_change(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'on_dpi_change'
            module_type_store = module_type_store.open_function_context('on_dpi_change', 489, 8, False)
            
            # Passed parameters checking function
            on_dpi_change.stypy_localization = localization
            on_dpi_change.stypy_type_of_self = None
            on_dpi_change.stypy_type_store = module_type_store
            on_dpi_change.stypy_function_name = 'on_dpi_change'
            on_dpi_change.stypy_param_names_list = ['fig']
            on_dpi_change.stypy_varargs_param_name = None
            on_dpi_change.stypy_kwargs_param_name = None
            on_dpi_change.stypy_call_defaults = defaults
            on_dpi_change.stypy_call_varargs = varargs
            on_dpi_change.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'on_dpi_change', ['fig'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'on_dpi_change', localization, ['fig'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'on_dpi_change(...)' code ##################

            
            # Assigning a Call to a Name (line 490):
            
            # Assigning a Call to a Name (line 490):
            
            # Call to weak_self(...): (line 490)
            # Processing the call keyword arguments (line 490)
            kwargs_121790 = {}
            # Getting the type of 'weak_self' (line 490)
            weak_self_121789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 27), 'weak_self', False)
            # Calling weak_self(args, kwargs) (line 490)
            weak_self_call_result_121791 = invoke(stypy.reporting.localization.Localization(__file__, 490, 27), weak_self_121789, *[], **kwargs_121790)
            
            # Assigning a type to the variable 'self_weakref' (line 490)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'self_weakref', weak_self_call_result_121791)
            
            # Type idiom detected: calculating its left and rigth part (line 491)
            # Getting the type of 'self_weakref' (line 491)
            self_weakref_121792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'self_weakref')
            # Getting the type of 'None' (line 491)
            None_121793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 35), 'None')
            
            (may_be_121794, more_types_in_union_121795) = may_not_be_none(self_weakref_121792, None_121793)

            if may_be_121794:

                if more_types_in_union_121795:
                    # Runtime conditional SSA (line 491)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Name to a Attribute (line 492):
                
                # Assigning a Name to a Attribute (line 492):
                # Getting the type of 'True' (line 492)
                True_121796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 39), 'True')
                # Getting the type of 'self_weakref' (line 492)
                self_weakref_121797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 16), 'self_weakref')
                # Setting the type of the member '_new_UV' of a type (line 492)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 16), self_weakref_121797, '_new_UV', True_121796)
                
                # Assigning a Name to a Attribute (line 494):
                
                # Assigning a Name to a Attribute (line 494):
                # Getting the type of 'False' (line 494)
                False_121798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 44), 'False')
                # Getting the type of 'self_weakref' (line 494)
                self_weakref_121799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 16), 'self_weakref')
                # Setting the type of the member '_initialized' of a type (line 494)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 16), self_weakref_121799, '_initialized', False_121798)

                if more_types_in_union_121795:
                    # SSA join for if statement (line 491)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # ################# End of 'on_dpi_change(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'on_dpi_change' in the type store
            # Getting the type of 'stypy_return_type' (line 489)
            stypy_return_type_121800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_121800)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'on_dpi_change'
            return stypy_return_type_121800

        # Assigning a type to the variable 'on_dpi_change' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'on_dpi_change', on_dpi_change)
        
        # Assigning a Call to a Attribute (line 499):
        
        # Assigning a Call to a Attribute (line 499):
        
        # Call to connect(...): (line 499)
        # Processing the call arguments (line 499)
        unicode_121806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 53), 'unicode', u'dpi_changed')
        # Getting the type of 'on_dpi_change' (line 500)
        on_dpi_change_121807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 53), 'on_dpi_change', False)
        # Processing the call keyword arguments (line 499)
        kwargs_121808 = {}
        # Getting the type of 'self' (line 499)
        self_121801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 20), 'self', False)
        # Obtaining the member 'ax' of a type (line 499)
        ax_121802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 20), self_121801, 'ax')
        # Obtaining the member 'figure' of a type (line 499)
        figure_121803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 20), ax_121802, 'figure')
        # Obtaining the member 'callbacks' of a type (line 499)
        callbacks_121804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 20), figure_121803, 'callbacks')
        # Obtaining the member 'connect' of a type (line 499)
        connect_121805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 20), callbacks_121804, 'connect')
        # Calling connect(args, kwargs) (line 499)
        connect_call_result_121809 = invoke(stypy.reporting.localization.Localization(__file__, 499, 20), connect_121805, *[unicode_121806, on_dpi_change_121807], **kwargs_121808)
        
        # Getting the type of 'self' (line 499)
        self_121810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'self')
        # Setting the type of the member '_cid' of a type (line 499)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), self_121810, '_cid', connect_call_result_121809)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def remove(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove'
        module_type_store = module_type_store.open_function_context('remove', 502, 4, False)
        # Assigning a type to the variable 'self' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Quiver.remove.__dict__.__setitem__('stypy_localization', localization)
        Quiver.remove.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Quiver.remove.__dict__.__setitem__('stypy_type_store', module_type_store)
        Quiver.remove.__dict__.__setitem__('stypy_function_name', 'Quiver.remove')
        Quiver.remove.__dict__.__setitem__('stypy_param_names_list', [])
        Quiver.remove.__dict__.__setitem__('stypy_varargs_param_name', None)
        Quiver.remove.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Quiver.remove.__dict__.__setitem__('stypy_call_defaults', defaults)
        Quiver.remove.__dict__.__setitem__('stypy_call_varargs', varargs)
        Quiver.remove.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Quiver.remove.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Quiver.remove', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove(...)' code ##################

        unicode_121811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, (-1)), 'unicode', u'\n        Overload the remove method\n        ')
        
        # Call to disconnect(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'self' (line 507)
        self_121817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 44), 'self', False)
        # Obtaining the member '_cid' of a type (line 507)
        _cid_121818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 44), self_121817, '_cid')
        # Processing the call keyword arguments (line 507)
        kwargs_121819 = {}
        # Getting the type of 'self' (line 507)
        self_121812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'self', False)
        # Obtaining the member 'ax' of a type (line 507)
        ax_121813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 8), self_121812, 'ax')
        # Obtaining the member 'figure' of a type (line 507)
        figure_121814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 8), ax_121813, 'figure')
        # Obtaining the member 'callbacks' of a type (line 507)
        callbacks_121815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 8), figure_121814, 'callbacks')
        # Obtaining the member 'disconnect' of a type (line 507)
        disconnect_121816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 8), callbacks_121815, 'disconnect')
        # Calling disconnect(args, kwargs) (line 507)
        disconnect_call_result_121820 = invoke(stypy.reporting.localization.Localization(__file__, 507, 8), disconnect_121816, *[_cid_121818], **kwargs_121819)
        
        
        # Assigning a Name to a Attribute (line 508):
        
        # Assigning a Name to a Attribute (line 508):
        # Getting the type of 'None' (line 508)
        None_121821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 20), 'None')
        # Getting the type of 'self' (line 508)
        self_121822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'self')
        # Setting the type of the member '_cid' of a type (line 508)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 8), self_121822, '_cid', None_121821)
        
        # Call to remove(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'self' (line 510)
        self_121826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 43), 'self', False)
        # Processing the call keyword arguments (line 510)
        kwargs_121827 = {}
        # Getting the type of 'mcollections' (line 510)
        mcollections_121823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'mcollections', False)
        # Obtaining the member 'PolyCollection' of a type (line 510)
        PolyCollection_121824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), mcollections_121823, 'PolyCollection')
        # Obtaining the member 'remove' of a type (line 510)
        remove_121825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), PolyCollection_121824, 'remove')
        # Calling remove(args, kwargs) (line 510)
        remove_call_result_121828 = invoke(stypy.reporting.localization.Localization(__file__, 510, 8), remove_121825, *[self_121826], **kwargs_121827)
        
        
        # ################# End of 'remove(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove' in the type store
        # Getting the type of 'stypy_return_type' (line 502)
        stypy_return_type_121829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121829)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove'
        return stypy_return_type_121829


    @norecursion
    def _init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init'
        module_type_store = module_type_store.open_function_context('_init', 512, 4, False)
        # Assigning a type to the variable 'self' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Quiver._init.__dict__.__setitem__('stypy_localization', localization)
        Quiver._init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Quiver._init.__dict__.__setitem__('stypy_type_store', module_type_store)
        Quiver._init.__dict__.__setitem__('stypy_function_name', 'Quiver._init')
        Quiver._init.__dict__.__setitem__('stypy_param_names_list', [])
        Quiver._init.__dict__.__setitem__('stypy_varargs_param_name', None)
        Quiver._init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Quiver._init.__dict__.__setitem__('stypy_call_defaults', defaults)
        Quiver._init.__dict__.__setitem__('stypy_call_varargs', varargs)
        Quiver._init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Quiver._init.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Quiver._init', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init(...)' code ##################

        unicode_121830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, (-1)), 'unicode', u'\n        Initialization delayed until first draw;\n        allow time for axes setup.\n        ')
        
        # Getting the type of 'True' (line 519)
        True_121831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 11), 'True')
        # Testing the type of an if condition (line 519)
        if_condition_121832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 519, 8), True_121831)
        # Assigning a type to the variable 'if_condition_121832' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'if_condition_121832', if_condition_121832)
        # SSA begins for if statement (line 519)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 520):
        
        # Assigning a Call to a Name (line 520):
        
        # Call to _set_transform(...): (line 520)
        # Processing the call keyword arguments (line 520)
        kwargs_121835 = {}
        # Getting the type of 'self' (line 520)
        self_121833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 20), 'self', False)
        # Obtaining the member '_set_transform' of a type (line 520)
        _set_transform_121834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 20), self_121833, '_set_transform')
        # Calling _set_transform(args, kwargs) (line 520)
        _set_transform_call_result_121836 = invoke(stypy.reporting.localization.Localization(__file__, 520, 20), _set_transform_121834, *[], **kwargs_121835)
        
        # Assigning a type to the variable 'trans' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'trans', _set_transform_call_result_121836)
        
        # Assigning a Attribute to a Name (line 521):
        
        # Assigning a Attribute to a Name (line 521):
        # Getting the type of 'self' (line 521)
        self_121837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 17), 'self')
        # Obtaining the member 'ax' of a type (line 521)
        ax_121838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 17), self_121837, 'ax')
        # Assigning a type to the variable 'ax' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'ax', ax_121838)
        
        # Assigning a Call to a Tuple (line 522):
        
        # Assigning a Call to a Name:
        
        # Call to transform_point(...): (line 522)
        # Processing the call arguments (line 522)
        
        # Obtaining an instance of the builtin type 'tuple' (line 523)
        tuple_121844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 523)
        # Adding element type (line 523)
        # Getting the type of 'ax' (line 523)
        ax_121845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 45), 'ax', False)
        # Obtaining the member 'bbox' of a type (line 523)
        bbox_121846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 45), ax_121845, 'bbox')
        # Obtaining the member 'width' of a type (line 523)
        width_121847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 45), bbox_121846, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 45), tuple_121844, width_121847)
        # Adding element type (line 523)
        # Getting the type of 'ax' (line 523)
        ax_121848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 60), 'ax', False)
        # Obtaining the member 'bbox' of a type (line 523)
        bbox_121849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 60), ax_121848, 'bbox')
        # Obtaining the member 'height' of a type (line 523)
        height_121850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 60), bbox_121849, 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 45), tuple_121844, height_121850)
        
        # Processing the call keyword arguments (line 522)
        kwargs_121851 = {}
        
        # Call to inverted(...): (line 522)
        # Processing the call keyword arguments (line 522)
        kwargs_121841 = {}
        # Getting the type of 'trans' (line 522)
        trans_121839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 21), 'trans', False)
        # Obtaining the member 'inverted' of a type (line 522)
        inverted_121840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 21), trans_121839, 'inverted')
        # Calling inverted(args, kwargs) (line 522)
        inverted_call_result_121842 = invoke(stypy.reporting.localization.Localization(__file__, 522, 21), inverted_121840, *[], **kwargs_121841)
        
        # Obtaining the member 'transform_point' of a type (line 522)
        transform_point_121843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 21), inverted_call_result_121842, 'transform_point')
        # Calling transform_point(args, kwargs) (line 522)
        transform_point_call_result_121852 = invoke(stypy.reporting.localization.Localization(__file__, 522, 21), transform_point_121843, *[tuple_121844], **kwargs_121851)
        
        # Assigning a type to the variable 'call_assignment_120680' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'call_assignment_120680', transform_point_call_result_121852)
        
        # Assigning a Call to a Name (line 522):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_121855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 12), 'int')
        # Processing the call keyword arguments
        kwargs_121856 = {}
        # Getting the type of 'call_assignment_120680' (line 522)
        call_assignment_120680_121853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'call_assignment_120680', False)
        # Obtaining the member '__getitem__' of a type (line 522)
        getitem___121854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 12), call_assignment_120680_121853, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_121857 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121854, *[int_121855], **kwargs_121856)
        
        # Assigning a type to the variable 'call_assignment_120681' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'call_assignment_120681', getitem___call_result_121857)
        
        # Assigning a Name to a Name (line 522):
        # Getting the type of 'call_assignment_120681' (line 522)
        call_assignment_120681_121858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'call_assignment_120681')
        # Assigning a type to the variable 'sx' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'sx', call_assignment_120681_121858)
        
        # Assigning a Call to a Name (line 522):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_121861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 12), 'int')
        # Processing the call keyword arguments
        kwargs_121862 = {}
        # Getting the type of 'call_assignment_120680' (line 522)
        call_assignment_120680_121859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'call_assignment_120680', False)
        # Obtaining the member '__getitem__' of a type (line 522)
        getitem___121860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 12), call_assignment_120680_121859, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_121863 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___121860, *[int_121861], **kwargs_121862)
        
        # Assigning a type to the variable 'call_assignment_120682' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'call_assignment_120682', getitem___call_result_121863)
        
        # Assigning a Name to a Name (line 522):
        # Getting the type of 'call_assignment_120682' (line 522)
        call_assignment_120682_121864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'call_assignment_120682')
        # Assigning a type to the variable 'sy' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 16), 'sy', call_assignment_120682_121864)
        
        # Assigning a Name to a Attribute (line 524):
        
        # Assigning a Name to a Attribute (line 524):
        # Getting the type of 'sx' (line 524)
        sx_121865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 24), 'sx')
        # Getting the type of 'self' (line 524)
        self_121866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'self')
        # Setting the type of the member 'span' of a type (line 524)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 12), self_121866, 'span', sx_121865)
        
        # Type idiom detected: calculating its left and rigth part (line 525)
        # Getting the type of 'self' (line 525)
        self_121867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 15), 'self')
        # Obtaining the member 'width' of a type (line 525)
        width_121868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 15), self_121867, 'width')
        # Getting the type of 'None' (line 525)
        None_121869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 29), 'None')
        
        (may_be_121870, more_types_in_union_121871) = may_be_none(width_121868, None_121869)

        if may_be_121870:

            if more_types_in_union_121871:
                # Runtime conditional SSA (line 525)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 526):
            
            # Assigning a Call to a Name (line 526):
            
            # Call to clip(...): (line 526)
            # Processing the call arguments (line 526)
            
            # Call to sqrt(...): (line 526)
            # Processing the call arguments (line 526)
            # Getting the type of 'self' (line 526)
            self_121876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 39), 'self', False)
            # Obtaining the member 'N' of a type (line 526)
            N_121877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 39), self_121876, 'N')
            # Processing the call keyword arguments (line 526)
            kwargs_121878 = {}
            # Getting the type of 'math' (line 526)
            math_121874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 29), 'math', False)
            # Obtaining the member 'sqrt' of a type (line 526)
            sqrt_121875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 29), math_121874, 'sqrt')
            # Calling sqrt(args, kwargs) (line 526)
            sqrt_call_result_121879 = invoke(stypy.reporting.localization.Localization(__file__, 526, 29), sqrt_121875, *[N_121877], **kwargs_121878)
            
            int_121880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 48), 'int')
            int_121881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 51), 'int')
            # Processing the call keyword arguments (line 526)
            kwargs_121882 = {}
            # Getting the type of 'np' (line 526)
            np_121872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 21), 'np', False)
            # Obtaining the member 'clip' of a type (line 526)
            clip_121873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 21), np_121872, 'clip')
            # Calling clip(args, kwargs) (line 526)
            clip_call_result_121883 = invoke(stypy.reporting.localization.Localization(__file__, 526, 21), clip_121873, *[sqrt_call_result_121879, int_121880, int_121881], **kwargs_121882)
            
            # Assigning a type to the variable 'sn' (line 526)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 16), 'sn', clip_call_result_121883)
            
            # Assigning a BinOp to a Attribute (line 527):
            
            # Assigning a BinOp to a Attribute (line 527):
            float_121884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 29), 'float')
            # Getting the type of 'self' (line 527)
            self_121885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 36), 'self')
            # Obtaining the member 'span' of a type (line 527)
            span_121886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 36), self_121885, 'span')
            # Applying the binary operator '*' (line 527)
            result_mul_121887 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 29), '*', float_121884, span_121886)
            
            # Getting the type of 'sn' (line 527)
            sn_121888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 48), 'sn')
            # Applying the binary operator 'div' (line 527)
            result_div_121889 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 46), 'div', result_mul_121887, sn_121888)
            
            # Getting the type of 'self' (line 527)
            self_121890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 16), 'self')
            # Setting the type of the member 'width' of a type (line 527)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 16), self_121890, 'width', result_div_121889)

            if more_types_in_union_121871:
                # SSA join for if statement (line 525)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 530)
        self_121891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 19), 'self')
        # Obtaining the member '_initialized' of a type (line 530)
        _initialized_121892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 19), self_121891, '_initialized')
        # Applying the 'not' unary operator (line 530)
        result_not__121893 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 15), 'not', _initialized_121892)
        
        
        # Getting the type of 'self' (line 530)
        self_121894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 41), 'self')
        # Obtaining the member 'scale' of a type (line 530)
        scale_121895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 41), self_121894, 'scale')
        # Getting the type of 'None' (line 530)
        None_121896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 55), 'None')
        # Applying the binary operator 'is' (line 530)
        result_is__121897 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 41), 'is', scale_121895, None_121896)
        
        # Applying the binary operator 'and' (line 530)
        result_and_keyword_121898 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 15), 'and', result_not__121893, result_is__121897)
        
        # Testing the type of an if condition (line 530)
        if_condition_121899 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 530, 12), result_and_keyword_121898)
        # Assigning a type to the variable 'if_condition_121899' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'if_condition_121899', if_condition_121899)
        # SSA begins for if statement (line 530)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _make_verts(...): (line 531)
        # Processing the call arguments (line 531)
        # Getting the type of 'self' (line 531)
        self_121902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 33), 'self', False)
        # Obtaining the member 'U' of a type (line 531)
        U_121903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 33), self_121902, 'U')
        # Getting the type of 'self' (line 531)
        self_121904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 41), 'self', False)
        # Obtaining the member 'V' of a type (line 531)
        V_121905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 41), self_121904, 'V')
        # Getting the type of 'self' (line 531)
        self_121906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 49), 'self', False)
        # Obtaining the member 'angles' of a type (line 531)
        angles_121907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 49), self_121906, 'angles')
        # Processing the call keyword arguments (line 531)
        kwargs_121908 = {}
        # Getting the type of 'self' (line 531)
        self_121900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 16), 'self', False)
        # Obtaining the member '_make_verts' of a type (line 531)
        _make_verts_121901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 16), self_121900, '_make_verts')
        # Calling _make_verts(args, kwargs) (line 531)
        _make_verts_call_result_121909 = invoke(stypy.reporting.localization.Localization(__file__, 531, 16), _make_verts_121901, *[U_121903, V_121905, angles_121907], **kwargs_121908)
        
        # SSA join for if statement (line 530)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 533):
        
        # Assigning a Name to a Attribute (line 533):
        # Getting the type of 'True' (line 533)
        True_121910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 32), 'True')
        # Getting the type of 'self' (line 533)
        self_121911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'self')
        # Setting the type of the member '_initialized' of a type (line 533)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 12), self_121911, '_initialized', True_121910)
        # SSA join for if statement (line 519)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init' in the type store
        # Getting the type of 'stypy_return_type' (line 512)
        stypy_return_type_121912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121912)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init'
        return stypy_return_type_121912


    @norecursion
    def get_datalim(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_datalim'
        module_type_store = module_type_store.open_function_context('get_datalim', 535, 4, False)
        # Assigning a type to the variable 'self' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Quiver.get_datalim.__dict__.__setitem__('stypy_localization', localization)
        Quiver.get_datalim.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Quiver.get_datalim.__dict__.__setitem__('stypy_type_store', module_type_store)
        Quiver.get_datalim.__dict__.__setitem__('stypy_function_name', 'Quiver.get_datalim')
        Quiver.get_datalim.__dict__.__setitem__('stypy_param_names_list', ['transData'])
        Quiver.get_datalim.__dict__.__setitem__('stypy_varargs_param_name', None)
        Quiver.get_datalim.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Quiver.get_datalim.__dict__.__setitem__('stypy_call_defaults', defaults)
        Quiver.get_datalim.__dict__.__setitem__('stypy_call_varargs', varargs)
        Quiver.get_datalim.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Quiver.get_datalim.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Quiver.get_datalim', ['transData'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_datalim', localization, ['transData'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_datalim(...)' code ##################

        
        # Assigning a Call to a Name (line 536):
        
        # Assigning a Call to a Name (line 536):
        
        # Call to get_transform(...): (line 536)
        # Processing the call keyword arguments (line 536)
        kwargs_121915 = {}
        # Getting the type of 'self' (line 536)
        self_121913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'self', False)
        # Obtaining the member 'get_transform' of a type (line 536)
        get_transform_121914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 16), self_121913, 'get_transform')
        # Calling get_transform(args, kwargs) (line 536)
        get_transform_call_result_121916 = invoke(stypy.reporting.localization.Localization(__file__, 536, 16), get_transform_121914, *[], **kwargs_121915)
        
        # Assigning a type to the variable 'trans' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'trans', get_transform_call_result_121916)
        
        # Assigning a Call to a Name (line 537):
        
        # Assigning a Call to a Name (line 537):
        
        # Call to get_offset_transform(...): (line 537)
        # Processing the call keyword arguments (line 537)
        kwargs_121919 = {}
        # Getting the type of 'self' (line 537)
        self_121917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 22), 'self', False)
        # Obtaining the member 'get_offset_transform' of a type (line 537)
        get_offset_transform_121918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 22), self_121917, 'get_offset_transform')
        # Calling get_offset_transform(args, kwargs) (line 537)
        get_offset_transform_call_result_121920 = invoke(stypy.reporting.localization.Localization(__file__, 537, 22), get_offset_transform_121918, *[], **kwargs_121919)
        
        # Assigning a type to the variable 'transOffset' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'transOffset', get_offset_transform_call_result_121920)
        
        # Assigning a BinOp to a Name (line 538):
        
        # Assigning a BinOp to a Name (line 538):
        # Getting the type of 'trans' (line 538)
        trans_121921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 26), 'trans')
        # Getting the type of 'transData' (line 538)
        transData_121922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 34), 'transData')
        # Applying the binary operator '-' (line 538)
        result_sub_121923 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 26), '-', trans_121921, transData_121922)
        
        # Getting the type of 'transOffset' (line 538)
        transOffset_121924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 48), 'transOffset')
        # Getting the type of 'transData' (line 538)
        transData_121925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 62), 'transData')
        # Applying the binary operator '-' (line 538)
        result_sub_121926 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 48), '-', transOffset_121924, transData_121925)
        
        # Applying the binary operator '+' (line 538)
        result_add_121927 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 25), '+', result_sub_121923, result_sub_121926)
        
        # Assigning a type to the variable 'full_transform' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'full_transform', result_add_121927)
        
        # Assigning a Call to a Name (line 539):
        
        # Assigning a Call to a Name (line 539):
        
        # Call to transform(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'self' (line 539)
        self_121930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 38), 'self', False)
        # Obtaining the member 'XY' of a type (line 539)
        XY_121931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 38), self_121930, 'XY')
        # Processing the call keyword arguments (line 539)
        kwargs_121932 = {}
        # Getting the type of 'full_transform' (line 539)
        full_transform_121928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 13), 'full_transform', False)
        # Obtaining the member 'transform' of a type (line 539)
        transform_121929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 13), full_transform_121928, 'transform')
        # Calling transform(args, kwargs) (line 539)
        transform_call_result_121933 = invoke(stypy.reporting.localization.Localization(__file__, 539, 13), transform_121929, *[XY_121931], **kwargs_121932)
        
        # Assigning a type to the variable 'XY' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'XY', transform_call_result_121933)
        
        # Assigning a Call to a Name (line 540):
        
        # Assigning a Call to a Name (line 540):
        
        # Call to null(...): (line 540)
        # Processing the call keyword arguments (line 540)
        kwargs_121937 = {}
        # Getting the type of 'transforms' (line 540)
        transforms_121934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 15), 'transforms', False)
        # Obtaining the member 'Bbox' of a type (line 540)
        Bbox_121935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 15), transforms_121934, 'Bbox')
        # Obtaining the member 'null' of a type (line 540)
        null_121936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 15), Bbox_121935, 'null')
        # Calling null(args, kwargs) (line 540)
        null_call_result_121938 = invoke(stypy.reporting.localization.Localization(__file__, 540, 15), null_121936, *[], **kwargs_121937)
        
        # Assigning a type to the variable 'bbox' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'bbox', null_call_result_121938)
        
        # Call to update_from_data_xy(...): (line 541)
        # Processing the call arguments (line 541)
        # Getting the type of 'XY' (line 541)
        XY_121941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 33), 'XY', False)
        # Processing the call keyword arguments (line 541)
        # Getting the type of 'True' (line 541)
        True_121942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 44), 'True', False)
        keyword_121943 = True_121942
        kwargs_121944 = {'ignore': keyword_121943}
        # Getting the type of 'bbox' (line 541)
        bbox_121939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'bbox', False)
        # Obtaining the member 'update_from_data_xy' of a type (line 541)
        update_from_data_xy_121940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 8), bbox_121939, 'update_from_data_xy')
        # Calling update_from_data_xy(args, kwargs) (line 541)
        update_from_data_xy_call_result_121945 = invoke(stypy.reporting.localization.Localization(__file__, 541, 8), update_from_data_xy_121940, *[XY_121941], **kwargs_121944)
        
        # Getting the type of 'bbox' (line 542)
        bbox_121946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 15), 'bbox')
        # Assigning a type to the variable 'stypy_return_type' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'stypy_return_type', bbox_121946)
        
        # ################# End of 'get_datalim(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_datalim' in the type store
        # Getting the type of 'stypy_return_type' (line 535)
        stypy_return_type_121947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121947)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_datalim'
        return stypy_return_type_121947


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 544, 4, False)
        # Assigning a type to the variable 'self' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Quiver.draw.__dict__.__setitem__('stypy_localization', localization)
        Quiver.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Quiver.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        Quiver.draw.__dict__.__setitem__('stypy_function_name', 'Quiver.draw')
        Quiver.draw.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Quiver.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        Quiver.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Quiver.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        Quiver.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        Quiver.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Quiver.draw.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Quiver.draw', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw(...)' code ##################

        
        # Call to _init(...): (line 546)
        # Processing the call keyword arguments (line 546)
        kwargs_121950 = {}
        # Getting the type of 'self' (line 546)
        self_121948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'self', False)
        # Obtaining the member '_init' of a type (line 546)
        _init_121949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 8), self_121948, '_init')
        # Calling _init(args, kwargs) (line 546)
        _init_call_result_121951 = invoke(stypy.reporting.localization.Localization(__file__, 546, 8), _init_121949, *[], **kwargs_121950)
        
        
        # Assigning a Call to a Name (line 547):
        
        # Assigning a Call to a Name (line 547):
        
        # Call to _make_verts(...): (line 547)
        # Processing the call arguments (line 547)
        # Getting the type of 'self' (line 547)
        self_121954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 33), 'self', False)
        # Obtaining the member 'U' of a type (line 547)
        U_121955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 33), self_121954, 'U')
        # Getting the type of 'self' (line 547)
        self_121956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 41), 'self', False)
        # Obtaining the member 'V' of a type (line 547)
        V_121957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 41), self_121956, 'V')
        # Getting the type of 'self' (line 547)
        self_121958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 49), 'self', False)
        # Obtaining the member 'angles' of a type (line 547)
        angles_121959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 49), self_121958, 'angles')
        # Processing the call keyword arguments (line 547)
        kwargs_121960 = {}
        # Getting the type of 'self' (line 547)
        self_121952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'self', False)
        # Obtaining the member '_make_verts' of a type (line 547)
        _make_verts_121953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 16), self_121952, '_make_verts')
        # Calling _make_verts(args, kwargs) (line 547)
        _make_verts_call_result_121961 = invoke(stypy.reporting.localization.Localization(__file__, 547, 16), _make_verts_121953, *[U_121955, V_121957, angles_121959], **kwargs_121960)
        
        # Assigning a type to the variable 'verts' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'verts', _make_verts_call_result_121961)
        
        # Call to set_verts(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'verts' (line 548)
        verts_121964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 23), 'verts', False)
        # Processing the call keyword arguments (line 548)
        # Getting the type of 'False' (line 548)
        False_121965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 37), 'False', False)
        keyword_121966 = False_121965
        kwargs_121967 = {'closed': keyword_121966}
        # Getting the type of 'self' (line 548)
        self_121962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'self', False)
        # Obtaining the member 'set_verts' of a type (line 548)
        set_verts_121963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 8), self_121962, 'set_verts')
        # Calling set_verts(args, kwargs) (line 548)
        set_verts_call_result_121968 = invoke(stypy.reporting.localization.Localization(__file__, 548, 8), set_verts_121963, *[verts_121964], **kwargs_121967)
        
        
        # Assigning a Name to a Attribute (line 549):
        
        # Assigning a Name to a Attribute (line 549):
        # Getting the type of 'False' (line 549)
        False_121969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 23), 'False')
        # Getting the type of 'self' (line 549)
        self_121970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'self')
        # Setting the type of the member '_new_UV' of a type (line 549)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 8), self_121970, '_new_UV', False_121969)
        
        # Call to draw(...): (line 550)
        # Processing the call arguments (line 550)
        # Getting the type of 'self' (line 550)
        self_121974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 41), 'self', False)
        # Getting the type of 'renderer' (line 550)
        renderer_121975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 47), 'renderer', False)
        # Processing the call keyword arguments (line 550)
        kwargs_121976 = {}
        # Getting the type of 'mcollections' (line 550)
        mcollections_121971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'mcollections', False)
        # Obtaining the member 'PolyCollection' of a type (line 550)
        PolyCollection_121972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), mcollections_121971, 'PolyCollection')
        # Obtaining the member 'draw' of a type (line 550)
        draw_121973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), PolyCollection_121972, 'draw')
        # Calling draw(args, kwargs) (line 550)
        draw_call_result_121977 = invoke(stypy.reporting.localization.Localization(__file__, 550, 8), draw_121973, *[self_121974, renderer_121975], **kwargs_121976)
        
        
        # Assigning a Name to a Attribute (line 551):
        
        # Assigning a Name to a Attribute (line 551):
        # Getting the type of 'False' (line 551)
        False_121978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 21), 'False')
        # Getting the type of 'self' (line 551)
        self_121979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 551)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 8), self_121979, 'stale', False_121978)
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 544)
        stypy_return_type_121980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121980)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_121980


    @norecursion
    def set_UVC(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 553)
        None_121981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 30), 'None')
        defaults = [None_121981]
        # Create a new context for function 'set_UVC'
        module_type_store = module_type_store.open_function_context('set_UVC', 553, 4, False)
        # Assigning a type to the variable 'self' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Quiver.set_UVC.__dict__.__setitem__('stypy_localization', localization)
        Quiver.set_UVC.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Quiver.set_UVC.__dict__.__setitem__('stypy_type_store', module_type_store)
        Quiver.set_UVC.__dict__.__setitem__('stypy_function_name', 'Quiver.set_UVC')
        Quiver.set_UVC.__dict__.__setitem__('stypy_param_names_list', ['U', 'V', 'C'])
        Quiver.set_UVC.__dict__.__setitem__('stypy_varargs_param_name', None)
        Quiver.set_UVC.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Quiver.set_UVC.__dict__.__setitem__('stypy_call_defaults', defaults)
        Quiver.set_UVC.__dict__.__setitem__('stypy_call_varargs', varargs)
        Quiver.set_UVC.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Quiver.set_UVC.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Quiver.set_UVC', ['U', 'V', 'C'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_UVC', localization, ['U', 'V', 'C'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_UVC(...)' code ##################

        
        # Assigning a Call to a Name (line 556):
        
        # Assigning a Call to a Name (line 556):
        
        # Call to ravel(...): (line 556)
        # Processing the call keyword arguments (line 556)
        kwargs_121990 = {}
        
        # Call to masked_invalid(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'U' (line 556)
        U_121984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 30), 'U', False)
        # Processing the call keyword arguments (line 556)
        # Getting the type of 'True' (line 556)
        True_121985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 38), 'True', False)
        keyword_121986 = True_121985
        kwargs_121987 = {'copy': keyword_121986}
        # Getting the type of 'ma' (line 556)
        ma_121982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'ma', False)
        # Obtaining the member 'masked_invalid' of a type (line 556)
        masked_invalid_121983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 12), ma_121982, 'masked_invalid')
        # Calling masked_invalid(args, kwargs) (line 556)
        masked_invalid_call_result_121988 = invoke(stypy.reporting.localization.Localization(__file__, 556, 12), masked_invalid_121983, *[U_121984], **kwargs_121987)
        
        # Obtaining the member 'ravel' of a type (line 556)
        ravel_121989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 12), masked_invalid_call_result_121988, 'ravel')
        # Calling ravel(args, kwargs) (line 556)
        ravel_call_result_121991 = invoke(stypy.reporting.localization.Localization(__file__, 556, 12), ravel_121989, *[], **kwargs_121990)
        
        # Assigning a type to the variable 'U' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'U', ravel_call_result_121991)
        
        # Assigning a Call to a Name (line 557):
        
        # Assigning a Call to a Name (line 557):
        
        # Call to ravel(...): (line 557)
        # Processing the call keyword arguments (line 557)
        kwargs_122000 = {}
        
        # Call to masked_invalid(...): (line 557)
        # Processing the call arguments (line 557)
        # Getting the type of 'V' (line 557)
        V_121994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 30), 'V', False)
        # Processing the call keyword arguments (line 557)
        # Getting the type of 'True' (line 557)
        True_121995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 38), 'True', False)
        keyword_121996 = True_121995
        kwargs_121997 = {'copy': keyword_121996}
        # Getting the type of 'ma' (line 557)
        ma_121992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'ma', False)
        # Obtaining the member 'masked_invalid' of a type (line 557)
        masked_invalid_121993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 12), ma_121992, 'masked_invalid')
        # Calling masked_invalid(args, kwargs) (line 557)
        masked_invalid_call_result_121998 = invoke(stypy.reporting.localization.Localization(__file__, 557, 12), masked_invalid_121993, *[V_121994], **kwargs_121997)
        
        # Obtaining the member 'ravel' of a type (line 557)
        ravel_121999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 12), masked_invalid_call_result_121998, 'ravel')
        # Calling ravel(args, kwargs) (line 557)
        ravel_call_result_122001 = invoke(stypy.reporting.localization.Localization(__file__, 557, 12), ravel_121999, *[], **kwargs_122000)
        
        # Assigning a type to the variable 'V' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'V', ravel_call_result_122001)
        
        # Assigning a Call to a Name (line 558):
        
        # Assigning a Call to a Name (line 558):
        
        # Call to mask_or(...): (line 558)
        # Processing the call arguments (line 558)
        # Getting the type of 'U' (line 558)
        U_122004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 26), 'U', False)
        # Obtaining the member 'mask' of a type (line 558)
        mask_122005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 26), U_122004, 'mask')
        # Getting the type of 'V' (line 558)
        V_122006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 34), 'V', False)
        # Obtaining the member 'mask' of a type (line 558)
        mask_122007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 34), V_122006, 'mask')
        # Processing the call keyword arguments (line 558)
        # Getting the type of 'False' (line 558)
        False_122008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 47), 'False', False)
        keyword_122009 = False_122008
        # Getting the type of 'True' (line 558)
        True_122010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 61), 'True', False)
        keyword_122011 = True_122010
        kwargs_122012 = {'copy': keyword_122009, 'shrink': keyword_122011}
        # Getting the type of 'ma' (line 558)
        ma_122002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 15), 'ma', False)
        # Obtaining the member 'mask_or' of a type (line 558)
        mask_or_122003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 15), ma_122002, 'mask_or')
        # Calling mask_or(args, kwargs) (line 558)
        mask_or_call_result_122013 = invoke(stypy.reporting.localization.Localization(__file__, 558, 15), mask_or_122003, *[mask_122005, mask_122007], **kwargs_122012)
        
        # Assigning a type to the variable 'mask' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'mask', mask_or_call_result_122013)
        
        # Type idiom detected: calculating its left and rigth part (line 559)
        # Getting the type of 'C' (line 559)
        C_122014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'C')
        # Getting the type of 'None' (line 559)
        None_122015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 20), 'None')
        
        (may_be_122016, more_types_in_union_122017) = may_not_be_none(C_122014, None_122015)

        if may_be_122016:

            if more_types_in_union_122017:
                # Runtime conditional SSA (line 559)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 560):
            
            # Assigning a Call to a Name (line 560):
            
            # Call to ravel(...): (line 560)
            # Processing the call keyword arguments (line 560)
            kwargs_122026 = {}
            
            # Call to masked_invalid(...): (line 560)
            # Processing the call arguments (line 560)
            # Getting the type of 'C' (line 560)
            C_122020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 34), 'C', False)
            # Processing the call keyword arguments (line 560)
            # Getting the type of 'True' (line 560)
            True_122021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 42), 'True', False)
            keyword_122022 = True_122021
            kwargs_122023 = {'copy': keyword_122022}
            # Getting the type of 'ma' (line 560)
            ma_122018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'ma', False)
            # Obtaining the member 'masked_invalid' of a type (line 560)
            masked_invalid_122019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 16), ma_122018, 'masked_invalid')
            # Calling masked_invalid(args, kwargs) (line 560)
            masked_invalid_call_result_122024 = invoke(stypy.reporting.localization.Localization(__file__, 560, 16), masked_invalid_122019, *[C_122020], **kwargs_122023)
            
            # Obtaining the member 'ravel' of a type (line 560)
            ravel_122025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 16), masked_invalid_call_result_122024, 'ravel')
            # Calling ravel(args, kwargs) (line 560)
            ravel_call_result_122027 = invoke(stypy.reporting.localization.Localization(__file__, 560, 16), ravel_122025, *[], **kwargs_122026)
            
            # Assigning a type to the variable 'C' (line 560)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'C', ravel_call_result_122027)
            
            # Assigning a Call to a Name (line 561):
            
            # Assigning a Call to a Name (line 561):
            
            # Call to mask_or(...): (line 561)
            # Processing the call arguments (line 561)
            # Getting the type of 'mask' (line 561)
            mask_122030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 30), 'mask', False)
            # Getting the type of 'C' (line 561)
            C_122031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 36), 'C', False)
            # Obtaining the member 'mask' of a type (line 561)
            mask_122032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 36), C_122031, 'mask')
            # Processing the call keyword arguments (line 561)
            # Getting the type of 'False' (line 561)
            False_122033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 49), 'False', False)
            keyword_122034 = False_122033
            # Getting the type of 'True' (line 561)
            True_122035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 63), 'True', False)
            keyword_122036 = True_122035
            kwargs_122037 = {'copy': keyword_122034, 'shrink': keyword_122036}
            # Getting the type of 'ma' (line 561)
            ma_122028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 19), 'ma', False)
            # Obtaining the member 'mask_or' of a type (line 561)
            mask_or_122029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 19), ma_122028, 'mask_or')
            # Calling mask_or(args, kwargs) (line 561)
            mask_or_call_result_122038 = invoke(stypy.reporting.localization.Localization(__file__, 561, 19), mask_or_122029, *[mask_122030, mask_122032], **kwargs_122037)
            
            # Assigning a type to the variable 'mask' (line 561)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'mask', mask_or_call_result_122038)
            
            
            # Getting the type of 'mask' (line 562)
            mask_122039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 15), 'mask')
            # Getting the type of 'ma' (line 562)
            ma_122040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 23), 'ma')
            # Obtaining the member 'nomask' of a type (line 562)
            nomask_122041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 23), ma_122040, 'nomask')
            # Applying the binary operator 'is' (line 562)
            result_is__122042 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 15), 'is', mask_122039, nomask_122041)
            
            # Testing the type of an if condition (line 562)
            if_condition_122043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 12), result_is__122042)
            # Assigning a type to the variable 'if_condition_122043' (line 562)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'if_condition_122043', if_condition_122043)
            # SSA begins for if statement (line 562)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 563):
            
            # Assigning a Call to a Name (line 563):
            
            # Call to filled(...): (line 563)
            # Processing the call keyword arguments (line 563)
            kwargs_122046 = {}
            # Getting the type of 'C' (line 563)
            C_122044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 20), 'C', False)
            # Obtaining the member 'filled' of a type (line 563)
            filled_122045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 20), C_122044, 'filled')
            # Calling filled(args, kwargs) (line 563)
            filled_call_result_122047 = invoke(stypy.reporting.localization.Localization(__file__, 563, 20), filled_122045, *[], **kwargs_122046)
            
            # Assigning a type to the variable 'C' (line 563)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 16), 'C', filled_call_result_122047)
            # SSA branch for the else part of an if statement (line 562)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 565):
            
            # Assigning a Call to a Name (line 565):
            
            # Call to array(...): (line 565)
            # Processing the call arguments (line 565)
            # Getting the type of 'C' (line 565)
            C_122050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 29), 'C', False)
            # Processing the call keyword arguments (line 565)
            # Getting the type of 'mask' (line 565)
            mask_122051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 37), 'mask', False)
            keyword_122052 = mask_122051
            # Getting the type of 'False' (line 565)
            False_122053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 48), 'False', False)
            keyword_122054 = False_122053
            kwargs_122055 = {'copy': keyword_122054, 'mask': keyword_122052}
            # Getting the type of 'ma' (line 565)
            ma_122048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 20), 'ma', False)
            # Obtaining the member 'array' of a type (line 565)
            array_122049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 20), ma_122048, 'array')
            # Calling array(args, kwargs) (line 565)
            array_call_result_122056 = invoke(stypy.reporting.localization.Localization(__file__, 565, 20), array_122049, *[C_122050], **kwargs_122055)
            
            # Assigning a type to the variable 'C' (line 565)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'C', array_call_result_122056)
            # SSA join for if statement (line 562)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_122017:
                # SSA join for if statement (line 559)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 566):
        
        # Assigning a Call to a Attribute (line 566):
        
        # Call to filled(...): (line 566)
        # Processing the call arguments (line 566)
        int_122059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 26), 'int')
        # Processing the call keyword arguments (line 566)
        kwargs_122060 = {}
        # Getting the type of 'U' (line 566)
        U_122057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 17), 'U', False)
        # Obtaining the member 'filled' of a type (line 566)
        filled_122058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 17), U_122057, 'filled')
        # Calling filled(args, kwargs) (line 566)
        filled_call_result_122061 = invoke(stypy.reporting.localization.Localization(__file__, 566, 17), filled_122058, *[int_122059], **kwargs_122060)
        
        # Getting the type of 'self' (line 566)
        self_122062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'self')
        # Setting the type of the member 'U' of a type (line 566)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 8), self_122062, 'U', filled_call_result_122061)
        
        # Assigning a Call to a Attribute (line 567):
        
        # Assigning a Call to a Attribute (line 567):
        
        # Call to filled(...): (line 567)
        # Processing the call arguments (line 567)
        int_122065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 26), 'int')
        # Processing the call keyword arguments (line 567)
        kwargs_122066 = {}
        # Getting the type of 'V' (line 567)
        V_122063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 17), 'V', False)
        # Obtaining the member 'filled' of a type (line 567)
        filled_122064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 17), V_122063, 'filled')
        # Calling filled(args, kwargs) (line 567)
        filled_call_result_122067 = invoke(stypy.reporting.localization.Localization(__file__, 567, 17), filled_122064, *[int_122065], **kwargs_122066)
        
        # Getting the type of 'self' (line 567)
        self_122068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'self')
        # Setting the type of the member 'V' of a type (line 567)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 8), self_122068, 'V', filled_call_result_122067)
        
        # Assigning a Name to a Attribute (line 568):
        
        # Assigning a Name to a Attribute (line 568):
        # Getting the type of 'mask' (line 568)
        mask_122069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 21), 'mask')
        # Getting the type of 'self' (line 568)
        self_122070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'self')
        # Setting the type of the member 'Umask' of a type (line 568)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 8), self_122070, 'Umask', mask_122069)
        
        # Type idiom detected: calculating its left and rigth part (line 569)
        # Getting the type of 'C' (line 569)
        C_122071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'C')
        # Getting the type of 'None' (line 569)
        None_122072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 20), 'None')
        
        (may_be_122073, more_types_in_union_122074) = may_not_be_none(C_122071, None_122072)

        if may_be_122073:

            if more_types_in_union_122074:
                # Runtime conditional SSA (line 569)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to set_array(...): (line 570)
            # Processing the call arguments (line 570)
            # Getting the type of 'C' (line 570)
            C_122077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 27), 'C', False)
            # Processing the call keyword arguments (line 570)
            kwargs_122078 = {}
            # Getting the type of 'self' (line 570)
            self_122075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'self', False)
            # Obtaining the member 'set_array' of a type (line 570)
            set_array_122076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 12), self_122075, 'set_array')
            # Calling set_array(args, kwargs) (line 570)
            set_array_call_result_122079 = invoke(stypy.reporting.localization.Localization(__file__, 570, 12), set_array_122076, *[C_122077], **kwargs_122078)
            

            if more_types_in_union_122074:
                # SSA join for if statement (line 569)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 571):
        
        # Assigning a Name to a Attribute (line 571):
        # Getting the type of 'True' (line 571)
        True_122080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 23), 'True')
        # Getting the type of 'self' (line 571)
        self_122081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'self')
        # Setting the type of the member '_new_UV' of a type (line 571)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 8), self_122081, '_new_UV', True_122080)
        
        # Assigning a Name to a Attribute (line 572):
        
        # Assigning a Name to a Attribute (line 572):
        # Getting the type of 'True' (line 572)
        True_122082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 21), 'True')
        # Getting the type of 'self' (line 572)
        self_122083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 572)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 8), self_122083, 'stale', True_122082)
        
        # ################# End of 'set_UVC(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_UVC' in the type store
        # Getting the type of 'stypy_return_type' (line 553)
        stypy_return_type_122084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_UVC'
        return stypy_return_type_122084


    @norecursion
    def _dots_per_unit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dots_per_unit'
        module_type_store = module_type_store.open_function_context('_dots_per_unit', 574, 4, False)
        # Assigning a type to the variable 'self' (line 575)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Quiver._dots_per_unit.__dict__.__setitem__('stypy_localization', localization)
        Quiver._dots_per_unit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Quiver._dots_per_unit.__dict__.__setitem__('stypy_type_store', module_type_store)
        Quiver._dots_per_unit.__dict__.__setitem__('stypy_function_name', 'Quiver._dots_per_unit')
        Quiver._dots_per_unit.__dict__.__setitem__('stypy_param_names_list', ['units'])
        Quiver._dots_per_unit.__dict__.__setitem__('stypy_varargs_param_name', None)
        Quiver._dots_per_unit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Quiver._dots_per_unit.__dict__.__setitem__('stypy_call_defaults', defaults)
        Quiver._dots_per_unit.__dict__.__setitem__('stypy_call_varargs', varargs)
        Quiver._dots_per_unit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Quiver._dots_per_unit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Quiver._dots_per_unit', ['units'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dots_per_unit', localization, ['units'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dots_per_unit(...)' code ##################

        unicode_122085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, (-1)), 'unicode', u'\n        Return a scale factor for converting from units to pixels\n        ')
        
        # Assigning a Attribute to a Name (line 578):
        
        # Assigning a Attribute to a Name (line 578):
        # Getting the type of 'self' (line 578)
        self_122086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 13), 'self')
        # Obtaining the member 'ax' of a type (line 578)
        ax_122087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 13), self_122086, 'ax')
        # Assigning a type to the variable 'ax' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'ax', ax_122087)
        
        
        # Getting the type of 'units' (line 579)
        units_122088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 11), 'units')
        
        # Obtaining an instance of the builtin type 'tuple' (line 579)
        tuple_122089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 579)
        # Adding element type (line 579)
        unicode_122090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 21), 'unicode', u'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 21), tuple_122089, unicode_122090)
        # Adding element type (line 579)
        unicode_122091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 26), 'unicode', u'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 21), tuple_122089, unicode_122091)
        # Adding element type (line 579)
        unicode_122092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 31), 'unicode', u'xy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 21), tuple_122089, unicode_122092)
        
        # Applying the binary operator 'in' (line 579)
        result_contains_122093 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 11), 'in', units_122088, tuple_122089)
        
        # Testing the type of an if condition (line 579)
        if_condition_122094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 579, 8), result_contains_122093)
        # Assigning a type to the variable 'if_condition_122094' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'if_condition_122094', if_condition_122094)
        # SSA begins for if statement (line 579)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'units' (line 580)
        units_122095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 15), 'units')
        unicode_122096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 24), 'unicode', u'x')
        # Applying the binary operator '==' (line 580)
        result_eq_122097 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 15), '==', units_122095, unicode_122096)
        
        # Testing the type of an if condition (line 580)
        if_condition_122098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 580, 12), result_eq_122097)
        # Assigning a type to the variable 'if_condition_122098' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'if_condition_122098', if_condition_122098)
        # SSA begins for if statement (line 580)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 581):
        
        # Assigning a Attribute to a Name (line 581):
        # Getting the type of 'ax' (line 581)
        ax_122099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 22), 'ax')
        # Obtaining the member 'viewLim' of a type (line 581)
        viewLim_122100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 22), ax_122099, 'viewLim')
        # Obtaining the member 'width' of a type (line 581)
        width_122101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 22), viewLim_122100, 'width')
        # Assigning a type to the variable 'dx0' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'dx0', width_122101)
        
        # Assigning a Attribute to a Name (line 582):
        
        # Assigning a Attribute to a Name (line 582):
        # Getting the type of 'ax' (line 582)
        ax_122102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 22), 'ax')
        # Obtaining the member 'bbox' of a type (line 582)
        bbox_122103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 22), ax_122102, 'bbox')
        # Obtaining the member 'width' of a type (line 582)
        width_122104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 22), bbox_122103, 'width')
        # Assigning a type to the variable 'dx1' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'dx1', width_122104)
        # SSA branch for the else part of an if statement (line 580)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'units' (line 583)
        units_122105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 17), 'units')
        unicode_122106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 26), 'unicode', u'y')
        # Applying the binary operator '==' (line 583)
        result_eq_122107 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 17), '==', units_122105, unicode_122106)
        
        # Testing the type of an if condition (line 583)
        if_condition_122108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 583, 17), result_eq_122107)
        # Assigning a type to the variable 'if_condition_122108' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 17), 'if_condition_122108', if_condition_122108)
        # SSA begins for if statement (line 583)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 584):
        
        # Assigning a Attribute to a Name (line 584):
        # Getting the type of 'ax' (line 584)
        ax_122109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 22), 'ax')
        # Obtaining the member 'viewLim' of a type (line 584)
        viewLim_122110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 22), ax_122109, 'viewLim')
        # Obtaining the member 'height' of a type (line 584)
        height_122111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 22), viewLim_122110, 'height')
        # Assigning a type to the variable 'dx0' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'dx0', height_122111)
        
        # Assigning a Attribute to a Name (line 585):
        
        # Assigning a Attribute to a Name (line 585):
        # Getting the type of 'ax' (line 585)
        ax_122112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 22), 'ax')
        # Obtaining the member 'bbox' of a type (line 585)
        bbox_122113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 22), ax_122112, 'bbox')
        # Obtaining the member 'height' of a type (line 585)
        height_122114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 22), bbox_122113, 'height')
        # Assigning a type to the variable 'dx1' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 16), 'dx1', height_122114)
        # SSA branch for the else part of an if statement (line 583)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 587):
        
        # Assigning a Attribute to a Name (line 587):
        # Getting the type of 'ax' (line 587)
        ax_122115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 23), 'ax')
        # Obtaining the member 'viewLim' of a type (line 587)
        viewLim_122116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 23), ax_122115, 'viewLim')
        # Obtaining the member 'width' of a type (line 587)
        width_122117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 23), viewLim_122116, 'width')
        # Assigning a type to the variable 'dxx0' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'dxx0', width_122117)
        
        # Assigning a Attribute to a Name (line 588):
        
        # Assigning a Attribute to a Name (line 588):
        # Getting the type of 'ax' (line 588)
        ax_122118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 23), 'ax')
        # Obtaining the member 'bbox' of a type (line 588)
        bbox_122119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 23), ax_122118, 'bbox')
        # Obtaining the member 'width' of a type (line 588)
        width_122120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 23), bbox_122119, 'width')
        # Assigning a type to the variable 'dxx1' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 16), 'dxx1', width_122120)
        
        # Assigning a Attribute to a Name (line 589):
        
        # Assigning a Attribute to a Name (line 589):
        # Getting the type of 'ax' (line 589)
        ax_122121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 23), 'ax')
        # Obtaining the member 'viewLim' of a type (line 589)
        viewLim_122122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 23), ax_122121, 'viewLim')
        # Obtaining the member 'height' of a type (line 589)
        height_122123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 23), viewLim_122122, 'height')
        # Assigning a type to the variable 'dyy0' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'dyy0', height_122123)
        
        # Assigning a Attribute to a Name (line 590):
        
        # Assigning a Attribute to a Name (line 590):
        # Getting the type of 'ax' (line 590)
        ax_122124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 23), 'ax')
        # Obtaining the member 'bbox' of a type (line 590)
        bbox_122125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 23), ax_122124, 'bbox')
        # Obtaining the member 'height' of a type (line 590)
        height_122126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 23), bbox_122125, 'height')
        # Assigning a type to the variable 'dyy1' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 16), 'dyy1', height_122126)
        
        # Assigning a Call to a Name (line 591):
        
        # Assigning a Call to a Name (line 591):
        
        # Call to hypot(...): (line 591)
        # Processing the call arguments (line 591)
        # Getting the type of 'dxx1' (line 591)
        dxx1_122129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 31), 'dxx1', False)
        # Getting the type of 'dyy1' (line 591)
        dyy1_122130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 37), 'dyy1', False)
        # Processing the call keyword arguments (line 591)
        kwargs_122131 = {}
        # Getting the type of 'np' (line 591)
        np_122127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 22), 'np', False)
        # Obtaining the member 'hypot' of a type (line 591)
        hypot_122128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 22), np_122127, 'hypot')
        # Calling hypot(args, kwargs) (line 591)
        hypot_call_result_122132 = invoke(stypy.reporting.localization.Localization(__file__, 591, 22), hypot_122128, *[dxx1_122129, dyy1_122130], **kwargs_122131)
        
        # Assigning a type to the variable 'dx1' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 16), 'dx1', hypot_call_result_122132)
        
        # Assigning a Call to a Name (line 592):
        
        # Assigning a Call to a Name (line 592):
        
        # Call to hypot(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'dxx0' (line 592)
        dxx0_122135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 31), 'dxx0', False)
        # Getting the type of 'dyy0' (line 592)
        dyy0_122136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 37), 'dyy0', False)
        # Processing the call keyword arguments (line 592)
        kwargs_122137 = {}
        # Getting the type of 'np' (line 592)
        np_122133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 22), 'np', False)
        # Obtaining the member 'hypot' of a type (line 592)
        hypot_122134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 22), np_122133, 'hypot')
        # Calling hypot(args, kwargs) (line 592)
        hypot_call_result_122138 = invoke(stypy.reporting.localization.Localization(__file__, 592, 22), hypot_122134, *[dxx0_122135, dyy0_122136], **kwargs_122137)
        
        # Assigning a type to the variable 'dx0' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'dx0', hypot_call_result_122138)
        # SSA join for if statement (line 583)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 580)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 593):
        
        # Assigning a BinOp to a Name (line 593):
        # Getting the type of 'dx1' (line 593)
        dx1_122139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 17), 'dx1')
        # Getting the type of 'dx0' (line 593)
        dx0_122140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 23), 'dx0')
        # Applying the binary operator 'div' (line 593)
        result_div_122141 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 17), 'div', dx1_122139, dx0_122140)
        
        # Assigning a type to the variable 'dx' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'dx', result_div_122141)
        # SSA branch for the else part of an if statement (line 579)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'units' (line 595)
        units_122142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 15), 'units')
        unicode_122143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 24), 'unicode', u'width')
        # Applying the binary operator '==' (line 595)
        result_eq_122144 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 15), '==', units_122142, unicode_122143)
        
        # Testing the type of an if condition (line 595)
        if_condition_122145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 595, 12), result_eq_122144)
        # Assigning a type to the variable 'if_condition_122145' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'if_condition_122145', if_condition_122145)
        # SSA begins for if statement (line 595)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 596):
        
        # Assigning a Attribute to a Name (line 596):
        # Getting the type of 'ax' (line 596)
        ax_122146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 21), 'ax')
        # Obtaining the member 'bbox' of a type (line 596)
        bbox_122147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 21), ax_122146, 'bbox')
        # Obtaining the member 'width' of a type (line 596)
        width_122148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 21), bbox_122147, 'width')
        # Assigning a type to the variable 'dx' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'dx', width_122148)
        # SSA branch for the else part of an if statement (line 595)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'units' (line 597)
        units_122149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 17), 'units')
        unicode_122150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 26), 'unicode', u'height')
        # Applying the binary operator '==' (line 597)
        result_eq_122151 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 17), '==', units_122149, unicode_122150)
        
        # Testing the type of an if condition (line 597)
        if_condition_122152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 17), result_eq_122151)
        # Assigning a type to the variable 'if_condition_122152' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 17), 'if_condition_122152', if_condition_122152)
        # SSA begins for if statement (line 597)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 598):
        
        # Assigning a Attribute to a Name (line 598):
        # Getting the type of 'ax' (line 598)
        ax_122153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 21), 'ax')
        # Obtaining the member 'bbox' of a type (line 598)
        bbox_122154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 21), ax_122153, 'bbox')
        # Obtaining the member 'height' of a type (line 598)
        height_122155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 21), bbox_122154, 'height')
        # Assigning a type to the variable 'dx' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'dx', height_122155)
        # SSA branch for the else part of an if statement (line 597)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'units' (line 599)
        units_122156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 17), 'units')
        unicode_122157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 26), 'unicode', u'dots')
        # Applying the binary operator '==' (line 599)
        result_eq_122158 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 17), '==', units_122156, unicode_122157)
        
        # Testing the type of an if condition (line 599)
        if_condition_122159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 599, 17), result_eq_122158)
        # Assigning a type to the variable 'if_condition_122159' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 17), 'if_condition_122159', if_condition_122159)
        # SSA begins for if statement (line 599)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 600):
        
        # Assigning a Num to a Name (line 600):
        float_122160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 21), 'float')
        # Assigning a type to the variable 'dx' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'dx', float_122160)
        # SSA branch for the else part of an if statement (line 599)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'units' (line 601)
        units_122161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 17), 'units')
        unicode_122162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 26), 'unicode', u'inches')
        # Applying the binary operator '==' (line 601)
        result_eq_122163 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 17), '==', units_122161, unicode_122162)
        
        # Testing the type of an if condition (line 601)
        if_condition_122164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 601, 17), result_eq_122163)
        # Assigning a type to the variable 'if_condition_122164' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 17), 'if_condition_122164', if_condition_122164)
        # SSA begins for if statement (line 601)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 602):
        
        # Assigning a Attribute to a Name (line 602):
        # Getting the type of 'ax' (line 602)
        ax_122165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 21), 'ax')
        # Obtaining the member 'figure' of a type (line 602)
        figure_122166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 21), ax_122165, 'figure')
        # Obtaining the member 'dpi' of a type (line 602)
        dpi_122167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 21), figure_122166, 'dpi')
        # Assigning a type to the variable 'dx' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 16), 'dx', dpi_122167)
        # SSA branch for the else part of an if statement (line 601)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 604)
        # Processing the call arguments (line 604)
        unicode_122169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 33), 'unicode', u'unrecognized units')
        # Processing the call keyword arguments (line 604)
        kwargs_122170 = {}
        # Getting the type of 'ValueError' (line 604)
        ValueError_122168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 604)
        ValueError_call_result_122171 = invoke(stypy.reporting.localization.Localization(__file__, 604, 22), ValueError_122168, *[unicode_122169], **kwargs_122170)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 604, 16), ValueError_call_result_122171, 'raise parameter', BaseException)
        # SSA join for if statement (line 601)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 599)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 597)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 595)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 579)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'dx' (line 605)
        dx_122172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 15), 'dx')
        # Assigning a type to the variable 'stypy_return_type' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'stypy_return_type', dx_122172)
        
        # ################# End of '_dots_per_unit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dots_per_unit' in the type store
        # Getting the type of 'stypy_return_type' (line 574)
        stypy_return_type_122173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dots_per_unit'
        return stypy_return_type_122173


    @norecursion
    def _set_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_transform'
        module_type_store = module_type_store.open_function_context('_set_transform', 607, 4, False)
        # Assigning a type to the variable 'self' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Quiver._set_transform.__dict__.__setitem__('stypy_localization', localization)
        Quiver._set_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Quiver._set_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        Quiver._set_transform.__dict__.__setitem__('stypy_function_name', 'Quiver._set_transform')
        Quiver._set_transform.__dict__.__setitem__('stypy_param_names_list', [])
        Quiver._set_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        Quiver._set_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Quiver._set_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        Quiver._set_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        Quiver._set_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Quiver._set_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Quiver._set_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_transform(...)' code ##################

        unicode_122174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, (-1)), 'unicode', u'\n        Sets the PolygonCollection transform to go\n        from arrow width units to pixels.\n        ')
        
        # Assigning a Call to a Name (line 612):
        
        # Assigning a Call to a Name (line 612):
        
        # Call to _dots_per_unit(...): (line 612)
        # Processing the call arguments (line 612)
        # Getting the type of 'self' (line 612)
        self_122177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 33), 'self', False)
        # Obtaining the member 'units' of a type (line 612)
        units_122178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 33), self_122177, 'units')
        # Processing the call keyword arguments (line 612)
        kwargs_122179 = {}
        # Getting the type of 'self' (line 612)
        self_122175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 13), 'self', False)
        # Obtaining the member '_dots_per_unit' of a type (line 612)
        _dots_per_unit_122176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 612, 13), self_122175, '_dots_per_unit')
        # Calling _dots_per_unit(args, kwargs) (line 612)
        _dots_per_unit_call_result_122180 = invoke(stypy.reporting.localization.Localization(__file__, 612, 13), _dots_per_unit_122176, *[units_122178], **kwargs_122179)
        
        # Assigning a type to the variable 'dx' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'dx', _dots_per_unit_call_result_122180)
        
        # Assigning a Name to a Attribute (line 613):
        
        # Assigning a Name to a Attribute (line 613):
        # Getting the type of 'dx' (line 613)
        dx_122181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 28), 'dx')
        # Getting the type of 'self' (line 613)
        self_122182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'self')
        # Setting the type of the member '_trans_scale' of a type (line 613)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 8), self_122182, '_trans_scale', dx_122181)
        
        # Assigning a Call to a Name (line 614):
        
        # Assigning a Call to a Name (line 614):
        
        # Call to scale(...): (line 614)
        # Processing the call arguments (line 614)
        # Getting the type of 'dx' (line 614)
        dx_122188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 44), 'dx', False)
        # Processing the call keyword arguments (line 614)
        kwargs_122189 = {}
        
        # Call to Affine2D(...): (line 614)
        # Processing the call keyword arguments (line 614)
        kwargs_122185 = {}
        # Getting the type of 'transforms' (line 614)
        transforms_122183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 16), 'transforms', False)
        # Obtaining the member 'Affine2D' of a type (line 614)
        Affine2D_122184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 16), transforms_122183, 'Affine2D')
        # Calling Affine2D(args, kwargs) (line 614)
        Affine2D_call_result_122186 = invoke(stypy.reporting.localization.Localization(__file__, 614, 16), Affine2D_122184, *[], **kwargs_122185)
        
        # Obtaining the member 'scale' of a type (line 614)
        scale_122187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 16), Affine2D_call_result_122186, 'scale')
        # Calling scale(args, kwargs) (line 614)
        scale_call_result_122190 = invoke(stypy.reporting.localization.Localization(__file__, 614, 16), scale_122187, *[dx_122188], **kwargs_122189)
        
        # Assigning a type to the variable 'trans' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'trans', scale_call_result_122190)
        
        # Call to set_transform(...): (line 615)
        # Processing the call arguments (line 615)
        # Getting the type of 'trans' (line 615)
        trans_122193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 27), 'trans', False)
        # Processing the call keyword arguments (line 615)
        kwargs_122194 = {}
        # Getting the type of 'self' (line 615)
        self_122191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'self', False)
        # Obtaining the member 'set_transform' of a type (line 615)
        set_transform_122192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 8), self_122191, 'set_transform')
        # Calling set_transform(args, kwargs) (line 615)
        set_transform_call_result_122195 = invoke(stypy.reporting.localization.Localization(__file__, 615, 8), set_transform_122192, *[trans_122193], **kwargs_122194)
        
        # Getting the type of 'trans' (line 616)
        trans_122196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 15), 'trans')
        # Assigning a type to the variable 'stypy_return_type' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'stypy_return_type', trans_122196)
        
        # ################# End of '_set_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 607)
        stypy_return_type_122197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122197)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_transform'
        return stypy_return_type_122197


    @norecursion
    def _angles_lengths(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_122198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 40), 'int')
        defaults = [int_122198]
        # Create a new context for function '_angles_lengths'
        module_type_store = module_type_store.open_function_context('_angles_lengths', 618, 4, False)
        # Assigning a type to the variable 'self' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Quiver._angles_lengths.__dict__.__setitem__('stypy_localization', localization)
        Quiver._angles_lengths.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Quiver._angles_lengths.__dict__.__setitem__('stypy_type_store', module_type_store)
        Quiver._angles_lengths.__dict__.__setitem__('stypy_function_name', 'Quiver._angles_lengths')
        Quiver._angles_lengths.__dict__.__setitem__('stypy_param_names_list', ['U', 'V', 'eps'])
        Quiver._angles_lengths.__dict__.__setitem__('stypy_varargs_param_name', None)
        Quiver._angles_lengths.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Quiver._angles_lengths.__dict__.__setitem__('stypy_call_defaults', defaults)
        Quiver._angles_lengths.__dict__.__setitem__('stypy_call_varargs', varargs)
        Quiver._angles_lengths.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Quiver._angles_lengths.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Quiver._angles_lengths', ['U', 'V', 'eps'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_angles_lengths', localization, ['U', 'V', 'eps'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_angles_lengths(...)' code ##################

        
        # Assigning a Call to a Name (line 619):
        
        # Assigning a Call to a Name (line 619):
        
        # Call to transform(...): (line 619)
        # Processing the call arguments (line 619)
        # Getting the type of 'self' (line 619)
        self_122203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 41), 'self', False)
        # Obtaining the member 'XY' of a type (line 619)
        XY_122204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 41), self_122203, 'XY')
        # Processing the call keyword arguments (line 619)
        kwargs_122205 = {}
        # Getting the type of 'self' (line 619)
        self_122199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 13), 'self', False)
        # Obtaining the member 'ax' of a type (line 619)
        ax_122200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 13), self_122199, 'ax')
        # Obtaining the member 'transData' of a type (line 619)
        transData_122201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 13), ax_122200, 'transData')
        # Obtaining the member 'transform' of a type (line 619)
        transform_122202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 13), transData_122201, 'transform')
        # Calling transform(args, kwargs) (line 619)
        transform_call_result_122206 = invoke(stypy.reporting.localization.Localization(__file__, 619, 13), transform_122202, *[XY_122204], **kwargs_122205)
        
        # Assigning a type to the variable 'xy' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'xy', transform_call_result_122206)
        
        # Assigning a Call to a Name (line 620):
        
        # Assigning a Call to a Name (line 620):
        
        # Call to hstack(...): (line 620)
        # Processing the call arguments (line 620)
        
        # Obtaining an instance of the builtin type 'tuple' (line 620)
        tuple_122209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 620)
        # Adding element type (line 620)
        
        # Obtaining the type of the subscript
        slice_122210 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 620, 24), None, None, None)
        # Getting the type of 'np' (line 620)
        np_122211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 29), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 620)
        newaxis_122212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 29), np_122211, 'newaxis')
        # Getting the type of 'U' (line 620)
        U_122213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 24), 'U', False)
        # Obtaining the member '__getitem__' of a type (line 620)
        getitem___122214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 24), U_122213, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 620)
        subscript_call_result_122215 = invoke(stypy.reporting.localization.Localization(__file__, 620, 24), getitem___122214, (slice_122210, newaxis_122212))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 24), tuple_122209, subscript_call_result_122215)
        # Adding element type (line 620)
        
        # Obtaining the type of the subscript
        slice_122216 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 620, 42), None, None, None)
        # Getting the type of 'np' (line 620)
        np_122217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 47), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 620)
        newaxis_122218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 47), np_122217, 'newaxis')
        # Getting the type of 'V' (line 620)
        V_122219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 42), 'V', False)
        # Obtaining the member '__getitem__' of a type (line 620)
        getitem___122220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 42), V_122219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 620)
        subscript_call_result_122221 = invoke(stypy.reporting.localization.Localization(__file__, 620, 42), getitem___122220, (slice_122216, newaxis_122218))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 620, 24), tuple_122209, subscript_call_result_122221)
        
        # Processing the call keyword arguments (line 620)
        kwargs_122222 = {}
        # Getting the type of 'np' (line 620)
        np_122207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 13), 'np', False)
        # Obtaining the member 'hstack' of a type (line 620)
        hstack_122208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 13), np_122207, 'hstack')
        # Calling hstack(args, kwargs) (line 620)
        hstack_call_result_122223 = invoke(stypy.reporting.localization.Localization(__file__, 620, 13), hstack_122208, *[tuple_122209], **kwargs_122222)
        
        # Assigning a type to the variable 'uv' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'uv', hstack_call_result_122223)
        
        # Assigning a Call to a Name (line 621):
        
        # Assigning a Call to a Name (line 621):
        
        # Call to transform(...): (line 621)
        # Processing the call arguments (line 621)
        # Getting the type of 'self' (line 621)
        self_122228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 42), 'self', False)
        # Obtaining the member 'XY' of a type (line 621)
        XY_122229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 42), self_122228, 'XY')
        # Getting the type of 'eps' (line 621)
        eps_122230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 52), 'eps', False)
        # Getting the type of 'uv' (line 621)
        uv_122231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 58), 'uv', False)
        # Applying the binary operator '*' (line 621)
        result_mul_122232 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 52), '*', eps_122230, uv_122231)
        
        # Applying the binary operator '+' (line 621)
        result_add_122233 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 42), '+', XY_122229, result_mul_122232)
        
        # Processing the call keyword arguments (line 621)
        kwargs_122234 = {}
        # Getting the type of 'self' (line 621)
        self_122224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 14), 'self', False)
        # Obtaining the member 'ax' of a type (line 621)
        ax_122225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 14), self_122224, 'ax')
        # Obtaining the member 'transData' of a type (line 621)
        transData_122226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 14), ax_122225, 'transData')
        # Obtaining the member 'transform' of a type (line 621)
        transform_122227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 14), transData_122226, 'transform')
        # Calling transform(args, kwargs) (line 621)
        transform_call_result_122235 = invoke(stypy.reporting.localization.Localization(__file__, 621, 14), transform_122227, *[result_add_122233], **kwargs_122234)
        
        # Assigning a type to the variable 'xyp' (line 621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'xyp', transform_call_result_122235)
        
        # Assigning a BinOp to a Name (line 622):
        
        # Assigning a BinOp to a Name (line 622):
        # Getting the type of 'xyp' (line 622)
        xyp_122236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 14), 'xyp')
        # Getting the type of 'xy' (line 622)
        xy_122237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 20), 'xy')
        # Applying the binary operator '-' (line 622)
        result_sub_122238 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 14), '-', xyp_122236, xy_122237)
        
        # Assigning a type to the variable 'dxy' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'dxy', result_sub_122238)
        
        # Assigning a Call to a Name (line 623):
        
        # Assigning a Call to a Name (line 623):
        
        # Call to arctan2(...): (line 623)
        # Processing the call arguments (line 623)
        
        # Obtaining the type of the subscript
        slice_122241 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 623, 28), None, None, None)
        int_122242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 35), 'int')
        # Getting the type of 'dxy' (line 623)
        dxy_122243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 28), 'dxy', False)
        # Obtaining the member '__getitem__' of a type (line 623)
        getitem___122244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 28), dxy_122243, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 623)
        subscript_call_result_122245 = invoke(stypy.reporting.localization.Localization(__file__, 623, 28), getitem___122244, (slice_122241, int_122242))
        
        
        # Obtaining the type of the subscript
        slice_122246 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 623, 39), None, None, None)
        int_122247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 46), 'int')
        # Getting the type of 'dxy' (line 623)
        dxy_122248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 39), 'dxy', False)
        # Obtaining the member '__getitem__' of a type (line 623)
        getitem___122249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 39), dxy_122248, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 623)
        subscript_call_result_122250 = invoke(stypy.reporting.localization.Localization(__file__, 623, 39), getitem___122249, (slice_122246, int_122247))
        
        # Processing the call keyword arguments (line 623)
        kwargs_122251 = {}
        # Getting the type of 'np' (line 623)
        np_122239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 17), 'np', False)
        # Obtaining the member 'arctan2' of a type (line 623)
        arctan2_122240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 17), np_122239, 'arctan2')
        # Calling arctan2(args, kwargs) (line 623)
        arctan2_call_result_122252 = invoke(stypy.reporting.localization.Localization(__file__, 623, 17), arctan2_122240, *[subscript_call_result_122245, subscript_call_result_122250], **kwargs_122251)
        
        # Assigning a type to the variable 'angles' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'angles', arctan2_call_result_122252)
        
        # Assigning a BinOp to a Name (line 624):
        
        # Assigning a BinOp to a Name (line 624):
        
        # Call to hypot(...): (line 624)
        # Getting the type of 'dxy' (line 624)
        dxy_122255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 28), 'dxy', False)
        # Obtaining the member 'T' of a type (line 624)
        T_122256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 28), dxy_122255, 'T')
        # Processing the call keyword arguments (line 624)
        kwargs_122257 = {}
        # Getting the type of 'np' (line 624)
        np_122253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 18), 'np', False)
        # Obtaining the member 'hypot' of a type (line 624)
        hypot_122254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 18), np_122253, 'hypot')
        # Calling hypot(args, kwargs) (line 624)
        hypot_call_result_122258 = invoke(stypy.reporting.localization.Localization(__file__, 624, 18), hypot_122254, *[T_122256], **kwargs_122257)
        
        # Getting the type of 'eps' (line 624)
        eps_122259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 37), 'eps')
        # Applying the binary operator 'div' (line 624)
        result_div_122260 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 18), 'div', hypot_call_result_122258, eps_122259)
        
        # Assigning a type to the variable 'lengths' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'lengths', result_div_122260)
        
        # Obtaining an instance of the builtin type 'tuple' (line 625)
        tuple_122261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 625)
        # Adding element type (line 625)
        # Getting the type of 'angles' (line 625)
        angles_122262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 'angles')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 15), tuple_122261, angles_122262)
        # Adding element type (line 625)
        # Getting the type of 'lengths' (line 625)
        lengths_122263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 23), 'lengths')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 15), tuple_122261, lengths_122263)
        
        # Assigning a type to the variable 'stypy_return_type' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'stypy_return_type', tuple_122261)
        
        # ################# End of '_angles_lengths(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_angles_lengths' in the type store
        # Getting the type of 'stypy_return_type' (line 618)
        stypy_return_type_122264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122264)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_angles_lengths'
        return stypy_return_type_122264


    @norecursion
    def _make_verts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_make_verts'
        module_type_store = module_type_store.open_function_context('_make_verts', 627, 4, False)
        # Assigning a type to the variable 'self' (line 628)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Quiver._make_verts.__dict__.__setitem__('stypy_localization', localization)
        Quiver._make_verts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Quiver._make_verts.__dict__.__setitem__('stypy_type_store', module_type_store)
        Quiver._make_verts.__dict__.__setitem__('stypy_function_name', 'Quiver._make_verts')
        Quiver._make_verts.__dict__.__setitem__('stypy_param_names_list', ['U', 'V', 'angles'])
        Quiver._make_verts.__dict__.__setitem__('stypy_varargs_param_name', None)
        Quiver._make_verts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Quiver._make_verts.__dict__.__setitem__('stypy_call_defaults', defaults)
        Quiver._make_verts.__dict__.__setitem__('stypy_call_varargs', varargs)
        Quiver._make_verts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Quiver._make_verts.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Quiver._make_verts', ['U', 'V', 'angles'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_make_verts', localization, ['U', 'V', 'angles'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_make_verts(...)' code ##################

        
        # Assigning a BinOp to a Name (line 628):
        
        # Assigning a BinOp to a Name (line 628):
        # Getting the type of 'U' (line 628)
        U_122265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 14), 'U')
        # Getting the type of 'V' (line 628)
        V_122266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 18), 'V')
        complex_122267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 22), 'complex')
        # Applying the binary operator '*' (line 628)
        result_mul_122268 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 18), '*', V_122266, complex_122267)
        
        # Applying the binary operator '+' (line 628)
        result_add_122269 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 14), '+', U_122265, result_mul_122268)
        
        # Assigning a type to the variable 'uv' (line 628)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'uv', result_add_122269)
        
        # Assigning a IfExp to a Name (line 629):
        
        # Assigning a IfExp to a Name (line 629):
        
        
        # Call to isinstance(...): (line 629)
        # Processing the call arguments (line 629)
        # Getting the type of 'angles' (line 629)
        angles_122271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 42), 'angles', False)
        # Getting the type of 'six' (line 629)
        six_122272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 50), 'six', False)
        # Obtaining the member 'string_types' of a type (line 629)
        string_types_122273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 50), six_122272, 'string_types')
        # Processing the call keyword arguments (line 629)
        kwargs_122274 = {}
        # Getting the type of 'isinstance' (line 629)
        isinstance_122270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 31), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 629)
        isinstance_call_result_122275 = invoke(stypy.reporting.localization.Localization(__file__, 629, 31), isinstance_122270, *[angles_122271, string_types_122273], **kwargs_122274)
        
        # Testing the type of an if expression (line 629)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 629, 21), isinstance_call_result_122275)
        # SSA begins for if expression (line 629)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'angles' (line 629)
        angles_122276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 21), 'angles')
        # SSA branch for the else part of an if expression (line 629)
        module_type_store.open_ssa_branch('if expression else')
        unicode_122277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 73), 'unicode', u'')
        # SSA join for if expression (line 629)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_122278 = union_type.UnionType.add(angles_122276, unicode_122277)
        
        # Assigning a type to the variable 'str_angles' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'str_angles', if_exp_122278)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'str_angles' (line 630)
        str_angles_122279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 11), 'str_angles')
        unicode_122280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 25), 'unicode', u'xy')
        # Applying the binary operator '==' (line 630)
        result_eq_122281 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 11), '==', str_angles_122279, unicode_122280)
        
        
        # Getting the type of 'self' (line 630)
        self_122282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 34), 'self')
        # Obtaining the member 'scale_units' of a type (line 630)
        scale_units_122283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 34), self_122282, 'scale_units')
        unicode_122284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 54), 'unicode', u'xy')
        # Applying the binary operator '==' (line 630)
        result_eq_122285 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 34), '==', scale_units_122283, unicode_122284)
        
        # Applying the binary operator 'and' (line 630)
        result_and_keyword_122286 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 11), 'and', result_eq_122281, result_eq_122285)
        
        # Testing the type of an if condition (line 630)
        if_condition_122287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 630, 8), result_and_keyword_122286)
        # Assigning a type to the variable 'if_condition_122287' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'if_condition_122287', if_condition_122287)
        # SSA begins for if statement (line 630)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 634):
        
        # Assigning a Call to a Name:
        
        # Call to _angles_lengths(...): (line 634)
        # Processing the call arguments (line 634)
        # Getting the type of 'U' (line 634)
        U_122290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 51), 'U', False)
        # Getting the type of 'V' (line 634)
        V_122291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 54), 'V', False)
        # Processing the call keyword arguments (line 634)
        int_122292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 61), 'int')
        keyword_122293 = int_122292
        kwargs_122294 = {'eps': keyword_122293}
        # Getting the type of 'self' (line 634)
        self_122288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 30), 'self', False)
        # Obtaining the member '_angles_lengths' of a type (line 634)
        _angles_lengths_122289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 30), self_122288, '_angles_lengths')
        # Calling _angles_lengths(args, kwargs) (line 634)
        _angles_lengths_call_result_122295 = invoke(stypy.reporting.localization.Localization(__file__, 634, 30), _angles_lengths_122289, *[U_122290, V_122291], **kwargs_122294)
        
        # Assigning a type to the variable 'call_assignment_120683' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'call_assignment_120683', _angles_lengths_call_result_122295)
        
        # Assigning a Call to a Name (line 634):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_122298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 12), 'int')
        # Processing the call keyword arguments
        kwargs_122299 = {}
        # Getting the type of 'call_assignment_120683' (line 634)
        call_assignment_120683_122296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'call_assignment_120683', False)
        # Obtaining the member '__getitem__' of a type (line 634)
        getitem___122297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 12), call_assignment_120683_122296, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_122300 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___122297, *[int_122298], **kwargs_122299)
        
        # Assigning a type to the variable 'call_assignment_120684' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'call_assignment_120684', getitem___call_result_122300)
        
        # Assigning a Name to a Name (line 634):
        # Getting the type of 'call_assignment_120684' (line 634)
        call_assignment_120684_122301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'call_assignment_120684')
        # Assigning a type to the variable 'angles' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'angles', call_assignment_120684_122301)
        
        # Assigning a Call to a Name (line 634):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_122304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 12), 'int')
        # Processing the call keyword arguments
        kwargs_122305 = {}
        # Getting the type of 'call_assignment_120683' (line 634)
        call_assignment_120683_122302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'call_assignment_120683', False)
        # Obtaining the member '__getitem__' of a type (line 634)
        getitem___122303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 12), call_assignment_120683_122302, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_122306 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___122303, *[int_122304], **kwargs_122305)
        
        # Assigning a type to the variable 'call_assignment_120685' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'call_assignment_120685', getitem___call_result_122306)
        
        # Assigning a Name to a Name (line 634):
        # Getting the type of 'call_assignment_120685' (line 634)
        call_assignment_120685_122307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'call_assignment_120685')
        # Assigning a type to the variable 'lengths' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 20), 'lengths', call_assignment_120685_122307)
        # SSA branch for the else part of an if statement (line 630)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'str_angles' (line 635)
        str_angles_122308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 13), 'str_angles')
        unicode_122309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 27), 'unicode', u'xy')
        # Applying the binary operator '==' (line 635)
        result_eq_122310 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 13), '==', str_angles_122308, unicode_122309)
        
        
        # Getting the type of 'self' (line 635)
        self_122311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 35), 'self')
        # Obtaining the member 'scale_units' of a type (line 635)
        scale_units_122312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 35), self_122311, 'scale_units')
        unicode_122313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 55), 'unicode', u'xy')
        # Applying the binary operator '==' (line 635)
        result_eq_122314 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 35), '==', scale_units_122312, unicode_122313)
        
        # Applying the binary operator 'or' (line 635)
        result_or_keyword_122315 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 13), 'or', result_eq_122310, result_eq_122314)
        
        # Testing the type of an if condition (line 635)
        if_condition_122316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 635, 13), result_or_keyword_122315)
        # Assigning a type to the variable 'if_condition_122316' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 13), 'if_condition_122316', if_condition_122316)
        # SSA begins for if statement (line 635)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 639):
        
        # Assigning a BinOp to a Name (line 639):
        
        # Call to max(...): (line 639)
        # Processing the call keyword arguments (line 639)
        kwargs_122326 = {}
        
        # Call to abs(...): (line 639)
        # Processing the call arguments (line 639)
        # Getting the type of 'self' (line 639)
        self_122319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 25), 'self', False)
        # Obtaining the member 'ax' of a type (line 639)
        ax_122320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 25), self_122319, 'ax')
        # Obtaining the member 'dataLim' of a type (line 639)
        dataLim_122321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 25), ax_122320, 'dataLim')
        # Obtaining the member 'extents' of a type (line 639)
        extents_122322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 25), dataLim_122321, 'extents')
        # Processing the call keyword arguments (line 639)
        kwargs_122323 = {}
        # Getting the type of 'np' (line 639)
        np_122317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 18), 'np', False)
        # Obtaining the member 'abs' of a type (line 639)
        abs_122318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 18), np_122317, 'abs')
        # Calling abs(args, kwargs) (line 639)
        abs_call_result_122324 = invoke(stypy.reporting.localization.Localization(__file__, 639, 18), abs_122318, *[extents_122322], **kwargs_122323)
        
        # Obtaining the member 'max' of a type (line 639)
        max_122325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 18), abs_call_result_122324, 'max')
        # Calling max(args, kwargs) (line 639)
        max_call_result_122327 = invoke(stypy.reporting.localization.Localization(__file__, 639, 18), max_122325, *[], **kwargs_122326)
        
        float_122328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 58), 'float')
        # Applying the binary operator '*' (line 639)
        result_mul_122329 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 18), '*', max_call_result_122327, float_122328)
        
        # Assigning a type to the variable 'eps' (line 639)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'eps', result_mul_122329)
        
        # Assigning a Call to a Tuple (line 640):
        
        # Assigning a Call to a Name:
        
        # Call to _angles_lengths(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 'U' (line 640)
        U_122332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 51), 'U', False)
        # Getting the type of 'V' (line 640)
        V_122333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 54), 'V', False)
        # Processing the call keyword arguments (line 640)
        # Getting the type of 'eps' (line 640)
        eps_122334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 61), 'eps', False)
        keyword_122335 = eps_122334
        kwargs_122336 = {'eps': keyword_122335}
        # Getting the type of 'self' (line 640)
        self_122330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 30), 'self', False)
        # Obtaining the member '_angles_lengths' of a type (line 640)
        _angles_lengths_122331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 30), self_122330, '_angles_lengths')
        # Calling _angles_lengths(args, kwargs) (line 640)
        _angles_lengths_call_result_122337 = invoke(stypy.reporting.localization.Localization(__file__, 640, 30), _angles_lengths_122331, *[U_122332, V_122333], **kwargs_122336)
        
        # Assigning a type to the variable 'call_assignment_120686' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'call_assignment_120686', _angles_lengths_call_result_122337)
        
        # Assigning a Call to a Name (line 640):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_122340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 12), 'int')
        # Processing the call keyword arguments
        kwargs_122341 = {}
        # Getting the type of 'call_assignment_120686' (line 640)
        call_assignment_120686_122338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'call_assignment_120686', False)
        # Obtaining the member '__getitem__' of a type (line 640)
        getitem___122339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 12), call_assignment_120686_122338, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_122342 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___122339, *[int_122340], **kwargs_122341)
        
        # Assigning a type to the variable 'call_assignment_120687' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'call_assignment_120687', getitem___call_result_122342)
        
        # Assigning a Name to a Name (line 640):
        # Getting the type of 'call_assignment_120687' (line 640)
        call_assignment_120687_122343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'call_assignment_120687')
        # Assigning a type to the variable 'angles' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'angles', call_assignment_120687_122343)
        
        # Assigning a Call to a Name (line 640):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_122346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 12), 'int')
        # Processing the call keyword arguments
        kwargs_122347 = {}
        # Getting the type of 'call_assignment_120686' (line 640)
        call_assignment_120686_122344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'call_assignment_120686', False)
        # Obtaining the member '__getitem__' of a type (line 640)
        getitem___122345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 12), call_assignment_120686_122344, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_122348 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___122345, *[int_122346], **kwargs_122347)
        
        # Assigning a type to the variable 'call_assignment_120688' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'call_assignment_120688', getitem___call_result_122348)
        
        # Assigning a Name to a Name (line 640):
        # Getting the type of 'call_assignment_120688' (line 640)
        call_assignment_120688_122349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'call_assignment_120688')
        # Assigning a type to the variable 'lengths' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 20), 'lengths', call_assignment_120688_122349)
        # SSA join for if statement (line 635)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 630)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'str_angles' (line 641)
        str_angles_122350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 11), 'str_angles')
        
        # Getting the type of 'self' (line 641)
        self_122351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 26), 'self')
        # Obtaining the member 'scale_units' of a type (line 641)
        scale_units_122352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 26), self_122351, 'scale_units')
        unicode_122353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 46), 'unicode', u'xy')
        # Applying the binary operator '==' (line 641)
        result_eq_122354 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 26), '==', scale_units_122352, unicode_122353)
        
        # Applying the binary operator 'and' (line 641)
        result_and_keyword_122355 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 11), 'and', str_angles_122350, result_eq_122354)
        
        # Testing the type of an if condition (line 641)
        if_condition_122356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 641, 8), result_and_keyword_122355)
        # Assigning a type to the variable 'if_condition_122356' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'if_condition_122356', if_condition_122356)
        # SSA begins for if statement (line 641)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 642):
        
        # Assigning a Name to a Name (line 642):
        # Getting the type of 'lengths' (line 642)
        lengths_122357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 16), 'lengths')
        # Assigning a type to the variable 'a' (line 642)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 12), 'a', lengths_122357)
        # SSA branch for the else part of an if statement (line 641)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 644):
        
        # Assigning a Call to a Name (line 644):
        
        # Call to abs(...): (line 644)
        # Processing the call arguments (line 644)
        # Getting the type of 'uv' (line 644)
        uv_122360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 23), 'uv', False)
        # Processing the call keyword arguments (line 644)
        kwargs_122361 = {}
        # Getting the type of 'np' (line 644)
        np_122358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 16), 'np', False)
        # Obtaining the member 'abs' of a type (line 644)
        abs_122359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 16), np_122358, 'abs')
        # Calling abs(args, kwargs) (line 644)
        abs_call_result_122362 = invoke(stypy.reporting.localization.Localization(__file__, 644, 16), abs_122359, *[uv_122360], **kwargs_122361)
        
        # Assigning a type to the variable 'a' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'a', abs_call_result_122362)
        # SSA join for if statement (line 641)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 645)
        # Getting the type of 'self' (line 645)
        self_122363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 11), 'self')
        # Obtaining the member 'scale' of a type (line 645)
        scale_122364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 11), self_122363, 'scale')
        # Getting the type of 'None' (line 645)
        None_122365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 25), 'None')
        
        (may_be_122366, more_types_in_union_122367) = may_be_none(scale_122364, None_122365)

        if may_be_122366:

            if more_types_in_union_122367:
                # Runtime conditional SSA (line 645)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 646):
            
            # Assigning a Call to a Name (line 646):
            
            # Call to max(...): (line 646)
            # Processing the call arguments (line 646)
            int_122369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 21), 'int')
            
            # Call to sqrt(...): (line 646)
            # Processing the call arguments (line 646)
            # Getting the type of 'self' (line 646)
            self_122372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 35), 'self', False)
            # Obtaining the member 'N' of a type (line 646)
            N_122373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 35), self_122372, 'N')
            # Processing the call keyword arguments (line 646)
            kwargs_122374 = {}
            # Getting the type of 'math' (line 646)
            math_122370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 25), 'math', False)
            # Obtaining the member 'sqrt' of a type (line 646)
            sqrt_122371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 25), math_122370, 'sqrt')
            # Calling sqrt(args, kwargs) (line 646)
            sqrt_call_result_122375 = invoke(stypy.reporting.localization.Localization(__file__, 646, 25), sqrt_122371, *[N_122373], **kwargs_122374)
            
            # Processing the call keyword arguments (line 646)
            kwargs_122376 = {}
            # Getting the type of 'max' (line 646)
            max_122368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 17), 'max', False)
            # Calling max(args, kwargs) (line 646)
            max_call_result_122377 = invoke(stypy.reporting.localization.Localization(__file__, 646, 17), max_122368, *[int_122369, sqrt_call_result_122375], **kwargs_122376)
            
            # Assigning a type to the variable 'sn' (line 646)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 12), 'sn', max_call_result_122377)
            
            
            # Getting the type of 'self' (line 647)
            self_122378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 15), 'self')
            # Obtaining the member 'Umask' of a type (line 647)
            Umask_122379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 15), self_122378, 'Umask')
            # Getting the type of 'ma' (line 647)
            ma_122380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 33), 'ma')
            # Obtaining the member 'nomask' of a type (line 647)
            nomask_122381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 33), ma_122380, 'nomask')
            # Applying the binary operator 'isnot' (line 647)
            result_is_not_122382 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 15), 'isnot', Umask_122379, nomask_122381)
            
            # Testing the type of an if condition (line 647)
            if_condition_122383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 647, 12), result_is_not_122382)
            # Assigning a type to the variable 'if_condition_122383' (line 647)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 12), 'if_condition_122383', if_condition_122383)
            # SSA begins for if statement (line 647)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 648):
            
            # Assigning a Call to a Name (line 648):
            
            # Call to mean(...): (line 648)
            # Processing the call keyword arguments (line 648)
            kwargs_122391 = {}
            
            # Obtaining the type of the subscript
            
            # Getting the type of 'self' (line 648)
            self_122384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 27), 'self', False)
            # Obtaining the member 'Umask' of a type (line 648)
            Umask_122385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 27), self_122384, 'Umask')
            # Applying the '~' unary operator (line 648)
            result_inv_122386 = python_operator(stypy.reporting.localization.Localization(__file__, 648, 26), '~', Umask_122385)
            
            # Getting the type of 'a' (line 648)
            a_122387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 24), 'a', False)
            # Obtaining the member '__getitem__' of a type (line 648)
            getitem___122388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 24), a_122387, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 648)
            subscript_call_result_122389 = invoke(stypy.reporting.localization.Localization(__file__, 648, 24), getitem___122388, result_inv_122386)
            
            # Obtaining the member 'mean' of a type (line 648)
            mean_122390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 24), subscript_call_result_122389, 'mean')
            # Calling mean(args, kwargs) (line 648)
            mean_call_result_122392 = invoke(stypy.reporting.localization.Localization(__file__, 648, 24), mean_122390, *[], **kwargs_122391)
            
            # Assigning a type to the variable 'amean' (line 648)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 16), 'amean', mean_call_result_122392)
            # SSA branch for the else part of an if statement (line 647)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 650):
            
            # Assigning a Call to a Name (line 650):
            
            # Call to mean(...): (line 650)
            # Processing the call keyword arguments (line 650)
            kwargs_122395 = {}
            # Getting the type of 'a' (line 650)
            a_122393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 24), 'a', False)
            # Obtaining the member 'mean' of a type (line 650)
            mean_122394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 24), a_122393, 'mean')
            # Calling mean(args, kwargs) (line 650)
            mean_call_result_122396 = invoke(stypy.reporting.localization.Localization(__file__, 650, 24), mean_122394, *[], **kwargs_122395)
            
            # Assigning a type to the variable 'amean' (line 650)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 16), 'amean', mean_call_result_122396)
            # SSA join for if statement (line 647)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 653):
            
            # Assigning a BinOp to a Name (line 653):
            float_122397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 20), 'float')
            # Getting the type of 'amean' (line 653)
            amean_122398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 26), 'amean')
            # Applying the binary operator '*' (line 653)
            result_mul_122399 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 20), '*', float_122397, amean_122398)
            
            # Getting the type of 'sn' (line 653)
            sn_122400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 34), 'sn')
            # Applying the binary operator '*' (line 653)
            result_mul_122401 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 32), '*', result_mul_122399, sn_122400)
            
            # Getting the type of 'self' (line 653)
            self_122402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 39), 'self')
            # Obtaining the member 'span' of a type (line 653)
            span_122403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 39), self_122402, 'span')
            # Applying the binary operator 'div' (line 653)
            result_div_122404 = python_operator(stypy.reporting.localization.Localization(__file__, 653, 37), 'div', result_mul_122401, span_122403)
            
            # Assigning a type to the variable 'scale' (line 653)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 12), 'scale', result_div_122404)

            if more_types_in_union_122367:
                # SSA join for if statement (line 645)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 654)
        # Getting the type of 'self' (line 654)
        self_122405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 11), 'self')
        # Obtaining the member 'scale_units' of a type (line 654)
        scale_units_122406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 11), self_122405, 'scale_units')
        # Getting the type of 'None' (line 654)
        None_122407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 31), 'None')
        
        (may_be_122408, more_types_in_union_122409) = may_be_none(scale_units_122406, None_122407)

        if may_be_122408:

            if more_types_in_union_122409:
                # Runtime conditional SSA (line 654)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 655)
            # Getting the type of 'self' (line 655)
            self_122410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 15), 'self')
            # Obtaining the member 'scale' of a type (line 655)
            scale_122411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 15), self_122410, 'scale')
            # Getting the type of 'None' (line 655)
            None_122412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 29), 'None')
            
            (may_be_122413, more_types_in_union_122414) = may_be_none(scale_122411, None_122412)

            if may_be_122413:

                if more_types_in_union_122414:
                    # Runtime conditional SSA (line 655)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Name to a Attribute (line 656):
                
                # Assigning a Name to a Attribute (line 656):
                # Getting the type of 'scale' (line 656)
                scale_122415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 29), 'scale')
                # Getting the type of 'self' (line 656)
                self_122416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 16), 'self')
                # Setting the type of the member 'scale' of a type (line 656)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 16), self_122416, 'scale', scale_122415)

                if more_types_in_union_122414:
                    # SSA join for if statement (line 655)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Num to a Name (line 657):
            
            # Assigning a Num to a Name (line 657):
            float_122417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 30), 'float')
            # Assigning a type to the variable 'widthu_per_lenu' (line 657)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'widthu_per_lenu', float_122417)

            if more_types_in_union_122409:
                # Runtime conditional SSA for else branch (line 654)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_122408) or more_types_in_union_122409):
            
            
            # Getting the type of 'self' (line 659)
            self_122418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 15), 'self')
            # Obtaining the member 'scale_units' of a type (line 659)
            scale_units_122419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 15), self_122418, 'scale_units')
            unicode_122420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 35), 'unicode', u'xy')
            # Applying the binary operator '==' (line 659)
            result_eq_122421 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 15), '==', scale_units_122419, unicode_122420)
            
            # Testing the type of an if condition (line 659)
            if_condition_122422 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 659, 12), result_eq_122421)
            # Assigning a type to the variable 'if_condition_122422' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'if_condition_122422', if_condition_122422)
            # SSA begins for if statement (line 659)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 660):
            
            # Assigning a Num to a Name (line 660):
            int_122423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 21), 'int')
            # Assigning a type to the variable 'dx' (line 660)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 16), 'dx', int_122423)
            # SSA branch for the else part of an if statement (line 659)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 662):
            
            # Assigning a Call to a Name (line 662):
            
            # Call to _dots_per_unit(...): (line 662)
            # Processing the call arguments (line 662)
            # Getting the type of 'self' (line 662)
            self_122426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 41), 'self', False)
            # Obtaining the member 'scale_units' of a type (line 662)
            scale_units_122427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 41), self_122426, 'scale_units')
            # Processing the call keyword arguments (line 662)
            kwargs_122428 = {}
            # Getting the type of 'self' (line 662)
            self_122424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 21), 'self', False)
            # Obtaining the member '_dots_per_unit' of a type (line 662)
            _dots_per_unit_122425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 21), self_122424, '_dots_per_unit')
            # Calling _dots_per_unit(args, kwargs) (line 662)
            _dots_per_unit_call_result_122429 = invoke(stypy.reporting.localization.Localization(__file__, 662, 21), _dots_per_unit_122425, *[scale_units_122427], **kwargs_122428)
            
            # Assigning a type to the variable 'dx' (line 662)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 16), 'dx', _dots_per_unit_call_result_122429)
            # SSA join for if statement (line 659)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 663):
            
            # Assigning a BinOp to a Name (line 663):
            # Getting the type of 'dx' (line 663)
            dx_122430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 30), 'dx')
            # Getting the type of 'self' (line 663)
            self_122431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 35), 'self')
            # Obtaining the member '_trans_scale' of a type (line 663)
            _trans_scale_122432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 35), self_122431, '_trans_scale')
            # Applying the binary operator 'div' (line 663)
            result_div_122433 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 30), 'div', dx_122430, _trans_scale_122432)
            
            # Assigning a type to the variable 'widthu_per_lenu' (line 663)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), 'widthu_per_lenu', result_div_122433)
            
            # Type idiom detected: calculating its left and rigth part (line 664)
            # Getting the type of 'self' (line 664)
            self_122434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 15), 'self')
            # Obtaining the member 'scale' of a type (line 664)
            scale_122435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 15), self_122434, 'scale')
            # Getting the type of 'None' (line 664)
            None_122436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 29), 'None')
            
            (may_be_122437, more_types_in_union_122438) = may_be_none(scale_122435, None_122436)

            if may_be_122437:

                if more_types_in_union_122438:
                    # Runtime conditional SSA (line 664)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a BinOp to a Attribute (line 665):
                
                # Assigning a BinOp to a Attribute (line 665):
                # Getting the type of 'scale' (line 665)
                scale_122439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 29), 'scale')
                # Getting the type of 'widthu_per_lenu' (line 665)
                widthu_per_lenu_122440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 37), 'widthu_per_lenu')
                # Applying the binary operator '*' (line 665)
                result_mul_122441 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 29), '*', scale_122439, widthu_per_lenu_122440)
                
                # Getting the type of 'self' (line 665)
                self_122442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 16), 'self')
                # Setting the type of the member 'scale' of a type (line 665)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 16), self_122442, 'scale', result_mul_122441)

                if more_types_in_union_122438:
                    # SSA join for if statement (line 664)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_122408 and more_types_in_union_122409):
                # SSA join for if statement (line 654)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 666):
        
        # Assigning a BinOp to a Name (line 666):
        # Getting the type of 'a' (line 666)
        a_122443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 17), 'a')
        # Getting the type of 'widthu_per_lenu' (line 666)
        widthu_per_lenu_122444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 22), 'widthu_per_lenu')
        # Getting the type of 'self' (line 666)
        self_122445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 41), 'self')
        # Obtaining the member 'scale' of a type (line 666)
        scale_122446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 41), self_122445, 'scale')
        # Getting the type of 'self' (line 666)
        self_122447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 54), 'self')
        # Obtaining the member 'width' of a type (line 666)
        width_122448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 54), self_122447, 'width')
        # Applying the binary operator '*' (line 666)
        result_mul_122449 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 41), '*', scale_122446, width_122448)
        
        # Applying the binary operator 'div' (line 666)
        result_div_122450 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 22), 'div', widthu_per_lenu_122444, result_mul_122449)
        
        # Applying the binary operator '*' (line 666)
        result_mul_122451 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 17), '*', a_122443, result_div_122450)
        
        # Assigning a type to the variable 'length' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'length', result_mul_122451)
        
        # Assigning a Call to a Tuple (line 667):
        
        # Assigning a Call to a Name:
        
        # Call to _h_arrows(...): (line 667)
        # Processing the call arguments (line 667)
        # Getting the type of 'length' (line 667)
        length_122454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 30), 'length', False)
        # Processing the call keyword arguments (line 667)
        kwargs_122455 = {}
        # Getting the type of 'self' (line 667)
        self_122452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 15), 'self', False)
        # Obtaining the member '_h_arrows' of a type (line 667)
        _h_arrows_122453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 15), self_122452, '_h_arrows')
        # Calling _h_arrows(args, kwargs) (line 667)
        _h_arrows_call_result_122456 = invoke(stypy.reporting.localization.Localization(__file__, 667, 15), _h_arrows_122453, *[length_122454], **kwargs_122455)
        
        # Assigning a type to the variable 'call_assignment_120689' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'call_assignment_120689', _h_arrows_call_result_122456)
        
        # Assigning a Call to a Name (line 667):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_122459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 8), 'int')
        # Processing the call keyword arguments
        kwargs_122460 = {}
        # Getting the type of 'call_assignment_120689' (line 667)
        call_assignment_120689_122457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'call_assignment_120689', False)
        # Obtaining the member '__getitem__' of a type (line 667)
        getitem___122458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 8), call_assignment_120689_122457, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_122461 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___122458, *[int_122459], **kwargs_122460)
        
        # Assigning a type to the variable 'call_assignment_120690' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'call_assignment_120690', getitem___call_result_122461)
        
        # Assigning a Name to a Name (line 667):
        # Getting the type of 'call_assignment_120690' (line 667)
        call_assignment_120690_122462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'call_assignment_120690')
        # Assigning a type to the variable 'X' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'X', call_assignment_120690_122462)
        
        # Assigning a Call to a Name (line 667):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_122465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 8), 'int')
        # Processing the call keyword arguments
        kwargs_122466 = {}
        # Getting the type of 'call_assignment_120689' (line 667)
        call_assignment_120689_122463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'call_assignment_120689', False)
        # Obtaining the member '__getitem__' of a type (line 667)
        getitem___122464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 8), call_assignment_120689_122463, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_122467 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___122464, *[int_122465], **kwargs_122466)
        
        # Assigning a type to the variable 'call_assignment_120691' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'call_assignment_120691', getitem___call_result_122467)
        
        # Assigning a Name to a Name (line 667):
        # Getting the type of 'call_assignment_120691' (line 667)
        call_assignment_120691_122468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'call_assignment_120691')
        # Assigning a type to the variable 'Y' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 11), 'Y', call_assignment_120691_122468)
        
        
        # Getting the type of 'str_angles' (line 668)
        str_angles_122469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 11), 'str_angles')
        unicode_122470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 25), 'unicode', u'xy')
        # Applying the binary operator '==' (line 668)
        result_eq_122471 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 11), '==', str_angles_122469, unicode_122470)
        
        # Testing the type of an if condition (line 668)
        if_condition_122472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 668, 8), result_eq_122471)
        # Assigning a type to the variable 'if_condition_122472' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'if_condition_122472', if_condition_122472)
        # SSA begins for if statement (line 668)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 669):
        
        # Assigning a Name to a Name (line 669):
        # Getting the type of 'angles' (line 669)
        angles_122473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 20), 'angles')
        # Assigning a type to the variable 'theta' (line 669)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'theta', angles_122473)
        # SSA branch for the else part of an if statement (line 668)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'str_angles' (line 670)
        str_angles_122474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 13), 'str_angles')
        unicode_122475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 27), 'unicode', u'uv')
        # Applying the binary operator '==' (line 670)
        result_eq_122476 = python_operator(stypy.reporting.localization.Localization(__file__, 670, 13), '==', str_angles_122474, unicode_122475)
        
        # Testing the type of an if condition (line 670)
        if_condition_122477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 670, 13), result_eq_122476)
        # Assigning a type to the variable 'if_condition_122477' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 13), 'if_condition_122477', if_condition_122477)
        # SSA begins for if statement (line 670)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 671):
        
        # Assigning a Call to a Name (line 671):
        
        # Call to angle(...): (line 671)
        # Processing the call arguments (line 671)
        # Getting the type of 'uv' (line 671)
        uv_122480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 29), 'uv', False)
        # Processing the call keyword arguments (line 671)
        kwargs_122481 = {}
        # Getting the type of 'np' (line 671)
        np_122478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 20), 'np', False)
        # Obtaining the member 'angle' of a type (line 671)
        angle_122479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 20), np_122478, 'angle')
        # Calling angle(args, kwargs) (line 671)
        angle_call_result_122482 = invoke(stypy.reporting.localization.Localization(__file__, 671, 20), angle_122479, *[uv_122480], **kwargs_122481)
        
        # Assigning a type to the variable 'theta' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 12), 'theta', angle_call_result_122482)
        # SSA branch for the else part of an if statement (line 670)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 673):
        
        # Assigning a Call to a Name (line 673):
        
        # Call to filled(...): (line 673)
        # Processing the call arguments (line 673)
        int_122493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 65), 'int')
        # Processing the call keyword arguments (line 673)
        kwargs_122494 = {}
        
        # Call to masked_invalid(...): (line 673)
        # Processing the call arguments (line 673)
        
        # Call to deg2rad(...): (line 673)
        # Processing the call arguments (line 673)
        # Getting the type of 'angles' (line 673)
        angles_122487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 49), 'angles', False)
        # Processing the call keyword arguments (line 673)
        kwargs_122488 = {}
        # Getting the type of 'np' (line 673)
        np_122485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 38), 'np', False)
        # Obtaining the member 'deg2rad' of a type (line 673)
        deg2rad_122486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 38), np_122485, 'deg2rad')
        # Calling deg2rad(args, kwargs) (line 673)
        deg2rad_call_result_122489 = invoke(stypy.reporting.localization.Localization(__file__, 673, 38), deg2rad_122486, *[angles_122487], **kwargs_122488)
        
        # Processing the call keyword arguments (line 673)
        kwargs_122490 = {}
        # Getting the type of 'ma' (line 673)
        ma_122483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 20), 'ma', False)
        # Obtaining the member 'masked_invalid' of a type (line 673)
        masked_invalid_122484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 20), ma_122483, 'masked_invalid')
        # Calling masked_invalid(args, kwargs) (line 673)
        masked_invalid_call_result_122491 = invoke(stypy.reporting.localization.Localization(__file__, 673, 20), masked_invalid_122484, *[deg2rad_call_result_122489], **kwargs_122490)
        
        # Obtaining the member 'filled' of a type (line 673)
        filled_122492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 20), masked_invalid_call_result_122491, 'filled')
        # Calling filled(args, kwargs) (line 673)
        filled_call_result_122495 = invoke(stypy.reporting.localization.Localization(__file__, 673, 20), filled_122492, *[int_122493], **kwargs_122494)
        
        # Assigning a type to the variable 'theta' (line 673)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 12), 'theta', filled_call_result_122495)
        # SSA join for if statement (line 670)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 668)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 674):
        
        # Assigning a Call to a Name (line 674):
        
        # Call to reshape(...): (line 674)
        # Processing the call arguments (line 674)
        
        # Obtaining an instance of the builtin type 'tuple' (line 674)
        tuple_122498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 674)
        # Adding element type (line 674)
        int_122499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 674, 31), tuple_122498, int_122499)
        # Adding element type (line 674)
        int_122500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 674, 31), tuple_122498, int_122500)
        
        # Processing the call keyword arguments (line 674)
        kwargs_122501 = {}
        # Getting the type of 'theta' (line 674)
        theta_122496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 16), 'theta', False)
        # Obtaining the member 'reshape' of a type (line 674)
        reshape_122497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 16), theta_122496, 'reshape')
        # Calling reshape(args, kwargs) (line 674)
        reshape_call_result_122502 = invoke(stypy.reporting.localization.Localization(__file__, 674, 16), reshape_122497, *[tuple_122498], **kwargs_122501)
        
        # Assigning a type to the variable 'theta' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'theta', reshape_call_result_122502)
        
        # Assigning a BinOp to a Name (line 675):
        
        # Assigning a BinOp to a Name (line 675):
        # Getting the type of 'X' (line 675)
        X_122503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 14), 'X')
        # Getting the type of 'Y' (line 675)
        Y_122504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 18), 'Y')
        complex_122505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 22), 'complex')
        # Applying the binary operator '*' (line 675)
        result_mul_122506 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 18), '*', Y_122504, complex_122505)
        
        # Applying the binary operator '+' (line 675)
        result_add_122507 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 14), '+', X_122503, result_mul_122506)
        
        
        # Call to exp(...): (line 675)
        # Processing the call arguments (line 675)
        complex_122510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 35), 'complex')
        # Getting the type of 'theta' (line 675)
        theta_122511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 40), 'theta', False)
        # Applying the binary operator '*' (line 675)
        result_mul_122512 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 35), '*', complex_122510, theta_122511)
        
        # Processing the call keyword arguments (line 675)
        kwargs_122513 = {}
        # Getting the type of 'np' (line 675)
        np_122508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 28), 'np', False)
        # Obtaining the member 'exp' of a type (line 675)
        exp_122509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 28), np_122508, 'exp')
        # Calling exp(args, kwargs) (line 675)
        exp_call_result_122514 = invoke(stypy.reporting.localization.Localization(__file__, 675, 28), exp_122509, *[result_mul_122512], **kwargs_122513)
        
        # Applying the binary operator '*' (line 675)
        result_mul_122515 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 13), '*', result_add_122507, exp_call_result_122514)
        
        # Getting the type of 'self' (line 675)
        self_122516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 49), 'self')
        # Obtaining the member 'width' of a type (line 675)
        width_122517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 49), self_122516, 'width')
        # Applying the binary operator '*' (line 675)
        result_mul_122518 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 47), '*', result_mul_122515, width_122517)
        
        # Assigning a type to the variable 'xy' (line 675)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'xy', result_mul_122518)
        
        # Assigning a Subscript to a Name (line 676):
        
        # Assigning a Subscript to a Name (line 676):
        
        # Obtaining the type of the subscript
        slice_122519 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 676, 13), None, None, None)
        slice_122520 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 676, 13), None, None, None)
        # Getting the type of 'np' (line 676)
        np_122521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 22), 'np')
        # Obtaining the member 'newaxis' of a type (line 676)
        newaxis_122522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 22), np_122521, 'newaxis')
        # Getting the type of 'xy' (line 676)
        xy_122523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 13), 'xy')
        # Obtaining the member '__getitem__' of a type (line 676)
        getitem___122524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 13), xy_122523, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 676)
        subscript_call_result_122525 = invoke(stypy.reporting.localization.Localization(__file__, 676, 13), getitem___122524, (slice_122519, slice_122520, newaxis_122522))
        
        # Assigning a type to the variable 'xy' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'xy', subscript_call_result_122525)
        
        # Assigning a Call to a Name (line 677):
        
        # Assigning a Call to a Name (line 677):
        
        # Call to concatenate(...): (line 677)
        # Processing the call arguments (line 677)
        
        # Obtaining an instance of the builtin type 'tuple' (line 677)
        tuple_122528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 677)
        # Adding element type (line 677)
        # Getting the type of 'xy' (line 677)
        xy_122529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 29), 'xy', False)
        # Obtaining the member 'real' of a type (line 677)
        real_122530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 29), xy_122529, 'real')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 29), tuple_122528, real_122530)
        # Adding element type (line 677)
        # Getting the type of 'xy' (line 677)
        xy_122531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 38), 'xy', False)
        # Obtaining the member 'imag' of a type (line 677)
        imag_122532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 38), xy_122531, 'imag')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 29), tuple_122528, imag_122532)
        
        # Processing the call keyword arguments (line 677)
        int_122533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 53), 'int')
        keyword_122534 = int_122533
        kwargs_122535 = {'axis': keyword_122534}
        # Getting the type of 'np' (line 677)
        np_122526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 13), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 677)
        concatenate_122527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 13), np_122526, 'concatenate')
        # Calling concatenate(args, kwargs) (line 677)
        concatenate_call_result_122536 = invoke(stypy.reporting.localization.Localization(__file__, 677, 13), concatenate_122527, *[tuple_122528], **kwargs_122535)
        
        # Assigning a type to the variable 'XY' (line 677)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 8), 'XY', concatenate_call_result_122536)
        
        
        # Getting the type of 'self' (line 678)
        self_122537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 11), 'self')
        # Obtaining the member 'Umask' of a type (line 678)
        Umask_122538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 11), self_122537, 'Umask')
        # Getting the type of 'ma' (line 678)
        ma_122539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 29), 'ma')
        # Obtaining the member 'nomask' of a type (line 678)
        nomask_122540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 29), ma_122539, 'nomask')
        # Applying the binary operator 'isnot' (line 678)
        result_is_not_122541 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 11), 'isnot', Umask_122538, nomask_122540)
        
        # Testing the type of an if condition (line 678)
        if_condition_122542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 678, 8), result_is_not_122541)
        # Assigning a type to the variable 'if_condition_122542' (line 678)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 8), 'if_condition_122542', if_condition_122542)
        # SSA begins for if statement (line 678)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 679):
        
        # Assigning a Call to a Name (line 679):
        
        # Call to array(...): (line 679)
        # Processing the call arguments (line 679)
        # Getting the type of 'XY' (line 679)
        XY_122545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 26), 'XY', False)
        # Processing the call keyword arguments (line 679)
        kwargs_122546 = {}
        # Getting the type of 'ma' (line 679)
        ma_122543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 17), 'ma', False)
        # Obtaining the member 'array' of a type (line 679)
        array_122544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 17), ma_122543, 'array')
        # Calling array(args, kwargs) (line 679)
        array_call_result_122547 = invoke(stypy.reporting.localization.Localization(__file__, 679, 17), array_122544, *[XY_122545], **kwargs_122546)
        
        # Assigning a type to the variable 'XY' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 12), 'XY', array_call_result_122547)
        
        # Assigning a Attribute to a Subscript (line 680):
        
        # Assigning a Attribute to a Subscript (line 680):
        # Getting the type of 'ma' (line 680)
        ma_122548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 29), 'ma')
        # Obtaining the member 'masked' of a type (line 680)
        masked_122549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 29), ma_122548, 'masked')
        # Getting the type of 'XY' (line 680)
        XY_122550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 12), 'XY')
        # Getting the type of 'self' (line 680)
        self_122551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 15), 'self')
        # Obtaining the member 'Umask' of a type (line 680)
        Umask_122552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 15), self_122551, 'Umask')
        # Storing an element on a container (line 680)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 680, 12), XY_122550, (Umask_122552, masked_122549))
        # SSA join for if statement (line 678)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'XY' (line 684)
        XY_122553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 15), 'XY')
        # Assigning a type to the variable 'stypy_return_type' (line 684)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'stypy_return_type', XY_122553)
        
        # ################# End of '_make_verts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_make_verts' in the type store
        # Getting the type of 'stypy_return_type' (line 627)
        stypy_return_type_122554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122554)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_make_verts'
        return stypy_return_type_122554


    @norecursion
    def _h_arrows(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_h_arrows'
        module_type_store = module_type_store.open_function_context('_h_arrows', 686, 4, False)
        # Assigning a type to the variable 'self' (line 687)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Quiver._h_arrows.__dict__.__setitem__('stypy_localization', localization)
        Quiver._h_arrows.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Quiver._h_arrows.__dict__.__setitem__('stypy_type_store', module_type_store)
        Quiver._h_arrows.__dict__.__setitem__('stypy_function_name', 'Quiver._h_arrows')
        Quiver._h_arrows.__dict__.__setitem__('stypy_param_names_list', ['length'])
        Quiver._h_arrows.__dict__.__setitem__('stypy_varargs_param_name', None)
        Quiver._h_arrows.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Quiver._h_arrows.__dict__.__setitem__('stypy_call_defaults', defaults)
        Quiver._h_arrows.__dict__.__setitem__('stypy_call_varargs', varargs)
        Quiver._h_arrows.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Quiver._h_arrows.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Quiver._h_arrows', ['length'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_h_arrows', localization, ['length'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_h_arrows(...)' code ##################

        unicode_122555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 8), 'unicode', u' length is in arrow width units ')
        
        # Assigning a BinOp to a Name (line 691):
        
        # Assigning a BinOp to a Name (line 691):
        # Getting the type of 'self' (line 691)
        self_122556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'self')
        # Obtaining the member 'minshaft' of a type (line 691)
        minshaft_122557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 16), self_122556, 'minshaft')
        # Getting the type of 'self' (line 691)
        self_122558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 32), 'self')
        # Obtaining the member 'headlength' of a type (line 691)
        headlength_122559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 32), self_122558, 'headlength')
        # Applying the binary operator '*' (line 691)
        result_mul_122560 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 16), '*', minshaft_122557, headlength_122559)
        
        # Assigning a type to the variable 'minsh' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 8), 'minsh', result_mul_122560)
        
        # Assigning a Call to a Name (line 692):
        
        # Assigning a Call to a Name (line 692):
        
        # Call to len(...): (line 692)
        # Processing the call arguments (line 692)
        # Getting the type of 'length' (line 692)
        length_122562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'length', False)
        # Processing the call keyword arguments (line 692)
        kwargs_122563 = {}
        # Getting the type of 'len' (line 692)
        len_122561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 12), 'len', False)
        # Calling len(args, kwargs) (line 692)
        len_call_result_122564 = invoke(stypy.reporting.localization.Localization(__file__, 692, 12), len_122561, *[length_122562], **kwargs_122563)
        
        # Assigning a type to the variable 'N' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 8), 'N', len_call_result_122564)
        
        # Assigning a Call to a Name (line 693):
        
        # Assigning a Call to a Name (line 693):
        
        # Call to reshape(...): (line 693)
        # Processing the call arguments (line 693)
        # Getting the type of 'N' (line 693)
        N_122567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 32), 'N', False)
        int_122568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 35), 'int')
        # Processing the call keyword arguments (line 693)
        kwargs_122569 = {}
        # Getting the type of 'length' (line 693)
        length_122565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 17), 'length', False)
        # Obtaining the member 'reshape' of a type (line 693)
        reshape_122566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 17), length_122565, 'reshape')
        # Calling reshape(args, kwargs) (line 693)
        reshape_call_result_122570 = invoke(stypy.reporting.localization.Localization(__file__, 693, 17), reshape_122566, *[N_122567, int_122568], **kwargs_122569)
        
        # Assigning a type to the variable 'length' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'length', reshape_call_result_122570)
        
        # Call to clip(...): (line 697)
        # Processing the call arguments (line 697)
        # Getting the type of 'length' (line 697)
        length_122573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'length', False)
        int_122574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 24), 'int')
        int_122575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 27), 'int')
        int_122576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 32), 'int')
        # Applying the binary operator '**' (line 697)
        result_pow_122577 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 27), '**', int_122575, int_122576)
        
        # Processing the call keyword arguments (line 697)
        # Getting the type of 'length' (line 697)
        length_122578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 40), 'length', False)
        keyword_122579 = length_122578
        kwargs_122580 = {'out': keyword_122579}
        # Getting the type of 'np' (line 697)
        np_122571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'np', False)
        # Obtaining the member 'clip' of a type (line 697)
        clip_122572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), np_122571, 'clip')
        # Calling clip(args, kwargs) (line 697)
        clip_call_result_122581 = invoke(stypy.reporting.localization.Localization(__file__, 697, 8), clip_122572, *[length_122573, int_122574, result_pow_122577], **kwargs_122580)
        
        
        # Assigning a Call to a Name (line 699):
        
        # Assigning a Call to a Name (line 699):
        
        # Call to array(...): (line 699)
        # Processing the call arguments (line 699)
        
        # Obtaining an instance of the builtin type 'list' (line 699)
        list_122584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 699)
        # Adding element type (line 699)
        int_122585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 699, 21), list_122584, int_122585)
        # Adding element type (line 699)
        
        # Getting the type of 'self' (line 699)
        self_122586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 26), 'self', False)
        # Obtaining the member 'headaxislength' of a type (line 699)
        headaxislength_122587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 26), self_122586, 'headaxislength')
        # Applying the 'usub' unary operator (line 699)
        result___neg___122588 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 25), 'usub', headaxislength_122587)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 699, 21), list_122584, result___neg___122588)
        # Adding element type (line 699)
        
        # Getting the type of 'self' (line 700)
        self_122589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 23), 'self', False)
        # Obtaining the member 'headlength' of a type (line 700)
        headlength_122590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 23), self_122589, 'headlength')
        # Applying the 'usub' unary operator (line 700)
        result___neg___122591 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 22), 'usub', headlength_122590)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 699, 21), list_122584, result___neg___122591)
        # Adding element type (line 699)
        int_122592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 699, 21), list_122584, int_122592)
        
        # Getting the type of 'np' (line 701)
        np_122593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 21), 'np', False)
        # Obtaining the member 'float64' of a type (line 701)
        float64_122594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 21), np_122593, 'float64')
        # Processing the call keyword arguments (line 699)
        kwargs_122595 = {}
        # Getting the type of 'np' (line 699)
        np_122582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 699)
        array_122583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 12), np_122582, 'array')
        # Calling array(args, kwargs) (line 699)
        array_call_result_122596 = invoke(stypy.reporting.localization.Localization(__file__, 699, 12), array_122583, *[list_122584, float64_122594], **kwargs_122595)
        
        # Assigning a type to the variable 'x' (line 699)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'x', array_call_result_122596)
        
        # Assigning a BinOp to a Name (line 702):
        
        # Assigning a BinOp to a Name (line 702):
        # Getting the type of 'x' (line 702)
        x_122597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'x')
        
        # Call to array(...): (line 702)
        # Processing the call arguments (line 702)
        
        # Obtaining an instance of the builtin type 'list' (line 702)
        list_122600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 702)
        # Adding element type (line 702)
        int_122601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 25), list_122600, int_122601)
        # Adding element type (line 702)
        int_122602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 25), list_122600, int_122602)
        # Adding element type (line 702)
        int_122603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 25), list_122600, int_122603)
        # Adding element type (line 702)
        int_122604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 702, 25), list_122600, int_122604)
        
        # Processing the call keyword arguments (line 702)
        kwargs_122605 = {}
        # Getting the type of 'np' (line 702)
        np_122598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 702)
        array_122599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 16), np_122598, 'array')
        # Calling array(args, kwargs) (line 702)
        array_call_result_122606 = invoke(stypy.reporting.localization.Localization(__file__, 702, 16), array_122599, *[list_122600], **kwargs_122605)
        
        # Getting the type of 'length' (line 702)
        length_122607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 41), 'length')
        # Applying the binary operator '*' (line 702)
        result_mul_122608 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 16), '*', array_call_result_122606, length_122607)
        
        # Applying the binary operator '+' (line 702)
        result_add_122609 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 12), '+', x_122597, result_mul_122608)
        
        # Assigning a type to the variable 'x' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'x', result_add_122609)
        
        # Assigning a BinOp to a Name (line 703):
        
        # Assigning a BinOp to a Name (line 703):
        float_122610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 12), 'float')
        
        # Call to array(...): (line 703)
        # Processing the call arguments (line 703)
        
        # Obtaining an instance of the builtin type 'list' (line 703)
        list_122613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 703)
        # Adding element type (line 703)
        int_122614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 27), list_122613, int_122614)
        # Adding element type (line 703)
        int_122615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 27), list_122613, int_122615)
        # Adding element type (line 703)
        # Getting the type of 'self' (line 703)
        self_122616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 34), 'self', False)
        # Obtaining the member 'headwidth' of a type (line 703)
        headwidth_122617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 34), self_122616, 'headwidth')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 27), list_122613, headwidth_122617)
        # Adding element type (line 703)
        int_122618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 703, 27), list_122613, int_122618)
        
        # Getting the type of 'np' (line 703)
        np_122619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 54), 'np', False)
        # Obtaining the member 'float64' of a type (line 703)
        float64_122620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 54), np_122619, 'float64')
        # Processing the call keyword arguments (line 703)
        kwargs_122621 = {}
        # Getting the type of 'np' (line 703)
        np_122611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 703)
        array_122612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 18), np_122611, 'array')
        # Calling array(args, kwargs) (line 703)
        array_call_result_122622 = invoke(stypy.reporting.localization.Localization(__file__, 703, 18), array_122612, *[list_122613, float64_122620], **kwargs_122621)
        
        # Applying the binary operator '*' (line 703)
        result_mul_122623 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 12), '*', float_122610, array_call_result_122622)
        
        # Assigning a type to the variable 'y' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'y', result_mul_122623)
        
        # Assigning a Call to a Name (line 704):
        
        # Assigning a Call to a Name (line 704):
        
        # Call to repeat(...): (line 704)
        # Processing the call arguments (line 704)
        
        # Obtaining the type of the subscript
        # Getting the type of 'np' (line 704)
        np_122626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 24), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 704)
        newaxis_122627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 24), np_122626, 'newaxis')
        slice_122628 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 704, 22), None, None, None)
        # Getting the type of 'y' (line 704)
        y_122629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 22), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 704)
        getitem___122630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 22), y_122629, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 704)
        subscript_call_result_122631 = invoke(stypy.reporting.localization.Localization(__file__, 704, 22), getitem___122630, (newaxis_122627, slice_122628))
        
        # Getting the type of 'N' (line 704)
        N_122632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 40), 'N', False)
        # Processing the call keyword arguments (line 704)
        int_122633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 48), 'int')
        keyword_122634 = int_122633
        kwargs_122635 = {'axis': keyword_122634}
        # Getting the type of 'np' (line 704)
        np_122624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 12), 'np', False)
        # Obtaining the member 'repeat' of a type (line 704)
        repeat_122625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 12), np_122624, 'repeat')
        # Calling repeat(args, kwargs) (line 704)
        repeat_call_result_122636 = invoke(stypy.reporting.localization.Localization(__file__, 704, 12), repeat_122625, *[subscript_call_result_122631, N_122632], **kwargs_122635)
        
        # Assigning a type to the variable 'y' (line 704)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'y', repeat_call_result_122636)
        
        # Assigning a Call to a Name (line 706):
        
        # Assigning a Call to a Name (line 706):
        
        # Call to array(...): (line 706)
        # Processing the call arguments (line 706)
        
        # Obtaining an instance of the builtin type 'list' (line 706)
        list_122639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 706)
        # Adding element type (line 706)
        int_122640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 22), list_122639, int_122640)
        # Adding element type (line 706)
        # Getting the type of 'minsh' (line 706)
        minsh_122641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 26), 'minsh', False)
        # Getting the type of 'self' (line 706)
        self_122642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 34), 'self', False)
        # Obtaining the member 'headaxislength' of a type (line 706)
        headaxislength_122643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 34), self_122642, 'headaxislength')
        # Applying the binary operator '-' (line 706)
        result_sub_122644 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 26), '-', minsh_122641, headaxislength_122643)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 22), list_122639, result_sub_122644)
        # Adding element type (line 706)
        # Getting the type of 'minsh' (line 707)
        minsh_122645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 23), 'minsh', False)
        # Getting the type of 'self' (line 707)
        self_122646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 31), 'self', False)
        # Obtaining the member 'headlength' of a type (line 707)
        headlength_122647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 31), self_122646, 'headlength')
        # Applying the binary operator '-' (line 707)
        result_sub_122648 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 23), '-', minsh_122645, headlength_122647)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 22), list_122639, result_sub_122648)
        # Adding element type (line 706)
        # Getting the type of 'minsh' (line 707)
        minsh_122649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 48), 'minsh', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 22), list_122639, minsh_122649)
        
        # Getting the type of 'np' (line 707)
        np_122650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 56), 'np', False)
        # Obtaining the member 'float64' of a type (line 707)
        float64_122651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 56), np_122650, 'float64')
        # Processing the call keyword arguments (line 706)
        kwargs_122652 = {}
        # Getting the type of 'np' (line 706)
        np_122637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 706)
        array_122638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 13), np_122637, 'array')
        # Calling array(args, kwargs) (line 706)
        array_call_result_122653 = invoke(stypy.reporting.localization.Localization(__file__, 706, 13), array_122638, *[list_122639, float64_122651], **kwargs_122652)
        
        # Assigning a type to the variable 'x0' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'x0', array_call_result_122653)
        
        # Assigning a BinOp to a Name (line 708):
        
        # Assigning a BinOp to a Name (line 708):
        float_122654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 13), 'float')
        
        # Call to array(...): (line 708)
        # Processing the call arguments (line 708)
        
        # Obtaining an instance of the builtin type 'list' (line 708)
        list_122657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 708)
        # Adding element type (line 708)
        int_122658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 28), list_122657, int_122658)
        # Adding element type (line 708)
        int_122659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 28), list_122657, int_122659)
        # Adding element type (line 708)
        # Getting the type of 'self' (line 708)
        self_122660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 35), 'self', False)
        # Obtaining the member 'headwidth' of a type (line 708)
        headwidth_122661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 35), self_122660, 'headwidth')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 28), list_122657, headwidth_122661)
        # Adding element type (line 708)
        int_122662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 28), list_122657, int_122662)
        
        # Getting the type of 'np' (line 708)
        np_122663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 55), 'np', False)
        # Obtaining the member 'float64' of a type (line 708)
        float64_122664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 55), np_122663, 'float64')
        # Processing the call keyword arguments (line 708)
        kwargs_122665 = {}
        # Getting the type of 'np' (line 708)
        np_122655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 708)
        array_122656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 19), np_122655, 'array')
        # Calling array(args, kwargs) (line 708)
        array_call_result_122666 = invoke(stypy.reporting.localization.Localization(__file__, 708, 19), array_122656, *[list_122657, float64_122664], **kwargs_122665)
        
        # Applying the binary operator '*' (line 708)
        result_mul_122667 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 13), '*', float_122654, array_call_result_122666)
        
        # Assigning a type to the variable 'y0' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'y0', result_mul_122667)
        
        # Assigning a List to a Name (line 709):
        
        # Assigning a List to a Name (line 709):
        
        # Obtaining an instance of the builtin type 'list' (line 709)
        list_122668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 709)
        # Adding element type (line 709)
        int_122669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 13), list_122668, int_122669)
        # Adding element type (line 709)
        int_122670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 13), list_122668, int_122670)
        # Adding element type (line 709)
        int_122671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 13), list_122668, int_122671)
        # Adding element type (line 709)
        int_122672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 13), list_122668, int_122672)
        # Adding element type (line 709)
        int_122673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 13), list_122668, int_122673)
        # Adding element type (line 709)
        int_122674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 13), list_122668, int_122674)
        # Adding element type (line 709)
        int_122675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 13), list_122668, int_122675)
        # Adding element type (line 709)
        int_122676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 709, 13), list_122668, int_122676)
        
        # Assigning a type to the variable 'ii' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'ii', list_122668)
        
        # Assigning a Call to a Name (line 710):
        
        # Assigning a Call to a Name (line 710):
        
        # Call to take(...): (line 710)
        # Processing the call arguments (line 710)
        # Getting the type of 'ii' (line 710)
        ii_122679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 19), 'ii', False)
        int_122680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 23), 'int')
        # Processing the call keyword arguments (line 710)
        kwargs_122681 = {}
        # Getting the type of 'x' (line 710)
        x_122677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 12), 'x', False)
        # Obtaining the member 'take' of a type (line 710)
        take_122678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 12), x_122677, 'take')
        # Calling take(args, kwargs) (line 710)
        take_call_result_122682 = invoke(stypy.reporting.localization.Localization(__file__, 710, 12), take_122678, *[ii_122679, int_122680], **kwargs_122681)
        
        # Assigning a type to the variable 'X' (line 710)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 8), 'X', take_call_result_122682)
        
        # Assigning a Call to a Name (line 711):
        
        # Assigning a Call to a Name (line 711):
        
        # Call to take(...): (line 711)
        # Processing the call arguments (line 711)
        # Getting the type of 'ii' (line 711)
        ii_122685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 19), 'ii', False)
        int_122686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 23), 'int')
        # Processing the call keyword arguments (line 711)
        kwargs_122687 = {}
        # Getting the type of 'y' (line 711)
        y_122683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 12), 'y', False)
        # Obtaining the member 'take' of a type (line 711)
        take_122684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 12), y_122683, 'take')
        # Calling take(args, kwargs) (line 711)
        take_call_result_122688 = invoke(stypy.reporting.localization.Localization(__file__, 711, 12), take_122684, *[ii_122685, int_122686], **kwargs_122687)
        
        # Assigning a type to the variable 'Y' (line 711)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'Y', take_call_result_122688)
        
        # Getting the type of 'Y' (line 712)
        Y_122689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'Y')
        
        # Obtaining the type of the subscript
        slice_122690 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 712, 8), None, None, None)
        int_122691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 13), 'int')
        int_122692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 15), 'int')
        slice_122693 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 712, 8), int_122691, int_122692, None)
        # Getting the type of 'Y' (line 712)
        Y_122694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'Y')
        # Obtaining the member '__getitem__' of a type (line 712)
        getitem___122695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 8), Y_122694, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 712)
        subscript_call_result_122696 = invoke(stypy.reporting.localization.Localization(__file__, 712, 8), getitem___122695, (slice_122690, slice_122693))
        
        int_122697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 22), 'int')
        # Applying the binary operator '*=' (line 712)
        result_imul_122698 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 8), '*=', subscript_call_result_122696, int_122697)
        # Getting the type of 'Y' (line 712)
        Y_122699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'Y')
        slice_122700 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 712, 8), None, None, None)
        int_122701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 13), 'int')
        int_122702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 15), 'int')
        slice_122703 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 712, 8), int_122701, int_122702, None)
        # Storing an element on a container (line 712)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 8), Y_122699, ((slice_122700, slice_122703), result_imul_122698))
        
        
        # Assigning a Call to a Name (line 713):
        
        # Assigning a Call to a Name (line 713):
        
        # Call to take(...): (line 713)
        # Processing the call arguments (line 713)
        # Getting the type of 'ii' (line 713)
        ii_122706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 21), 'ii', False)
        # Processing the call keyword arguments (line 713)
        kwargs_122707 = {}
        # Getting the type of 'x0' (line 713)
        x0_122704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 13), 'x0', False)
        # Obtaining the member 'take' of a type (line 713)
        take_122705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 13), x0_122704, 'take')
        # Calling take(args, kwargs) (line 713)
        take_call_result_122708 = invoke(stypy.reporting.localization.Localization(__file__, 713, 13), take_122705, *[ii_122706], **kwargs_122707)
        
        # Assigning a type to the variable 'X0' (line 713)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'X0', take_call_result_122708)
        
        # Assigning a Call to a Name (line 714):
        
        # Assigning a Call to a Name (line 714):
        
        # Call to take(...): (line 714)
        # Processing the call arguments (line 714)
        # Getting the type of 'ii' (line 714)
        ii_122711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 21), 'ii', False)
        # Processing the call keyword arguments (line 714)
        kwargs_122712 = {}
        # Getting the type of 'y0' (line 714)
        y0_122709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 13), 'y0', False)
        # Obtaining the member 'take' of a type (line 714)
        take_122710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 13), y0_122709, 'take')
        # Calling take(args, kwargs) (line 714)
        take_call_result_122713 = invoke(stypy.reporting.localization.Localization(__file__, 714, 13), take_122710, *[ii_122711], **kwargs_122712)
        
        # Assigning a type to the variable 'Y0' (line 714)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'Y0', take_call_result_122713)
        
        # Getting the type of 'Y0' (line 715)
        Y0_122714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'Y0')
        
        # Obtaining the type of the subscript
        int_122715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 11), 'int')
        int_122716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 13), 'int')
        slice_122717 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 715, 8), int_122715, int_122716, None)
        # Getting the type of 'Y0' (line 715)
        Y0_122718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'Y0')
        # Obtaining the member '__getitem__' of a type (line 715)
        getitem___122719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 8), Y0_122718, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 715)
        subscript_call_result_122720 = invoke(stypy.reporting.localization.Localization(__file__, 715, 8), getitem___122719, slice_122717)
        
        int_122721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 20), 'int')
        # Applying the binary operator '*=' (line 715)
        result_imul_122722 = python_operator(stypy.reporting.localization.Localization(__file__, 715, 8), '*=', subscript_call_result_122720, int_122721)
        # Getting the type of 'Y0' (line 715)
        Y0_122723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'Y0')
        int_122724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 11), 'int')
        int_122725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 13), 'int')
        slice_122726 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 715, 8), int_122724, int_122725, None)
        # Storing an element on a container (line 715)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 8), Y0_122723, (slice_122726, result_imul_122722))
        
        
        # Assigning a IfExp to a Name (line 716):
        
        # Assigning a IfExp to a Name (line 716):
        
        
        # Getting the type of 'minsh' (line 716)
        minsh_122727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 35), 'minsh')
        float_122728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 44), 'float')
        # Applying the binary operator '!=' (line 716)
        result_ne_122729 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 35), '!=', minsh_122727, float_122728)
        
        # Testing the type of an if expression (line 716)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 716, 17), result_ne_122729)
        # SSA begins for if expression (line 716)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'length' (line 716)
        length_122730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 17), 'length')
        # Getting the type of 'minsh' (line 716)
        minsh_122731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 26), 'minsh')
        # Applying the binary operator 'div' (line 716)
        result_div_122732 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 17), 'div', length_122730, minsh_122731)
        
        # SSA branch for the else part of an if expression (line 716)
        module_type_store.open_ssa_branch('if expression else')
        float_122733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 52), 'float')
        # SSA join for if expression (line 716)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_122734 = union_type.UnionType.add(result_div_122732, float_122733)
        
        # Assigning a type to the variable 'shrink' (line 716)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'shrink', if_exp_122734)
        
        # Assigning a BinOp to a Name (line 717):
        
        # Assigning a BinOp to a Name (line 717):
        # Getting the type of 'shrink' (line 717)
        shrink_122735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 13), 'shrink')
        
        # Obtaining the type of the subscript
        # Getting the type of 'np' (line 717)
        np_122736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 25), 'np')
        # Obtaining the member 'newaxis' of a type (line 717)
        newaxis_122737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 25), np_122736, 'newaxis')
        slice_122738 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 717, 22), None, None, None)
        # Getting the type of 'X0' (line 717)
        X0_122739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 22), 'X0')
        # Obtaining the member '__getitem__' of a type (line 717)
        getitem___122740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 22), X0_122739, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 717)
        subscript_call_result_122741 = invoke(stypy.reporting.localization.Localization(__file__, 717, 22), getitem___122740, (newaxis_122737, slice_122738))
        
        # Applying the binary operator '*' (line 717)
        result_mul_122742 = python_operator(stypy.reporting.localization.Localization(__file__, 717, 13), '*', shrink_122735, subscript_call_result_122741)
        
        # Assigning a type to the variable 'X0' (line 717)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'X0', result_mul_122742)
        
        # Assigning a BinOp to a Name (line 718):
        
        # Assigning a BinOp to a Name (line 718):
        # Getting the type of 'shrink' (line 718)
        shrink_122743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 13), 'shrink')
        
        # Obtaining the type of the subscript
        # Getting the type of 'np' (line 718)
        np_122744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 25), 'np')
        # Obtaining the member 'newaxis' of a type (line 718)
        newaxis_122745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 25), np_122744, 'newaxis')
        slice_122746 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 718, 22), None, None, None)
        # Getting the type of 'Y0' (line 718)
        Y0_122747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 22), 'Y0')
        # Obtaining the member '__getitem__' of a type (line 718)
        getitem___122748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 22), Y0_122747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 718)
        subscript_call_result_122749 = invoke(stypy.reporting.localization.Localization(__file__, 718, 22), getitem___122748, (newaxis_122745, slice_122746))
        
        # Applying the binary operator '*' (line 718)
        result_mul_122750 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 13), '*', shrink_122743, subscript_call_result_122749)
        
        # Assigning a type to the variable 'Y0' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'Y0', result_mul_122750)
        
        # Assigning a Call to a Name (line 719):
        
        # Assigning a Call to a Name (line 719):
        
        # Call to repeat(...): (line 719)
        # Processing the call arguments (line 719)
        
        # Getting the type of 'length' (line 719)
        length_122753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 26), 'length', False)
        # Getting the type of 'minsh' (line 719)
        minsh_122754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 35), 'minsh', False)
        # Applying the binary operator '<' (line 719)
        result_lt_122755 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 26), '<', length_122753, minsh_122754)
        
        int_122756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 42), 'int')
        # Processing the call keyword arguments (line 719)
        int_122757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 50), 'int')
        keyword_122758 = int_122757
        kwargs_122759 = {'axis': keyword_122758}
        # Getting the type of 'np' (line 719)
        np_122751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 16), 'np', False)
        # Obtaining the member 'repeat' of a type (line 719)
        repeat_122752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 16), np_122751, 'repeat')
        # Calling repeat(args, kwargs) (line 719)
        repeat_call_result_122760 = invoke(stypy.reporting.localization.Localization(__file__, 719, 16), repeat_122752, *[result_lt_122755, int_122756], **kwargs_122759)
        
        # Assigning a type to the variable 'short' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'short', repeat_call_result_122760)
        
        # Call to copyto(...): (line 721)
        # Processing the call arguments (line 721)
        # Getting the type of 'X' (line 721)
        X_122763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 18), 'X', False)
        # Getting the type of 'X0' (line 721)
        X0_122764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 21), 'X0', False)
        # Processing the call keyword arguments (line 721)
        # Getting the type of 'short' (line 721)
        short_122765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 31), 'short', False)
        keyword_122766 = short_122765
        kwargs_122767 = {'where': keyword_122766}
        # Getting the type of 'np' (line 721)
        np_122761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'np', False)
        # Obtaining the member 'copyto' of a type (line 721)
        copyto_122762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 8), np_122761, 'copyto')
        # Calling copyto(args, kwargs) (line 721)
        copyto_call_result_122768 = invoke(stypy.reporting.localization.Localization(__file__, 721, 8), copyto_122762, *[X_122763, X0_122764], **kwargs_122767)
        
        
        # Call to copyto(...): (line 722)
        # Processing the call arguments (line 722)
        # Getting the type of 'Y' (line 722)
        Y_122771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 18), 'Y', False)
        # Getting the type of 'Y0' (line 722)
        Y0_122772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 21), 'Y0', False)
        # Processing the call keyword arguments (line 722)
        # Getting the type of 'short' (line 722)
        short_122773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 31), 'short', False)
        keyword_122774 = short_122773
        kwargs_122775 = {'where': keyword_122774}
        # Getting the type of 'np' (line 722)
        np_122769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'np', False)
        # Obtaining the member 'copyto' of a type (line 722)
        copyto_122770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 8), np_122769, 'copyto')
        # Calling copyto(args, kwargs) (line 722)
        copyto_call_result_122776 = invoke(stypy.reporting.localization.Localization(__file__, 722, 8), copyto_122770, *[Y_122771, Y0_122772], **kwargs_122775)
        
        
        
        # Getting the type of 'self' (line 723)
        self_122777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 11), 'self')
        # Obtaining the member 'pivot' of a type (line 723)
        pivot_122778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 11), self_122777, 'pivot')
        unicode_122779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 25), 'unicode', u'middle')
        # Applying the binary operator '==' (line 723)
        result_eq_122780 = python_operator(stypy.reporting.localization.Localization(__file__, 723, 11), '==', pivot_122778, unicode_122779)
        
        # Testing the type of an if condition (line 723)
        if_condition_122781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 723, 8), result_eq_122780)
        # Assigning a type to the variable 'if_condition_122781' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'if_condition_122781', if_condition_122781)
        # SSA begins for if statement (line 723)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'X' (line 724)
        X_122782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 12), 'X')
        float_122783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 17), 'float')
        
        # Obtaining the type of the subscript
        slice_122784 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 724, 23), None, None, None)
        int_122785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 28), 'int')
        # Getting the type of 'np' (line 724)
        np_122786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 31), 'np')
        # Obtaining the member 'newaxis' of a type (line 724)
        newaxis_122787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 31), np_122786, 'newaxis')
        # Getting the type of 'X' (line 724)
        X_122788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 23), 'X')
        # Obtaining the member '__getitem__' of a type (line 724)
        getitem___122789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 23), X_122788, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 724)
        subscript_call_result_122790 = invoke(stypy.reporting.localization.Localization(__file__, 724, 23), getitem___122789, (slice_122784, int_122785, newaxis_122787))
        
        # Applying the binary operator '*' (line 724)
        result_mul_122791 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 17), '*', float_122783, subscript_call_result_122790)
        
        # Applying the binary operator '-=' (line 724)
        result_isub_122792 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 12), '-=', X_122782, result_mul_122791)
        # Assigning a type to the variable 'X' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 12), 'X', result_isub_122792)
        
        # SSA branch for the else part of an if statement (line 723)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 725)
        self_122793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 13), 'self')
        # Obtaining the member 'pivot' of a type (line 725)
        pivot_122794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 13), self_122793, 'pivot')
        unicode_122795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 27), 'unicode', u'tip')
        # Applying the binary operator '==' (line 725)
        result_eq_122796 = python_operator(stypy.reporting.localization.Localization(__file__, 725, 13), '==', pivot_122794, unicode_122795)
        
        # Testing the type of an if condition (line 725)
        if_condition_122797 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 725, 13), result_eq_122796)
        # Assigning a type to the variable 'if_condition_122797' (line 725)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 13), 'if_condition_122797', if_condition_122797)
        # SSA begins for if statement (line 725)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 726):
        
        # Assigning a BinOp to a Name (line 726):
        # Getting the type of 'X' (line 726)
        X_122798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 16), 'X')
        
        # Obtaining the type of the subscript
        slice_122799 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 726, 20), None, None, None)
        int_122800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 25), 'int')
        # Getting the type of 'np' (line 726)
        np_122801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 28), 'np')
        # Obtaining the member 'newaxis' of a type (line 726)
        newaxis_122802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 28), np_122801, 'newaxis')
        # Getting the type of 'X' (line 726)
        X_122803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 20), 'X')
        # Obtaining the member '__getitem__' of a type (line 726)
        getitem___122804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 20), X_122803, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 726)
        subscript_call_result_122805 = invoke(stypy.reporting.localization.Localization(__file__, 726, 20), getitem___122804, (slice_122799, int_122800, newaxis_122802))
        
        # Applying the binary operator '-' (line 726)
        result_sub_122806 = python_operator(stypy.reporting.localization.Localization(__file__, 726, 16), '-', X_122798, subscript_call_result_122805)
        
        # Assigning a type to the variable 'X' (line 726)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 12), 'X', result_sub_122806)
        # SSA branch for the else part of an if statement (line 725)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 729)
        self_122807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 13), 'self')
        # Obtaining the member 'pivot' of a type (line 729)
        pivot_122808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 729, 13), self_122807, 'pivot')
        unicode_122809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 27), 'unicode', u'tail')
        # Applying the binary operator '!=' (line 729)
        result_ne_122810 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 13), '!=', pivot_122808, unicode_122809)
        
        # Testing the type of an if condition (line 729)
        if_condition_122811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 13), result_ne_122810)
        # Assigning a type to the variable 'if_condition_122811' (line 729)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 13), 'if_condition_122811', if_condition_122811)
        # SSA begins for if statement (line 729)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 730)
        # Processing the call arguments (line 730)
        
        # Call to format(...): (line 730)
        # Processing the call arguments (line 730)
        # Getting the type of 'self' (line 731)
        self_122815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 64), 'self', False)
        # Obtaining the member 'pivot' of a type (line 731)
        pivot_122816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 64), self_122815, 'pivot')
        # Processing the call keyword arguments (line 730)
        kwargs_122817 = {}
        unicode_122813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 30), 'unicode', u"Quiver.pivot must have value in {{'middle', 'tip', 'tail'}} not {0}")
        # Obtaining the member 'format' of a type (line 730)
        format_122814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 30), unicode_122813, 'format')
        # Calling format(args, kwargs) (line 730)
        format_call_result_122818 = invoke(stypy.reporting.localization.Localization(__file__, 730, 30), format_122814, *[pivot_122816], **kwargs_122817)
        
        # Processing the call keyword arguments (line 730)
        kwargs_122819 = {}
        # Getting the type of 'ValueError' (line 730)
        ValueError_122812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 730)
        ValueError_call_result_122820 = invoke(stypy.reporting.localization.Localization(__file__, 730, 18), ValueError_122812, *[format_call_result_122818], **kwargs_122819)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 730, 12), ValueError_call_result_122820, 'raise parameter', BaseException)
        # SSA join for if statement (line 729)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 725)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 723)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Compare to a Name (line 733):
        
        # Assigning a Compare to a Name (line 733):
        
        # Getting the type of 'length' (line 733)
        length_122821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 19), 'length')
        # Getting the type of 'self' (line 733)
        self_122822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 28), 'self')
        # Obtaining the member 'minlength' of a type (line 733)
        minlength_122823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 28), self_122822, 'minlength')
        # Applying the binary operator '<' (line 733)
        result_lt_122824 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 19), '<', length_122821, minlength_122823)
        
        # Assigning a type to the variable 'tooshort' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'tooshort', result_lt_122824)
        
        
        # Call to any(...): (line 734)
        # Processing the call keyword arguments (line 734)
        kwargs_122827 = {}
        # Getting the type of 'tooshort' (line 734)
        tooshort_122825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 11), 'tooshort', False)
        # Obtaining the member 'any' of a type (line 734)
        any_122826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 11), tooshort_122825, 'any')
        # Calling any(args, kwargs) (line 734)
        any_call_result_122828 = invoke(stypy.reporting.localization.Localization(__file__, 734, 11), any_122826, *[], **kwargs_122827)
        
        # Testing the type of an if condition (line 734)
        if_condition_122829 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 734, 8), any_call_result_122828)
        # Assigning a type to the variable 'if_condition_122829' (line 734)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'if_condition_122829', if_condition_122829)
        # SSA begins for if statement (line 734)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 736):
        
        # Assigning a BinOp to a Name (line 736):
        
        # Call to arange(...): (line 736)
        # Processing the call arguments (line 736)
        int_122832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 27), 'int')
        int_122833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 30), 'int')
        int_122834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 33), 'int')
        # Getting the type of 'np' (line 736)
        np_122835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 36), 'np', False)
        # Obtaining the member 'float64' of a type (line 736)
        float64_122836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 36), np_122835, 'float64')
        # Processing the call keyword arguments (line 736)
        kwargs_122837 = {}
        # Getting the type of 'np' (line 736)
        np_122830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 736)
        arange_122831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 17), np_122830, 'arange')
        # Calling arange(args, kwargs) (line 736)
        arange_call_result_122838 = invoke(stypy.reporting.localization.Localization(__file__, 736, 17), arange_122831, *[int_122832, int_122833, int_122834, float64_122836], **kwargs_122837)
        
        # Getting the type of 'np' (line 736)
        np_122839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 51), 'np')
        # Obtaining the member 'pi' of a type (line 736)
        pi_122840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 51), np_122839, 'pi')
        float_122841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 59), 'float')
        # Applying the binary operator 'div' (line 736)
        result_div_122842 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 51), 'div', pi_122840, float_122841)
        
        # Applying the binary operator '*' (line 736)
        result_mul_122843 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 17), '*', arange_call_result_122838, result_div_122842)
        
        # Assigning a type to the variable 'th' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 12), 'th', result_mul_122843)
        
        # Assigning a BinOp to a Name (line 737):
        
        # Assigning a BinOp to a Name (line 737):
        
        # Call to cos(...): (line 737)
        # Processing the call arguments (line 737)
        # Getting the type of 'th' (line 737)
        th_122846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 24), 'th', False)
        # Processing the call keyword arguments (line 737)
        kwargs_122847 = {}
        # Getting the type of 'np' (line 737)
        np_122844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 17), 'np', False)
        # Obtaining the member 'cos' of a type (line 737)
        cos_122845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 17), np_122844, 'cos')
        # Calling cos(args, kwargs) (line 737)
        cos_call_result_122848 = invoke(stypy.reporting.localization.Localization(__file__, 737, 17), cos_122845, *[th_122846], **kwargs_122847)
        
        # Getting the type of 'self' (line 737)
        self_122849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 30), 'self')
        # Obtaining the member 'minlength' of a type (line 737)
        minlength_122850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 30), self_122849, 'minlength')
        # Applying the binary operator '*' (line 737)
        result_mul_122851 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 17), '*', cos_call_result_122848, minlength_122850)
        
        float_122852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 47), 'float')
        # Applying the binary operator '*' (line 737)
        result_mul_122853 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 45), '*', result_mul_122851, float_122852)
        
        # Assigning a type to the variable 'x1' (line 737)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 12), 'x1', result_mul_122853)
        
        # Assigning a BinOp to a Name (line 738):
        
        # Assigning a BinOp to a Name (line 738):
        
        # Call to sin(...): (line 738)
        # Processing the call arguments (line 738)
        # Getting the type of 'th' (line 738)
        th_122856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 24), 'th', False)
        # Processing the call keyword arguments (line 738)
        kwargs_122857 = {}
        # Getting the type of 'np' (line 738)
        np_122854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 17), 'np', False)
        # Obtaining the member 'sin' of a type (line 738)
        sin_122855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 17), np_122854, 'sin')
        # Calling sin(args, kwargs) (line 738)
        sin_call_result_122858 = invoke(stypy.reporting.localization.Localization(__file__, 738, 17), sin_122855, *[th_122856], **kwargs_122857)
        
        # Getting the type of 'self' (line 738)
        self_122859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 30), 'self')
        # Obtaining the member 'minlength' of a type (line 738)
        minlength_122860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 30), self_122859, 'minlength')
        # Applying the binary operator '*' (line 738)
        result_mul_122861 = python_operator(stypy.reporting.localization.Localization(__file__, 738, 17), '*', sin_call_result_122858, minlength_122860)
        
        float_122862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 47), 'float')
        # Applying the binary operator '*' (line 738)
        result_mul_122863 = python_operator(stypy.reporting.localization.Localization(__file__, 738, 45), '*', result_mul_122861, float_122862)
        
        # Assigning a type to the variable 'y1' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'y1', result_mul_122863)
        
        # Assigning a Call to a Name (line 739):
        
        # Assigning a Call to a Name (line 739):
        
        # Call to repeat(...): (line 739)
        # Processing the call arguments (line 739)
        
        # Obtaining the type of the subscript
        # Getting the type of 'np' (line 739)
        np_122866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 30), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 739)
        newaxis_122867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 30), np_122866, 'newaxis')
        slice_122868 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 739, 27), None, None, None)
        # Getting the type of 'x1' (line 739)
        x1_122869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 27), 'x1', False)
        # Obtaining the member '__getitem__' of a type (line 739)
        getitem___122870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 27), x1_122869, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 739)
        subscript_call_result_122871 = invoke(stypy.reporting.localization.Localization(__file__, 739, 27), getitem___122870, (newaxis_122867, slice_122868))
        
        # Getting the type of 'N' (line 739)
        N_122872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 46), 'N', False)
        # Processing the call keyword arguments (line 739)
        int_122873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 54), 'int')
        keyword_122874 = int_122873
        kwargs_122875 = {'axis': keyword_122874}
        # Getting the type of 'np' (line 739)
        np_122864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 17), 'np', False)
        # Obtaining the member 'repeat' of a type (line 739)
        repeat_122865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 17), np_122864, 'repeat')
        # Calling repeat(args, kwargs) (line 739)
        repeat_call_result_122876 = invoke(stypy.reporting.localization.Localization(__file__, 739, 17), repeat_122865, *[subscript_call_result_122871, N_122872], **kwargs_122875)
        
        # Assigning a type to the variable 'X1' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'X1', repeat_call_result_122876)
        
        # Assigning a Call to a Name (line 740):
        
        # Assigning a Call to a Name (line 740):
        
        # Call to repeat(...): (line 740)
        # Processing the call arguments (line 740)
        
        # Obtaining the type of the subscript
        # Getting the type of 'np' (line 740)
        np_122879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 30), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 740)
        newaxis_122880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 30), np_122879, 'newaxis')
        slice_122881 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 740, 27), None, None, None)
        # Getting the type of 'y1' (line 740)
        y1_122882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 27), 'y1', False)
        # Obtaining the member '__getitem__' of a type (line 740)
        getitem___122883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 27), y1_122882, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 740)
        subscript_call_result_122884 = invoke(stypy.reporting.localization.Localization(__file__, 740, 27), getitem___122883, (newaxis_122880, slice_122881))
        
        # Getting the type of 'N' (line 740)
        N_122885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 46), 'N', False)
        # Processing the call keyword arguments (line 740)
        int_122886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 54), 'int')
        keyword_122887 = int_122886
        kwargs_122888 = {'axis': keyword_122887}
        # Getting the type of 'np' (line 740)
        np_122877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 17), 'np', False)
        # Obtaining the member 'repeat' of a type (line 740)
        repeat_122878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 17), np_122877, 'repeat')
        # Calling repeat(args, kwargs) (line 740)
        repeat_call_result_122889 = invoke(stypy.reporting.localization.Localization(__file__, 740, 17), repeat_122878, *[subscript_call_result_122884, N_122885], **kwargs_122888)
        
        # Assigning a type to the variable 'Y1' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 'Y1', repeat_call_result_122889)
        
        # Assigning a Call to a Name (line 741):
        
        # Assigning a Call to a Name (line 741):
        
        # Call to repeat(...): (line 741)
        # Processing the call arguments (line 741)
        # Getting the type of 'tooshort' (line 741)
        tooshort_122892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 33), 'tooshort', False)
        int_122893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 43), 'int')
        int_122894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 46), 'int')
        # Processing the call keyword arguments (line 741)
        kwargs_122895 = {}
        # Getting the type of 'np' (line 741)
        np_122890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 23), 'np', False)
        # Obtaining the member 'repeat' of a type (line 741)
        repeat_122891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 23), np_122890, 'repeat')
        # Calling repeat(args, kwargs) (line 741)
        repeat_call_result_122896 = invoke(stypy.reporting.localization.Localization(__file__, 741, 23), repeat_122891, *[tooshort_122892, int_122893, int_122894], **kwargs_122895)
        
        # Assigning a type to the variable 'tooshort' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'tooshort', repeat_call_result_122896)
        
        # Call to copyto(...): (line 742)
        # Processing the call arguments (line 742)
        # Getting the type of 'X' (line 742)
        X_122899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 22), 'X', False)
        # Getting the type of 'X1' (line 742)
        X1_122900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 25), 'X1', False)
        # Processing the call keyword arguments (line 742)
        # Getting the type of 'tooshort' (line 742)
        tooshort_122901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 35), 'tooshort', False)
        keyword_122902 = tooshort_122901
        kwargs_122903 = {'where': keyword_122902}
        # Getting the type of 'np' (line 742)
        np_122897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 12), 'np', False)
        # Obtaining the member 'copyto' of a type (line 742)
        copyto_122898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 12), np_122897, 'copyto')
        # Calling copyto(args, kwargs) (line 742)
        copyto_call_result_122904 = invoke(stypy.reporting.localization.Localization(__file__, 742, 12), copyto_122898, *[X_122899, X1_122900], **kwargs_122903)
        
        
        # Call to copyto(...): (line 743)
        # Processing the call arguments (line 743)
        # Getting the type of 'Y' (line 743)
        Y_122907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 22), 'Y', False)
        # Getting the type of 'Y1' (line 743)
        Y1_122908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 25), 'Y1', False)
        # Processing the call keyword arguments (line 743)
        # Getting the type of 'tooshort' (line 743)
        tooshort_122909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 35), 'tooshort', False)
        keyword_122910 = tooshort_122909
        kwargs_122911 = {'where': keyword_122910}
        # Getting the type of 'np' (line 743)
        np_122905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 12), 'np', False)
        # Obtaining the member 'copyto' of a type (line 743)
        copyto_122906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 12), np_122905, 'copyto')
        # Calling copyto(args, kwargs) (line 743)
        copyto_call_result_122912 = invoke(stypy.reporting.localization.Localization(__file__, 743, 12), copyto_122906, *[Y_122907, Y1_122908], **kwargs_122911)
        
        # SSA join for if statement (line 734)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 745)
        tuple_122913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 745)
        # Adding element type (line 745)
        # Getting the type of 'X' (line 745)
        X_122914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 15), 'X')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 15), tuple_122913, X_122914)
        # Adding element type (line 745)
        # Getting the type of 'Y' (line 745)
        Y_122915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 18), 'Y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 15), tuple_122913, Y_122915)
        
        # Assigning a type to the variable 'stypy_return_type' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'stypy_return_type', tuple_122913)
        
        # ################# End of '_h_arrows(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_h_arrows' in the type store
        # Getting the type of 'stypy_return_type' (line 686)
        stypy_return_type_122916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122916)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_h_arrows'
        return stypy_return_type_122916

    
    # Assigning a Name to a Name (line 747):

# Assigning a type to the variable 'Quiver' (line 415)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 0), 'Quiver', Quiver)

# Assigning a Tuple to a Name (line 433):

# Obtaining an instance of the builtin type 'tuple' (line 433)
tuple_122917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 433)
# Adding element type (line 433)
unicode_122918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 19), 'unicode', u'tail')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 19), tuple_122917, unicode_122918)
# Adding element type (line 433)
unicode_122919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 27), 'unicode', u'mid')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 19), tuple_122917, unicode_122919)
# Adding element type (line 433)
unicode_122920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 34), 'unicode', u'middle')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 19), tuple_122917, unicode_122920)
# Adding element type (line 433)
unicode_122921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 44), 'unicode', u'tip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 19), tuple_122917, unicode_122921)

# Getting the type of 'Quiver'
Quiver_122922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Quiver')
# Setting the type of the member '_PIVOT_VALS' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Quiver_122922, '_PIVOT_VALS', tuple_122917)

# Assigning a Name to a Name (line 747):
# Getting the type of '_quiver_doc' (line 747)
_quiver_doc_122923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 17), '_quiver_doc')
# Getting the type of 'Quiver'
Quiver_122924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Quiver')
# Setting the type of the member 'quiver_doc' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Quiver_122924, 'quiver_doc', _quiver_doc_122923)

# Assigning a BinOp to a Name (line 750):

# Assigning a BinOp to a Name (line 750):
unicode_122925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, (-1)), 'unicode', u'\nPlot a 2-D field of barbs.\n\nCall signatures::\n\n  barb(U, V, **kw)\n  barb(U, V, C, **kw)\n  barb(X, Y, U, V, **kw)\n  barb(X, Y, U, V, C, **kw)\n\nArguments:\n\n  *X*, *Y*:\n    The x and y coordinates of the barb locations\n    (default is head of barb; see *pivot* kwarg)\n\n  *U*, *V*:\n    Give the x and y components of the barb shaft\n\n  *C*:\n    An optional array used to map colors to the barbs\n\nAll arguments may be 1-D or 2-D arrays or sequences. If *X* and *Y*\nare absent, they will be generated as a uniform grid.  If *U* and *V*\nare 2-D arrays but *X* and *Y* are 1-D, and if ``len(X)`` and ``len(Y)``\nmatch the column and row dimensions of *U*, then *X* and *Y* will be\nexpanded with :func:`numpy.meshgrid`.\n\n*U*, *V*, *C* may be masked arrays, but masked *X*, *Y* are not\nsupported at present.\n\nKeyword arguments:\n\n  *length*:\n    Length of the barb in points; the other parts of the barb\n    are scaled against this.\n    Default is 7.\n\n  *pivot*: [ \'tip\' | \'middle\' | float ]\n    The part of the arrow that is at the grid point; the arrow rotates\n    about this point, hence the name *pivot*.  Default is \'tip\'. Can\n    also be a number, which shifts the start of the barb that many\n    points from the origin.\n\n  *barbcolor*: [ color | color sequence ]\n    Specifies the color all parts of the barb except any flags.  This\n    parameter is analagous to the *edgecolor* parameter for polygons,\n    which can be used instead. However this parameter will override\n    facecolor.\n\n  *flagcolor*: [ color | color sequence ]\n    Specifies the color of any flags on the barb.  This parameter is\n    analagous to the *facecolor* parameter for polygons, which can be\n    used instead. However this parameter will override facecolor.  If\n    this is not set (and *C* has not either) then *flagcolor* will be\n    set to match *barbcolor* so that the barb has a uniform color. If\n    *C* has been set, *flagcolor* has no effect.\n\n  *sizes*:\n    A dictionary of coefficients specifying the ratio of a given\n    feature to the length of the barb. Only those values one wishes to\n    override need to be included.  These features include:\n\n        - \'spacing\' - space between features (flags, full/half barbs)\n\n        - \'height\' - height (distance from shaft to top) of a flag or\n          full barb\n\n        - \'width\' - width of a flag, twice the width of a full barb\n\n        - \'emptybarb\' - radius of the circle used for low magnitudes\n\n  *fill_empty*:\n    A flag on whether the empty barbs (circles) that are drawn should\n    be filled with the flag color.  If they are not filled, they will\n    be drawn such that no color is applied to the center.  Default is\n    False\n\n  *rounding*:\n    A flag to indicate whether the vector magnitude should be rounded\n    when allocating barb components.  If True, the magnitude is\n    rounded to the nearest multiple of the half-barb increment.  If\n    False, the magnitude is simply truncated to the next lowest\n    multiple.  Default is True\n\n  *barb_increments*:\n    A dictionary of increments specifying values to associate with\n    different parts of the barb. Only those values one wishes to\n    override need to be included.\n\n        - \'half\' - half barbs (Default is 5)\n\n        - \'full\' - full barbs (Default is 10)\n\n        - \'flag\' - flags (default is 50)\n\n  *flip_barb*:\n    Either a single boolean flag or an array of booleans.  Single\n    boolean indicates whether the lines and flags should point\n    opposite to normal for all barbs.  An array (which should be the\n    same size as the other data arrays) indicates whether to flip for\n    each individual barb.  Normal behavior is for the barbs and lines\n    to point right (comes from wind barbs having these features point\n    towards low pressure in the Northern Hemisphere.)  Default is\n    False\n\nBarbs are traditionally used in meteorology as a way to plot the speed\nand direction of wind observations, but can technically be used to\nplot any two dimensional vector quantity.  As opposed to arrows, which\ngive vector magnitude by the length of the arrow, the barbs give more\nquantitative information about the vector magnitude by putting slanted\nlines or a triangle for various increments in magnitude, as show\nschematically below::\n\n :     /\\    \\\\\n :    /  \\    \\\\\n :   /    \\    \\    \\\\\n :  /      \\    \\    \\\\\n : ------------------------------\n\n.. note the double \\\\ at the end of each line to make the figure\n.. render correctly\n\nThe largest increment is given by a triangle (or "flag"). After those\ncome full lines (barbs). The smallest increment is a half line.  There\nis only, of course, ever at most 1 half line.  If the magnitude is\nsmall and only needs a single half-line and no full lines or\ntriangles, the half-line is offset from the end of the barb so that it\ncan be easily distinguished from barbs with a single full line.  The\nmagnitude for the barb shown above would nominally be 65, using the\nstandard increments of 50, 10, and 5.\n\nlinewidths and edgecolors can be used to customize the barb.\nAdditional :class:`~matplotlib.collections.PolyCollection` keyword\narguments:\n\n%(PolyCollection)s\n')
# Getting the type of 'docstring' (line 887)
docstring_122926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 6), 'docstring')
# Obtaining the member 'interpd' of a type (line 887)
interpd_122927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 6), docstring_122926, 'interpd')
# Obtaining the member 'params' of a type (line 887)
params_122928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 6), interpd_122927, 'params')
# Applying the binary operator '%' (line 887)
result_mod_122929 = python_operator(stypy.reporting.localization.Localization(__file__, 887, (-1)), '%', unicode_122925, params_122928)

# Assigning a type to the variable '_barbs_doc' (line 750)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 0), '_barbs_doc', result_mod_122929)

# Call to update(...): (line 889)
# Processing the call keyword arguments (line 889)
# Getting the type of '_barbs_doc' (line 889)
_barbs_doc_122933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 35), '_barbs_doc', False)
keyword_122934 = _barbs_doc_122933
kwargs_122935 = {'barbs_doc': keyword_122934}
# Getting the type of 'docstring' (line 889)
docstring_122930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 0), 'docstring', False)
# Obtaining the member 'interpd' of a type (line 889)
interpd_122931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 0), docstring_122930, 'interpd')
# Obtaining the member 'update' of a type (line 889)
update_122932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 0), interpd_122931, 'update')
# Calling update(args, kwargs) (line 889)
update_call_result_122936 = invoke(stypy.reporting.localization.Localization(__file__, 889, 0), update_122932, *[], **kwargs_122935)

# Declaration of the 'Barbs' class
# Getting the type of 'mcollections' (line 892)
mcollections_122937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 12), 'mcollections')
# Obtaining the member 'PolyCollection' of a type (line 892)
PolyCollection_122938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 12), mcollections_122937, 'PolyCollection')

class Barbs(PolyCollection_122938, ):
    unicode_122939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, (-1)), 'unicode', u'\n    Specialized PolyCollection for barbs.\n\n    The only API method is :meth:`set_UVC`, which can be used to\n    change the size, orientation, and color of the arrows.  Locations\n    are changed using the :meth:`set_offsets` collection method.\n    Possibly this method will be useful in animations.\n\n    There is one internal function :meth:`_find_tails` which finds\n    exactly what should be put on the barb given the vector magnitude.\n    From there :meth:`_make_barbs` is used to find the vertices of the\n    polygon to represent the barb based on this information.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 909, 4, False)
        # Assigning a type to the variable 'self' (line 910)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Barbs.__init__', ['ax'], 'args', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['ax'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_122940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, (-1)), 'unicode', u'\n        The constructor takes one required argument, an Axes\n        instance, followed by the args and kwargs described\n        by the following pylab interface documentation:\n        %(barbs_doc)s\n        ')
        
        # Assigning a Call to a Attribute (line 917):
        
        # Assigning a Call to a Attribute (line 917):
        
        # Call to pop(...): (line 917)
        # Processing the call arguments (line 917)
        unicode_122943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 917, 29), 'unicode', u'pivot')
        unicode_122944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 917, 38), 'unicode', u'tip')
        # Processing the call keyword arguments (line 917)
        kwargs_122945 = {}
        # Getting the type of 'kw' (line 917)
        kw_122941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 22), 'kw', False)
        # Obtaining the member 'pop' of a type (line 917)
        pop_122942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 22), kw_122941, 'pop')
        # Calling pop(args, kwargs) (line 917)
        pop_call_result_122946 = invoke(stypy.reporting.localization.Localization(__file__, 917, 22), pop_122942, *[unicode_122943, unicode_122944], **kwargs_122945)
        
        # Getting the type of 'self' (line 917)
        self_122947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 8), 'self')
        # Setting the type of the member '_pivot' of a type (line 917)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 8), self_122947, '_pivot', pop_call_result_122946)
        
        # Assigning a Call to a Attribute (line 918):
        
        # Assigning a Call to a Attribute (line 918):
        
        # Call to pop(...): (line 918)
        # Processing the call arguments (line 918)
        unicode_122950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 30), 'unicode', u'length')
        int_122951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 40), 'int')
        # Processing the call keyword arguments (line 918)
        kwargs_122952 = {}
        # Getting the type of 'kw' (line 918)
        kw_122948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 23), 'kw', False)
        # Obtaining the member 'pop' of a type (line 918)
        pop_122949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 23), kw_122948, 'pop')
        # Calling pop(args, kwargs) (line 918)
        pop_call_result_122953 = invoke(stypy.reporting.localization.Localization(__file__, 918, 23), pop_122949, *[unicode_122950, int_122951], **kwargs_122952)
        
        # Getting the type of 'self' (line 918)
        self_122954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 8), 'self')
        # Setting the type of the member '_length' of a type (line 918)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 8), self_122954, '_length', pop_call_result_122953)
        
        # Assigning a Call to a Name (line 919):
        
        # Assigning a Call to a Name (line 919):
        
        # Call to pop(...): (line 919)
        # Processing the call arguments (line 919)
        unicode_122957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, 27), 'unicode', u'barbcolor')
        # Getting the type of 'None' (line 919)
        None_122958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 40), 'None', False)
        # Processing the call keyword arguments (line 919)
        kwargs_122959 = {}
        # Getting the type of 'kw' (line 919)
        kw_122955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 20), 'kw', False)
        # Obtaining the member 'pop' of a type (line 919)
        pop_122956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 20), kw_122955, 'pop')
        # Calling pop(args, kwargs) (line 919)
        pop_call_result_122960 = invoke(stypy.reporting.localization.Localization(__file__, 919, 20), pop_122956, *[unicode_122957, None_122958], **kwargs_122959)
        
        # Assigning a type to the variable 'barbcolor' (line 919)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 8), 'barbcolor', pop_call_result_122960)
        
        # Assigning a Call to a Name (line 920):
        
        # Assigning a Call to a Name (line 920):
        
        # Call to pop(...): (line 920)
        # Processing the call arguments (line 920)
        unicode_122963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 27), 'unicode', u'flagcolor')
        # Getting the type of 'None' (line 920)
        None_122964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 40), 'None', False)
        # Processing the call keyword arguments (line 920)
        kwargs_122965 = {}
        # Getting the type of 'kw' (line 920)
        kw_122961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 20), 'kw', False)
        # Obtaining the member 'pop' of a type (line 920)
        pop_122962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 20), kw_122961, 'pop')
        # Calling pop(args, kwargs) (line 920)
        pop_call_result_122966 = invoke(stypy.reporting.localization.Localization(__file__, 920, 20), pop_122962, *[unicode_122963, None_122964], **kwargs_122965)
        
        # Assigning a type to the variable 'flagcolor' (line 920)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 8), 'flagcolor', pop_call_result_122966)
        
        # Assigning a Call to a Attribute (line 921):
        
        # Assigning a Call to a Attribute (line 921):
        
        # Call to pop(...): (line 921)
        # Processing the call arguments (line 921)
        unicode_122969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 28), 'unicode', u'sizes')
        
        # Call to dict(...): (line 921)
        # Processing the call keyword arguments (line 921)
        kwargs_122971 = {}
        # Getting the type of 'dict' (line 921)
        dict_122970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 37), 'dict', False)
        # Calling dict(args, kwargs) (line 921)
        dict_call_result_122972 = invoke(stypy.reporting.localization.Localization(__file__, 921, 37), dict_122970, *[], **kwargs_122971)
        
        # Processing the call keyword arguments (line 921)
        kwargs_122973 = {}
        # Getting the type of 'kw' (line 921)
        kw_122967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 21), 'kw', False)
        # Obtaining the member 'pop' of a type (line 921)
        pop_122968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 21), kw_122967, 'pop')
        # Calling pop(args, kwargs) (line 921)
        pop_call_result_122974 = invoke(stypy.reporting.localization.Localization(__file__, 921, 21), pop_122968, *[unicode_122969, dict_call_result_122972], **kwargs_122973)
        
        # Getting the type of 'self' (line 921)
        self_122975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 8), 'self')
        # Setting the type of the member 'sizes' of a type (line 921)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 8), self_122975, 'sizes', pop_call_result_122974)
        
        # Assigning a Call to a Attribute (line 922):
        
        # Assigning a Call to a Attribute (line 922):
        
        # Call to pop(...): (line 922)
        # Processing the call arguments (line 922)
        unicode_122978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 33), 'unicode', u'fill_empty')
        # Getting the type of 'False' (line 922)
        False_122979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 47), 'False', False)
        # Processing the call keyword arguments (line 922)
        kwargs_122980 = {}
        # Getting the type of 'kw' (line 922)
        kw_122976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 26), 'kw', False)
        # Obtaining the member 'pop' of a type (line 922)
        pop_122977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 26), kw_122976, 'pop')
        # Calling pop(args, kwargs) (line 922)
        pop_call_result_122981 = invoke(stypy.reporting.localization.Localization(__file__, 922, 26), pop_122977, *[unicode_122978, False_122979], **kwargs_122980)
        
        # Getting the type of 'self' (line 922)
        self_122982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'self')
        # Setting the type of the member 'fill_empty' of a type (line 922)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 8), self_122982, 'fill_empty', pop_call_result_122981)
        
        # Assigning a Call to a Attribute (line 923):
        
        # Assigning a Call to a Attribute (line 923):
        
        # Call to pop(...): (line 923)
        # Processing the call arguments (line 923)
        unicode_122985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 38), 'unicode', u'barb_increments')
        
        # Call to dict(...): (line 923)
        # Processing the call keyword arguments (line 923)
        kwargs_122987 = {}
        # Getting the type of 'dict' (line 923)
        dict_122986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 57), 'dict', False)
        # Calling dict(args, kwargs) (line 923)
        dict_call_result_122988 = invoke(stypy.reporting.localization.Localization(__file__, 923, 57), dict_122986, *[], **kwargs_122987)
        
        # Processing the call keyword arguments (line 923)
        kwargs_122989 = {}
        # Getting the type of 'kw' (line 923)
        kw_122983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 31), 'kw', False)
        # Obtaining the member 'pop' of a type (line 923)
        pop_122984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 31), kw_122983, 'pop')
        # Calling pop(args, kwargs) (line 923)
        pop_call_result_122990 = invoke(stypy.reporting.localization.Localization(__file__, 923, 31), pop_122984, *[unicode_122985, dict_call_result_122988], **kwargs_122989)
        
        # Getting the type of 'self' (line 923)
        self_122991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 8), 'self')
        # Setting the type of the member 'barb_increments' of a type (line 923)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 8), self_122991, 'barb_increments', pop_call_result_122990)
        
        # Assigning a Call to a Attribute (line 924):
        
        # Assigning a Call to a Attribute (line 924):
        
        # Call to pop(...): (line 924)
        # Processing the call arguments (line 924)
        unicode_122994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 31), 'unicode', u'rounding')
        # Getting the type of 'True' (line 924)
        True_122995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 43), 'True', False)
        # Processing the call keyword arguments (line 924)
        kwargs_122996 = {}
        # Getting the type of 'kw' (line 924)
        kw_122992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 24), 'kw', False)
        # Obtaining the member 'pop' of a type (line 924)
        pop_122993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 24), kw_122992, 'pop')
        # Calling pop(args, kwargs) (line 924)
        pop_call_result_122997 = invoke(stypy.reporting.localization.Localization(__file__, 924, 24), pop_122993, *[unicode_122994, True_122995], **kwargs_122996)
        
        # Getting the type of 'self' (line 924)
        self_122998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'self')
        # Setting the type of the member 'rounding' of a type (line 924)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 8), self_122998, 'rounding', pop_call_result_122997)
        
        # Assigning a Call to a Attribute (line 925):
        
        # Assigning a Call to a Attribute (line 925):
        
        # Call to pop(...): (line 925)
        # Processing the call arguments (line 925)
        unicode_123001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 27), 'unicode', u'flip_barb')
        # Getting the type of 'False' (line 925)
        False_123002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 40), 'False', False)
        # Processing the call keyword arguments (line 925)
        kwargs_123003 = {}
        # Getting the type of 'kw' (line 925)
        kw_122999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 20), 'kw', False)
        # Obtaining the member 'pop' of a type (line 925)
        pop_123000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 20), kw_122999, 'pop')
        # Calling pop(args, kwargs) (line 925)
        pop_call_result_123004 = invoke(stypy.reporting.localization.Localization(__file__, 925, 20), pop_123000, *[unicode_123001, False_123002], **kwargs_123003)
        
        # Getting the type of 'self' (line 925)
        self_123005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 8), 'self')
        # Setting the type of the member 'flip' of a type (line 925)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 8), self_123005, 'flip', pop_call_result_123004)
        
        # Assigning a Call to a Name (line 926):
        
        # Assigning a Call to a Name (line 926):
        
        # Call to pop(...): (line 926)
        # Processing the call arguments (line 926)
        unicode_123008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 27), 'unicode', u'transform')
        # Getting the type of 'ax' (line 926)
        ax_123009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 40), 'ax', False)
        # Obtaining the member 'transData' of a type (line 926)
        transData_123010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 40), ax_123009, 'transData')
        # Processing the call keyword arguments (line 926)
        kwargs_123011 = {}
        # Getting the type of 'kw' (line 926)
        kw_123006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 20), 'kw', False)
        # Obtaining the member 'pop' of a type (line 926)
        pop_123007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 20), kw_123006, 'pop')
        # Calling pop(args, kwargs) (line 926)
        pop_call_result_123012 = invoke(stypy.reporting.localization.Localization(__file__, 926, 20), pop_123007, *[unicode_123008, transData_123010], **kwargs_123011)
        
        # Assigning a type to the variable 'transform' (line 926)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 8), 'transform', pop_call_result_123012)
        
        
        # Getting the type of 'None' (line 933)
        None_123013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 11), 'None')
        
        # Obtaining an instance of the builtin type 'tuple' (line 933)
        tuple_123014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 933)
        # Adding element type (line 933)
        # Getting the type of 'barbcolor' (line 933)
        barbcolor_123015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 20), 'barbcolor')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 933, 20), tuple_123014, barbcolor_123015)
        # Adding element type (line 933)
        # Getting the type of 'flagcolor' (line 933)
        flagcolor_123016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 31), 'flagcolor')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 933, 20), tuple_123014, flagcolor_123016)
        
        # Applying the binary operator 'in' (line 933)
        result_contains_123017 = python_operator(stypy.reporting.localization.Localization(__file__, 933, 11), 'in', None_123013, tuple_123014)
        
        # Testing the type of an if condition (line 933)
        if_condition_123018 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 933, 8), result_contains_123017)
        # Assigning a type to the variable 'if_condition_123018' (line 933)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 8), 'if_condition_123018', if_condition_123018)
        # SSA begins for if statement (line 933)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Subscript (line 934):
        
        # Assigning a Str to a Subscript (line 934):
        unicode_123019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 31), 'unicode', u'face')
        # Getting the type of 'kw' (line 934)
        kw_123020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 12), 'kw')
        unicode_123021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 15), 'unicode', u'edgecolors')
        # Storing an element on a container (line 934)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 934, 12), kw_123020, (unicode_123021, unicode_123019))
        
        # Getting the type of 'flagcolor' (line 935)
        flagcolor_123022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 15), 'flagcolor')
        # Testing the type of an if condition (line 935)
        if_condition_123023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 935, 12), flagcolor_123022)
        # Assigning a type to the variable 'if_condition_123023' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 12), 'if_condition_123023', if_condition_123023)
        # SSA begins for if statement (line 935)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 936):
        
        # Assigning a Name to a Subscript (line 936):
        # Getting the type of 'flagcolor' (line 936)
        flagcolor_123024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 35), 'flagcolor')
        # Getting the type of 'kw' (line 936)
        kw_123025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 16), 'kw')
        unicode_123026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 19), 'unicode', u'facecolors')
        # Storing an element on a container (line 936)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 936, 16), kw_123025, (unicode_123026, flagcolor_123024))
        # SSA branch for the else part of an if statement (line 935)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'barbcolor' (line 937)
        barbcolor_123027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 17), 'barbcolor')
        # Testing the type of an if condition (line 937)
        if_condition_123028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 937, 17), barbcolor_123027)
        # Assigning a type to the variable 'if_condition_123028' (line 937)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 17), 'if_condition_123028', if_condition_123028)
        # SSA begins for if statement (line 937)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 938):
        
        # Assigning a Name to a Subscript (line 938):
        # Getting the type of 'barbcolor' (line 938)
        barbcolor_123029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 35), 'barbcolor')
        # Getting the type of 'kw' (line 938)
        kw_123030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 16), 'kw')
        unicode_123031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 19), 'unicode', u'facecolors')
        # Storing an element on a container (line 938)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 938, 16), kw_123030, (unicode_123031, barbcolor_123029))
        # SSA branch for the else part of an if statement (line 937)
        module_type_store.open_ssa_branch('else')
        
        # Call to setdefault(...): (line 941)
        # Processing the call arguments (line 941)
        unicode_123034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 30), 'unicode', u'facecolors')
        unicode_123035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 44), 'unicode', u'k')
        # Processing the call keyword arguments (line 941)
        kwargs_123036 = {}
        # Getting the type of 'kw' (line 941)
        kw_123032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 16), 'kw', False)
        # Obtaining the member 'setdefault' of a type (line 941)
        setdefault_123033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 941, 16), kw_123032, 'setdefault')
        # Calling setdefault(args, kwargs) (line 941)
        setdefault_call_result_123037 = invoke(stypy.reporting.localization.Localization(__file__, 941, 16), setdefault_123033, *[unicode_123034, unicode_123035], **kwargs_123036)
        
        # SSA join for if statement (line 937)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 935)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 933)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Subscript (line 943):
        
        # Assigning a Name to a Subscript (line 943):
        # Getting the type of 'barbcolor' (line 943)
        barbcolor_123038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 31), 'barbcolor')
        # Getting the type of 'kw' (line 943)
        kw_123039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 12), 'kw')
        unicode_123040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 15), 'unicode', u'edgecolors')
        # Storing an element on a container (line 943)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 943, 12), kw_123039, (unicode_123040, barbcolor_123038))
        
        # Assigning a Name to a Subscript (line 944):
        
        # Assigning a Name to a Subscript (line 944):
        # Getting the type of 'flagcolor' (line 944)
        flagcolor_123041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 31), 'flagcolor')
        # Getting the type of 'kw' (line 944)
        kw_123042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 12), 'kw')
        unicode_123043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 15), 'unicode', u'facecolors')
        # Storing an element on a container (line 944)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 944, 12), kw_123042, (unicode_123043, flagcolor_123041))
        # SSA join for if statement (line 933)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        unicode_123044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 11), 'unicode', u'linewidth')
        # Getting the type of 'kw' (line 948)
        kw_123045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 30), 'kw')
        # Applying the binary operator 'notin' (line 948)
        result_contains_123046 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 11), 'notin', unicode_123044, kw_123045)
        
        
        unicode_123047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 37), 'unicode', u'lw')
        # Getting the type of 'kw' (line 948)
        kw_123048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 49), 'kw')
        # Applying the binary operator 'notin' (line 948)
        result_contains_123049 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 37), 'notin', unicode_123047, kw_123048)
        
        # Applying the binary operator 'and' (line 948)
        result_and_keyword_123050 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 11), 'and', result_contains_123046, result_contains_123049)
        
        # Testing the type of an if condition (line 948)
        if_condition_123051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 948, 8), result_and_keyword_123050)
        # Assigning a type to the variable 'if_condition_123051' (line 948)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 8), 'if_condition_123051', if_condition_123051)
        # SSA begins for if statement (line 948)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 949):
        
        # Assigning a Num to a Subscript (line 949):
        int_123052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, 30), 'int')
        # Getting the type of 'kw' (line 949)
        kw_123053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 12), 'kw')
        unicode_123054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, 15), 'unicode', u'linewidth')
        # Storing an element on a container (line 949)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 949, 12), kw_123053, (unicode_123054, int_123052))
        # SSA join for if statement (line 948)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 952):
        
        # Assigning a Call to a Name:
        
        # Call to _parse_args(...): (line 952)
        # Getting the type of 'args' (line 952)
        args_123056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 37), 'args', False)
        # Processing the call keyword arguments (line 952)
        kwargs_123057 = {}
        # Getting the type of '_parse_args' (line 952)
        _parse_args_123055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 24), '_parse_args', False)
        # Calling _parse_args(args, kwargs) (line 952)
        _parse_args_call_result_123058 = invoke(stypy.reporting.localization.Localization(__file__, 952, 24), _parse_args_123055, *[args_123056], **kwargs_123057)
        
        # Assigning a type to the variable 'call_assignment_120692' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120692', _parse_args_call_result_123058)
        
        # Assigning a Call to a Name (line 952):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123062 = {}
        # Getting the type of 'call_assignment_120692' (line 952)
        call_assignment_120692_123059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120692', False)
        # Obtaining the member '__getitem__' of a type (line 952)
        getitem___123060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 8), call_assignment_120692_123059, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123063 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123060, *[int_123061], **kwargs_123062)
        
        # Assigning a type to the variable 'call_assignment_120693' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120693', getitem___call_result_123063)
        
        # Assigning a Name to a Name (line 952):
        # Getting the type of 'call_assignment_120693' (line 952)
        call_assignment_120693_123064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120693')
        # Assigning a type to the variable 'x' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'x', call_assignment_120693_123064)
        
        # Assigning a Call to a Name (line 952):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123068 = {}
        # Getting the type of 'call_assignment_120692' (line 952)
        call_assignment_120692_123065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120692', False)
        # Obtaining the member '__getitem__' of a type (line 952)
        getitem___123066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 8), call_assignment_120692_123065, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123069 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123066, *[int_123067], **kwargs_123068)
        
        # Assigning a type to the variable 'call_assignment_120694' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120694', getitem___call_result_123069)
        
        # Assigning a Name to a Name (line 952):
        # Getting the type of 'call_assignment_120694' (line 952)
        call_assignment_120694_123070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120694')
        # Assigning a type to the variable 'y' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 11), 'y', call_assignment_120694_123070)
        
        # Assigning a Call to a Name (line 952):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123074 = {}
        # Getting the type of 'call_assignment_120692' (line 952)
        call_assignment_120692_123071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120692', False)
        # Obtaining the member '__getitem__' of a type (line 952)
        getitem___123072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 8), call_assignment_120692_123071, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123075 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123072, *[int_123073], **kwargs_123074)
        
        # Assigning a type to the variable 'call_assignment_120695' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120695', getitem___call_result_123075)
        
        # Assigning a Name to a Name (line 952):
        # Getting the type of 'call_assignment_120695' (line 952)
        call_assignment_120695_123076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120695')
        # Assigning a type to the variable 'u' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 14), 'u', call_assignment_120695_123076)
        
        # Assigning a Call to a Name (line 952):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123080 = {}
        # Getting the type of 'call_assignment_120692' (line 952)
        call_assignment_120692_123077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120692', False)
        # Obtaining the member '__getitem__' of a type (line 952)
        getitem___123078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 8), call_assignment_120692_123077, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123081 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123078, *[int_123079], **kwargs_123080)
        
        # Assigning a type to the variable 'call_assignment_120696' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120696', getitem___call_result_123081)
        
        # Assigning a Name to a Name (line 952):
        # Getting the type of 'call_assignment_120696' (line 952)
        call_assignment_120696_123082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120696')
        # Assigning a type to the variable 'v' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 17), 'v', call_assignment_120696_123082)
        
        # Assigning a Call to a Name (line 952):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123086 = {}
        # Getting the type of 'call_assignment_120692' (line 952)
        call_assignment_120692_123083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120692', False)
        # Obtaining the member '__getitem__' of a type (line 952)
        getitem___123084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 8), call_assignment_120692_123083, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123087 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123084, *[int_123085], **kwargs_123086)
        
        # Assigning a type to the variable 'call_assignment_120697' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120697', getitem___call_result_123087)
        
        # Assigning a Name to a Name (line 952):
        # Getting the type of 'call_assignment_120697' (line 952)
        call_assignment_120697_123088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'call_assignment_120697')
        # Assigning a type to the variable 'c' (line 952)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 20), 'c', call_assignment_120697_123088)
        
        # Assigning a Name to a Attribute (line 953):
        
        # Assigning a Name to a Attribute (line 953):
        # Getting the type of 'x' (line 953)
        x_123089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 17), 'x')
        # Getting the type of 'self' (line 953)
        self_123090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 8), 'self')
        # Setting the type of the member 'x' of a type (line 953)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 953, 8), self_123090, 'x', x_123089)
        
        # Assigning a Name to a Attribute (line 954):
        
        # Assigning a Name to a Attribute (line 954):
        # Getting the type of 'y' (line 954)
        y_123091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 17), 'y')
        # Getting the type of 'self' (line 954)
        self_123092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 8), 'self')
        # Setting the type of the member 'y' of a type (line 954)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 954, 8), self_123092, 'y', y_123091)
        
        # Assigning a Call to a Name (line 955):
        
        # Assigning a Call to a Name (line 955):
        
        # Call to hstack(...): (line 955)
        # Processing the call arguments (line 955)
        
        # Obtaining an instance of the builtin type 'tuple' (line 955)
        tuple_123095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 955)
        # Adding element type (line 955)
        
        # Obtaining the type of the subscript
        slice_123096 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 955, 24), None, None, None)
        # Getting the type of 'np' (line 955)
        np_123097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 29), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 955)
        newaxis_123098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 29), np_123097, 'newaxis')
        # Getting the type of 'x' (line 955)
        x_123099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 24), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 955)
        getitem___123100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 24), x_123099, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 955)
        subscript_call_result_123101 = invoke(stypy.reporting.localization.Localization(__file__, 955, 24), getitem___123100, (slice_123096, newaxis_123098))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 955, 24), tuple_123095, subscript_call_result_123101)
        # Adding element type (line 955)
        
        # Obtaining the type of the subscript
        slice_123102 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 955, 42), None, None, None)
        # Getting the type of 'np' (line 955)
        np_123103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 47), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 955)
        newaxis_123104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 47), np_123103, 'newaxis')
        # Getting the type of 'y' (line 955)
        y_123105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 42), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 955)
        getitem___123106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 42), y_123105, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 955)
        subscript_call_result_123107 = invoke(stypy.reporting.localization.Localization(__file__, 955, 42), getitem___123106, (slice_123102, newaxis_123104))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 955, 24), tuple_123095, subscript_call_result_123107)
        
        # Processing the call keyword arguments (line 955)
        kwargs_123108 = {}
        # Getting the type of 'np' (line 955)
        np_123093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 13), 'np', False)
        # Obtaining the member 'hstack' of a type (line 955)
        hstack_123094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 13), np_123093, 'hstack')
        # Calling hstack(args, kwargs) (line 955)
        hstack_call_result_123109 = invoke(stypy.reporting.localization.Localization(__file__, 955, 13), hstack_123094, *[tuple_123095], **kwargs_123108)
        
        # Assigning a type to the variable 'xy' (line 955)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 8), 'xy', hstack_call_result_123109)
        
        # Assigning a BinOp to a Name (line 958):
        
        # Assigning a BinOp to a Name (line 958):
        # Getting the type of 'self' (line 958)
        self_123110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 20), 'self')
        # Obtaining the member '_length' of a type (line 958)
        _length_123111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 958, 20), self_123110, '_length')
        int_123112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 36), 'int')
        # Applying the binary operator '**' (line 958)
        result_pow_123113 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 20), '**', _length_123111, int_123112)
        
        int_123114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 40), 'int')
        # Applying the binary operator 'div' (line 958)
        result_div_123115 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 20), 'div', result_pow_123113, int_123114)
        
        # Assigning a type to the variable 'barb_size' (line 958)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 958, 8), 'barb_size', result_div_123115)
        
        # Call to __init__(...): (line 959)
        # Processing the call arguments (line 959)
        # Getting the type of 'self' (line 959)
        self_123119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 45), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 959)
        list_123120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 959)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 959)
        tuple_123121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 959)
        # Adding element type (line 959)
        # Getting the type of 'barb_size' (line 959)
        barb_size_123122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 56), 'barb_size', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 959, 56), tuple_123121, barb_size_123122)
        
        # Processing the call keyword arguments (line 959)
        # Getting the type of 'xy' (line 960)
        xy_123123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 53), 'xy', False)
        keyword_123124 = xy_123123
        # Getting the type of 'transform' (line 961)
        transform_123125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 57), 'transform', False)
        keyword_123126 = transform_123125
        # Getting the type of 'kw' (line 961)
        kw_123127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 70), 'kw', False)
        kwargs_123128 = {'kw_123127': kw_123127, 'transOffset': keyword_123126, 'offsets': keyword_123124}
        # Getting the type of 'mcollections' (line 959)
        mcollections_123116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 8), 'mcollections', False)
        # Obtaining the member 'PolyCollection' of a type (line 959)
        PolyCollection_123117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 959, 8), mcollections_123116, 'PolyCollection')
        # Obtaining the member '__init__' of a type (line 959)
        init___123118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 959, 8), PolyCollection_123117, '__init__')
        # Calling __init__(args, kwargs) (line 959)
        init___call_result_123129 = invoke(stypy.reporting.localization.Localization(__file__, 959, 8), init___123118, *[self_123119, list_123120, tuple_123121], **kwargs_123128)
        
        
        # Call to set_transform(...): (line 962)
        # Processing the call arguments (line 962)
        
        # Call to IdentityTransform(...): (line 962)
        # Processing the call keyword arguments (line 962)
        kwargs_123134 = {}
        # Getting the type of 'transforms' (line 962)
        transforms_123132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 27), 'transforms', False)
        # Obtaining the member 'IdentityTransform' of a type (line 962)
        IdentityTransform_123133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 27), transforms_123132, 'IdentityTransform')
        # Calling IdentityTransform(args, kwargs) (line 962)
        IdentityTransform_call_result_123135 = invoke(stypy.reporting.localization.Localization(__file__, 962, 27), IdentityTransform_123133, *[], **kwargs_123134)
        
        # Processing the call keyword arguments (line 962)
        kwargs_123136 = {}
        # Getting the type of 'self' (line 962)
        self_123130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 8), 'self', False)
        # Obtaining the member 'set_transform' of a type (line 962)
        set_transform_123131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 8), self_123130, 'set_transform')
        # Calling set_transform(args, kwargs) (line 962)
        set_transform_call_result_123137 = invoke(stypy.reporting.localization.Localization(__file__, 962, 8), set_transform_123131, *[IdentityTransform_call_result_123135], **kwargs_123136)
        
        
        # Call to set_UVC(...): (line 964)
        # Processing the call arguments (line 964)
        # Getting the type of 'u' (line 964)
        u_123140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 21), 'u', False)
        # Getting the type of 'v' (line 964)
        v_123141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 24), 'v', False)
        # Getting the type of 'c' (line 964)
        c_123142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 27), 'c', False)
        # Processing the call keyword arguments (line 964)
        kwargs_123143 = {}
        # Getting the type of 'self' (line 964)
        self_123138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 8), 'self', False)
        # Obtaining the member 'set_UVC' of a type (line 964)
        set_UVC_123139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 964, 8), self_123138, 'set_UVC')
        # Calling set_UVC(args, kwargs) (line 964)
        set_UVC_call_result_123144 = invoke(stypy.reporting.localization.Localization(__file__, 964, 8), set_UVC_123139, *[u_123140, v_123141, c_123142], **kwargs_123143)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _find_tails(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 966)
        True_123145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 40), 'True')
        int_123146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 51), 'int')
        int_123147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 59), 'int')
        int_123148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 68), 'int')
        defaults = [True_123145, int_123146, int_123147, int_123148]
        # Create a new context for function '_find_tails'
        module_type_store = module_type_store.open_function_context('_find_tails', 966, 4, False)
        # Assigning a type to the variable 'self' (line 967)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Barbs._find_tails.__dict__.__setitem__('stypy_localization', localization)
        Barbs._find_tails.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Barbs._find_tails.__dict__.__setitem__('stypy_type_store', module_type_store)
        Barbs._find_tails.__dict__.__setitem__('stypy_function_name', 'Barbs._find_tails')
        Barbs._find_tails.__dict__.__setitem__('stypy_param_names_list', ['mag', 'rounding', 'half', 'full', 'flag'])
        Barbs._find_tails.__dict__.__setitem__('stypy_varargs_param_name', None)
        Barbs._find_tails.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Barbs._find_tails.__dict__.__setitem__('stypy_call_defaults', defaults)
        Barbs._find_tails.__dict__.__setitem__('stypy_call_varargs', varargs)
        Barbs._find_tails.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Barbs._find_tails.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Barbs._find_tails', ['mag', 'rounding', 'half', 'full', 'flag'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_find_tails', localization, ['mag', 'rounding', 'half', 'full', 'flag'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_find_tails(...)' code ##################

        unicode_123149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, (-1)), 'unicode', u'\n        Find how many of each of the tail pieces is necessary.  Flag\n        specifies the increment for a flag, barb for a full barb, and half for\n        half a barb. Mag should be the magnitude of a vector (i.e., >= 0).\n\n        This returns a tuple of:\n\n            (*number of flags*, *number of barbs*, *half_flag*, *empty_flag*)\n\n        *half_flag* is a boolean whether half of a barb is needed,\n        since there should only ever be one half on a given\n        barb. *empty_flag* flag is an array of flags to easily tell if\n        a barb is empty (too low to plot any barbs/flags.\n        ')
        
        # Getting the type of 'rounding' (line 984)
        rounding_123150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 11), 'rounding')
        # Testing the type of an if condition (line 984)
        if_condition_123151 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 984, 8), rounding_123150)
        # Assigning a type to the variable 'if_condition_123151' (line 984)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 8), 'if_condition_123151', if_condition_123151)
        # SSA begins for if statement (line 984)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 985):
        
        # Assigning a BinOp to a Name (line 985):
        # Getting the type of 'half' (line 985)
        half_123152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 18), 'half')
        
        # Call to astype(...): (line 985)
        # Processing the call arguments (line 985)
        # Getting the type of 'int' (line 985)
        int_123159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 51), 'int', False)
        # Processing the call keyword arguments (line 985)
        kwargs_123160 = {}
        # Getting the type of 'mag' (line 985)
        mag_123153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 26), 'mag', False)
        # Getting the type of 'half' (line 985)
        half_123154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 32), 'half', False)
        # Applying the binary operator 'div' (line 985)
        result_div_123155 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 26), 'div', mag_123153, half_123154)
        
        float_123156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 985, 39), 'float')
        # Applying the binary operator '+' (line 985)
        result_add_123157 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 26), '+', result_div_123155, float_123156)
        
        # Obtaining the member 'astype' of a type (line 985)
        astype_123158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 985, 26), result_add_123157, 'astype')
        # Calling astype(args, kwargs) (line 985)
        astype_call_result_123161 = invoke(stypy.reporting.localization.Localization(__file__, 985, 26), astype_123158, *[int_123159], **kwargs_123160)
        
        # Applying the binary operator '*' (line 985)
        result_mul_123162 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 18), '*', half_123152, astype_call_result_123161)
        
        # Assigning a type to the variable 'mag' (line 985)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 12), 'mag', result_mul_123162)
        # SSA join for if statement (line 984)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 987):
        
        # Assigning a Call to a Name (line 987):
        
        # Call to astype(...): (line 987)
        # Processing the call arguments (line 987)
        # Getting the type of 'int' (line 987)
        int_123171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 48), 'int', False)
        # Processing the call keyword arguments (line 987)
        kwargs_123172 = {}
        
        # Call to floor(...): (line 987)
        # Processing the call arguments (line 987)
        # Getting the type of 'mag' (line 987)
        mag_123165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 29), 'mag', False)
        # Getting the type of 'flag' (line 987)
        flag_123166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 35), 'flag', False)
        # Applying the binary operator 'div' (line 987)
        result_div_123167 = python_operator(stypy.reporting.localization.Localization(__file__, 987, 29), 'div', mag_123165, flag_123166)
        
        # Processing the call keyword arguments (line 987)
        kwargs_123168 = {}
        # Getting the type of 'np' (line 987)
        np_123163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 20), 'np', False)
        # Obtaining the member 'floor' of a type (line 987)
        floor_123164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 987, 20), np_123163, 'floor')
        # Calling floor(args, kwargs) (line 987)
        floor_call_result_123169 = invoke(stypy.reporting.localization.Localization(__file__, 987, 20), floor_123164, *[result_div_123167], **kwargs_123168)
        
        # Obtaining the member 'astype' of a type (line 987)
        astype_123170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 987, 20), floor_call_result_123169, 'astype')
        # Calling astype(args, kwargs) (line 987)
        astype_call_result_123173 = invoke(stypy.reporting.localization.Localization(__file__, 987, 20), astype_123170, *[int_123171], **kwargs_123172)
        
        # Assigning a type to the variable 'num_flags' (line 987)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 987, 8), 'num_flags', astype_call_result_123173)
        
        # Assigning a Call to a Name (line 988):
        
        # Assigning a Call to a Name (line 988):
        
        # Call to mod(...): (line 988)
        # Processing the call arguments (line 988)
        # Getting the type of 'mag' (line 988)
        mag_123176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 21), 'mag', False)
        # Getting the type of 'flag' (line 988)
        flag_123177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 26), 'flag', False)
        # Processing the call keyword arguments (line 988)
        kwargs_123178 = {}
        # Getting the type of 'np' (line 988)
        np_123174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 14), 'np', False)
        # Obtaining the member 'mod' of a type (line 988)
        mod_123175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 988, 14), np_123174, 'mod')
        # Calling mod(args, kwargs) (line 988)
        mod_call_result_123179 = invoke(stypy.reporting.localization.Localization(__file__, 988, 14), mod_123175, *[mag_123176, flag_123177], **kwargs_123178)
        
        # Assigning a type to the variable 'mag' (line 988)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 8), 'mag', mod_call_result_123179)
        
        # Assigning a Call to a Name (line 990):
        
        # Assigning a Call to a Name (line 990):
        
        # Call to astype(...): (line 990)
        # Processing the call arguments (line 990)
        # Getting the type of 'int' (line 990)
        int_123188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 47), 'int', False)
        # Processing the call keyword arguments (line 990)
        kwargs_123189 = {}
        
        # Call to floor(...): (line 990)
        # Processing the call arguments (line 990)
        # Getting the type of 'mag' (line 990)
        mag_123182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 28), 'mag', False)
        # Getting the type of 'full' (line 990)
        full_123183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 34), 'full', False)
        # Applying the binary operator 'div' (line 990)
        result_div_123184 = python_operator(stypy.reporting.localization.Localization(__file__, 990, 28), 'div', mag_123182, full_123183)
        
        # Processing the call keyword arguments (line 990)
        kwargs_123185 = {}
        # Getting the type of 'np' (line 990)
        np_123180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 19), 'np', False)
        # Obtaining the member 'floor' of a type (line 990)
        floor_123181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 990, 19), np_123180, 'floor')
        # Calling floor(args, kwargs) (line 990)
        floor_call_result_123186 = invoke(stypy.reporting.localization.Localization(__file__, 990, 19), floor_123181, *[result_div_123184], **kwargs_123185)
        
        # Obtaining the member 'astype' of a type (line 990)
        astype_123187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 990, 19), floor_call_result_123186, 'astype')
        # Calling astype(args, kwargs) (line 990)
        astype_call_result_123190 = invoke(stypy.reporting.localization.Localization(__file__, 990, 19), astype_123187, *[int_123188], **kwargs_123189)
        
        # Assigning a type to the variable 'num_barb' (line 990)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 990, 8), 'num_barb', astype_call_result_123190)
        
        # Assigning a Call to a Name (line 991):
        
        # Assigning a Call to a Name (line 991):
        
        # Call to mod(...): (line 991)
        # Processing the call arguments (line 991)
        # Getting the type of 'mag' (line 991)
        mag_123193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 21), 'mag', False)
        # Getting the type of 'full' (line 991)
        full_123194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 26), 'full', False)
        # Processing the call keyword arguments (line 991)
        kwargs_123195 = {}
        # Getting the type of 'np' (line 991)
        np_123191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 14), 'np', False)
        # Obtaining the member 'mod' of a type (line 991)
        mod_123192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 14), np_123191, 'mod')
        # Calling mod(args, kwargs) (line 991)
        mod_call_result_123196 = invoke(stypy.reporting.localization.Localization(__file__, 991, 14), mod_123192, *[mag_123193, full_123194], **kwargs_123195)
        
        # Assigning a type to the variable 'mag' (line 991)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 991, 8), 'mag', mod_call_result_123196)
        
        # Assigning a Compare to a Name (line 993):
        
        # Assigning a Compare to a Name (line 993):
        
        # Getting the type of 'mag' (line 993)
        mag_123197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 20), 'mag')
        # Getting the type of 'half' (line 993)
        half_123198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 27), 'half')
        # Applying the binary operator '>=' (line 993)
        result_ge_123199 = python_operator(stypy.reporting.localization.Localization(__file__, 993, 20), '>=', mag_123197, half_123198)
        
        # Assigning a type to the variable 'half_flag' (line 993)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'half_flag', result_ge_123199)
        
        # Assigning a UnaryOp to a Name (line 994):
        
        # Assigning a UnaryOp to a Name (line 994):
        
        # Getting the type of 'half_flag' (line 994)
        half_flag_123200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 23), 'half_flag')
        
        # Getting the type of 'num_flags' (line 994)
        num_flags_123201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 36), 'num_flags')
        int_123202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 48), 'int')
        # Applying the binary operator '>' (line 994)
        result_gt_123203 = python_operator(stypy.reporting.localization.Localization(__file__, 994, 36), '>', num_flags_123201, int_123202)
        
        # Applying the binary operator '|' (line 994)
        result_or__123204 = python_operator(stypy.reporting.localization.Localization(__file__, 994, 23), '|', half_flag_123200, result_gt_123203)
        
        
        # Getting the type of 'num_barb' (line 994)
        num_barb_123205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 54), 'num_barb')
        int_123206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 65), 'int')
        # Applying the binary operator '>' (line 994)
        result_gt_123207 = python_operator(stypy.reporting.localization.Localization(__file__, 994, 54), '>', num_barb_123205, int_123206)
        
        # Applying the binary operator '|' (line 994)
        result_or__123208 = python_operator(stypy.reporting.localization.Localization(__file__, 994, 51), '|', result_or__123204, result_gt_123207)
        
        # Applying the '~' unary operator (line 994)
        result_inv_123209 = python_operator(stypy.reporting.localization.Localization(__file__, 994, 21), '~', result_or__123208)
        
        # Assigning a type to the variable 'empty_flag' (line 994)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 994, 8), 'empty_flag', result_inv_123209)
        
        # Obtaining an instance of the builtin type 'tuple' (line 996)
        tuple_123210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 996)
        # Adding element type (line 996)
        # Getting the type of 'num_flags' (line 996)
        num_flags_123211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 15), 'num_flags')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 996, 15), tuple_123210, num_flags_123211)
        # Adding element type (line 996)
        # Getting the type of 'num_barb' (line 996)
        num_barb_123212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 26), 'num_barb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 996, 15), tuple_123210, num_barb_123212)
        # Adding element type (line 996)
        # Getting the type of 'half_flag' (line 996)
        half_flag_123213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 36), 'half_flag')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 996, 15), tuple_123210, half_flag_123213)
        # Adding element type (line 996)
        # Getting the type of 'empty_flag' (line 996)
        empty_flag_123214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 47), 'empty_flag')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 996, 15), tuple_123210, empty_flag_123214)
        
        # Assigning a type to the variable 'stypy_return_type' (line 996)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 996, 8), 'stypy_return_type', tuple_123210)
        
        # ################# End of '_find_tails(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_find_tails' in the type store
        # Getting the type of 'stypy_return_type' (line 966)
        stypy_return_type_123215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123215)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_find_tails'
        return stypy_return_type_123215


    @norecursion
    def _make_barbs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_make_barbs'
        module_type_store = module_type_store.open_function_context('_make_barbs', 998, 4, False)
        # Assigning a type to the variable 'self' (line 999)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 999, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Barbs._make_barbs.__dict__.__setitem__('stypy_localization', localization)
        Barbs._make_barbs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Barbs._make_barbs.__dict__.__setitem__('stypy_type_store', module_type_store)
        Barbs._make_barbs.__dict__.__setitem__('stypy_function_name', 'Barbs._make_barbs')
        Barbs._make_barbs.__dict__.__setitem__('stypy_param_names_list', ['u', 'v', 'nflags', 'nbarbs', 'half_barb', 'empty_flag', 'length', 'pivot', 'sizes', 'fill_empty', 'flip'])
        Barbs._make_barbs.__dict__.__setitem__('stypy_varargs_param_name', None)
        Barbs._make_barbs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Barbs._make_barbs.__dict__.__setitem__('stypy_call_defaults', defaults)
        Barbs._make_barbs.__dict__.__setitem__('stypy_call_varargs', varargs)
        Barbs._make_barbs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Barbs._make_barbs.__dict__.__setitem__('stypy_declared_arg_number', 12)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Barbs._make_barbs', ['u', 'v', 'nflags', 'nbarbs', 'half_barb', 'empty_flag', 'length', 'pivot', 'sizes', 'fill_empty', 'flip'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_make_barbs', localization, ['u', 'v', 'nflags', 'nbarbs', 'half_barb', 'empty_flag', 'length', 'pivot', 'sizes', 'fill_empty', 'flip'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_make_barbs(...)' code ##################

        unicode_123216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, (-1)), 'unicode', u"\n        This function actually creates the wind barbs.  *u* and *v*\n        are components of the vector in the *x* and *y* directions,\n        respectively.\n\n        *nflags*, *nbarbs*, and *half_barb*, empty_flag* are,\n        *respectively, the number of flags, number of barbs, flag for\n        *half a barb, and flag for empty barb, ostensibly obtained\n        *from :meth:`_find_tails`.\n\n        *length* is the length of the barb staff in points.\n\n        *pivot* specifies the point on the barb around which the\n        entire barb should be rotated.  Right now, valid options are\n        'tip' and 'middle'. Can also be a number, which shifts the start\n        of the barb that many points from the origin.\n\n        *sizes* is a dictionary of coefficients specifying the ratio\n        of a given feature to the length of the barb. These features\n        include:\n\n            - *spacing*: space between features (flags, full/half\n               barbs)\n\n            - *height*: distance from shaft of top of a flag or full\n               barb\n\n            - *width* - width of a flag, twice the width of a full barb\n\n            - *emptybarb* - radius of the circle used for low\n               magnitudes\n\n        *fill_empty* specifies whether the circle representing an\n        empty barb should be filled or not (this changes the drawing\n        of the polygon).\n\n        *flip* is a flag indicating whether the features should be flipped to\n        the other side of the barb (useful for winds in the southern\n        hemisphere).\n\n        This function returns list of arrays of vertices, defining a polygon\n        for each of the wind barbs.  These polygons have been rotated to\n        properly align with the vector direction.\n        ")
        
        # Assigning a BinOp to a Name (line 1047):
        
        # Assigning a BinOp to a Name (line 1047):
        # Getting the type of 'length' (line 1047)
        length_123217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 18), 'length')
        
        # Call to get(...): (line 1047)
        # Processing the call arguments (line 1047)
        unicode_123220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 37), 'unicode', u'spacing')
        float_123221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 48), 'float')
        # Processing the call keyword arguments (line 1047)
        kwargs_123222 = {}
        # Getting the type of 'sizes' (line 1047)
        sizes_123218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 27), 'sizes', False)
        # Obtaining the member 'get' of a type (line 1047)
        get_123219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1047, 27), sizes_123218, 'get')
        # Calling get(args, kwargs) (line 1047)
        get_call_result_123223 = invoke(stypy.reporting.localization.Localization(__file__, 1047, 27), get_123219, *[unicode_123220, float_123221], **kwargs_123222)
        
        # Applying the binary operator '*' (line 1047)
        result_mul_123224 = python_operator(stypy.reporting.localization.Localization(__file__, 1047, 18), '*', length_123217, get_call_result_123223)
        
        # Assigning a type to the variable 'spacing' (line 1047)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1047, 8), 'spacing', result_mul_123224)
        
        # Assigning a BinOp to a Name (line 1048):
        
        # Assigning a BinOp to a Name (line 1048):
        # Getting the type of 'length' (line 1048)
        length_123225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 22), 'length')
        
        # Call to get(...): (line 1048)
        # Processing the call arguments (line 1048)
        unicode_123228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1048, 41), 'unicode', u'height')
        float_123229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1048, 51), 'float')
        # Processing the call keyword arguments (line 1048)
        kwargs_123230 = {}
        # Getting the type of 'sizes' (line 1048)
        sizes_123226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 31), 'sizes', False)
        # Obtaining the member 'get' of a type (line 1048)
        get_123227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1048, 31), sizes_123226, 'get')
        # Calling get(args, kwargs) (line 1048)
        get_call_result_123231 = invoke(stypy.reporting.localization.Localization(__file__, 1048, 31), get_123227, *[unicode_123228, float_123229], **kwargs_123230)
        
        # Applying the binary operator '*' (line 1048)
        result_mul_123232 = python_operator(stypy.reporting.localization.Localization(__file__, 1048, 22), '*', length_123225, get_call_result_123231)
        
        # Assigning a type to the variable 'full_height' (line 1048)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1048, 8), 'full_height', result_mul_123232)
        
        # Assigning a BinOp to a Name (line 1049):
        
        # Assigning a BinOp to a Name (line 1049):
        # Getting the type of 'length' (line 1049)
        length_123233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 21), 'length')
        
        # Call to get(...): (line 1049)
        # Processing the call arguments (line 1049)
        unicode_123236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 40), 'unicode', u'width')
        float_123237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 49), 'float')
        # Processing the call keyword arguments (line 1049)
        kwargs_123238 = {}
        # Getting the type of 'sizes' (line 1049)
        sizes_123234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 30), 'sizes', False)
        # Obtaining the member 'get' of a type (line 1049)
        get_123235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1049, 30), sizes_123234, 'get')
        # Calling get(args, kwargs) (line 1049)
        get_call_result_123239 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 30), get_123235, *[unicode_123236, float_123237], **kwargs_123238)
        
        # Applying the binary operator '*' (line 1049)
        result_mul_123240 = python_operator(stypy.reporting.localization.Localization(__file__, 1049, 21), '*', length_123233, get_call_result_123239)
        
        # Assigning a type to the variable 'full_width' (line 1049)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'full_width', result_mul_123240)
        
        # Assigning a BinOp to a Name (line 1050):
        
        # Assigning a BinOp to a Name (line 1050):
        # Getting the type of 'length' (line 1050)
        length_123241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 20), 'length')
        
        # Call to get(...): (line 1050)
        # Processing the call arguments (line 1050)
        unicode_123244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 39), 'unicode', u'emptybarb')
        float_123245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 52), 'float')
        # Processing the call keyword arguments (line 1050)
        kwargs_123246 = {}
        # Getting the type of 'sizes' (line 1050)
        sizes_123242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 29), 'sizes', False)
        # Obtaining the member 'get' of a type (line 1050)
        get_123243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 29), sizes_123242, 'get')
        # Calling get(args, kwargs) (line 1050)
        get_call_result_123247 = invoke(stypy.reporting.localization.Localization(__file__, 1050, 29), get_123243, *[unicode_123244, float_123245], **kwargs_123246)
        
        # Applying the binary operator '*' (line 1050)
        result_mul_123248 = python_operator(stypy.reporting.localization.Localization(__file__, 1050, 20), '*', length_123241, get_call_result_123247)
        
        # Assigning a type to the variable 'empty_rad' (line 1050)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 8), 'empty_rad', result_mul_123248)
        
        # Assigning a Call to a Name (line 1053):
        
        # Assigning a Call to a Name (line 1053):
        
        # Call to dict(...): (line 1053)
        # Processing the call keyword arguments (line 1053)
        float_123250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 32), 'float')
        keyword_123251 = float_123250
        
        # Getting the type of 'length' (line 1053)
        length_123252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 45), 'length', False)
        # Applying the 'usub' unary operator (line 1053)
        result___neg___123253 = python_operator(stypy.reporting.localization.Localization(__file__, 1053, 44), 'usub', length_123252)
        
        float_123254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 54), 'float')
        # Applying the binary operator 'div' (line 1053)
        result_div_123255 = python_operator(stypy.reporting.localization.Localization(__file__, 1053, 44), 'div', result___neg___123253, float_123254)
        
        keyword_123256 = result_div_123255
        kwargs_123257 = {'middle': keyword_123256, 'tip': keyword_123251}
        # Getting the type of 'dict' (line 1053)
        dict_123249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 23), 'dict', False)
        # Calling dict(args, kwargs) (line 1053)
        dict_call_result_123258 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 23), dict_123249, *[], **kwargs_123257)
        
        # Assigning a type to the variable 'pivot_points' (line 1053)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1053, 8), 'pivot_points', dict_call_result_123258)
        
        # Getting the type of 'flip' (line 1056)
        flip_123259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 11), 'flip')
        # Testing the type of an if condition (line 1056)
        if_condition_123260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1056, 8), flip_123259)
        # Assigning a type to the variable 'if_condition_123260' (line 1056)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 8), 'if_condition_123260', if_condition_123260)
        # SSA begins for if statement (line 1056)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a UnaryOp to a Name (line 1057):
        
        # Assigning a UnaryOp to a Name (line 1057):
        
        # Getting the type of 'full_height' (line 1057)
        full_height_123261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 27), 'full_height')
        # Applying the 'usub' unary operator (line 1057)
        result___neg___123262 = python_operator(stypy.reporting.localization.Localization(__file__, 1057, 26), 'usub', full_height_123261)
        
        # Assigning a type to the variable 'full_height' (line 1057)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 12), 'full_height', result___neg___123262)
        # SSA join for if statement (line 1056)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 1059):
        
        # Assigning a Num to a Name (line 1059):
        float_123263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1059, 15), 'float')
        # Assigning a type to the variable 'endx' (line 1059)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 8), 'endx', float_123263)
        
        
        # SSA begins for try-except statement (line 1060)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 1061):
        
        # Assigning a Call to a Name (line 1061):
        
        # Call to float(...): (line 1061)
        # Processing the call arguments (line 1061)
        # Getting the type of 'pivot' (line 1061)
        pivot_123265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 25), 'pivot', False)
        # Processing the call keyword arguments (line 1061)
        kwargs_123266 = {}
        # Getting the type of 'float' (line 1061)
        float_123264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 19), 'float', False)
        # Calling float(args, kwargs) (line 1061)
        float_call_result_123267 = invoke(stypy.reporting.localization.Localization(__file__, 1061, 19), float_123264, *[pivot_123265], **kwargs_123266)
        
        # Assigning a type to the variable 'endy' (line 1061)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1061, 12), 'endy', float_call_result_123267)
        # SSA branch for the except part of a try statement (line 1060)
        # SSA branch for the except 'ValueError' branch of a try statement (line 1060)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Subscript to a Name (line 1063):
        
        # Assigning a Subscript to a Name (line 1063):
        
        # Obtaining the type of the subscript
        
        # Call to lower(...): (line 1063)
        # Processing the call keyword arguments (line 1063)
        kwargs_123270 = {}
        # Getting the type of 'pivot' (line 1063)
        pivot_123268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 32), 'pivot', False)
        # Obtaining the member 'lower' of a type (line 1063)
        lower_123269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1063, 32), pivot_123268, 'lower')
        # Calling lower(args, kwargs) (line 1063)
        lower_call_result_123271 = invoke(stypy.reporting.localization.Localization(__file__, 1063, 32), lower_123269, *[], **kwargs_123270)
        
        # Getting the type of 'pivot_points' (line 1063)
        pivot_points_123272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 19), 'pivot_points')
        # Obtaining the member '__getitem__' of a type (line 1063)
        getitem___123273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1063, 19), pivot_points_123272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1063)
        subscript_call_result_123274 = invoke(stypy.reporting.localization.Localization(__file__, 1063, 19), getitem___123273, lower_call_result_123271)
        
        # Assigning a type to the variable 'endy' (line 1063)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1063, 12), 'endy', subscript_call_result_123274)
        # SSA join for try-except statement (line 1060)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a UnaryOp to a Name (line 1069):
        
        # Assigning a UnaryOp to a Name (line 1069):
        
        
        # Call to arctan2(...): (line 1069)
        # Processing the call arguments (line 1069)
        # Getting the type of 'v' (line 1069)
        v_123277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 30), 'v', False)
        # Getting the type of 'u' (line 1069)
        u_123278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 33), 'u', False)
        # Processing the call keyword arguments (line 1069)
        kwargs_123279 = {}
        # Getting the type of 'ma' (line 1069)
        ma_123275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 19), 'ma', False)
        # Obtaining the member 'arctan2' of a type (line 1069)
        arctan2_123276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 19), ma_123275, 'arctan2')
        # Calling arctan2(args, kwargs) (line 1069)
        arctan2_call_result_123280 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 19), arctan2_123276, *[v_123277, u_123278], **kwargs_123279)
        
        # Getting the type of 'np' (line 1069)
        np_123281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 38), 'np')
        # Obtaining the member 'pi' of a type (line 1069)
        pi_123282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 38), np_123281, 'pi')
        int_123283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 46), 'int')
        # Applying the binary operator 'div' (line 1069)
        result_div_123284 = python_operator(stypy.reporting.localization.Localization(__file__, 1069, 38), 'div', pi_123282, int_123283)
        
        # Applying the binary operator '+' (line 1069)
        result_add_123285 = python_operator(stypy.reporting.localization.Localization(__file__, 1069, 19), '+', arctan2_call_result_123280, result_div_123284)
        
        # Applying the 'usub' unary operator (line 1069)
        result___neg___123286 = python_operator(stypy.reporting.localization.Localization(__file__, 1069, 17), 'usub', result_add_123285)
        
        # Assigning a type to the variable 'angles' (line 1069)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 8), 'angles', result___neg___123286)
        
        # Assigning a Call to a Name (line 1075):
        
        # Assigning a Call to a Name (line 1075):
        
        # Call to get_verts(...): (line 1075)
        # Processing the call keyword arguments (line 1075)
        kwargs_123296 = {}
        
        # Call to CirclePolygon(...): (line 1075)
        # Processing the call arguments (line 1075)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1075)
        tuple_123288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1075)
        # Adding element type (line 1075)
        int_123289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1075, 30), tuple_123288, int_123289)
        # Adding element type (line 1075)
        int_123290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1075, 30), tuple_123288, int_123290)
        
        # Processing the call keyword arguments (line 1075)
        # Getting the type of 'empty_rad' (line 1075)
        empty_rad_123291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 44), 'empty_rad', False)
        keyword_123292 = empty_rad_123291
        kwargs_123293 = {'radius': keyword_123292}
        # Getting the type of 'CirclePolygon' (line 1075)
        CirclePolygon_123287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 15), 'CirclePolygon', False)
        # Calling CirclePolygon(args, kwargs) (line 1075)
        CirclePolygon_call_result_123294 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 15), CirclePolygon_123287, *[tuple_123288], **kwargs_123293)
        
        # Obtaining the member 'get_verts' of a type (line 1075)
        get_verts_123295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 15), CirclePolygon_call_result_123294, 'get_verts')
        # Calling get_verts(args, kwargs) (line 1075)
        get_verts_call_result_123297 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 15), get_verts_123295, *[], **kwargs_123296)
        
        # Assigning a type to the variable 'circ' (line 1075)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'circ', get_verts_call_result_123297)
        
        # Getting the type of 'fill_empty' (line 1076)
        fill_empty_123298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 11), 'fill_empty')
        # Testing the type of an if condition (line 1076)
        if_condition_123299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1076, 8), fill_empty_123298)
        # Assigning a type to the variable 'if_condition_123299' (line 1076)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1076, 8), 'if_condition_123299', if_condition_123299)
        # SSA begins for if statement (line 1076)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 1077):
        
        # Assigning a Name to a Name (line 1077):
        # Getting the type of 'circ' (line 1077)
        circ_123300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 25), 'circ')
        # Assigning a type to the variable 'empty_barb' (line 1077)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 12), 'empty_barb', circ_123300)
        # SSA branch for the else part of an if statement (line 1076)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1081):
        
        # Assigning a Call to a Name (line 1081):
        
        # Call to concatenate(...): (line 1081)
        # Processing the call arguments (line 1081)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1081)
        tuple_123303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1081)
        # Adding element type (line 1081)
        # Getting the type of 'circ' (line 1081)
        circ_123304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 41), 'circ', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1081, 41), tuple_123303, circ_123304)
        # Adding element type (line 1081)
        
        # Obtaining the type of the subscript
        int_123305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 54), 'int')
        slice_123306 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1081, 47), None, None, int_123305)
        # Getting the type of 'circ' (line 1081)
        circ_123307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 47), 'circ', False)
        # Obtaining the member '__getitem__' of a type (line 1081)
        getitem___123308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1081, 47), circ_123307, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1081)
        subscript_call_result_123309 = invoke(stypy.reporting.localization.Localization(__file__, 1081, 47), getitem___123308, slice_123306)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1081, 41), tuple_123303, subscript_call_result_123309)
        
        # Processing the call keyword arguments (line 1081)
        kwargs_123310 = {}
        # Getting the type of 'np' (line 1081)
        np_123301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 25), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 1081)
        concatenate_123302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1081, 25), np_123301, 'concatenate')
        # Calling concatenate(args, kwargs) (line 1081)
        concatenate_call_result_123311 = invoke(stypy.reporting.localization.Localization(__file__, 1081, 25), concatenate_123302, *[tuple_123303], **kwargs_123310)
        
        # Assigning a type to the variable 'empty_barb' (line 1081)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1081, 12), 'empty_barb', concatenate_call_result_123311)
        # SSA join for if statement (line 1076)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 1083):
        
        # Assigning a List to a Name (line 1083):
        
        # Obtaining an instance of the builtin type 'list' (line 1083)
        list_123312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1083, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1083)
        
        # Assigning a type to the variable 'barb_list' (line 1083)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1083, 8), 'barb_list', list_123312)
        
        
        # Call to ndenumerate(...): (line 1084)
        # Processing the call arguments (line 1084)
        # Getting the type of 'angles' (line 1084)
        angles_123315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 43), 'angles', False)
        # Processing the call keyword arguments (line 1084)
        kwargs_123316 = {}
        # Getting the type of 'np' (line 1084)
        np_123313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 28), 'np', False)
        # Obtaining the member 'ndenumerate' of a type (line 1084)
        ndenumerate_123314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1084, 28), np_123313, 'ndenumerate')
        # Calling ndenumerate(args, kwargs) (line 1084)
        ndenumerate_call_result_123317 = invoke(stypy.reporting.localization.Localization(__file__, 1084, 28), ndenumerate_123314, *[angles_123315], **kwargs_123316)
        
        # Testing the type of a for loop iterable (line 1084)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1084, 8), ndenumerate_call_result_123317)
        # Getting the type of the for loop variable (line 1084)
        for_loop_var_123318 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1084, 8), ndenumerate_call_result_123317)
        # Assigning a type to the variable 'index' (line 1084)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1084, 8), 'index', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1084, 8), for_loop_var_123318))
        # Assigning a type to the variable 'angle' (line 1084)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1084, 8), 'angle', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1084, 8), for_loop_var_123318))
        # SSA begins for a for statement (line 1084)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 1087)
        index_123319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 26), 'index')
        # Getting the type of 'empty_flag' (line 1087)
        empty_flag_123320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 15), 'empty_flag')
        # Obtaining the member '__getitem__' of a type (line 1087)
        getitem___123321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1087, 15), empty_flag_123320, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1087)
        subscript_call_result_123322 = invoke(stypy.reporting.localization.Localization(__file__, 1087, 15), getitem___123321, index_123319)
        
        # Testing the type of an if condition (line 1087)
        if_condition_123323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1087, 12), subscript_call_result_123322)
        # Assigning a type to the variable 'if_condition_123323' (line 1087)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1087, 12), 'if_condition_123323', if_condition_123323)
        # SSA begins for if statement (line 1087)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 1090)
        # Processing the call arguments (line 1090)
        # Getting the type of 'empty_barb' (line 1090)
        empty_barb_123326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 33), 'empty_barb', False)
        # Processing the call keyword arguments (line 1090)
        kwargs_123327 = {}
        # Getting the type of 'barb_list' (line 1090)
        barb_list_123324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 16), 'barb_list', False)
        # Obtaining the member 'append' of a type (line 1090)
        append_123325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 16), barb_list_123324, 'append')
        # Calling append(args, kwargs) (line 1090)
        append_call_result_123328 = invoke(stypy.reporting.localization.Localization(__file__, 1090, 16), append_123325, *[empty_barb_123326], **kwargs_123327)
        
        # SSA join for if statement (line 1087)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 1093):
        
        # Assigning a List to a Name (line 1093):
        
        # Obtaining an instance of the builtin type 'list' (line 1093)
        list_123329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1093)
        # Adding element type (line 1093)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1093)
        tuple_123330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1093)
        # Adding element type (line 1093)
        # Getting the type of 'endx' (line 1093)
        endx_123331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 27), 'endx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1093, 27), tuple_123330, endx_123331)
        # Adding element type (line 1093)
        # Getting the type of 'endy' (line 1093)
        endy_123332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 33), 'endy')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1093, 27), tuple_123330, endy_123332)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1093, 25), list_123329, tuple_123330)
        
        # Assigning a type to the variable 'poly_verts' (line 1093)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 12), 'poly_verts', list_123329)
        
        # Assigning a Name to a Name (line 1094):
        
        # Assigning a Name to a Name (line 1094):
        # Getting the type of 'length' (line 1094)
        length_123333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 21), 'length')
        # Assigning a type to the variable 'offset' (line 1094)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 12), 'offset', length_123333)
        
        
        # Call to range(...): (line 1097)
        # Processing the call arguments (line 1097)
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 1097)
        index_123335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 34), 'index', False)
        # Getting the type of 'nflags' (line 1097)
        nflags_123336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 27), 'nflags', False)
        # Obtaining the member '__getitem__' of a type (line 1097)
        getitem___123337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 27), nflags_123336, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1097)
        subscript_call_result_123338 = invoke(stypy.reporting.localization.Localization(__file__, 1097, 27), getitem___123337, index_123335)
        
        # Processing the call keyword arguments (line 1097)
        kwargs_123339 = {}
        # Getting the type of 'range' (line 1097)
        range_123334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 21), 'range', False)
        # Calling range(args, kwargs) (line 1097)
        range_call_result_123340 = invoke(stypy.reporting.localization.Localization(__file__, 1097, 21), range_123334, *[subscript_call_result_123338], **kwargs_123339)
        
        # Testing the type of a for loop iterable (line 1097)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1097, 12), range_call_result_123340)
        # Getting the type of the for loop variable (line 1097)
        for_loop_var_123341 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1097, 12), range_call_result_123340)
        # Assigning a type to the variable 'i' (line 1097)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 12), 'i', for_loop_var_123341)
        # SSA begins for a for statement (line 1097)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'offset' (line 1101)
        offset_123342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 19), 'offset')
        # Getting the type of 'length' (line 1101)
        length_123343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 29), 'length')
        # Applying the binary operator '!=' (line 1101)
        result_ne_123344 = python_operator(stypy.reporting.localization.Localization(__file__, 1101, 19), '!=', offset_123342, length_123343)
        
        # Testing the type of an if condition (line 1101)
        if_condition_123345 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1101, 16), result_ne_123344)
        # Assigning a type to the variable 'if_condition_123345' (line 1101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1101, 16), 'if_condition_123345', if_condition_123345)
        # SSA begins for if statement (line 1101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'offset' (line 1102)
        offset_123346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 20), 'offset')
        # Getting the type of 'spacing' (line 1102)
        spacing_123347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 30), 'spacing')
        float_123348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 40), 'float')
        # Applying the binary operator 'div' (line 1102)
        result_div_123349 = python_operator(stypy.reporting.localization.Localization(__file__, 1102, 30), 'div', spacing_123347, float_123348)
        
        # Applying the binary operator '+=' (line 1102)
        result_iadd_123350 = python_operator(stypy.reporting.localization.Localization(__file__, 1102, 20), '+=', offset_123346, result_div_123349)
        # Assigning a type to the variable 'offset' (line 1102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1102, 20), 'offset', result_iadd_123350)
        
        # SSA join for if statement (line 1101)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to extend(...): (line 1103)
        # Processing the call arguments (line 1103)
        
        # Obtaining an instance of the builtin type 'list' (line 1104)
        list_123353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1104)
        # Adding element type (line 1104)
        
        # Obtaining an instance of the builtin type 'list' (line 1104)
        list_123354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1104)
        # Adding element type (line 1104)
        # Getting the type of 'endx' (line 1104)
        endx_123355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 22), 'endx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 21), list_123354, endx_123355)
        # Adding element type (line 1104)
        # Getting the type of 'endy' (line 1104)
        endy_123356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 28), 'endy', False)
        # Getting the type of 'offset' (line 1104)
        offset_123357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 35), 'offset', False)
        # Applying the binary operator '+' (line 1104)
        result_add_123358 = python_operator(stypy.reporting.localization.Localization(__file__, 1104, 28), '+', endy_123356, offset_123357)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 21), list_123354, result_add_123358)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 20), list_123353, list_123354)
        # Adding element type (line 1104)
        
        # Obtaining an instance of the builtin type 'list' (line 1105)
        list_123359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1105, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1105)
        # Adding element type (line 1105)
        # Getting the type of 'endx' (line 1105)
        endx_123360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 22), 'endx', False)
        # Getting the type of 'full_height' (line 1105)
        full_height_123361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 29), 'full_height', False)
        # Applying the binary operator '+' (line 1105)
        result_add_123362 = python_operator(stypy.reporting.localization.Localization(__file__, 1105, 22), '+', endx_123360, full_height_123361)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1105, 21), list_123359, result_add_123362)
        # Adding element type (line 1105)
        # Getting the type of 'endy' (line 1105)
        endy_123363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 42), 'endy', False)
        # Getting the type of 'full_width' (line 1105)
        full_width_123364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 49), 'full_width', False)
        int_123365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1105, 62), 'int')
        # Applying the binary operator 'div' (line 1105)
        result_div_123366 = python_operator(stypy.reporting.localization.Localization(__file__, 1105, 49), 'div', full_width_123364, int_123365)
        
        # Applying the binary operator '-' (line 1105)
        result_sub_123367 = python_operator(stypy.reporting.localization.Localization(__file__, 1105, 42), '-', endy_123363, result_div_123366)
        
        # Getting the type of 'offset' (line 1105)
        offset_123368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 66), 'offset', False)
        # Applying the binary operator '+' (line 1105)
        result_add_123369 = python_operator(stypy.reporting.localization.Localization(__file__, 1105, 64), '+', result_sub_123367, offset_123368)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1105, 21), list_123359, result_add_123369)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 20), list_123353, list_123359)
        # Adding element type (line 1104)
        
        # Obtaining an instance of the builtin type 'list' (line 1106)
        list_123370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1106)
        # Adding element type (line 1106)
        # Getting the type of 'endx' (line 1106)
        endx_123371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 22), 'endx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 21), list_123370, endx_123371)
        # Adding element type (line 1106)
        # Getting the type of 'endy' (line 1106)
        endy_123372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 28), 'endy', False)
        # Getting the type of 'full_width' (line 1106)
        full_width_123373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 35), 'full_width', False)
        # Applying the binary operator '-' (line 1106)
        result_sub_123374 = python_operator(stypy.reporting.localization.Localization(__file__, 1106, 28), '-', endy_123372, full_width_123373)
        
        # Getting the type of 'offset' (line 1106)
        offset_123375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 48), 'offset', False)
        # Applying the binary operator '+' (line 1106)
        result_add_123376 = python_operator(stypy.reporting.localization.Localization(__file__, 1106, 46), '+', result_sub_123374, offset_123375)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 21), list_123370, result_add_123376)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 20), list_123353, list_123370)
        
        # Processing the call keyword arguments (line 1103)
        kwargs_123377 = {}
        # Getting the type of 'poly_verts' (line 1103)
        poly_verts_123351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 16), 'poly_verts', False)
        # Obtaining the member 'extend' of a type (line 1103)
        extend_123352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1103, 16), poly_verts_123351, 'extend')
        # Calling extend(args, kwargs) (line 1103)
        extend_call_result_123378 = invoke(stypy.reporting.localization.Localization(__file__, 1103, 16), extend_123352, *[list_123353], **kwargs_123377)
        
        
        # Getting the type of 'offset' (line 1108)
        offset_123379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 16), 'offset')
        # Getting the type of 'full_width' (line 1108)
        full_width_123380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 26), 'full_width')
        # Getting the type of 'spacing' (line 1108)
        spacing_123381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 39), 'spacing')
        # Applying the binary operator '+' (line 1108)
        result_add_123382 = python_operator(stypy.reporting.localization.Localization(__file__, 1108, 26), '+', full_width_123380, spacing_123381)
        
        # Applying the binary operator '-=' (line 1108)
        result_isub_123383 = python_operator(stypy.reporting.localization.Localization(__file__, 1108, 16), '-=', offset_123379, result_add_123382)
        # Assigning a type to the variable 'offset' (line 1108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 16), 'offset', result_isub_123383)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to range(...): (line 1113)
        # Processing the call arguments (line 1113)
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 1113)
        index_123385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 34), 'index', False)
        # Getting the type of 'nbarbs' (line 1113)
        nbarbs_123386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 27), 'nbarbs', False)
        # Obtaining the member '__getitem__' of a type (line 1113)
        getitem___123387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1113, 27), nbarbs_123386, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1113)
        subscript_call_result_123388 = invoke(stypy.reporting.localization.Localization(__file__, 1113, 27), getitem___123387, index_123385)
        
        # Processing the call keyword arguments (line 1113)
        kwargs_123389 = {}
        # Getting the type of 'range' (line 1113)
        range_123384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 21), 'range', False)
        # Calling range(args, kwargs) (line 1113)
        range_call_result_123390 = invoke(stypy.reporting.localization.Localization(__file__, 1113, 21), range_123384, *[subscript_call_result_123388], **kwargs_123389)
        
        # Testing the type of a for loop iterable (line 1113)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1113, 12), range_call_result_123390)
        # Getting the type of the for loop variable (line 1113)
        for_loop_var_123391 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1113, 12), range_call_result_123390)
        # Assigning a type to the variable 'i' (line 1113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1113, 12), 'i', for_loop_var_123391)
        # SSA begins for a for statement (line 1113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to extend(...): (line 1114)
        # Processing the call arguments (line 1114)
        
        # Obtaining an instance of the builtin type 'list' (line 1115)
        list_123394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1115, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1115)
        # Adding element type (line 1115)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1115)
        tuple_123395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1115, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1115)
        # Adding element type (line 1115)
        # Getting the type of 'endx' (line 1115)
        endx_123396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 22), 'endx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1115, 22), tuple_123395, endx_123396)
        # Adding element type (line 1115)
        # Getting the type of 'endy' (line 1115)
        endy_123397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 28), 'endy', False)
        # Getting the type of 'offset' (line 1115)
        offset_123398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 35), 'offset', False)
        # Applying the binary operator '+' (line 1115)
        result_add_123399 = python_operator(stypy.reporting.localization.Localization(__file__, 1115, 28), '+', endy_123397, offset_123398)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1115, 22), tuple_123395, result_add_123399)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1115, 20), list_123394, tuple_123395)
        # Adding element type (line 1115)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1116)
        tuple_123400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1116)
        # Adding element type (line 1116)
        # Getting the type of 'endx' (line 1116)
        endx_123401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 22), 'endx', False)
        # Getting the type of 'full_height' (line 1116)
        full_height_123402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 29), 'full_height', False)
        # Applying the binary operator '+' (line 1116)
        result_add_123403 = python_operator(stypy.reporting.localization.Localization(__file__, 1116, 22), '+', endx_123401, full_height_123402)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1116, 22), tuple_123400, result_add_123403)
        # Adding element type (line 1116)
        # Getting the type of 'endy' (line 1116)
        endy_123404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 42), 'endy', False)
        # Getting the type of 'offset' (line 1116)
        offset_123405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 49), 'offset', False)
        # Applying the binary operator '+' (line 1116)
        result_add_123406 = python_operator(stypy.reporting.localization.Localization(__file__, 1116, 42), '+', endy_123404, offset_123405)
        
        # Getting the type of 'full_width' (line 1116)
        full_width_123407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 58), 'full_width', False)
        int_123408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 71), 'int')
        # Applying the binary operator 'div' (line 1116)
        result_div_123409 = python_operator(stypy.reporting.localization.Localization(__file__, 1116, 58), 'div', full_width_123407, int_123408)
        
        # Applying the binary operator '+' (line 1116)
        result_add_123410 = python_operator(stypy.reporting.localization.Localization(__file__, 1116, 56), '+', result_add_123406, result_div_123409)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1116, 22), tuple_123400, result_add_123410)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1115, 20), list_123394, tuple_123400)
        # Adding element type (line 1115)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1117)
        tuple_123411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1117)
        # Adding element type (line 1117)
        # Getting the type of 'endx' (line 1117)
        endx_123412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 22), 'endx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1117, 22), tuple_123411, endx_123412)
        # Adding element type (line 1117)
        # Getting the type of 'endy' (line 1117)
        endy_123413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 28), 'endy', False)
        # Getting the type of 'offset' (line 1117)
        offset_123414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 35), 'offset', False)
        # Applying the binary operator '+' (line 1117)
        result_add_123415 = python_operator(stypy.reporting.localization.Localization(__file__, 1117, 28), '+', endy_123413, offset_123414)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1117, 22), tuple_123411, result_add_123415)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1115, 20), list_123394, tuple_123411)
        
        # Processing the call keyword arguments (line 1114)
        kwargs_123416 = {}
        # Getting the type of 'poly_verts' (line 1114)
        poly_verts_123392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 16), 'poly_verts', False)
        # Obtaining the member 'extend' of a type (line 1114)
        extend_123393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 16), poly_verts_123392, 'extend')
        # Calling extend(args, kwargs) (line 1114)
        extend_call_result_123417 = invoke(stypy.reporting.localization.Localization(__file__, 1114, 16), extend_123393, *[list_123394], **kwargs_123416)
        
        
        # Getting the type of 'offset' (line 1119)
        offset_123418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 16), 'offset')
        # Getting the type of 'spacing' (line 1119)
        spacing_123419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 26), 'spacing')
        # Applying the binary operator '-=' (line 1119)
        result_isub_123420 = python_operator(stypy.reporting.localization.Localization(__file__, 1119, 16), '-=', offset_123418, spacing_123419)
        # Assigning a type to the variable 'offset' (line 1119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1119, 16), 'offset', result_isub_123420)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 1122)
        index_123421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 25), 'index')
        # Getting the type of 'half_barb' (line 1122)
        half_barb_123422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 15), 'half_barb')
        # Obtaining the member '__getitem__' of a type (line 1122)
        getitem___123423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1122, 15), half_barb_123422, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1122)
        subscript_call_result_123424 = invoke(stypy.reporting.localization.Localization(__file__, 1122, 15), getitem___123423, index_123421)
        
        # Testing the type of an if condition (line 1122)
        if_condition_123425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1122, 12), subscript_call_result_123424)
        # Assigning a type to the variable 'if_condition_123425' (line 1122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1122, 12), 'if_condition_123425', if_condition_123425)
        # SSA begins for if statement (line 1122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'offset' (line 1126)
        offset_123426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 19), 'offset')
        # Getting the type of 'length' (line 1126)
        length_123427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 29), 'length')
        # Applying the binary operator '==' (line 1126)
        result_eq_123428 = python_operator(stypy.reporting.localization.Localization(__file__, 1126, 19), '==', offset_123426, length_123427)
        
        # Testing the type of an if condition (line 1126)
        if_condition_123429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1126, 16), result_eq_123428)
        # Assigning a type to the variable 'if_condition_123429' (line 1126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 16), 'if_condition_123429', if_condition_123429)
        # SSA begins for if statement (line 1126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 1127)
        # Processing the call arguments (line 1127)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1127)
        tuple_123432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1127)
        # Adding element type (line 1127)
        # Getting the type of 'endx' (line 1127)
        endx_123433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 39), 'endx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1127, 39), tuple_123432, endx_123433)
        # Adding element type (line 1127)
        # Getting the type of 'endy' (line 1127)
        endy_123434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 45), 'endy', False)
        # Getting the type of 'offset' (line 1127)
        offset_123435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 52), 'offset', False)
        # Applying the binary operator '+' (line 1127)
        result_add_123436 = python_operator(stypy.reporting.localization.Localization(__file__, 1127, 45), '+', endy_123434, offset_123435)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1127, 39), tuple_123432, result_add_123436)
        
        # Processing the call keyword arguments (line 1127)
        kwargs_123437 = {}
        # Getting the type of 'poly_verts' (line 1127)
        poly_verts_123430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 20), 'poly_verts', False)
        # Obtaining the member 'append' of a type (line 1127)
        append_123431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1127, 20), poly_verts_123430, 'append')
        # Calling append(args, kwargs) (line 1127)
        append_call_result_123438 = invoke(stypy.reporting.localization.Localization(__file__, 1127, 20), append_123431, *[tuple_123432], **kwargs_123437)
        
        
        # Getting the type of 'offset' (line 1128)
        offset_123439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 20), 'offset')
        float_123440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1128, 30), 'float')
        # Getting the type of 'spacing' (line 1128)
        spacing_123441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 36), 'spacing')
        # Applying the binary operator '*' (line 1128)
        result_mul_123442 = python_operator(stypy.reporting.localization.Localization(__file__, 1128, 30), '*', float_123440, spacing_123441)
        
        # Applying the binary operator '-=' (line 1128)
        result_isub_123443 = python_operator(stypy.reporting.localization.Localization(__file__, 1128, 20), '-=', offset_123439, result_mul_123442)
        # Assigning a type to the variable 'offset' (line 1128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1128, 20), 'offset', result_isub_123443)
        
        # SSA join for if statement (line 1126)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to extend(...): (line 1129)
        # Processing the call arguments (line 1129)
        
        # Obtaining an instance of the builtin type 'list' (line 1130)
        list_123446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1130)
        # Adding element type (line 1130)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1130)
        tuple_123447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1130)
        # Adding element type (line 1130)
        # Getting the type of 'endx' (line 1130)
        endx_123448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 22), 'endx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1130, 22), tuple_123447, endx_123448)
        # Adding element type (line 1130)
        # Getting the type of 'endy' (line 1130)
        endy_123449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 28), 'endy', False)
        # Getting the type of 'offset' (line 1130)
        offset_123450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 35), 'offset', False)
        # Applying the binary operator '+' (line 1130)
        result_add_123451 = python_operator(stypy.reporting.localization.Localization(__file__, 1130, 28), '+', endy_123449, offset_123450)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1130, 22), tuple_123447, result_add_123451)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1130, 20), list_123446, tuple_123447)
        # Adding element type (line 1130)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1131)
        tuple_123452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1131)
        # Adding element type (line 1131)
        # Getting the type of 'endx' (line 1131)
        endx_123453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 22), 'endx', False)
        # Getting the type of 'full_height' (line 1131)
        full_height_123454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 29), 'full_height', False)
        int_123455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 43), 'int')
        # Applying the binary operator 'div' (line 1131)
        result_div_123456 = python_operator(stypy.reporting.localization.Localization(__file__, 1131, 29), 'div', full_height_123454, int_123455)
        
        # Applying the binary operator '+' (line 1131)
        result_add_123457 = python_operator(stypy.reporting.localization.Localization(__file__, 1131, 22), '+', endx_123453, result_div_123456)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1131, 22), tuple_123452, result_add_123457)
        # Adding element type (line 1131)
        # Getting the type of 'endy' (line 1131)
        endy_123458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 46), 'endy', False)
        # Getting the type of 'offset' (line 1131)
        offset_123459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 53), 'offset', False)
        # Applying the binary operator '+' (line 1131)
        result_add_123460 = python_operator(stypy.reporting.localization.Localization(__file__, 1131, 46), '+', endy_123458, offset_123459)
        
        # Getting the type of 'full_width' (line 1131)
        full_width_123461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 62), 'full_width', False)
        int_123462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 75), 'int')
        # Applying the binary operator 'div' (line 1131)
        result_div_123463 = python_operator(stypy.reporting.localization.Localization(__file__, 1131, 62), 'div', full_width_123461, int_123462)
        
        # Applying the binary operator '+' (line 1131)
        result_add_123464 = python_operator(stypy.reporting.localization.Localization(__file__, 1131, 60), '+', result_add_123460, result_div_123463)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1131, 22), tuple_123452, result_add_123464)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1130, 20), list_123446, tuple_123452)
        # Adding element type (line 1130)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1132)
        tuple_123465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1132)
        # Adding element type (line 1132)
        # Getting the type of 'endx' (line 1132)
        endx_123466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 22), 'endx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1132, 22), tuple_123465, endx_123466)
        # Adding element type (line 1132)
        # Getting the type of 'endy' (line 1132)
        endy_123467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 28), 'endy', False)
        # Getting the type of 'offset' (line 1132)
        offset_123468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 35), 'offset', False)
        # Applying the binary operator '+' (line 1132)
        result_add_123469 = python_operator(stypy.reporting.localization.Localization(__file__, 1132, 28), '+', endy_123467, offset_123468)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1132, 22), tuple_123465, result_add_123469)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1130, 20), list_123446, tuple_123465)
        
        # Processing the call keyword arguments (line 1129)
        kwargs_123470 = {}
        # Getting the type of 'poly_verts' (line 1129)
        poly_verts_123444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 16), 'poly_verts', False)
        # Obtaining the member 'extend' of a type (line 1129)
        extend_123445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1129, 16), poly_verts_123444, 'extend')
        # Calling extend(args, kwargs) (line 1129)
        extend_call_result_123471 = invoke(stypy.reporting.localization.Localization(__file__, 1129, 16), extend_123445, *[list_123446], **kwargs_123470)
        
        # SSA join for if statement (line 1122)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1137):
        
        # Assigning a Call to a Name (line 1137):
        
        # Call to transform(...): (line 1137)
        # Processing the call arguments (line 1137)
        # Getting the type of 'poly_verts' (line 1138)
        poly_verts_123482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 16), 'poly_verts', False)
        # Processing the call keyword arguments (line 1137)
        kwargs_123483 = {}
        
        # Call to rotate(...): (line 1137)
        # Processing the call arguments (line 1137)
        
        # Getting the type of 'angle' (line 1137)
        angle_123477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 55), 'angle', False)
        # Applying the 'usub' unary operator (line 1137)
        result___neg___123478 = python_operator(stypy.reporting.localization.Localization(__file__, 1137, 54), 'usub', angle_123477)
        
        # Processing the call keyword arguments (line 1137)
        kwargs_123479 = {}
        
        # Call to Affine2D(...): (line 1137)
        # Processing the call keyword arguments (line 1137)
        kwargs_123474 = {}
        # Getting the type of 'transforms' (line 1137)
        transforms_123472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 25), 'transforms', False)
        # Obtaining the member 'Affine2D' of a type (line 1137)
        Affine2D_123473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1137, 25), transforms_123472, 'Affine2D')
        # Calling Affine2D(args, kwargs) (line 1137)
        Affine2D_call_result_123475 = invoke(stypy.reporting.localization.Localization(__file__, 1137, 25), Affine2D_123473, *[], **kwargs_123474)
        
        # Obtaining the member 'rotate' of a type (line 1137)
        rotate_123476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1137, 25), Affine2D_call_result_123475, 'rotate')
        # Calling rotate(args, kwargs) (line 1137)
        rotate_call_result_123480 = invoke(stypy.reporting.localization.Localization(__file__, 1137, 25), rotate_123476, *[result___neg___123478], **kwargs_123479)
        
        # Obtaining the member 'transform' of a type (line 1137)
        transform_123481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1137, 25), rotate_call_result_123480, 'transform')
        # Calling transform(args, kwargs) (line 1137)
        transform_call_result_123484 = invoke(stypy.reporting.localization.Localization(__file__, 1137, 25), transform_123481, *[poly_verts_123482], **kwargs_123483)
        
        # Assigning a type to the variable 'poly_verts' (line 1137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1137, 12), 'poly_verts', transform_call_result_123484)
        
        # Call to append(...): (line 1139)
        # Processing the call arguments (line 1139)
        # Getting the type of 'poly_verts' (line 1139)
        poly_verts_123487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 29), 'poly_verts', False)
        # Processing the call keyword arguments (line 1139)
        kwargs_123488 = {}
        # Getting the type of 'barb_list' (line 1139)
        barb_list_123485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 12), 'barb_list', False)
        # Obtaining the member 'append' of a type (line 1139)
        append_123486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 12), barb_list_123485, 'append')
        # Calling append(args, kwargs) (line 1139)
        append_call_result_123489 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 12), append_123486, *[poly_verts_123487], **kwargs_123488)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'barb_list' (line 1141)
        barb_list_123490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 15), 'barb_list')
        # Assigning a type to the variable 'stypy_return_type' (line 1141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1141, 8), 'stypy_return_type', barb_list_123490)
        
        # ################# End of '_make_barbs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_make_barbs' in the type store
        # Getting the type of 'stypy_return_type' (line 998)
        stypy_return_type_123491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123491)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_make_barbs'
        return stypy_return_type_123491


    @norecursion
    def set_UVC(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 1143)
        None_123492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 30), 'None')
        defaults = [None_123492]
        # Create a new context for function 'set_UVC'
        module_type_store = module_type_store.open_function_context('set_UVC', 1143, 4, False)
        # Assigning a type to the variable 'self' (line 1144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Barbs.set_UVC.__dict__.__setitem__('stypy_localization', localization)
        Barbs.set_UVC.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Barbs.set_UVC.__dict__.__setitem__('stypy_type_store', module_type_store)
        Barbs.set_UVC.__dict__.__setitem__('stypy_function_name', 'Barbs.set_UVC')
        Barbs.set_UVC.__dict__.__setitem__('stypy_param_names_list', ['U', 'V', 'C'])
        Barbs.set_UVC.__dict__.__setitem__('stypy_varargs_param_name', None)
        Barbs.set_UVC.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Barbs.set_UVC.__dict__.__setitem__('stypy_call_defaults', defaults)
        Barbs.set_UVC.__dict__.__setitem__('stypy_call_varargs', varargs)
        Barbs.set_UVC.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Barbs.set_UVC.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Barbs.set_UVC', ['U', 'V', 'C'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_UVC', localization, ['U', 'V', 'C'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_UVC(...)' code ##################

        
        # Assigning a Call to a Attribute (line 1144):
        
        # Assigning a Call to a Attribute (line 1144):
        
        # Call to ravel(...): (line 1144)
        # Processing the call keyword arguments (line 1144)
        kwargs_123501 = {}
        
        # Call to masked_invalid(...): (line 1144)
        # Processing the call arguments (line 1144)
        # Getting the type of 'U' (line 1144)
        U_123495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 35), 'U', False)
        # Processing the call keyword arguments (line 1144)
        # Getting the type of 'False' (line 1144)
        False_123496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 43), 'False', False)
        keyword_123497 = False_123496
        kwargs_123498 = {'copy': keyword_123497}
        # Getting the type of 'ma' (line 1144)
        ma_123493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 17), 'ma', False)
        # Obtaining the member 'masked_invalid' of a type (line 1144)
        masked_invalid_123494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1144, 17), ma_123493, 'masked_invalid')
        # Calling masked_invalid(args, kwargs) (line 1144)
        masked_invalid_call_result_123499 = invoke(stypy.reporting.localization.Localization(__file__, 1144, 17), masked_invalid_123494, *[U_123495], **kwargs_123498)
        
        # Obtaining the member 'ravel' of a type (line 1144)
        ravel_123500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1144, 17), masked_invalid_call_result_123499, 'ravel')
        # Calling ravel(args, kwargs) (line 1144)
        ravel_call_result_123502 = invoke(stypy.reporting.localization.Localization(__file__, 1144, 17), ravel_123500, *[], **kwargs_123501)
        
        # Getting the type of 'self' (line 1144)
        self_123503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 8), 'self')
        # Setting the type of the member 'u' of a type (line 1144)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1144, 8), self_123503, 'u', ravel_call_result_123502)
        
        # Assigning a Call to a Attribute (line 1145):
        
        # Assigning a Call to a Attribute (line 1145):
        
        # Call to ravel(...): (line 1145)
        # Processing the call keyword arguments (line 1145)
        kwargs_123512 = {}
        
        # Call to masked_invalid(...): (line 1145)
        # Processing the call arguments (line 1145)
        # Getting the type of 'V' (line 1145)
        V_123506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 35), 'V', False)
        # Processing the call keyword arguments (line 1145)
        # Getting the type of 'False' (line 1145)
        False_123507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 43), 'False', False)
        keyword_123508 = False_123507
        kwargs_123509 = {'copy': keyword_123508}
        # Getting the type of 'ma' (line 1145)
        ma_123504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 17), 'ma', False)
        # Obtaining the member 'masked_invalid' of a type (line 1145)
        masked_invalid_123505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1145, 17), ma_123504, 'masked_invalid')
        # Calling masked_invalid(args, kwargs) (line 1145)
        masked_invalid_call_result_123510 = invoke(stypy.reporting.localization.Localization(__file__, 1145, 17), masked_invalid_123505, *[V_123506], **kwargs_123509)
        
        # Obtaining the member 'ravel' of a type (line 1145)
        ravel_123511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1145, 17), masked_invalid_call_result_123510, 'ravel')
        # Calling ravel(args, kwargs) (line 1145)
        ravel_call_result_123513 = invoke(stypy.reporting.localization.Localization(__file__, 1145, 17), ravel_123511, *[], **kwargs_123512)
        
        # Getting the type of 'self' (line 1145)
        self_123514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 8), 'self')
        # Setting the type of the member 'v' of a type (line 1145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1145, 8), self_123514, 'v', ravel_call_result_123513)
        
        # Type idiom detected: calculating its left and rigth part (line 1146)
        # Getting the type of 'C' (line 1146)
        C_123515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 8), 'C')
        # Getting the type of 'None' (line 1146)
        None_123516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 20), 'None')
        
        (may_be_123517, more_types_in_union_123518) = may_not_be_none(C_123515, None_123516)

        if may_be_123517:

            if more_types_in_union_123518:
                # Runtime conditional SSA (line 1146)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 1147):
            
            # Assigning a Call to a Name (line 1147):
            
            # Call to ravel(...): (line 1147)
            # Processing the call keyword arguments (line 1147)
            kwargs_123527 = {}
            
            # Call to masked_invalid(...): (line 1147)
            # Processing the call arguments (line 1147)
            # Getting the type of 'C' (line 1147)
            C_123521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 34), 'C', False)
            # Processing the call keyword arguments (line 1147)
            # Getting the type of 'False' (line 1147)
            False_123522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 42), 'False', False)
            keyword_123523 = False_123522
            kwargs_123524 = {'copy': keyword_123523}
            # Getting the type of 'ma' (line 1147)
            ma_123519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 16), 'ma', False)
            # Obtaining the member 'masked_invalid' of a type (line 1147)
            masked_invalid_123520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1147, 16), ma_123519, 'masked_invalid')
            # Calling masked_invalid(args, kwargs) (line 1147)
            masked_invalid_call_result_123525 = invoke(stypy.reporting.localization.Localization(__file__, 1147, 16), masked_invalid_123520, *[C_123521], **kwargs_123524)
            
            # Obtaining the member 'ravel' of a type (line 1147)
            ravel_123526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1147, 16), masked_invalid_call_result_123525, 'ravel')
            # Calling ravel(args, kwargs) (line 1147)
            ravel_call_result_123528 = invoke(stypy.reporting.localization.Localization(__file__, 1147, 16), ravel_123526, *[], **kwargs_123527)
            
            # Assigning a type to the variable 'c' (line 1147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1147, 12), 'c', ravel_call_result_123528)
            
            # Assigning a Call to a Tuple (line 1148):
            
            # Assigning a Call to a Name:
            
            # Call to delete_masked_points(...): (line 1148)
            # Processing the call arguments (line 1148)
            
            # Call to ravel(...): (line 1148)
            # Processing the call keyword arguments (line 1148)
            kwargs_123533 = {}
            # Getting the type of 'self' (line 1148)
            self_123530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 49), 'self', False)
            # Obtaining the member 'x' of a type (line 1148)
            x_123531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1148, 49), self_123530, 'x')
            # Obtaining the member 'ravel' of a type (line 1148)
            ravel_123532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1148, 49), x_123531, 'ravel')
            # Calling ravel(args, kwargs) (line 1148)
            ravel_call_result_123534 = invoke(stypy.reporting.localization.Localization(__file__, 1148, 49), ravel_123532, *[], **kwargs_123533)
            
            
            # Call to ravel(...): (line 1149)
            # Processing the call keyword arguments (line 1149)
            kwargs_123538 = {}
            # Getting the type of 'self' (line 1149)
            self_123535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 49), 'self', False)
            # Obtaining the member 'y' of a type (line 1149)
            y_123536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1149, 49), self_123535, 'y')
            # Obtaining the member 'ravel' of a type (line 1149)
            ravel_123537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1149, 49), y_123536, 'ravel')
            # Calling ravel(args, kwargs) (line 1149)
            ravel_call_result_123539 = invoke(stypy.reporting.localization.Localization(__file__, 1149, 49), ravel_123537, *[], **kwargs_123538)
            
            # Getting the type of 'self' (line 1150)
            self_123540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 49), 'self', False)
            # Obtaining the member 'u' of a type (line 1150)
            u_123541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1150, 49), self_123540, 'u')
            # Getting the type of 'self' (line 1150)
            self_123542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 57), 'self', False)
            # Obtaining the member 'v' of a type (line 1150)
            v_123543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1150, 57), self_123542, 'v')
            # Getting the type of 'c' (line 1150)
            c_123544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 65), 'c', False)
            # Processing the call keyword arguments (line 1148)
            kwargs_123545 = {}
            # Getting the type of 'delete_masked_points' (line 1148)
            delete_masked_points_123529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 28), 'delete_masked_points', False)
            # Calling delete_masked_points(args, kwargs) (line 1148)
            delete_masked_points_call_result_123546 = invoke(stypy.reporting.localization.Localization(__file__, 1148, 28), delete_masked_points_123529, *[ravel_call_result_123534, ravel_call_result_123539, u_123541, v_123543, c_123544], **kwargs_123545)
            
            # Assigning a type to the variable 'call_assignment_120698' (line 1148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120698', delete_masked_points_call_result_123546)
            
            # Assigning a Call to a Name (line 1148):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_123549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1148, 12), 'int')
            # Processing the call keyword arguments
            kwargs_123550 = {}
            # Getting the type of 'call_assignment_120698' (line 1148)
            call_assignment_120698_123547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120698', False)
            # Obtaining the member '__getitem__' of a type (line 1148)
            getitem___123548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1148, 12), call_assignment_120698_123547, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_123551 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123548, *[int_123549], **kwargs_123550)
            
            # Assigning a type to the variable 'call_assignment_120699' (line 1148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120699', getitem___call_result_123551)
            
            # Assigning a Name to a Name (line 1148):
            # Getting the type of 'call_assignment_120699' (line 1148)
            call_assignment_120699_123552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120699')
            # Assigning a type to the variable 'x' (line 1148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'x', call_assignment_120699_123552)
            
            # Assigning a Call to a Name (line 1148):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_123555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1148, 12), 'int')
            # Processing the call keyword arguments
            kwargs_123556 = {}
            # Getting the type of 'call_assignment_120698' (line 1148)
            call_assignment_120698_123553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120698', False)
            # Obtaining the member '__getitem__' of a type (line 1148)
            getitem___123554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1148, 12), call_assignment_120698_123553, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_123557 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123554, *[int_123555], **kwargs_123556)
            
            # Assigning a type to the variable 'call_assignment_120700' (line 1148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120700', getitem___call_result_123557)
            
            # Assigning a Name to a Name (line 1148):
            # Getting the type of 'call_assignment_120700' (line 1148)
            call_assignment_120700_123558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120700')
            # Assigning a type to the variable 'y' (line 1148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 15), 'y', call_assignment_120700_123558)
            
            # Assigning a Call to a Name (line 1148):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_123561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1148, 12), 'int')
            # Processing the call keyword arguments
            kwargs_123562 = {}
            # Getting the type of 'call_assignment_120698' (line 1148)
            call_assignment_120698_123559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120698', False)
            # Obtaining the member '__getitem__' of a type (line 1148)
            getitem___123560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1148, 12), call_assignment_120698_123559, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_123563 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123560, *[int_123561], **kwargs_123562)
            
            # Assigning a type to the variable 'call_assignment_120701' (line 1148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120701', getitem___call_result_123563)
            
            # Assigning a Name to a Name (line 1148):
            # Getting the type of 'call_assignment_120701' (line 1148)
            call_assignment_120701_123564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120701')
            # Assigning a type to the variable 'u' (line 1148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 18), 'u', call_assignment_120701_123564)
            
            # Assigning a Call to a Name (line 1148):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_123567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1148, 12), 'int')
            # Processing the call keyword arguments
            kwargs_123568 = {}
            # Getting the type of 'call_assignment_120698' (line 1148)
            call_assignment_120698_123565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120698', False)
            # Obtaining the member '__getitem__' of a type (line 1148)
            getitem___123566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1148, 12), call_assignment_120698_123565, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_123569 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123566, *[int_123567], **kwargs_123568)
            
            # Assigning a type to the variable 'call_assignment_120702' (line 1148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120702', getitem___call_result_123569)
            
            # Assigning a Name to a Name (line 1148):
            # Getting the type of 'call_assignment_120702' (line 1148)
            call_assignment_120702_123570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120702')
            # Assigning a type to the variable 'v' (line 1148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 21), 'v', call_assignment_120702_123570)
            
            # Assigning a Call to a Name (line 1148):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_123573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1148, 12), 'int')
            # Processing the call keyword arguments
            kwargs_123574 = {}
            # Getting the type of 'call_assignment_120698' (line 1148)
            call_assignment_120698_123571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120698', False)
            # Obtaining the member '__getitem__' of a type (line 1148)
            getitem___123572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1148, 12), call_assignment_120698_123571, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_123575 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123572, *[int_123573], **kwargs_123574)
            
            # Assigning a type to the variable 'call_assignment_120703' (line 1148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120703', getitem___call_result_123575)
            
            # Assigning a Name to a Name (line 1148):
            # Getting the type of 'call_assignment_120703' (line 1148)
            call_assignment_120703_123576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 12), 'call_assignment_120703')
            # Assigning a type to the variable 'c' (line 1148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 24), 'c', call_assignment_120703_123576)
            
            # Call to _check_consistent_shapes(...): (line 1151)
            # Processing the call arguments (line 1151)
            # Getting the type of 'x' (line 1151)
            x_123578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 37), 'x', False)
            # Getting the type of 'y' (line 1151)
            y_123579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 40), 'y', False)
            # Getting the type of 'u' (line 1151)
            u_123580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 43), 'u', False)
            # Getting the type of 'v' (line 1151)
            v_123581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 46), 'v', False)
            # Getting the type of 'c' (line 1151)
            c_123582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 49), 'c', False)
            # Processing the call keyword arguments (line 1151)
            kwargs_123583 = {}
            # Getting the type of '_check_consistent_shapes' (line 1151)
            _check_consistent_shapes_123577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 12), '_check_consistent_shapes', False)
            # Calling _check_consistent_shapes(args, kwargs) (line 1151)
            _check_consistent_shapes_call_result_123584 = invoke(stypy.reporting.localization.Localization(__file__, 1151, 12), _check_consistent_shapes_123577, *[x_123578, y_123579, u_123580, v_123581, c_123582], **kwargs_123583)
            

            if more_types_in_union_123518:
                # Runtime conditional SSA for else branch (line 1146)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_123517) or more_types_in_union_123518):
            
            # Assigning a Call to a Tuple (line 1153):
            
            # Assigning a Call to a Name:
            
            # Call to delete_masked_points(...): (line 1153)
            # Processing the call arguments (line 1153)
            
            # Call to ravel(...): (line 1153)
            # Processing the call keyword arguments (line 1153)
            kwargs_123589 = {}
            # Getting the type of 'self' (line 1153)
            self_123586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 46), 'self', False)
            # Obtaining the member 'x' of a type (line 1153)
            x_123587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 46), self_123586, 'x')
            # Obtaining the member 'ravel' of a type (line 1153)
            ravel_123588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 46), x_123587, 'ravel')
            # Calling ravel(args, kwargs) (line 1153)
            ravel_call_result_123590 = invoke(stypy.reporting.localization.Localization(__file__, 1153, 46), ravel_123588, *[], **kwargs_123589)
            
            
            # Call to ravel(...): (line 1153)
            # Processing the call keyword arguments (line 1153)
            kwargs_123594 = {}
            # Getting the type of 'self' (line 1153)
            self_123591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 62), 'self', False)
            # Obtaining the member 'y' of a type (line 1153)
            y_123592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 62), self_123591, 'y')
            # Obtaining the member 'ravel' of a type (line 1153)
            ravel_123593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 62), y_123592, 'ravel')
            # Calling ravel(args, kwargs) (line 1153)
            ravel_call_result_123595 = invoke(stypy.reporting.localization.Localization(__file__, 1153, 62), ravel_123593, *[], **kwargs_123594)
            
            # Getting the type of 'self' (line 1154)
            self_123596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 46), 'self', False)
            # Obtaining the member 'u' of a type (line 1154)
            u_123597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1154, 46), self_123596, 'u')
            # Getting the type of 'self' (line 1154)
            self_123598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 54), 'self', False)
            # Obtaining the member 'v' of a type (line 1154)
            v_123599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1154, 54), self_123598, 'v')
            # Processing the call keyword arguments (line 1153)
            kwargs_123600 = {}
            # Getting the type of 'delete_masked_points' (line 1153)
            delete_masked_points_123585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 25), 'delete_masked_points', False)
            # Calling delete_masked_points(args, kwargs) (line 1153)
            delete_masked_points_call_result_123601 = invoke(stypy.reporting.localization.Localization(__file__, 1153, 25), delete_masked_points_123585, *[ravel_call_result_123590, ravel_call_result_123595, u_123597, v_123599], **kwargs_123600)
            
            # Assigning a type to the variable 'call_assignment_120704' (line 1153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120704', delete_masked_points_call_result_123601)
            
            # Assigning a Call to a Name (line 1153):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_123604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 12), 'int')
            # Processing the call keyword arguments
            kwargs_123605 = {}
            # Getting the type of 'call_assignment_120704' (line 1153)
            call_assignment_120704_123602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120704', False)
            # Obtaining the member '__getitem__' of a type (line 1153)
            getitem___123603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 12), call_assignment_120704_123602, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_123606 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123603, *[int_123604], **kwargs_123605)
            
            # Assigning a type to the variable 'call_assignment_120705' (line 1153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120705', getitem___call_result_123606)
            
            # Assigning a Name to a Name (line 1153):
            # Getting the type of 'call_assignment_120705' (line 1153)
            call_assignment_120705_123607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120705')
            # Assigning a type to the variable 'x' (line 1153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'x', call_assignment_120705_123607)
            
            # Assigning a Call to a Name (line 1153):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_123610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 12), 'int')
            # Processing the call keyword arguments
            kwargs_123611 = {}
            # Getting the type of 'call_assignment_120704' (line 1153)
            call_assignment_120704_123608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120704', False)
            # Obtaining the member '__getitem__' of a type (line 1153)
            getitem___123609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 12), call_assignment_120704_123608, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_123612 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123609, *[int_123610], **kwargs_123611)
            
            # Assigning a type to the variable 'call_assignment_120706' (line 1153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120706', getitem___call_result_123612)
            
            # Assigning a Name to a Name (line 1153):
            # Getting the type of 'call_assignment_120706' (line 1153)
            call_assignment_120706_123613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120706')
            # Assigning a type to the variable 'y' (line 1153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 15), 'y', call_assignment_120706_123613)
            
            # Assigning a Call to a Name (line 1153):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_123616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 12), 'int')
            # Processing the call keyword arguments
            kwargs_123617 = {}
            # Getting the type of 'call_assignment_120704' (line 1153)
            call_assignment_120704_123614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120704', False)
            # Obtaining the member '__getitem__' of a type (line 1153)
            getitem___123615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 12), call_assignment_120704_123614, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_123618 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123615, *[int_123616], **kwargs_123617)
            
            # Assigning a type to the variable 'call_assignment_120707' (line 1153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120707', getitem___call_result_123618)
            
            # Assigning a Name to a Name (line 1153):
            # Getting the type of 'call_assignment_120707' (line 1153)
            call_assignment_120707_123619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120707')
            # Assigning a type to the variable 'u' (line 1153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 18), 'u', call_assignment_120707_123619)
            
            # Assigning a Call to a Name (line 1153):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_123622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 12), 'int')
            # Processing the call keyword arguments
            kwargs_123623 = {}
            # Getting the type of 'call_assignment_120704' (line 1153)
            call_assignment_120704_123620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120704', False)
            # Obtaining the member '__getitem__' of a type (line 1153)
            getitem___123621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 12), call_assignment_120704_123620, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_123624 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123621, *[int_123622], **kwargs_123623)
            
            # Assigning a type to the variable 'call_assignment_120708' (line 1153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120708', getitem___call_result_123624)
            
            # Assigning a Name to a Name (line 1153):
            # Getting the type of 'call_assignment_120708' (line 1153)
            call_assignment_120708_123625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 12), 'call_assignment_120708')
            # Assigning a type to the variable 'v' (line 1153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 21), 'v', call_assignment_120708_123625)
            
            # Call to _check_consistent_shapes(...): (line 1155)
            # Processing the call arguments (line 1155)
            # Getting the type of 'x' (line 1155)
            x_123627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 37), 'x', False)
            # Getting the type of 'y' (line 1155)
            y_123628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 40), 'y', False)
            # Getting the type of 'u' (line 1155)
            u_123629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 43), 'u', False)
            # Getting the type of 'v' (line 1155)
            v_123630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 46), 'v', False)
            # Processing the call keyword arguments (line 1155)
            kwargs_123631 = {}
            # Getting the type of '_check_consistent_shapes' (line 1155)
            _check_consistent_shapes_123626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 12), '_check_consistent_shapes', False)
            # Calling _check_consistent_shapes(args, kwargs) (line 1155)
            _check_consistent_shapes_call_result_123632 = invoke(stypy.reporting.localization.Localization(__file__, 1155, 12), _check_consistent_shapes_123626, *[x_123627, y_123628, u_123629, v_123630], **kwargs_123631)
            

            if (may_be_123517 and more_types_in_union_123518):
                # SSA join for if statement (line 1146)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 1157):
        
        # Assigning a Call to a Name (line 1157):
        
        # Call to hypot(...): (line 1157)
        # Processing the call arguments (line 1157)
        # Getting the type of 'u' (line 1157)
        u_123635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 29), 'u', False)
        # Getting the type of 'v' (line 1157)
        v_123636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 32), 'v', False)
        # Processing the call keyword arguments (line 1157)
        kwargs_123637 = {}
        # Getting the type of 'np' (line 1157)
        np_123633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 20), 'np', False)
        # Obtaining the member 'hypot' of a type (line 1157)
        hypot_123634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1157, 20), np_123633, 'hypot')
        # Calling hypot(args, kwargs) (line 1157)
        hypot_call_result_123638 = invoke(stypy.reporting.localization.Localization(__file__, 1157, 20), hypot_123634, *[u_123635, v_123636], **kwargs_123637)
        
        # Assigning a type to the variable 'magnitude' (line 1157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1157, 8), 'magnitude', hypot_call_result_123638)
        
        # Assigning a Call to a Tuple (line 1158):
        
        # Assigning a Call to a Name:
        
        # Call to _find_tails(...): (line 1158)
        # Processing the call arguments (line 1158)
        # Getting the type of 'magnitude' (line 1158)
        magnitude_123641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 55), 'magnitude', False)
        # Getting the type of 'self' (line 1159)
        self_123642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 55), 'self', False)
        # Obtaining the member 'rounding' of a type (line 1159)
        rounding_123643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1159, 55), self_123642, 'rounding')
        # Processing the call keyword arguments (line 1158)
        # Getting the type of 'self' (line 1160)
        self_123644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 57), 'self', False)
        # Obtaining the member 'barb_increments' of a type (line 1160)
        barb_increments_123645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1160, 57), self_123644, 'barb_increments')
        kwargs_123646 = {'barb_increments_123645': barb_increments_123645}
        # Getting the type of 'self' (line 1158)
        self_123639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 38), 'self', False)
        # Obtaining the member '_find_tails' of a type (line 1158)
        _find_tails_123640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 38), self_123639, '_find_tails')
        # Calling _find_tails(args, kwargs) (line 1158)
        _find_tails_call_result_123647 = invoke(stypy.reporting.localization.Localization(__file__, 1158, 38), _find_tails_123640, *[magnitude_123641, rounding_123643], **kwargs_123646)
        
        # Assigning a type to the variable 'call_assignment_120709' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120709', _find_tails_call_result_123647)
        
        # Assigning a Call to a Name (line 1158):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1158, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123651 = {}
        # Getting the type of 'call_assignment_120709' (line 1158)
        call_assignment_120709_123648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120709', False)
        # Obtaining the member '__getitem__' of a type (line 1158)
        getitem___123649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 8), call_assignment_120709_123648, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123652 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123649, *[int_123650], **kwargs_123651)
        
        # Assigning a type to the variable 'call_assignment_120710' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120710', getitem___call_result_123652)
        
        # Assigning a Name to a Name (line 1158):
        # Getting the type of 'call_assignment_120710' (line 1158)
        call_assignment_120710_123653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120710')
        # Assigning a type to the variable 'flags' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'flags', call_assignment_120710_123653)
        
        # Assigning a Call to a Name (line 1158):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1158, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123657 = {}
        # Getting the type of 'call_assignment_120709' (line 1158)
        call_assignment_120709_123654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120709', False)
        # Obtaining the member '__getitem__' of a type (line 1158)
        getitem___123655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 8), call_assignment_120709_123654, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123658 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123655, *[int_123656], **kwargs_123657)
        
        # Assigning a type to the variable 'call_assignment_120711' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120711', getitem___call_result_123658)
        
        # Assigning a Name to a Name (line 1158):
        # Getting the type of 'call_assignment_120711' (line 1158)
        call_assignment_120711_123659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120711')
        # Assigning a type to the variable 'barbs' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 15), 'barbs', call_assignment_120711_123659)
        
        # Assigning a Call to a Name (line 1158):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1158, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123663 = {}
        # Getting the type of 'call_assignment_120709' (line 1158)
        call_assignment_120709_123660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120709', False)
        # Obtaining the member '__getitem__' of a type (line 1158)
        getitem___123661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 8), call_assignment_120709_123660, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123664 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123661, *[int_123662], **kwargs_123663)
        
        # Assigning a type to the variable 'call_assignment_120712' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120712', getitem___call_result_123664)
        
        # Assigning a Name to a Name (line 1158):
        # Getting the type of 'call_assignment_120712' (line 1158)
        call_assignment_120712_123665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120712')
        # Assigning a type to the variable 'halves' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 22), 'halves', call_assignment_120712_123665)
        
        # Assigning a Call to a Name (line 1158):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1158, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123669 = {}
        # Getting the type of 'call_assignment_120709' (line 1158)
        call_assignment_120709_123666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120709', False)
        # Obtaining the member '__getitem__' of a type (line 1158)
        getitem___123667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 8), call_assignment_120709_123666, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123670 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123667, *[int_123668], **kwargs_123669)
        
        # Assigning a type to the variable 'call_assignment_120713' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120713', getitem___call_result_123670)
        
        # Assigning a Name to a Name (line 1158):
        # Getting the type of 'call_assignment_120713' (line 1158)
        call_assignment_120713_123671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 8), 'call_assignment_120713')
        # Assigning a type to the variable 'empty' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 30), 'empty', call_assignment_120713_123671)
        
        # Assigning a Call to a Name (line 1164):
        
        # Assigning a Call to a Name (line 1164):
        
        # Call to _make_barbs(...): (line 1164)
        # Processing the call arguments (line 1164)
        # Getting the type of 'u' (line 1164)
        u_123674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 38), 'u', False)
        # Getting the type of 'v' (line 1164)
        v_123675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 41), 'v', False)
        # Getting the type of 'flags' (line 1164)
        flags_123676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 44), 'flags', False)
        # Getting the type of 'barbs' (line 1164)
        barbs_123677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 51), 'barbs', False)
        # Getting the type of 'halves' (line 1164)
        halves_123678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 58), 'halves', False)
        # Getting the type of 'empty' (line 1164)
        empty_123679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 66), 'empty', False)
        # Getting the type of 'self' (line 1165)
        self_123680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 38), 'self', False)
        # Obtaining the member '_length' of a type (line 1165)
        _length_123681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1165, 38), self_123680, '_length')
        # Getting the type of 'self' (line 1165)
        self_123682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 52), 'self', False)
        # Obtaining the member '_pivot' of a type (line 1165)
        _pivot_123683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1165, 52), self_123682, '_pivot')
        # Getting the type of 'self' (line 1165)
        self_123684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 65), 'self', False)
        # Obtaining the member 'sizes' of a type (line 1165)
        sizes_123685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1165, 65), self_123684, 'sizes')
        # Getting the type of 'self' (line 1166)
        self_123686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 38), 'self', False)
        # Obtaining the member 'fill_empty' of a type (line 1166)
        fill_empty_123687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1166, 38), self_123686, 'fill_empty')
        # Getting the type of 'self' (line 1166)
        self_123688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 55), 'self', False)
        # Obtaining the member 'flip' of a type (line 1166)
        flip_123689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1166, 55), self_123688, 'flip')
        # Processing the call keyword arguments (line 1164)
        kwargs_123690 = {}
        # Getting the type of 'self' (line 1164)
        self_123672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 21), 'self', False)
        # Obtaining the member '_make_barbs' of a type (line 1164)
        _make_barbs_123673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1164, 21), self_123672, '_make_barbs')
        # Calling _make_barbs(args, kwargs) (line 1164)
        _make_barbs_call_result_123691 = invoke(stypy.reporting.localization.Localization(__file__, 1164, 21), _make_barbs_123673, *[u_123674, v_123675, flags_123676, barbs_123677, halves_123678, empty_123679, _length_123681, _pivot_123683, sizes_123685, fill_empty_123687, flip_123689], **kwargs_123690)
        
        # Assigning a type to the variable 'plot_barbs' (line 1164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 8), 'plot_barbs', _make_barbs_call_result_123691)
        
        # Call to set_verts(...): (line 1167)
        # Processing the call arguments (line 1167)
        # Getting the type of 'plot_barbs' (line 1167)
        plot_barbs_123694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 23), 'plot_barbs', False)
        # Processing the call keyword arguments (line 1167)
        kwargs_123695 = {}
        # Getting the type of 'self' (line 1167)
        self_123692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 8), 'self', False)
        # Obtaining the member 'set_verts' of a type (line 1167)
        set_verts_123693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1167, 8), self_123692, 'set_verts')
        # Calling set_verts(args, kwargs) (line 1167)
        set_verts_call_result_123696 = invoke(stypy.reporting.localization.Localization(__file__, 1167, 8), set_verts_123693, *[plot_barbs_123694], **kwargs_123695)
        
        
        # Type idiom detected: calculating its left and rigth part (line 1170)
        # Getting the type of 'C' (line 1170)
        C_123697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 8), 'C')
        # Getting the type of 'None' (line 1170)
        None_123698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 20), 'None')
        
        (may_be_123699, more_types_in_union_123700) = may_not_be_none(C_123697, None_123698)

        if may_be_123699:

            if more_types_in_union_123700:
                # Runtime conditional SSA (line 1170)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to set_array(...): (line 1171)
            # Processing the call arguments (line 1171)
            # Getting the type of 'c' (line 1171)
            c_123703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 27), 'c', False)
            # Processing the call keyword arguments (line 1171)
            kwargs_123704 = {}
            # Getting the type of 'self' (line 1171)
            self_123701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 12), 'self', False)
            # Obtaining the member 'set_array' of a type (line 1171)
            set_array_123702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1171, 12), self_123701, 'set_array')
            # Calling set_array(args, kwargs) (line 1171)
            set_array_call_result_123705 = invoke(stypy.reporting.localization.Localization(__file__, 1171, 12), set_array_123702, *[c_123703], **kwargs_123704)
            

            if more_types_in_union_123700:
                # SSA join for if statement (line 1170)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 1174):
        
        # Assigning a Call to a Name (line 1174):
        
        # Call to hstack(...): (line 1174)
        # Processing the call arguments (line 1174)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1174)
        tuple_123708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1174, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1174)
        # Adding element type (line 1174)
        
        # Obtaining the type of the subscript
        slice_123709 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1174, 24), None, None, None)
        # Getting the type of 'np' (line 1174)
        np_123710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 29), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 1174)
        newaxis_123711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1174, 29), np_123710, 'newaxis')
        # Getting the type of 'x' (line 1174)
        x_123712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 24), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 1174)
        getitem___123713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1174, 24), x_123712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1174)
        subscript_call_result_123714 = invoke(stypy.reporting.localization.Localization(__file__, 1174, 24), getitem___123713, (slice_123709, newaxis_123711))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1174, 24), tuple_123708, subscript_call_result_123714)
        # Adding element type (line 1174)
        
        # Obtaining the type of the subscript
        slice_123715 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1174, 42), None, None, None)
        # Getting the type of 'np' (line 1174)
        np_123716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 47), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 1174)
        newaxis_123717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1174, 47), np_123716, 'newaxis')
        # Getting the type of 'y' (line 1174)
        y_123718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 42), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 1174)
        getitem___123719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1174, 42), y_123718, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1174)
        subscript_call_result_123720 = invoke(stypy.reporting.localization.Localization(__file__, 1174, 42), getitem___123719, (slice_123715, newaxis_123717))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1174, 24), tuple_123708, subscript_call_result_123720)
        
        # Processing the call keyword arguments (line 1174)
        kwargs_123721 = {}
        # Getting the type of 'np' (line 1174)
        np_123706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 13), 'np', False)
        # Obtaining the member 'hstack' of a type (line 1174)
        hstack_123707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1174, 13), np_123706, 'hstack')
        # Calling hstack(args, kwargs) (line 1174)
        hstack_call_result_123722 = invoke(stypy.reporting.localization.Localization(__file__, 1174, 13), hstack_123707, *[tuple_123708], **kwargs_123721)
        
        # Assigning a type to the variable 'xy' (line 1174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1174, 8), 'xy', hstack_call_result_123722)
        
        # Assigning a Name to a Attribute (line 1175):
        
        # Assigning a Name to a Attribute (line 1175):
        # Getting the type of 'xy' (line 1175)
        xy_123723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 24), 'xy')
        # Getting the type of 'self' (line 1175)
        self_123724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 8), 'self')
        # Setting the type of the member '_offsets' of a type (line 1175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1175, 8), self_123724, '_offsets', xy_123723)
        
        # Assigning a Name to a Attribute (line 1176):
        
        # Assigning a Name to a Attribute (line 1176):
        # Getting the type of 'True' (line 1176)
        True_123725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 21), 'True')
        # Getting the type of 'self' (line 1176)
        self_123726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 1176)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1176, 8), self_123726, 'stale', True_123725)
        
        # ################# End of 'set_UVC(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_UVC' in the type store
        # Getting the type of 'stypy_return_type' (line 1143)
        stypy_return_type_123727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123727)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_UVC'
        return stypy_return_type_123727


    @norecursion
    def set_offsets(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_offsets'
        module_type_store = module_type_store.open_function_context('set_offsets', 1178, 4, False)
        # Assigning a type to the variable 'self' (line 1179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Barbs.set_offsets.__dict__.__setitem__('stypy_localization', localization)
        Barbs.set_offsets.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Barbs.set_offsets.__dict__.__setitem__('stypy_type_store', module_type_store)
        Barbs.set_offsets.__dict__.__setitem__('stypy_function_name', 'Barbs.set_offsets')
        Barbs.set_offsets.__dict__.__setitem__('stypy_param_names_list', ['xy'])
        Barbs.set_offsets.__dict__.__setitem__('stypy_varargs_param_name', None)
        Barbs.set_offsets.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Barbs.set_offsets.__dict__.__setitem__('stypy_call_defaults', defaults)
        Barbs.set_offsets.__dict__.__setitem__('stypy_call_varargs', varargs)
        Barbs.set_offsets.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Barbs.set_offsets.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Barbs.set_offsets', ['xy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_offsets', localization, ['xy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_offsets(...)' code ##################

        unicode_123728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1185, (-1)), 'unicode', u'\n        Set the offsets for the barb polygons.  This saves the offsets passed\n        in and actually sets version masked as appropriate for the existing\n        U/V data. *offsets* should be a sequence.\n\n        ACCEPTS: sequence of pairs of floats\n        ')
        
        # Assigning a Subscript to a Attribute (line 1186):
        
        # Assigning a Subscript to a Attribute (line 1186):
        
        # Obtaining the type of the subscript
        slice_123729 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1186, 17), None, None, None)
        int_123730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1186, 23), 'int')
        # Getting the type of 'xy' (line 1186)
        xy_123731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 17), 'xy')
        # Obtaining the member '__getitem__' of a type (line 1186)
        getitem___123732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1186, 17), xy_123731, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1186)
        subscript_call_result_123733 = invoke(stypy.reporting.localization.Localization(__file__, 1186, 17), getitem___123732, (slice_123729, int_123730))
        
        # Getting the type of 'self' (line 1186)
        self_123734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 8), 'self')
        # Setting the type of the member 'x' of a type (line 1186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1186, 8), self_123734, 'x', subscript_call_result_123733)
        
        # Assigning a Subscript to a Attribute (line 1187):
        
        # Assigning a Subscript to a Attribute (line 1187):
        
        # Obtaining the type of the subscript
        slice_123735 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1187, 17), None, None, None)
        int_123736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1187, 23), 'int')
        # Getting the type of 'xy' (line 1187)
        xy_123737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 17), 'xy')
        # Obtaining the member '__getitem__' of a type (line 1187)
        getitem___123738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1187, 17), xy_123737, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1187)
        subscript_call_result_123739 = invoke(stypy.reporting.localization.Localization(__file__, 1187, 17), getitem___123738, (slice_123735, int_123736))
        
        # Getting the type of 'self' (line 1187)
        self_123740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 8), 'self')
        # Setting the type of the member 'y' of a type (line 1187)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1187, 8), self_123740, 'y', subscript_call_result_123739)
        
        # Assigning a Call to a Tuple (line 1188):
        
        # Assigning a Call to a Name:
        
        # Call to delete_masked_points(...): (line 1188)
        # Processing the call arguments (line 1188)
        
        # Call to ravel(...): (line 1188)
        # Processing the call keyword arguments (line 1188)
        kwargs_123745 = {}
        # Getting the type of 'self' (line 1188)
        self_123742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 42), 'self', False)
        # Obtaining the member 'x' of a type (line 1188)
        x_123743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1188, 42), self_123742, 'x')
        # Obtaining the member 'ravel' of a type (line 1188)
        ravel_123744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1188, 42), x_123743, 'ravel')
        # Calling ravel(args, kwargs) (line 1188)
        ravel_call_result_123746 = invoke(stypy.reporting.localization.Localization(__file__, 1188, 42), ravel_123744, *[], **kwargs_123745)
        
        
        # Call to ravel(...): (line 1188)
        # Processing the call keyword arguments (line 1188)
        kwargs_123750 = {}
        # Getting the type of 'self' (line 1188)
        self_123747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 58), 'self', False)
        # Obtaining the member 'y' of a type (line 1188)
        y_123748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1188, 58), self_123747, 'y')
        # Obtaining the member 'ravel' of a type (line 1188)
        ravel_123749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1188, 58), y_123748, 'ravel')
        # Calling ravel(args, kwargs) (line 1188)
        ravel_call_result_123751 = invoke(stypy.reporting.localization.Localization(__file__, 1188, 58), ravel_123749, *[], **kwargs_123750)
        
        # Getting the type of 'self' (line 1189)
        self_123752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 42), 'self', False)
        # Obtaining the member 'u' of a type (line 1189)
        u_123753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1189, 42), self_123752, 'u')
        # Getting the type of 'self' (line 1189)
        self_123754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 50), 'self', False)
        # Obtaining the member 'v' of a type (line 1189)
        v_123755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1189, 50), self_123754, 'v')
        # Processing the call keyword arguments (line 1188)
        kwargs_123756 = {}
        # Getting the type of 'delete_masked_points' (line 1188)
        delete_masked_points_123741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 21), 'delete_masked_points', False)
        # Calling delete_masked_points(args, kwargs) (line 1188)
        delete_masked_points_call_result_123757 = invoke(stypy.reporting.localization.Localization(__file__, 1188, 21), delete_masked_points_123741, *[ravel_call_result_123746, ravel_call_result_123751, u_123753, v_123755], **kwargs_123756)
        
        # Assigning a type to the variable 'call_assignment_120714' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120714', delete_masked_points_call_result_123757)
        
        # Assigning a Call to a Name (line 1188):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1188, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123761 = {}
        # Getting the type of 'call_assignment_120714' (line 1188)
        call_assignment_120714_123758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120714', False)
        # Obtaining the member '__getitem__' of a type (line 1188)
        getitem___123759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1188, 8), call_assignment_120714_123758, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123762 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123759, *[int_123760], **kwargs_123761)
        
        # Assigning a type to the variable 'call_assignment_120715' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120715', getitem___call_result_123762)
        
        # Assigning a Name to a Name (line 1188):
        # Getting the type of 'call_assignment_120715' (line 1188)
        call_assignment_120715_123763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120715')
        # Assigning a type to the variable 'x' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'x', call_assignment_120715_123763)
        
        # Assigning a Call to a Name (line 1188):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1188, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123767 = {}
        # Getting the type of 'call_assignment_120714' (line 1188)
        call_assignment_120714_123764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120714', False)
        # Obtaining the member '__getitem__' of a type (line 1188)
        getitem___123765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1188, 8), call_assignment_120714_123764, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123768 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123765, *[int_123766], **kwargs_123767)
        
        # Assigning a type to the variable 'call_assignment_120716' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120716', getitem___call_result_123768)
        
        # Assigning a Name to a Name (line 1188):
        # Getting the type of 'call_assignment_120716' (line 1188)
        call_assignment_120716_123769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120716')
        # Assigning a type to the variable 'y' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 11), 'y', call_assignment_120716_123769)
        
        # Assigning a Call to a Name (line 1188):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1188, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123773 = {}
        # Getting the type of 'call_assignment_120714' (line 1188)
        call_assignment_120714_123770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120714', False)
        # Obtaining the member '__getitem__' of a type (line 1188)
        getitem___123771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1188, 8), call_assignment_120714_123770, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123774 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123771, *[int_123772], **kwargs_123773)
        
        # Assigning a type to the variable 'call_assignment_120717' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120717', getitem___call_result_123774)
        
        # Assigning a Name to a Name (line 1188):
        # Getting the type of 'call_assignment_120717' (line 1188)
        call_assignment_120717_123775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120717')
        # Assigning a type to the variable 'u' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 14), 'u', call_assignment_120717_123775)
        
        # Assigning a Call to a Name (line 1188):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_123778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1188, 8), 'int')
        # Processing the call keyword arguments
        kwargs_123779 = {}
        # Getting the type of 'call_assignment_120714' (line 1188)
        call_assignment_120714_123776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120714', False)
        # Obtaining the member '__getitem__' of a type (line 1188)
        getitem___123777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1188, 8), call_assignment_120714_123776, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_123780 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___123777, *[int_123778], **kwargs_123779)
        
        # Assigning a type to the variable 'call_assignment_120718' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120718', getitem___call_result_123780)
        
        # Assigning a Name to a Name (line 1188):
        # Getting the type of 'call_assignment_120718' (line 1188)
        call_assignment_120718_123781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'call_assignment_120718')
        # Assigning a type to the variable 'v' (line 1188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 17), 'v', call_assignment_120718_123781)
        
        # Call to _check_consistent_shapes(...): (line 1190)
        # Processing the call arguments (line 1190)
        # Getting the type of 'x' (line 1190)
        x_123783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 33), 'x', False)
        # Getting the type of 'y' (line 1190)
        y_123784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 36), 'y', False)
        # Getting the type of 'u' (line 1190)
        u_123785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 39), 'u', False)
        # Getting the type of 'v' (line 1190)
        v_123786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 42), 'v', False)
        # Processing the call keyword arguments (line 1190)
        kwargs_123787 = {}
        # Getting the type of '_check_consistent_shapes' (line 1190)
        _check_consistent_shapes_123782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 8), '_check_consistent_shapes', False)
        # Calling _check_consistent_shapes(args, kwargs) (line 1190)
        _check_consistent_shapes_call_result_123788 = invoke(stypy.reporting.localization.Localization(__file__, 1190, 8), _check_consistent_shapes_123782, *[x_123783, y_123784, u_123785, v_123786], **kwargs_123787)
        
        
        # Assigning a Call to a Name (line 1191):
        
        # Assigning a Call to a Name (line 1191):
        
        # Call to hstack(...): (line 1191)
        # Processing the call arguments (line 1191)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1191)
        tuple_123791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1191, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1191)
        # Adding element type (line 1191)
        
        # Obtaining the type of the subscript
        slice_123792 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1191, 24), None, None, None)
        # Getting the type of 'np' (line 1191)
        np_123793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 29), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 1191)
        newaxis_123794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1191, 29), np_123793, 'newaxis')
        # Getting the type of 'x' (line 1191)
        x_123795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 24), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 1191)
        getitem___123796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1191, 24), x_123795, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1191)
        subscript_call_result_123797 = invoke(stypy.reporting.localization.Localization(__file__, 1191, 24), getitem___123796, (slice_123792, newaxis_123794))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 24), tuple_123791, subscript_call_result_123797)
        # Adding element type (line 1191)
        
        # Obtaining the type of the subscript
        slice_123798 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1191, 42), None, None, None)
        # Getting the type of 'np' (line 1191)
        np_123799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 47), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 1191)
        newaxis_123800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1191, 47), np_123799, 'newaxis')
        # Getting the type of 'y' (line 1191)
        y_123801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 42), 'y', False)
        # Obtaining the member '__getitem__' of a type (line 1191)
        getitem___123802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1191, 42), y_123801, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1191)
        subscript_call_result_123803 = invoke(stypy.reporting.localization.Localization(__file__, 1191, 42), getitem___123802, (slice_123798, newaxis_123800))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1191, 24), tuple_123791, subscript_call_result_123803)
        
        # Processing the call keyword arguments (line 1191)
        kwargs_123804 = {}
        # Getting the type of 'np' (line 1191)
        np_123789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 13), 'np', False)
        # Obtaining the member 'hstack' of a type (line 1191)
        hstack_123790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1191, 13), np_123789, 'hstack')
        # Calling hstack(args, kwargs) (line 1191)
        hstack_call_result_123805 = invoke(stypy.reporting.localization.Localization(__file__, 1191, 13), hstack_123790, *[tuple_123791], **kwargs_123804)
        
        # Assigning a type to the variable 'xy' (line 1191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1191, 8), 'xy', hstack_call_result_123805)
        
        # Call to set_offsets(...): (line 1192)
        # Processing the call arguments (line 1192)
        # Getting the type of 'self' (line 1192)
        self_123809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 48), 'self', False)
        # Getting the type of 'xy' (line 1192)
        xy_123810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 54), 'xy', False)
        # Processing the call keyword arguments (line 1192)
        kwargs_123811 = {}
        # Getting the type of 'mcollections' (line 1192)
        mcollections_123806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 8), 'mcollections', False)
        # Obtaining the member 'PolyCollection' of a type (line 1192)
        PolyCollection_123807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1192, 8), mcollections_123806, 'PolyCollection')
        # Obtaining the member 'set_offsets' of a type (line 1192)
        set_offsets_123808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1192, 8), PolyCollection_123807, 'set_offsets')
        # Calling set_offsets(args, kwargs) (line 1192)
        set_offsets_call_result_123812 = invoke(stypy.reporting.localization.Localization(__file__, 1192, 8), set_offsets_123808, *[self_123809, xy_123810], **kwargs_123811)
        
        
        # Assigning a Name to a Attribute (line 1193):
        
        # Assigning a Name to a Attribute (line 1193):
        # Getting the type of 'True' (line 1193)
        True_123813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 21), 'True')
        # Getting the type of 'self' (line 1193)
        self_123814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 1193)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1193, 8), self_123814, 'stale', True_123813)
        
        # ################# End of 'set_offsets(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_offsets' in the type store
        # Getting the type of 'stypy_return_type' (line 1178)
        stypy_return_type_123815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123815)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_offsets'
        return stypy_return_type_123815

    
    # Assigning a Attribute to a Attribute (line 1195):
    
    # Assigning a Name to a Name (line 1197):

# Assigning a type to the variable 'Barbs' (line 892)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 0), 'Barbs', Barbs)

# Assigning a Attribute to a Attribute (line 1195):
# Getting the type of 'mcollections' (line 1195)
mcollections_123816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 26), 'mcollections')
# Obtaining the member 'PolyCollection' of a type (line 1195)
PolyCollection_123817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1195, 26), mcollections_123816, 'PolyCollection')
# Obtaining the member 'set_offsets' of a type (line 1195)
set_offsets_123818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1195, 26), PolyCollection_123817, 'set_offsets')
# Obtaining the member '__doc__' of a type (line 1195)
doc___123819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1195, 26), set_offsets_123818, '__doc__')
# Getting the type of 'Barbs'
Barbs_123820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Barbs')
# Obtaining the member 'set_offsets' of a type
set_offsets_123821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Barbs_123820, 'set_offsets')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), set_offsets_123821, '__doc__', doc___123819)

# Assigning a Name to a Name (line 1197):
# Getting the type of '_barbs_doc' (line 1197)
_barbs_doc_123822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 16), '_barbs_doc')
# Getting the type of 'Barbs'
Barbs_123823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Barbs')
# Setting the type of the member 'barbs_doc' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Barbs_123823, 'barbs_doc', _barbs_doc_123822)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
