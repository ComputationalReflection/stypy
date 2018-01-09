
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: from collections import OrderedDict, namedtuple
7: from functools import wraps
8: import inspect
9: import re
10: import warnings
11: 
12: import numpy as np
13: 
14: import matplotlib
15: from . import cbook, docstring, rcParams
16: from .path import Path
17: from .transforms import (Bbox, IdentityTransform, Transform, TransformedBbox,
18:                          TransformedPatchPath, TransformedPath)
19: # Note, matplotlib artists use the doc strings for set and get
20: # methods to enable the introspection methods of setp and getp.  Every
21: # set_* method should have a docstring containing the line
22: #
23: # ACCEPTS: [ legal | values ]
24: #
25: # and aliases for setters and getters should have a docstring that
26: # starts with 'alias for ', as in 'alias for set_somemethod'
27: #
28: # You may wonder why we use so much boiler-plate manually defining the
29: # set_alias and get_alias functions, rather than using some clever
30: # python trick.  The answer is that I need to be able to manipulate
31: # the docstring, and there is no clever way to do that in python 2.2,
32: # as far as I can see - see
33: #
34: # https://mail.python.org/pipermail/python-list/2004-October/242925.html
35: 
36: 
37: def allow_rasterization(draw):
38:     '''
39:     Decorator for Artist.draw method. Provides routines
40:     that run before and after the draw call. The before and after functions
41:     are useful for changing artist-dependent renderer attributes or making
42:     other setup function calls, such as starting and flushing a mixed-mode
43:     renderer.
44:     '''
45: 
46:     # the axes class has a second argument inframe for its draw method.
47:     @wraps(draw)
48:     def draw_wrapper(artist, renderer, *args, **kwargs):
49:         try:
50:             if artist.get_rasterized():
51:                 renderer.start_rasterizing()
52:             if artist.get_agg_filter() is not None:
53:                 renderer.start_filter()
54: 
55:             return draw(artist, renderer, *args, **kwargs)
56:         finally:
57:             if artist.get_agg_filter() is not None:
58:                 renderer.stop_filter(artist.get_agg_filter())
59:             if artist.get_rasterized():
60:                 renderer.stop_rasterizing()
61: 
62:     draw_wrapper._supports_rasterization = True
63:     return draw_wrapper
64: 
65: 
66: def _stale_axes_callback(self, val):
67:     if self.axes:
68:         self.axes.stale = val
69: 
70: 
71: _XYPair = namedtuple("_XYPair", "x y")
72: 
73: 
74: class Artist(object):
75:     '''
76:     Abstract base class for someone who renders into a
77:     :class:`FigureCanvas`.
78:     '''
79: 
80:     aname = 'Artist'
81:     zorder = 0
82:     # order of precedence when bulk setting/updating properties
83:     # via update.  The keys should be property names and the values
84:     # integers
85:     _prop_order = dict(color=-1)
86: 
87:     def __init__(self):
88:         self._stale = True
89:         self.stale_callback = None
90:         self._axes = None
91:         self.figure = None
92: 
93:         self._transform = None
94:         self._transformSet = False
95:         self._visible = True
96:         self._animated = False
97:         self._alpha = None
98:         self.clipbox = None
99:         self._clippath = None
100:         self._clipon = True
101:         self._label = ''
102:         self._picker = None
103:         self._contains = None
104:         self._rasterized = None
105:         self._agg_filter = None
106:         self._mouseover = False
107:         self.eventson = False  # fire events only if eventson
108:         self._oid = 0  # an observer id
109:         self._propobservers = {}  # a dict from oids to funcs
110:         try:
111:             self.axes = None
112:         except AttributeError:
113:             # Handle self.axes as a read-only property, as in Figure.
114:             pass
115:         self._remove_method = None
116:         self._url = None
117:         self._gid = None
118:         self._snap = None
119:         self._sketch = rcParams['path.sketch']
120:         self._path_effects = rcParams['path.effects']
121:         self._sticky_edges = _XYPair([], [])
122: 
123:     def __getstate__(self):
124:         d = self.__dict__.copy()
125:         # remove the unpicklable remove method, this will get re-added on load
126:         # (by the axes) if the artist lives on an axes.
127:         d['_remove_method'] = None
128:         d['stale_callback'] = None
129:         return d
130: 
131:     def remove(self):
132:         '''
133:         Remove the artist from the figure if possible.  The effect
134:         will not be visible until the figure is redrawn, e.g., with
135:         :meth:`matplotlib.axes.Axes.draw_idle`.  Call
136:         :meth:`matplotlib.axes.Axes.relim` to update the axes limits
137:         if desired.
138: 
139:         Note: :meth:`~matplotlib.axes.Axes.relim` will not see
140:         collections even if the collection was added to axes with
141:         *autolim* = True.
142: 
143:         Note: there is no support for removing the artist's legend entry.
144:         '''
145: 
146:         # There is no method to set the callback.  Instead the parent should
147:         # set the _remove_method attribute directly.  This would be a
148:         # protected attribute if Python supported that sort of thing.  The
149:         # callback has one parameter, which is the child to be removed.
150:         if self._remove_method is not None:
151:             self._remove_method(self)
152:             # clear stale callback
153:             self.stale_callback = None
154:             _ax_flag = False
155:             if hasattr(self, 'axes') and self.axes:
156:                 # remove from the mouse hit list
157:                 self.axes.mouseover_set.discard(self)
158:                 # mark the axes as stale
159:                 self.axes.stale = True
160:                 # decouple the artist from the axes
161:                 self.axes = None
162:                 _ax_flag = True
163: 
164:             if self.figure:
165:                 self.figure = None
166:                 if not _ax_flag:
167:                     self.figure = True
168: 
169:         else:
170:             raise NotImplementedError('cannot remove artist')
171:         # TODO: the fix for the collections relim problem is to move the
172:         # limits calculation into the artist itself, including the property of
173:         # whether or not the artist should affect the limits.  Then there will
174:         # be no distinction between axes.add_line, axes.add_patch, etc.
175:         # TODO: add legend support
176: 
177:     def have_units(self):
178:         'Return *True* if units are set on the *x* or *y* axes'
179:         ax = self.axes
180:         if ax is None or ax.xaxis is None:
181:             return False
182:         return ax.xaxis.have_units() or ax.yaxis.have_units()
183: 
184:     def convert_xunits(self, x):
185:         '''For artists in an axes, if the xaxis has units support,
186:         convert *x* using xaxis unit type
187:         '''
188:         ax = getattr(self, 'axes', None)
189:         if ax is None or ax.xaxis is None:
190:             return x
191:         return ax.xaxis.convert_units(x)
192: 
193:     def convert_yunits(self, y):
194:         '''For artists in an axes, if the yaxis has units support,
195:         convert *y* using yaxis unit type
196:         '''
197:         ax = getattr(self, 'axes', None)
198:         if ax is None or ax.yaxis is None:
199:             return y
200:         return ax.yaxis.convert_units(y)
201: 
202:     @property
203:     def axes(self):
204:         '''
205:         The :class:`~matplotlib.axes.Axes` instance the artist
206:         resides in, or *None*.
207:         '''
208:         return self._axes
209: 
210:     @axes.setter
211:     def axes(self, new_axes):
212:         if (new_axes is not None and self._axes is not None
213:                 and new_axes != self._axes):
214:             raise ValueError("Can not reset the axes.  You are probably "
215:                              "trying to re-use an artist in more than one "
216:                              "Axes which is not supported")
217:         self._axes = new_axes
218:         if new_axes is not None and new_axes is not self:
219:             self.stale_callback = _stale_axes_callback
220:         return new_axes
221: 
222:     @property
223:     def stale(self):
224:         '''
225:         If the artist is 'stale' and needs to be re-drawn for the output to
226:         match the internal state of the artist.
227:         '''
228:         return self._stale
229: 
230:     @stale.setter
231:     def stale(self, val):
232:         self._stale = val
233: 
234:         # if the artist is animated it does not take normal part in the
235:         # draw stack and is not expected to be drawn as part of the normal
236:         # draw loop (when not saving) so do not propagate this change
237:         if self.get_animated():
238:             return
239: 
240:         if val and self.stale_callback is not None:
241:             self.stale_callback(self, val)
242: 
243:     def get_window_extent(self, renderer):
244:         '''
245:         Get the axes bounding box in display space.
246:         Subclasses should override for inclusion in the bounding box
247:         "tight" calculation. Default is to return an empty bounding
248:         box at 0, 0.
249: 
250:         Be careful when using this function, the results will not update
251:         if the artist window extent of the artist changes.  The extent
252:         can change due to any changes in the transform stack, such as
253:         changing the axes limits, the figure size, or the canvas used
254:         (as is done when saving a figure).  This can lead to unexpected
255:         behavior where interactive figures will look fine on the screen,
256:         but will save incorrectly.
257:         '''
258:         return Bbox([[0, 0], [0, 0]])
259: 
260:     def add_callback(self, func):
261:         '''
262:         Adds a callback function that will be called whenever one of
263:         the :class:`Artist`'s properties changes.
264: 
265:         Returns an *id* that is useful for removing the callback with
266:         :meth:`remove_callback` later.
267:         '''
268:         oid = self._oid
269:         self._propobservers[oid] = func
270:         self._oid += 1
271:         return oid
272: 
273:     def remove_callback(self, oid):
274:         '''
275:         Remove a callback based on its *id*.
276: 
277:         .. seealso::
278: 
279:             :meth:`add_callback`
280:                For adding callbacks
281: 
282:         '''
283:         try:
284:             del self._propobservers[oid]
285:         except KeyError:
286:             pass
287: 
288:     def pchanged(self):
289:         '''
290:         Fire an event when property changed, calling all of the
291:         registered callbacks.
292:         '''
293:         for oid, func in six.iteritems(self._propobservers):
294:             func(self)
295: 
296:     def is_transform_set(self):
297:         '''
298:         Returns *True* if :class:`Artist` has a transform explicitly
299:         set.
300:         '''
301:         return self._transformSet
302: 
303:     def set_transform(self, t):
304:         '''
305:         Set the :class:`~matplotlib.transforms.Transform` instance
306:         used by this artist.
307: 
308:         ACCEPTS: :class:`~matplotlib.transforms.Transform` instance
309:         '''
310:         self._transform = t
311:         self._transformSet = True
312:         self.pchanged()
313:         self.stale = True
314: 
315:     def get_transform(self):
316:         '''
317:         Return the :class:`~matplotlib.transforms.Transform`
318:         instance used by this artist.
319:         '''
320:         if self._transform is None:
321:             self._transform = IdentityTransform()
322:         elif (not isinstance(self._transform, Transform)
323:               and hasattr(self._transform, '_as_mpl_transform')):
324:             self._transform = self._transform._as_mpl_transform(self.axes)
325:         return self._transform
326: 
327:     def hitlist(self, event):
328:         '''
329:         List the children of the artist which contain the mouse event *event*.
330:         '''
331:         L = []
332:         try:
333:             hascursor, info = self.contains(event)
334:             if hascursor:
335:                 L.append(self)
336:         except:
337:             import traceback
338:             traceback.print_exc()
339:             print("while checking", self.__class__)
340: 
341:         for a in self.get_children():
342:             L.extend(a.hitlist(event))
343:         return L
344: 
345:     def get_children(self):
346:         '''
347:         Return a list of the child :class:`Artist`s this
348:         :class:`Artist` contains.
349:         '''
350:         return []
351: 
352:     def contains(self, mouseevent):
353:         '''Test whether the artist contains the mouse event.
354: 
355:         Returns the truth value and a dictionary of artist specific details of
356:         selection, such as which points are contained in the pick radius.  See
357:         individual artists for details.
358:         '''
359:         if callable(self._contains):
360:             return self._contains(self, mouseevent)
361:         warnings.warn("'%s' needs 'contains' method" % self.__class__.__name__)
362:         return False, {}
363: 
364:     def set_contains(self, picker):
365:         '''
366:         Replace the contains test used by this artist. The new picker
367:         should be a callable function which determines whether the
368:         artist is hit by the mouse event::
369: 
370:             hit, props = picker(artist, mouseevent)
371: 
372:         If the mouse event is over the artist, return *hit* = *True*
373:         and *props* is a dictionary of properties you want returned
374:         with the contains test.
375: 
376:         ACCEPTS: a callable function
377:         '''
378:         self._contains = picker
379: 
380:     def get_contains(self):
381:         '''
382:         Return the _contains test used by the artist, or *None* for default.
383:         '''
384:         return self._contains
385: 
386:     def pickable(self):
387:         'Return *True* if :class:`Artist` is pickable.'
388:         return (self.figure is not None and
389:                 self.figure.canvas is not None and
390:                 self._picker is not None)
391: 
392:     def pick(self, mouseevent):
393:         '''
394:         Process pick event
395: 
396:         each child artist will fire a pick event if *mouseevent* is over
397:         the artist and the artist has picker set
398:         '''
399:         # Pick self
400:         if self.pickable():
401:             picker = self.get_picker()
402:             if callable(picker):
403:                 inside, prop = picker(self, mouseevent)
404:             else:
405:                 inside, prop = self.contains(mouseevent)
406:             if inside:
407:                 self.figure.canvas.pick_event(mouseevent, self, **prop)
408: 
409:         # Pick children
410:         for a in self.get_children():
411:             # make sure the event happened in the same axes
412:             ax = getattr(a, 'axes', None)
413:             if (mouseevent.inaxes is None or ax is None
414:                     or mouseevent.inaxes == ax):
415:                 # we need to check if mouseevent.inaxes is None
416:                 # because some objects associated with an axes (e.g., a
417:                 # tick label) can be outside the bounding box of the
418:                 # axes and inaxes will be None
419:                 # also check that ax is None so that it traverse objects
420:                 # which do no have an axes property but children might
421:                 a.pick(mouseevent)
422: 
423:     def set_picker(self, picker):
424:         '''
425:         Set the epsilon for picking used by this artist
426: 
427:         *picker* can be one of the following:
428: 
429:           * *None*: picking is disabled for this artist (default)
430: 
431:           * A boolean: if *True* then picking will be enabled and the
432:             artist will fire a pick event if the mouse event is over
433:             the artist
434: 
435:           * A float: if picker is a number it is interpreted as an
436:             epsilon tolerance in points and the artist will fire
437:             off an event if it's data is within epsilon of the mouse
438:             event.  For some artists like lines and patch collections,
439:             the artist may provide additional data to the pick event
440:             that is generated, e.g., the indices of the data within
441:             epsilon of the pick event
442: 
443:           * A function: if picker is callable, it is a user supplied
444:             function which determines whether the artist is hit by the
445:             mouse event::
446: 
447:               hit, props = picker(artist, mouseevent)
448: 
449:             to determine the hit test.  if the mouse event is over the
450:             artist, return *hit=True* and props is a dictionary of
451:             properties you want added to the PickEvent attributes.
452: 
453:         ACCEPTS: [None|float|boolean|callable]
454:         '''
455:         self._picker = picker
456: 
457:     def get_picker(self):
458:         'Return the picker object used by this artist'
459:         return self._picker
460: 
461:     def is_figure_set(self):
462:         '''
463:         Returns True if the artist is assigned to a
464:         :class:`~matplotlib.figure.Figure`.
465:         '''
466:         return self.figure is not None
467: 
468:     def get_url(self):
469:         '''
470:         Returns the url
471:         '''
472:         return self._url
473: 
474:     def set_url(self, url):
475:         '''
476:         Sets the url for the artist
477: 
478:         ACCEPTS: a url string
479:         '''
480:         self._url = url
481: 
482:     def get_gid(self):
483:         '''
484:         Returns the group id
485:         '''
486:         return self._gid
487: 
488:     def set_gid(self, gid):
489:         '''
490:         Sets the (group) id for the artist
491: 
492:         ACCEPTS: an id string
493:         '''
494:         self._gid = gid
495: 
496:     def get_snap(self):
497:         '''
498:         Returns the snap setting which may be:
499: 
500:           * True: snap vertices to the nearest pixel center
501: 
502:           * False: leave vertices as-is
503: 
504:           * None: (auto) If the path contains only rectilinear line
505:             segments, round to the nearest pixel center
506: 
507:         Only supported by the Agg and MacOSX backends.
508:         '''
509:         if rcParams['path.snap']:
510:             return self._snap
511:         else:
512:             return False
513: 
514:     def set_snap(self, snap):
515:         '''
516:         Sets the snap setting which may be:
517: 
518:           * True: snap vertices to the nearest pixel center
519: 
520:           * False: leave vertices as-is
521: 
522:           * None: (auto) If the path contains only rectilinear line
523:             segments, round to the nearest pixel center
524: 
525:         Only supported by the Agg and MacOSX backends.
526:         '''
527:         self._snap = snap
528:         self.stale = True
529: 
530:     def get_sketch_params(self):
531:         '''
532:         Returns the sketch parameters for the artist.
533: 
534:         Returns
535:         -------
536:         sketch_params : tuple or `None`
537: 
538:         A 3-tuple with the following elements:
539: 
540:           * `scale`: The amplitude of the wiggle perpendicular to the
541:             source line.
542: 
543:           * `length`: The length of the wiggle along the line.
544: 
545:           * `randomness`: The scale factor by which the length is
546:             shrunken or expanded.
547: 
548:         May return `None` if no sketch parameters were set.
549:         '''
550:         return self._sketch
551: 
552:     def set_sketch_params(self, scale=None, length=None, randomness=None):
553:         '''
554:         Sets the sketch parameters.
555: 
556:         Parameters
557:         ----------
558: 
559:         scale : float, optional
560:             The amplitude of the wiggle perpendicular to the source
561:             line, in pixels.  If scale is `None`, or not provided, no
562:             sketch filter will be provided.
563: 
564:         length : float, optional
565:              The length of the wiggle along the line, in pixels
566:              (default 128.0)
567: 
568:         randomness : float, optional
569:             The scale factor by which the length is shrunken or
570:             expanded (default 16.0)
571:         '''
572:         if scale is None:
573:             self._sketch = None
574:         else:
575:             self._sketch = (scale, length or 128.0, randomness or 16.0)
576:         self.stale = True
577: 
578:     def set_path_effects(self, path_effects):
579:         '''
580:         set path_effects, which should be a list of instances of
581:         matplotlib.patheffect._Base class or its derivatives.
582:         '''
583:         self._path_effects = path_effects
584:         self.stale = True
585: 
586:     def get_path_effects(self):
587:         return self._path_effects
588: 
589:     def get_figure(self):
590:         '''
591:         Return the :class:`~matplotlib.figure.Figure` instance the
592:         artist belongs to.
593:         '''
594:         return self.figure
595: 
596:     def set_figure(self, fig):
597:         '''
598:         Set the :class:`~matplotlib.figure.Figure` instance the artist
599:         belongs to.
600: 
601:         ACCEPTS: a :class:`matplotlib.figure.Figure` instance
602:         '''
603:         # if this is a no-op just return
604:         if self.figure is fig:
605:             return
606:         # if we currently have a figure (the case of both `self.figure`
607:         # and `fig` being none is taken care of above) we then user is
608:         # trying to change the figure an artist is associated with which
609:         # is not allowed for the same reason as adding the same instance
610:         # to more than one Axes
611:         if self.figure is not None:
612:             raise RuntimeError("Can not put single artist in "
613:                                "more than one figure")
614:         self.figure = fig
615:         if self.figure and self.figure is not self:
616:             self.pchanged()
617:         self.stale = True
618: 
619:     def set_clip_box(self, clipbox):
620:         '''
621:         Set the artist's clip :class:`~matplotlib.transforms.Bbox`.
622: 
623:         ACCEPTS: a :class:`matplotlib.transforms.Bbox` instance
624:         '''
625:         self.clipbox = clipbox
626:         self.pchanged()
627:         self.stale = True
628: 
629:     def set_clip_path(self, path, transform=None):
630:         '''
631:         Set the artist's clip path, which may be:
632: 
633:         - a :class:`~matplotlib.patches.Patch` (or subclass) instance; or
634:         - a :class:`~matplotlib.path.Path` instance, in which case a
635:           :class:`~matplotlib.transforms.Transform` instance, which will be
636:           applied to the path before using it for clipping, must be provided;
637:           or
638:         - ``None``, to remove a previously set clipping path.
639: 
640:         For efficiency, if the path happens to be an axis-aligned rectangle,
641:         this method will set the clipping box to the corresponding rectangle
642:         and set the clipping path to ``None``.
643: 
644:         ACCEPTS: [ (:class:`~matplotlib.path.Path`,
645:         :class:`~matplotlib.transforms.Transform`) |
646:         :class:`~matplotlib.patches.Patch` | None ]
647:         '''
648:         from matplotlib.patches import Patch, Rectangle
649: 
650:         success = False
651:         if transform is None:
652:             if isinstance(path, Rectangle):
653:                 self.clipbox = TransformedBbox(Bbox.unit(),
654:                                                path.get_transform())
655:                 self._clippath = None
656:                 success = True
657:             elif isinstance(path, Patch):
658:                 self._clippath = TransformedPatchPath(path)
659:                 success = True
660:             elif isinstance(path, tuple):
661:                 path, transform = path
662: 
663:         if path is None:
664:             self._clippath = None
665:             success = True
666:         elif isinstance(path, Path):
667:             self._clippath = TransformedPath(path, transform)
668:             success = True
669:         elif isinstance(path, TransformedPatchPath):
670:             self._clippath = path
671:             success = True
672:         elif isinstance(path, TransformedPath):
673:             self._clippath = path
674:             success = True
675: 
676:         if not success:
677:             raise TypeError(
678:                 "Invalid arguments to set_clip_path, of type {} and {}"
679:                 .format(type(path).__name__, type(transform).__name__))
680:         # This may result in the callbacks being hit twice, but guarantees they
681:         # will be hit at least once.
682:         self.pchanged()
683:         self.stale = True
684: 
685:     def get_alpha(self):
686:         '''
687:         Return the alpha value used for blending - not supported on all
688:         backends
689:         '''
690:         return self._alpha
691: 
692:     def get_visible(self):
693:         "Return the artist's visiblity"
694:         return self._visible
695: 
696:     def get_animated(self):
697:         "Return the artist's animated state"
698:         return self._animated
699: 
700:     def get_clip_on(self):
701:         'Return whether artist uses clipping'
702:         return self._clipon
703: 
704:     def get_clip_box(self):
705:         'Return artist clipbox'
706:         return self.clipbox
707: 
708:     def get_clip_path(self):
709:         'Return artist clip path'
710:         return self._clippath
711: 
712:     def get_transformed_clip_path_and_affine(self):
713:         '''
714:         Return the clip path with the non-affine part of its
715:         transformation applied, and the remaining affine part of its
716:         transformation.
717:         '''
718:         if self._clippath is not None:
719:             return self._clippath.get_transformed_path_and_affine()
720:         return None, None
721: 
722:     def set_clip_on(self, b):
723:         '''
724:         Set whether artist uses clipping.
725: 
726:         When False artists will be visible out side of the axes which
727:         can lead to unexpected results.
728: 
729:         ACCEPTS: [True | False]
730:         '''
731:         self._clipon = b
732:         # This may result in the callbacks being hit twice, but ensures they
733:         # are hit at least once
734:         self.pchanged()
735:         self.stale = True
736: 
737:     def _set_gc_clip(self, gc):
738:         'Set the clip properly for the gc'
739:         if self._clipon:
740:             if self.clipbox is not None:
741:                 gc.set_clip_rectangle(self.clipbox)
742:             gc.set_clip_path(self._clippath)
743:         else:
744:             gc.set_clip_rectangle(None)
745:             gc.set_clip_path(None)
746: 
747:     def get_rasterized(self):
748:         "return True if the artist is to be rasterized"
749:         return self._rasterized
750: 
751:     def set_rasterized(self, rasterized):
752:         '''
753:         Force rasterized (bitmap) drawing in vector backend output.
754: 
755:         Defaults to None, which implies the backend's default behavior
756: 
757:         ACCEPTS: [True | False | None]
758:         '''
759:         if rasterized and not hasattr(self.draw, "_supports_rasterization"):
760:             warnings.warn("Rasterization of '%s' will be ignored" % self)
761: 
762:         self._rasterized = rasterized
763: 
764:     def get_agg_filter(self):
765:         "return filter function to be used for agg filter"
766:         return self._agg_filter
767: 
768:     def set_agg_filter(self, filter_func):
769:         '''
770:         set agg_filter function.
771: 
772:         '''
773:         self._agg_filter = filter_func
774:         self.stale = True
775: 
776:     def draw(self, renderer, *args, **kwargs):
777:         'Derived classes drawing method'
778:         if not self.get_visible():
779:             return
780:         self.stale = False
781: 
782:     def set_alpha(self, alpha):
783:         '''
784:         Set the alpha value used for blending - not supported on
785:         all backends.
786: 
787:         ACCEPTS: float (0.0 transparent through 1.0 opaque)
788:         '''
789:         self._alpha = alpha
790:         self.pchanged()
791:         self.stale = True
792: 
793:     def set_visible(self, b):
794:         '''
795:         Set the artist's visiblity.
796: 
797:         ACCEPTS: [True | False]
798:         '''
799:         self._visible = b
800:         self.pchanged()
801:         self.stale = True
802: 
803:     def set_animated(self, b):
804:         '''
805:         Set the artist's animation state.
806: 
807:         ACCEPTS: [True | False]
808:         '''
809:         if self._animated != b:
810:             self._animated = b
811:             self.pchanged()
812: 
813:     def update(self, props):
814:         '''
815:         Update the properties of this :class:`Artist` from the
816:         dictionary *prop*.
817:         '''
818:         def _update_property(self, k, v):
819:             '''sorting out how to update property (setter or setattr)
820: 
821:             Parameters
822:             ----------
823:             k : str
824:                 The name of property to update
825:             v : obj
826:                 The value to assign to the property
827:             Returns
828:             -------
829:             ret : obj or None
830:                 If using a `set_*` method return it's return, else None.
831:             '''
832:             k = k.lower()
833:             # white list attributes we want to be able to update through
834:             # art.update, art.set, setp
835:             if k in {'axes'}:
836:                 return setattr(self, k, v)
837:             else:
838:                 func = getattr(self, 'set_' + k, None)
839:                 if not callable(func):
840:                     raise AttributeError('Unknown property %s' % k)
841:                 return func(v)
842: 
843:         store = self.eventson
844:         self.eventson = False
845:         try:
846:             ret = [_update_property(self, k, v)
847:                    for k, v in props.items()]
848:         finally:
849:             self.eventson = store
850: 
851:         if len(ret):
852:             self.pchanged()
853:             self.stale = True
854:         return ret
855: 
856:     def get_label(self):
857:         '''
858:         Get the label used for this artist in the legend.
859:         '''
860:         return self._label
861: 
862:     def set_label(self, s):
863:         '''
864:         Set the label to *s* for auto legend.
865: 
866:         ACCEPTS: string or anything printable with '%s' conversion.
867:         '''
868:         if s is not None:
869:             self._label = '%s' % (s, )
870:         else:
871:             self._label = None
872:         self.pchanged()
873:         self.stale = True
874: 
875:     def get_zorder(self):
876:         '''
877:         Return the :class:`Artist`'s zorder.
878:         '''
879:         return self.zorder
880: 
881:     def set_zorder(self, level):
882:         '''
883:         Set the zorder for the artist.  Artists with lower zorder
884:         values are drawn first.
885: 
886:         ACCEPTS: any number
887:         '''
888:         self.zorder = level
889:         self.pchanged()
890:         self.stale = True
891: 
892:     @property
893:     def sticky_edges(self):
894:         '''
895:         `x` and `y` sticky edge lists.
896: 
897:         When performing autoscaling, if a data limit coincides with a value in
898:         the corresponding sticky_edges list, then no margin will be added--the
899:         view limit "sticks" to the edge. A typical usecase is histograms,
900:         where one usually expects no margin on the bottom edge (0) of the
901:         histogram.
902: 
903:         This attribute cannot be assigned to; however, the `x` and `y` lists
904:         can be modified in place as needed.
905: 
906:         Examples
907:         --------
908: 
909:         >>> artist.sticky_edges.x[:] = (xmin, xmax)
910:         >>> artist.sticky_edges.y[:] = (ymin, ymax)
911: 
912:         '''
913:         return self._sticky_edges
914: 
915:     def update_from(self, other):
916:         'Copy properties from *other* to *self*.'
917:         self._transform = other._transform
918:         self._transformSet = other._transformSet
919:         self._visible = other._visible
920:         self._alpha = other._alpha
921:         self.clipbox = other.clipbox
922:         self._clipon = other._clipon
923:         self._clippath = other._clippath
924:         self._label = other._label
925:         self._sketch = other._sketch
926:         self._path_effects = other._path_effects
927:         self.sticky_edges.x[:] = other.sticky_edges.x[:]
928:         self.sticky_edges.y[:] = other.sticky_edges.y[:]
929:         self.pchanged()
930:         self.stale = True
931: 
932:     def properties(self):
933:         '''
934:         return a dictionary mapping property name -> value for all Artist props
935:         '''
936:         return ArtistInspector(self).properties()
937: 
938:     def set(self, **kwargs):
939:         '''A property batch setter. Pass *kwargs* to set properties.
940:         '''
941:         props = OrderedDict(
942:             sorted(kwargs.items(), reverse=True,
943:                    key=lambda x: (self._prop_order.get(x[0], 0), x[0])))
944: 
945:         return self.update(props)
946: 
947:     def findobj(self, match=None, include_self=True):
948:         '''
949:         Find artist objects.
950: 
951:         Recursively find all :class:`~matplotlib.artist.Artist` instances
952:         contained in self.
953: 
954:         *match* can be
955: 
956:           - None: return all objects contained in artist.
957: 
958:           - function with signature ``boolean = match(artist)``
959:             used to filter matches
960: 
961:           - class instance: e.g., Line2D.  Only return artists of class type.
962: 
963:         If *include_self* is True (default), include self in the list to be
964:         checked for a match.
965: 
966:         '''
967:         if match is None:  # always return True
968:             def matchfunc(x):
969:                 return True
970:         elif isinstance(match, type) and issubclass(match, Artist):
971:             def matchfunc(x):
972:                 return isinstance(x, match)
973:         elif callable(match):
974:             matchfunc = match
975:         else:
976:             raise ValueError('match must be None, a matplotlib.artist.Artist '
977:                              'subclass, or a callable')
978: 
979:         artists = sum([c.findobj(matchfunc) for c in self.get_children()], [])
980:         if include_self and matchfunc(self):
981:             artists.append(self)
982:         return artists
983: 
984:     def get_cursor_data(self, event):
985:         '''
986:         Get the cursor data for a given event.
987:         '''
988:         return None
989: 
990:     def format_cursor_data(self, data):
991:         '''
992:         Return *cursor data* string formatted.
993:         '''
994:         try:
995:             data[0]
996:         except (TypeError, IndexError):
997:             data = [data]
998:         return ', '.join('{:0.3g}'.format(item) for item in data if
999:                 isinstance(item, (np.floating, np.integer, int, float)))
1000: 
1001:     @property
1002:     def mouseover(self):
1003:         return self._mouseover
1004: 
1005:     @mouseover.setter
1006:     def mouseover(self, val):
1007:         val = bool(val)
1008:         self._mouseover = val
1009:         ax = self.axes
1010:         if ax:
1011:             if val:
1012:                 ax.mouseover_set.add(self)
1013:             else:
1014:                 ax.mouseover_set.discard(self)
1015: 
1016: 
1017: class ArtistInspector(object):
1018:     '''
1019:     A helper class to inspect an :class:`~matplotlib.artist.Artist`
1020:     and return information about it's settable properties and their
1021:     current values.
1022:     '''
1023:     def __init__(self, o):
1024:         '''
1025:         Initialize the artist inspector with an
1026:         :class:`~matplotlib.artist.Artist` or iterable of :class:`Artists`.
1027:         If an iterable is used, we assume it is a homogeneous sequence (all
1028:         :class:`Artists` are of the same type) and it is your responsibility
1029:         to make sure this is so.
1030:         '''
1031:         if cbook.iterable(o):
1032:             # Wrapped in list instead of doing try-except around next(iter(o))
1033:             o = list(o)
1034:             if len(o):
1035:                 o = o[0]
1036: 
1037:         self.oorig = o
1038:         if not inspect.isclass(o):
1039:             o = type(o)
1040:         self.o = o
1041: 
1042:         self.aliasd = self.get_aliases()
1043: 
1044:     def get_aliases(self):
1045:         '''
1046:         Get a dict mapping *fullname* -> *alias* for each *alias* in
1047:         the :class:`~matplotlib.artist.ArtistInspector`.
1048: 
1049:         e.g., for lines::
1050: 
1051:           {'markerfacecolor': 'mfc',
1052:            'linewidth'      : 'lw',
1053:           }
1054: 
1055:         '''
1056:         names = [name for name in dir(self.o)
1057:                  if name.startswith(('set_', 'get_'))
1058:                     and callable(getattr(self.o, name))]
1059:         aliases = {}
1060:         for name in names:
1061:             func = getattr(self.o, name)
1062:             if not self.is_alias(func):
1063:                 continue
1064:             docstring = func.__doc__
1065:             fullname = docstring[10:]
1066:             aliases.setdefault(fullname[4:], {})[name[4:]] = None
1067:         return aliases
1068: 
1069:     _get_valid_values_regex = re.compile(
1070:         r"\n\s*ACCEPTS:\s*((?:.|\n)*?)(?:$|(?:\n\n))"
1071:     )
1072: 
1073:     def get_valid_values(self, attr):
1074:         '''
1075:         Get the legal arguments for the setter associated with *attr*.
1076: 
1077:         This is done by querying the docstring of the function *set_attr*
1078:         for a line that begins with ACCEPTS:
1079: 
1080:         e.g., for a line linestyle, return
1081:         "[ ``'-'`` | ``'--'`` | ``'-.'`` | ``':'`` | ``'steps'`` | ``'None'``
1082:         ]"
1083:         '''
1084: 
1085:         name = 'set_%s' % attr
1086:         if not hasattr(self.o, name):
1087:             raise AttributeError('%s has no function %s' % (self.o, name))
1088:         func = getattr(self.o, name)
1089: 
1090:         docstring = func.__doc__
1091:         if docstring is None:
1092:             return 'unknown'
1093: 
1094:         if docstring.startswith('alias for '):
1095:             return None
1096: 
1097:         match = self._get_valid_values_regex.search(docstring)
1098:         if match is not None:
1099:             return re.sub("\n *", " ", match.group(1))
1100:         return 'unknown'
1101: 
1102:     def _get_setters_and_targets(self):
1103:         '''
1104:         Get the attribute strings and a full path to where the setter
1105:         is defined for all setters in an object.
1106:         '''
1107: 
1108:         setters = []
1109:         for name in dir(self.o):
1110:             if not name.startswith('set_'):
1111:                 continue
1112:             func = getattr(self.o, name)
1113:             if not callable(func):
1114:                 continue
1115:             if six.PY2:
1116:                 nargs = len(inspect.getargspec(func)[0])
1117:             else:
1118:                 nargs = len(inspect.getfullargspec(func)[0])
1119:             if nargs < 2 or self.is_alias(func):
1120:                 continue
1121:             source_class = self.o.__module__ + "." + self.o.__name__
1122:             for cls in self.o.mro():
1123:                 if name in cls.__dict__:
1124:                     source_class = cls.__module__ + "." + cls.__name__
1125:                     break
1126:             setters.append((name[4:], source_class + "." + name))
1127:         return setters
1128: 
1129:     def get_setters(self):
1130:         '''
1131:         Get the attribute strings with setters for object.  e.g., for a line,
1132:         return ``['markerfacecolor', 'linewidth', ....]``.
1133:         '''
1134: 
1135:         return [prop for prop, target in self._get_setters_and_targets()]
1136: 
1137:     def is_alias(self, o):
1138:         '''
1139:         Return *True* if method object *o* is an alias for another
1140:         function.
1141:         '''
1142:         ds = o.__doc__
1143:         if ds is None:
1144:             return False
1145:         return ds.startswith('alias for ')
1146: 
1147:     def aliased_name(self, s):
1148:         '''
1149:         return 'PROPNAME or alias' if *s* has an alias, else return
1150:         PROPNAME.
1151: 
1152:         e.g., for the line markerfacecolor property, which has an
1153:         alias, return 'markerfacecolor or mfc' and for the transform
1154:         property, which does not, return 'transform'
1155:         '''
1156: 
1157:         if s in self.aliasd:
1158:             return s + ''.join([' or %s' % x
1159:                                 for x in sorted(self.aliasd[s])])
1160:         else:
1161:             return s
1162: 
1163:     def aliased_name_rest(self, s, target):
1164:         '''
1165:         return 'PROPNAME or alias' if *s* has an alias, else return
1166:         PROPNAME formatted for ReST
1167: 
1168:         e.g., for the line markerfacecolor property, which has an
1169:         alias, return 'markerfacecolor or mfc' and for the transform
1170:         property, which does not, return 'transform'
1171:         '''
1172: 
1173:         if s in self.aliasd:
1174:             aliases = ''.join([' or %s' % x
1175:                                for x in sorted(self.aliasd[s])])
1176:         else:
1177:             aliases = ''
1178:         return ':meth:`%s <%s>`%s' % (s, target, aliases)
1179: 
1180:     def pprint_setters(self, prop=None, leadingspace=2):
1181:         '''
1182:         If *prop* is *None*, return a list of strings of all settable properies
1183:         and their valid values.
1184: 
1185:         If *prop* is not *None*, it is a valid property name and that
1186:         property will be returned as a string of property : valid
1187:         values.
1188:         '''
1189:         if leadingspace:
1190:             pad = ' ' * leadingspace
1191:         else:
1192:             pad = ''
1193:         if prop is not None:
1194:             accepts = self.get_valid_values(prop)
1195:             return '%s%s: %s' % (pad, prop, accepts)
1196: 
1197:         attrs = self._get_setters_and_targets()
1198:         attrs.sort()
1199:         lines = []
1200: 
1201:         for prop, path in attrs:
1202:             accepts = self.get_valid_values(prop)
1203:             name = self.aliased_name(prop)
1204: 
1205:             lines.append('%s%s: %s' % (pad, name, accepts))
1206:         return lines
1207: 
1208:     def pprint_setters_rest(self, prop=None, leadingspace=2):
1209:         '''
1210:         If *prop* is *None*, return a list of strings of all settable properies
1211:         and their valid values.  Format the output for ReST
1212: 
1213:         If *prop* is not *None*, it is a valid property name and that
1214:         property will be returned as a string of property : valid
1215:         values.
1216:         '''
1217:         if leadingspace:
1218:             pad = ' ' * leadingspace
1219:         else:
1220:             pad = ''
1221:         if prop is not None:
1222:             accepts = self.get_valid_values(prop)
1223:             return '%s%s: %s' % (pad, prop, accepts)
1224: 
1225:         attrs = self._get_setters_and_targets()
1226:         attrs.sort()
1227:         lines = []
1228: 
1229:         ########
1230:         names = [self.aliased_name_rest(prop, target)
1231:                  for prop, target in attrs]
1232:         accepts = [self.get_valid_values(prop) for prop, target in attrs]
1233: 
1234:         col0_len = max(len(n) for n in names)
1235:         col1_len = max(len(a) for a in accepts)
1236:         table_formatstr = pad + '=' * col0_len + '   ' + '=' * col1_len
1237: 
1238:         lines.append('')
1239:         lines.append(table_formatstr)
1240:         lines.append(pad + 'Property'.ljust(col0_len + 3) +
1241:                      'Description'.ljust(col1_len))
1242:         lines.append(table_formatstr)
1243: 
1244:         lines.extend([pad + n.ljust(col0_len + 3) + a.ljust(col1_len)
1245:                       for n, a in zip(names, accepts)])
1246: 
1247:         lines.append(table_formatstr)
1248:         lines.append('')
1249:         return lines
1250: 
1251:     def properties(self):
1252:         '''
1253:         return a dictionary mapping property name -> value
1254:         '''
1255:         o = self.oorig
1256:         getters = [name for name in dir(o)
1257:                    if name.startswith('get_') and callable(getattr(o, name))]
1258:         getters.sort()
1259:         d = dict()
1260:         for name in getters:
1261:             func = getattr(o, name)
1262:             if self.is_alias(func):
1263:                 continue
1264: 
1265:             try:
1266:                 with warnings.catch_warnings():
1267:                     warnings.simplefilter('ignore')
1268:                     val = func()
1269:             except:
1270:                 continue
1271:             else:
1272:                 d[name[4:]] = val
1273: 
1274:         return d
1275: 
1276:     def pprint_getters(self):
1277:         '''
1278:         Return the getters and actual values as list of strings.
1279:         '''
1280: 
1281:         lines = []
1282:         for name, val in sorted(six.iteritems(self.properties())):
1283:             if getattr(val, 'shape', ()) != () and len(val) > 6:
1284:                 s = str(val[:6]) + '...'
1285:             else:
1286:                 s = str(val)
1287:             s = s.replace('\n', ' ')
1288:             if len(s) > 50:
1289:                 s = s[:50] + '...'
1290:             name = self.aliased_name(name)
1291:             lines.append('    %s = %s' % (name, s))
1292:         return lines
1293: 
1294: 
1295: def getp(obj, property=None):
1296:     '''
1297:     Return the value of object's property.  *property* is an optional string
1298:     for the property you want to return
1299: 
1300:     Example usage::
1301: 
1302:         getp(obj)  # get all the object properties
1303:         getp(obj, 'linestyle')  # get the linestyle property
1304: 
1305:     *obj* is a :class:`Artist` instance, e.g.,
1306:     :class:`~matplotllib.lines.Line2D` or an instance of a
1307:     :class:`~matplotlib.axes.Axes` or :class:`matplotlib.text.Text`.
1308:     If the *property* is 'somename', this function returns
1309: 
1310:       obj.get_somename()
1311: 
1312:     :func:`getp` can be used to query all the gettable properties with
1313:     ``getp(obj)``. Many properties have aliases for shorter typing, e.g.
1314:     'lw' is an alias for 'linewidth'.  In the output, aliases and full
1315:     property names will be listed as:
1316: 
1317:       property or alias = value
1318: 
1319:     e.g.:
1320: 
1321:       linewidth or lw = 2
1322:     '''
1323:     if property is None:
1324:         insp = ArtistInspector(obj)
1325:         ret = insp.pprint_getters()
1326:         print('\n'.join(ret))
1327:         return
1328: 
1329:     func = getattr(obj, 'get_' + property)
1330:     return func()
1331: 
1332: # alias
1333: get = getp
1334: 
1335: 
1336: def setp(obj, *args, **kwargs):
1337:     '''
1338:     Set a property on an artist object.
1339: 
1340:     matplotlib supports the use of :func:`setp` ("set property") and
1341:     :func:`getp` to set and get object properties, as well as to do
1342:     introspection on the object.  For example, to set the linestyle of a
1343:     line to be dashed, you can do::
1344: 
1345:       >>> line, = plot([1,2,3])
1346:       >>> setp(line, linestyle='--')
1347: 
1348:     If you want to know the valid types of arguments, you can provide
1349:     the name of the property you want to set without a value::
1350: 
1351:       >>> setp(line, 'linestyle')
1352:           linestyle: [ '-' | '--' | '-.' | ':' | 'steps' | 'None' ]
1353: 
1354:     If you want to see all the properties that can be set, and their
1355:     possible values, you can do::
1356: 
1357:       >>> setp(line)
1358:           ... long output listing omitted
1359: 
1360:     You may specify another output file to `setp` if `sys.stdout` is not
1361:     acceptable for some reason using the `file` keyword-only argument::
1362: 
1363:       >>> with fopen('output.log') as f:
1364:       >>>     setp(line, file=f)
1365: 
1366:     :func:`setp` operates on a single instance or a iterable of
1367:     instances. If you are in query mode introspecting the possible
1368:     values, only the first instance in the sequence is used. When
1369:     actually setting values, all the instances will be set.  e.g.,
1370:     suppose you have a list of two lines, the following will make both
1371:     lines thicker and red::
1372: 
1373:       >>> x = arange(0,1.0,0.01)
1374:       >>> y1 = sin(2*pi*x)
1375:       >>> y2 = sin(4*pi*x)
1376:       >>> lines = plot(x, y1, x, y2)
1377:       >>> setp(lines, linewidth=2, color='r')
1378: 
1379:     :func:`setp` works with the MATLAB style string/value pairs or
1380:     with python kwargs.  For example, the following are equivalent::
1381: 
1382:       >>> setp(lines, 'linewidth', 2, 'color', 'r')  # MATLAB style
1383:       >>> setp(lines, linewidth=2, color='r')        # python style
1384:     '''
1385: 
1386:     if not cbook.iterable(obj):
1387:         objs = [obj]
1388:     else:
1389:         objs = list(cbook.flatten(obj))
1390: 
1391:     if not objs:
1392:         return
1393: 
1394:     insp = ArtistInspector(objs[0])
1395: 
1396:     # file has to be popped before checking if kwargs is empty
1397:     printArgs = {}
1398:     if 'file' in kwargs:
1399:         printArgs['file'] = kwargs.pop('file')
1400: 
1401:     if not kwargs and len(args) < 2:
1402:         if args:
1403:             print(insp.pprint_setters(prop=args[0]), **printArgs)
1404:         else:
1405:             print('\n'.join(insp.pprint_setters()), **printArgs)
1406:         return
1407: 
1408:     if len(args) % 2:
1409:         raise ValueError('The set args must be string, value pairs')
1410: 
1411:     # put args into ordereddict to maintain order
1412:     funcvals = OrderedDict()
1413:     for i in range(0, len(args) - 1, 2):
1414:         funcvals[args[i]] = args[i + 1]
1415: 
1416:     ret = [o.update(funcvals) for o in objs]
1417:     ret.extend([o.set(**kwargs) for o in objs])
1418:     return [x for x in cbook.flatten(ret)]
1419: 
1420: 
1421: def kwdoc(a):
1422:     hardcopy = matplotlib.rcParams['docstring.hardcopy']
1423:     if hardcopy:
1424:         return '\n'.join(ArtistInspector(a).pprint_setters_rest(
1425:                          leadingspace=2))
1426:     else:
1427:         return '\n'.join(ArtistInspector(a).pprint_setters(leadingspace=2))
1428: 
1429: docstring.interpd.update(Artist=kwdoc(Artist))
1430: 
1431: _get_axes_msg = '''{0} has been deprecated in mpl 1.5, please use the
1432: axes property.  A removal date has not been set.'''
1433: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_4603 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_4603) is not StypyTypeError):

    if (import_4603 != 'pyd_module'):
        __import__(import_4603)
        sys_modules_4604 = sys.modules[import_4603]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_4604.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_4603)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from collections import OrderedDict, namedtuple' statement (line 6)
try:
    from collections import OrderedDict, namedtuple

except:
    OrderedDict = UndefinedType
    namedtuple = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'collections', None, module_type_store, ['OrderedDict', 'namedtuple'], [OrderedDict, namedtuple])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from functools import wraps' statement (line 7)
try:
    from functools import wraps

except:
    wraps = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'functools', None, module_type_store, ['wraps'], [wraps])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import inspect' statement (line 8)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import re' statement (line 9)
import re

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import warnings' statement (line 10)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import numpy' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_4605 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy')

if (type(import_4605) is not StypyTypeError):

    if (import_4605 != 'pyd_module'):
        __import__(import_4605)
        sys_modules_4606 = sys.modules[import_4605]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'np', sys_modules_4606.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', import_4605)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import matplotlib' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_4607 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib')

if (type(import_4607) is not StypyTypeError):

    if (import_4607 != 'pyd_module'):
        __import__(import_4607)
        sys_modules_4608 = sys.modules[import_4607]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', sys_modules_4608.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', import_4607)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib import cbook, docstring, rcParams' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_4609 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib')

if (type(import_4609) is not StypyTypeError):

    if (import_4609 != 'pyd_module'):
        __import__(import_4609)
        sys_modules_4610 = sys.modules[import_4609]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', sys_modules_4610.module_type_store, module_type_store, ['cbook', 'docstring', 'rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_4610, sys_modules_4610.module_type_store, module_type_store)
    else:
        from matplotlib import cbook, docstring, rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', None, module_type_store, ['cbook', 'docstring', 'rcParams'], [cbook, docstring, rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', import_4609)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from matplotlib.path import Path' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_4611 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.path')

if (type(import_4611) is not StypyTypeError):

    if (import_4611 != 'pyd_module'):
        __import__(import_4611)
        sys_modules_4612 = sys.modules[import_4611]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.path', sys_modules_4612.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_4612, sys_modules_4612.module_type_store, module_type_store)
    else:
        from matplotlib.path import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.path', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'matplotlib.path' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib.path', import_4611)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from matplotlib.transforms import Bbox, IdentityTransform, Transform, TransformedBbox, TransformedPatchPath, TransformedPath' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_4613 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.transforms')

if (type(import_4613) is not StypyTypeError):

    if (import_4613 != 'pyd_module'):
        __import__(import_4613)
        sys_modules_4614 = sys.modules[import_4613]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.transforms', sys_modules_4614.module_type_store, module_type_store, ['Bbox', 'IdentityTransform', 'Transform', 'TransformedBbox', 'TransformedPatchPath', 'TransformedPath'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_4614, sys_modules_4614.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Bbox, IdentityTransform, Transform, TransformedBbox, TransformedPatchPath, TransformedPath

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.transforms', None, module_type_store, ['Bbox', 'IdentityTransform', 'Transform', 'TransformedBbox', 'TransformedPatchPath', 'TransformedPath'], [Bbox, IdentityTransform, Transform, TransformedBbox, TransformedPatchPath, TransformedPath])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.transforms', import_4613)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


@norecursion
def allow_rasterization(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'allow_rasterization'
    module_type_store = module_type_store.open_function_context('allow_rasterization', 37, 0, False)
    
    # Passed parameters checking function
    allow_rasterization.stypy_localization = localization
    allow_rasterization.stypy_type_of_self = None
    allow_rasterization.stypy_type_store = module_type_store
    allow_rasterization.stypy_function_name = 'allow_rasterization'
    allow_rasterization.stypy_param_names_list = ['draw']
    allow_rasterization.stypy_varargs_param_name = None
    allow_rasterization.stypy_kwargs_param_name = None
    allow_rasterization.stypy_call_defaults = defaults
    allow_rasterization.stypy_call_varargs = varargs
    allow_rasterization.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'allow_rasterization', ['draw'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'allow_rasterization', localization, ['draw'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'allow_rasterization(...)' code ##################

    unicode_4615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'unicode', u'\n    Decorator for Artist.draw method. Provides routines\n    that run before and after the draw call. The before and after functions\n    are useful for changing artist-dependent renderer attributes or making\n    other setup function calls, such as starting and flushing a mixed-mode\n    renderer.\n    ')

    @norecursion
    def draw_wrapper(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_wrapper'
        module_type_store = module_type_store.open_function_context('draw_wrapper', 47, 4, False)
        
        # Passed parameters checking function
        draw_wrapper.stypy_localization = localization
        draw_wrapper.stypy_type_of_self = None
        draw_wrapper.stypy_type_store = module_type_store
        draw_wrapper.stypy_function_name = 'draw_wrapper'
        draw_wrapper.stypy_param_names_list = ['artist', 'renderer']
        draw_wrapper.stypy_varargs_param_name = 'args'
        draw_wrapper.stypy_kwargs_param_name = 'kwargs'
        draw_wrapper.stypy_call_defaults = defaults
        draw_wrapper.stypy_call_varargs = varargs
        draw_wrapper.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'draw_wrapper', ['artist', 'renderer'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_wrapper', localization, ['artist', 'renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_wrapper(...)' code ##################

        
        # Try-finally block (line 49)
        
        
        # Call to get_rasterized(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_4618 = {}
        # Getting the type of 'artist' (line 50)
        artist_4616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'artist', False)
        # Obtaining the member 'get_rasterized' of a type (line 50)
        get_rasterized_4617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), artist_4616, 'get_rasterized')
        # Calling get_rasterized(args, kwargs) (line 50)
        get_rasterized_call_result_4619 = invoke(stypy.reporting.localization.Localization(__file__, 50, 15), get_rasterized_4617, *[], **kwargs_4618)
        
        # Testing the type of an if condition (line 50)
        if_condition_4620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 12), get_rasterized_call_result_4619)
        # Assigning a type to the variable 'if_condition_4620' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'if_condition_4620', if_condition_4620)
        # SSA begins for if statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to start_rasterizing(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_4623 = {}
        # Getting the type of 'renderer' (line 51)
        renderer_4621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'renderer', False)
        # Obtaining the member 'start_rasterizing' of a type (line 51)
        start_rasterizing_4622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), renderer_4621, 'start_rasterizing')
        # Calling start_rasterizing(args, kwargs) (line 51)
        start_rasterizing_call_result_4624 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), start_rasterizing_4622, *[], **kwargs_4623)
        
        # SSA join for if statement (line 50)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to get_agg_filter(...): (line 52)
        # Processing the call keyword arguments (line 52)
        kwargs_4627 = {}
        # Getting the type of 'artist' (line 52)
        artist_4625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'artist', False)
        # Obtaining the member 'get_agg_filter' of a type (line 52)
        get_agg_filter_4626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 15), artist_4625, 'get_agg_filter')
        # Calling get_agg_filter(args, kwargs) (line 52)
        get_agg_filter_call_result_4628 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), get_agg_filter_4626, *[], **kwargs_4627)
        
        # Getting the type of 'None' (line 52)
        None_4629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 46), 'None')
        # Applying the binary operator 'isnot' (line 52)
        result_is_not_4630 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 15), 'isnot', get_agg_filter_call_result_4628, None_4629)
        
        # Testing the type of an if condition (line 52)
        if_condition_4631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 12), result_is_not_4630)
        # Assigning a type to the variable 'if_condition_4631' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'if_condition_4631', if_condition_4631)
        # SSA begins for if statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to start_filter(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_4634 = {}
        # Getting the type of 'renderer' (line 53)
        renderer_4632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'renderer', False)
        # Obtaining the member 'start_filter' of a type (line 53)
        start_filter_4633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 16), renderer_4632, 'start_filter')
        # Calling start_filter(args, kwargs) (line 53)
        start_filter_call_result_4635 = invoke(stypy.reporting.localization.Localization(__file__, 53, 16), start_filter_4633, *[], **kwargs_4634)
        
        # SSA join for if statement (line 52)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'artist' (line 55)
        artist_4637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'artist', False)
        # Getting the type of 'renderer' (line 55)
        renderer_4638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 32), 'renderer', False)
        # Getting the type of 'args' (line 55)
        args_4639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 43), 'args', False)
        # Processing the call keyword arguments (line 55)
        # Getting the type of 'kwargs' (line 55)
        kwargs_4640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 51), 'kwargs', False)
        kwargs_4641 = {'kwargs_4640': kwargs_4640}
        # Getting the type of 'draw' (line 55)
        draw_4636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'draw', False)
        # Calling draw(args, kwargs) (line 55)
        draw_call_result_4642 = invoke(stypy.reporting.localization.Localization(__file__, 55, 19), draw_4636, *[artist_4637, renderer_4638, args_4639], **kwargs_4641)
        
        # Assigning a type to the variable 'stypy_return_type' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'stypy_return_type', draw_call_result_4642)
        
        # finally branch of the try-finally block (line 49)
        
        
        
        # Call to get_agg_filter(...): (line 57)
        # Processing the call keyword arguments (line 57)
        kwargs_4645 = {}
        # Getting the type of 'artist' (line 57)
        artist_4643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'artist', False)
        # Obtaining the member 'get_agg_filter' of a type (line 57)
        get_agg_filter_4644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 15), artist_4643, 'get_agg_filter')
        # Calling get_agg_filter(args, kwargs) (line 57)
        get_agg_filter_call_result_4646 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), get_agg_filter_4644, *[], **kwargs_4645)
        
        # Getting the type of 'None' (line 57)
        None_4647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 46), 'None')
        # Applying the binary operator 'isnot' (line 57)
        result_is_not_4648 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 15), 'isnot', get_agg_filter_call_result_4646, None_4647)
        
        # Testing the type of an if condition (line 57)
        if_condition_4649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 12), result_is_not_4648)
        # Assigning a type to the variable 'if_condition_4649' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'if_condition_4649', if_condition_4649)
        # SSA begins for if statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to stop_filter(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to get_agg_filter(...): (line 58)
        # Processing the call keyword arguments (line 58)
        kwargs_4654 = {}
        # Getting the type of 'artist' (line 58)
        artist_4652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'artist', False)
        # Obtaining the member 'get_agg_filter' of a type (line 58)
        get_agg_filter_4653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 37), artist_4652, 'get_agg_filter')
        # Calling get_agg_filter(args, kwargs) (line 58)
        get_agg_filter_call_result_4655 = invoke(stypy.reporting.localization.Localization(__file__, 58, 37), get_agg_filter_4653, *[], **kwargs_4654)
        
        # Processing the call keyword arguments (line 58)
        kwargs_4656 = {}
        # Getting the type of 'renderer' (line 58)
        renderer_4650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'renderer', False)
        # Obtaining the member 'stop_filter' of a type (line 58)
        stop_filter_4651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), renderer_4650, 'stop_filter')
        # Calling stop_filter(args, kwargs) (line 58)
        stop_filter_call_result_4657 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), stop_filter_4651, *[get_agg_filter_call_result_4655], **kwargs_4656)
        
        # SSA join for if statement (line 57)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to get_rasterized(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_4660 = {}
        # Getting the type of 'artist' (line 59)
        artist_4658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'artist', False)
        # Obtaining the member 'get_rasterized' of a type (line 59)
        get_rasterized_4659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), artist_4658, 'get_rasterized')
        # Calling get_rasterized(args, kwargs) (line 59)
        get_rasterized_call_result_4661 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), get_rasterized_4659, *[], **kwargs_4660)
        
        # Testing the type of an if condition (line 59)
        if_condition_4662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 12), get_rasterized_call_result_4661)
        # Assigning a type to the variable 'if_condition_4662' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'if_condition_4662', if_condition_4662)
        # SSA begins for if statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to stop_rasterizing(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_4665 = {}
        # Getting the type of 'renderer' (line 60)
        renderer_4663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'renderer', False)
        # Obtaining the member 'stop_rasterizing' of a type (line 60)
        stop_rasterizing_4664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), renderer_4663, 'stop_rasterizing')
        # Calling stop_rasterizing(args, kwargs) (line 60)
        stop_rasterizing_call_result_4666 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), stop_rasterizing_4664, *[], **kwargs_4665)
        
        # SSA join for if statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # ################# End of 'draw_wrapper(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_wrapper' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_4667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4667)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_wrapper'
        return stypy_return_type_4667

    # Assigning a type to the variable 'draw_wrapper' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'draw_wrapper', draw_wrapper)
    
    # Assigning a Name to a Attribute (line 62):
    
    # Assigning a Name to a Attribute (line 62):
    # Getting the type of 'True' (line 62)
    True_4668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 43), 'True')
    # Getting the type of 'draw_wrapper' (line 62)
    draw_wrapper_4669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'draw_wrapper')
    # Setting the type of the member '_supports_rasterization' of a type (line 62)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), draw_wrapper_4669, '_supports_rasterization', True_4668)
    # Getting the type of 'draw_wrapper' (line 63)
    draw_wrapper_4670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'draw_wrapper')
    # Assigning a type to the variable 'stypy_return_type' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type', draw_wrapper_4670)
    
    # ################# End of 'allow_rasterization(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'allow_rasterization' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_4671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4671)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'allow_rasterization'
    return stypy_return_type_4671

# Assigning a type to the variable 'allow_rasterization' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'allow_rasterization', allow_rasterization)

@norecursion
def _stale_axes_callback(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stale_axes_callback'
    module_type_store = module_type_store.open_function_context('_stale_axes_callback', 66, 0, False)
    
    # Passed parameters checking function
    _stale_axes_callback.stypy_localization = localization
    _stale_axes_callback.stypy_type_of_self = None
    _stale_axes_callback.stypy_type_store = module_type_store
    _stale_axes_callback.stypy_function_name = '_stale_axes_callback'
    _stale_axes_callback.stypy_param_names_list = ['self', 'val']
    _stale_axes_callback.stypy_varargs_param_name = None
    _stale_axes_callback.stypy_kwargs_param_name = None
    _stale_axes_callback.stypy_call_defaults = defaults
    _stale_axes_callback.stypy_call_varargs = varargs
    _stale_axes_callback.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stale_axes_callback', ['self', 'val'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_stale_axes_callback', localization, ['self', 'val'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_stale_axes_callback(...)' code ##################

    
    # Getting the type of 'self' (line 67)
    self_4672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 7), 'self')
    # Obtaining the member 'axes' of a type (line 67)
    axes_4673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 7), self_4672, 'axes')
    # Testing the type of an if condition (line 67)
    if_condition_4674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 4), axes_4673)
    # Assigning a type to the variable 'if_condition_4674' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'if_condition_4674', if_condition_4674)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Attribute (line 68):
    
    # Assigning a Name to a Attribute (line 68):
    # Getting the type of 'val' (line 68)
    val_4675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'val')
    # Getting the type of 'self' (line 68)
    self_4676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
    # Obtaining the member 'axes' of a type (line 68)
    axes_4677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_4676, 'axes')
    # Setting the type of the member 'stale' of a type (line 68)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), axes_4677, 'stale', val_4675)
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_stale_axes_callback(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_stale_axes_callback' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_4678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4678)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stale_axes_callback'
    return stypy_return_type_4678

# Assigning a type to the variable '_stale_axes_callback' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), '_stale_axes_callback', _stale_axes_callback)

# Assigning a Call to a Name (line 71):

# Assigning a Call to a Name (line 71):

# Call to namedtuple(...): (line 71)
# Processing the call arguments (line 71)
unicode_4680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'unicode', u'_XYPair')
unicode_4681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 32), 'unicode', u'x y')
# Processing the call keyword arguments (line 71)
kwargs_4682 = {}
# Getting the type of 'namedtuple' (line 71)
namedtuple_4679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 71)
namedtuple_call_result_4683 = invoke(stypy.reporting.localization.Localization(__file__, 71, 10), namedtuple_4679, *[unicode_4680, unicode_4681], **kwargs_4682)

# Assigning a type to the variable '_XYPair' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), '_XYPair', namedtuple_call_result_4683)
# Declaration of the 'Artist' class

class Artist(object, ):
    unicode_4684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'unicode', u'\n    Abstract base class for someone who renders into a\n    :class:`FigureCanvas`.\n    ')
    
    # Assigning a Str to a Name (line 80):
    
    # Assigning a Num to a Name (line 81):
    
    # Assigning a Call to a Name (line 85):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 87, 4, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 88):
        
        # Assigning a Name to a Attribute (line 88):
        # Getting the type of 'True' (line 88)
        True_4685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'True')
        # Getting the type of 'self' (line 88)
        self_4686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self')
        # Setting the type of the member '_stale' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_4686, '_stale', True_4685)
        
        # Assigning a Name to a Attribute (line 89):
        
        # Assigning a Name to a Attribute (line 89):
        # Getting the type of 'None' (line 89)
        None_4687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'None')
        # Getting the type of 'self' (line 89)
        self_4688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self')
        # Setting the type of the member 'stale_callback' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_4688, 'stale_callback', None_4687)
        
        # Assigning a Name to a Attribute (line 90):
        
        # Assigning a Name to a Attribute (line 90):
        # Getting the type of 'None' (line 90)
        None_4689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'None')
        # Getting the type of 'self' (line 90)
        self_4690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member '_axes' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_4690, '_axes', None_4689)
        
        # Assigning a Name to a Attribute (line 91):
        
        # Assigning a Name to a Attribute (line 91):
        # Getting the type of 'None' (line 91)
        None_4691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'None')
        # Getting the type of 'self' (line 91)
        self_4692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'figure' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_4692, 'figure', None_4691)
        
        # Assigning a Name to a Attribute (line 93):
        
        # Assigning a Name to a Attribute (line 93):
        # Getting the type of 'None' (line 93)
        None_4693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), 'None')
        # Getting the type of 'self' (line 93)
        self_4694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'self')
        # Setting the type of the member '_transform' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), self_4694, '_transform', None_4693)
        
        # Assigning a Name to a Attribute (line 94):
        
        # Assigning a Name to a Attribute (line 94):
        # Getting the type of 'False' (line 94)
        False_4695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 29), 'False')
        # Getting the type of 'self' (line 94)
        self_4696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member '_transformSet' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_4696, '_transformSet', False_4695)
        
        # Assigning a Name to a Attribute (line 95):
        
        # Assigning a Name to a Attribute (line 95):
        # Getting the type of 'True' (line 95)
        True_4697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'True')
        # Getting the type of 'self' (line 95)
        self_4698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member '_visible' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_4698, '_visible', True_4697)
        
        # Assigning a Name to a Attribute (line 96):
        
        # Assigning a Name to a Attribute (line 96):
        # Getting the type of 'False' (line 96)
        False_4699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'False')
        # Getting the type of 'self' (line 96)
        self_4700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Setting the type of the member '_animated' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_4700, '_animated', False_4699)
        
        # Assigning a Name to a Attribute (line 97):
        
        # Assigning a Name to a Attribute (line 97):
        # Getting the type of 'None' (line 97)
        None_4701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 22), 'None')
        # Getting the type of 'self' (line 97)
        self_4702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self')
        # Setting the type of the member '_alpha' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_4702, '_alpha', None_4701)
        
        # Assigning a Name to a Attribute (line 98):
        
        # Assigning a Name to a Attribute (line 98):
        # Getting the type of 'None' (line 98)
        None_4703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'None')
        # Getting the type of 'self' (line 98)
        self_4704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self')
        # Setting the type of the member 'clipbox' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_4704, 'clipbox', None_4703)
        
        # Assigning a Name to a Attribute (line 99):
        
        # Assigning a Name to a Attribute (line 99):
        # Getting the type of 'None' (line 99)
        None_4705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'None')
        # Getting the type of 'self' (line 99)
        self_4706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Setting the type of the member '_clippath' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_4706, '_clippath', None_4705)
        
        # Assigning a Name to a Attribute (line 100):
        
        # Assigning a Name to a Attribute (line 100):
        # Getting the type of 'True' (line 100)
        True_4707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'True')
        # Getting the type of 'self' (line 100)
        self_4708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Setting the type of the member '_clipon' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_4708, '_clipon', True_4707)
        
        # Assigning a Str to a Attribute (line 101):
        
        # Assigning a Str to a Attribute (line 101):
        unicode_4709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'unicode', u'')
        # Getting the type of 'self' (line 101)
        self_4710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self')
        # Setting the type of the member '_label' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_4710, '_label', unicode_4709)
        
        # Assigning a Name to a Attribute (line 102):
        
        # Assigning a Name to a Attribute (line 102):
        # Getting the type of 'None' (line 102)
        None_4711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'None')
        # Getting the type of 'self' (line 102)
        self_4712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self')
        # Setting the type of the member '_picker' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_4712, '_picker', None_4711)
        
        # Assigning a Name to a Attribute (line 103):
        
        # Assigning a Name to a Attribute (line 103):
        # Getting the type of 'None' (line 103)
        None_4713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'None')
        # Getting the type of 'self' (line 103)
        self_4714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self')
        # Setting the type of the member '_contains' of a type (line 103)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_4714, '_contains', None_4713)
        
        # Assigning a Name to a Attribute (line 104):
        
        # Assigning a Name to a Attribute (line 104):
        # Getting the type of 'None' (line 104)
        None_4715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'None')
        # Getting the type of 'self' (line 104)
        self_4716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self')
        # Setting the type of the member '_rasterized' of a type (line 104)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_4716, '_rasterized', None_4715)
        
        # Assigning a Name to a Attribute (line 105):
        
        # Assigning a Name to a Attribute (line 105):
        # Getting the type of 'None' (line 105)
        None_4717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'None')
        # Getting the type of 'self' (line 105)
        self_4718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self')
        # Setting the type of the member '_agg_filter' of a type (line 105)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_4718, '_agg_filter', None_4717)
        
        # Assigning a Name to a Attribute (line 106):
        
        # Assigning a Name to a Attribute (line 106):
        # Getting the type of 'False' (line 106)
        False_4719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'False')
        # Getting the type of 'self' (line 106)
        self_4720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self')
        # Setting the type of the member '_mouseover' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_4720, '_mouseover', False_4719)
        
        # Assigning a Name to a Attribute (line 107):
        
        # Assigning a Name to a Attribute (line 107):
        # Getting the type of 'False' (line 107)
        False_4721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'False')
        # Getting the type of 'self' (line 107)
        self_4722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'self')
        # Setting the type of the member 'eventson' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), self_4722, 'eventson', False_4721)
        
        # Assigning a Num to a Attribute (line 108):
        
        # Assigning a Num to a Attribute (line 108):
        int_4723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 20), 'int')
        # Getting the type of 'self' (line 108)
        self_4724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self')
        # Setting the type of the member '_oid' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_4724, '_oid', int_4723)
        
        # Assigning a Dict to a Attribute (line 109):
        
        # Assigning a Dict to a Attribute (line 109):
        
        # Obtaining an instance of the builtin type 'dict' (line 109)
        dict_4725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 109)
        
        # Getting the type of 'self' (line 109)
        self_4726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member '_propobservers' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_4726, '_propobservers', dict_4725)
        
        
        # SSA begins for try-except statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Attribute (line 111):
        
        # Assigning a Name to a Attribute (line 111):
        # Getting the type of 'None' (line 111)
        None_4727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'None')
        # Getting the type of 'self' (line 111)
        self_4728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'self')
        # Setting the type of the member 'axes' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), self_4728, 'axes', None_4727)
        # SSA branch for the except part of a try statement (line 110)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 110)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 115):
        
        # Assigning a Name to a Attribute (line 115):
        # Getting the type of 'None' (line 115)
        None_4729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 30), 'None')
        # Getting the type of 'self' (line 115)
        self_4730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'self')
        # Setting the type of the member '_remove_method' of a type (line 115)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), self_4730, '_remove_method', None_4729)
        
        # Assigning a Name to a Attribute (line 116):
        
        # Assigning a Name to a Attribute (line 116):
        # Getting the type of 'None' (line 116)
        None_4731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'None')
        # Getting the type of 'self' (line 116)
        self_4732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self')
        # Setting the type of the member '_url' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_4732, '_url', None_4731)
        
        # Assigning a Name to a Attribute (line 117):
        
        # Assigning a Name to a Attribute (line 117):
        # Getting the type of 'None' (line 117)
        None_4733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 20), 'None')
        # Getting the type of 'self' (line 117)
        self_4734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self')
        # Setting the type of the member '_gid' of a type (line 117)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_4734, '_gid', None_4733)
        
        # Assigning a Name to a Attribute (line 118):
        
        # Assigning a Name to a Attribute (line 118):
        # Getting the type of 'None' (line 118)
        None_4735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'None')
        # Getting the type of 'self' (line 118)
        self_4736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self')
        # Setting the type of the member '_snap' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_4736, '_snap', None_4735)
        
        # Assigning a Subscript to a Attribute (line 119):
        
        # Assigning a Subscript to a Attribute (line 119):
        
        # Obtaining the type of the subscript
        unicode_4737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'unicode', u'path.sketch')
        # Getting the type of 'rcParams' (line 119)
        rcParams_4738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___4739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 23), rcParams_4738, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_4740 = invoke(stypy.reporting.localization.Localization(__file__, 119, 23), getitem___4739, unicode_4737)
        
        # Getting the type of 'self' (line 119)
        self_4741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member '_sketch' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_4741, '_sketch', subscript_call_result_4740)
        
        # Assigning a Subscript to a Attribute (line 120):
        
        # Assigning a Subscript to a Attribute (line 120):
        
        # Obtaining the type of the subscript
        unicode_4742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 38), 'unicode', u'path.effects')
        # Getting the type of 'rcParams' (line 120)
        rcParams_4743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___4744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 29), rcParams_4743, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_4745 = invoke(stypy.reporting.localization.Localization(__file__, 120, 29), getitem___4744, unicode_4742)
        
        # Getting the type of 'self' (line 120)
        self_4746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Setting the type of the member '_path_effects' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_4746, '_path_effects', subscript_call_result_4745)
        
        # Assigning a Call to a Attribute (line 121):
        
        # Assigning a Call to a Attribute (line 121):
        
        # Call to _XYPair(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_4748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_4749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        
        # Processing the call keyword arguments (line 121)
        kwargs_4750 = {}
        # Getting the type of '_XYPair' (line 121)
        _XYPair_4747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), '_XYPair', False)
        # Calling _XYPair(args, kwargs) (line 121)
        _XYPair_call_result_4751 = invoke(stypy.reporting.localization.Localization(__file__, 121, 29), _XYPair_4747, *[list_4748, list_4749], **kwargs_4750)
        
        # Getting the type of 'self' (line 121)
        self_4752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self')
        # Setting the type of the member '_sticky_edges' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_4752, '_sticky_edges', _XYPair_call_result_4751)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getstate__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getstate__'
        module_type_store = module_type_store.open_function_context('__getstate__', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.__getstate__.__dict__.__setitem__('stypy_localization', localization)
        Artist.__getstate__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.__getstate__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.__getstate__.__dict__.__setitem__('stypy_function_name', 'Artist.__getstate__')
        Artist.__getstate__.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.__getstate__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.__getstate__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.__getstate__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.__getstate__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.__getstate__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.__getstate__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.__getstate__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getstate__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getstate__(...)' code ##################

        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to copy(...): (line 124)
        # Processing the call keyword arguments (line 124)
        kwargs_4756 = {}
        # Getting the type of 'self' (line 124)
        self_4753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'self', False)
        # Obtaining the member '__dict__' of a type (line 124)
        dict___4754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), self_4753, '__dict__')
        # Obtaining the member 'copy' of a type (line 124)
        copy_4755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), dict___4754, 'copy')
        # Calling copy(args, kwargs) (line 124)
        copy_call_result_4757 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), copy_4755, *[], **kwargs_4756)
        
        # Assigning a type to the variable 'd' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'd', copy_call_result_4757)
        
        # Assigning a Name to a Subscript (line 127):
        
        # Assigning a Name to a Subscript (line 127):
        # Getting the type of 'None' (line 127)
        None_4758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'None')
        # Getting the type of 'd' (line 127)
        d_4759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'd')
        unicode_4760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 10), 'unicode', u'_remove_method')
        # Storing an element on a container (line 127)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 8), d_4759, (unicode_4760, None_4758))
        
        # Assigning a Name to a Subscript (line 128):
        
        # Assigning a Name to a Subscript (line 128):
        # Getting the type of 'None' (line 128)
        None_4761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 30), 'None')
        # Getting the type of 'd' (line 128)
        d_4762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'd')
        unicode_4763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 10), 'unicode', u'stale_callback')
        # Storing an element on a container (line 128)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 8), d_4762, (unicode_4763, None_4761))
        # Getting the type of 'd' (line 129)
        d_4764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'd')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', d_4764)
        
        # ################# End of '__getstate__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getstate__' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_4765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4765)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getstate__'
        return stypy_return_type_4765


    @norecursion
    def remove(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove'
        module_type_store = module_type_store.open_function_context('remove', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.remove.__dict__.__setitem__('stypy_localization', localization)
        Artist.remove.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.remove.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.remove.__dict__.__setitem__('stypy_function_name', 'Artist.remove')
        Artist.remove.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.remove.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.remove.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.remove.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.remove.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.remove.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.remove.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.remove', [], None, None, defaults, varargs, kwargs)

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

        unicode_4766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'unicode', u"\n        Remove the artist from the figure if possible.  The effect\n        will not be visible until the figure is redrawn, e.g., with\n        :meth:`matplotlib.axes.Axes.draw_idle`.  Call\n        :meth:`matplotlib.axes.Axes.relim` to update the axes limits\n        if desired.\n\n        Note: :meth:`~matplotlib.axes.Axes.relim` will not see\n        collections even if the collection was added to axes with\n        *autolim* = True.\n\n        Note: there is no support for removing the artist's legend entry.\n        ")
        
        
        # Getting the type of 'self' (line 150)
        self_4767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'self')
        # Obtaining the member '_remove_method' of a type (line 150)
        _remove_method_4768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 11), self_4767, '_remove_method')
        # Getting the type of 'None' (line 150)
        None_4769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 38), 'None')
        # Applying the binary operator 'isnot' (line 150)
        result_is_not_4770 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 11), 'isnot', _remove_method_4768, None_4769)
        
        # Testing the type of an if condition (line 150)
        if_condition_4771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), result_is_not_4770)
        # Assigning a type to the variable 'if_condition_4771' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_4771', if_condition_4771)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _remove_method(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'self' (line 151)
        self_4774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 32), 'self', False)
        # Processing the call keyword arguments (line 151)
        kwargs_4775 = {}
        # Getting the type of 'self' (line 151)
        self_4772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'self', False)
        # Obtaining the member '_remove_method' of a type (line 151)
        _remove_method_4773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), self_4772, '_remove_method')
        # Calling _remove_method(args, kwargs) (line 151)
        _remove_method_call_result_4776 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), _remove_method_4773, *[self_4774], **kwargs_4775)
        
        
        # Assigning a Name to a Attribute (line 153):
        
        # Assigning a Name to a Attribute (line 153):
        # Getting the type of 'None' (line 153)
        None_4777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'None')
        # Getting the type of 'self' (line 153)
        self_4778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'self')
        # Setting the type of the member 'stale_callback' of a type (line 153)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), self_4778, 'stale_callback', None_4777)
        
        # Assigning a Name to a Name (line 154):
        
        # Assigning a Name to a Name (line 154):
        # Getting the type of 'False' (line 154)
        False_4779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'False')
        # Assigning a type to the variable '_ax_flag' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), '_ax_flag', False_4779)
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'self' (line 155)
        self_4781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 23), 'self', False)
        unicode_4782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 29), 'unicode', u'axes')
        # Processing the call keyword arguments (line 155)
        kwargs_4783 = {}
        # Getting the type of 'hasattr' (line 155)
        hasattr_4780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 155)
        hasattr_call_result_4784 = invoke(stypy.reporting.localization.Localization(__file__, 155, 15), hasattr_4780, *[self_4781, unicode_4782], **kwargs_4783)
        
        # Getting the type of 'self' (line 155)
        self_4785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 41), 'self')
        # Obtaining the member 'axes' of a type (line 155)
        axes_4786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 41), self_4785, 'axes')
        # Applying the binary operator 'and' (line 155)
        result_and_keyword_4787 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 15), 'and', hasattr_call_result_4784, axes_4786)
        
        # Testing the type of an if condition (line 155)
        if_condition_4788 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 12), result_and_keyword_4787)
        # Assigning a type to the variable 'if_condition_4788' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'if_condition_4788', if_condition_4788)
        # SSA begins for if statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to discard(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'self' (line 157)
        self_4793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 48), 'self', False)
        # Processing the call keyword arguments (line 157)
        kwargs_4794 = {}
        # Getting the type of 'self' (line 157)
        self_4789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'self', False)
        # Obtaining the member 'axes' of a type (line 157)
        axes_4790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 16), self_4789, 'axes')
        # Obtaining the member 'mouseover_set' of a type (line 157)
        mouseover_set_4791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 16), axes_4790, 'mouseover_set')
        # Obtaining the member 'discard' of a type (line 157)
        discard_4792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 16), mouseover_set_4791, 'discard')
        # Calling discard(args, kwargs) (line 157)
        discard_call_result_4795 = invoke(stypy.reporting.localization.Localization(__file__, 157, 16), discard_4792, *[self_4793], **kwargs_4794)
        
        
        # Assigning a Name to a Attribute (line 159):
        
        # Assigning a Name to a Attribute (line 159):
        # Getting the type of 'True' (line 159)
        True_4796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'True')
        # Getting the type of 'self' (line 159)
        self_4797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'self')
        # Obtaining the member 'axes' of a type (line 159)
        axes_4798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 16), self_4797, 'axes')
        # Setting the type of the member 'stale' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 16), axes_4798, 'stale', True_4796)
        
        # Assigning a Name to a Attribute (line 161):
        
        # Assigning a Name to a Attribute (line 161):
        # Getting the type of 'None' (line 161)
        None_4799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'None')
        # Getting the type of 'self' (line 161)
        self_4800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'self')
        # Setting the type of the member 'axes' of a type (line 161)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 16), self_4800, 'axes', None_4799)
        
        # Assigning a Name to a Name (line 162):
        
        # Assigning a Name to a Name (line 162):
        # Getting the type of 'True' (line 162)
        True_4801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'True')
        # Assigning a type to the variable '_ax_flag' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), '_ax_flag', True_4801)
        # SSA join for if statement (line 155)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 164)
        self_4802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'self')
        # Obtaining the member 'figure' of a type (line 164)
        figure_4803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 15), self_4802, 'figure')
        # Testing the type of an if condition (line 164)
        if_condition_4804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 12), figure_4803)
        # Assigning a type to the variable 'if_condition_4804' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'if_condition_4804', if_condition_4804)
        # SSA begins for if statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 165):
        
        # Assigning a Name to a Attribute (line 165):
        # Getting the type of 'None' (line 165)
        None_4805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), 'None')
        # Getting the type of 'self' (line 165)
        self_4806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'self')
        # Setting the type of the member 'figure' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), self_4806, 'figure', None_4805)
        
        
        # Getting the type of '_ax_flag' (line 166)
        _ax_flag_4807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 23), '_ax_flag')
        # Applying the 'not' unary operator (line 166)
        result_not__4808 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 19), 'not', _ax_flag_4807)
        
        # Testing the type of an if condition (line 166)
        if_condition_4809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 16), result_not__4808)
        # Assigning a type to the variable 'if_condition_4809' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'if_condition_4809', if_condition_4809)
        # SSA begins for if statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 167):
        
        # Assigning a Name to a Attribute (line 167):
        # Getting the type of 'True' (line 167)
        True_4810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 34), 'True')
        # Getting the type of 'self' (line 167)
        self_4811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'self')
        # Setting the type of the member 'figure' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 20), self_4811, 'figure', True_4810)
        # SSA join for if statement (line 166)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 164)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 150)
        module_type_store.open_ssa_branch('else')
        
        # Call to NotImplementedError(...): (line 170)
        # Processing the call arguments (line 170)
        unicode_4813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 38), 'unicode', u'cannot remove artist')
        # Processing the call keyword arguments (line 170)
        kwargs_4814 = {}
        # Getting the type of 'NotImplementedError' (line 170)
        NotImplementedError_4812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 170)
        NotImplementedError_call_result_4815 = invoke(stypy.reporting.localization.Localization(__file__, 170, 18), NotImplementedError_4812, *[unicode_4813], **kwargs_4814)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 170, 12), NotImplementedError_call_result_4815, 'raise parameter', BaseException)
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'remove(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_4816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4816)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove'
        return stypy_return_type_4816


    @norecursion
    def have_units(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'have_units'
        module_type_store = module_type_store.open_function_context('have_units', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.have_units.__dict__.__setitem__('stypy_localization', localization)
        Artist.have_units.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.have_units.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.have_units.__dict__.__setitem__('stypy_function_name', 'Artist.have_units')
        Artist.have_units.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.have_units.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.have_units.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.have_units.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.have_units.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.have_units.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.have_units.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.have_units', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'have_units', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'have_units(...)' code ##################

        unicode_4817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 8), 'unicode', u'Return *True* if units are set on the *x* or *y* axes')
        
        # Assigning a Attribute to a Name (line 179):
        
        # Assigning a Attribute to a Name (line 179):
        # Getting the type of 'self' (line 179)
        self_4818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'self')
        # Obtaining the member 'axes' of a type (line 179)
        axes_4819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 13), self_4818, 'axes')
        # Assigning a type to the variable 'ax' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'ax', axes_4819)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ax' (line 180)
        ax_4820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'ax')
        # Getting the type of 'None' (line 180)
        None_4821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'None')
        # Applying the binary operator 'is' (line 180)
        result_is__4822 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 11), 'is', ax_4820, None_4821)
        
        
        # Getting the type of 'ax' (line 180)
        ax_4823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 25), 'ax')
        # Obtaining the member 'xaxis' of a type (line 180)
        xaxis_4824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 25), ax_4823, 'xaxis')
        # Getting the type of 'None' (line 180)
        None_4825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 37), 'None')
        # Applying the binary operator 'is' (line 180)
        result_is__4826 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 25), 'is', xaxis_4824, None_4825)
        
        # Applying the binary operator 'or' (line 180)
        result_or_keyword_4827 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 11), 'or', result_is__4822, result_is__4826)
        
        # Testing the type of an if condition (line 180)
        if_condition_4828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 8), result_or_keyword_4827)
        # Assigning a type to the variable 'if_condition_4828' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'if_condition_4828', if_condition_4828)
        # SSA begins for if statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 181)
        False_4829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'stypy_return_type', False_4829)
        # SSA join for if statement (line 180)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Evaluating a boolean operation
        
        # Call to have_units(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_4833 = {}
        # Getting the type of 'ax' (line 182)
        ax_4830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'ax', False)
        # Obtaining the member 'xaxis' of a type (line 182)
        xaxis_4831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), ax_4830, 'xaxis')
        # Obtaining the member 'have_units' of a type (line 182)
        have_units_4832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), xaxis_4831, 'have_units')
        # Calling have_units(args, kwargs) (line 182)
        have_units_call_result_4834 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), have_units_4832, *[], **kwargs_4833)
        
        
        # Call to have_units(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_4838 = {}
        # Getting the type of 'ax' (line 182)
        ax_4835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 40), 'ax', False)
        # Obtaining the member 'yaxis' of a type (line 182)
        yaxis_4836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 40), ax_4835, 'yaxis')
        # Obtaining the member 'have_units' of a type (line 182)
        have_units_4837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 40), yaxis_4836, 'have_units')
        # Calling have_units(args, kwargs) (line 182)
        have_units_call_result_4839 = invoke(stypy.reporting.localization.Localization(__file__, 182, 40), have_units_4837, *[], **kwargs_4838)
        
        # Applying the binary operator 'or' (line 182)
        result_or_keyword_4840 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 15), 'or', have_units_call_result_4834, have_units_call_result_4839)
        
        # Assigning a type to the variable 'stypy_return_type' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'stypy_return_type', result_or_keyword_4840)
        
        # ################# End of 'have_units(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'have_units' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_4841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4841)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'have_units'
        return stypy_return_type_4841


    @norecursion
    def convert_xunits(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert_xunits'
        module_type_store = module_type_store.open_function_context('convert_xunits', 184, 4, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.convert_xunits.__dict__.__setitem__('stypy_localization', localization)
        Artist.convert_xunits.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.convert_xunits.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.convert_xunits.__dict__.__setitem__('stypy_function_name', 'Artist.convert_xunits')
        Artist.convert_xunits.__dict__.__setitem__('stypy_param_names_list', ['x'])
        Artist.convert_xunits.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.convert_xunits.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.convert_xunits.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.convert_xunits.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.convert_xunits.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.convert_xunits.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.convert_xunits', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert_xunits', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert_xunits(...)' code ##################

        unicode_4842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'unicode', u'For artists in an axes, if the xaxis has units support,\n        convert *x* using xaxis unit type\n        ')
        
        # Assigning a Call to a Name (line 188):
        
        # Assigning a Call to a Name (line 188):
        
        # Call to getattr(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'self' (line 188)
        self_4844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'self', False)
        unicode_4845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 27), 'unicode', u'axes')
        # Getting the type of 'None' (line 188)
        None_4846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 35), 'None', False)
        # Processing the call keyword arguments (line 188)
        kwargs_4847 = {}
        # Getting the type of 'getattr' (line 188)
        getattr_4843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 13), 'getattr', False)
        # Calling getattr(args, kwargs) (line 188)
        getattr_call_result_4848 = invoke(stypy.reporting.localization.Localization(__file__, 188, 13), getattr_4843, *[self_4844, unicode_4845, None_4846], **kwargs_4847)
        
        # Assigning a type to the variable 'ax' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'ax', getattr_call_result_4848)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ax' (line 189)
        ax_4849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'ax')
        # Getting the type of 'None' (line 189)
        None_4850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 17), 'None')
        # Applying the binary operator 'is' (line 189)
        result_is__4851 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 11), 'is', ax_4849, None_4850)
        
        
        # Getting the type of 'ax' (line 189)
        ax_4852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'ax')
        # Obtaining the member 'xaxis' of a type (line 189)
        xaxis_4853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 25), ax_4852, 'xaxis')
        # Getting the type of 'None' (line 189)
        None_4854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'None')
        # Applying the binary operator 'is' (line 189)
        result_is__4855 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 25), 'is', xaxis_4853, None_4854)
        
        # Applying the binary operator 'or' (line 189)
        result_or_keyword_4856 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 11), 'or', result_is__4851, result_is__4855)
        
        # Testing the type of an if condition (line 189)
        if_condition_4857 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 8), result_or_keyword_4856)
        # Assigning a type to the variable 'if_condition_4857' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'if_condition_4857', if_condition_4857)
        # SSA begins for if statement (line 189)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'x' (line 190)
        x_4858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'stypy_return_type', x_4858)
        # SSA join for if statement (line 189)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to convert_units(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'x' (line 191)
        x_4862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 38), 'x', False)
        # Processing the call keyword arguments (line 191)
        kwargs_4863 = {}
        # Getting the type of 'ax' (line 191)
        ax_4859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'ax', False)
        # Obtaining the member 'xaxis' of a type (line 191)
        xaxis_4860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 15), ax_4859, 'xaxis')
        # Obtaining the member 'convert_units' of a type (line 191)
        convert_units_4861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 15), xaxis_4860, 'convert_units')
        # Calling convert_units(args, kwargs) (line 191)
        convert_units_call_result_4864 = invoke(stypy.reporting.localization.Localization(__file__, 191, 15), convert_units_4861, *[x_4862], **kwargs_4863)
        
        # Assigning a type to the variable 'stypy_return_type' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'stypy_return_type', convert_units_call_result_4864)
        
        # ################# End of 'convert_xunits(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert_xunits' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_4865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4865)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert_xunits'
        return stypy_return_type_4865


    @norecursion
    def convert_yunits(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert_yunits'
        module_type_store = module_type_store.open_function_context('convert_yunits', 193, 4, False)
        # Assigning a type to the variable 'self' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.convert_yunits.__dict__.__setitem__('stypy_localization', localization)
        Artist.convert_yunits.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.convert_yunits.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.convert_yunits.__dict__.__setitem__('stypy_function_name', 'Artist.convert_yunits')
        Artist.convert_yunits.__dict__.__setitem__('stypy_param_names_list', ['y'])
        Artist.convert_yunits.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.convert_yunits.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.convert_yunits.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.convert_yunits.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.convert_yunits.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.convert_yunits.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.convert_yunits', ['y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert_yunits', localization, ['y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert_yunits(...)' code ##################

        unicode_4866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, (-1)), 'unicode', u'For artists in an axes, if the yaxis has units support,\n        convert *y* using yaxis unit type\n        ')
        
        # Assigning a Call to a Name (line 197):
        
        # Assigning a Call to a Name (line 197):
        
        # Call to getattr(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'self' (line 197)
        self_4868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 21), 'self', False)
        unicode_4869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 27), 'unicode', u'axes')
        # Getting the type of 'None' (line 197)
        None_4870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 35), 'None', False)
        # Processing the call keyword arguments (line 197)
        kwargs_4871 = {}
        # Getting the type of 'getattr' (line 197)
        getattr_4867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 13), 'getattr', False)
        # Calling getattr(args, kwargs) (line 197)
        getattr_call_result_4872 = invoke(stypy.reporting.localization.Localization(__file__, 197, 13), getattr_4867, *[self_4868, unicode_4869, None_4870], **kwargs_4871)
        
        # Assigning a type to the variable 'ax' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'ax', getattr_call_result_4872)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ax' (line 198)
        ax_4873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'ax')
        # Getting the type of 'None' (line 198)
        None_4874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'None')
        # Applying the binary operator 'is' (line 198)
        result_is__4875 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), 'is', ax_4873, None_4874)
        
        
        # Getting the type of 'ax' (line 198)
        ax_4876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 25), 'ax')
        # Obtaining the member 'yaxis' of a type (line 198)
        yaxis_4877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 25), ax_4876, 'yaxis')
        # Getting the type of 'None' (line 198)
        None_4878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 37), 'None')
        # Applying the binary operator 'is' (line 198)
        result_is__4879 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 25), 'is', yaxis_4877, None_4878)
        
        # Applying the binary operator 'or' (line 198)
        result_or_keyword_4880 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), 'or', result_is__4875, result_is__4879)
        
        # Testing the type of an if condition (line 198)
        if_condition_4881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_or_keyword_4880)
        # Assigning a type to the variable 'if_condition_4881' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_4881', if_condition_4881)
        # SSA begins for if statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'y' (line 199)
        y_4882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'stypy_return_type', y_4882)
        # SSA join for if statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to convert_units(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'y' (line 200)
        y_4886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 38), 'y', False)
        # Processing the call keyword arguments (line 200)
        kwargs_4887 = {}
        # Getting the type of 'ax' (line 200)
        ax_4883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'ax', False)
        # Obtaining the member 'yaxis' of a type (line 200)
        yaxis_4884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), ax_4883, 'yaxis')
        # Obtaining the member 'convert_units' of a type (line 200)
        convert_units_4885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), yaxis_4884, 'convert_units')
        # Calling convert_units(args, kwargs) (line 200)
        convert_units_call_result_4888 = invoke(stypy.reporting.localization.Localization(__file__, 200, 15), convert_units_4885, *[y_4886], **kwargs_4887)
        
        # Assigning a type to the variable 'stypy_return_type' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'stypy_return_type', convert_units_call_result_4888)
        
        # ################# End of 'convert_yunits(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert_yunits' in the type store
        # Getting the type of 'stypy_return_type' (line 193)
        stypy_return_type_4889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4889)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert_yunits'
        return stypy_return_type_4889


    @norecursion
    def axes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'axes'
        module_type_store = module_type_store.open_function_context('axes', 202, 4, False)
        # Assigning a type to the variable 'self' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.axes.__dict__.__setitem__('stypy_localization', localization)
        Artist.axes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.axes.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.axes.__dict__.__setitem__('stypy_function_name', 'Artist.axes')
        Artist.axes.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.axes.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.axes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.axes.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.axes.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.axes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.axes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.axes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'axes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'axes(...)' code ##################

        unicode_4890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, (-1)), 'unicode', u'\n        The :class:`~matplotlib.axes.Axes` instance the artist\n        resides in, or *None*.\n        ')
        # Getting the type of 'self' (line 208)
        self_4891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'self')
        # Obtaining the member '_axes' of a type (line 208)
        _axes_4892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 15), self_4891, '_axes')
        # Assigning a type to the variable 'stypy_return_type' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', _axes_4892)
        
        # ################# End of 'axes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'axes' in the type store
        # Getting the type of 'stypy_return_type' (line 202)
        stypy_return_type_4893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4893)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'axes'
        return stypy_return_type_4893


    @norecursion
    def axes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'axes'
        module_type_store = module_type_store.open_function_context('axes', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.axes.__dict__.__setitem__('stypy_localization', localization)
        Artist.axes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.axes.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.axes.__dict__.__setitem__('stypy_function_name', 'Artist.axes')
        Artist.axes.__dict__.__setitem__('stypy_param_names_list', ['new_axes'])
        Artist.axes.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.axes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.axes.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.axes.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.axes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.axes.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.axes', ['new_axes'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'axes', localization, ['new_axes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'axes(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'new_axes' (line 212)
        new_axes_4894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'new_axes')
        # Getting the type of 'None' (line 212)
        None_4895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'None')
        # Applying the binary operator 'isnot' (line 212)
        result_is_not_4896 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 12), 'isnot', new_axes_4894, None_4895)
        
        
        # Getting the type of 'self' (line 212)
        self_4897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'self')
        # Obtaining the member '_axes' of a type (line 212)
        _axes_4898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 37), self_4897, '_axes')
        # Getting the type of 'None' (line 212)
        None_4899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 55), 'None')
        # Applying the binary operator 'isnot' (line 212)
        result_is_not_4900 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 37), 'isnot', _axes_4898, None_4899)
        
        # Applying the binary operator 'and' (line 212)
        result_and_keyword_4901 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 12), 'and', result_is_not_4896, result_is_not_4900)
        
        # Getting the type of 'new_axes' (line 213)
        new_axes_4902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'new_axes')
        # Getting the type of 'self' (line 213)
        self_4903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 32), 'self')
        # Obtaining the member '_axes' of a type (line 213)
        _axes_4904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 32), self_4903, '_axes')
        # Applying the binary operator '!=' (line 213)
        result_ne_4905 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 20), '!=', new_axes_4902, _axes_4904)
        
        # Applying the binary operator 'and' (line 212)
        result_and_keyword_4906 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 12), 'and', result_and_keyword_4901, result_ne_4905)
        
        # Testing the type of an if condition (line 212)
        if_condition_4907 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 8), result_and_keyword_4906)
        # Assigning a type to the variable 'if_condition_4907' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'if_condition_4907', if_condition_4907)
        # SSA begins for if statement (line 212)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 214)
        # Processing the call arguments (line 214)
        unicode_4909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 29), 'unicode', u'Can not reset the axes.  You are probably trying to re-use an artist in more than one Axes which is not supported')
        # Processing the call keyword arguments (line 214)
        kwargs_4910 = {}
        # Getting the type of 'ValueError' (line 214)
        ValueError_4908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 214)
        ValueError_call_result_4911 = invoke(stypy.reporting.localization.Localization(__file__, 214, 18), ValueError_4908, *[unicode_4909], **kwargs_4910)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 214, 12), ValueError_call_result_4911, 'raise parameter', BaseException)
        # SSA join for if statement (line 212)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 217):
        
        # Assigning a Name to a Attribute (line 217):
        # Getting the type of 'new_axes' (line 217)
        new_axes_4912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'new_axes')
        # Getting the type of 'self' (line 217)
        self_4913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'self')
        # Setting the type of the member '_axes' of a type (line 217)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), self_4913, '_axes', new_axes_4912)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'new_axes' (line 218)
        new_axes_4914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'new_axes')
        # Getting the type of 'None' (line 218)
        None_4915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'None')
        # Applying the binary operator 'isnot' (line 218)
        result_is_not_4916 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 11), 'isnot', new_axes_4914, None_4915)
        
        
        # Getting the type of 'new_axes' (line 218)
        new_axes_4917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 36), 'new_axes')
        # Getting the type of 'self' (line 218)
        self_4918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 52), 'self')
        # Applying the binary operator 'isnot' (line 218)
        result_is_not_4919 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 36), 'isnot', new_axes_4917, self_4918)
        
        # Applying the binary operator 'and' (line 218)
        result_and_keyword_4920 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 11), 'and', result_is_not_4916, result_is_not_4919)
        
        # Testing the type of an if condition (line 218)
        if_condition_4921 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 8), result_and_keyword_4920)
        # Assigning a type to the variable 'if_condition_4921' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'if_condition_4921', if_condition_4921)
        # SSA begins for if statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 219):
        
        # Assigning a Name to a Attribute (line 219):
        # Getting the type of '_stale_axes_callback' (line 219)
        _stale_axes_callback_4922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 34), '_stale_axes_callback')
        # Getting the type of 'self' (line 219)
        self_4923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'self')
        # Setting the type of the member 'stale_callback' of a type (line 219)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 12), self_4923, 'stale_callback', _stale_axes_callback_4922)
        # SSA join for if statement (line 218)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new_axes' (line 220)
        new_axes_4924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), 'new_axes')
        # Assigning a type to the variable 'stypy_return_type' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'stypy_return_type', new_axes_4924)
        
        # ################# End of 'axes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'axes' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_4925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4925)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'axes'
        return stypy_return_type_4925


    @norecursion
    def stale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'stale'
        module_type_store = module_type_store.open_function_context('stale', 222, 4, False)
        # Assigning a type to the variable 'self' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.stale.__dict__.__setitem__('stypy_localization', localization)
        Artist.stale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.stale.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.stale.__dict__.__setitem__('stypy_function_name', 'Artist.stale')
        Artist.stale.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.stale.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.stale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.stale.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.stale.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.stale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.stale.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.stale', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'stale', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'stale(...)' code ##################

        unicode_4926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, (-1)), 'unicode', u"\n        If the artist is 'stale' and needs to be re-drawn for the output to\n        match the internal state of the artist.\n        ")
        # Getting the type of 'self' (line 228)
        self_4927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'self')
        # Obtaining the member '_stale' of a type (line 228)
        _stale_4928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 15), self_4927, '_stale')
        # Assigning a type to the variable 'stypy_return_type' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stypy_return_type', _stale_4928)
        
        # ################# End of 'stale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stale' in the type store
        # Getting the type of 'stypy_return_type' (line 222)
        stypy_return_type_4929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4929)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stale'
        return stypy_return_type_4929


    @norecursion
    def stale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'stale'
        module_type_store = module_type_store.open_function_context('stale', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.stale.__dict__.__setitem__('stypy_localization', localization)
        Artist.stale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.stale.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.stale.__dict__.__setitem__('stypy_function_name', 'Artist.stale')
        Artist.stale.__dict__.__setitem__('stypy_param_names_list', ['val'])
        Artist.stale.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.stale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.stale.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.stale.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.stale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.stale.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.stale', ['val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'stale', localization, ['val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'stale(...)' code ##################

        
        # Assigning a Name to a Attribute (line 232):
        
        # Assigning a Name to a Attribute (line 232):
        # Getting the type of 'val' (line 232)
        val_4930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 22), 'val')
        # Getting the type of 'self' (line 232)
        self_4931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'self')
        # Setting the type of the member '_stale' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), self_4931, '_stale', val_4930)
        
        
        # Call to get_animated(...): (line 237)
        # Processing the call keyword arguments (line 237)
        kwargs_4934 = {}
        # Getting the type of 'self' (line 237)
        self_4932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'self', False)
        # Obtaining the member 'get_animated' of a type (line 237)
        get_animated_4933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 11), self_4932, 'get_animated')
        # Calling get_animated(args, kwargs) (line 237)
        get_animated_call_result_4935 = invoke(stypy.reporting.localization.Localization(__file__, 237, 11), get_animated_4933, *[], **kwargs_4934)
        
        # Testing the type of an if condition (line 237)
        if_condition_4936 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 8), get_animated_call_result_4935)
        # Assigning a type to the variable 'if_condition_4936' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'if_condition_4936', if_condition_4936)
        # SSA begins for if statement (line 237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 237)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'val' (line 240)
        val_4937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'val')
        
        # Getting the type of 'self' (line 240)
        self_4938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'self')
        # Obtaining the member 'stale_callback' of a type (line 240)
        stale_callback_4939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 19), self_4938, 'stale_callback')
        # Getting the type of 'None' (line 240)
        None_4940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 46), 'None')
        # Applying the binary operator 'isnot' (line 240)
        result_is_not_4941 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 19), 'isnot', stale_callback_4939, None_4940)
        
        # Applying the binary operator 'and' (line 240)
        result_and_keyword_4942 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), 'and', val_4937, result_is_not_4941)
        
        # Testing the type of an if condition (line 240)
        if_condition_4943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), result_and_keyword_4942)
        # Assigning a type to the variable 'if_condition_4943' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_4943', if_condition_4943)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to stale_callback(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'self' (line 241)
        self_4946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 32), 'self', False)
        # Getting the type of 'val' (line 241)
        val_4947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 38), 'val', False)
        # Processing the call keyword arguments (line 241)
        kwargs_4948 = {}
        # Getting the type of 'self' (line 241)
        self_4944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'self', False)
        # Obtaining the member 'stale_callback' of a type (line 241)
        stale_callback_4945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), self_4944, 'stale_callback')
        # Calling stale_callback(args, kwargs) (line 241)
        stale_callback_call_result_4949 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), stale_callback_4945, *[self_4946, val_4947], **kwargs_4948)
        
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'stale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stale' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_4950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4950)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stale'
        return stypy_return_type_4950


    @norecursion
    def get_window_extent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_window_extent'
        module_type_store = module_type_store.open_function_context('get_window_extent', 243, 4, False)
        # Assigning a type to the variable 'self' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_window_extent.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_window_extent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_window_extent.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_window_extent.__dict__.__setitem__('stypy_function_name', 'Artist.get_window_extent')
        Artist.get_window_extent.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Artist.get_window_extent.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_window_extent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_window_extent.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_window_extent.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_window_extent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_window_extent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_window_extent', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_window_extent', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_window_extent(...)' code ##################

        unicode_4951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, (-1)), 'unicode', u'\n        Get the axes bounding box in display space.\n        Subclasses should override for inclusion in the bounding box\n        "tight" calculation. Default is to return an empty bounding\n        box at 0, 0.\n\n        Be careful when using this function, the results will not update\n        if the artist window extent of the artist changes.  The extent\n        can change due to any changes in the transform stack, such as\n        changing the axes limits, the figure size, or the canvas used\n        (as is done when saving a figure).  This can lead to unexpected\n        behavior where interactive figures will look fine on the screen,\n        but will save incorrectly.\n        ')
        
        # Call to Bbox(...): (line 258)
        # Processing the call arguments (line 258)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_4953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_4954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        int_4955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 21), list_4954, int_4955)
        # Adding element type (line 258)
        int_4956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 21), list_4954, int_4956)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 20), list_4953, list_4954)
        # Adding element type (line 258)
        
        # Obtaining an instance of the builtin type 'list' (line 258)
        list_4957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 258)
        # Adding element type (line 258)
        int_4958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 29), list_4957, int_4958)
        # Adding element type (line 258)
        int_4959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 29), list_4957, int_4959)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 20), list_4953, list_4957)
        
        # Processing the call keyword arguments (line 258)
        kwargs_4960 = {}
        # Getting the type of 'Bbox' (line 258)
        Bbox_4952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'Bbox', False)
        # Calling Bbox(args, kwargs) (line 258)
        Bbox_call_result_4961 = invoke(stypy.reporting.localization.Localization(__file__, 258, 15), Bbox_4952, *[list_4953], **kwargs_4960)
        
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', Bbox_call_result_4961)
        
        # ################# End of 'get_window_extent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_window_extent' in the type store
        # Getting the type of 'stypy_return_type' (line 243)
        stypy_return_type_4962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4962)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_window_extent'
        return stypy_return_type_4962


    @norecursion
    def add_callback(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_callback'
        module_type_store = module_type_store.open_function_context('add_callback', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.add_callback.__dict__.__setitem__('stypy_localization', localization)
        Artist.add_callback.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.add_callback.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.add_callback.__dict__.__setitem__('stypy_function_name', 'Artist.add_callback')
        Artist.add_callback.__dict__.__setitem__('stypy_param_names_list', ['func'])
        Artist.add_callback.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.add_callback.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.add_callback.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.add_callback.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.add_callback.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.add_callback.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.add_callback', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_callback', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_callback(...)' code ##################

        unicode_4963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'unicode', u"\n        Adds a callback function that will be called whenever one of\n        the :class:`Artist`'s properties changes.\n\n        Returns an *id* that is useful for removing the callback with\n        :meth:`remove_callback` later.\n        ")
        
        # Assigning a Attribute to a Name (line 268):
        
        # Assigning a Attribute to a Name (line 268):
        # Getting the type of 'self' (line 268)
        self_4964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 14), 'self')
        # Obtaining the member '_oid' of a type (line 268)
        _oid_4965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 14), self_4964, '_oid')
        # Assigning a type to the variable 'oid' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'oid', _oid_4965)
        
        # Assigning a Name to a Subscript (line 269):
        
        # Assigning a Name to a Subscript (line 269):
        # Getting the type of 'func' (line 269)
        func_4966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 35), 'func')
        # Getting the type of 'self' (line 269)
        self_4967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'self')
        # Obtaining the member '_propobservers' of a type (line 269)
        _propobservers_4968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), self_4967, '_propobservers')
        # Getting the type of 'oid' (line 269)
        oid_4969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 28), 'oid')
        # Storing an element on a container (line 269)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 8), _propobservers_4968, (oid_4969, func_4966))
        
        # Getting the type of 'self' (line 270)
        self_4970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self')
        # Obtaining the member '_oid' of a type (line 270)
        _oid_4971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_4970, '_oid')
        int_4972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 21), 'int')
        # Applying the binary operator '+=' (line 270)
        result_iadd_4973 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 8), '+=', _oid_4971, int_4972)
        # Getting the type of 'self' (line 270)
        self_4974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self')
        # Setting the type of the member '_oid' of a type (line 270)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_4974, '_oid', result_iadd_4973)
        
        # Getting the type of 'oid' (line 271)
        oid_4975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'oid')
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'stypy_return_type', oid_4975)
        
        # ################# End of 'add_callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_callback' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_4976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4976)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_callback'
        return stypy_return_type_4976


    @norecursion
    def remove_callback(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove_callback'
        module_type_store = module_type_store.open_function_context('remove_callback', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.remove_callback.__dict__.__setitem__('stypy_localization', localization)
        Artist.remove_callback.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.remove_callback.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.remove_callback.__dict__.__setitem__('stypy_function_name', 'Artist.remove_callback')
        Artist.remove_callback.__dict__.__setitem__('stypy_param_names_list', ['oid'])
        Artist.remove_callback.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.remove_callback.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.remove_callback.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.remove_callback.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.remove_callback.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.remove_callback.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.remove_callback', ['oid'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove_callback', localization, ['oid'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove_callback(...)' code ##################

        unicode_4977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, (-1)), 'unicode', u'\n        Remove a callback based on its *id*.\n\n        .. seealso::\n\n            :meth:`add_callback`\n               For adding callbacks\n\n        ')
        
        
        # SSA begins for try-except statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Deleting a member
        # Getting the type of 'self' (line 284)
        self_4978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'self')
        # Obtaining the member '_propobservers' of a type (line 284)
        _propobservers_4979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), self_4978, '_propobservers')
        
        # Obtaining the type of the subscript
        # Getting the type of 'oid' (line 284)
        oid_4980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 36), 'oid')
        # Getting the type of 'self' (line 284)
        self_4981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'self')
        # Obtaining the member '_propobservers' of a type (line 284)
        _propobservers_4982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), self_4981, '_propobservers')
        # Obtaining the member '__getitem__' of a type (line 284)
        getitem___4983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), _propobservers_4982, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 284)
        subscript_call_result_4984 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), getitem___4983, oid_4980)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 12), _propobservers_4979, subscript_call_result_4984)
        # SSA branch for the except part of a try statement (line 283)
        # SSA branch for the except 'KeyError' branch of a try statement (line 283)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 283)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'remove_callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove_callback' in the type store
        # Getting the type of 'stypy_return_type' (line 273)
        stypy_return_type_4985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4985)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove_callback'
        return stypy_return_type_4985


    @norecursion
    def pchanged(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pchanged'
        module_type_store = module_type_store.open_function_context('pchanged', 288, 4, False)
        # Assigning a type to the variable 'self' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.pchanged.__dict__.__setitem__('stypy_localization', localization)
        Artist.pchanged.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.pchanged.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.pchanged.__dict__.__setitem__('stypy_function_name', 'Artist.pchanged')
        Artist.pchanged.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.pchanged.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.pchanged.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.pchanged.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.pchanged.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.pchanged.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.pchanged.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.pchanged', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pchanged', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pchanged(...)' code ##################

        unicode_4986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, (-1)), 'unicode', u'\n        Fire an event when property changed, calling all of the\n        registered callbacks.\n        ')
        
        
        # Call to iteritems(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'self' (line 293)
        self_4989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 39), 'self', False)
        # Obtaining the member '_propobservers' of a type (line 293)
        _propobservers_4990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 39), self_4989, '_propobservers')
        # Processing the call keyword arguments (line 293)
        kwargs_4991 = {}
        # Getting the type of 'six' (line 293)
        six_4987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 25), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 293)
        iteritems_4988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 25), six_4987, 'iteritems')
        # Calling iteritems(args, kwargs) (line 293)
        iteritems_call_result_4992 = invoke(stypy.reporting.localization.Localization(__file__, 293, 25), iteritems_4988, *[_propobservers_4990], **kwargs_4991)
        
        # Testing the type of a for loop iterable (line 293)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 293, 8), iteritems_call_result_4992)
        # Getting the type of the for loop variable (line 293)
        for_loop_var_4993 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 293, 8), iteritems_call_result_4992)
        # Assigning a type to the variable 'oid' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'oid', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 8), for_loop_var_4993))
        # Assigning a type to the variable 'func' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 8), for_loop_var_4993))
        # SSA begins for a for statement (line 293)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to func(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'self' (line 294)
        self_4995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 17), 'self', False)
        # Processing the call keyword arguments (line 294)
        kwargs_4996 = {}
        # Getting the type of 'func' (line 294)
        func_4994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'func', False)
        # Calling func(args, kwargs) (line 294)
        func_call_result_4997 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), func_4994, *[self_4995], **kwargs_4996)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'pchanged(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pchanged' in the type store
        # Getting the type of 'stypy_return_type' (line 288)
        stypy_return_type_4998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4998)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pchanged'
        return stypy_return_type_4998


    @norecursion
    def is_transform_set(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_transform_set'
        module_type_store = module_type_store.open_function_context('is_transform_set', 296, 4, False)
        # Assigning a type to the variable 'self' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.is_transform_set.__dict__.__setitem__('stypy_localization', localization)
        Artist.is_transform_set.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.is_transform_set.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.is_transform_set.__dict__.__setitem__('stypy_function_name', 'Artist.is_transform_set')
        Artist.is_transform_set.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.is_transform_set.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.is_transform_set.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.is_transform_set.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.is_transform_set.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.is_transform_set.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.is_transform_set.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.is_transform_set', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_transform_set', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_transform_set(...)' code ##################

        unicode_4999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, (-1)), 'unicode', u'\n        Returns *True* if :class:`Artist` has a transform explicitly\n        set.\n        ')
        # Getting the type of 'self' (line 301)
        self_5000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'self')
        # Obtaining the member '_transformSet' of a type (line 301)
        _transformSet_5001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 15), self_5000, '_transformSet')
        # Assigning a type to the variable 'stypy_return_type' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'stypy_return_type', _transformSet_5001)
        
        # ################# End of 'is_transform_set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_transform_set' in the type store
        # Getting the type of 'stypy_return_type' (line 296)
        stypy_return_type_5002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5002)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_transform_set'
        return stypy_return_type_5002


    @norecursion
    def set_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_transform'
        module_type_store = module_type_store.open_function_context('set_transform', 303, 4, False)
        # Assigning a type to the variable 'self' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_transform.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_transform.__dict__.__setitem__('stypy_function_name', 'Artist.set_transform')
        Artist.set_transform.__dict__.__setitem__('stypy_param_names_list', ['t'])
        Artist.set_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_transform', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_transform', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_transform(...)' code ##################

        unicode_5003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, (-1)), 'unicode', u'\n        Set the :class:`~matplotlib.transforms.Transform` instance\n        used by this artist.\n\n        ACCEPTS: :class:`~matplotlib.transforms.Transform` instance\n        ')
        
        # Assigning a Name to a Attribute (line 310):
        
        # Assigning a Name to a Attribute (line 310):
        # Getting the type of 't' (line 310)
        t_5004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 26), 't')
        # Getting the type of 'self' (line 310)
        self_5005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'self')
        # Setting the type of the member '_transform' of a type (line 310)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), self_5005, '_transform', t_5004)
        
        # Assigning a Name to a Attribute (line 311):
        
        # Assigning a Name to a Attribute (line 311):
        # Getting the type of 'True' (line 311)
        True_5006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 29), 'True')
        # Getting the type of 'self' (line 311)
        self_5007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'self')
        # Setting the type of the member '_transformSet' of a type (line 311)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), self_5007, '_transformSet', True_5006)
        
        # Call to pchanged(...): (line 312)
        # Processing the call keyword arguments (line 312)
        kwargs_5010 = {}
        # Getting the type of 'self' (line 312)
        self_5008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 312)
        pchanged_5009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), self_5008, 'pchanged')
        # Calling pchanged(args, kwargs) (line 312)
        pchanged_call_result_5011 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), pchanged_5009, *[], **kwargs_5010)
        
        
        # Assigning a Name to a Attribute (line 313):
        
        # Assigning a Name to a Attribute (line 313):
        # Getting the type of 'True' (line 313)
        True_5012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 21), 'True')
        # Getting the type of 'self' (line 313)
        self_5013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 313)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), self_5013, 'stale', True_5012)
        
        # ################# End of 'set_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 303)
        stypy_return_type_5014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5014)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_transform'
        return stypy_return_type_5014


    @norecursion
    def get_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_transform'
        module_type_store = module_type_store.open_function_context('get_transform', 315, 4, False)
        # Assigning a type to the variable 'self' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_transform.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_transform.__dict__.__setitem__('stypy_function_name', 'Artist.get_transform')
        Artist.get_transform.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_transform.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_transform', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_transform', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_transform(...)' code ##################

        unicode_5015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, (-1)), 'unicode', u'\n        Return the :class:`~matplotlib.transforms.Transform`\n        instance used by this artist.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 320)
        # Getting the type of 'self' (line 320)
        self_5016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 11), 'self')
        # Obtaining the member '_transform' of a type (line 320)
        _transform_5017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 11), self_5016, '_transform')
        # Getting the type of 'None' (line 320)
        None_5018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 30), 'None')
        
        (may_be_5019, more_types_in_union_5020) = may_be_none(_transform_5017, None_5018)

        if may_be_5019:

            if more_types_in_union_5020:
                # Runtime conditional SSA (line 320)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 321):
            
            # Assigning a Call to a Attribute (line 321):
            
            # Call to IdentityTransform(...): (line 321)
            # Processing the call keyword arguments (line 321)
            kwargs_5022 = {}
            # Getting the type of 'IdentityTransform' (line 321)
            IdentityTransform_5021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 30), 'IdentityTransform', False)
            # Calling IdentityTransform(args, kwargs) (line 321)
            IdentityTransform_call_result_5023 = invoke(stypy.reporting.localization.Localization(__file__, 321, 30), IdentityTransform_5021, *[], **kwargs_5022)
            
            # Getting the type of 'self' (line 321)
            self_5024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'self')
            # Setting the type of the member '_transform' of a type (line 321)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 12), self_5024, '_transform', IdentityTransform_call_result_5023)

            if more_types_in_union_5020:
                # Runtime conditional SSA for else branch (line 320)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_5019) or more_types_in_union_5020):
            
            
            # Evaluating a boolean operation
            
            
            # Call to isinstance(...): (line 322)
            # Processing the call arguments (line 322)
            # Getting the type of 'self' (line 322)
            self_5026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 29), 'self', False)
            # Obtaining the member '_transform' of a type (line 322)
            _transform_5027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 29), self_5026, '_transform')
            # Getting the type of 'Transform' (line 322)
            Transform_5028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 46), 'Transform', False)
            # Processing the call keyword arguments (line 322)
            kwargs_5029 = {}
            # Getting the type of 'isinstance' (line 322)
            isinstance_5025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 18), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 322)
            isinstance_call_result_5030 = invoke(stypy.reporting.localization.Localization(__file__, 322, 18), isinstance_5025, *[_transform_5027, Transform_5028], **kwargs_5029)
            
            # Applying the 'not' unary operator (line 322)
            result_not__5031 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 14), 'not', isinstance_call_result_5030)
            
            
            # Call to hasattr(...): (line 323)
            # Processing the call arguments (line 323)
            # Getting the type of 'self' (line 323)
            self_5033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 26), 'self', False)
            # Obtaining the member '_transform' of a type (line 323)
            _transform_5034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 26), self_5033, '_transform')
            unicode_5035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 43), 'unicode', u'_as_mpl_transform')
            # Processing the call keyword arguments (line 323)
            kwargs_5036 = {}
            # Getting the type of 'hasattr' (line 323)
            hasattr_5032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 18), 'hasattr', False)
            # Calling hasattr(args, kwargs) (line 323)
            hasattr_call_result_5037 = invoke(stypy.reporting.localization.Localization(__file__, 323, 18), hasattr_5032, *[_transform_5034, unicode_5035], **kwargs_5036)
            
            # Applying the binary operator 'and' (line 322)
            result_and_keyword_5038 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 14), 'and', result_not__5031, hasattr_call_result_5037)
            
            # Testing the type of an if condition (line 322)
            if_condition_5039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 13), result_and_keyword_5038)
            # Assigning a type to the variable 'if_condition_5039' (line 322)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 13), 'if_condition_5039', if_condition_5039)
            # SSA begins for if statement (line 322)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 324):
            
            # Assigning a Call to a Attribute (line 324):
            
            # Call to _as_mpl_transform(...): (line 324)
            # Processing the call arguments (line 324)
            # Getting the type of 'self' (line 324)
            self_5043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 64), 'self', False)
            # Obtaining the member 'axes' of a type (line 324)
            axes_5044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 64), self_5043, 'axes')
            # Processing the call keyword arguments (line 324)
            kwargs_5045 = {}
            # Getting the type of 'self' (line 324)
            self_5040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 30), 'self', False)
            # Obtaining the member '_transform' of a type (line 324)
            _transform_5041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 30), self_5040, '_transform')
            # Obtaining the member '_as_mpl_transform' of a type (line 324)
            _as_mpl_transform_5042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 30), _transform_5041, '_as_mpl_transform')
            # Calling _as_mpl_transform(args, kwargs) (line 324)
            _as_mpl_transform_call_result_5046 = invoke(stypy.reporting.localization.Localization(__file__, 324, 30), _as_mpl_transform_5042, *[axes_5044], **kwargs_5045)
            
            # Getting the type of 'self' (line 324)
            self_5047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'self')
            # Setting the type of the member '_transform' of a type (line 324)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), self_5047, '_transform', _as_mpl_transform_call_result_5046)
            # SSA join for if statement (line 322)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_5019 and more_types_in_union_5020):
                # SSA join for if statement (line 320)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 325)
        self_5048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'self')
        # Obtaining the member '_transform' of a type (line 325)
        _transform_5049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 15), self_5048, '_transform')
        # Assigning a type to the variable 'stypy_return_type' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'stypy_return_type', _transform_5049)
        
        # ################# End of 'get_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 315)
        stypy_return_type_5050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5050)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_transform'
        return stypy_return_type_5050


    @norecursion
    def hitlist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'hitlist'
        module_type_store = module_type_store.open_function_context('hitlist', 327, 4, False)
        # Assigning a type to the variable 'self' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.hitlist.__dict__.__setitem__('stypy_localization', localization)
        Artist.hitlist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.hitlist.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.hitlist.__dict__.__setitem__('stypy_function_name', 'Artist.hitlist')
        Artist.hitlist.__dict__.__setitem__('stypy_param_names_list', ['event'])
        Artist.hitlist.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.hitlist.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.hitlist.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.hitlist.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.hitlist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.hitlist.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.hitlist', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'hitlist', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'hitlist(...)' code ##################

        unicode_5051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, (-1)), 'unicode', u'\n        List the children of the artist which contain the mouse event *event*.\n        ')
        
        # Assigning a List to a Name (line 331):
        
        # Assigning a List to a Name (line 331):
        
        # Obtaining an instance of the builtin type 'list' (line 331)
        list_5052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 331)
        
        # Assigning a type to the variable 'L' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'L', list_5052)
        
        
        # SSA begins for try-except statement (line 332)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Tuple (line 333):
        
        # Assigning a Call to a Name:
        
        # Call to contains(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'event' (line 333)
        event_5055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 44), 'event', False)
        # Processing the call keyword arguments (line 333)
        kwargs_5056 = {}
        # Getting the type of 'self' (line 333)
        self_5053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 30), 'self', False)
        # Obtaining the member 'contains' of a type (line 333)
        contains_5054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 30), self_5053, 'contains')
        # Calling contains(args, kwargs) (line 333)
        contains_call_result_5057 = invoke(stypy.reporting.localization.Localization(__file__, 333, 30), contains_5054, *[event_5055], **kwargs_5056)
        
        # Assigning a type to the variable 'call_assignment_4592' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'call_assignment_4592', contains_call_result_5057)
        
        # Assigning a Call to a Name (line 333):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_5060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 12), 'int')
        # Processing the call keyword arguments
        kwargs_5061 = {}
        # Getting the type of 'call_assignment_4592' (line 333)
        call_assignment_4592_5058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'call_assignment_4592', False)
        # Obtaining the member '__getitem__' of a type (line 333)
        getitem___5059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), call_assignment_4592_5058, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_5062 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___5059, *[int_5060], **kwargs_5061)
        
        # Assigning a type to the variable 'call_assignment_4593' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'call_assignment_4593', getitem___call_result_5062)
        
        # Assigning a Name to a Name (line 333):
        # Getting the type of 'call_assignment_4593' (line 333)
        call_assignment_4593_5063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'call_assignment_4593')
        # Assigning a type to the variable 'hascursor' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'hascursor', call_assignment_4593_5063)
        
        # Assigning a Call to a Name (line 333):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_5066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 12), 'int')
        # Processing the call keyword arguments
        kwargs_5067 = {}
        # Getting the type of 'call_assignment_4592' (line 333)
        call_assignment_4592_5064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'call_assignment_4592', False)
        # Obtaining the member '__getitem__' of a type (line 333)
        getitem___5065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), call_assignment_4592_5064, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_5068 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___5065, *[int_5066], **kwargs_5067)
        
        # Assigning a type to the variable 'call_assignment_4594' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'call_assignment_4594', getitem___call_result_5068)
        
        # Assigning a Name to a Name (line 333):
        # Getting the type of 'call_assignment_4594' (line 333)
        call_assignment_4594_5069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'call_assignment_4594')
        # Assigning a type to the variable 'info' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 23), 'info', call_assignment_4594_5069)
        
        # Getting the type of 'hascursor' (line 334)
        hascursor_5070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 15), 'hascursor')
        # Testing the type of an if condition (line 334)
        if_condition_5071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 12), hascursor_5070)
        # Assigning a type to the variable 'if_condition_5071' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'if_condition_5071', if_condition_5071)
        # SSA begins for if statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'self' (line 335)
        self_5074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 25), 'self', False)
        # Processing the call keyword arguments (line 335)
        kwargs_5075 = {}
        # Getting the type of 'L' (line 335)
        L_5072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'L', False)
        # Obtaining the member 'append' of a type (line 335)
        append_5073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 16), L_5072, 'append')
        # Calling append(args, kwargs) (line 335)
        append_call_result_5076 = invoke(stypy.reporting.localization.Localization(__file__, 335, 16), append_5073, *[self_5074], **kwargs_5075)
        
        # SSA join for if statement (line 334)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 332)
        # SSA branch for the except '<any exception>' branch of a try statement (line 332)
        module_type_store.open_ssa_branch('except')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 337, 12))
        
        # 'import traceback' statement (line 337)
        import traceback

        import_module(stypy.reporting.localization.Localization(__file__, 337, 12), 'traceback', traceback, module_type_store)
        
        
        # Call to print_exc(...): (line 338)
        # Processing the call keyword arguments (line 338)
        kwargs_5079 = {}
        # Getting the type of 'traceback' (line 338)
        traceback_5077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'traceback', False)
        # Obtaining the member 'print_exc' of a type (line 338)
        print_exc_5078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), traceback_5077, 'print_exc')
        # Calling print_exc(args, kwargs) (line 338)
        print_exc_call_result_5080 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), print_exc_5078, *[], **kwargs_5079)
        
        
        # Call to print(...): (line 339)
        # Processing the call arguments (line 339)
        unicode_5082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 18), 'unicode', u'while checking')
        # Getting the type of 'self' (line 339)
        self_5083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 36), 'self', False)
        # Obtaining the member '__class__' of a type (line 339)
        class___5084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 36), self_5083, '__class__')
        # Processing the call keyword arguments (line 339)
        kwargs_5085 = {}
        # Getting the type of 'print' (line 339)
        print_5081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'print', False)
        # Calling print(args, kwargs) (line 339)
        print_call_result_5086 = invoke(stypy.reporting.localization.Localization(__file__, 339, 12), print_5081, *[unicode_5082, class___5084], **kwargs_5085)
        
        # SSA join for try-except statement (line 332)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to get_children(...): (line 341)
        # Processing the call keyword arguments (line 341)
        kwargs_5089 = {}
        # Getting the type of 'self' (line 341)
        self_5087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 17), 'self', False)
        # Obtaining the member 'get_children' of a type (line 341)
        get_children_5088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 17), self_5087, 'get_children')
        # Calling get_children(args, kwargs) (line 341)
        get_children_call_result_5090 = invoke(stypy.reporting.localization.Localization(__file__, 341, 17), get_children_5088, *[], **kwargs_5089)
        
        # Testing the type of a for loop iterable (line 341)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 341, 8), get_children_call_result_5090)
        # Getting the type of the for loop variable (line 341)
        for_loop_var_5091 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 341, 8), get_children_call_result_5090)
        # Assigning a type to the variable 'a' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'a', for_loop_var_5091)
        # SSA begins for a for statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to extend(...): (line 342)
        # Processing the call arguments (line 342)
        
        # Call to hitlist(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'event' (line 342)
        event_5096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 31), 'event', False)
        # Processing the call keyword arguments (line 342)
        kwargs_5097 = {}
        # Getting the type of 'a' (line 342)
        a_5094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 21), 'a', False)
        # Obtaining the member 'hitlist' of a type (line 342)
        hitlist_5095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 21), a_5094, 'hitlist')
        # Calling hitlist(args, kwargs) (line 342)
        hitlist_call_result_5098 = invoke(stypy.reporting.localization.Localization(__file__, 342, 21), hitlist_5095, *[event_5096], **kwargs_5097)
        
        # Processing the call keyword arguments (line 342)
        kwargs_5099 = {}
        # Getting the type of 'L' (line 342)
        L_5092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'L', False)
        # Obtaining the member 'extend' of a type (line 342)
        extend_5093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 12), L_5092, 'extend')
        # Calling extend(args, kwargs) (line 342)
        extend_call_result_5100 = invoke(stypy.reporting.localization.Localization(__file__, 342, 12), extend_5093, *[hitlist_call_result_5098], **kwargs_5099)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'L' (line 343)
        L_5101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 15), 'L')
        # Assigning a type to the variable 'stypy_return_type' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'stypy_return_type', L_5101)
        
        # ################# End of 'hitlist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'hitlist' in the type store
        # Getting the type of 'stypy_return_type' (line 327)
        stypy_return_type_5102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5102)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'hitlist'
        return stypy_return_type_5102


    @norecursion
    def get_children(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_children'
        module_type_store = module_type_store.open_function_context('get_children', 345, 4, False)
        # Assigning a type to the variable 'self' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_children.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_children.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_children.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_children.__dict__.__setitem__('stypy_function_name', 'Artist.get_children')
        Artist.get_children.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_children.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_children.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_children.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_children.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_children.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_children.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_children', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_children', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_children(...)' code ##################

        unicode_5103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, (-1)), 'unicode', u'\n        Return a list of the child :class:`Artist`s this\n        :class:`Artist` contains.\n        ')
        
        # Obtaining an instance of the builtin type 'list' (line 350)
        list_5104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 350)
        
        # Assigning a type to the variable 'stypy_return_type' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'stypy_return_type', list_5104)
        
        # ################# End of 'get_children(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_children' in the type store
        # Getting the type of 'stypy_return_type' (line 345)
        stypy_return_type_5105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5105)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_children'
        return stypy_return_type_5105


    @norecursion
    def contains(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'contains'
        module_type_store = module_type_store.open_function_context('contains', 352, 4, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.contains.__dict__.__setitem__('stypy_localization', localization)
        Artist.contains.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.contains.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.contains.__dict__.__setitem__('stypy_function_name', 'Artist.contains')
        Artist.contains.__dict__.__setitem__('stypy_param_names_list', ['mouseevent'])
        Artist.contains.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.contains.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.contains.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.contains.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.contains.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.contains.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.contains', ['mouseevent'], None, None, defaults, varargs, kwargs)

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

        unicode_5106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, (-1)), 'unicode', u'Test whether the artist contains the mouse event.\n\n        Returns the truth value and a dictionary of artist specific details of\n        selection, such as which points are contained in the pick radius.  See\n        individual artists for details.\n        ')
        
        
        # Call to callable(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'self' (line 359)
        self_5108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 20), 'self', False)
        # Obtaining the member '_contains' of a type (line 359)
        _contains_5109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 20), self_5108, '_contains')
        # Processing the call keyword arguments (line 359)
        kwargs_5110 = {}
        # Getting the type of 'callable' (line 359)
        callable_5107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 11), 'callable', False)
        # Calling callable(args, kwargs) (line 359)
        callable_call_result_5111 = invoke(stypy.reporting.localization.Localization(__file__, 359, 11), callable_5107, *[_contains_5109], **kwargs_5110)
        
        # Testing the type of an if condition (line 359)
        if_condition_5112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 8), callable_call_result_5111)
        # Assigning a type to the variable 'if_condition_5112' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'if_condition_5112', if_condition_5112)
        # SSA begins for if statement (line 359)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _contains(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'self' (line 360)
        self_5115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 34), 'self', False)
        # Getting the type of 'mouseevent' (line 360)
        mouseevent_5116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 40), 'mouseevent', False)
        # Processing the call keyword arguments (line 360)
        kwargs_5117 = {}
        # Getting the type of 'self' (line 360)
        self_5113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 19), 'self', False)
        # Obtaining the member '_contains' of a type (line 360)
        _contains_5114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 19), self_5113, '_contains')
        # Calling _contains(args, kwargs) (line 360)
        _contains_call_result_5118 = invoke(stypy.reporting.localization.Localization(__file__, 360, 19), _contains_5114, *[self_5115, mouseevent_5116], **kwargs_5117)
        
        # Assigning a type to the variable 'stypy_return_type' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'stypy_return_type', _contains_call_result_5118)
        # SSA join for if statement (line 359)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to warn(...): (line 361)
        # Processing the call arguments (line 361)
        unicode_5121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 22), 'unicode', u"'%s' needs 'contains' method")
        # Getting the type of 'self' (line 361)
        self_5122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 55), 'self', False)
        # Obtaining the member '__class__' of a type (line 361)
        class___5123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 55), self_5122, '__class__')
        # Obtaining the member '__name__' of a type (line 361)
        name___5124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 55), class___5123, '__name__')
        # Applying the binary operator '%' (line 361)
        result_mod_5125 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 22), '%', unicode_5121, name___5124)
        
        # Processing the call keyword arguments (line 361)
        kwargs_5126 = {}
        # Getting the type of 'warnings' (line 361)
        warnings_5119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 361)
        warn_5120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), warnings_5119, 'warn')
        # Calling warn(args, kwargs) (line 361)
        warn_call_result_5127 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), warn_5120, *[result_mod_5125], **kwargs_5126)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 362)
        tuple_5128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 362)
        # Adding element type (line 362)
        # Getting the type of 'False' (line 362)
        False_5129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 15), tuple_5128, False_5129)
        # Adding element type (line 362)
        
        # Obtaining an instance of the builtin type 'dict' (line 362)
        dict_5130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 362)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 15), tuple_5128, dict_5130)
        
        # Assigning a type to the variable 'stypy_return_type' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'stypy_return_type', tuple_5128)
        
        # ################# End of 'contains(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'contains' in the type store
        # Getting the type of 'stypy_return_type' (line 352)
        stypy_return_type_5131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5131)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'contains'
        return stypy_return_type_5131


    @norecursion
    def set_contains(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_contains'
        module_type_store = module_type_store.open_function_context('set_contains', 364, 4, False)
        # Assigning a type to the variable 'self' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_contains.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_contains.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_contains.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_contains.__dict__.__setitem__('stypy_function_name', 'Artist.set_contains')
        Artist.set_contains.__dict__.__setitem__('stypy_param_names_list', ['picker'])
        Artist.set_contains.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_contains.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_contains.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_contains.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_contains.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_contains.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_contains', ['picker'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_contains', localization, ['picker'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_contains(...)' code ##################

        unicode_5132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, (-1)), 'unicode', u'\n        Replace the contains test used by this artist. The new picker\n        should be a callable function which determines whether the\n        artist is hit by the mouse event::\n\n            hit, props = picker(artist, mouseevent)\n\n        If the mouse event is over the artist, return *hit* = *True*\n        and *props* is a dictionary of properties you want returned\n        with the contains test.\n\n        ACCEPTS: a callable function\n        ')
        
        # Assigning a Name to a Attribute (line 378):
        
        # Assigning a Name to a Attribute (line 378):
        # Getting the type of 'picker' (line 378)
        picker_5133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 25), 'picker')
        # Getting the type of 'self' (line 378)
        self_5134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self')
        # Setting the type of the member '_contains' of a type (line 378)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_5134, '_contains', picker_5133)
        
        # ################# End of 'set_contains(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_contains' in the type store
        # Getting the type of 'stypy_return_type' (line 364)
        stypy_return_type_5135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5135)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_contains'
        return stypy_return_type_5135


    @norecursion
    def get_contains(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_contains'
        module_type_store = module_type_store.open_function_context('get_contains', 380, 4, False)
        # Assigning a type to the variable 'self' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_contains.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_contains.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_contains.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_contains.__dict__.__setitem__('stypy_function_name', 'Artist.get_contains')
        Artist.get_contains.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_contains.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_contains.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_contains.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_contains.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_contains.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_contains.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_contains', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_contains', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_contains(...)' code ##################

        unicode_5136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, (-1)), 'unicode', u'\n        Return the _contains test used by the artist, or *None* for default.\n        ')
        # Getting the type of 'self' (line 384)
        self_5137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 15), 'self')
        # Obtaining the member '_contains' of a type (line 384)
        _contains_5138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 15), self_5137, '_contains')
        # Assigning a type to the variable 'stypy_return_type' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'stypy_return_type', _contains_5138)
        
        # ################# End of 'get_contains(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_contains' in the type store
        # Getting the type of 'stypy_return_type' (line 380)
        stypy_return_type_5139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5139)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_contains'
        return stypy_return_type_5139


    @norecursion
    def pickable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pickable'
        module_type_store = module_type_store.open_function_context('pickable', 386, 4, False)
        # Assigning a type to the variable 'self' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.pickable.__dict__.__setitem__('stypy_localization', localization)
        Artist.pickable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.pickable.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.pickable.__dict__.__setitem__('stypy_function_name', 'Artist.pickable')
        Artist.pickable.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.pickable.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.pickable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.pickable.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.pickable.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.pickable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.pickable.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.pickable', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pickable', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pickable(...)' code ##################

        unicode_5140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 8), 'unicode', u'Return *True* if :class:`Artist` is pickable.')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 388)
        self_5141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 16), 'self')
        # Obtaining the member 'figure' of a type (line 388)
        figure_5142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 16), self_5141, 'figure')
        # Getting the type of 'None' (line 388)
        None_5143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 35), 'None')
        # Applying the binary operator 'isnot' (line 388)
        result_is_not_5144 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 16), 'isnot', figure_5142, None_5143)
        
        
        # Getting the type of 'self' (line 389)
        self_5145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'self')
        # Obtaining the member 'figure' of a type (line 389)
        figure_5146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 16), self_5145, 'figure')
        # Obtaining the member 'canvas' of a type (line 389)
        canvas_5147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 16), figure_5146, 'canvas')
        # Getting the type of 'None' (line 389)
        None_5148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 42), 'None')
        # Applying the binary operator 'isnot' (line 389)
        result_is_not_5149 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 16), 'isnot', canvas_5147, None_5148)
        
        # Applying the binary operator 'and' (line 388)
        result_and_keyword_5150 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 16), 'and', result_is_not_5144, result_is_not_5149)
        
        # Getting the type of 'self' (line 390)
        self_5151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'self')
        # Obtaining the member '_picker' of a type (line 390)
        _picker_5152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 16), self_5151, '_picker')
        # Getting the type of 'None' (line 390)
        None_5153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 36), 'None')
        # Applying the binary operator 'isnot' (line 390)
        result_is_not_5154 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 16), 'isnot', _picker_5152, None_5153)
        
        # Applying the binary operator 'and' (line 388)
        result_and_keyword_5155 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 16), 'and', result_and_keyword_5150, result_is_not_5154)
        
        # Assigning a type to the variable 'stypy_return_type' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'stypy_return_type', result_and_keyword_5155)
        
        # ################# End of 'pickable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pickable' in the type store
        # Getting the type of 'stypy_return_type' (line 386)
        stypy_return_type_5156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5156)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pickable'
        return stypy_return_type_5156


    @norecursion
    def pick(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pick'
        module_type_store = module_type_store.open_function_context('pick', 392, 4, False)
        # Assigning a type to the variable 'self' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.pick.__dict__.__setitem__('stypy_localization', localization)
        Artist.pick.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.pick.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.pick.__dict__.__setitem__('stypy_function_name', 'Artist.pick')
        Artist.pick.__dict__.__setitem__('stypy_param_names_list', ['mouseevent'])
        Artist.pick.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.pick.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.pick.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.pick.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.pick.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.pick.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.pick', ['mouseevent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pick', localization, ['mouseevent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pick(...)' code ##################

        unicode_5157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, (-1)), 'unicode', u'\n        Process pick event\n\n        each child artist will fire a pick event if *mouseevent* is over\n        the artist and the artist has picker set\n        ')
        
        
        # Call to pickable(...): (line 400)
        # Processing the call keyword arguments (line 400)
        kwargs_5160 = {}
        # Getting the type of 'self' (line 400)
        self_5158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 11), 'self', False)
        # Obtaining the member 'pickable' of a type (line 400)
        pickable_5159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 11), self_5158, 'pickable')
        # Calling pickable(args, kwargs) (line 400)
        pickable_call_result_5161 = invoke(stypy.reporting.localization.Localization(__file__, 400, 11), pickable_5159, *[], **kwargs_5160)
        
        # Testing the type of an if condition (line 400)
        if_condition_5162 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 400, 8), pickable_call_result_5161)
        # Assigning a type to the variable 'if_condition_5162' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'if_condition_5162', if_condition_5162)
        # SSA begins for if statement (line 400)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 401):
        
        # Assigning a Call to a Name (line 401):
        
        # Call to get_picker(...): (line 401)
        # Processing the call keyword arguments (line 401)
        kwargs_5165 = {}
        # Getting the type of 'self' (line 401)
        self_5163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 21), 'self', False)
        # Obtaining the member 'get_picker' of a type (line 401)
        get_picker_5164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 21), self_5163, 'get_picker')
        # Calling get_picker(args, kwargs) (line 401)
        get_picker_call_result_5166 = invoke(stypy.reporting.localization.Localization(__file__, 401, 21), get_picker_5164, *[], **kwargs_5165)
        
        # Assigning a type to the variable 'picker' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'picker', get_picker_call_result_5166)
        
        
        # Call to callable(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'picker' (line 402)
        picker_5168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 24), 'picker', False)
        # Processing the call keyword arguments (line 402)
        kwargs_5169 = {}
        # Getting the type of 'callable' (line 402)
        callable_5167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'callable', False)
        # Calling callable(args, kwargs) (line 402)
        callable_call_result_5170 = invoke(stypy.reporting.localization.Localization(__file__, 402, 15), callable_5167, *[picker_5168], **kwargs_5169)
        
        # Testing the type of an if condition (line 402)
        if_condition_5171 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 12), callable_call_result_5170)
        # Assigning a type to the variable 'if_condition_5171' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'if_condition_5171', if_condition_5171)
        # SSA begins for if statement (line 402)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 403):
        
        # Assigning a Call to a Name:
        
        # Call to picker(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'self' (line 403)
        self_5173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 38), 'self', False)
        # Getting the type of 'mouseevent' (line 403)
        mouseevent_5174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 44), 'mouseevent', False)
        # Processing the call keyword arguments (line 403)
        kwargs_5175 = {}
        # Getting the type of 'picker' (line 403)
        picker_5172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 31), 'picker', False)
        # Calling picker(args, kwargs) (line 403)
        picker_call_result_5176 = invoke(stypy.reporting.localization.Localization(__file__, 403, 31), picker_5172, *[self_5173, mouseevent_5174], **kwargs_5175)
        
        # Assigning a type to the variable 'call_assignment_4595' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'call_assignment_4595', picker_call_result_5176)
        
        # Assigning a Call to a Name (line 403):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_5179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 16), 'int')
        # Processing the call keyword arguments
        kwargs_5180 = {}
        # Getting the type of 'call_assignment_4595' (line 403)
        call_assignment_4595_5177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'call_assignment_4595', False)
        # Obtaining the member '__getitem__' of a type (line 403)
        getitem___5178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 16), call_assignment_4595_5177, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_5181 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___5178, *[int_5179], **kwargs_5180)
        
        # Assigning a type to the variable 'call_assignment_4596' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'call_assignment_4596', getitem___call_result_5181)
        
        # Assigning a Name to a Name (line 403):
        # Getting the type of 'call_assignment_4596' (line 403)
        call_assignment_4596_5182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'call_assignment_4596')
        # Assigning a type to the variable 'inside' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'inside', call_assignment_4596_5182)
        
        # Assigning a Call to a Name (line 403):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_5185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 16), 'int')
        # Processing the call keyword arguments
        kwargs_5186 = {}
        # Getting the type of 'call_assignment_4595' (line 403)
        call_assignment_4595_5183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'call_assignment_4595', False)
        # Obtaining the member '__getitem__' of a type (line 403)
        getitem___5184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 16), call_assignment_4595_5183, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_5187 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___5184, *[int_5185], **kwargs_5186)
        
        # Assigning a type to the variable 'call_assignment_4597' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'call_assignment_4597', getitem___call_result_5187)
        
        # Assigning a Name to a Name (line 403):
        # Getting the type of 'call_assignment_4597' (line 403)
        call_assignment_4597_5188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'call_assignment_4597')
        # Assigning a type to the variable 'prop' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 24), 'prop', call_assignment_4597_5188)
        # SSA branch for the else part of an if statement (line 402)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 405):
        
        # Assigning a Call to a Name:
        
        # Call to contains(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'mouseevent' (line 405)
        mouseevent_5191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 45), 'mouseevent', False)
        # Processing the call keyword arguments (line 405)
        kwargs_5192 = {}
        # Getting the type of 'self' (line 405)
        self_5189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 31), 'self', False)
        # Obtaining the member 'contains' of a type (line 405)
        contains_5190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 31), self_5189, 'contains')
        # Calling contains(args, kwargs) (line 405)
        contains_call_result_5193 = invoke(stypy.reporting.localization.Localization(__file__, 405, 31), contains_5190, *[mouseevent_5191], **kwargs_5192)
        
        # Assigning a type to the variable 'call_assignment_4598' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'call_assignment_4598', contains_call_result_5193)
        
        # Assigning a Call to a Name (line 405):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_5196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 16), 'int')
        # Processing the call keyword arguments
        kwargs_5197 = {}
        # Getting the type of 'call_assignment_4598' (line 405)
        call_assignment_4598_5194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'call_assignment_4598', False)
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___5195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 16), call_assignment_4598_5194, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_5198 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___5195, *[int_5196], **kwargs_5197)
        
        # Assigning a type to the variable 'call_assignment_4599' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'call_assignment_4599', getitem___call_result_5198)
        
        # Assigning a Name to a Name (line 405):
        # Getting the type of 'call_assignment_4599' (line 405)
        call_assignment_4599_5199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'call_assignment_4599')
        # Assigning a type to the variable 'inside' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'inside', call_assignment_4599_5199)
        
        # Assigning a Call to a Name (line 405):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_5202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 16), 'int')
        # Processing the call keyword arguments
        kwargs_5203 = {}
        # Getting the type of 'call_assignment_4598' (line 405)
        call_assignment_4598_5200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'call_assignment_4598', False)
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___5201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 16), call_assignment_4598_5200, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_5204 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___5201, *[int_5202], **kwargs_5203)
        
        # Assigning a type to the variable 'call_assignment_4600' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'call_assignment_4600', getitem___call_result_5204)
        
        # Assigning a Name to a Name (line 405):
        # Getting the type of 'call_assignment_4600' (line 405)
        call_assignment_4600_5205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'call_assignment_4600')
        # Assigning a type to the variable 'prop' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 24), 'prop', call_assignment_4600_5205)
        # SSA join for if statement (line 402)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'inside' (line 406)
        inside_5206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 15), 'inside')
        # Testing the type of an if condition (line 406)
        if_condition_5207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 406, 12), inside_5206)
        # Assigning a type to the variable 'if_condition_5207' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'if_condition_5207', if_condition_5207)
        # SSA begins for if statement (line 406)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to pick_event(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'mouseevent' (line 407)
        mouseevent_5212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 46), 'mouseevent', False)
        # Getting the type of 'self' (line 407)
        self_5213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 58), 'self', False)
        # Processing the call keyword arguments (line 407)
        # Getting the type of 'prop' (line 407)
        prop_5214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 66), 'prop', False)
        kwargs_5215 = {'prop_5214': prop_5214}
        # Getting the type of 'self' (line 407)
        self_5208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 'self', False)
        # Obtaining the member 'figure' of a type (line 407)
        figure_5209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 16), self_5208, 'figure')
        # Obtaining the member 'canvas' of a type (line 407)
        canvas_5210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 16), figure_5209, 'canvas')
        # Obtaining the member 'pick_event' of a type (line 407)
        pick_event_5211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 16), canvas_5210, 'pick_event')
        # Calling pick_event(args, kwargs) (line 407)
        pick_event_call_result_5216 = invoke(stypy.reporting.localization.Localization(__file__, 407, 16), pick_event_5211, *[mouseevent_5212, self_5213], **kwargs_5215)
        
        # SSA join for if statement (line 406)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 400)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to get_children(...): (line 410)
        # Processing the call keyword arguments (line 410)
        kwargs_5219 = {}
        # Getting the type of 'self' (line 410)
        self_5217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 17), 'self', False)
        # Obtaining the member 'get_children' of a type (line 410)
        get_children_5218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 17), self_5217, 'get_children')
        # Calling get_children(args, kwargs) (line 410)
        get_children_call_result_5220 = invoke(stypy.reporting.localization.Localization(__file__, 410, 17), get_children_5218, *[], **kwargs_5219)
        
        # Testing the type of a for loop iterable (line 410)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 410, 8), get_children_call_result_5220)
        # Getting the type of the for loop variable (line 410)
        for_loop_var_5221 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 410, 8), get_children_call_result_5220)
        # Assigning a type to the variable 'a' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'a', for_loop_var_5221)
        # SSA begins for a for statement (line 410)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 412):
        
        # Assigning a Call to a Name (line 412):
        
        # Call to getattr(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'a' (line 412)
        a_5223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 25), 'a', False)
        unicode_5224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 28), 'unicode', u'axes')
        # Getting the type of 'None' (line 412)
        None_5225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 36), 'None', False)
        # Processing the call keyword arguments (line 412)
        kwargs_5226 = {}
        # Getting the type of 'getattr' (line 412)
        getattr_5222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 17), 'getattr', False)
        # Calling getattr(args, kwargs) (line 412)
        getattr_call_result_5227 = invoke(stypy.reporting.localization.Localization(__file__, 412, 17), getattr_5222, *[a_5223, unicode_5224, None_5225], **kwargs_5226)
        
        # Assigning a type to the variable 'ax' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'ax', getattr_call_result_5227)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'mouseevent' (line 413)
        mouseevent_5228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 16), 'mouseevent')
        # Obtaining the member 'inaxes' of a type (line 413)
        inaxes_5229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 16), mouseevent_5228, 'inaxes')
        # Getting the type of 'None' (line 413)
        None_5230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 37), 'None')
        # Applying the binary operator 'is' (line 413)
        result_is__5231 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 16), 'is', inaxes_5229, None_5230)
        
        
        # Getting the type of 'ax' (line 413)
        ax_5232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 45), 'ax')
        # Getting the type of 'None' (line 413)
        None_5233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 51), 'None')
        # Applying the binary operator 'is' (line 413)
        result_is__5234 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 45), 'is', ax_5232, None_5233)
        
        # Applying the binary operator 'or' (line 413)
        result_or_keyword_5235 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 16), 'or', result_is__5231, result_is__5234)
        
        # Getting the type of 'mouseevent' (line 414)
        mouseevent_5236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 23), 'mouseevent')
        # Obtaining the member 'inaxes' of a type (line 414)
        inaxes_5237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 23), mouseevent_5236, 'inaxes')
        # Getting the type of 'ax' (line 414)
        ax_5238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 44), 'ax')
        # Applying the binary operator '==' (line 414)
        result_eq_5239 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 23), '==', inaxes_5237, ax_5238)
        
        # Applying the binary operator 'or' (line 413)
        result_or_keyword_5240 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 16), 'or', result_or_keyword_5235, result_eq_5239)
        
        # Testing the type of an if condition (line 413)
        if_condition_5241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 12), result_or_keyword_5240)
        # Assigning a type to the variable 'if_condition_5241' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'if_condition_5241', if_condition_5241)
        # SSA begins for if statement (line 413)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to pick(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'mouseevent' (line 421)
        mouseevent_5244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 23), 'mouseevent', False)
        # Processing the call keyword arguments (line 421)
        kwargs_5245 = {}
        # Getting the type of 'a' (line 421)
        a_5242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 16), 'a', False)
        # Obtaining the member 'pick' of a type (line 421)
        pick_5243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 16), a_5242, 'pick')
        # Calling pick(args, kwargs) (line 421)
        pick_call_result_5246 = invoke(stypy.reporting.localization.Localization(__file__, 421, 16), pick_5243, *[mouseevent_5244], **kwargs_5245)
        
        # SSA join for if statement (line 413)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'pick(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pick' in the type store
        # Getting the type of 'stypy_return_type' (line 392)
        stypy_return_type_5247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5247)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pick'
        return stypy_return_type_5247


    @norecursion
    def set_picker(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_picker'
        module_type_store = module_type_store.open_function_context('set_picker', 423, 4, False)
        # Assigning a type to the variable 'self' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_picker.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_picker.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_picker.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_picker.__dict__.__setitem__('stypy_function_name', 'Artist.set_picker')
        Artist.set_picker.__dict__.__setitem__('stypy_param_names_list', ['picker'])
        Artist.set_picker.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_picker.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_picker.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_picker.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_picker.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_picker.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_picker', ['picker'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_picker', localization, ['picker'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_picker(...)' code ##################

        unicode_5248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, (-1)), 'unicode', u"\n        Set the epsilon for picking used by this artist\n\n        *picker* can be one of the following:\n\n          * *None*: picking is disabled for this artist (default)\n\n          * A boolean: if *True* then picking will be enabled and the\n            artist will fire a pick event if the mouse event is over\n            the artist\n\n          * A float: if picker is a number it is interpreted as an\n            epsilon tolerance in points and the artist will fire\n            off an event if it's data is within epsilon of the mouse\n            event.  For some artists like lines and patch collections,\n            the artist may provide additional data to the pick event\n            that is generated, e.g., the indices of the data within\n            epsilon of the pick event\n\n          * A function: if picker is callable, it is a user supplied\n            function which determines whether the artist is hit by the\n            mouse event::\n\n              hit, props = picker(artist, mouseevent)\n\n            to determine the hit test.  if the mouse event is over the\n            artist, return *hit=True* and props is a dictionary of\n            properties you want added to the PickEvent attributes.\n\n        ACCEPTS: [None|float|boolean|callable]\n        ")
        
        # Assigning a Name to a Attribute (line 455):
        
        # Assigning a Name to a Attribute (line 455):
        # Getting the type of 'picker' (line 455)
        picker_5249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 23), 'picker')
        # Getting the type of 'self' (line 455)
        self_5250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'self')
        # Setting the type of the member '_picker' of a type (line 455)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), self_5250, '_picker', picker_5249)
        
        # ################# End of 'set_picker(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_picker' in the type store
        # Getting the type of 'stypy_return_type' (line 423)
        stypy_return_type_5251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5251)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_picker'
        return stypy_return_type_5251


    @norecursion
    def get_picker(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_picker'
        module_type_store = module_type_store.open_function_context('get_picker', 457, 4, False)
        # Assigning a type to the variable 'self' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_picker.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_picker.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_picker.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_picker.__dict__.__setitem__('stypy_function_name', 'Artist.get_picker')
        Artist.get_picker.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_picker.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_picker.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_picker.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_picker.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_picker.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_picker.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_picker', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_picker', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_picker(...)' code ##################

        unicode_5252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 8), 'unicode', u'Return the picker object used by this artist')
        # Getting the type of 'self' (line 459)
        self_5253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'self')
        # Obtaining the member '_picker' of a type (line 459)
        _picker_5254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 15), self_5253, '_picker')
        # Assigning a type to the variable 'stypy_return_type' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'stypy_return_type', _picker_5254)
        
        # ################# End of 'get_picker(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_picker' in the type store
        # Getting the type of 'stypy_return_type' (line 457)
        stypy_return_type_5255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5255)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_picker'
        return stypy_return_type_5255


    @norecursion
    def is_figure_set(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_figure_set'
        module_type_store = module_type_store.open_function_context('is_figure_set', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.is_figure_set.__dict__.__setitem__('stypy_localization', localization)
        Artist.is_figure_set.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.is_figure_set.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.is_figure_set.__dict__.__setitem__('stypy_function_name', 'Artist.is_figure_set')
        Artist.is_figure_set.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.is_figure_set.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.is_figure_set.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.is_figure_set.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.is_figure_set.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.is_figure_set.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.is_figure_set.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.is_figure_set', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_figure_set', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_figure_set(...)' code ##################

        unicode_5256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, (-1)), 'unicode', u'\n        Returns True if the artist is assigned to a\n        :class:`~matplotlib.figure.Figure`.\n        ')
        
        # Getting the type of 'self' (line 466)
        self_5257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 15), 'self')
        # Obtaining the member 'figure' of a type (line 466)
        figure_5258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 15), self_5257, 'figure')
        # Getting the type of 'None' (line 466)
        None_5259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 34), 'None')
        # Applying the binary operator 'isnot' (line 466)
        result_is_not_5260 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 15), 'isnot', figure_5258, None_5259)
        
        # Assigning a type to the variable 'stypy_return_type' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'stypy_return_type', result_is_not_5260)
        
        # ################# End of 'is_figure_set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_figure_set' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_5261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_figure_set'
        return stypy_return_type_5261


    @norecursion
    def get_url(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_url'
        module_type_store = module_type_store.open_function_context('get_url', 468, 4, False)
        # Assigning a type to the variable 'self' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_url.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_url.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_url.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_url.__dict__.__setitem__('stypy_function_name', 'Artist.get_url')
        Artist.get_url.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_url.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_url.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_url.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_url.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_url.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_url.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_url', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_url', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_url(...)' code ##################

        unicode_5262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, (-1)), 'unicode', u'\n        Returns the url\n        ')
        # Getting the type of 'self' (line 472)
        self_5263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'self')
        # Obtaining the member '_url' of a type (line 472)
        _url_5264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 15), self_5263, '_url')
        # Assigning a type to the variable 'stypy_return_type' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'stypy_return_type', _url_5264)
        
        # ################# End of 'get_url(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_url' in the type store
        # Getting the type of 'stypy_return_type' (line 468)
        stypy_return_type_5265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5265)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_url'
        return stypy_return_type_5265


    @norecursion
    def set_url(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_url'
        module_type_store = module_type_store.open_function_context('set_url', 474, 4, False)
        # Assigning a type to the variable 'self' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_url.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_url.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_url.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_url.__dict__.__setitem__('stypy_function_name', 'Artist.set_url')
        Artist.set_url.__dict__.__setitem__('stypy_param_names_list', ['url'])
        Artist.set_url.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_url.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_url.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_url.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_url.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_url.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_url', ['url'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_url', localization, ['url'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_url(...)' code ##################

        unicode_5266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, (-1)), 'unicode', u'\n        Sets the url for the artist\n\n        ACCEPTS: a url string\n        ')
        
        # Assigning a Name to a Attribute (line 480):
        
        # Assigning a Name to a Attribute (line 480):
        # Getting the type of 'url' (line 480)
        url_5267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 20), 'url')
        # Getting the type of 'self' (line 480)
        self_5268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'self')
        # Setting the type of the member '_url' of a type (line 480)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), self_5268, '_url', url_5267)
        
        # ################# End of 'set_url(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_url' in the type store
        # Getting the type of 'stypy_return_type' (line 474)
        stypy_return_type_5269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5269)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_url'
        return stypy_return_type_5269


    @norecursion
    def get_gid(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_gid'
        module_type_store = module_type_store.open_function_context('get_gid', 482, 4, False)
        # Assigning a type to the variable 'self' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_gid.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_gid.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_gid.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_gid.__dict__.__setitem__('stypy_function_name', 'Artist.get_gid')
        Artist.get_gid.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_gid.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_gid.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_gid.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_gid.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_gid.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_gid.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_gid', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_gid', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_gid(...)' code ##################

        unicode_5270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, (-1)), 'unicode', u'\n        Returns the group id\n        ')
        # Getting the type of 'self' (line 486)
        self_5271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 15), 'self')
        # Obtaining the member '_gid' of a type (line 486)
        _gid_5272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 15), self_5271, '_gid')
        # Assigning a type to the variable 'stypy_return_type' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'stypy_return_type', _gid_5272)
        
        # ################# End of 'get_gid(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_gid' in the type store
        # Getting the type of 'stypy_return_type' (line 482)
        stypy_return_type_5273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5273)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_gid'
        return stypy_return_type_5273


    @norecursion
    def set_gid(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_gid'
        module_type_store = module_type_store.open_function_context('set_gid', 488, 4, False)
        # Assigning a type to the variable 'self' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_gid.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_gid.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_gid.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_gid.__dict__.__setitem__('stypy_function_name', 'Artist.set_gid')
        Artist.set_gid.__dict__.__setitem__('stypy_param_names_list', ['gid'])
        Artist.set_gid.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_gid.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_gid.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_gid.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_gid.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_gid.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_gid', ['gid'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_gid', localization, ['gid'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_gid(...)' code ##################

        unicode_5274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, (-1)), 'unicode', u'\n        Sets the (group) id for the artist\n\n        ACCEPTS: an id string\n        ')
        
        # Assigning a Name to a Attribute (line 494):
        
        # Assigning a Name to a Attribute (line 494):
        # Getting the type of 'gid' (line 494)
        gid_5275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 20), 'gid')
        # Getting the type of 'self' (line 494)
        self_5276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'self')
        # Setting the type of the member '_gid' of a type (line 494)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 8), self_5276, '_gid', gid_5275)
        
        # ################# End of 'set_gid(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_gid' in the type store
        # Getting the type of 'stypy_return_type' (line 488)
        stypy_return_type_5277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5277)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_gid'
        return stypy_return_type_5277


    @norecursion
    def get_snap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_snap'
        module_type_store = module_type_store.open_function_context('get_snap', 496, 4, False)
        # Assigning a type to the variable 'self' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_snap.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_snap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_snap.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_snap.__dict__.__setitem__('stypy_function_name', 'Artist.get_snap')
        Artist.get_snap.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_snap.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_snap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_snap.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_snap.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_snap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_snap.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_snap', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_snap', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_snap(...)' code ##################

        unicode_5278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, (-1)), 'unicode', u'\n        Returns the snap setting which may be:\n\n          * True: snap vertices to the nearest pixel center\n\n          * False: leave vertices as-is\n\n          * None: (auto) If the path contains only rectilinear line\n            segments, round to the nearest pixel center\n\n        Only supported by the Agg and MacOSX backends.\n        ')
        
        
        # Obtaining the type of the subscript
        unicode_5279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 20), 'unicode', u'path.snap')
        # Getting the type of 'rcParams' (line 509)
        rcParams_5280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 509)
        getitem___5281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 11), rcParams_5280, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 509)
        subscript_call_result_5282 = invoke(stypy.reporting.localization.Localization(__file__, 509, 11), getitem___5281, unicode_5279)
        
        # Testing the type of an if condition (line 509)
        if_condition_5283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 509, 8), subscript_call_result_5282)
        # Assigning a type to the variable 'if_condition_5283' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'if_condition_5283', if_condition_5283)
        # SSA begins for if statement (line 509)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 510)
        self_5284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 19), 'self')
        # Obtaining the member '_snap' of a type (line 510)
        _snap_5285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 19), self_5284, '_snap')
        # Assigning a type to the variable 'stypy_return_type' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'stypy_return_type', _snap_5285)
        # SSA branch for the else part of an if statement (line 509)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'False' (line 512)
        False_5286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'stypy_return_type', False_5286)
        # SSA join for if statement (line 509)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_snap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_snap' in the type store
        # Getting the type of 'stypy_return_type' (line 496)
        stypy_return_type_5287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5287)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_snap'
        return stypy_return_type_5287


    @norecursion
    def set_snap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_snap'
        module_type_store = module_type_store.open_function_context('set_snap', 514, 4, False)
        # Assigning a type to the variable 'self' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_snap.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_snap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_snap.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_snap.__dict__.__setitem__('stypy_function_name', 'Artist.set_snap')
        Artist.set_snap.__dict__.__setitem__('stypy_param_names_list', ['snap'])
        Artist.set_snap.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_snap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_snap.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_snap.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_snap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_snap.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_snap', ['snap'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_snap', localization, ['snap'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_snap(...)' code ##################

        unicode_5288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, (-1)), 'unicode', u'\n        Sets the snap setting which may be:\n\n          * True: snap vertices to the nearest pixel center\n\n          * False: leave vertices as-is\n\n          * None: (auto) If the path contains only rectilinear line\n            segments, round to the nearest pixel center\n\n        Only supported by the Agg and MacOSX backends.\n        ')
        
        # Assigning a Name to a Attribute (line 527):
        
        # Assigning a Name to a Attribute (line 527):
        # Getting the type of 'snap' (line 527)
        snap_5289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 21), 'snap')
        # Getting the type of 'self' (line 527)
        self_5290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'self')
        # Setting the type of the member '_snap' of a type (line 527)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 8), self_5290, '_snap', snap_5289)
        
        # Assigning a Name to a Attribute (line 528):
        
        # Assigning a Name to a Attribute (line 528):
        # Getting the type of 'True' (line 528)
        True_5291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 21), 'True')
        # Getting the type of 'self' (line 528)
        self_5292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 528)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 8), self_5292, 'stale', True_5291)
        
        # ################# End of 'set_snap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_snap' in the type store
        # Getting the type of 'stypy_return_type' (line 514)
        stypy_return_type_5293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5293)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_snap'
        return stypy_return_type_5293


    @norecursion
    def get_sketch_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_sketch_params'
        module_type_store = module_type_store.open_function_context('get_sketch_params', 530, 4, False)
        # Assigning a type to the variable 'self' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_sketch_params.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_sketch_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_sketch_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_sketch_params.__dict__.__setitem__('stypy_function_name', 'Artist.get_sketch_params')
        Artist.get_sketch_params.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_sketch_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_sketch_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_sketch_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_sketch_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_sketch_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_sketch_params.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_sketch_params', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_sketch_params', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_sketch_params(...)' code ##################

        unicode_5294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, (-1)), 'unicode', u'\n        Returns the sketch parameters for the artist.\n\n        Returns\n        -------\n        sketch_params : tuple or `None`\n\n        A 3-tuple with the following elements:\n\n          * `scale`: The amplitude of the wiggle perpendicular to the\n            source line.\n\n          * `length`: The length of the wiggle along the line.\n\n          * `randomness`: The scale factor by which the length is\n            shrunken or expanded.\n\n        May return `None` if no sketch parameters were set.\n        ')
        # Getting the type of 'self' (line 550)
        self_5295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 15), 'self')
        # Obtaining the member '_sketch' of a type (line 550)
        _sketch_5296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 15), self_5295, '_sketch')
        # Assigning a type to the variable 'stypy_return_type' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'stypy_return_type', _sketch_5296)
        
        # ################# End of 'get_sketch_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_sketch_params' in the type store
        # Getting the type of 'stypy_return_type' (line 530)
        stypy_return_type_5297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_sketch_params'
        return stypy_return_type_5297


    @norecursion
    def set_sketch_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 552)
        None_5298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 38), 'None')
        # Getting the type of 'None' (line 552)
        None_5299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 51), 'None')
        # Getting the type of 'None' (line 552)
        None_5300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 68), 'None')
        defaults = [None_5298, None_5299, None_5300]
        # Create a new context for function 'set_sketch_params'
        module_type_store = module_type_store.open_function_context('set_sketch_params', 552, 4, False)
        # Assigning a type to the variable 'self' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_sketch_params.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_sketch_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_sketch_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_sketch_params.__dict__.__setitem__('stypy_function_name', 'Artist.set_sketch_params')
        Artist.set_sketch_params.__dict__.__setitem__('stypy_param_names_list', ['scale', 'length', 'randomness'])
        Artist.set_sketch_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_sketch_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_sketch_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_sketch_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_sketch_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_sketch_params.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_sketch_params', ['scale', 'length', 'randomness'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_sketch_params', localization, ['scale', 'length', 'randomness'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_sketch_params(...)' code ##################

        unicode_5301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, (-1)), 'unicode', u'\n        Sets the sketch parameters.\n\n        Parameters\n        ----------\n\n        scale : float, optional\n            The amplitude of the wiggle perpendicular to the source\n            line, in pixels.  If scale is `None`, or not provided, no\n            sketch filter will be provided.\n\n        length : float, optional\n             The length of the wiggle along the line, in pixels\n             (default 128.0)\n\n        randomness : float, optional\n            The scale factor by which the length is shrunken or\n            expanded (default 16.0)\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 572)
        # Getting the type of 'scale' (line 572)
        scale_5302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 11), 'scale')
        # Getting the type of 'None' (line 572)
        None_5303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 20), 'None')
        
        (may_be_5304, more_types_in_union_5305) = may_be_none(scale_5302, None_5303)

        if may_be_5304:

            if more_types_in_union_5305:
                # Runtime conditional SSA (line 572)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 573):
            
            # Assigning a Name to a Attribute (line 573):
            # Getting the type of 'None' (line 573)
            None_5306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 27), 'None')
            # Getting the type of 'self' (line 573)
            self_5307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'self')
            # Setting the type of the member '_sketch' of a type (line 573)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 12), self_5307, '_sketch', None_5306)

            if more_types_in_union_5305:
                # Runtime conditional SSA for else branch (line 572)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_5304) or more_types_in_union_5305):
            
            # Assigning a Tuple to a Attribute (line 575):
            
            # Assigning a Tuple to a Attribute (line 575):
            
            # Obtaining an instance of the builtin type 'tuple' (line 575)
            tuple_5308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 28), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 575)
            # Adding element type (line 575)
            # Getting the type of 'scale' (line 575)
            scale_5309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 28), 'scale')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 28), tuple_5308, scale_5309)
            # Adding element type (line 575)
            
            # Evaluating a boolean operation
            # Getting the type of 'length' (line 575)
            length_5310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 35), 'length')
            float_5311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 45), 'float')
            # Applying the binary operator 'or' (line 575)
            result_or_keyword_5312 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 35), 'or', length_5310, float_5311)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 28), tuple_5308, result_or_keyword_5312)
            # Adding element type (line 575)
            
            # Evaluating a boolean operation
            # Getting the type of 'randomness' (line 575)
            randomness_5313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 52), 'randomness')
            float_5314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 66), 'float')
            # Applying the binary operator 'or' (line 575)
            result_or_keyword_5315 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 52), 'or', randomness_5313, float_5314)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 28), tuple_5308, result_or_keyword_5315)
            
            # Getting the type of 'self' (line 575)
            self_5316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'self')
            # Setting the type of the member '_sketch' of a type (line 575)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), self_5316, '_sketch', tuple_5308)

            if (may_be_5304 and more_types_in_union_5305):
                # SSA join for if statement (line 572)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 576):
        
        # Assigning a Name to a Attribute (line 576):
        # Getting the type of 'True' (line 576)
        True_5317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 21), 'True')
        # Getting the type of 'self' (line 576)
        self_5318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 576)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 8), self_5318, 'stale', True_5317)
        
        # ################# End of 'set_sketch_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_sketch_params' in the type store
        # Getting the type of 'stypy_return_type' (line 552)
        stypy_return_type_5319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5319)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_sketch_params'
        return stypy_return_type_5319


    @norecursion
    def set_path_effects(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_path_effects'
        module_type_store = module_type_store.open_function_context('set_path_effects', 578, 4, False)
        # Assigning a type to the variable 'self' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_path_effects.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_path_effects.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_path_effects.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_path_effects.__dict__.__setitem__('stypy_function_name', 'Artist.set_path_effects')
        Artist.set_path_effects.__dict__.__setitem__('stypy_param_names_list', ['path_effects'])
        Artist.set_path_effects.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_path_effects.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_path_effects.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_path_effects.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_path_effects.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_path_effects.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_path_effects', ['path_effects'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_path_effects', localization, ['path_effects'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_path_effects(...)' code ##################

        unicode_5320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, (-1)), 'unicode', u'\n        set path_effects, which should be a list of instances of\n        matplotlib.patheffect._Base class or its derivatives.\n        ')
        
        # Assigning a Name to a Attribute (line 583):
        
        # Assigning a Name to a Attribute (line 583):
        # Getting the type of 'path_effects' (line 583)
        path_effects_5321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 29), 'path_effects')
        # Getting the type of 'self' (line 583)
        self_5322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'self')
        # Setting the type of the member '_path_effects' of a type (line 583)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 8), self_5322, '_path_effects', path_effects_5321)
        
        # Assigning a Name to a Attribute (line 584):
        
        # Assigning a Name to a Attribute (line 584):
        # Getting the type of 'True' (line 584)
        True_5323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 21), 'True')
        # Getting the type of 'self' (line 584)
        self_5324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 584)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 8), self_5324, 'stale', True_5323)
        
        # ################# End of 'set_path_effects(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_path_effects' in the type store
        # Getting the type of 'stypy_return_type' (line 578)
        stypy_return_type_5325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5325)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_path_effects'
        return stypy_return_type_5325


    @norecursion
    def get_path_effects(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_path_effects'
        module_type_store = module_type_store.open_function_context('get_path_effects', 586, 4, False)
        # Assigning a type to the variable 'self' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_path_effects.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_path_effects.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_path_effects.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_path_effects.__dict__.__setitem__('stypy_function_name', 'Artist.get_path_effects')
        Artist.get_path_effects.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_path_effects.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_path_effects.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_path_effects.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_path_effects.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_path_effects.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_path_effects.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_path_effects', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_path_effects', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_path_effects(...)' code ##################

        # Getting the type of 'self' (line 587)
        self_5326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 15), 'self')
        # Obtaining the member '_path_effects' of a type (line 587)
        _path_effects_5327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 15), self_5326, '_path_effects')
        # Assigning a type to the variable 'stypy_return_type' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'stypy_return_type', _path_effects_5327)
        
        # ################# End of 'get_path_effects(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_path_effects' in the type store
        # Getting the type of 'stypy_return_type' (line 586)
        stypy_return_type_5328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5328)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_path_effects'
        return stypy_return_type_5328


    @norecursion
    def get_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_figure'
        module_type_store = module_type_store.open_function_context('get_figure', 589, 4, False)
        # Assigning a type to the variable 'self' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_figure.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_figure.__dict__.__setitem__('stypy_function_name', 'Artist.get_figure')
        Artist.get_figure.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_figure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_figure', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_figure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_figure(...)' code ##################

        unicode_5329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, (-1)), 'unicode', u'\n        Return the :class:`~matplotlib.figure.Figure` instance the\n        artist belongs to.\n        ')
        # Getting the type of 'self' (line 594)
        self_5330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 15), 'self')
        # Obtaining the member 'figure' of a type (line 594)
        figure_5331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 15), self_5330, 'figure')
        # Assigning a type to the variable 'stypy_return_type' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'stypy_return_type', figure_5331)
        
        # ################# End of 'get_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 589)
        stypy_return_type_5332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5332)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_figure'
        return stypy_return_type_5332


    @norecursion
    def set_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_figure'
        module_type_store = module_type_store.open_function_context('set_figure', 596, 4, False)
        # Assigning a type to the variable 'self' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_figure.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_figure.__dict__.__setitem__('stypy_function_name', 'Artist.set_figure')
        Artist.set_figure.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        Artist.set_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_figure', ['fig'], None, None, defaults, varargs, kwargs)

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

        unicode_5333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, (-1)), 'unicode', u'\n        Set the :class:`~matplotlib.figure.Figure` instance the artist\n        belongs to.\n\n        ACCEPTS: a :class:`matplotlib.figure.Figure` instance\n        ')
        
        
        # Getting the type of 'self' (line 604)
        self_5334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 11), 'self')
        # Obtaining the member 'figure' of a type (line 604)
        figure_5335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 11), self_5334, 'figure')
        # Getting the type of 'fig' (line 604)
        fig_5336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 26), 'fig')
        # Applying the binary operator 'is' (line 604)
        result_is__5337 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 11), 'is', figure_5335, fig_5336)
        
        # Testing the type of an if condition (line 604)
        if_condition_5338 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 8), result_is__5337)
        # Assigning a type to the variable 'if_condition_5338' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'if_condition_5338', if_condition_5338)
        # SSA begins for if statement (line 604)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 604)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 611)
        self_5339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 11), 'self')
        # Obtaining the member 'figure' of a type (line 611)
        figure_5340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 11), self_5339, 'figure')
        # Getting the type of 'None' (line 611)
        None_5341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 30), 'None')
        # Applying the binary operator 'isnot' (line 611)
        result_is_not_5342 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 11), 'isnot', figure_5340, None_5341)
        
        # Testing the type of an if condition (line 611)
        if_condition_5343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 8), result_is_not_5342)
        # Assigning a type to the variable 'if_condition_5343' (line 611)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 8), 'if_condition_5343', if_condition_5343)
        # SSA begins for if statement (line 611)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 612)
        # Processing the call arguments (line 612)
        unicode_5345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 31), 'unicode', u'Can not put single artist in more than one figure')
        # Processing the call keyword arguments (line 612)
        kwargs_5346 = {}
        # Getting the type of 'RuntimeError' (line 612)
        RuntimeError_5344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 612)
        RuntimeError_call_result_5347 = invoke(stypy.reporting.localization.Localization(__file__, 612, 18), RuntimeError_5344, *[unicode_5345], **kwargs_5346)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 612, 12), RuntimeError_call_result_5347, 'raise parameter', BaseException)
        # SSA join for if statement (line 611)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 614):
        
        # Assigning a Name to a Attribute (line 614):
        # Getting the type of 'fig' (line 614)
        fig_5348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 22), 'fig')
        # Getting the type of 'self' (line 614)
        self_5349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'self')
        # Setting the type of the member 'figure' of a type (line 614)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 8), self_5349, 'figure', fig_5348)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 615)
        self_5350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 11), 'self')
        # Obtaining the member 'figure' of a type (line 615)
        figure_5351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 11), self_5350, 'figure')
        
        # Getting the type of 'self' (line 615)
        self_5352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 27), 'self')
        # Obtaining the member 'figure' of a type (line 615)
        figure_5353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 27), self_5352, 'figure')
        # Getting the type of 'self' (line 615)
        self_5354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 46), 'self')
        # Applying the binary operator 'isnot' (line 615)
        result_is_not_5355 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 27), 'isnot', figure_5353, self_5354)
        
        # Applying the binary operator 'and' (line 615)
        result_and_keyword_5356 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 11), 'and', figure_5351, result_is_not_5355)
        
        # Testing the type of an if condition (line 615)
        if_condition_5357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 615, 8), result_and_keyword_5356)
        # Assigning a type to the variable 'if_condition_5357' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'if_condition_5357', if_condition_5357)
        # SSA begins for if statement (line 615)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to pchanged(...): (line 616)
        # Processing the call keyword arguments (line 616)
        kwargs_5360 = {}
        # Getting the type of 'self' (line 616)
        self_5358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 616)
        pchanged_5359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 12), self_5358, 'pchanged')
        # Calling pchanged(args, kwargs) (line 616)
        pchanged_call_result_5361 = invoke(stypy.reporting.localization.Localization(__file__, 616, 12), pchanged_5359, *[], **kwargs_5360)
        
        # SSA join for if statement (line 615)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 617):
        
        # Assigning a Name to a Attribute (line 617):
        # Getting the type of 'True' (line 617)
        True_5362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 21), 'True')
        # Getting the type of 'self' (line 617)
        self_5363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 617)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 8), self_5363, 'stale', True_5362)
        
        # ################# End of 'set_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 596)
        stypy_return_type_5364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5364)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_figure'
        return stypy_return_type_5364


    @norecursion
    def set_clip_box(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_clip_box'
        module_type_store = module_type_store.open_function_context('set_clip_box', 619, 4, False)
        # Assigning a type to the variable 'self' (line 620)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_clip_box.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_clip_box.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_clip_box.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_clip_box.__dict__.__setitem__('stypy_function_name', 'Artist.set_clip_box')
        Artist.set_clip_box.__dict__.__setitem__('stypy_param_names_list', ['clipbox'])
        Artist.set_clip_box.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_clip_box.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_clip_box.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_clip_box.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_clip_box.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_clip_box.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_clip_box', ['clipbox'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_clip_box', localization, ['clipbox'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_clip_box(...)' code ##################

        unicode_5365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, (-1)), 'unicode', u"\n        Set the artist's clip :class:`~matplotlib.transforms.Bbox`.\n\n        ACCEPTS: a :class:`matplotlib.transforms.Bbox` instance\n        ")
        
        # Assigning a Name to a Attribute (line 625):
        
        # Assigning a Name to a Attribute (line 625):
        # Getting the type of 'clipbox' (line 625)
        clipbox_5366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 23), 'clipbox')
        # Getting the type of 'self' (line 625)
        self_5367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'self')
        # Setting the type of the member 'clipbox' of a type (line 625)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 8), self_5367, 'clipbox', clipbox_5366)
        
        # Call to pchanged(...): (line 626)
        # Processing the call keyword arguments (line 626)
        kwargs_5370 = {}
        # Getting the type of 'self' (line 626)
        self_5368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 626)
        pchanged_5369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 8), self_5368, 'pchanged')
        # Calling pchanged(args, kwargs) (line 626)
        pchanged_call_result_5371 = invoke(stypy.reporting.localization.Localization(__file__, 626, 8), pchanged_5369, *[], **kwargs_5370)
        
        
        # Assigning a Name to a Attribute (line 627):
        
        # Assigning a Name to a Attribute (line 627):
        # Getting the type of 'True' (line 627)
        True_5372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 21), 'True')
        # Getting the type of 'self' (line 627)
        self_5373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 627)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 8), self_5373, 'stale', True_5372)
        
        # ################# End of 'set_clip_box(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_clip_box' in the type store
        # Getting the type of 'stypy_return_type' (line 619)
        stypy_return_type_5374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5374)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_clip_box'
        return stypy_return_type_5374


    @norecursion
    def set_clip_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 629)
        None_5375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 44), 'None')
        defaults = [None_5375]
        # Create a new context for function 'set_clip_path'
        module_type_store = module_type_store.open_function_context('set_clip_path', 629, 4, False)
        # Assigning a type to the variable 'self' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_clip_path.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_clip_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_clip_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_clip_path.__dict__.__setitem__('stypy_function_name', 'Artist.set_clip_path')
        Artist.set_clip_path.__dict__.__setitem__('stypy_param_names_list', ['path', 'transform'])
        Artist.set_clip_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_clip_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_clip_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_clip_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_clip_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_clip_path.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_clip_path', ['path', 'transform'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_clip_path', localization, ['path', 'transform'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_clip_path(...)' code ##################

        unicode_5376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, (-1)), 'unicode', u"\n        Set the artist's clip path, which may be:\n\n        - a :class:`~matplotlib.patches.Patch` (or subclass) instance; or\n        - a :class:`~matplotlib.path.Path` instance, in which case a\n          :class:`~matplotlib.transforms.Transform` instance, which will be\n          applied to the path before using it for clipping, must be provided;\n          or\n        - ``None``, to remove a previously set clipping path.\n\n        For efficiency, if the path happens to be an axis-aligned rectangle,\n        this method will set the clipping box to the corresponding rectangle\n        and set the clipping path to ``None``.\n\n        ACCEPTS: [ (:class:`~matplotlib.path.Path`,\n        :class:`~matplotlib.transforms.Transform`) |\n        :class:`~matplotlib.patches.Patch` | None ]\n        ")
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 648, 8))
        
        # 'from matplotlib.patches import Patch, Rectangle' statement (line 648)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
        import_5377 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 648, 8), 'matplotlib.patches')

        if (type(import_5377) is not StypyTypeError):

            if (import_5377 != 'pyd_module'):
                __import__(import_5377)
                sys_modules_5378 = sys.modules[import_5377]
                import_from_module(stypy.reporting.localization.Localization(__file__, 648, 8), 'matplotlib.patches', sys_modules_5378.module_type_store, module_type_store, ['Patch', 'Rectangle'])
                nest_module(stypy.reporting.localization.Localization(__file__, 648, 8), __file__, sys_modules_5378, sys_modules_5378.module_type_store, module_type_store)
            else:
                from matplotlib.patches import Patch, Rectangle

                import_from_module(stypy.reporting.localization.Localization(__file__, 648, 8), 'matplotlib.patches', None, module_type_store, ['Patch', 'Rectangle'], [Patch, Rectangle])

        else:
            # Assigning a type to the variable 'matplotlib.patches' (line 648)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'matplotlib.patches', import_5377)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
        
        
        # Assigning a Name to a Name (line 650):
        
        # Assigning a Name to a Name (line 650):
        # Getting the type of 'False' (line 650)
        False_5379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 18), 'False')
        # Assigning a type to the variable 'success' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'success', False_5379)
        
        # Type idiom detected: calculating its left and rigth part (line 651)
        # Getting the type of 'transform' (line 651)
        transform_5380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 11), 'transform')
        # Getting the type of 'None' (line 651)
        None_5381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 24), 'None')
        
        (may_be_5382, more_types_in_union_5383) = may_be_none(transform_5380, None_5381)

        if may_be_5382:

            if more_types_in_union_5383:
                # Runtime conditional SSA (line 651)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Call to isinstance(...): (line 652)
            # Processing the call arguments (line 652)
            # Getting the type of 'path' (line 652)
            path_5385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 26), 'path', False)
            # Getting the type of 'Rectangle' (line 652)
            Rectangle_5386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 32), 'Rectangle', False)
            # Processing the call keyword arguments (line 652)
            kwargs_5387 = {}
            # Getting the type of 'isinstance' (line 652)
            isinstance_5384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 652)
            isinstance_call_result_5388 = invoke(stypy.reporting.localization.Localization(__file__, 652, 15), isinstance_5384, *[path_5385, Rectangle_5386], **kwargs_5387)
            
            # Testing the type of an if condition (line 652)
            if_condition_5389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 652, 12), isinstance_call_result_5388)
            # Assigning a type to the variable 'if_condition_5389' (line 652)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'if_condition_5389', if_condition_5389)
            # SSA begins for if statement (line 652)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 653):
            
            # Assigning a Call to a Attribute (line 653):
            
            # Call to TransformedBbox(...): (line 653)
            # Processing the call arguments (line 653)
            
            # Call to unit(...): (line 653)
            # Processing the call keyword arguments (line 653)
            kwargs_5393 = {}
            # Getting the type of 'Bbox' (line 653)
            Bbox_5391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 47), 'Bbox', False)
            # Obtaining the member 'unit' of a type (line 653)
            unit_5392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 47), Bbox_5391, 'unit')
            # Calling unit(args, kwargs) (line 653)
            unit_call_result_5394 = invoke(stypy.reporting.localization.Localization(__file__, 653, 47), unit_5392, *[], **kwargs_5393)
            
            
            # Call to get_transform(...): (line 654)
            # Processing the call keyword arguments (line 654)
            kwargs_5397 = {}
            # Getting the type of 'path' (line 654)
            path_5395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 47), 'path', False)
            # Obtaining the member 'get_transform' of a type (line 654)
            get_transform_5396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 47), path_5395, 'get_transform')
            # Calling get_transform(args, kwargs) (line 654)
            get_transform_call_result_5398 = invoke(stypy.reporting.localization.Localization(__file__, 654, 47), get_transform_5396, *[], **kwargs_5397)
            
            # Processing the call keyword arguments (line 653)
            kwargs_5399 = {}
            # Getting the type of 'TransformedBbox' (line 653)
            TransformedBbox_5390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 31), 'TransformedBbox', False)
            # Calling TransformedBbox(args, kwargs) (line 653)
            TransformedBbox_call_result_5400 = invoke(stypy.reporting.localization.Localization(__file__, 653, 31), TransformedBbox_5390, *[unit_call_result_5394, get_transform_call_result_5398], **kwargs_5399)
            
            # Getting the type of 'self' (line 653)
            self_5401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 16), 'self')
            # Setting the type of the member 'clipbox' of a type (line 653)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 16), self_5401, 'clipbox', TransformedBbox_call_result_5400)
            
            # Assigning a Name to a Attribute (line 655):
            
            # Assigning a Name to a Attribute (line 655):
            # Getting the type of 'None' (line 655)
            None_5402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 33), 'None')
            # Getting the type of 'self' (line 655)
            self_5403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 16), 'self')
            # Setting the type of the member '_clippath' of a type (line 655)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 16), self_5403, '_clippath', None_5402)
            
            # Assigning a Name to a Name (line 656):
            
            # Assigning a Name to a Name (line 656):
            # Getting the type of 'True' (line 656)
            True_5404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 26), 'True')
            # Assigning a type to the variable 'success' (line 656)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 16), 'success', True_5404)
            # SSA branch for the else part of an if statement (line 652)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 657)
            # Processing the call arguments (line 657)
            # Getting the type of 'path' (line 657)
            path_5406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 28), 'path', False)
            # Getting the type of 'Patch' (line 657)
            Patch_5407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 34), 'Patch', False)
            # Processing the call keyword arguments (line 657)
            kwargs_5408 = {}
            # Getting the type of 'isinstance' (line 657)
            isinstance_5405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 657)
            isinstance_call_result_5409 = invoke(stypy.reporting.localization.Localization(__file__, 657, 17), isinstance_5405, *[path_5406, Patch_5407], **kwargs_5408)
            
            # Testing the type of an if condition (line 657)
            if_condition_5410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 657, 17), isinstance_call_result_5409)
            # Assigning a type to the variable 'if_condition_5410' (line 657)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 17), 'if_condition_5410', if_condition_5410)
            # SSA begins for if statement (line 657)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 658):
            
            # Assigning a Call to a Attribute (line 658):
            
            # Call to TransformedPatchPath(...): (line 658)
            # Processing the call arguments (line 658)
            # Getting the type of 'path' (line 658)
            path_5412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 54), 'path', False)
            # Processing the call keyword arguments (line 658)
            kwargs_5413 = {}
            # Getting the type of 'TransformedPatchPath' (line 658)
            TransformedPatchPath_5411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 33), 'TransformedPatchPath', False)
            # Calling TransformedPatchPath(args, kwargs) (line 658)
            TransformedPatchPath_call_result_5414 = invoke(stypy.reporting.localization.Localization(__file__, 658, 33), TransformedPatchPath_5411, *[path_5412], **kwargs_5413)
            
            # Getting the type of 'self' (line 658)
            self_5415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 16), 'self')
            # Setting the type of the member '_clippath' of a type (line 658)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 16), self_5415, '_clippath', TransformedPatchPath_call_result_5414)
            
            # Assigning a Name to a Name (line 659):
            
            # Assigning a Name to a Name (line 659):
            # Getting the type of 'True' (line 659)
            True_5416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 26), 'True')
            # Assigning a type to the variable 'success' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'success', True_5416)
            # SSA branch for the else part of an if statement (line 657)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 660)
            # Getting the type of 'tuple' (line 660)
            tuple_5417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 34), 'tuple')
            # Getting the type of 'path' (line 660)
            path_5418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 28), 'path')
            
            (may_be_5419, more_types_in_union_5420) = may_be_subtype(tuple_5417, path_5418)

            if may_be_5419:

                if more_types_in_union_5420:
                    # Runtime conditional SSA (line 660)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'path' (line 660)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 17), 'path', remove_not_subtype_from_union(path_5418, tuple))
                
                # Assigning a Name to a Tuple (line 661):
                
                # Assigning a Subscript to a Name (line 661):
                
                # Obtaining the type of the subscript
                int_5421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 16), 'int')
                # Getting the type of 'path' (line 661)
                path_5422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 34), 'path')
                # Obtaining the member '__getitem__' of a type (line 661)
                getitem___5423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 16), path_5422, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 661)
                subscript_call_result_5424 = invoke(stypy.reporting.localization.Localization(__file__, 661, 16), getitem___5423, int_5421)
                
                # Assigning a type to the variable 'tuple_var_assignment_4601' (line 661)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'tuple_var_assignment_4601', subscript_call_result_5424)
                
                # Assigning a Subscript to a Name (line 661):
                
                # Obtaining the type of the subscript
                int_5425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 16), 'int')
                # Getting the type of 'path' (line 661)
                path_5426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 34), 'path')
                # Obtaining the member '__getitem__' of a type (line 661)
                getitem___5427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 16), path_5426, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 661)
                subscript_call_result_5428 = invoke(stypy.reporting.localization.Localization(__file__, 661, 16), getitem___5427, int_5425)
                
                # Assigning a type to the variable 'tuple_var_assignment_4602' (line 661)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'tuple_var_assignment_4602', subscript_call_result_5428)
                
                # Assigning a Name to a Name (line 661):
                # Getting the type of 'tuple_var_assignment_4601' (line 661)
                tuple_var_assignment_4601_5429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'tuple_var_assignment_4601')
                # Assigning a type to the variable 'path' (line 661)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'path', tuple_var_assignment_4601_5429)
                
                # Assigning a Name to a Name (line 661):
                # Getting the type of 'tuple_var_assignment_4602' (line 661)
                tuple_var_assignment_4602_5430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'tuple_var_assignment_4602')
                # Assigning a type to the variable 'transform' (line 661)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 22), 'transform', tuple_var_assignment_4602_5430)

                if more_types_in_union_5420:
                    # SSA join for if statement (line 660)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 657)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 652)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_5383:
                # SSA join for if statement (line 651)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 663)
        # Getting the type of 'path' (line 663)
        path_5431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 11), 'path')
        # Getting the type of 'None' (line 663)
        None_5432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 19), 'None')
        
        (may_be_5433, more_types_in_union_5434) = may_be_none(path_5431, None_5432)

        if may_be_5433:

            if more_types_in_union_5434:
                # Runtime conditional SSA (line 663)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 664):
            
            # Assigning a Name to a Attribute (line 664):
            # Getting the type of 'None' (line 664)
            None_5435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 29), 'None')
            # Getting the type of 'self' (line 664)
            self_5436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'self')
            # Setting the type of the member '_clippath' of a type (line 664)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 12), self_5436, '_clippath', None_5435)
            
            # Assigning a Name to a Name (line 665):
            
            # Assigning a Name to a Name (line 665):
            # Getting the type of 'True' (line 665)
            True_5437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 22), 'True')
            # Assigning a type to the variable 'success' (line 665)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'success', True_5437)

            if more_types_in_union_5434:
                # Runtime conditional SSA for else branch (line 663)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_5433) or more_types_in_union_5434):
            
            
            # Call to isinstance(...): (line 666)
            # Processing the call arguments (line 666)
            # Getting the type of 'path' (line 666)
            path_5439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 24), 'path', False)
            # Getting the type of 'Path' (line 666)
            Path_5440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 30), 'Path', False)
            # Processing the call keyword arguments (line 666)
            kwargs_5441 = {}
            # Getting the type of 'isinstance' (line 666)
            isinstance_5438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 666)
            isinstance_call_result_5442 = invoke(stypy.reporting.localization.Localization(__file__, 666, 13), isinstance_5438, *[path_5439, Path_5440], **kwargs_5441)
            
            # Testing the type of an if condition (line 666)
            if_condition_5443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 13), isinstance_call_result_5442)
            # Assigning a type to the variable 'if_condition_5443' (line 666)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 13), 'if_condition_5443', if_condition_5443)
            # SSA begins for if statement (line 666)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 667):
            
            # Assigning a Call to a Attribute (line 667):
            
            # Call to TransformedPath(...): (line 667)
            # Processing the call arguments (line 667)
            # Getting the type of 'path' (line 667)
            path_5445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 45), 'path', False)
            # Getting the type of 'transform' (line 667)
            transform_5446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 51), 'transform', False)
            # Processing the call keyword arguments (line 667)
            kwargs_5447 = {}
            # Getting the type of 'TransformedPath' (line 667)
            TransformedPath_5444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 29), 'TransformedPath', False)
            # Calling TransformedPath(args, kwargs) (line 667)
            TransformedPath_call_result_5448 = invoke(stypy.reporting.localization.Localization(__file__, 667, 29), TransformedPath_5444, *[path_5445, transform_5446], **kwargs_5447)
            
            # Getting the type of 'self' (line 667)
            self_5449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'self')
            # Setting the type of the member '_clippath' of a type (line 667)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 12), self_5449, '_clippath', TransformedPath_call_result_5448)
            
            # Assigning a Name to a Name (line 668):
            
            # Assigning a Name to a Name (line 668):
            # Getting the type of 'True' (line 668)
            True_5450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 22), 'True')
            # Assigning a type to the variable 'success' (line 668)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), 'success', True_5450)
            # SSA branch for the else part of an if statement (line 666)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 669)
            # Processing the call arguments (line 669)
            # Getting the type of 'path' (line 669)
            path_5452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 24), 'path', False)
            # Getting the type of 'TransformedPatchPath' (line 669)
            TransformedPatchPath_5453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 30), 'TransformedPatchPath', False)
            # Processing the call keyword arguments (line 669)
            kwargs_5454 = {}
            # Getting the type of 'isinstance' (line 669)
            isinstance_5451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 669)
            isinstance_call_result_5455 = invoke(stypy.reporting.localization.Localization(__file__, 669, 13), isinstance_5451, *[path_5452, TransformedPatchPath_5453], **kwargs_5454)
            
            # Testing the type of an if condition (line 669)
            if_condition_5456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 669, 13), isinstance_call_result_5455)
            # Assigning a type to the variable 'if_condition_5456' (line 669)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 13), 'if_condition_5456', if_condition_5456)
            # SSA begins for if statement (line 669)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 670):
            
            # Assigning a Name to a Attribute (line 670):
            # Getting the type of 'path' (line 670)
            path_5457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 29), 'path')
            # Getting the type of 'self' (line 670)
            self_5458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), 'self')
            # Setting the type of the member '_clippath' of a type (line 670)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 12), self_5458, '_clippath', path_5457)
            
            # Assigning a Name to a Name (line 671):
            
            # Assigning a Name to a Name (line 671):
            # Getting the type of 'True' (line 671)
            True_5459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 22), 'True')
            # Assigning a type to the variable 'success' (line 671)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 12), 'success', True_5459)
            # SSA branch for the else part of an if statement (line 669)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 672)
            # Processing the call arguments (line 672)
            # Getting the type of 'path' (line 672)
            path_5461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 24), 'path', False)
            # Getting the type of 'TransformedPath' (line 672)
            TransformedPath_5462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 30), 'TransformedPath', False)
            # Processing the call keyword arguments (line 672)
            kwargs_5463 = {}
            # Getting the type of 'isinstance' (line 672)
            isinstance_5460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 672)
            isinstance_call_result_5464 = invoke(stypy.reporting.localization.Localization(__file__, 672, 13), isinstance_5460, *[path_5461, TransformedPath_5462], **kwargs_5463)
            
            # Testing the type of an if condition (line 672)
            if_condition_5465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 672, 13), isinstance_call_result_5464)
            # Assigning a type to the variable 'if_condition_5465' (line 672)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 13), 'if_condition_5465', if_condition_5465)
            # SSA begins for if statement (line 672)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 673):
            
            # Assigning a Name to a Attribute (line 673):
            # Getting the type of 'path' (line 673)
            path_5466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 29), 'path')
            # Getting the type of 'self' (line 673)
            self_5467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 12), 'self')
            # Setting the type of the member '_clippath' of a type (line 673)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 12), self_5467, '_clippath', path_5466)
            
            # Assigning a Name to a Name (line 674):
            
            # Assigning a Name to a Name (line 674):
            # Getting the type of 'True' (line 674)
            True_5468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 22), 'True')
            # Assigning a type to the variable 'success' (line 674)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'success', True_5468)
            # SSA join for if statement (line 672)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 669)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 666)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_5433 and more_types_in_union_5434):
                # SSA join for if statement (line 663)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'success' (line 676)
        success_5469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 15), 'success')
        # Applying the 'not' unary operator (line 676)
        result_not__5470 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 11), 'not', success_5469)
        
        # Testing the type of an if condition (line 676)
        if_condition_5471 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 676, 8), result_not__5470)
        # Assigning a type to the variable 'if_condition_5471' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'if_condition_5471', if_condition_5471)
        # SSA begins for if statement (line 676)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 677)
        # Processing the call arguments (line 677)
        
        # Call to format(...): (line 678)
        # Processing the call arguments (line 678)
        
        # Call to type(...): (line 679)
        # Processing the call arguments (line 679)
        # Getting the type of 'path' (line 679)
        path_5476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 29), 'path', False)
        # Processing the call keyword arguments (line 679)
        kwargs_5477 = {}
        # Getting the type of 'type' (line 679)
        type_5475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 24), 'type', False)
        # Calling type(args, kwargs) (line 679)
        type_call_result_5478 = invoke(stypy.reporting.localization.Localization(__file__, 679, 24), type_5475, *[path_5476], **kwargs_5477)
        
        # Obtaining the member '__name__' of a type (line 679)
        name___5479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 24), type_call_result_5478, '__name__')
        
        # Call to type(...): (line 679)
        # Processing the call arguments (line 679)
        # Getting the type of 'transform' (line 679)
        transform_5481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 50), 'transform', False)
        # Processing the call keyword arguments (line 679)
        kwargs_5482 = {}
        # Getting the type of 'type' (line 679)
        type_5480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 45), 'type', False)
        # Calling type(args, kwargs) (line 679)
        type_call_result_5483 = invoke(stypy.reporting.localization.Localization(__file__, 679, 45), type_5480, *[transform_5481], **kwargs_5482)
        
        # Obtaining the member '__name__' of a type (line 679)
        name___5484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 45), type_call_result_5483, '__name__')
        # Processing the call keyword arguments (line 678)
        kwargs_5485 = {}
        unicode_5473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 16), 'unicode', u'Invalid arguments to set_clip_path, of type {} and {}')
        # Obtaining the member 'format' of a type (line 678)
        format_5474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 16), unicode_5473, 'format')
        # Calling format(args, kwargs) (line 678)
        format_call_result_5486 = invoke(stypy.reporting.localization.Localization(__file__, 678, 16), format_5474, *[name___5479, name___5484], **kwargs_5485)
        
        # Processing the call keyword arguments (line 677)
        kwargs_5487 = {}
        # Getting the type of 'TypeError' (line 677)
        TypeError_5472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 677)
        TypeError_call_result_5488 = invoke(stypy.reporting.localization.Localization(__file__, 677, 18), TypeError_5472, *[format_call_result_5486], **kwargs_5487)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 677, 12), TypeError_call_result_5488, 'raise parameter', BaseException)
        # SSA join for if statement (line 676)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to pchanged(...): (line 682)
        # Processing the call keyword arguments (line 682)
        kwargs_5491 = {}
        # Getting the type of 'self' (line 682)
        self_5489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 682)
        pchanged_5490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 8), self_5489, 'pchanged')
        # Calling pchanged(args, kwargs) (line 682)
        pchanged_call_result_5492 = invoke(stypy.reporting.localization.Localization(__file__, 682, 8), pchanged_5490, *[], **kwargs_5491)
        
        
        # Assigning a Name to a Attribute (line 683):
        
        # Assigning a Name to a Attribute (line 683):
        # Getting the type of 'True' (line 683)
        True_5493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 21), 'True')
        # Getting the type of 'self' (line 683)
        self_5494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 683)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 8), self_5494, 'stale', True_5493)
        
        # ################# End of 'set_clip_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_clip_path' in the type store
        # Getting the type of 'stypy_return_type' (line 629)
        stypy_return_type_5495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5495)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_clip_path'
        return stypy_return_type_5495


    @norecursion
    def get_alpha(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_alpha'
        module_type_store = module_type_store.open_function_context('get_alpha', 685, 4, False)
        # Assigning a type to the variable 'self' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_alpha.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_alpha.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_alpha.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_alpha.__dict__.__setitem__('stypy_function_name', 'Artist.get_alpha')
        Artist.get_alpha.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_alpha.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_alpha.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_alpha.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_alpha.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_alpha.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_alpha.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_alpha', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_alpha', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_alpha(...)' code ##################

        unicode_5496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, (-1)), 'unicode', u'\n        Return the alpha value used for blending - not supported on all\n        backends\n        ')
        # Getting the type of 'self' (line 690)
        self_5497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 15), 'self')
        # Obtaining the member '_alpha' of a type (line 690)
        _alpha_5498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 15), self_5497, '_alpha')
        # Assigning a type to the variable 'stypy_return_type' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'stypy_return_type', _alpha_5498)
        
        # ################# End of 'get_alpha(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_alpha' in the type store
        # Getting the type of 'stypy_return_type' (line 685)
        stypy_return_type_5499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5499)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_alpha'
        return stypy_return_type_5499


    @norecursion
    def get_visible(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_visible'
        module_type_store = module_type_store.open_function_context('get_visible', 692, 4, False)
        # Assigning a type to the variable 'self' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_visible.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_visible.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_visible.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_visible.__dict__.__setitem__('stypy_function_name', 'Artist.get_visible')
        Artist.get_visible.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_visible.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_visible.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_visible.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_visible.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_visible.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_visible.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_visible', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_visible', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_visible(...)' code ##################

        unicode_5500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 8), 'unicode', u"Return the artist's visiblity")
        # Getting the type of 'self' (line 694)
        self_5501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 15), 'self')
        # Obtaining the member '_visible' of a type (line 694)
        _visible_5502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 15), self_5501, '_visible')
        # Assigning a type to the variable 'stypy_return_type' (line 694)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'stypy_return_type', _visible_5502)
        
        # ################# End of 'get_visible(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_visible' in the type store
        # Getting the type of 'stypy_return_type' (line 692)
        stypy_return_type_5503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5503)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_visible'
        return stypy_return_type_5503


    @norecursion
    def get_animated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_animated'
        module_type_store = module_type_store.open_function_context('get_animated', 696, 4, False)
        # Assigning a type to the variable 'self' (line 697)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_animated.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_animated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_animated.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_animated.__dict__.__setitem__('stypy_function_name', 'Artist.get_animated')
        Artist.get_animated.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_animated.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_animated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_animated.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_animated.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_animated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_animated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_animated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_animated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_animated(...)' code ##################

        unicode_5504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 8), 'unicode', u"Return the artist's animated state")
        # Getting the type of 'self' (line 698)
        self_5505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 15), 'self')
        # Obtaining the member '_animated' of a type (line 698)
        _animated_5506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 15), self_5505, '_animated')
        # Assigning a type to the variable 'stypy_return_type' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'stypy_return_type', _animated_5506)
        
        # ################# End of 'get_animated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_animated' in the type store
        # Getting the type of 'stypy_return_type' (line 696)
        stypy_return_type_5507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5507)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_animated'
        return stypy_return_type_5507


    @norecursion
    def get_clip_on(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_clip_on'
        module_type_store = module_type_store.open_function_context('get_clip_on', 700, 4, False)
        # Assigning a type to the variable 'self' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_clip_on.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_clip_on.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_clip_on.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_clip_on.__dict__.__setitem__('stypy_function_name', 'Artist.get_clip_on')
        Artist.get_clip_on.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_clip_on.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_clip_on.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_clip_on.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_clip_on.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_clip_on.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_clip_on.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_clip_on', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_clip_on', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_clip_on(...)' code ##################

        unicode_5508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 8), 'unicode', u'Return whether artist uses clipping')
        # Getting the type of 'self' (line 702)
        self_5509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 15), 'self')
        # Obtaining the member '_clipon' of a type (line 702)
        _clipon_5510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 15), self_5509, '_clipon')
        # Assigning a type to the variable 'stypy_return_type' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'stypy_return_type', _clipon_5510)
        
        # ################# End of 'get_clip_on(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_clip_on' in the type store
        # Getting the type of 'stypy_return_type' (line 700)
        stypy_return_type_5511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5511)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_clip_on'
        return stypy_return_type_5511


    @norecursion
    def get_clip_box(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_clip_box'
        module_type_store = module_type_store.open_function_context('get_clip_box', 704, 4, False)
        # Assigning a type to the variable 'self' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_clip_box.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_clip_box.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_clip_box.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_clip_box.__dict__.__setitem__('stypy_function_name', 'Artist.get_clip_box')
        Artist.get_clip_box.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_clip_box.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_clip_box.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_clip_box.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_clip_box.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_clip_box.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_clip_box.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_clip_box', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_clip_box', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_clip_box(...)' code ##################

        unicode_5512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 8), 'unicode', u'Return artist clipbox')
        # Getting the type of 'self' (line 706)
        self_5513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 15), 'self')
        # Obtaining the member 'clipbox' of a type (line 706)
        clipbox_5514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 15), self_5513, 'clipbox')
        # Assigning a type to the variable 'stypy_return_type' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'stypy_return_type', clipbox_5514)
        
        # ################# End of 'get_clip_box(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_clip_box' in the type store
        # Getting the type of 'stypy_return_type' (line 704)
        stypy_return_type_5515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5515)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_clip_box'
        return stypy_return_type_5515


    @norecursion
    def get_clip_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_clip_path'
        module_type_store = module_type_store.open_function_context('get_clip_path', 708, 4, False)
        # Assigning a type to the variable 'self' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_clip_path.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_clip_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_clip_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_clip_path.__dict__.__setitem__('stypy_function_name', 'Artist.get_clip_path')
        Artist.get_clip_path.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_clip_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_clip_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_clip_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_clip_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_clip_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_clip_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_clip_path', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_clip_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_clip_path(...)' code ##################

        unicode_5516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 8), 'unicode', u'Return artist clip path')
        # Getting the type of 'self' (line 710)
        self_5517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 15), 'self')
        # Obtaining the member '_clippath' of a type (line 710)
        _clippath_5518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 15), self_5517, '_clippath')
        # Assigning a type to the variable 'stypy_return_type' (line 710)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 8), 'stypy_return_type', _clippath_5518)
        
        # ################# End of 'get_clip_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_clip_path' in the type store
        # Getting the type of 'stypy_return_type' (line 708)
        stypy_return_type_5519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5519)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_clip_path'
        return stypy_return_type_5519


    @norecursion
    def get_transformed_clip_path_and_affine(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_transformed_clip_path_and_affine'
        module_type_store = module_type_store.open_function_context('get_transformed_clip_path_and_affine', 712, 4, False)
        # Assigning a type to the variable 'self' (line 713)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_transformed_clip_path_and_affine.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_transformed_clip_path_and_affine.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_transformed_clip_path_and_affine.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_transformed_clip_path_and_affine.__dict__.__setitem__('stypy_function_name', 'Artist.get_transformed_clip_path_and_affine')
        Artist.get_transformed_clip_path_and_affine.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_transformed_clip_path_and_affine.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_transformed_clip_path_and_affine.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_transformed_clip_path_and_affine.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_transformed_clip_path_and_affine.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_transformed_clip_path_and_affine.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_transformed_clip_path_and_affine.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_transformed_clip_path_and_affine', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_transformed_clip_path_and_affine', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_transformed_clip_path_and_affine(...)' code ##################

        unicode_5520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, (-1)), 'unicode', u'\n        Return the clip path with the non-affine part of its\n        transformation applied, and the remaining affine part of its\n        transformation.\n        ')
        
        
        # Getting the type of 'self' (line 718)
        self_5521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 11), 'self')
        # Obtaining the member '_clippath' of a type (line 718)
        _clippath_5522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 11), self_5521, '_clippath')
        # Getting the type of 'None' (line 718)
        None_5523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 33), 'None')
        # Applying the binary operator 'isnot' (line 718)
        result_is_not_5524 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 11), 'isnot', _clippath_5522, None_5523)
        
        # Testing the type of an if condition (line 718)
        if_condition_5525 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 718, 8), result_is_not_5524)
        # Assigning a type to the variable 'if_condition_5525' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'if_condition_5525', if_condition_5525)
        # SSA begins for if statement (line 718)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to get_transformed_path_and_affine(...): (line 719)
        # Processing the call keyword arguments (line 719)
        kwargs_5529 = {}
        # Getting the type of 'self' (line 719)
        self_5526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 19), 'self', False)
        # Obtaining the member '_clippath' of a type (line 719)
        _clippath_5527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 19), self_5526, '_clippath')
        # Obtaining the member 'get_transformed_path_and_affine' of a type (line 719)
        get_transformed_path_and_affine_5528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 19), _clippath_5527, 'get_transformed_path_and_affine')
        # Calling get_transformed_path_and_affine(args, kwargs) (line 719)
        get_transformed_path_and_affine_call_result_5530 = invoke(stypy.reporting.localization.Localization(__file__, 719, 19), get_transformed_path_and_affine_5528, *[], **kwargs_5529)
        
        # Assigning a type to the variable 'stypy_return_type' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'stypy_return_type', get_transformed_path_and_affine_call_result_5530)
        # SSA join for if statement (line 718)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 720)
        tuple_5531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 720)
        # Adding element type (line 720)
        # Getting the type of 'None' (line 720)
        None_5532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 15), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 15), tuple_5531, None_5532)
        # Adding element type (line 720)
        # Getting the type of 'None' (line 720)
        None_5533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 21), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 15), tuple_5531, None_5533)
        
        # Assigning a type to the variable 'stypy_return_type' (line 720)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'stypy_return_type', tuple_5531)
        
        # ################# End of 'get_transformed_clip_path_and_affine(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_transformed_clip_path_and_affine' in the type store
        # Getting the type of 'stypy_return_type' (line 712)
        stypy_return_type_5534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5534)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_transformed_clip_path_and_affine'
        return stypy_return_type_5534


    @norecursion
    def set_clip_on(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_clip_on'
        module_type_store = module_type_store.open_function_context('set_clip_on', 722, 4, False)
        # Assigning a type to the variable 'self' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_clip_on.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_clip_on.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_clip_on.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_clip_on.__dict__.__setitem__('stypy_function_name', 'Artist.set_clip_on')
        Artist.set_clip_on.__dict__.__setitem__('stypy_param_names_list', ['b'])
        Artist.set_clip_on.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_clip_on.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_clip_on.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_clip_on.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_clip_on.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_clip_on.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_clip_on', ['b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_clip_on', localization, ['b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_clip_on(...)' code ##################

        unicode_5535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, (-1)), 'unicode', u'\n        Set whether artist uses clipping.\n\n        When False artists will be visible out side of the axes which\n        can lead to unexpected results.\n\n        ACCEPTS: [True | False]\n        ')
        
        # Assigning a Name to a Attribute (line 731):
        
        # Assigning a Name to a Attribute (line 731):
        # Getting the type of 'b' (line 731)
        b_5536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 23), 'b')
        # Getting the type of 'self' (line 731)
        self_5537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 8), 'self')
        # Setting the type of the member '_clipon' of a type (line 731)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 8), self_5537, '_clipon', b_5536)
        
        # Call to pchanged(...): (line 734)
        # Processing the call keyword arguments (line 734)
        kwargs_5540 = {}
        # Getting the type of 'self' (line 734)
        self_5538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 734)
        pchanged_5539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 8), self_5538, 'pchanged')
        # Calling pchanged(args, kwargs) (line 734)
        pchanged_call_result_5541 = invoke(stypy.reporting.localization.Localization(__file__, 734, 8), pchanged_5539, *[], **kwargs_5540)
        
        
        # Assigning a Name to a Attribute (line 735):
        
        # Assigning a Name to a Attribute (line 735):
        # Getting the type of 'True' (line 735)
        True_5542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 21), 'True')
        # Getting the type of 'self' (line 735)
        self_5543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 735)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 8), self_5543, 'stale', True_5542)
        
        # ################# End of 'set_clip_on(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_clip_on' in the type store
        # Getting the type of 'stypy_return_type' (line 722)
        stypy_return_type_5544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5544)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_clip_on'
        return stypy_return_type_5544


    @norecursion
    def _set_gc_clip(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_gc_clip'
        module_type_store = module_type_store.open_function_context('_set_gc_clip', 737, 4, False)
        # Assigning a type to the variable 'self' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist._set_gc_clip.__dict__.__setitem__('stypy_localization', localization)
        Artist._set_gc_clip.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist._set_gc_clip.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist._set_gc_clip.__dict__.__setitem__('stypy_function_name', 'Artist._set_gc_clip')
        Artist._set_gc_clip.__dict__.__setitem__('stypy_param_names_list', ['gc'])
        Artist._set_gc_clip.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist._set_gc_clip.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist._set_gc_clip.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist._set_gc_clip.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist._set_gc_clip.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist._set_gc_clip.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist._set_gc_clip', ['gc'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_gc_clip', localization, ['gc'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_gc_clip(...)' code ##################

        unicode_5545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 8), 'unicode', u'Set the clip properly for the gc')
        
        # Getting the type of 'self' (line 739)
        self_5546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 11), 'self')
        # Obtaining the member '_clipon' of a type (line 739)
        _clipon_5547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 11), self_5546, '_clipon')
        # Testing the type of an if condition (line 739)
        if_condition_5548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 739, 8), _clipon_5547)
        # Assigning a type to the variable 'if_condition_5548' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'if_condition_5548', if_condition_5548)
        # SSA begins for if statement (line 739)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 740)
        self_5549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 15), 'self')
        # Obtaining the member 'clipbox' of a type (line 740)
        clipbox_5550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 15), self_5549, 'clipbox')
        # Getting the type of 'None' (line 740)
        None_5551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 35), 'None')
        # Applying the binary operator 'isnot' (line 740)
        result_is_not_5552 = python_operator(stypy.reporting.localization.Localization(__file__, 740, 15), 'isnot', clipbox_5550, None_5551)
        
        # Testing the type of an if condition (line 740)
        if_condition_5553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 740, 12), result_is_not_5552)
        # Assigning a type to the variable 'if_condition_5553' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 'if_condition_5553', if_condition_5553)
        # SSA begins for if statement (line 740)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_clip_rectangle(...): (line 741)
        # Processing the call arguments (line 741)
        # Getting the type of 'self' (line 741)
        self_5556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 38), 'self', False)
        # Obtaining the member 'clipbox' of a type (line 741)
        clipbox_5557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 38), self_5556, 'clipbox')
        # Processing the call keyword arguments (line 741)
        kwargs_5558 = {}
        # Getting the type of 'gc' (line 741)
        gc_5554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 16), 'gc', False)
        # Obtaining the member 'set_clip_rectangle' of a type (line 741)
        set_clip_rectangle_5555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 16), gc_5554, 'set_clip_rectangle')
        # Calling set_clip_rectangle(args, kwargs) (line 741)
        set_clip_rectangle_call_result_5559 = invoke(stypy.reporting.localization.Localization(__file__, 741, 16), set_clip_rectangle_5555, *[clipbox_5557], **kwargs_5558)
        
        # SSA join for if statement (line 740)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_clip_path(...): (line 742)
        # Processing the call arguments (line 742)
        # Getting the type of 'self' (line 742)
        self_5562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 29), 'self', False)
        # Obtaining the member '_clippath' of a type (line 742)
        _clippath_5563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 29), self_5562, '_clippath')
        # Processing the call keyword arguments (line 742)
        kwargs_5564 = {}
        # Getting the type of 'gc' (line 742)
        gc_5560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 12), 'gc', False)
        # Obtaining the member 'set_clip_path' of a type (line 742)
        set_clip_path_5561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 12), gc_5560, 'set_clip_path')
        # Calling set_clip_path(args, kwargs) (line 742)
        set_clip_path_call_result_5565 = invoke(stypy.reporting.localization.Localization(__file__, 742, 12), set_clip_path_5561, *[_clippath_5563], **kwargs_5564)
        
        # SSA branch for the else part of an if statement (line 739)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_clip_rectangle(...): (line 744)
        # Processing the call arguments (line 744)
        # Getting the type of 'None' (line 744)
        None_5568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 34), 'None', False)
        # Processing the call keyword arguments (line 744)
        kwargs_5569 = {}
        # Getting the type of 'gc' (line 744)
        gc_5566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 12), 'gc', False)
        # Obtaining the member 'set_clip_rectangle' of a type (line 744)
        set_clip_rectangle_5567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 12), gc_5566, 'set_clip_rectangle')
        # Calling set_clip_rectangle(args, kwargs) (line 744)
        set_clip_rectangle_call_result_5570 = invoke(stypy.reporting.localization.Localization(__file__, 744, 12), set_clip_rectangle_5567, *[None_5568], **kwargs_5569)
        
        
        # Call to set_clip_path(...): (line 745)
        # Processing the call arguments (line 745)
        # Getting the type of 'None' (line 745)
        None_5573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 29), 'None', False)
        # Processing the call keyword arguments (line 745)
        kwargs_5574 = {}
        # Getting the type of 'gc' (line 745)
        gc_5571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 12), 'gc', False)
        # Obtaining the member 'set_clip_path' of a type (line 745)
        set_clip_path_5572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 12), gc_5571, 'set_clip_path')
        # Calling set_clip_path(args, kwargs) (line 745)
        set_clip_path_call_result_5575 = invoke(stypy.reporting.localization.Localization(__file__, 745, 12), set_clip_path_5572, *[None_5573], **kwargs_5574)
        
        # SSA join for if statement (line 739)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_set_gc_clip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_gc_clip' in the type store
        # Getting the type of 'stypy_return_type' (line 737)
        stypy_return_type_5576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5576)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_gc_clip'
        return stypy_return_type_5576


    @norecursion
    def get_rasterized(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_rasterized'
        module_type_store = module_type_store.open_function_context('get_rasterized', 747, 4, False)
        # Assigning a type to the variable 'self' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_rasterized.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_rasterized.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_rasterized.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_rasterized.__dict__.__setitem__('stypy_function_name', 'Artist.get_rasterized')
        Artist.get_rasterized.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_rasterized.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_rasterized.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_rasterized.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_rasterized.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_rasterized.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_rasterized.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_rasterized', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_rasterized', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_rasterized(...)' code ##################

        unicode_5577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 8), 'unicode', u'return True if the artist is to be rasterized')
        # Getting the type of 'self' (line 749)
        self_5578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 15), 'self')
        # Obtaining the member '_rasterized' of a type (line 749)
        _rasterized_5579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 15), self_5578, '_rasterized')
        # Assigning a type to the variable 'stypy_return_type' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'stypy_return_type', _rasterized_5579)
        
        # ################# End of 'get_rasterized(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_rasterized' in the type store
        # Getting the type of 'stypy_return_type' (line 747)
        stypy_return_type_5580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5580)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_rasterized'
        return stypy_return_type_5580


    @norecursion
    def set_rasterized(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_rasterized'
        module_type_store = module_type_store.open_function_context('set_rasterized', 751, 4, False)
        # Assigning a type to the variable 'self' (line 752)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_rasterized.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_rasterized.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_rasterized.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_rasterized.__dict__.__setitem__('stypy_function_name', 'Artist.set_rasterized')
        Artist.set_rasterized.__dict__.__setitem__('stypy_param_names_list', ['rasterized'])
        Artist.set_rasterized.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_rasterized.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_rasterized.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_rasterized.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_rasterized.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_rasterized.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_rasterized', ['rasterized'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_rasterized', localization, ['rasterized'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_rasterized(...)' code ##################

        unicode_5581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, (-1)), 'unicode', u"\n        Force rasterized (bitmap) drawing in vector backend output.\n\n        Defaults to None, which implies the backend's default behavior\n\n        ACCEPTS: [True | False | None]\n        ")
        
        
        # Evaluating a boolean operation
        # Getting the type of 'rasterized' (line 759)
        rasterized_5582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 11), 'rasterized')
        
        
        # Call to hasattr(...): (line 759)
        # Processing the call arguments (line 759)
        # Getting the type of 'self' (line 759)
        self_5584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 38), 'self', False)
        # Obtaining the member 'draw' of a type (line 759)
        draw_5585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 38), self_5584, 'draw')
        unicode_5586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 49), 'unicode', u'_supports_rasterization')
        # Processing the call keyword arguments (line 759)
        kwargs_5587 = {}
        # Getting the type of 'hasattr' (line 759)
        hasattr_5583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 30), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 759)
        hasattr_call_result_5588 = invoke(stypy.reporting.localization.Localization(__file__, 759, 30), hasattr_5583, *[draw_5585, unicode_5586], **kwargs_5587)
        
        # Applying the 'not' unary operator (line 759)
        result_not__5589 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 26), 'not', hasattr_call_result_5588)
        
        # Applying the binary operator 'and' (line 759)
        result_and_keyword_5590 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 11), 'and', rasterized_5582, result_not__5589)
        
        # Testing the type of an if condition (line 759)
        if_condition_5591 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 759, 8), result_and_keyword_5590)
        # Assigning a type to the variable 'if_condition_5591' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'if_condition_5591', if_condition_5591)
        # SSA begins for if statement (line 759)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 760)
        # Processing the call arguments (line 760)
        unicode_5594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 26), 'unicode', u"Rasterization of '%s' will be ignored")
        # Getting the type of 'self' (line 760)
        self_5595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 68), 'self', False)
        # Applying the binary operator '%' (line 760)
        result_mod_5596 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 26), '%', unicode_5594, self_5595)
        
        # Processing the call keyword arguments (line 760)
        kwargs_5597 = {}
        # Getting the type of 'warnings' (line 760)
        warnings_5592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 760)
        warn_5593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 12), warnings_5592, 'warn')
        # Calling warn(args, kwargs) (line 760)
        warn_call_result_5598 = invoke(stypy.reporting.localization.Localization(__file__, 760, 12), warn_5593, *[result_mod_5596], **kwargs_5597)
        
        # SSA join for if statement (line 759)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 762):
        
        # Assigning a Name to a Attribute (line 762):
        # Getting the type of 'rasterized' (line 762)
        rasterized_5599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 27), 'rasterized')
        # Getting the type of 'self' (line 762)
        self_5600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'self')
        # Setting the type of the member '_rasterized' of a type (line 762)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 8), self_5600, '_rasterized', rasterized_5599)
        
        # ################# End of 'set_rasterized(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_rasterized' in the type store
        # Getting the type of 'stypy_return_type' (line 751)
        stypy_return_type_5601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5601)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_rasterized'
        return stypy_return_type_5601


    @norecursion
    def get_agg_filter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_agg_filter'
        module_type_store = module_type_store.open_function_context('get_agg_filter', 764, 4, False)
        # Assigning a type to the variable 'self' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_agg_filter.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_agg_filter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_agg_filter.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_agg_filter.__dict__.__setitem__('stypy_function_name', 'Artist.get_agg_filter')
        Artist.get_agg_filter.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_agg_filter.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_agg_filter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_agg_filter.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_agg_filter.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_agg_filter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_agg_filter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_agg_filter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_agg_filter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_agg_filter(...)' code ##################

        unicode_5602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 8), 'unicode', u'return filter function to be used for agg filter')
        # Getting the type of 'self' (line 766)
        self_5603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 15), 'self')
        # Obtaining the member '_agg_filter' of a type (line 766)
        _agg_filter_5604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 15), self_5603, '_agg_filter')
        # Assigning a type to the variable 'stypy_return_type' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'stypy_return_type', _agg_filter_5604)
        
        # ################# End of 'get_agg_filter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_agg_filter' in the type store
        # Getting the type of 'stypy_return_type' (line 764)
        stypy_return_type_5605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5605)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_agg_filter'
        return stypy_return_type_5605


    @norecursion
    def set_agg_filter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_agg_filter'
        module_type_store = module_type_store.open_function_context('set_agg_filter', 768, 4, False)
        # Assigning a type to the variable 'self' (line 769)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_agg_filter.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_agg_filter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_agg_filter.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_agg_filter.__dict__.__setitem__('stypy_function_name', 'Artist.set_agg_filter')
        Artist.set_agg_filter.__dict__.__setitem__('stypy_param_names_list', ['filter_func'])
        Artist.set_agg_filter.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_agg_filter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_agg_filter.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_agg_filter.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_agg_filter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_agg_filter.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_agg_filter', ['filter_func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_agg_filter', localization, ['filter_func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_agg_filter(...)' code ##################

        unicode_5606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, (-1)), 'unicode', u'\n        set agg_filter function.\n\n        ')
        
        # Assigning a Name to a Attribute (line 773):
        
        # Assigning a Name to a Attribute (line 773):
        # Getting the type of 'filter_func' (line 773)
        filter_func_5607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 27), 'filter_func')
        # Getting the type of 'self' (line 773)
        self_5608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'self')
        # Setting the type of the member '_agg_filter' of a type (line 773)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 8), self_5608, '_agg_filter', filter_func_5607)
        
        # Assigning a Name to a Attribute (line 774):
        
        # Assigning a Name to a Attribute (line 774):
        # Getting the type of 'True' (line 774)
        True_5609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 21), 'True')
        # Getting the type of 'self' (line 774)
        self_5610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 774)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 8), self_5610, 'stale', True_5609)
        
        # ################# End of 'set_agg_filter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_agg_filter' in the type store
        # Getting the type of 'stypy_return_type' (line 768)
        stypy_return_type_5611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5611)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_agg_filter'
        return stypy_return_type_5611


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 776, 4, False)
        # Assigning a type to the variable 'self' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.draw.__dict__.__setitem__('stypy_localization', localization)
        Artist.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.draw.__dict__.__setitem__('stypy_function_name', 'Artist.draw')
        Artist.draw.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Artist.draw.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Artist.draw.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Artist.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.draw.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.draw', ['renderer'], 'args', 'kwargs', defaults, varargs, kwargs)

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

        unicode_5612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 8), 'unicode', u'Derived classes drawing method')
        
        
        
        # Call to get_visible(...): (line 778)
        # Processing the call keyword arguments (line 778)
        kwargs_5615 = {}
        # Getting the type of 'self' (line 778)
        self_5613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 15), 'self', False)
        # Obtaining the member 'get_visible' of a type (line 778)
        get_visible_5614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 15), self_5613, 'get_visible')
        # Calling get_visible(args, kwargs) (line 778)
        get_visible_call_result_5616 = invoke(stypy.reporting.localization.Localization(__file__, 778, 15), get_visible_5614, *[], **kwargs_5615)
        
        # Applying the 'not' unary operator (line 778)
        result_not__5617 = python_operator(stypy.reporting.localization.Localization(__file__, 778, 11), 'not', get_visible_call_result_5616)
        
        # Testing the type of an if condition (line 778)
        if_condition_5618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 778, 8), result_not__5617)
        # Assigning a type to the variable 'if_condition_5618' (line 778)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 8), 'if_condition_5618', if_condition_5618)
        # SSA begins for if statement (line 778)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 778)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 780):
        
        # Assigning a Name to a Attribute (line 780):
        # Getting the type of 'False' (line 780)
        False_5619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 21), 'False')
        # Getting the type of 'self' (line 780)
        self_5620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 780)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 8), self_5620, 'stale', False_5619)
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 776)
        stypy_return_type_5621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5621)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_5621


    @norecursion
    def set_alpha(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_alpha'
        module_type_store = module_type_store.open_function_context('set_alpha', 782, 4, False)
        # Assigning a type to the variable 'self' (line 783)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_alpha.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_alpha.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_alpha.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_alpha.__dict__.__setitem__('stypy_function_name', 'Artist.set_alpha')
        Artist.set_alpha.__dict__.__setitem__('stypy_param_names_list', ['alpha'])
        Artist.set_alpha.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_alpha.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_alpha.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_alpha.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_alpha.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_alpha.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_alpha', ['alpha'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_alpha', localization, ['alpha'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_alpha(...)' code ##################

        unicode_5622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, (-1)), 'unicode', u'\n        Set the alpha value used for blending - not supported on\n        all backends.\n\n        ACCEPTS: float (0.0 transparent through 1.0 opaque)\n        ')
        
        # Assigning a Name to a Attribute (line 789):
        
        # Assigning a Name to a Attribute (line 789):
        # Getting the type of 'alpha' (line 789)
        alpha_5623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 22), 'alpha')
        # Getting the type of 'self' (line 789)
        self_5624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 8), 'self')
        # Setting the type of the member '_alpha' of a type (line 789)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 8), self_5624, '_alpha', alpha_5623)
        
        # Call to pchanged(...): (line 790)
        # Processing the call keyword arguments (line 790)
        kwargs_5627 = {}
        # Getting the type of 'self' (line 790)
        self_5625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 790)
        pchanged_5626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 8), self_5625, 'pchanged')
        # Calling pchanged(args, kwargs) (line 790)
        pchanged_call_result_5628 = invoke(stypy.reporting.localization.Localization(__file__, 790, 8), pchanged_5626, *[], **kwargs_5627)
        
        
        # Assigning a Name to a Attribute (line 791):
        
        # Assigning a Name to a Attribute (line 791):
        # Getting the type of 'True' (line 791)
        True_5629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 21), 'True')
        # Getting the type of 'self' (line 791)
        self_5630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 791)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 8), self_5630, 'stale', True_5629)
        
        # ################# End of 'set_alpha(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_alpha' in the type store
        # Getting the type of 'stypy_return_type' (line 782)
        stypy_return_type_5631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5631)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_alpha'
        return stypy_return_type_5631


    @norecursion
    def set_visible(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_visible'
        module_type_store = module_type_store.open_function_context('set_visible', 793, 4, False)
        # Assigning a type to the variable 'self' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_visible.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_visible.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_visible.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_visible.__dict__.__setitem__('stypy_function_name', 'Artist.set_visible')
        Artist.set_visible.__dict__.__setitem__('stypy_param_names_list', ['b'])
        Artist.set_visible.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_visible.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_visible.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_visible.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_visible.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_visible.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_visible', ['b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_visible', localization, ['b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_visible(...)' code ##################

        unicode_5632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, (-1)), 'unicode', u"\n        Set the artist's visiblity.\n\n        ACCEPTS: [True | False]\n        ")
        
        # Assigning a Name to a Attribute (line 799):
        
        # Assigning a Name to a Attribute (line 799):
        # Getting the type of 'b' (line 799)
        b_5633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 24), 'b')
        # Getting the type of 'self' (line 799)
        self_5634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 8), 'self')
        # Setting the type of the member '_visible' of a type (line 799)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 8), self_5634, '_visible', b_5633)
        
        # Call to pchanged(...): (line 800)
        # Processing the call keyword arguments (line 800)
        kwargs_5637 = {}
        # Getting the type of 'self' (line 800)
        self_5635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 8), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 800)
        pchanged_5636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 8), self_5635, 'pchanged')
        # Calling pchanged(args, kwargs) (line 800)
        pchanged_call_result_5638 = invoke(stypy.reporting.localization.Localization(__file__, 800, 8), pchanged_5636, *[], **kwargs_5637)
        
        
        # Assigning a Name to a Attribute (line 801):
        
        # Assigning a Name to a Attribute (line 801):
        # Getting the type of 'True' (line 801)
        True_5639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 21), 'True')
        # Getting the type of 'self' (line 801)
        self_5640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 801)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 8), self_5640, 'stale', True_5639)
        
        # ################# End of 'set_visible(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_visible' in the type store
        # Getting the type of 'stypy_return_type' (line 793)
        stypy_return_type_5641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_visible'
        return stypy_return_type_5641


    @norecursion
    def set_animated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_animated'
        module_type_store = module_type_store.open_function_context('set_animated', 803, 4, False)
        # Assigning a type to the variable 'self' (line 804)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_animated.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_animated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_animated.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_animated.__dict__.__setitem__('stypy_function_name', 'Artist.set_animated')
        Artist.set_animated.__dict__.__setitem__('stypy_param_names_list', ['b'])
        Artist.set_animated.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_animated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_animated.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_animated.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_animated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_animated.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_animated', ['b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_animated', localization, ['b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_animated(...)' code ##################

        unicode_5642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, (-1)), 'unicode', u"\n        Set the artist's animation state.\n\n        ACCEPTS: [True | False]\n        ")
        
        
        # Getting the type of 'self' (line 809)
        self_5643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 11), 'self')
        # Obtaining the member '_animated' of a type (line 809)
        _animated_5644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 11), self_5643, '_animated')
        # Getting the type of 'b' (line 809)
        b_5645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 29), 'b')
        # Applying the binary operator '!=' (line 809)
        result_ne_5646 = python_operator(stypy.reporting.localization.Localization(__file__, 809, 11), '!=', _animated_5644, b_5645)
        
        # Testing the type of an if condition (line 809)
        if_condition_5647 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 809, 8), result_ne_5646)
        # Assigning a type to the variable 'if_condition_5647' (line 809)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 8), 'if_condition_5647', if_condition_5647)
        # SSA begins for if statement (line 809)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 810):
        
        # Assigning a Name to a Attribute (line 810):
        # Getting the type of 'b' (line 810)
        b_5648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 29), 'b')
        # Getting the type of 'self' (line 810)
        self_5649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 12), 'self')
        # Setting the type of the member '_animated' of a type (line 810)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 12), self_5649, '_animated', b_5648)
        
        # Call to pchanged(...): (line 811)
        # Processing the call keyword arguments (line 811)
        kwargs_5652 = {}
        # Getting the type of 'self' (line 811)
        self_5650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 12), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 811)
        pchanged_5651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 12), self_5650, 'pchanged')
        # Calling pchanged(args, kwargs) (line 811)
        pchanged_call_result_5653 = invoke(stypy.reporting.localization.Localization(__file__, 811, 12), pchanged_5651, *[], **kwargs_5652)
        
        # SSA join for if statement (line 809)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_animated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_animated' in the type store
        # Getting the type of 'stypy_return_type' (line 803)
        stypy_return_type_5654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5654)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_animated'
        return stypy_return_type_5654


    @norecursion
    def update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update'
        module_type_store = module_type_store.open_function_context('update', 813, 4, False)
        # Assigning a type to the variable 'self' (line 814)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.update.__dict__.__setitem__('stypy_localization', localization)
        Artist.update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.update.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.update.__dict__.__setitem__('stypy_function_name', 'Artist.update')
        Artist.update.__dict__.__setitem__('stypy_param_names_list', ['props'])
        Artist.update.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.update.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.update.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.update.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.update', ['props'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update', localization, ['props'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update(...)' code ##################

        unicode_5655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, (-1)), 'unicode', u'\n        Update the properties of this :class:`Artist` from the\n        dictionary *prop*.\n        ')

        @norecursion
        def _update_property(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_update_property'
            module_type_store = module_type_store.open_function_context('_update_property', 818, 8, False)
            
            # Passed parameters checking function
            _update_property.stypy_localization = localization
            _update_property.stypy_type_of_self = None
            _update_property.stypy_type_store = module_type_store
            _update_property.stypy_function_name = '_update_property'
            _update_property.stypy_param_names_list = ['self', 'k', 'v']
            _update_property.stypy_varargs_param_name = None
            _update_property.stypy_kwargs_param_name = None
            _update_property.stypy_call_defaults = defaults
            _update_property.stypy_call_varargs = varargs
            _update_property.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_update_property', ['self', 'k', 'v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_update_property', localization, ['self', 'k', 'v'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_update_property(...)' code ##################

            unicode_5656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, (-1)), 'unicode', u"sorting out how to update property (setter or setattr)\n\n            Parameters\n            ----------\n            k : str\n                The name of property to update\n            v : obj\n                The value to assign to the property\n            Returns\n            -------\n            ret : obj or None\n                If using a `set_*` method return it's return, else None.\n            ")
            
            # Assigning a Call to a Name (line 832):
            
            # Assigning a Call to a Name (line 832):
            
            # Call to lower(...): (line 832)
            # Processing the call keyword arguments (line 832)
            kwargs_5659 = {}
            # Getting the type of 'k' (line 832)
            k_5657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 16), 'k', False)
            # Obtaining the member 'lower' of a type (line 832)
            lower_5658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 16), k_5657, 'lower')
            # Calling lower(args, kwargs) (line 832)
            lower_call_result_5660 = invoke(stypy.reporting.localization.Localization(__file__, 832, 16), lower_5658, *[], **kwargs_5659)
            
            # Assigning a type to the variable 'k' (line 832)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 12), 'k', lower_call_result_5660)
            
            
            # Getting the type of 'k' (line 835)
            k_5661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 15), 'k')
            
            # Obtaining an instance of the builtin type 'set' (line 835)
            set_5662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 20), 'set')
            # Adding type elements to the builtin type 'set' instance (line 835)
            # Adding element type (line 835)
            unicode_5663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 21), 'unicode', u'axes')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 835, 20), set_5662, unicode_5663)
            
            # Applying the binary operator 'in' (line 835)
            result_contains_5664 = python_operator(stypy.reporting.localization.Localization(__file__, 835, 15), 'in', k_5661, set_5662)
            
            # Testing the type of an if condition (line 835)
            if_condition_5665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 835, 12), result_contains_5664)
            # Assigning a type to the variable 'if_condition_5665' (line 835)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 12), 'if_condition_5665', if_condition_5665)
            # SSA begins for if statement (line 835)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to setattr(...): (line 836)
            # Processing the call arguments (line 836)
            # Getting the type of 'self' (line 836)
            self_5667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 31), 'self', False)
            # Getting the type of 'k' (line 836)
            k_5668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 37), 'k', False)
            # Getting the type of 'v' (line 836)
            v_5669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 40), 'v', False)
            # Processing the call keyword arguments (line 836)
            kwargs_5670 = {}
            # Getting the type of 'setattr' (line 836)
            setattr_5666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 23), 'setattr', False)
            # Calling setattr(args, kwargs) (line 836)
            setattr_call_result_5671 = invoke(stypy.reporting.localization.Localization(__file__, 836, 23), setattr_5666, *[self_5667, k_5668, v_5669], **kwargs_5670)
            
            # Assigning a type to the variable 'stypy_return_type' (line 836)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 16), 'stypy_return_type', setattr_call_result_5671)
            # SSA branch for the else part of an if statement (line 835)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 838):
            
            # Assigning a Call to a Name (line 838):
            
            # Call to getattr(...): (line 838)
            # Processing the call arguments (line 838)
            # Getting the type of 'self' (line 838)
            self_5673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 31), 'self', False)
            unicode_5674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 37), 'unicode', u'set_')
            # Getting the type of 'k' (line 838)
            k_5675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 46), 'k', False)
            # Applying the binary operator '+' (line 838)
            result_add_5676 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 37), '+', unicode_5674, k_5675)
            
            # Getting the type of 'None' (line 838)
            None_5677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 49), 'None', False)
            # Processing the call keyword arguments (line 838)
            kwargs_5678 = {}
            # Getting the type of 'getattr' (line 838)
            getattr_5672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 23), 'getattr', False)
            # Calling getattr(args, kwargs) (line 838)
            getattr_call_result_5679 = invoke(stypy.reporting.localization.Localization(__file__, 838, 23), getattr_5672, *[self_5673, result_add_5676, None_5677], **kwargs_5678)
            
            # Assigning a type to the variable 'func' (line 838)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 16), 'func', getattr_call_result_5679)
            
            
            
            # Call to callable(...): (line 839)
            # Processing the call arguments (line 839)
            # Getting the type of 'func' (line 839)
            func_5681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 32), 'func', False)
            # Processing the call keyword arguments (line 839)
            kwargs_5682 = {}
            # Getting the type of 'callable' (line 839)
            callable_5680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 23), 'callable', False)
            # Calling callable(args, kwargs) (line 839)
            callable_call_result_5683 = invoke(stypy.reporting.localization.Localization(__file__, 839, 23), callable_5680, *[func_5681], **kwargs_5682)
            
            # Applying the 'not' unary operator (line 839)
            result_not__5684 = python_operator(stypy.reporting.localization.Localization(__file__, 839, 19), 'not', callable_call_result_5683)
            
            # Testing the type of an if condition (line 839)
            if_condition_5685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 839, 16), result_not__5684)
            # Assigning a type to the variable 'if_condition_5685' (line 839)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 16), 'if_condition_5685', if_condition_5685)
            # SSA begins for if statement (line 839)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to AttributeError(...): (line 840)
            # Processing the call arguments (line 840)
            unicode_5687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 41), 'unicode', u'Unknown property %s')
            # Getting the type of 'k' (line 840)
            k_5688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 65), 'k', False)
            # Applying the binary operator '%' (line 840)
            result_mod_5689 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 41), '%', unicode_5687, k_5688)
            
            # Processing the call keyword arguments (line 840)
            kwargs_5690 = {}
            # Getting the type of 'AttributeError' (line 840)
            AttributeError_5686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 26), 'AttributeError', False)
            # Calling AttributeError(args, kwargs) (line 840)
            AttributeError_call_result_5691 = invoke(stypy.reporting.localization.Localization(__file__, 840, 26), AttributeError_5686, *[result_mod_5689], **kwargs_5690)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 840, 20), AttributeError_call_result_5691, 'raise parameter', BaseException)
            # SSA join for if statement (line 839)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to func(...): (line 841)
            # Processing the call arguments (line 841)
            # Getting the type of 'v' (line 841)
            v_5693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 28), 'v', False)
            # Processing the call keyword arguments (line 841)
            kwargs_5694 = {}
            # Getting the type of 'func' (line 841)
            func_5692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 23), 'func', False)
            # Calling func(args, kwargs) (line 841)
            func_call_result_5695 = invoke(stypy.reporting.localization.Localization(__file__, 841, 23), func_5692, *[v_5693], **kwargs_5694)
            
            # Assigning a type to the variable 'stypy_return_type' (line 841)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 16), 'stypy_return_type', func_call_result_5695)
            # SSA join for if statement (line 835)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of '_update_property(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_update_property' in the type store
            # Getting the type of 'stypy_return_type' (line 818)
            stypy_return_type_5696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5696)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_update_property'
            return stypy_return_type_5696

        # Assigning a type to the variable '_update_property' (line 818)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 8), '_update_property', _update_property)
        
        # Assigning a Attribute to a Name (line 843):
        
        # Assigning a Attribute to a Name (line 843):
        # Getting the type of 'self' (line 843)
        self_5697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 16), 'self')
        # Obtaining the member 'eventson' of a type (line 843)
        eventson_5698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 16), self_5697, 'eventson')
        # Assigning a type to the variable 'store' (line 843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 8), 'store', eventson_5698)
        
        # Assigning a Name to a Attribute (line 844):
        
        # Assigning a Name to a Attribute (line 844):
        # Getting the type of 'False' (line 844)
        False_5699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 24), 'False')
        # Getting the type of 'self' (line 844)
        self_5700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'self')
        # Setting the type of the member 'eventson' of a type (line 844)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 8), self_5700, 'eventson', False_5699)
        
        # Try-finally block (line 845)
        
        # Assigning a ListComp to a Name (line 846):
        
        # Assigning a ListComp to a Name (line 846):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to items(...): (line 847)
        # Processing the call keyword arguments (line 847)
        kwargs_5709 = {}
        # Getting the type of 'props' (line 847)
        props_5707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 31), 'props', False)
        # Obtaining the member 'items' of a type (line 847)
        items_5708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 31), props_5707, 'items')
        # Calling items(args, kwargs) (line 847)
        items_call_result_5710 = invoke(stypy.reporting.localization.Localization(__file__, 847, 31), items_5708, *[], **kwargs_5709)
        
        comprehension_5711 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 19), items_call_result_5710)
        # Assigning a type to the variable 'k' (line 846)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 19), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 19), comprehension_5711))
        # Assigning a type to the variable 'v' (line 846)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 19), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 19), comprehension_5711))
        
        # Call to _update_property(...): (line 846)
        # Processing the call arguments (line 846)
        # Getting the type of 'self' (line 846)
        self_5702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 36), 'self', False)
        # Getting the type of 'k' (line 846)
        k_5703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 42), 'k', False)
        # Getting the type of 'v' (line 846)
        v_5704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 45), 'v', False)
        # Processing the call keyword arguments (line 846)
        kwargs_5705 = {}
        # Getting the type of '_update_property' (line 846)
        _update_property_5701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 19), '_update_property', False)
        # Calling _update_property(args, kwargs) (line 846)
        _update_property_call_result_5706 = invoke(stypy.reporting.localization.Localization(__file__, 846, 19), _update_property_5701, *[self_5702, k_5703, v_5704], **kwargs_5705)
        
        list_5712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 19), list_5712, _update_property_call_result_5706)
        # Assigning a type to the variable 'ret' (line 846)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 12), 'ret', list_5712)
        
        # finally branch of the try-finally block (line 845)
        
        # Assigning a Name to a Attribute (line 849):
        
        # Assigning a Name to a Attribute (line 849):
        # Getting the type of 'store' (line 849)
        store_5713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 28), 'store')
        # Getting the type of 'self' (line 849)
        self_5714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 12), 'self')
        # Setting the type of the member 'eventson' of a type (line 849)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 12), self_5714, 'eventson', store_5713)
        
        
        
        # Call to len(...): (line 851)
        # Processing the call arguments (line 851)
        # Getting the type of 'ret' (line 851)
        ret_5716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 15), 'ret', False)
        # Processing the call keyword arguments (line 851)
        kwargs_5717 = {}
        # Getting the type of 'len' (line 851)
        len_5715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 11), 'len', False)
        # Calling len(args, kwargs) (line 851)
        len_call_result_5718 = invoke(stypy.reporting.localization.Localization(__file__, 851, 11), len_5715, *[ret_5716], **kwargs_5717)
        
        # Testing the type of an if condition (line 851)
        if_condition_5719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 851, 8), len_call_result_5718)
        # Assigning a type to the variable 'if_condition_5719' (line 851)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'if_condition_5719', if_condition_5719)
        # SSA begins for if statement (line 851)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to pchanged(...): (line 852)
        # Processing the call keyword arguments (line 852)
        kwargs_5722 = {}
        # Getting the type of 'self' (line 852)
        self_5720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 12), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 852)
        pchanged_5721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 12), self_5720, 'pchanged')
        # Calling pchanged(args, kwargs) (line 852)
        pchanged_call_result_5723 = invoke(stypy.reporting.localization.Localization(__file__, 852, 12), pchanged_5721, *[], **kwargs_5722)
        
        
        # Assigning a Name to a Attribute (line 853):
        
        # Assigning a Name to a Attribute (line 853):
        # Getting the type of 'True' (line 853)
        True_5724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 25), 'True')
        # Getting the type of 'self' (line 853)
        self_5725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 12), 'self')
        # Setting the type of the member 'stale' of a type (line 853)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 12), self_5725, 'stale', True_5724)
        # SSA join for if statement (line 851)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'ret' (line 854)
        ret_5726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 8), 'stypy_return_type', ret_5726)
        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 813)
        stypy_return_type_5727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5727)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_5727


    @norecursion
    def get_label(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_label'
        module_type_store = module_type_store.open_function_context('get_label', 856, 4, False)
        # Assigning a type to the variable 'self' (line 857)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_label.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_label.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_label.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_label.__dict__.__setitem__('stypy_function_name', 'Artist.get_label')
        Artist.get_label.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_label.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_label.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_label.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_label.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_label.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_label.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_label', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_label', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_label(...)' code ##################

        unicode_5728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, (-1)), 'unicode', u'\n        Get the label used for this artist in the legend.\n        ')
        # Getting the type of 'self' (line 860)
        self_5729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 15), 'self')
        # Obtaining the member '_label' of a type (line 860)
        _label_5730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 15), self_5729, '_label')
        # Assigning a type to the variable 'stypy_return_type' (line 860)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 8), 'stypy_return_type', _label_5730)
        
        # ################# End of 'get_label(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_label' in the type store
        # Getting the type of 'stypy_return_type' (line 856)
        stypy_return_type_5731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5731)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_label'
        return stypy_return_type_5731


    @norecursion
    def set_label(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_label'
        module_type_store = module_type_store.open_function_context('set_label', 862, 4, False)
        # Assigning a type to the variable 'self' (line 863)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_label.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_label.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_label.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_label.__dict__.__setitem__('stypy_function_name', 'Artist.set_label')
        Artist.set_label.__dict__.__setitem__('stypy_param_names_list', ['s'])
        Artist.set_label.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_label.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_label.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_label.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_label.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_label.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_label', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_label', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_label(...)' code ##################

        unicode_5732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, (-1)), 'unicode', u"\n        Set the label to *s* for auto legend.\n\n        ACCEPTS: string or anything printable with '%s' conversion.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 868)
        # Getting the type of 's' (line 868)
        s_5733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 's')
        # Getting the type of 'None' (line 868)
        None_5734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 20), 'None')
        
        (may_be_5735, more_types_in_union_5736) = may_not_be_none(s_5733, None_5734)

        if may_be_5735:

            if more_types_in_union_5736:
                # Runtime conditional SSA (line 868)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Attribute (line 869):
            
            # Assigning a BinOp to a Attribute (line 869):
            unicode_5737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 26), 'unicode', u'%s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 869)
            tuple_5738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 34), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 869)
            # Adding element type (line 869)
            # Getting the type of 's' (line 869)
            s_5739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 34), 's')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 869, 34), tuple_5738, s_5739)
            
            # Applying the binary operator '%' (line 869)
            result_mod_5740 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 26), '%', unicode_5737, tuple_5738)
            
            # Getting the type of 'self' (line 869)
            self_5741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 12), 'self')
            # Setting the type of the member '_label' of a type (line 869)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 12), self_5741, '_label', result_mod_5740)

            if more_types_in_union_5736:
                # Runtime conditional SSA for else branch (line 868)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_5735) or more_types_in_union_5736):
            
            # Assigning a Name to a Attribute (line 871):
            
            # Assigning a Name to a Attribute (line 871):
            # Getting the type of 'None' (line 871)
            None_5742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 26), 'None')
            # Getting the type of 'self' (line 871)
            self_5743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 12), 'self')
            # Setting the type of the member '_label' of a type (line 871)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 12), self_5743, '_label', None_5742)

            if (may_be_5735 and more_types_in_union_5736):
                # SSA join for if statement (line 868)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to pchanged(...): (line 872)
        # Processing the call keyword arguments (line 872)
        kwargs_5746 = {}
        # Getting the type of 'self' (line 872)
        self_5744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 8), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 872)
        pchanged_5745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 8), self_5744, 'pchanged')
        # Calling pchanged(args, kwargs) (line 872)
        pchanged_call_result_5747 = invoke(stypy.reporting.localization.Localization(__file__, 872, 8), pchanged_5745, *[], **kwargs_5746)
        
        
        # Assigning a Name to a Attribute (line 873):
        
        # Assigning a Name to a Attribute (line 873):
        # Getting the type of 'True' (line 873)
        True_5748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 21), 'True')
        # Getting the type of 'self' (line 873)
        self_5749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 873)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 873, 8), self_5749, 'stale', True_5748)
        
        # ################# End of 'set_label(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_label' in the type store
        # Getting the type of 'stypy_return_type' (line 862)
        stypy_return_type_5750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5750)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_label'
        return stypy_return_type_5750


    @norecursion
    def get_zorder(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_zorder'
        module_type_store = module_type_store.open_function_context('get_zorder', 875, 4, False)
        # Assigning a type to the variable 'self' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_zorder.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_zorder.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_zorder.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_zorder.__dict__.__setitem__('stypy_function_name', 'Artist.get_zorder')
        Artist.get_zorder.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.get_zorder.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_zorder.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_zorder.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_zorder.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_zorder.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_zorder.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_zorder', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_zorder', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_zorder(...)' code ##################

        unicode_5751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 878, (-1)), 'unicode', u"\n        Return the :class:`Artist`'s zorder.\n        ")
        # Getting the type of 'self' (line 879)
        self_5752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 15), 'self')
        # Obtaining the member 'zorder' of a type (line 879)
        zorder_5753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 15), self_5752, 'zorder')
        # Assigning a type to the variable 'stypy_return_type' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'stypy_return_type', zorder_5753)
        
        # ################# End of 'get_zorder(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_zorder' in the type store
        # Getting the type of 'stypy_return_type' (line 875)
        stypy_return_type_5754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5754)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_zorder'
        return stypy_return_type_5754


    @norecursion
    def set_zorder(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_zorder'
        module_type_store = module_type_store.open_function_context('set_zorder', 881, 4, False)
        # Assigning a type to the variable 'self' (line 882)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set_zorder.__dict__.__setitem__('stypy_localization', localization)
        Artist.set_zorder.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set_zorder.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set_zorder.__dict__.__setitem__('stypy_function_name', 'Artist.set_zorder')
        Artist.set_zorder.__dict__.__setitem__('stypy_param_names_list', ['level'])
        Artist.set_zorder.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set_zorder.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.set_zorder.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set_zorder.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set_zorder.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set_zorder.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set_zorder', ['level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_zorder', localization, ['level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_zorder(...)' code ##################

        unicode_5755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, (-1)), 'unicode', u'\n        Set the zorder for the artist.  Artists with lower zorder\n        values are drawn first.\n\n        ACCEPTS: any number\n        ')
        
        # Assigning a Name to a Attribute (line 888):
        
        # Assigning a Name to a Attribute (line 888):
        # Getting the type of 'level' (line 888)
        level_5756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 22), 'level')
        # Getting the type of 'self' (line 888)
        self_5757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'self')
        # Setting the type of the member 'zorder' of a type (line 888)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 8), self_5757, 'zorder', level_5756)
        
        # Call to pchanged(...): (line 889)
        # Processing the call keyword arguments (line 889)
        kwargs_5760 = {}
        # Getting the type of 'self' (line 889)
        self_5758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 8), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 889)
        pchanged_5759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 8), self_5758, 'pchanged')
        # Calling pchanged(args, kwargs) (line 889)
        pchanged_call_result_5761 = invoke(stypy.reporting.localization.Localization(__file__, 889, 8), pchanged_5759, *[], **kwargs_5760)
        
        
        # Assigning a Name to a Attribute (line 890):
        
        # Assigning a Name to a Attribute (line 890):
        # Getting the type of 'True' (line 890)
        True_5762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 21), 'True')
        # Getting the type of 'self' (line 890)
        self_5763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 890)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 890, 8), self_5763, 'stale', True_5762)
        
        # ################# End of 'set_zorder(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_zorder' in the type store
        # Getting the type of 'stypy_return_type' (line 881)
        stypy_return_type_5764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5764)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_zorder'
        return stypy_return_type_5764


    @norecursion
    def sticky_edges(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sticky_edges'
        module_type_store = module_type_store.open_function_context('sticky_edges', 892, 4, False)
        # Assigning a type to the variable 'self' (line 893)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.sticky_edges.__dict__.__setitem__('stypy_localization', localization)
        Artist.sticky_edges.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.sticky_edges.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.sticky_edges.__dict__.__setitem__('stypy_function_name', 'Artist.sticky_edges')
        Artist.sticky_edges.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.sticky_edges.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.sticky_edges.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.sticky_edges.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.sticky_edges.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.sticky_edges.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.sticky_edges.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.sticky_edges', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sticky_edges', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sticky_edges(...)' code ##################

        unicode_5765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, (-1)), 'unicode', u'\n        `x` and `y` sticky edge lists.\n\n        When performing autoscaling, if a data limit coincides with a value in\n        the corresponding sticky_edges list, then no margin will be added--the\n        view limit "sticks" to the edge. A typical usecase is histograms,\n        where one usually expects no margin on the bottom edge (0) of the\n        histogram.\n\n        This attribute cannot be assigned to; however, the `x` and `y` lists\n        can be modified in place as needed.\n\n        Examples\n        --------\n\n        >>> artist.sticky_edges.x[:] = (xmin, xmax)\n        >>> artist.sticky_edges.y[:] = (ymin, ymax)\n\n        ')
        # Getting the type of 'self' (line 913)
        self_5766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 15), 'self')
        # Obtaining the member '_sticky_edges' of a type (line 913)
        _sticky_edges_5767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 15), self_5766, '_sticky_edges')
        # Assigning a type to the variable 'stypy_return_type' (line 913)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 8), 'stypy_return_type', _sticky_edges_5767)
        
        # ################# End of 'sticky_edges(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sticky_edges' in the type store
        # Getting the type of 'stypy_return_type' (line 892)
        stypy_return_type_5768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5768)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sticky_edges'
        return stypy_return_type_5768


    @norecursion
    def update_from(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_from'
        module_type_store = module_type_store.open_function_context('update_from', 915, 4, False)
        # Assigning a type to the variable 'self' (line 916)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.update_from.__dict__.__setitem__('stypy_localization', localization)
        Artist.update_from.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.update_from.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.update_from.__dict__.__setitem__('stypy_function_name', 'Artist.update_from')
        Artist.update_from.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Artist.update_from.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.update_from.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.update_from.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.update_from.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.update_from.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.update_from.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.update_from', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_from', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_from(...)' code ##################

        unicode_5769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 8), 'unicode', u'Copy properties from *other* to *self*.')
        
        # Assigning a Attribute to a Attribute (line 917):
        
        # Assigning a Attribute to a Attribute (line 917):
        # Getting the type of 'other' (line 917)
        other_5770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 26), 'other')
        # Obtaining the member '_transform' of a type (line 917)
        _transform_5771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 26), other_5770, '_transform')
        # Getting the type of 'self' (line 917)
        self_5772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 8), 'self')
        # Setting the type of the member '_transform' of a type (line 917)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 8), self_5772, '_transform', _transform_5771)
        
        # Assigning a Attribute to a Attribute (line 918):
        
        # Assigning a Attribute to a Attribute (line 918):
        # Getting the type of 'other' (line 918)
        other_5773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 29), 'other')
        # Obtaining the member '_transformSet' of a type (line 918)
        _transformSet_5774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 29), other_5773, '_transformSet')
        # Getting the type of 'self' (line 918)
        self_5775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 8), 'self')
        # Setting the type of the member '_transformSet' of a type (line 918)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 8), self_5775, '_transformSet', _transformSet_5774)
        
        # Assigning a Attribute to a Attribute (line 919):
        
        # Assigning a Attribute to a Attribute (line 919):
        # Getting the type of 'other' (line 919)
        other_5776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 24), 'other')
        # Obtaining the member '_visible' of a type (line 919)
        _visible_5777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 24), other_5776, '_visible')
        # Getting the type of 'self' (line 919)
        self_5778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 8), 'self')
        # Setting the type of the member '_visible' of a type (line 919)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 8), self_5778, '_visible', _visible_5777)
        
        # Assigning a Attribute to a Attribute (line 920):
        
        # Assigning a Attribute to a Attribute (line 920):
        # Getting the type of 'other' (line 920)
        other_5779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 22), 'other')
        # Obtaining the member '_alpha' of a type (line 920)
        _alpha_5780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 22), other_5779, '_alpha')
        # Getting the type of 'self' (line 920)
        self_5781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 8), 'self')
        # Setting the type of the member '_alpha' of a type (line 920)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 8), self_5781, '_alpha', _alpha_5780)
        
        # Assigning a Attribute to a Attribute (line 921):
        
        # Assigning a Attribute to a Attribute (line 921):
        # Getting the type of 'other' (line 921)
        other_5782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 23), 'other')
        # Obtaining the member 'clipbox' of a type (line 921)
        clipbox_5783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 23), other_5782, 'clipbox')
        # Getting the type of 'self' (line 921)
        self_5784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 8), 'self')
        # Setting the type of the member 'clipbox' of a type (line 921)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 8), self_5784, 'clipbox', clipbox_5783)
        
        # Assigning a Attribute to a Attribute (line 922):
        
        # Assigning a Attribute to a Attribute (line 922):
        # Getting the type of 'other' (line 922)
        other_5785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 23), 'other')
        # Obtaining the member '_clipon' of a type (line 922)
        _clipon_5786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 23), other_5785, '_clipon')
        # Getting the type of 'self' (line 922)
        self_5787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'self')
        # Setting the type of the member '_clipon' of a type (line 922)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 8), self_5787, '_clipon', _clipon_5786)
        
        # Assigning a Attribute to a Attribute (line 923):
        
        # Assigning a Attribute to a Attribute (line 923):
        # Getting the type of 'other' (line 923)
        other_5788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 25), 'other')
        # Obtaining the member '_clippath' of a type (line 923)
        _clippath_5789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 25), other_5788, '_clippath')
        # Getting the type of 'self' (line 923)
        self_5790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 8), 'self')
        # Setting the type of the member '_clippath' of a type (line 923)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 8), self_5790, '_clippath', _clippath_5789)
        
        # Assigning a Attribute to a Attribute (line 924):
        
        # Assigning a Attribute to a Attribute (line 924):
        # Getting the type of 'other' (line 924)
        other_5791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 22), 'other')
        # Obtaining the member '_label' of a type (line 924)
        _label_5792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 22), other_5791, '_label')
        # Getting the type of 'self' (line 924)
        self_5793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'self')
        # Setting the type of the member '_label' of a type (line 924)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 8), self_5793, '_label', _label_5792)
        
        # Assigning a Attribute to a Attribute (line 925):
        
        # Assigning a Attribute to a Attribute (line 925):
        # Getting the type of 'other' (line 925)
        other_5794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 23), 'other')
        # Obtaining the member '_sketch' of a type (line 925)
        _sketch_5795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 23), other_5794, '_sketch')
        # Getting the type of 'self' (line 925)
        self_5796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 8), 'self')
        # Setting the type of the member '_sketch' of a type (line 925)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 8), self_5796, '_sketch', _sketch_5795)
        
        # Assigning a Attribute to a Attribute (line 926):
        
        # Assigning a Attribute to a Attribute (line 926):
        # Getting the type of 'other' (line 926)
        other_5797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 29), 'other')
        # Obtaining the member '_path_effects' of a type (line 926)
        _path_effects_5798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 29), other_5797, '_path_effects')
        # Getting the type of 'self' (line 926)
        self_5799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 8), 'self')
        # Setting the type of the member '_path_effects' of a type (line 926)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 8), self_5799, '_path_effects', _path_effects_5798)
        
        # Assigning a Subscript to a Subscript (line 927):
        
        # Assigning a Subscript to a Subscript (line 927):
        
        # Obtaining the type of the subscript
        slice_5800 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 927, 33), None, None, None)
        # Getting the type of 'other' (line 927)
        other_5801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 33), 'other')
        # Obtaining the member 'sticky_edges' of a type (line 927)
        sticky_edges_5802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 33), other_5801, 'sticky_edges')
        # Obtaining the member 'x' of a type (line 927)
        x_5803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 33), sticky_edges_5802, 'x')
        # Obtaining the member '__getitem__' of a type (line 927)
        getitem___5804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 33), x_5803, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 927)
        subscript_call_result_5805 = invoke(stypy.reporting.localization.Localization(__file__, 927, 33), getitem___5804, slice_5800)
        
        # Getting the type of 'self' (line 927)
        self_5806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 8), 'self')
        # Obtaining the member 'sticky_edges' of a type (line 927)
        sticky_edges_5807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 8), self_5806, 'sticky_edges')
        # Obtaining the member 'x' of a type (line 927)
        x_5808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 8), sticky_edges_5807, 'x')
        slice_5809 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 927, 8), None, None, None)
        # Storing an element on a container (line 927)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 927, 8), x_5808, (slice_5809, subscript_call_result_5805))
        
        # Assigning a Subscript to a Subscript (line 928):
        
        # Assigning a Subscript to a Subscript (line 928):
        
        # Obtaining the type of the subscript
        slice_5810 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 928, 33), None, None, None)
        # Getting the type of 'other' (line 928)
        other_5811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 33), 'other')
        # Obtaining the member 'sticky_edges' of a type (line 928)
        sticky_edges_5812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 33), other_5811, 'sticky_edges')
        # Obtaining the member 'y' of a type (line 928)
        y_5813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 33), sticky_edges_5812, 'y')
        # Obtaining the member '__getitem__' of a type (line 928)
        getitem___5814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 33), y_5813, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 928)
        subscript_call_result_5815 = invoke(stypy.reporting.localization.Localization(__file__, 928, 33), getitem___5814, slice_5810)
        
        # Getting the type of 'self' (line 928)
        self_5816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 8), 'self')
        # Obtaining the member 'sticky_edges' of a type (line 928)
        sticky_edges_5817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 8), self_5816, 'sticky_edges')
        # Obtaining the member 'y' of a type (line 928)
        y_5818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 8), sticky_edges_5817, 'y')
        slice_5819 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 928, 8), None, None, None)
        # Storing an element on a container (line 928)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 928, 8), y_5818, (slice_5819, subscript_call_result_5815))
        
        # Call to pchanged(...): (line 929)
        # Processing the call keyword arguments (line 929)
        kwargs_5822 = {}
        # Getting the type of 'self' (line 929)
        self_5820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 8), 'self', False)
        # Obtaining the member 'pchanged' of a type (line 929)
        pchanged_5821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 8), self_5820, 'pchanged')
        # Calling pchanged(args, kwargs) (line 929)
        pchanged_call_result_5823 = invoke(stypy.reporting.localization.Localization(__file__, 929, 8), pchanged_5821, *[], **kwargs_5822)
        
        
        # Assigning a Name to a Attribute (line 930):
        
        # Assigning a Name to a Attribute (line 930):
        # Getting the type of 'True' (line 930)
        True_5824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 21), 'True')
        # Getting the type of 'self' (line 930)
        self_5825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 930)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 8), self_5825, 'stale', True_5824)
        
        # ################# End of 'update_from(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_from' in the type store
        # Getting the type of 'stypy_return_type' (line 915)
        stypy_return_type_5826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5826)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_from'
        return stypy_return_type_5826


    @norecursion
    def properties(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'properties'
        module_type_store = module_type_store.open_function_context('properties', 932, 4, False)
        # Assigning a type to the variable 'self' (line 933)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.properties.__dict__.__setitem__('stypy_localization', localization)
        Artist.properties.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.properties.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.properties.__dict__.__setitem__('stypy_function_name', 'Artist.properties')
        Artist.properties.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.properties.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.properties.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.properties.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.properties.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.properties.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.properties.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.properties', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'properties', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'properties(...)' code ##################

        unicode_5827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, (-1)), 'unicode', u'\n        return a dictionary mapping property name -> value for all Artist props\n        ')
        
        # Call to properties(...): (line 936)
        # Processing the call keyword arguments (line 936)
        kwargs_5833 = {}
        
        # Call to ArtistInspector(...): (line 936)
        # Processing the call arguments (line 936)
        # Getting the type of 'self' (line 936)
        self_5829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 31), 'self', False)
        # Processing the call keyword arguments (line 936)
        kwargs_5830 = {}
        # Getting the type of 'ArtistInspector' (line 936)
        ArtistInspector_5828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 15), 'ArtistInspector', False)
        # Calling ArtistInspector(args, kwargs) (line 936)
        ArtistInspector_call_result_5831 = invoke(stypy.reporting.localization.Localization(__file__, 936, 15), ArtistInspector_5828, *[self_5829], **kwargs_5830)
        
        # Obtaining the member 'properties' of a type (line 936)
        properties_5832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 936, 15), ArtistInspector_call_result_5831, 'properties')
        # Calling properties(args, kwargs) (line 936)
        properties_call_result_5834 = invoke(stypy.reporting.localization.Localization(__file__, 936, 15), properties_5832, *[], **kwargs_5833)
        
        # Assigning a type to the variable 'stypy_return_type' (line 936)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 8), 'stypy_return_type', properties_call_result_5834)
        
        # ################# End of 'properties(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'properties' in the type store
        # Getting the type of 'stypy_return_type' (line 932)
        stypy_return_type_5835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5835)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'properties'
        return stypy_return_type_5835


    @norecursion
    def set(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set'
        module_type_store = module_type_store.open_function_context('set', 938, 4, False)
        # Assigning a type to the variable 'self' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.set.__dict__.__setitem__('stypy_localization', localization)
        Artist.set.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.set.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.set.__dict__.__setitem__('stypy_function_name', 'Artist.set')
        Artist.set.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.set.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.set.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Artist.set.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.set.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.set.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.set.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.set', [], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set(...)' code ##################

        unicode_5836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, (-1)), 'unicode', u'A property batch setter. Pass *kwargs* to set properties.\n        ')
        
        # Assigning a Call to a Name (line 941):
        
        # Assigning a Call to a Name (line 941):
        
        # Call to OrderedDict(...): (line 941)
        # Processing the call arguments (line 941)
        
        # Call to sorted(...): (line 942)
        # Processing the call arguments (line 942)
        
        # Call to items(...): (line 942)
        # Processing the call keyword arguments (line 942)
        kwargs_5841 = {}
        # Getting the type of 'kwargs' (line 942)
        kwargs_5839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 19), 'kwargs', False)
        # Obtaining the member 'items' of a type (line 942)
        items_5840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 19), kwargs_5839, 'items')
        # Calling items(args, kwargs) (line 942)
        items_call_result_5842 = invoke(stypy.reporting.localization.Localization(__file__, 942, 19), items_5840, *[], **kwargs_5841)
        
        # Processing the call keyword arguments (line 942)
        # Getting the type of 'True' (line 942)
        True_5843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 43), 'True', False)
        keyword_5844 = True_5843

        @norecursion
        def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_3'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 943, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_3.stypy_localization = localization
            _stypy_temp_lambda_3.stypy_type_of_self = None
            _stypy_temp_lambda_3.stypy_type_store = module_type_store
            _stypy_temp_lambda_3.stypy_function_name = '_stypy_temp_lambda_3'
            _stypy_temp_lambda_3.stypy_param_names_list = ['x']
            _stypy_temp_lambda_3.stypy_varargs_param_name = None
            _stypy_temp_lambda_3.stypy_kwargs_param_name = None
            _stypy_temp_lambda_3.stypy_call_defaults = defaults
            _stypy_temp_lambda_3.stypy_call_varargs = varargs
            _stypy_temp_lambda_3.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_3', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_3', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining an instance of the builtin type 'tuple' (line 943)
            tuple_5845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 34), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 943)
            # Adding element type (line 943)
            
            # Call to get(...): (line 943)
            # Processing the call arguments (line 943)
            
            # Obtaining the type of the subscript
            int_5849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 57), 'int')
            # Getting the type of 'x' (line 943)
            x_5850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 55), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 943)
            getitem___5851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 943, 55), x_5850, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 943)
            subscript_call_result_5852 = invoke(stypy.reporting.localization.Localization(__file__, 943, 55), getitem___5851, int_5849)
            
            int_5853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 61), 'int')
            # Processing the call keyword arguments (line 943)
            kwargs_5854 = {}
            # Getting the type of 'self' (line 943)
            self_5846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 34), 'self', False)
            # Obtaining the member '_prop_order' of a type (line 943)
            _prop_order_5847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 943, 34), self_5846, '_prop_order')
            # Obtaining the member 'get' of a type (line 943)
            get_5848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 943, 34), _prop_order_5847, 'get')
            # Calling get(args, kwargs) (line 943)
            get_call_result_5855 = invoke(stypy.reporting.localization.Localization(__file__, 943, 34), get_5848, *[subscript_call_result_5852, int_5853], **kwargs_5854)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 943, 34), tuple_5845, get_call_result_5855)
            # Adding element type (line 943)
            
            # Obtaining the type of the subscript
            int_5856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 67), 'int')
            # Getting the type of 'x' (line 943)
            x_5857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 65), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 943)
            getitem___5858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 943, 65), x_5857, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 943)
            subscript_call_result_5859 = invoke(stypy.reporting.localization.Localization(__file__, 943, 65), getitem___5858, int_5856)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 943, 34), tuple_5845, subscript_call_result_5859)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 943)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 943, 23), 'stypy_return_type', tuple_5845)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_3' in the type store
            # Getting the type of 'stypy_return_type' (line 943)
            stypy_return_type_5860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5860)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_3'
            return stypy_return_type_5860

        # Assigning a type to the variable '_stypy_temp_lambda_3' (line 943)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 943, 23), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
        # Getting the type of '_stypy_temp_lambda_3' (line 943)
        _stypy_temp_lambda_3_5861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 23), '_stypy_temp_lambda_3')
        keyword_5862 = _stypy_temp_lambda_3_5861
        kwargs_5863 = {'reverse': keyword_5844, 'key': keyword_5862}
        # Getting the type of 'sorted' (line 942)
        sorted_5838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 12), 'sorted', False)
        # Calling sorted(args, kwargs) (line 942)
        sorted_call_result_5864 = invoke(stypy.reporting.localization.Localization(__file__, 942, 12), sorted_5838, *[items_call_result_5842], **kwargs_5863)
        
        # Processing the call keyword arguments (line 941)
        kwargs_5865 = {}
        # Getting the type of 'OrderedDict' (line 941)
        OrderedDict_5837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 16), 'OrderedDict', False)
        # Calling OrderedDict(args, kwargs) (line 941)
        OrderedDict_call_result_5866 = invoke(stypy.reporting.localization.Localization(__file__, 941, 16), OrderedDict_5837, *[sorted_call_result_5864], **kwargs_5865)
        
        # Assigning a type to the variable 'props' (line 941)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 8), 'props', OrderedDict_call_result_5866)
        
        # Call to update(...): (line 945)
        # Processing the call arguments (line 945)
        # Getting the type of 'props' (line 945)
        props_5869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 27), 'props', False)
        # Processing the call keyword arguments (line 945)
        kwargs_5870 = {}
        # Getting the type of 'self' (line 945)
        self_5867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 15), 'self', False)
        # Obtaining the member 'update' of a type (line 945)
        update_5868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 945, 15), self_5867, 'update')
        # Calling update(args, kwargs) (line 945)
        update_call_result_5871 = invoke(stypy.reporting.localization.Localization(__file__, 945, 15), update_5868, *[props_5869], **kwargs_5870)
        
        # Assigning a type to the variable 'stypy_return_type' (line 945)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 8), 'stypy_return_type', update_call_result_5871)
        
        # ################# End of 'set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set' in the type store
        # Getting the type of 'stypy_return_type' (line 938)
        stypy_return_type_5872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5872)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set'
        return stypy_return_type_5872


    @norecursion
    def findobj(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 947)
        None_5873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 28), 'None')
        # Getting the type of 'True' (line 947)
        True_5874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 47), 'True')
        defaults = [None_5873, True_5874]
        # Create a new context for function 'findobj'
        module_type_store = module_type_store.open_function_context('findobj', 947, 4, False)
        # Assigning a type to the variable 'self' (line 948)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.findobj.__dict__.__setitem__('stypy_localization', localization)
        Artist.findobj.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.findobj.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.findobj.__dict__.__setitem__('stypy_function_name', 'Artist.findobj')
        Artist.findobj.__dict__.__setitem__('stypy_param_names_list', ['match', 'include_self'])
        Artist.findobj.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.findobj.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.findobj.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.findobj.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.findobj.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.findobj.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.findobj', ['match', 'include_self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'findobj', localization, ['match', 'include_self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'findobj(...)' code ##################

        unicode_5875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, (-1)), 'unicode', u'\n        Find artist objects.\n\n        Recursively find all :class:`~matplotlib.artist.Artist` instances\n        contained in self.\n\n        *match* can be\n\n          - None: return all objects contained in artist.\n\n          - function with signature ``boolean = match(artist)``\n            used to filter matches\n\n          - class instance: e.g., Line2D.  Only return artists of class type.\n\n        If *include_self* is True (default), include self in the list to be\n        checked for a match.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 967)
        # Getting the type of 'match' (line 967)
        match_5876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 11), 'match')
        # Getting the type of 'None' (line 967)
        None_5877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 20), 'None')
        
        (may_be_5878, more_types_in_union_5879) = may_be_none(match_5876, None_5877)

        if may_be_5878:

            if more_types_in_union_5879:
                # Runtime conditional SSA (line 967)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store


            @norecursion
            def matchfunc(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'matchfunc'
                module_type_store = module_type_store.open_function_context('matchfunc', 968, 12, False)
                
                # Passed parameters checking function
                matchfunc.stypy_localization = localization
                matchfunc.stypy_type_of_self = None
                matchfunc.stypy_type_store = module_type_store
                matchfunc.stypy_function_name = 'matchfunc'
                matchfunc.stypy_param_names_list = ['x']
                matchfunc.stypy_varargs_param_name = None
                matchfunc.stypy_kwargs_param_name = None
                matchfunc.stypy_call_defaults = defaults
                matchfunc.stypy_call_varargs = varargs
                matchfunc.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'matchfunc', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'matchfunc', localization, ['x'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'matchfunc(...)' code ##################

                # Getting the type of 'True' (line 969)
                True_5880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 969)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 16), 'stypy_return_type', True_5880)
                
                # ################# End of 'matchfunc(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'matchfunc' in the type store
                # Getting the type of 'stypy_return_type' (line 968)
                stypy_return_type_5881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_5881)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'matchfunc'
                return stypy_return_type_5881

            # Assigning a type to the variable 'matchfunc' (line 968)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 968, 12), 'matchfunc', matchfunc)

            if more_types_in_union_5879:
                # Runtime conditional SSA for else branch (line 967)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_5878) or more_types_in_union_5879):
            
            
            # Evaluating a boolean operation
            
            # Call to isinstance(...): (line 970)
            # Processing the call arguments (line 970)
            # Getting the type of 'match' (line 970)
            match_5883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 24), 'match', False)
            # Getting the type of 'type' (line 970)
            type_5884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 31), 'type', False)
            # Processing the call keyword arguments (line 970)
            kwargs_5885 = {}
            # Getting the type of 'isinstance' (line 970)
            isinstance_5882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 970)
            isinstance_call_result_5886 = invoke(stypy.reporting.localization.Localization(__file__, 970, 13), isinstance_5882, *[match_5883, type_5884], **kwargs_5885)
            
            
            # Call to issubclass(...): (line 970)
            # Processing the call arguments (line 970)
            # Getting the type of 'match' (line 970)
            match_5888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 52), 'match', False)
            # Getting the type of 'Artist' (line 970)
            Artist_5889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 59), 'Artist', False)
            # Processing the call keyword arguments (line 970)
            kwargs_5890 = {}
            # Getting the type of 'issubclass' (line 970)
            issubclass_5887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 41), 'issubclass', False)
            # Calling issubclass(args, kwargs) (line 970)
            issubclass_call_result_5891 = invoke(stypy.reporting.localization.Localization(__file__, 970, 41), issubclass_5887, *[match_5888, Artist_5889], **kwargs_5890)
            
            # Applying the binary operator 'and' (line 970)
            result_and_keyword_5892 = python_operator(stypy.reporting.localization.Localization(__file__, 970, 13), 'and', isinstance_call_result_5886, issubclass_call_result_5891)
            
            # Testing the type of an if condition (line 970)
            if_condition_5893 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 970, 13), result_and_keyword_5892)
            # Assigning a type to the variable 'if_condition_5893' (line 970)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 13), 'if_condition_5893', if_condition_5893)
            # SSA begins for if statement (line 970)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

            @norecursion
            def matchfunc(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'matchfunc'
                module_type_store = module_type_store.open_function_context('matchfunc', 971, 12, False)
                
                # Passed parameters checking function
                matchfunc.stypy_localization = localization
                matchfunc.stypy_type_of_self = None
                matchfunc.stypy_type_store = module_type_store
                matchfunc.stypy_function_name = 'matchfunc'
                matchfunc.stypy_param_names_list = ['x']
                matchfunc.stypy_varargs_param_name = None
                matchfunc.stypy_kwargs_param_name = None
                matchfunc.stypy_call_defaults = defaults
                matchfunc.stypy_call_varargs = varargs
                matchfunc.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'matchfunc', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'matchfunc', localization, ['x'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'matchfunc(...)' code ##################

                
                # Call to isinstance(...): (line 972)
                # Processing the call arguments (line 972)
                # Getting the type of 'x' (line 972)
                x_5895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 34), 'x', False)
                # Getting the type of 'match' (line 972)
                match_5896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 37), 'match', False)
                # Processing the call keyword arguments (line 972)
                kwargs_5897 = {}
                # Getting the type of 'isinstance' (line 972)
                isinstance_5894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 23), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 972)
                isinstance_call_result_5898 = invoke(stypy.reporting.localization.Localization(__file__, 972, 23), isinstance_5894, *[x_5895, match_5896], **kwargs_5897)
                
                # Assigning a type to the variable 'stypy_return_type' (line 972)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 16), 'stypy_return_type', isinstance_call_result_5898)
                
                # ################# End of 'matchfunc(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'matchfunc' in the type store
                # Getting the type of 'stypy_return_type' (line 971)
                stypy_return_type_5899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_5899)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'matchfunc'
                return stypy_return_type_5899

            # Assigning a type to the variable 'matchfunc' (line 971)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 12), 'matchfunc', matchfunc)
            # SSA branch for the else part of an if statement (line 970)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to callable(...): (line 973)
            # Processing the call arguments (line 973)
            # Getting the type of 'match' (line 973)
            match_5901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 22), 'match', False)
            # Processing the call keyword arguments (line 973)
            kwargs_5902 = {}
            # Getting the type of 'callable' (line 973)
            callable_5900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 13), 'callable', False)
            # Calling callable(args, kwargs) (line 973)
            callable_call_result_5903 = invoke(stypy.reporting.localization.Localization(__file__, 973, 13), callable_5900, *[match_5901], **kwargs_5902)
            
            # Testing the type of an if condition (line 973)
            if_condition_5904 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 973, 13), callable_call_result_5903)
            # Assigning a type to the variable 'if_condition_5904' (line 973)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 973, 13), 'if_condition_5904', if_condition_5904)
            # SSA begins for if statement (line 973)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 974):
            
            # Assigning a Name to a Name (line 974):
            # Getting the type of 'match' (line 974)
            match_5905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 24), 'match')
            # Assigning a type to the variable 'matchfunc' (line 974)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 12), 'matchfunc', match_5905)
            # SSA branch for the else part of an if statement (line 973)
            module_type_store.open_ssa_branch('else')
            
            # Call to ValueError(...): (line 976)
            # Processing the call arguments (line 976)
            unicode_5907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 29), 'unicode', u'match must be None, a matplotlib.artist.Artist subclass, or a callable')
            # Processing the call keyword arguments (line 976)
            kwargs_5908 = {}
            # Getting the type of 'ValueError' (line 976)
            ValueError_5906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 976)
            ValueError_call_result_5909 = invoke(stypy.reporting.localization.Localization(__file__, 976, 18), ValueError_5906, *[unicode_5907], **kwargs_5908)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 976, 12), ValueError_call_result_5909, 'raise parameter', BaseException)
            # SSA join for if statement (line 973)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 970)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_5878 and more_types_in_union_5879):
                # SSA join for if statement (line 967)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 979):
        
        # Assigning a Call to a Name (line 979):
        
        # Call to sum(...): (line 979)
        # Processing the call arguments (line 979)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to get_children(...): (line 979)
        # Processing the call keyword arguments (line 979)
        kwargs_5918 = {}
        # Getting the type of 'self' (line 979)
        self_5916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 53), 'self', False)
        # Obtaining the member 'get_children' of a type (line 979)
        get_children_5917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 53), self_5916, 'get_children')
        # Calling get_children(args, kwargs) (line 979)
        get_children_call_result_5919 = invoke(stypy.reporting.localization.Localization(__file__, 979, 53), get_children_5917, *[], **kwargs_5918)
        
        comprehension_5920 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 979, 23), get_children_call_result_5919)
        # Assigning a type to the variable 'c' (line 979)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 23), 'c', comprehension_5920)
        
        # Call to findobj(...): (line 979)
        # Processing the call arguments (line 979)
        # Getting the type of 'matchfunc' (line 979)
        matchfunc_5913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 33), 'matchfunc', False)
        # Processing the call keyword arguments (line 979)
        kwargs_5914 = {}
        # Getting the type of 'c' (line 979)
        c_5911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 23), 'c', False)
        # Obtaining the member 'findobj' of a type (line 979)
        findobj_5912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 23), c_5911, 'findobj')
        # Calling findobj(args, kwargs) (line 979)
        findobj_call_result_5915 = invoke(stypy.reporting.localization.Localization(__file__, 979, 23), findobj_5912, *[matchfunc_5913], **kwargs_5914)
        
        list_5921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 979, 23), list_5921, findobj_call_result_5915)
        
        # Obtaining an instance of the builtin type 'list' (line 979)
        list_5922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 75), 'list')
        # Adding type elements to the builtin type 'list' instance (line 979)
        
        # Processing the call keyword arguments (line 979)
        kwargs_5923 = {}
        # Getting the type of 'sum' (line 979)
        sum_5910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 18), 'sum', False)
        # Calling sum(args, kwargs) (line 979)
        sum_call_result_5924 = invoke(stypy.reporting.localization.Localization(__file__, 979, 18), sum_5910, *[list_5921, list_5922], **kwargs_5923)
        
        # Assigning a type to the variable 'artists' (line 979)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 8), 'artists', sum_call_result_5924)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'include_self' (line 980)
        include_self_5925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 11), 'include_self')
        
        # Call to matchfunc(...): (line 980)
        # Processing the call arguments (line 980)
        # Getting the type of 'self' (line 980)
        self_5927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 38), 'self', False)
        # Processing the call keyword arguments (line 980)
        kwargs_5928 = {}
        # Getting the type of 'matchfunc' (line 980)
        matchfunc_5926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 28), 'matchfunc', False)
        # Calling matchfunc(args, kwargs) (line 980)
        matchfunc_call_result_5929 = invoke(stypy.reporting.localization.Localization(__file__, 980, 28), matchfunc_5926, *[self_5927], **kwargs_5928)
        
        # Applying the binary operator 'and' (line 980)
        result_and_keyword_5930 = python_operator(stypy.reporting.localization.Localization(__file__, 980, 11), 'and', include_self_5925, matchfunc_call_result_5929)
        
        # Testing the type of an if condition (line 980)
        if_condition_5931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 980, 8), result_and_keyword_5930)
        # Assigning a type to the variable 'if_condition_5931' (line 980)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 8), 'if_condition_5931', if_condition_5931)
        # SSA begins for if statement (line 980)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 981)
        # Processing the call arguments (line 981)
        # Getting the type of 'self' (line 981)
        self_5934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 27), 'self', False)
        # Processing the call keyword arguments (line 981)
        kwargs_5935 = {}
        # Getting the type of 'artists' (line 981)
        artists_5932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 12), 'artists', False)
        # Obtaining the member 'append' of a type (line 981)
        append_5933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 12), artists_5932, 'append')
        # Calling append(args, kwargs) (line 981)
        append_call_result_5936 = invoke(stypy.reporting.localization.Localization(__file__, 981, 12), append_5933, *[self_5934], **kwargs_5935)
        
        # SSA join for if statement (line 980)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'artists' (line 982)
        artists_5937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 15), 'artists')
        # Assigning a type to the variable 'stypy_return_type' (line 982)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 8), 'stypy_return_type', artists_5937)
        
        # ################# End of 'findobj(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'findobj' in the type store
        # Getting the type of 'stypy_return_type' (line 947)
        stypy_return_type_5938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'findobj'
        return stypy_return_type_5938


    @norecursion
    def get_cursor_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_cursor_data'
        module_type_store = module_type_store.open_function_context('get_cursor_data', 984, 4, False)
        # Assigning a type to the variable 'self' (line 985)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.get_cursor_data.__dict__.__setitem__('stypy_localization', localization)
        Artist.get_cursor_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.get_cursor_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.get_cursor_data.__dict__.__setitem__('stypy_function_name', 'Artist.get_cursor_data')
        Artist.get_cursor_data.__dict__.__setitem__('stypy_param_names_list', ['event'])
        Artist.get_cursor_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.get_cursor_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.get_cursor_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.get_cursor_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.get_cursor_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.get_cursor_data.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.get_cursor_data', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_cursor_data', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_cursor_data(...)' code ##################

        unicode_5939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 987, (-1)), 'unicode', u'\n        Get the cursor data for a given event.\n        ')
        # Getting the type of 'None' (line 988)
        None_5940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 988)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 8), 'stypy_return_type', None_5940)
        
        # ################# End of 'get_cursor_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_cursor_data' in the type store
        # Getting the type of 'stypy_return_type' (line 984)
        stypy_return_type_5941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5941)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_cursor_data'
        return stypy_return_type_5941


    @norecursion
    def format_cursor_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'format_cursor_data'
        module_type_store = module_type_store.open_function_context('format_cursor_data', 990, 4, False)
        # Assigning a type to the variable 'self' (line 991)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 991, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.format_cursor_data.__dict__.__setitem__('stypy_localization', localization)
        Artist.format_cursor_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.format_cursor_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.format_cursor_data.__dict__.__setitem__('stypy_function_name', 'Artist.format_cursor_data')
        Artist.format_cursor_data.__dict__.__setitem__('stypy_param_names_list', ['data'])
        Artist.format_cursor_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.format_cursor_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.format_cursor_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.format_cursor_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.format_cursor_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.format_cursor_data.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.format_cursor_data', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'format_cursor_data', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'format_cursor_data(...)' code ##################

        unicode_5942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, (-1)), 'unicode', u'\n        Return *cursor data* string formatted.\n        ')
        
        
        # SSA begins for try-except statement (line 994)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        int_5943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 17), 'int')
        # Getting the type of 'data' (line 995)
        data_5944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 12), 'data')
        # Obtaining the member '__getitem__' of a type (line 995)
        getitem___5945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 12), data_5944, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 995)
        subscript_call_result_5946 = invoke(stypy.reporting.localization.Localization(__file__, 995, 12), getitem___5945, int_5943)
        
        # SSA branch for the except part of a try statement (line 994)
        # SSA branch for the except 'Tuple' branch of a try statement (line 994)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a List to a Name (line 997):
        
        # Assigning a List to a Name (line 997):
        
        # Obtaining an instance of the builtin type 'list' (line 997)
        list_5947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 997, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 997)
        # Adding element type (line 997)
        # Getting the type of 'data' (line 997)
        data_5948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 20), 'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 997, 19), list_5947, data_5948)
        
        # Assigning a type to the variable 'data' (line 997)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 12), 'data', list_5947)
        # SSA join for try-except statement (line 994)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to join(...): (line 998)
        # Processing the call arguments (line 998)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 998, 25, True)
        # Calculating comprehension expression
        # Getting the type of 'data' (line 998)
        data_5967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 60), 'data', False)
        comprehension_5968 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 998, 25), data_5967)
        # Assigning a type to the variable 'item' (line 998)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 998, 25), 'item', comprehension_5968)
        
        # Call to isinstance(...): (line 999)
        # Processing the call arguments (line 999)
        # Getting the type of 'item' (line 999)
        item_5957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 27), 'item', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 999)
        tuple_5958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 999, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 999)
        # Adding element type (line 999)
        # Getting the type of 'np' (line 999)
        np_5959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 34), 'np', False)
        # Obtaining the member 'floating' of a type (line 999)
        floating_5960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 999, 34), np_5959, 'floating')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 34), tuple_5958, floating_5960)
        # Adding element type (line 999)
        # Getting the type of 'np' (line 999)
        np_5961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 47), 'np', False)
        # Obtaining the member 'integer' of a type (line 999)
        integer_5962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 999, 47), np_5961, 'integer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 34), tuple_5958, integer_5962)
        # Adding element type (line 999)
        # Getting the type of 'int' (line 999)
        int_5963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 59), 'int', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 34), tuple_5958, int_5963)
        # Adding element type (line 999)
        # Getting the type of 'float' (line 999)
        float_5964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 64), 'float', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 34), tuple_5958, float_5964)
        
        # Processing the call keyword arguments (line 999)
        kwargs_5965 = {}
        # Getting the type of 'isinstance' (line 999)
        isinstance_5956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 999)
        isinstance_call_result_5966 = invoke(stypy.reporting.localization.Localization(__file__, 999, 16), isinstance_5956, *[item_5957, tuple_5958], **kwargs_5965)
        
        
        # Call to format(...): (line 998)
        # Processing the call arguments (line 998)
        # Getting the type of 'item' (line 998)
        item_5953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 42), 'item', False)
        # Processing the call keyword arguments (line 998)
        kwargs_5954 = {}
        unicode_5951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 998, 25), 'unicode', u'{:0.3g}')
        # Obtaining the member 'format' of a type (line 998)
        format_5952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 998, 25), unicode_5951, 'format')
        # Calling format(args, kwargs) (line 998)
        format_call_result_5955 = invoke(stypy.reporting.localization.Localization(__file__, 998, 25), format_5952, *[item_5953], **kwargs_5954)
        
        list_5969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 998, 25), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 998, 25), list_5969, format_call_result_5955)
        # Processing the call keyword arguments (line 998)
        kwargs_5970 = {}
        unicode_5949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 998, 15), 'unicode', u', ')
        # Obtaining the member 'join' of a type (line 998)
        join_5950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 998, 15), unicode_5949, 'join')
        # Calling join(args, kwargs) (line 998)
        join_call_result_5971 = invoke(stypy.reporting.localization.Localization(__file__, 998, 15), join_5950, *[list_5969], **kwargs_5970)
        
        # Assigning a type to the variable 'stypy_return_type' (line 998)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 998, 8), 'stypy_return_type', join_call_result_5971)
        
        # ################# End of 'format_cursor_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'format_cursor_data' in the type store
        # Getting the type of 'stypy_return_type' (line 990)
        stypy_return_type_5972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5972)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'format_cursor_data'
        return stypy_return_type_5972


    @norecursion
    def mouseover(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mouseover'
        module_type_store = module_type_store.open_function_context('mouseover', 1001, 4, False)
        # Assigning a type to the variable 'self' (line 1002)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1002, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.mouseover.__dict__.__setitem__('stypy_localization', localization)
        Artist.mouseover.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.mouseover.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.mouseover.__dict__.__setitem__('stypy_function_name', 'Artist.mouseover')
        Artist.mouseover.__dict__.__setitem__('stypy_param_names_list', [])
        Artist.mouseover.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.mouseover.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.mouseover.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.mouseover.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.mouseover.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.mouseover.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.mouseover', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mouseover', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mouseover(...)' code ##################

        # Getting the type of 'self' (line 1003)
        self_5973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 15), 'self')
        # Obtaining the member '_mouseover' of a type (line 1003)
        _mouseover_5974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1003, 15), self_5973, '_mouseover')
        # Assigning a type to the variable 'stypy_return_type' (line 1003)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1003, 8), 'stypy_return_type', _mouseover_5974)
        
        # ################# End of 'mouseover(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mouseover' in the type store
        # Getting the type of 'stypy_return_type' (line 1001)
        stypy_return_type_5975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5975)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mouseover'
        return stypy_return_type_5975


    @norecursion
    def mouseover(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mouseover'
        module_type_store = module_type_store.open_function_context('mouseover', 1005, 4, False)
        # Assigning a type to the variable 'self' (line 1006)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1006, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Artist.mouseover.__dict__.__setitem__('stypy_localization', localization)
        Artist.mouseover.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Artist.mouseover.__dict__.__setitem__('stypy_type_store', module_type_store)
        Artist.mouseover.__dict__.__setitem__('stypy_function_name', 'Artist.mouseover')
        Artist.mouseover.__dict__.__setitem__('stypy_param_names_list', ['val'])
        Artist.mouseover.__dict__.__setitem__('stypy_varargs_param_name', None)
        Artist.mouseover.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Artist.mouseover.__dict__.__setitem__('stypy_call_defaults', defaults)
        Artist.mouseover.__dict__.__setitem__('stypy_call_varargs', varargs)
        Artist.mouseover.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Artist.mouseover.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Artist.mouseover', ['val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mouseover', localization, ['val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mouseover(...)' code ##################

        
        # Assigning a Call to a Name (line 1007):
        
        # Assigning a Call to a Name (line 1007):
        
        # Call to bool(...): (line 1007)
        # Processing the call arguments (line 1007)
        # Getting the type of 'val' (line 1007)
        val_5977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 19), 'val', False)
        # Processing the call keyword arguments (line 1007)
        kwargs_5978 = {}
        # Getting the type of 'bool' (line 1007)
        bool_5976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 14), 'bool', False)
        # Calling bool(args, kwargs) (line 1007)
        bool_call_result_5979 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 14), bool_5976, *[val_5977], **kwargs_5978)
        
        # Assigning a type to the variable 'val' (line 1007)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 8), 'val', bool_call_result_5979)
        
        # Assigning a Name to a Attribute (line 1008):
        
        # Assigning a Name to a Attribute (line 1008):
        # Getting the type of 'val' (line 1008)
        val_5980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 26), 'val')
        # Getting the type of 'self' (line 1008)
        self_5981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 8), 'self')
        # Setting the type of the member '_mouseover' of a type (line 1008)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1008, 8), self_5981, '_mouseover', val_5980)
        
        # Assigning a Attribute to a Name (line 1009):
        
        # Assigning a Attribute to a Name (line 1009):
        # Getting the type of 'self' (line 1009)
        self_5982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 13), 'self')
        # Obtaining the member 'axes' of a type (line 1009)
        axes_5983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1009, 13), self_5982, 'axes')
        # Assigning a type to the variable 'ax' (line 1009)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1009, 8), 'ax', axes_5983)
        
        # Getting the type of 'ax' (line 1010)
        ax_5984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 11), 'ax')
        # Testing the type of an if condition (line 1010)
        if_condition_5985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1010, 8), ax_5984)
        # Assigning a type to the variable 'if_condition_5985' (line 1010)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1010, 8), 'if_condition_5985', if_condition_5985)
        # SSA begins for if statement (line 1010)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'val' (line 1011)
        val_5986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 15), 'val')
        # Testing the type of an if condition (line 1011)
        if_condition_5987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1011, 12), val_5986)
        # Assigning a type to the variable 'if_condition_5987' (line 1011)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1011, 12), 'if_condition_5987', if_condition_5987)
        # SSA begins for if statement (line 1011)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add(...): (line 1012)
        # Processing the call arguments (line 1012)
        # Getting the type of 'self' (line 1012)
        self_5991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 37), 'self', False)
        # Processing the call keyword arguments (line 1012)
        kwargs_5992 = {}
        # Getting the type of 'ax' (line 1012)
        ax_5988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 16), 'ax', False)
        # Obtaining the member 'mouseover_set' of a type (line 1012)
        mouseover_set_5989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 16), ax_5988, 'mouseover_set')
        # Obtaining the member 'add' of a type (line 1012)
        add_5990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1012, 16), mouseover_set_5989, 'add')
        # Calling add(args, kwargs) (line 1012)
        add_call_result_5993 = invoke(stypy.reporting.localization.Localization(__file__, 1012, 16), add_5990, *[self_5991], **kwargs_5992)
        
        # SSA branch for the else part of an if statement (line 1011)
        module_type_store.open_ssa_branch('else')
        
        # Call to discard(...): (line 1014)
        # Processing the call arguments (line 1014)
        # Getting the type of 'self' (line 1014)
        self_5997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 41), 'self', False)
        # Processing the call keyword arguments (line 1014)
        kwargs_5998 = {}
        # Getting the type of 'ax' (line 1014)
        ax_5994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 16), 'ax', False)
        # Obtaining the member 'mouseover_set' of a type (line 1014)
        mouseover_set_5995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1014, 16), ax_5994, 'mouseover_set')
        # Obtaining the member 'discard' of a type (line 1014)
        discard_5996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1014, 16), mouseover_set_5995, 'discard')
        # Calling discard(args, kwargs) (line 1014)
        discard_call_result_5999 = invoke(stypy.reporting.localization.Localization(__file__, 1014, 16), discard_5996, *[self_5997], **kwargs_5998)
        
        # SSA join for if statement (line 1011)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1010)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'mouseover(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mouseover' in the type store
        # Getting the type of 'stypy_return_type' (line 1005)
        stypy_return_type_6000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6000)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mouseover'
        return stypy_return_type_6000


# Assigning a type to the variable 'Artist' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'Artist', Artist)

# Assigning a Str to a Name (line 80):
unicode_6001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 12), 'unicode', u'Artist')
# Getting the type of 'Artist'
Artist_6002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Artist')
# Setting the type of the member 'aname' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Artist_6002, 'aname', unicode_6001)

# Assigning a Num to a Name (line 81):
int_6003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 13), 'int')
# Getting the type of 'Artist'
Artist_6004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Artist')
# Setting the type of the member 'zorder' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Artist_6004, 'zorder', int_6003)

# Assigning a Call to a Name (line 85):

# Call to dict(...): (line 85)
# Processing the call keyword arguments (line 85)
int_6006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 29), 'int')
keyword_6007 = int_6006
kwargs_6008 = {'color': keyword_6007}
# Getting the type of 'dict' (line 85)
dict_6005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 18), 'dict', False)
# Calling dict(args, kwargs) (line 85)
dict_call_result_6009 = invoke(stypy.reporting.localization.Localization(__file__, 85, 18), dict_6005, *[], **kwargs_6008)

# Getting the type of 'Artist'
Artist_6010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Artist')
# Setting the type of the member '_prop_order' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Artist_6010, '_prop_order', dict_call_result_6009)
# Declaration of the 'ArtistInspector' class

class ArtistInspector(object, ):
    unicode_6011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1022, (-1)), 'unicode', u"\n    A helper class to inspect an :class:`~matplotlib.artist.Artist`\n    and return information about it's settable properties and their\n    current values.\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1023, 4, False)
        # Assigning a type to the variable 'self' (line 1024)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1024, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector.__init__', ['o'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['o'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_6012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, (-1)), 'unicode', u'\n        Initialize the artist inspector with an\n        :class:`~matplotlib.artist.Artist` or iterable of :class:`Artists`.\n        If an iterable is used, we assume it is a homogeneous sequence (all\n        :class:`Artists` are of the same type) and it is your responsibility\n        to make sure this is so.\n        ')
        
        
        # Call to iterable(...): (line 1031)
        # Processing the call arguments (line 1031)
        # Getting the type of 'o' (line 1031)
        o_6015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 26), 'o', False)
        # Processing the call keyword arguments (line 1031)
        kwargs_6016 = {}
        # Getting the type of 'cbook' (line 1031)
        cbook_6013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 11), 'cbook', False)
        # Obtaining the member 'iterable' of a type (line 1031)
        iterable_6014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1031, 11), cbook_6013, 'iterable')
        # Calling iterable(args, kwargs) (line 1031)
        iterable_call_result_6017 = invoke(stypy.reporting.localization.Localization(__file__, 1031, 11), iterable_6014, *[o_6015], **kwargs_6016)
        
        # Testing the type of an if condition (line 1031)
        if_condition_6018 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1031, 8), iterable_call_result_6017)
        # Assigning a type to the variable 'if_condition_6018' (line 1031)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 8), 'if_condition_6018', if_condition_6018)
        # SSA begins for if statement (line 1031)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1033):
        
        # Assigning a Call to a Name (line 1033):
        
        # Call to list(...): (line 1033)
        # Processing the call arguments (line 1033)
        # Getting the type of 'o' (line 1033)
        o_6020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 21), 'o', False)
        # Processing the call keyword arguments (line 1033)
        kwargs_6021 = {}
        # Getting the type of 'list' (line 1033)
        list_6019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 16), 'list', False)
        # Calling list(args, kwargs) (line 1033)
        list_call_result_6022 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 16), list_6019, *[o_6020], **kwargs_6021)
        
        # Assigning a type to the variable 'o' (line 1033)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 12), 'o', list_call_result_6022)
        
        
        # Call to len(...): (line 1034)
        # Processing the call arguments (line 1034)
        # Getting the type of 'o' (line 1034)
        o_6024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 19), 'o', False)
        # Processing the call keyword arguments (line 1034)
        kwargs_6025 = {}
        # Getting the type of 'len' (line 1034)
        len_6023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 15), 'len', False)
        # Calling len(args, kwargs) (line 1034)
        len_call_result_6026 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 15), len_6023, *[o_6024], **kwargs_6025)
        
        # Testing the type of an if condition (line 1034)
        if_condition_6027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1034, 12), len_call_result_6026)
        # Assigning a type to the variable 'if_condition_6027' (line 1034)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 12), 'if_condition_6027', if_condition_6027)
        # SSA begins for if statement (line 1034)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 1035):
        
        # Assigning a Subscript to a Name (line 1035):
        
        # Obtaining the type of the subscript
        int_6028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 22), 'int')
        # Getting the type of 'o' (line 1035)
        o_6029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 20), 'o')
        # Obtaining the member '__getitem__' of a type (line 1035)
        getitem___6030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1035, 20), o_6029, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1035)
        subscript_call_result_6031 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 20), getitem___6030, int_6028)
        
        # Assigning a type to the variable 'o' (line 1035)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 16), 'o', subscript_call_result_6031)
        # SSA join for if statement (line 1034)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1031)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 1037):
        
        # Assigning a Name to a Attribute (line 1037):
        # Getting the type of 'o' (line 1037)
        o_6032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 21), 'o')
        # Getting the type of 'self' (line 1037)
        self_6033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 8), 'self')
        # Setting the type of the member 'oorig' of a type (line 1037)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 8), self_6033, 'oorig', o_6032)
        
        
        
        # Call to isclass(...): (line 1038)
        # Processing the call arguments (line 1038)
        # Getting the type of 'o' (line 1038)
        o_6036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 31), 'o', False)
        # Processing the call keyword arguments (line 1038)
        kwargs_6037 = {}
        # Getting the type of 'inspect' (line 1038)
        inspect_6034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 15), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 1038)
        isclass_6035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1038, 15), inspect_6034, 'isclass')
        # Calling isclass(args, kwargs) (line 1038)
        isclass_call_result_6038 = invoke(stypy.reporting.localization.Localization(__file__, 1038, 15), isclass_6035, *[o_6036], **kwargs_6037)
        
        # Applying the 'not' unary operator (line 1038)
        result_not__6039 = python_operator(stypy.reporting.localization.Localization(__file__, 1038, 11), 'not', isclass_call_result_6038)
        
        # Testing the type of an if condition (line 1038)
        if_condition_6040 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1038, 8), result_not__6039)
        # Assigning a type to the variable 'if_condition_6040' (line 1038)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1038, 8), 'if_condition_6040', if_condition_6040)
        # SSA begins for if statement (line 1038)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1039):
        
        # Assigning a Call to a Name (line 1039):
        
        # Call to type(...): (line 1039)
        # Processing the call arguments (line 1039)
        # Getting the type of 'o' (line 1039)
        o_6042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 21), 'o', False)
        # Processing the call keyword arguments (line 1039)
        kwargs_6043 = {}
        # Getting the type of 'type' (line 1039)
        type_6041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 16), 'type', False)
        # Calling type(args, kwargs) (line 1039)
        type_call_result_6044 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 16), type_6041, *[o_6042], **kwargs_6043)
        
        # Assigning a type to the variable 'o' (line 1039)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 12), 'o', type_call_result_6044)
        # SSA join for if statement (line 1038)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 1040):
        
        # Assigning a Name to a Attribute (line 1040):
        # Getting the type of 'o' (line 1040)
        o_6045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 17), 'o')
        # Getting the type of 'self' (line 1040)
        self_6046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 8), 'self')
        # Setting the type of the member 'o' of a type (line 1040)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 8), self_6046, 'o', o_6045)
        
        # Assigning a Call to a Attribute (line 1042):
        
        # Assigning a Call to a Attribute (line 1042):
        
        # Call to get_aliases(...): (line 1042)
        # Processing the call keyword arguments (line 1042)
        kwargs_6049 = {}
        # Getting the type of 'self' (line 1042)
        self_6047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 22), 'self', False)
        # Obtaining the member 'get_aliases' of a type (line 1042)
        get_aliases_6048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 22), self_6047, 'get_aliases')
        # Calling get_aliases(args, kwargs) (line 1042)
        get_aliases_call_result_6050 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 22), get_aliases_6048, *[], **kwargs_6049)
        
        # Getting the type of 'self' (line 1042)
        self_6051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 8), 'self')
        # Setting the type of the member 'aliasd' of a type (line 1042)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1042, 8), self_6051, 'aliasd', get_aliases_call_result_6050)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_aliases(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_aliases'
        module_type_store = module_type_store.open_function_context('get_aliases', 1044, 4, False)
        # Assigning a type to the variable 'self' (line 1045)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1045, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArtistInspector.get_aliases.__dict__.__setitem__('stypy_localization', localization)
        ArtistInspector.get_aliases.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArtistInspector.get_aliases.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArtistInspector.get_aliases.__dict__.__setitem__('stypy_function_name', 'ArtistInspector.get_aliases')
        ArtistInspector.get_aliases.__dict__.__setitem__('stypy_param_names_list', [])
        ArtistInspector.get_aliases.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArtistInspector.get_aliases.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArtistInspector.get_aliases.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArtistInspector.get_aliases.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArtistInspector.get_aliases.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArtistInspector.get_aliases.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector.get_aliases', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_aliases', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_aliases(...)' code ##################

        unicode_6052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1055, (-1)), 'unicode', u"\n        Get a dict mapping *fullname* -> *alias* for each *alias* in\n        the :class:`~matplotlib.artist.ArtistInspector`.\n\n        e.g., for lines::\n\n          {'markerfacecolor': 'mfc',\n           'linewidth'      : 'lw',\n          }\n\n        ")
        
        # Assigning a ListComp to a Name (line 1056):
        
        # Assigning a ListComp to a Name (line 1056):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to dir(...): (line 1056)
        # Processing the call arguments (line 1056)
        # Getting the type of 'self' (line 1056)
        self_6072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 38), 'self', False)
        # Obtaining the member 'o' of a type (line 1056)
        o_6073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 38), self_6072, 'o')
        # Processing the call keyword arguments (line 1056)
        kwargs_6074 = {}
        # Getting the type of 'dir' (line 1056)
        dir_6071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 34), 'dir', False)
        # Calling dir(args, kwargs) (line 1056)
        dir_call_result_6075 = invoke(stypy.reporting.localization.Localization(__file__, 1056, 34), dir_6071, *[o_6073], **kwargs_6074)
        
        comprehension_6076 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1056, 17), dir_call_result_6075)
        # Assigning a type to the variable 'name' (line 1056)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 17), 'name', comprehension_6076)
        
        # Evaluating a boolean operation
        
        # Call to startswith(...): (line 1057)
        # Processing the call arguments (line 1057)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1057)
        tuple_6056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1057)
        # Adding element type (line 1057)
        unicode_6057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 37), 'unicode', u'set_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1057, 37), tuple_6056, unicode_6057)
        # Adding element type (line 1057)
        unicode_6058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 45), 'unicode', u'get_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1057, 37), tuple_6056, unicode_6058)
        
        # Processing the call keyword arguments (line 1057)
        kwargs_6059 = {}
        # Getting the type of 'name' (line 1057)
        name_6054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 20), 'name', False)
        # Obtaining the member 'startswith' of a type (line 1057)
        startswith_6055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1057, 20), name_6054, 'startswith')
        # Calling startswith(args, kwargs) (line 1057)
        startswith_call_result_6060 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 20), startswith_6055, *[tuple_6056], **kwargs_6059)
        
        
        # Call to callable(...): (line 1058)
        # Processing the call arguments (line 1058)
        
        # Call to getattr(...): (line 1058)
        # Processing the call arguments (line 1058)
        # Getting the type of 'self' (line 1058)
        self_6063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 41), 'self', False)
        # Obtaining the member 'o' of a type (line 1058)
        o_6064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1058, 41), self_6063, 'o')
        # Getting the type of 'name' (line 1058)
        name_6065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 49), 'name', False)
        # Processing the call keyword arguments (line 1058)
        kwargs_6066 = {}
        # Getting the type of 'getattr' (line 1058)
        getattr_6062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 33), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1058)
        getattr_call_result_6067 = invoke(stypy.reporting.localization.Localization(__file__, 1058, 33), getattr_6062, *[o_6064, name_6065], **kwargs_6066)
        
        # Processing the call keyword arguments (line 1058)
        kwargs_6068 = {}
        # Getting the type of 'callable' (line 1058)
        callable_6061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 24), 'callable', False)
        # Calling callable(args, kwargs) (line 1058)
        callable_call_result_6069 = invoke(stypy.reporting.localization.Localization(__file__, 1058, 24), callable_6061, *[getattr_call_result_6067], **kwargs_6068)
        
        # Applying the binary operator 'and' (line 1057)
        result_and_keyword_6070 = python_operator(stypy.reporting.localization.Localization(__file__, 1057, 20), 'and', startswith_call_result_6060, callable_call_result_6069)
        
        # Getting the type of 'name' (line 1056)
        name_6053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 17), 'name')
        list_6077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1056, 17), list_6077, name_6053)
        # Assigning a type to the variable 'names' (line 1056)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 8), 'names', list_6077)
        
        # Assigning a Dict to a Name (line 1059):
        
        # Assigning a Dict to a Name (line 1059):
        
        # Obtaining an instance of the builtin type 'dict' (line 1059)
        dict_6078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1059, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 1059)
        
        # Assigning a type to the variable 'aliases' (line 1059)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 8), 'aliases', dict_6078)
        
        # Getting the type of 'names' (line 1060)
        names_6079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 20), 'names')
        # Testing the type of a for loop iterable (line 1060)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1060, 8), names_6079)
        # Getting the type of the for loop variable (line 1060)
        for_loop_var_6080 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1060, 8), names_6079)
        # Assigning a type to the variable 'name' (line 1060)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1060, 8), 'name', for_loop_var_6080)
        # SSA begins for a for statement (line 1060)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 1061):
        
        # Assigning a Call to a Name (line 1061):
        
        # Call to getattr(...): (line 1061)
        # Processing the call arguments (line 1061)
        # Getting the type of 'self' (line 1061)
        self_6082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 27), 'self', False)
        # Obtaining the member 'o' of a type (line 1061)
        o_6083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1061, 27), self_6082, 'o')
        # Getting the type of 'name' (line 1061)
        name_6084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 35), 'name', False)
        # Processing the call keyword arguments (line 1061)
        kwargs_6085 = {}
        # Getting the type of 'getattr' (line 1061)
        getattr_6081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1061)
        getattr_call_result_6086 = invoke(stypy.reporting.localization.Localization(__file__, 1061, 19), getattr_6081, *[o_6083, name_6084], **kwargs_6085)
        
        # Assigning a type to the variable 'func' (line 1061)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1061, 12), 'func', getattr_call_result_6086)
        
        
        
        # Call to is_alias(...): (line 1062)
        # Processing the call arguments (line 1062)
        # Getting the type of 'func' (line 1062)
        func_6089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 33), 'func', False)
        # Processing the call keyword arguments (line 1062)
        kwargs_6090 = {}
        # Getting the type of 'self' (line 1062)
        self_6087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 19), 'self', False)
        # Obtaining the member 'is_alias' of a type (line 1062)
        is_alias_6088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1062, 19), self_6087, 'is_alias')
        # Calling is_alias(args, kwargs) (line 1062)
        is_alias_call_result_6091 = invoke(stypy.reporting.localization.Localization(__file__, 1062, 19), is_alias_6088, *[func_6089], **kwargs_6090)
        
        # Applying the 'not' unary operator (line 1062)
        result_not__6092 = python_operator(stypy.reporting.localization.Localization(__file__, 1062, 15), 'not', is_alias_call_result_6091)
        
        # Testing the type of an if condition (line 1062)
        if_condition_6093 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1062, 12), result_not__6092)
        # Assigning a type to the variable 'if_condition_6093' (line 1062)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1062, 12), 'if_condition_6093', if_condition_6093)
        # SSA begins for if statement (line 1062)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 1062)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 1064):
        
        # Assigning a Attribute to a Name (line 1064):
        # Getting the type of 'func' (line 1064)
        func_6094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 24), 'func')
        # Obtaining the member '__doc__' of a type (line 1064)
        doc___6095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1064, 24), func_6094, '__doc__')
        # Assigning a type to the variable 'docstring' (line 1064)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1064, 12), 'docstring', doc___6095)
        
        # Assigning a Subscript to a Name (line 1065):
        
        # Assigning a Subscript to a Name (line 1065):
        
        # Obtaining the type of the subscript
        int_6096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1065, 33), 'int')
        slice_6097 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1065, 23), int_6096, None, None)
        # Getting the type of 'docstring' (line 1065)
        docstring_6098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 23), 'docstring')
        # Obtaining the member '__getitem__' of a type (line 1065)
        getitem___6099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1065, 23), docstring_6098, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1065)
        subscript_call_result_6100 = invoke(stypy.reporting.localization.Localization(__file__, 1065, 23), getitem___6099, slice_6097)
        
        # Assigning a type to the variable 'fullname' (line 1065)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 12), 'fullname', subscript_call_result_6100)
        
        # Assigning a Name to a Subscript (line 1066):
        
        # Assigning a Name to a Subscript (line 1066):
        # Getting the type of 'None' (line 1066)
        None_6101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 61), 'None')
        
        # Call to setdefault(...): (line 1066)
        # Processing the call arguments (line 1066)
        
        # Obtaining the type of the subscript
        int_6104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, 40), 'int')
        slice_6105 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1066, 31), int_6104, None, None)
        # Getting the type of 'fullname' (line 1066)
        fullname_6106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 31), 'fullname', False)
        # Obtaining the member '__getitem__' of a type (line 1066)
        getitem___6107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 31), fullname_6106, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1066)
        subscript_call_result_6108 = invoke(stypy.reporting.localization.Localization(__file__, 1066, 31), getitem___6107, slice_6105)
        
        
        # Obtaining an instance of the builtin type 'dict' (line 1066)
        dict_6109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, 45), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 1066)
        
        # Processing the call keyword arguments (line 1066)
        kwargs_6110 = {}
        # Getting the type of 'aliases' (line 1066)
        aliases_6102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 12), 'aliases', False)
        # Obtaining the member 'setdefault' of a type (line 1066)
        setdefault_6103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 12), aliases_6102, 'setdefault')
        # Calling setdefault(args, kwargs) (line 1066)
        setdefault_call_result_6111 = invoke(stypy.reporting.localization.Localization(__file__, 1066, 12), setdefault_6103, *[subscript_call_result_6108, dict_6109], **kwargs_6110)
        
        
        # Obtaining the type of the subscript
        int_6112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, 54), 'int')
        slice_6113 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1066, 49), int_6112, None, None)
        # Getting the type of 'name' (line 1066)
        name_6114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 49), 'name')
        # Obtaining the member '__getitem__' of a type (line 1066)
        getitem___6115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 49), name_6114, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1066)
        subscript_call_result_6116 = invoke(stypy.reporting.localization.Localization(__file__, 1066, 49), getitem___6115, slice_6113)
        
        # Storing an element on a container (line 1066)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1066, 12), setdefault_call_result_6111, (subscript_call_result_6116, None_6101))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'aliases' (line 1067)
        aliases_6117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 15), 'aliases')
        # Assigning a type to the variable 'stypy_return_type' (line 1067)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1067, 8), 'stypy_return_type', aliases_6117)
        
        # ################# End of 'get_aliases(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_aliases' in the type store
        # Getting the type of 'stypy_return_type' (line 1044)
        stypy_return_type_6118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6118)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_aliases'
        return stypy_return_type_6118

    
    # Assigning a Call to a Name (line 1069):

    @norecursion
    def get_valid_values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_valid_values'
        module_type_store = module_type_store.open_function_context('get_valid_values', 1073, 4, False)
        # Assigning a type to the variable 'self' (line 1074)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1074, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArtistInspector.get_valid_values.__dict__.__setitem__('stypy_localization', localization)
        ArtistInspector.get_valid_values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArtistInspector.get_valid_values.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArtistInspector.get_valid_values.__dict__.__setitem__('stypy_function_name', 'ArtistInspector.get_valid_values')
        ArtistInspector.get_valid_values.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        ArtistInspector.get_valid_values.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArtistInspector.get_valid_values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArtistInspector.get_valid_values.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArtistInspector.get_valid_values.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArtistInspector.get_valid_values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArtistInspector.get_valid_values.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector.get_valid_values', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_valid_values', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_valid_values(...)' code ##################

        unicode_6119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1083, (-1)), 'unicode', u'\n        Get the legal arguments for the setter associated with *attr*.\n\n        This is done by querying the docstring of the function *set_attr*\n        for a line that begins with ACCEPTS:\n\n        e.g., for a line linestyle, return\n        "[ ``\'-\'`` | ``\'--\'`` | ``\'-.\'`` | ``\':\'`` | ``\'steps\'`` | ``\'None\'``\n        ]"\n        ')
        
        # Assigning a BinOp to a Name (line 1085):
        
        # Assigning a BinOp to a Name (line 1085):
        unicode_6120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 15), 'unicode', u'set_%s')
        # Getting the type of 'attr' (line 1085)
        attr_6121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 26), 'attr')
        # Applying the binary operator '%' (line 1085)
        result_mod_6122 = python_operator(stypy.reporting.localization.Localization(__file__, 1085, 15), '%', unicode_6120, attr_6121)
        
        # Assigning a type to the variable 'name' (line 1085)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1085, 8), 'name', result_mod_6122)
        
        
        
        # Call to hasattr(...): (line 1086)
        # Processing the call arguments (line 1086)
        # Getting the type of 'self' (line 1086)
        self_6124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 23), 'self', False)
        # Obtaining the member 'o' of a type (line 1086)
        o_6125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1086, 23), self_6124, 'o')
        # Getting the type of 'name' (line 1086)
        name_6126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 31), 'name', False)
        # Processing the call keyword arguments (line 1086)
        kwargs_6127 = {}
        # Getting the type of 'hasattr' (line 1086)
        hasattr_6123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 1086)
        hasattr_call_result_6128 = invoke(stypy.reporting.localization.Localization(__file__, 1086, 15), hasattr_6123, *[o_6125, name_6126], **kwargs_6127)
        
        # Applying the 'not' unary operator (line 1086)
        result_not__6129 = python_operator(stypy.reporting.localization.Localization(__file__, 1086, 11), 'not', hasattr_call_result_6128)
        
        # Testing the type of an if condition (line 1086)
        if_condition_6130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1086, 8), result_not__6129)
        # Assigning a type to the variable 'if_condition_6130' (line 1086)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1086, 8), 'if_condition_6130', if_condition_6130)
        # SSA begins for if statement (line 1086)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AttributeError(...): (line 1087)
        # Processing the call arguments (line 1087)
        unicode_6132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1087, 33), 'unicode', u'%s has no function %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1087)
        tuple_6133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1087, 60), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1087)
        # Adding element type (line 1087)
        # Getting the type of 'self' (line 1087)
        self_6134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 60), 'self', False)
        # Obtaining the member 'o' of a type (line 1087)
        o_6135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1087, 60), self_6134, 'o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1087, 60), tuple_6133, o_6135)
        # Adding element type (line 1087)
        # Getting the type of 'name' (line 1087)
        name_6136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 68), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1087, 60), tuple_6133, name_6136)
        
        # Applying the binary operator '%' (line 1087)
        result_mod_6137 = python_operator(stypy.reporting.localization.Localization(__file__, 1087, 33), '%', unicode_6132, tuple_6133)
        
        # Processing the call keyword arguments (line 1087)
        kwargs_6138 = {}
        # Getting the type of 'AttributeError' (line 1087)
        AttributeError_6131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 18), 'AttributeError', False)
        # Calling AttributeError(args, kwargs) (line 1087)
        AttributeError_call_result_6139 = invoke(stypy.reporting.localization.Localization(__file__, 1087, 18), AttributeError_6131, *[result_mod_6137], **kwargs_6138)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1087, 12), AttributeError_call_result_6139, 'raise parameter', BaseException)
        # SSA join for if statement (line 1086)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1088):
        
        # Assigning a Call to a Name (line 1088):
        
        # Call to getattr(...): (line 1088)
        # Processing the call arguments (line 1088)
        # Getting the type of 'self' (line 1088)
        self_6141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 23), 'self', False)
        # Obtaining the member 'o' of a type (line 1088)
        o_6142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1088, 23), self_6141, 'o')
        # Getting the type of 'name' (line 1088)
        name_6143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 31), 'name', False)
        # Processing the call keyword arguments (line 1088)
        kwargs_6144 = {}
        # Getting the type of 'getattr' (line 1088)
        getattr_6140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1088)
        getattr_call_result_6145 = invoke(stypy.reporting.localization.Localization(__file__, 1088, 15), getattr_6140, *[o_6142, name_6143], **kwargs_6144)
        
        # Assigning a type to the variable 'func' (line 1088)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1088, 8), 'func', getattr_call_result_6145)
        
        # Assigning a Attribute to a Name (line 1090):
        
        # Assigning a Attribute to a Name (line 1090):
        # Getting the type of 'func' (line 1090)
        func_6146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 20), 'func')
        # Obtaining the member '__doc__' of a type (line 1090)
        doc___6147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1090, 20), func_6146, '__doc__')
        # Assigning a type to the variable 'docstring' (line 1090)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1090, 8), 'docstring', doc___6147)
        
        # Type idiom detected: calculating its left and rigth part (line 1091)
        # Getting the type of 'docstring' (line 1091)
        docstring_6148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 11), 'docstring')
        # Getting the type of 'None' (line 1091)
        None_6149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 24), 'None')
        
        (may_be_6150, more_types_in_union_6151) = may_be_none(docstring_6148, None_6149)

        if may_be_6150:

            if more_types_in_union_6151:
                # Runtime conditional SSA (line 1091)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            unicode_6152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 19), 'unicode', u'unknown')
            # Assigning a type to the variable 'stypy_return_type' (line 1092)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1092, 12), 'stypy_return_type', unicode_6152)

            if more_types_in_union_6151:
                # SSA join for if statement (line 1091)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to startswith(...): (line 1094)
        # Processing the call arguments (line 1094)
        unicode_6155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 32), 'unicode', u'alias for ')
        # Processing the call keyword arguments (line 1094)
        kwargs_6156 = {}
        # Getting the type of 'docstring' (line 1094)
        docstring_6153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 11), 'docstring', False)
        # Obtaining the member 'startswith' of a type (line 1094)
        startswith_6154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 11), docstring_6153, 'startswith')
        # Calling startswith(args, kwargs) (line 1094)
        startswith_call_result_6157 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 11), startswith_6154, *[unicode_6155], **kwargs_6156)
        
        # Testing the type of an if condition (line 1094)
        if_condition_6158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1094, 8), startswith_call_result_6157)
        # Assigning a type to the variable 'if_condition_6158' (line 1094)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 8), 'if_condition_6158', if_condition_6158)
        # SSA begins for if statement (line 1094)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 1095)
        None_6159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 1095)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1095, 12), 'stypy_return_type', None_6159)
        # SSA join for if statement (line 1094)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1097):
        
        # Assigning a Call to a Name (line 1097):
        
        # Call to search(...): (line 1097)
        # Processing the call arguments (line 1097)
        # Getting the type of 'docstring' (line 1097)
        docstring_6163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 52), 'docstring', False)
        # Processing the call keyword arguments (line 1097)
        kwargs_6164 = {}
        # Getting the type of 'self' (line 1097)
        self_6160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 16), 'self', False)
        # Obtaining the member '_get_valid_values_regex' of a type (line 1097)
        _get_valid_values_regex_6161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 16), self_6160, '_get_valid_values_regex')
        # Obtaining the member 'search' of a type (line 1097)
        search_6162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 16), _get_valid_values_regex_6161, 'search')
        # Calling search(args, kwargs) (line 1097)
        search_call_result_6165 = invoke(stypy.reporting.localization.Localization(__file__, 1097, 16), search_6162, *[docstring_6163], **kwargs_6164)
        
        # Assigning a type to the variable 'match' (line 1097)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 8), 'match', search_call_result_6165)
        
        # Type idiom detected: calculating its left and rigth part (line 1098)
        # Getting the type of 'match' (line 1098)
        match_6166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 8), 'match')
        # Getting the type of 'None' (line 1098)
        None_6167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 24), 'None')
        
        (may_be_6168, more_types_in_union_6169) = may_not_be_none(match_6166, None_6167)

        if may_be_6168:

            if more_types_in_union_6169:
                # Runtime conditional SSA (line 1098)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to sub(...): (line 1099)
            # Processing the call arguments (line 1099)
            unicode_6172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 26), 'unicode', u'\n *')
            unicode_6173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 34), 'unicode', u' ')
            
            # Call to group(...): (line 1099)
            # Processing the call arguments (line 1099)
            int_6176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 51), 'int')
            # Processing the call keyword arguments (line 1099)
            kwargs_6177 = {}
            # Getting the type of 'match' (line 1099)
            match_6174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 39), 'match', False)
            # Obtaining the member 'group' of a type (line 1099)
            group_6175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1099, 39), match_6174, 'group')
            # Calling group(args, kwargs) (line 1099)
            group_call_result_6178 = invoke(stypy.reporting.localization.Localization(__file__, 1099, 39), group_6175, *[int_6176], **kwargs_6177)
            
            # Processing the call keyword arguments (line 1099)
            kwargs_6179 = {}
            # Getting the type of 're' (line 1099)
            re_6170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 19), 're', False)
            # Obtaining the member 'sub' of a type (line 1099)
            sub_6171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1099, 19), re_6170, 'sub')
            # Calling sub(args, kwargs) (line 1099)
            sub_call_result_6180 = invoke(stypy.reporting.localization.Localization(__file__, 1099, 19), sub_6171, *[unicode_6172, unicode_6173, group_call_result_6178], **kwargs_6179)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1099)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1099, 12), 'stypy_return_type', sub_call_result_6180)

            if more_types_in_union_6169:
                # SSA join for if statement (line 1098)
                module_type_store = module_type_store.join_ssa_context()


        
        unicode_6181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1100, 15), 'unicode', u'unknown')
        # Assigning a type to the variable 'stypy_return_type' (line 1100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1100, 8), 'stypy_return_type', unicode_6181)
        
        # ################# End of 'get_valid_values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_valid_values' in the type store
        # Getting the type of 'stypy_return_type' (line 1073)
        stypy_return_type_6182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6182)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_valid_values'
        return stypy_return_type_6182


    @norecursion
    def _get_setters_and_targets(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_setters_and_targets'
        module_type_store = module_type_store.open_function_context('_get_setters_and_targets', 1102, 4, False)
        # Assigning a type to the variable 'self' (line 1103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArtistInspector._get_setters_and_targets.__dict__.__setitem__('stypy_localization', localization)
        ArtistInspector._get_setters_and_targets.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArtistInspector._get_setters_and_targets.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArtistInspector._get_setters_and_targets.__dict__.__setitem__('stypy_function_name', 'ArtistInspector._get_setters_and_targets')
        ArtistInspector._get_setters_and_targets.__dict__.__setitem__('stypy_param_names_list', [])
        ArtistInspector._get_setters_and_targets.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArtistInspector._get_setters_and_targets.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArtistInspector._get_setters_and_targets.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArtistInspector._get_setters_and_targets.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArtistInspector._get_setters_and_targets.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArtistInspector._get_setters_and_targets.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector._get_setters_and_targets', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_setters_and_targets', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_setters_and_targets(...)' code ##################

        unicode_6183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, (-1)), 'unicode', u'\n        Get the attribute strings and a full path to where the setter\n        is defined for all setters in an object.\n        ')
        
        # Assigning a List to a Name (line 1108):
        
        # Assigning a List to a Name (line 1108):
        
        # Obtaining an instance of the builtin type 'list' (line 1108)
        list_6184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1108)
        
        # Assigning a type to the variable 'setters' (line 1108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'setters', list_6184)
        
        
        # Call to dir(...): (line 1109)
        # Processing the call arguments (line 1109)
        # Getting the type of 'self' (line 1109)
        self_6186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 24), 'self', False)
        # Obtaining the member 'o' of a type (line 1109)
        o_6187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1109, 24), self_6186, 'o')
        # Processing the call keyword arguments (line 1109)
        kwargs_6188 = {}
        # Getting the type of 'dir' (line 1109)
        dir_6185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 20), 'dir', False)
        # Calling dir(args, kwargs) (line 1109)
        dir_call_result_6189 = invoke(stypy.reporting.localization.Localization(__file__, 1109, 20), dir_6185, *[o_6187], **kwargs_6188)
        
        # Testing the type of a for loop iterable (line 1109)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1109, 8), dir_call_result_6189)
        # Getting the type of the for loop variable (line 1109)
        for_loop_var_6190 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1109, 8), dir_call_result_6189)
        # Assigning a type to the variable 'name' (line 1109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1109, 8), 'name', for_loop_var_6190)
        # SSA begins for a for statement (line 1109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to startswith(...): (line 1110)
        # Processing the call arguments (line 1110)
        unicode_6193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1110, 35), 'unicode', u'set_')
        # Processing the call keyword arguments (line 1110)
        kwargs_6194 = {}
        # Getting the type of 'name' (line 1110)
        name_6191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 19), 'name', False)
        # Obtaining the member 'startswith' of a type (line 1110)
        startswith_6192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1110, 19), name_6191, 'startswith')
        # Calling startswith(args, kwargs) (line 1110)
        startswith_call_result_6195 = invoke(stypy.reporting.localization.Localization(__file__, 1110, 19), startswith_6192, *[unicode_6193], **kwargs_6194)
        
        # Applying the 'not' unary operator (line 1110)
        result_not__6196 = python_operator(stypy.reporting.localization.Localization(__file__, 1110, 15), 'not', startswith_call_result_6195)
        
        # Testing the type of an if condition (line 1110)
        if_condition_6197 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1110, 12), result_not__6196)
        # Assigning a type to the variable 'if_condition_6197' (line 1110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1110, 12), 'if_condition_6197', if_condition_6197)
        # SSA begins for if statement (line 1110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 1110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1112):
        
        # Assigning a Call to a Name (line 1112):
        
        # Call to getattr(...): (line 1112)
        # Processing the call arguments (line 1112)
        # Getting the type of 'self' (line 1112)
        self_6199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 27), 'self', False)
        # Obtaining the member 'o' of a type (line 1112)
        o_6200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1112, 27), self_6199, 'o')
        # Getting the type of 'name' (line 1112)
        name_6201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 35), 'name', False)
        # Processing the call keyword arguments (line 1112)
        kwargs_6202 = {}
        # Getting the type of 'getattr' (line 1112)
        getattr_6198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1112)
        getattr_call_result_6203 = invoke(stypy.reporting.localization.Localization(__file__, 1112, 19), getattr_6198, *[o_6200, name_6201], **kwargs_6202)
        
        # Assigning a type to the variable 'func' (line 1112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1112, 12), 'func', getattr_call_result_6203)
        
        
        
        # Call to callable(...): (line 1113)
        # Processing the call arguments (line 1113)
        # Getting the type of 'func' (line 1113)
        func_6205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 28), 'func', False)
        # Processing the call keyword arguments (line 1113)
        kwargs_6206 = {}
        # Getting the type of 'callable' (line 1113)
        callable_6204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 19), 'callable', False)
        # Calling callable(args, kwargs) (line 1113)
        callable_call_result_6207 = invoke(stypy.reporting.localization.Localization(__file__, 1113, 19), callable_6204, *[func_6205], **kwargs_6206)
        
        # Applying the 'not' unary operator (line 1113)
        result_not__6208 = python_operator(stypy.reporting.localization.Localization(__file__, 1113, 15), 'not', callable_call_result_6207)
        
        # Testing the type of an if condition (line 1113)
        if_condition_6209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1113, 12), result_not__6208)
        # Assigning a type to the variable 'if_condition_6209' (line 1113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1113, 12), 'if_condition_6209', if_condition_6209)
        # SSA begins for if statement (line 1113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 1113)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'six' (line 1115)
        six_6210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 15), 'six')
        # Obtaining the member 'PY2' of a type (line 1115)
        PY2_6211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1115, 15), six_6210, 'PY2')
        # Testing the type of an if condition (line 1115)
        if_condition_6212 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1115, 12), PY2_6211)
        # Assigning a type to the variable 'if_condition_6212' (line 1115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 12), 'if_condition_6212', if_condition_6212)
        # SSA begins for if statement (line 1115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1116):
        
        # Assigning a Call to a Name (line 1116):
        
        # Call to len(...): (line 1116)
        # Processing the call arguments (line 1116)
        
        # Obtaining the type of the subscript
        int_6214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 53), 'int')
        
        # Call to getargspec(...): (line 1116)
        # Processing the call arguments (line 1116)
        # Getting the type of 'func' (line 1116)
        func_6217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 47), 'func', False)
        # Processing the call keyword arguments (line 1116)
        kwargs_6218 = {}
        # Getting the type of 'inspect' (line 1116)
        inspect_6215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 28), 'inspect', False)
        # Obtaining the member 'getargspec' of a type (line 1116)
        getargspec_6216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1116, 28), inspect_6215, 'getargspec')
        # Calling getargspec(args, kwargs) (line 1116)
        getargspec_call_result_6219 = invoke(stypy.reporting.localization.Localization(__file__, 1116, 28), getargspec_6216, *[func_6217], **kwargs_6218)
        
        # Obtaining the member '__getitem__' of a type (line 1116)
        getitem___6220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1116, 28), getargspec_call_result_6219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1116)
        subscript_call_result_6221 = invoke(stypy.reporting.localization.Localization(__file__, 1116, 28), getitem___6220, int_6214)
        
        # Processing the call keyword arguments (line 1116)
        kwargs_6222 = {}
        # Getting the type of 'len' (line 1116)
        len_6213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 24), 'len', False)
        # Calling len(args, kwargs) (line 1116)
        len_call_result_6223 = invoke(stypy.reporting.localization.Localization(__file__, 1116, 24), len_6213, *[subscript_call_result_6221], **kwargs_6222)
        
        # Assigning a type to the variable 'nargs' (line 1116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 16), 'nargs', len_call_result_6223)
        # SSA branch for the else part of an if statement (line 1115)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1118):
        
        # Assigning a Call to a Name (line 1118):
        
        # Call to len(...): (line 1118)
        # Processing the call arguments (line 1118)
        
        # Obtaining the type of the subscript
        int_6225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1118, 57), 'int')
        
        # Call to getfullargspec(...): (line 1118)
        # Processing the call arguments (line 1118)
        # Getting the type of 'func' (line 1118)
        func_6228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 51), 'func', False)
        # Processing the call keyword arguments (line 1118)
        kwargs_6229 = {}
        # Getting the type of 'inspect' (line 1118)
        inspect_6226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 28), 'inspect', False)
        # Obtaining the member 'getfullargspec' of a type (line 1118)
        getfullargspec_6227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1118, 28), inspect_6226, 'getfullargspec')
        # Calling getfullargspec(args, kwargs) (line 1118)
        getfullargspec_call_result_6230 = invoke(stypy.reporting.localization.Localization(__file__, 1118, 28), getfullargspec_6227, *[func_6228], **kwargs_6229)
        
        # Obtaining the member '__getitem__' of a type (line 1118)
        getitem___6231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1118, 28), getfullargspec_call_result_6230, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1118)
        subscript_call_result_6232 = invoke(stypy.reporting.localization.Localization(__file__, 1118, 28), getitem___6231, int_6225)
        
        # Processing the call keyword arguments (line 1118)
        kwargs_6233 = {}
        # Getting the type of 'len' (line 1118)
        len_6224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 24), 'len', False)
        # Calling len(args, kwargs) (line 1118)
        len_call_result_6234 = invoke(stypy.reporting.localization.Localization(__file__, 1118, 24), len_6224, *[subscript_call_result_6232], **kwargs_6233)
        
        # Assigning a type to the variable 'nargs' (line 1118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1118, 16), 'nargs', len_call_result_6234)
        # SSA join for if statement (line 1115)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'nargs' (line 1119)
        nargs_6235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 15), 'nargs')
        int_6236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 23), 'int')
        # Applying the binary operator '<' (line 1119)
        result_lt_6237 = python_operator(stypy.reporting.localization.Localization(__file__, 1119, 15), '<', nargs_6235, int_6236)
        
        
        # Call to is_alias(...): (line 1119)
        # Processing the call arguments (line 1119)
        # Getting the type of 'func' (line 1119)
        func_6240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 42), 'func', False)
        # Processing the call keyword arguments (line 1119)
        kwargs_6241 = {}
        # Getting the type of 'self' (line 1119)
        self_6238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 28), 'self', False)
        # Obtaining the member 'is_alias' of a type (line 1119)
        is_alias_6239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1119, 28), self_6238, 'is_alias')
        # Calling is_alias(args, kwargs) (line 1119)
        is_alias_call_result_6242 = invoke(stypy.reporting.localization.Localization(__file__, 1119, 28), is_alias_6239, *[func_6240], **kwargs_6241)
        
        # Applying the binary operator 'or' (line 1119)
        result_or_keyword_6243 = python_operator(stypy.reporting.localization.Localization(__file__, 1119, 15), 'or', result_lt_6237, is_alias_call_result_6242)
        
        # Testing the type of an if condition (line 1119)
        if_condition_6244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1119, 12), result_or_keyword_6243)
        # Assigning a type to the variable 'if_condition_6244' (line 1119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1119, 12), 'if_condition_6244', if_condition_6244)
        # SSA begins for if statement (line 1119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 1119)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1121):
        
        # Assigning a BinOp to a Name (line 1121):
        # Getting the type of 'self' (line 1121)
        self_6245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 27), 'self')
        # Obtaining the member 'o' of a type (line 1121)
        o_6246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 27), self_6245, 'o')
        # Obtaining the member '__module__' of a type (line 1121)
        module___6247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 27), o_6246, '__module__')
        unicode_6248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 47), 'unicode', u'.')
        # Applying the binary operator '+' (line 1121)
        result_add_6249 = python_operator(stypy.reporting.localization.Localization(__file__, 1121, 27), '+', module___6247, unicode_6248)
        
        # Getting the type of 'self' (line 1121)
        self_6250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 53), 'self')
        # Obtaining the member 'o' of a type (line 1121)
        o_6251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 53), self_6250, 'o')
        # Obtaining the member '__name__' of a type (line 1121)
        name___6252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 53), o_6251, '__name__')
        # Applying the binary operator '+' (line 1121)
        result_add_6253 = python_operator(stypy.reporting.localization.Localization(__file__, 1121, 51), '+', result_add_6249, name___6252)
        
        # Assigning a type to the variable 'source_class' (line 1121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1121, 12), 'source_class', result_add_6253)
        
        
        # Call to mro(...): (line 1122)
        # Processing the call keyword arguments (line 1122)
        kwargs_6257 = {}
        # Getting the type of 'self' (line 1122)
        self_6254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 23), 'self', False)
        # Obtaining the member 'o' of a type (line 1122)
        o_6255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1122, 23), self_6254, 'o')
        # Obtaining the member 'mro' of a type (line 1122)
        mro_6256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1122, 23), o_6255, 'mro')
        # Calling mro(args, kwargs) (line 1122)
        mro_call_result_6258 = invoke(stypy.reporting.localization.Localization(__file__, 1122, 23), mro_6256, *[], **kwargs_6257)
        
        # Testing the type of a for loop iterable (line 1122)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1122, 12), mro_call_result_6258)
        # Getting the type of the for loop variable (line 1122)
        for_loop_var_6259 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1122, 12), mro_call_result_6258)
        # Assigning a type to the variable 'cls' (line 1122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1122, 12), 'cls', for_loop_var_6259)
        # SSA begins for a for statement (line 1122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'name' (line 1123)
        name_6260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 19), 'name')
        # Getting the type of 'cls' (line 1123)
        cls_6261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1123, 27), 'cls')
        # Obtaining the member '__dict__' of a type (line 1123)
        dict___6262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1123, 27), cls_6261, '__dict__')
        # Applying the binary operator 'in' (line 1123)
        result_contains_6263 = python_operator(stypy.reporting.localization.Localization(__file__, 1123, 19), 'in', name_6260, dict___6262)
        
        # Testing the type of an if condition (line 1123)
        if_condition_6264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1123, 16), result_contains_6263)
        # Assigning a type to the variable 'if_condition_6264' (line 1123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1123, 16), 'if_condition_6264', if_condition_6264)
        # SSA begins for if statement (line 1123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1124):
        
        # Assigning a BinOp to a Name (line 1124):
        # Getting the type of 'cls' (line 1124)
        cls_6265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 35), 'cls')
        # Obtaining the member '__module__' of a type (line 1124)
        module___6266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 35), cls_6265, '__module__')
        unicode_6267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 52), 'unicode', u'.')
        # Applying the binary operator '+' (line 1124)
        result_add_6268 = python_operator(stypy.reporting.localization.Localization(__file__, 1124, 35), '+', module___6266, unicode_6267)
        
        # Getting the type of 'cls' (line 1124)
        cls_6269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 58), 'cls')
        # Obtaining the member '__name__' of a type (line 1124)
        name___6270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 58), cls_6269, '__name__')
        # Applying the binary operator '+' (line 1124)
        result_add_6271 = python_operator(stypy.reporting.localization.Localization(__file__, 1124, 56), '+', result_add_6268, name___6270)
        
        # Assigning a type to the variable 'source_class' (line 1124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1124, 20), 'source_class', result_add_6271)
        # SSA join for if statement (line 1123)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 1126)
        # Processing the call arguments (line 1126)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1126)
        tuple_6274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1126)
        # Adding element type (line 1126)
        
        # Obtaining the type of the subscript
        int_6275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 33), 'int')
        slice_6276 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1126, 28), int_6275, None, None)
        # Getting the type of 'name' (line 1126)
        name_6277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 28), 'name', False)
        # Obtaining the member '__getitem__' of a type (line 1126)
        getitem___6278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 28), name_6277, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
        subscript_call_result_6279 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 28), getitem___6278, slice_6276)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1126, 28), tuple_6274, subscript_call_result_6279)
        # Adding element type (line 1126)
        # Getting the type of 'source_class' (line 1126)
        source_class_6280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 38), 'source_class', False)
        unicode_6281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 53), 'unicode', u'.')
        # Applying the binary operator '+' (line 1126)
        result_add_6282 = python_operator(stypy.reporting.localization.Localization(__file__, 1126, 38), '+', source_class_6280, unicode_6281)
        
        # Getting the type of 'name' (line 1126)
        name_6283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 59), 'name', False)
        # Applying the binary operator '+' (line 1126)
        result_add_6284 = python_operator(stypy.reporting.localization.Localization(__file__, 1126, 57), '+', result_add_6282, name_6283)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1126, 28), tuple_6274, result_add_6284)
        
        # Processing the call keyword arguments (line 1126)
        kwargs_6285 = {}
        # Getting the type of 'setters' (line 1126)
        setters_6272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 12), 'setters', False)
        # Obtaining the member 'append' of a type (line 1126)
        append_6273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 12), setters_6272, 'append')
        # Calling append(args, kwargs) (line 1126)
        append_call_result_6286 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 12), append_6273, *[tuple_6274], **kwargs_6285)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'setters' (line 1127)
        setters_6287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 15), 'setters')
        # Assigning a type to the variable 'stypy_return_type' (line 1127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1127, 8), 'stypy_return_type', setters_6287)
        
        # ################# End of '_get_setters_and_targets(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_setters_and_targets' in the type store
        # Getting the type of 'stypy_return_type' (line 1102)
        stypy_return_type_6288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6288)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_setters_and_targets'
        return stypy_return_type_6288


    @norecursion
    def get_setters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_setters'
        module_type_store = module_type_store.open_function_context('get_setters', 1129, 4, False)
        # Assigning a type to the variable 'self' (line 1130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArtistInspector.get_setters.__dict__.__setitem__('stypy_localization', localization)
        ArtistInspector.get_setters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArtistInspector.get_setters.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArtistInspector.get_setters.__dict__.__setitem__('stypy_function_name', 'ArtistInspector.get_setters')
        ArtistInspector.get_setters.__dict__.__setitem__('stypy_param_names_list', [])
        ArtistInspector.get_setters.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArtistInspector.get_setters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArtistInspector.get_setters.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArtistInspector.get_setters.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArtistInspector.get_setters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArtistInspector.get_setters.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector.get_setters', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_setters', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_setters(...)' code ##################

        unicode_6289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1133, (-1)), 'unicode', u"\n        Get the attribute strings with setters for object.  e.g., for a line,\n        return ``['markerfacecolor', 'linewidth', ....]``.\n        ")
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to _get_setters_and_targets(...): (line 1135)
        # Processing the call keyword arguments (line 1135)
        kwargs_6293 = {}
        # Getting the type of 'self' (line 1135)
        self_6291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 41), 'self', False)
        # Obtaining the member '_get_setters_and_targets' of a type (line 1135)
        _get_setters_and_targets_6292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1135, 41), self_6291, '_get_setters_and_targets')
        # Calling _get_setters_and_targets(args, kwargs) (line 1135)
        _get_setters_and_targets_call_result_6294 = invoke(stypy.reporting.localization.Localization(__file__, 1135, 41), _get_setters_and_targets_6292, *[], **kwargs_6293)
        
        comprehension_6295 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 16), _get_setters_and_targets_call_result_6294)
        # Assigning a type to the variable 'prop' (line 1135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 16), 'prop', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 16), comprehension_6295))
        # Assigning a type to the variable 'target' (line 1135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 16), 'target', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 16), comprehension_6295))
        # Getting the type of 'prop' (line 1135)
        prop_6290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 16), 'prop')
        list_6296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 16), list_6296, prop_6290)
        # Assigning a type to the variable 'stypy_return_type' (line 1135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 8), 'stypy_return_type', list_6296)
        
        # ################# End of 'get_setters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_setters' in the type store
        # Getting the type of 'stypy_return_type' (line 1129)
        stypy_return_type_6297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_setters'
        return stypy_return_type_6297


    @norecursion
    def is_alias(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_alias'
        module_type_store = module_type_store.open_function_context('is_alias', 1137, 4, False)
        # Assigning a type to the variable 'self' (line 1138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArtistInspector.is_alias.__dict__.__setitem__('stypy_localization', localization)
        ArtistInspector.is_alias.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArtistInspector.is_alias.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArtistInspector.is_alias.__dict__.__setitem__('stypy_function_name', 'ArtistInspector.is_alias')
        ArtistInspector.is_alias.__dict__.__setitem__('stypy_param_names_list', ['o'])
        ArtistInspector.is_alias.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArtistInspector.is_alias.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArtistInspector.is_alias.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArtistInspector.is_alias.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArtistInspector.is_alias.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArtistInspector.is_alias.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector.is_alias', ['o'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_alias', localization, ['o'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_alias(...)' code ##################

        unicode_6298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1141, (-1)), 'unicode', u'\n        Return *True* if method object *o* is an alias for another\n        function.\n        ')
        
        # Assigning a Attribute to a Name (line 1142):
        
        # Assigning a Attribute to a Name (line 1142):
        # Getting the type of 'o' (line 1142)
        o_6299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 13), 'o')
        # Obtaining the member '__doc__' of a type (line 1142)
        doc___6300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1142, 13), o_6299, '__doc__')
        # Assigning a type to the variable 'ds' (line 1142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 8), 'ds', doc___6300)
        
        # Type idiom detected: calculating its left and rigth part (line 1143)
        # Getting the type of 'ds' (line 1143)
        ds_6301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 11), 'ds')
        # Getting the type of 'None' (line 1143)
        None_6302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 17), 'None')
        
        (may_be_6303, more_types_in_union_6304) = may_be_none(ds_6301, None_6302)

        if may_be_6303:

            if more_types_in_union_6304:
                # Runtime conditional SSA (line 1143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'False' (line 1144)
            False_6305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 1144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 12), 'stypy_return_type', False_6305)

            if more_types_in_union_6304:
                # SSA join for if statement (line 1143)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to startswith(...): (line 1145)
        # Processing the call arguments (line 1145)
        unicode_6308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1145, 29), 'unicode', u'alias for ')
        # Processing the call keyword arguments (line 1145)
        kwargs_6309 = {}
        # Getting the type of 'ds' (line 1145)
        ds_6306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 15), 'ds', False)
        # Obtaining the member 'startswith' of a type (line 1145)
        startswith_6307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1145, 15), ds_6306, 'startswith')
        # Calling startswith(args, kwargs) (line 1145)
        startswith_call_result_6310 = invoke(stypy.reporting.localization.Localization(__file__, 1145, 15), startswith_6307, *[unicode_6308], **kwargs_6309)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1145, 8), 'stypy_return_type', startswith_call_result_6310)
        
        # ################# End of 'is_alias(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_alias' in the type store
        # Getting the type of 'stypy_return_type' (line 1137)
        stypy_return_type_6311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6311)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_alias'
        return stypy_return_type_6311


    @norecursion
    def aliased_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'aliased_name'
        module_type_store = module_type_store.open_function_context('aliased_name', 1147, 4, False)
        # Assigning a type to the variable 'self' (line 1148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArtistInspector.aliased_name.__dict__.__setitem__('stypy_localization', localization)
        ArtistInspector.aliased_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArtistInspector.aliased_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArtistInspector.aliased_name.__dict__.__setitem__('stypy_function_name', 'ArtistInspector.aliased_name')
        ArtistInspector.aliased_name.__dict__.__setitem__('stypy_param_names_list', ['s'])
        ArtistInspector.aliased_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArtistInspector.aliased_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArtistInspector.aliased_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArtistInspector.aliased_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArtistInspector.aliased_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArtistInspector.aliased_name.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector.aliased_name', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'aliased_name', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'aliased_name(...)' code ##################

        unicode_6312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1155, (-1)), 'unicode', u"\n        return 'PROPNAME or alias' if *s* has an alias, else return\n        PROPNAME.\n\n        e.g., for the line markerfacecolor property, which has an\n        alias, return 'markerfacecolor or mfc' and for the transform\n        property, which does not, return 'transform'\n        ")
        
        
        # Getting the type of 's' (line 1157)
        s_6313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 11), 's')
        # Getting the type of 'self' (line 1157)
        self_6314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 16), 'self')
        # Obtaining the member 'aliasd' of a type (line 1157)
        aliasd_6315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1157, 16), self_6314, 'aliasd')
        # Applying the binary operator 'in' (line 1157)
        result_contains_6316 = python_operator(stypy.reporting.localization.Localization(__file__, 1157, 11), 'in', s_6313, aliasd_6315)
        
        # Testing the type of an if condition (line 1157)
        if_condition_6317 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1157, 8), result_contains_6316)
        # Assigning a type to the variable 'if_condition_6317' (line 1157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1157, 8), 'if_condition_6317', if_condition_6317)
        # SSA begins for if statement (line 1157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 's' (line 1158)
        s_6318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 19), 's')
        
        # Call to join(...): (line 1158)
        # Processing the call arguments (line 1158)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to sorted(...): (line 1159)
        # Processing the call arguments (line 1159)
        
        # Obtaining the type of the subscript
        # Getting the type of 's' (line 1159)
        s_6325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 60), 's', False)
        # Getting the type of 'self' (line 1159)
        self_6326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 48), 'self', False)
        # Obtaining the member 'aliasd' of a type (line 1159)
        aliasd_6327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1159, 48), self_6326, 'aliasd')
        # Obtaining the member '__getitem__' of a type (line 1159)
        getitem___6328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1159, 48), aliasd_6327, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1159)
        subscript_call_result_6329 = invoke(stypy.reporting.localization.Localization(__file__, 1159, 48), getitem___6328, s_6325)
        
        # Processing the call keyword arguments (line 1159)
        kwargs_6330 = {}
        # Getting the type of 'sorted' (line 1159)
        sorted_6324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 41), 'sorted', False)
        # Calling sorted(args, kwargs) (line 1159)
        sorted_call_result_6331 = invoke(stypy.reporting.localization.Localization(__file__, 1159, 41), sorted_6324, *[subscript_call_result_6329], **kwargs_6330)
        
        comprehension_6332 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1158, 32), sorted_call_result_6331)
        # Assigning a type to the variable 'x' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 32), 'x', comprehension_6332)
        unicode_6321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1158, 32), 'unicode', u' or %s')
        # Getting the type of 'x' (line 1158)
        x_6322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 43), 'x', False)
        # Applying the binary operator '%' (line 1158)
        result_mod_6323 = python_operator(stypy.reporting.localization.Localization(__file__, 1158, 32), '%', unicode_6321, x_6322)
        
        list_6333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1158, 32), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1158, 32), list_6333, result_mod_6323)
        # Processing the call keyword arguments (line 1158)
        kwargs_6334 = {}
        unicode_6319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1158, 23), 'unicode', u'')
        # Obtaining the member 'join' of a type (line 1158)
        join_6320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 23), unicode_6319, 'join')
        # Calling join(args, kwargs) (line 1158)
        join_call_result_6335 = invoke(stypy.reporting.localization.Localization(__file__, 1158, 23), join_6320, *[list_6333], **kwargs_6334)
        
        # Applying the binary operator '+' (line 1158)
        result_add_6336 = python_operator(stypy.reporting.localization.Localization(__file__, 1158, 19), '+', s_6318, join_call_result_6335)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 12), 'stypy_return_type', result_add_6336)
        # SSA branch for the else part of an if statement (line 1157)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 's' (line 1161)
        s_6337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 19), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 1161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 12), 'stypy_return_type', s_6337)
        # SSA join for if statement (line 1157)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'aliased_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'aliased_name' in the type store
        # Getting the type of 'stypy_return_type' (line 1147)
        stypy_return_type_6338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6338)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'aliased_name'
        return stypy_return_type_6338


    @norecursion
    def aliased_name_rest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'aliased_name_rest'
        module_type_store = module_type_store.open_function_context('aliased_name_rest', 1163, 4, False)
        # Assigning a type to the variable 'self' (line 1164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArtistInspector.aliased_name_rest.__dict__.__setitem__('stypy_localization', localization)
        ArtistInspector.aliased_name_rest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArtistInspector.aliased_name_rest.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArtistInspector.aliased_name_rest.__dict__.__setitem__('stypy_function_name', 'ArtistInspector.aliased_name_rest')
        ArtistInspector.aliased_name_rest.__dict__.__setitem__('stypy_param_names_list', ['s', 'target'])
        ArtistInspector.aliased_name_rest.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArtistInspector.aliased_name_rest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArtistInspector.aliased_name_rest.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArtistInspector.aliased_name_rest.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArtistInspector.aliased_name_rest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArtistInspector.aliased_name_rest.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector.aliased_name_rest', ['s', 'target'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'aliased_name_rest', localization, ['s', 'target'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'aliased_name_rest(...)' code ##################

        unicode_6339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1171, (-1)), 'unicode', u"\n        return 'PROPNAME or alias' if *s* has an alias, else return\n        PROPNAME formatted for ReST\n\n        e.g., for the line markerfacecolor property, which has an\n        alias, return 'markerfacecolor or mfc' and for the transform\n        property, which does not, return 'transform'\n        ")
        
        
        # Getting the type of 's' (line 1173)
        s_6340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 11), 's')
        # Getting the type of 'self' (line 1173)
        self_6341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 16), 'self')
        # Obtaining the member 'aliasd' of a type (line 1173)
        aliasd_6342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1173, 16), self_6341, 'aliasd')
        # Applying the binary operator 'in' (line 1173)
        result_contains_6343 = python_operator(stypy.reporting.localization.Localization(__file__, 1173, 11), 'in', s_6340, aliasd_6342)
        
        # Testing the type of an if condition (line 1173)
        if_condition_6344 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1173, 8), result_contains_6343)
        # Assigning a type to the variable 'if_condition_6344' (line 1173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1173, 8), 'if_condition_6344', if_condition_6344)
        # SSA begins for if statement (line 1173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1174):
        
        # Assigning a Call to a Name (line 1174):
        
        # Call to join(...): (line 1174)
        # Processing the call arguments (line 1174)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to sorted(...): (line 1175)
        # Processing the call arguments (line 1175)
        
        # Obtaining the type of the subscript
        # Getting the type of 's' (line 1175)
        s_6351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 59), 's', False)
        # Getting the type of 'self' (line 1175)
        self_6352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 47), 'self', False)
        # Obtaining the member 'aliasd' of a type (line 1175)
        aliasd_6353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1175, 47), self_6352, 'aliasd')
        # Obtaining the member '__getitem__' of a type (line 1175)
        getitem___6354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1175, 47), aliasd_6353, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1175)
        subscript_call_result_6355 = invoke(stypy.reporting.localization.Localization(__file__, 1175, 47), getitem___6354, s_6351)
        
        # Processing the call keyword arguments (line 1175)
        kwargs_6356 = {}
        # Getting the type of 'sorted' (line 1175)
        sorted_6350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 40), 'sorted', False)
        # Calling sorted(args, kwargs) (line 1175)
        sorted_call_result_6357 = invoke(stypy.reporting.localization.Localization(__file__, 1175, 40), sorted_6350, *[subscript_call_result_6355], **kwargs_6356)
        
        comprehension_6358 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1174, 31), sorted_call_result_6357)
        # Assigning a type to the variable 'x' (line 1174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1174, 31), 'x', comprehension_6358)
        unicode_6347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1174, 31), 'unicode', u' or %s')
        # Getting the type of 'x' (line 1174)
        x_6348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 42), 'x', False)
        # Applying the binary operator '%' (line 1174)
        result_mod_6349 = python_operator(stypy.reporting.localization.Localization(__file__, 1174, 31), '%', unicode_6347, x_6348)
        
        list_6359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1174, 31), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1174, 31), list_6359, result_mod_6349)
        # Processing the call keyword arguments (line 1174)
        kwargs_6360 = {}
        unicode_6345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1174, 22), 'unicode', u'')
        # Obtaining the member 'join' of a type (line 1174)
        join_6346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1174, 22), unicode_6345, 'join')
        # Calling join(args, kwargs) (line 1174)
        join_call_result_6361 = invoke(stypy.reporting.localization.Localization(__file__, 1174, 22), join_6346, *[list_6359], **kwargs_6360)
        
        # Assigning a type to the variable 'aliases' (line 1174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1174, 12), 'aliases', join_call_result_6361)
        # SSA branch for the else part of an if statement (line 1173)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 1177):
        
        # Assigning a Str to a Name (line 1177):
        unicode_6362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1177, 22), 'unicode', u'')
        # Assigning a type to the variable 'aliases' (line 1177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1177, 12), 'aliases', unicode_6362)
        # SSA join for if statement (line 1173)
        module_type_store = module_type_store.join_ssa_context()
        
        unicode_6363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1178, 15), 'unicode', u':meth:`%s <%s>`%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1178)
        tuple_6364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1178, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1178)
        # Adding element type (line 1178)
        # Getting the type of 's' (line 1178)
        s_6365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 38), 's')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1178, 38), tuple_6364, s_6365)
        # Adding element type (line 1178)
        # Getting the type of 'target' (line 1178)
        target_6366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 41), 'target')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1178, 38), tuple_6364, target_6366)
        # Adding element type (line 1178)
        # Getting the type of 'aliases' (line 1178)
        aliases_6367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 49), 'aliases')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1178, 38), tuple_6364, aliases_6367)
        
        # Applying the binary operator '%' (line 1178)
        result_mod_6368 = python_operator(stypy.reporting.localization.Localization(__file__, 1178, 15), '%', unicode_6363, tuple_6364)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1178, 8), 'stypy_return_type', result_mod_6368)
        
        # ################# End of 'aliased_name_rest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'aliased_name_rest' in the type store
        # Getting the type of 'stypy_return_type' (line 1163)
        stypy_return_type_6369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6369)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'aliased_name_rest'
        return stypy_return_type_6369


    @norecursion
    def pprint_setters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 1180)
        None_6370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 34), 'None')
        int_6371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1180, 53), 'int')
        defaults = [None_6370, int_6371]
        # Create a new context for function 'pprint_setters'
        module_type_store = module_type_store.open_function_context('pprint_setters', 1180, 4, False)
        # Assigning a type to the variable 'self' (line 1181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArtistInspector.pprint_setters.__dict__.__setitem__('stypy_localization', localization)
        ArtistInspector.pprint_setters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArtistInspector.pprint_setters.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArtistInspector.pprint_setters.__dict__.__setitem__('stypy_function_name', 'ArtistInspector.pprint_setters')
        ArtistInspector.pprint_setters.__dict__.__setitem__('stypy_param_names_list', ['prop', 'leadingspace'])
        ArtistInspector.pprint_setters.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArtistInspector.pprint_setters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArtistInspector.pprint_setters.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArtistInspector.pprint_setters.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArtistInspector.pprint_setters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArtistInspector.pprint_setters.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector.pprint_setters', ['prop', 'leadingspace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pprint_setters', localization, ['prop', 'leadingspace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pprint_setters(...)' code ##################

        unicode_6372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1188, (-1)), 'unicode', u'\n        If *prop* is *None*, return a list of strings of all settable properies\n        and their valid values.\n\n        If *prop* is not *None*, it is a valid property name and that\n        property will be returned as a string of property : valid\n        values.\n        ')
        
        # Getting the type of 'leadingspace' (line 1189)
        leadingspace_6373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 11), 'leadingspace')
        # Testing the type of an if condition (line 1189)
        if_condition_6374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1189, 8), leadingspace_6373)
        # Assigning a type to the variable 'if_condition_6374' (line 1189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1189, 8), 'if_condition_6374', if_condition_6374)
        # SSA begins for if statement (line 1189)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1190):
        
        # Assigning a BinOp to a Name (line 1190):
        unicode_6375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1190, 18), 'unicode', u' ')
        # Getting the type of 'leadingspace' (line 1190)
        leadingspace_6376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 24), 'leadingspace')
        # Applying the binary operator '*' (line 1190)
        result_mul_6377 = python_operator(stypy.reporting.localization.Localization(__file__, 1190, 18), '*', unicode_6375, leadingspace_6376)
        
        # Assigning a type to the variable 'pad' (line 1190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1190, 12), 'pad', result_mul_6377)
        # SSA branch for the else part of an if statement (line 1189)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 1192):
        
        # Assigning a Str to a Name (line 1192):
        unicode_6378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1192, 18), 'unicode', u'')
        # Assigning a type to the variable 'pad' (line 1192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1192, 12), 'pad', unicode_6378)
        # SSA join for if statement (line 1189)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 1193)
        # Getting the type of 'prop' (line 1193)
        prop_6379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 8), 'prop')
        # Getting the type of 'None' (line 1193)
        None_6380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 23), 'None')
        
        (may_be_6381, more_types_in_union_6382) = may_not_be_none(prop_6379, None_6380)

        if may_be_6381:

            if more_types_in_union_6382:
                # Runtime conditional SSA (line 1193)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 1194):
            
            # Assigning a Call to a Name (line 1194):
            
            # Call to get_valid_values(...): (line 1194)
            # Processing the call arguments (line 1194)
            # Getting the type of 'prop' (line 1194)
            prop_6385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 44), 'prop', False)
            # Processing the call keyword arguments (line 1194)
            kwargs_6386 = {}
            # Getting the type of 'self' (line 1194)
            self_6383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 22), 'self', False)
            # Obtaining the member 'get_valid_values' of a type (line 1194)
            get_valid_values_6384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1194, 22), self_6383, 'get_valid_values')
            # Calling get_valid_values(args, kwargs) (line 1194)
            get_valid_values_call_result_6387 = invoke(stypy.reporting.localization.Localization(__file__, 1194, 22), get_valid_values_6384, *[prop_6385], **kwargs_6386)
            
            # Assigning a type to the variable 'accepts' (line 1194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1194, 12), 'accepts', get_valid_values_call_result_6387)
            unicode_6388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1195, 19), 'unicode', u'%s%s: %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 1195)
            tuple_6389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1195, 33), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 1195)
            # Adding element type (line 1195)
            # Getting the type of 'pad' (line 1195)
            pad_6390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 33), 'pad')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1195, 33), tuple_6389, pad_6390)
            # Adding element type (line 1195)
            # Getting the type of 'prop' (line 1195)
            prop_6391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 38), 'prop')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1195, 33), tuple_6389, prop_6391)
            # Adding element type (line 1195)
            # Getting the type of 'accepts' (line 1195)
            accepts_6392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 44), 'accepts')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1195, 33), tuple_6389, accepts_6392)
            
            # Applying the binary operator '%' (line 1195)
            result_mod_6393 = python_operator(stypy.reporting.localization.Localization(__file__, 1195, 19), '%', unicode_6388, tuple_6389)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1195, 12), 'stypy_return_type', result_mod_6393)

            if more_types_in_union_6382:
                # SSA join for if statement (line 1193)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 1197):
        
        # Assigning a Call to a Name (line 1197):
        
        # Call to _get_setters_and_targets(...): (line 1197)
        # Processing the call keyword arguments (line 1197)
        kwargs_6396 = {}
        # Getting the type of 'self' (line 1197)
        self_6394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 16), 'self', False)
        # Obtaining the member '_get_setters_and_targets' of a type (line 1197)
        _get_setters_and_targets_6395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1197, 16), self_6394, '_get_setters_and_targets')
        # Calling _get_setters_and_targets(args, kwargs) (line 1197)
        _get_setters_and_targets_call_result_6397 = invoke(stypy.reporting.localization.Localization(__file__, 1197, 16), _get_setters_and_targets_6395, *[], **kwargs_6396)
        
        # Assigning a type to the variable 'attrs' (line 1197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1197, 8), 'attrs', _get_setters_and_targets_call_result_6397)
        
        # Call to sort(...): (line 1198)
        # Processing the call keyword arguments (line 1198)
        kwargs_6400 = {}
        # Getting the type of 'attrs' (line 1198)
        attrs_6398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 8), 'attrs', False)
        # Obtaining the member 'sort' of a type (line 1198)
        sort_6399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1198, 8), attrs_6398, 'sort')
        # Calling sort(args, kwargs) (line 1198)
        sort_call_result_6401 = invoke(stypy.reporting.localization.Localization(__file__, 1198, 8), sort_6399, *[], **kwargs_6400)
        
        
        # Assigning a List to a Name (line 1199):
        
        # Assigning a List to a Name (line 1199):
        
        # Obtaining an instance of the builtin type 'list' (line 1199)
        list_6402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1199, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1199)
        
        # Assigning a type to the variable 'lines' (line 1199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1199, 8), 'lines', list_6402)
        
        # Getting the type of 'attrs' (line 1201)
        attrs_6403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 26), 'attrs')
        # Testing the type of a for loop iterable (line 1201)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1201, 8), attrs_6403)
        # Getting the type of the for loop variable (line 1201)
        for_loop_var_6404 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1201, 8), attrs_6403)
        # Assigning a type to the variable 'prop' (line 1201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1201, 8), 'prop', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1201, 8), for_loop_var_6404))
        # Assigning a type to the variable 'path' (line 1201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1201, 8), 'path', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1201, 8), for_loop_var_6404))
        # SSA begins for a for statement (line 1201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 1202):
        
        # Assigning a Call to a Name (line 1202):
        
        # Call to get_valid_values(...): (line 1202)
        # Processing the call arguments (line 1202)
        # Getting the type of 'prop' (line 1202)
        prop_6407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1202, 44), 'prop', False)
        # Processing the call keyword arguments (line 1202)
        kwargs_6408 = {}
        # Getting the type of 'self' (line 1202)
        self_6405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1202, 22), 'self', False)
        # Obtaining the member 'get_valid_values' of a type (line 1202)
        get_valid_values_6406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1202, 22), self_6405, 'get_valid_values')
        # Calling get_valid_values(args, kwargs) (line 1202)
        get_valid_values_call_result_6409 = invoke(stypy.reporting.localization.Localization(__file__, 1202, 22), get_valid_values_6406, *[prop_6407], **kwargs_6408)
        
        # Assigning a type to the variable 'accepts' (line 1202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1202, 12), 'accepts', get_valid_values_call_result_6409)
        
        # Assigning a Call to a Name (line 1203):
        
        # Assigning a Call to a Name (line 1203):
        
        # Call to aliased_name(...): (line 1203)
        # Processing the call arguments (line 1203)
        # Getting the type of 'prop' (line 1203)
        prop_6412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 37), 'prop', False)
        # Processing the call keyword arguments (line 1203)
        kwargs_6413 = {}
        # Getting the type of 'self' (line 1203)
        self_6410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 19), 'self', False)
        # Obtaining the member 'aliased_name' of a type (line 1203)
        aliased_name_6411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1203, 19), self_6410, 'aliased_name')
        # Calling aliased_name(args, kwargs) (line 1203)
        aliased_name_call_result_6414 = invoke(stypy.reporting.localization.Localization(__file__, 1203, 19), aliased_name_6411, *[prop_6412], **kwargs_6413)
        
        # Assigning a type to the variable 'name' (line 1203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1203, 12), 'name', aliased_name_call_result_6414)
        
        # Call to append(...): (line 1205)
        # Processing the call arguments (line 1205)
        unicode_6417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 25), 'unicode', u'%s%s: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1205)
        tuple_6418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1205)
        # Adding element type (line 1205)
        # Getting the type of 'pad' (line 1205)
        pad_6419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 39), 'pad', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1205, 39), tuple_6418, pad_6419)
        # Adding element type (line 1205)
        # Getting the type of 'name' (line 1205)
        name_6420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 44), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1205, 39), tuple_6418, name_6420)
        # Adding element type (line 1205)
        # Getting the type of 'accepts' (line 1205)
        accepts_6421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 50), 'accepts', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1205, 39), tuple_6418, accepts_6421)
        
        # Applying the binary operator '%' (line 1205)
        result_mod_6422 = python_operator(stypy.reporting.localization.Localization(__file__, 1205, 25), '%', unicode_6417, tuple_6418)
        
        # Processing the call keyword arguments (line 1205)
        kwargs_6423 = {}
        # Getting the type of 'lines' (line 1205)
        lines_6415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'lines', False)
        # Obtaining the member 'append' of a type (line 1205)
        append_6416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 12), lines_6415, 'append')
        # Calling append(args, kwargs) (line 1205)
        append_call_result_6424 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 12), append_6416, *[result_mod_6422], **kwargs_6423)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'lines' (line 1206)
        lines_6425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 15), 'lines')
        # Assigning a type to the variable 'stypy_return_type' (line 1206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1206, 8), 'stypy_return_type', lines_6425)
        
        # ################# End of 'pprint_setters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pprint_setters' in the type store
        # Getting the type of 'stypy_return_type' (line 1180)
        stypy_return_type_6426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pprint_setters'
        return stypy_return_type_6426


    @norecursion
    def pprint_setters_rest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 1208)
        None_6427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 39), 'None')
        int_6428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1208, 58), 'int')
        defaults = [None_6427, int_6428]
        # Create a new context for function 'pprint_setters_rest'
        module_type_store = module_type_store.open_function_context('pprint_setters_rest', 1208, 4, False)
        # Assigning a type to the variable 'self' (line 1209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1209, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArtistInspector.pprint_setters_rest.__dict__.__setitem__('stypy_localization', localization)
        ArtistInspector.pprint_setters_rest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArtistInspector.pprint_setters_rest.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArtistInspector.pprint_setters_rest.__dict__.__setitem__('stypy_function_name', 'ArtistInspector.pprint_setters_rest')
        ArtistInspector.pprint_setters_rest.__dict__.__setitem__('stypy_param_names_list', ['prop', 'leadingspace'])
        ArtistInspector.pprint_setters_rest.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArtistInspector.pprint_setters_rest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArtistInspector.pprint_setters_rest.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArtistInspector.pprint_setters_rest.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArtistInspector.pprint_setters_rest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArtistInspector.pprint_setters_rest.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector.pprint_setters_rest', ['prop', 'leadingspace'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pprint_setters_rest', localization, ['prop', 'leadingspace'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pprint_setters_rest(...)' code ##################

        unicode_6429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, (-1)), 'unicode', u'\n        If *prop* is *None*, return a list of strings of all settable properies\n        and their valid values.  Format the output for ReST\n\n        If *prop* is not *None*, it is a valid property name and that\n        property will be returned as a string of property : valid\n        values.\n        ')
        
        # Getting the type of 'leadingspace' (line 1217)
        leadingspace_6430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 11), 'leadingspace')
        # Testing the type of an if condition (line 1217)
        if_condition_6431 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1217, 8), leadingspace_6430)
        # Assigning a type to the variable 'if_condition_6431' (line 1217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1217, 8), 'if_condition_6431', if_condition_6431)
        # SSA begins for if statement (line 1217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1218):
        
        # Assigning a BinOp to a Name (line 1218):
        unicode_6432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1218, 18), 'unicode', u' ')
        # Getting the type of 'leadingspace' (line 1218)
        leadingspace_6433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 24), 'leadingspace')
        # Applying the binary operator '*' (line 1218)
        result_mul_6434 = python_operator(stypy.reporting.localization.Localization(__file__, 1218, 18), '*', unicode_6432, leadingspace_6433)
        
        # Assigning a type to the variable 'pad' (line 1218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1218, 12), 'pad', result_mul_6434)
        # SSA branch for the else part of an if statement (line 1217)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 1220):
        
        # Assigning a Str to a Name (line 1220):
        unicode_6435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1220, 18), 'unicode', u'')
        # Assigning a type to the variable 'pad' (line 1220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1220, 12), 'pad', unicode_6435)
        # SSA join for if statement (line 1217)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 1221)
        # Getting the type of 'prop' (line 1221)
        prop_6436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 8), 'prop')
        # Getting the type of 'None' (line 1221)
        None_6437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 23), 'None')
        
        (may_be_6438, more_types_in_union_6439) = may_not_be_none(prop_6436, None_6437)

        if may_be_6438:

            if more_types_in_union_6439:
                # Runtime conditional SSA (line 1221)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 1222):
            
            # Assigning a Call to a Name (line 1222):
            
            # Call to get_valid_values(...): (line 1222)
            # Processing the call arguments (line 1222)
            # Getting the type of 'prop' (line 1222)
            prop_6442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 44), 'prop', False)
            # Processing the call keyword arguments (line 1222)
            kwargs_6443 = {}
            # Getting the type of 'self' (line 1222)
            self_6440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 22), 'self', False)
            # Obtaining the member 'get_valid_values' of a type (line 1222)
            get_valid_values_6441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1222, 22), self_6440, 'get_valid_values')
            # Calling get_valid_values(args, kwargs) (line 1222)
            get_valid_values_call_result_6444 = invoke(stypy.reporting.localization.Localization(__file__, 1222, 22), get_valid_values_6441, *[prop_6442], **kwargs_6443)
            
            # Assigning a type to the variable 'accepts' (line 1222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1222, 12), 'accepts', get_valid_values_call_result_6444)
            unicode_6445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1223, 19), 'unicode', u'%s%s: %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 1223)
            tuple_6446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1223, 33), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 1223)
            # Adding element type (line 1223)
            # Getting the type of 'pad' (line 1223)
            pad_6447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 33), 'pad')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1223, 33), tuple_6446, pad_6447)
            # Adding element type (line 1223)
            # Getting the type of 'prop' (line 1223)
            prop_6448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 38), 'prop')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1223, 33), tuple_6446, prop_6448)
            # Adding element type (line 1223)
            # Getting the type of 'accepts' (line 1223)
            accepts_6449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 44), 'accepts')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1223, 33), tuple_6446, accepts_6449)
            
            # Applying the binary operator '%' (line 1223)
            result_mod_6450 = python_operator(stypy.reporting.localization.Localization(__file__, 1223, 19), '%', unicode_6445, tuple_6446)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1223)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1223, 12), 'stypy_return_type', result_mod_6450)

            if more_types_in_union_6439:
                # SSA join for if statement (line 1221)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 1225):
        
        # Assigning a Call to a Name (line 1225):
        
        # Call to _get_setters_and_targets(...): (line 1225)
        # Processing the call keyword arguments (line 1225)
        kwargs_6453 = {}
        # Getting the type of 'self' (line 1225)
        self_6451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 16), 'self', False)
        # Obtaining the member '_get_setters_and_targets' of a type (line 1225)
        _get_setters_and_targets_6452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1225, 16), self_6451, '_get_setters_and_targets')
        # Calling _get_setters_and_targets(args, kwargs) (line 1225)
        _get_setters_and_targets_call_result_6454 = invoke(stypy.reporting.localization.Localization(__file__, 1225, 16), _get_setters_and_targets_6452, *[], **kwargs_6453)
        
        # Assigning a type to the variable 'attrs' (line 1225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1225, 8), 'attrs', _get_setters_and_targets_call_result_6454)
        
        # Call to sort(...): (line 1226)
        # Processing the call keyword arguments (line 1226)
        kwargs_6457 = {}
        # Getting the type of 'attrs' (line 1226)
        attrs_6455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 8), 'attrs', False)
        # Obtaining the member 'sort' of a type (line 1226)
        sort_6456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1226, 8), attrs_6455, 'sort')
        # Calling sort(args, kwargs) (line 1226)
        sort_call_result_6458 = invoke(stypy.reporting.localization.Localization(__file__, 1226, 8), sort_6456, *[], **kwargs_6457)
        
        
        # Assigning a List to a Name (line 1227):
        
        # Assigning a List to a Name (line 1227):
        
        # Obtaining an instance of the builtin type 'list' (line 1227)
        list_6459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1227)
        
        # Assigning a type to the variable 'lines' (line 1227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 8), 'lines', list_6459)
        
        # Assigning a ListComp to a Name (line 1230):
        
        # Assigning a ListComp to a Name (line 1230):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'attrs' (line 1231)
        attrs_6466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 37), 'attrs')
        comprehension_6467 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1230, 17), attrs_6466)
        # Assigning a type to the variable 'prop' (line 1230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1230, 17), 'prop', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1230, 17), comprehension_6467))
        # Assigning a type to the variable 'target' (line 1230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1230, 17), 'target', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1230, 17), comprehension_6467))
        
        # Call to aliased_name_rest(...): (line 1230)
        # Processing the call arguments (line 1230)
        # Getting the type of 'prop' (line 1230)
        prop_6462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 40), 'prop', False)
        # Getting the type of 'target' (line 1230)
        target_6463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 46), 'target', False)
        # Processing the call keyword arguments (line 1230)
        kwargs_6464 = {}
        # Getting the type of 'self' (line 1230)
        self_6460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 17), 'self', False)
        # Obtaining the member 'aliased_name_rest' of a type (line 1230)
        aliased_name_rest_6461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1230, 17), self_6460, 'aliased_name_rest')
        # Calling aliased_name_rest(args, kwargs) (line 1230)
        aliased_name_rest_call_result_6465 = invoke(stypy.reporting.localization.Localization(__file__, 1230, 17), aliased_name_rest_6461, *[prop_6462, target_6463], **kwargs_6464)
        
        list_6468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1230, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1230, 17), list_6468, aliased_name_rest_call_result_6465)
        # Assigning a type to the variable 'names' (line 1230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1230, 8), 'names', list_6468)
        
        # Assigning a ListComp to a Name (line 1232):
        
        # Assigning a ListComp to a Name (line 1232):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'attrs' (line 1232)
        attrs_6474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 67), 'attrs')
        comprehension_6475 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1232, 19), attrs_6474)
        # Assigning a type to the variable 'prop' (line 1232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 19), 'prop', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1232, 19), comprehension_6475))
        # Assigning a type to the variable 'target' (line 1232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 19), 'target', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1232, 19), comprehension_6475))
        
        # Call to get_valid_values(...): (line 1232)
        # Processing the call arguments (line 1232)
        # Getting the type of 'prop' (line 1232)
        prop_6471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 41), 'prop', False)
        # Processing the call keyword arguments (line 1232)
        kwargs_6472 = {}
        # Getting the type of 'self' (line 1232)
        self_6469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 19), 'self', False)
        # Obtaining the member 'get_valid_values' of a type (line 1232)
        get_valid_values_6470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1232, 19), self_6469, 'get_valid_values')
        # Calling get_valid_values(args, kwargs) (line 1232)
        get_valid_values_call_result_6473 = invoke(stypy.reporting.localization.Localization(__file__, 1232, 19), get_valid_values_6470, *[prop_6471], **kwargs_6472)
        
        list_6476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1232, 19), list_6476, get_valid_values_call_result_6473)
        # Assigning a type to the variable 'accepts' (line 1232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'accepts', list_6476)
        
        # Assigning a Call to a Name (line 1234):
        
        # Assigning a Call to a Name (line 1234):
        
        # Call to max(...): (line 1234)
        # Processing the call arguments (line 1234)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 1234, 23, True)
        # Calculating comprehension expression
        # Getting the type of 'names' (line 1234)
        names_6482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 39), 'names', False)
        comprehension_6483 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1234, 23), names_6482)
        # Assigning a type to the variable 'n' (line 1234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1234, 23), 'n', comprehension_6483)
        
        # Call to len(...): (line 1234)
        # Processing the call arguments (line 1234)
        # Getting the type of 'n' (line 1234)
        n_6479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 27), 'n', False)
        # Processing the call keyword arguments (line 1234)
        kwargs_6480 = {}
        # Getting the type of 'len' (line 1234)
        len_6478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 23), 'len', False)
        # Calling len(args, kwargs) (line 1234)
        len_call_result_6481 = invoke(stypy.reporting.localization.Localization(__file__, 1234, 23), len_6478, *[n_6479], **kwargs_6480)
        
        list_6484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1234, 23), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1234, 23), list_6484, len_call_result_6481)
        # Processing the call keyword arguments (line 1234)
        kwargs_6485 = {}
        # Getting the type of 'max' (line 1234)
        max_6477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 19), 'max', False)
        # Calling max(args, kwargs) (line 1234)
        max_call_result_6486 = invoke(stypy.reporting.localization.Localization(__file__, 1234, 19), max_6477, *[list_6484], **kwargs_6485)
        
        # Assigning a type to the variable 'col0_len' (line 1234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1234, 8), 'col0_len', max_call_result_6486)
        
        # Assigning a Call to a Name (line 1235):
        
        # Assigning a Call to a Name (line 1235):
        
        # Call to max(...): (line 1235)
        # Processing the call arguments (line 1235)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 1235, 23, True)
        # Calculating comprehension expression
        # Getting the type of 'accepts' (line 1235)
        accepts_6492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 39), 'accepts', False)
        comprehension_6493 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1235, 23), accepts_6492)
        # Assigning a type to the variable 'a' (line 1235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 23), 'a', comprehension_6493)
        
        # Call to len(...): (line 1235)
        # Processing the call arguments (line 1235)
        # Getting the type of 'a' (line 1235)
        a_6489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 27), 'a', False)
        # Processing the call keyword arguments (line 1235)
        kwargs_6490 = {}
        # Getting the type of 'len' (line 1235)
        len_6488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 23), 'len', False)
        # Calling len(args, kwargs) (line 1235)
        len_call_result_6491 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 23), len_6488, *[a_6489], **kwargs_6490)
        
        list_6494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 23), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1235, 23), list_6494, len_call_result_6491)
        # Processing the call keyword arguments (line 1235)
        kwargs_6495 = {}
        # Getting the type of 'max' (line 1235)
        max_6487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 19), 'max', False)
        # Calling max(args, kwargs) (line 1235)
        max_call_result_6496 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 19), max_6487, *[list_6494], **kwargs_6495)
        
        # Assigning a type to the variable 'col1_len' (line 1235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 8), 'col1_len', max_call_result_6496)
        
        # Assigning a BinOp to a Name (line 1236):
        
        # Assigning a BinOp to a Name (line 1236):
        # Getting the type of 'pad' (line 1236)
        pad_6497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 26), 'pad')
        unicode_6498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 32), 'unicode', u'=')
        # Getting the type of 'col0_len' (line 1236)
        col0_len_6499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 38), 'col0_len')
        # Applying the binary operator '*' (line 1236)
        result_mul_6500 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 32), '*', unicode_6498, col0_len_6499)
        
        # Applying the binary operator '+' (line 1236)
        result_add_6501 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 26), '+', pad_6497, result_mul_6500)
        
        unicode_6502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 49), 'unicode', u'   ')
        # Applying the binary operator '+' (line 1236)
        result_add_6503 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 47), '+', result_add_6501, unicode_6502)
        
        unicode_6504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 57), 'unicode', u'=')
        # Getting the type of 'col1_len' (line 1236)
        col1_len_6505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 63), 'col1_len')
        # Applying the binary operator '*' (line 1236)
        result_mul_6506 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 57), '*', unicode_6504, col1_len_6505)
        
        # Applying the binary operator '+' (line 1236)
        result_add_6507 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 55), '+', result_add_6503, result_mul_6506)
        
        # Assigning a type to the variable 'table_formatstr' (line 1236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 8), 'table_formatstr', result_add_6507)
        
        # Call to append(...): (line 1238)
        # Processing the call arguments (line 1238)
        unicode_6510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1238, 21), 'unicode', u'')
        # Processing the call keyword arguments (line 1238)
        kwargs_6511 = {}
        # Getting the type of 'lines' (line 1238)
        lines_6508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 1238)
        append_6509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1238, 8), lines_6508, 'append')
        # Calling append(args, kwargs) (line 1238)
        append_call_result_6512 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 8), append_6509, *[unicode_6510], **kwargs_6511)
        
        
        # Call to append(...): (line 1239)
        # Processing the call arguments (line 1239)
        # Getting the type of 'table_formatstr' (line 1239)
        table_formatstr_6515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 21), 'table_formatstr', False)
        # Processing the call keyword arguments (line 1239)
        kwargs_6516 = {}
        # Getting the type of 'lines' (line 1239)
        lines_6513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 1239)
        append_6514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1239, 8), lines_6513, 'append')
        # Calling append(args, kwargs) (line 1239)
        append_call_result_6517 = invoke(stypy.reporting.localization.Localization(__file__, 1239, 8), append_6514, *[table_formatstr_6515], **kwargs_6516)
        
        
        # Call to append(...): (line 1240)
        # Processing the call arguments (line 1240)
        # Getting the type of 'pad' (line 1240)
        pad_6520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 21), 'pad', False)
        
        # Call to ljust(...): (line 1240)
        # Processing the call arguments (line 1240)
        # Getting the type of 'col0_len' (line 1240)
        col0_len_6523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 44), 'col0_len', False)
        int_6524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1240, 55), 'int')
        # Applying the binary operator '+' (line 1240)
        result_add_6525 = python_operator(stypy.reporting.localization.Localization(__file__, 1240, 44), '+', col0_len_6523, int_6524)
        
        # Processing the call keyword arguments (line 1240)
        kwargs_6526 = {}
        unicode_6521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1240, 27), 'unicode', u'Property')
        # Obtaining the member 'ljust' of a type (line 1240)
        ljust_6522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 27), unicode_6521, 'ljust')
        # Calling ljust(args, kwargs) (line 1240)
        ljust_call_result_6527 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 27), ljust_6522, *[result_add_6525], **kwargs_6526)
        
        # Applying the binary operator '+' (line 1240)
        result_add_6528 = python_operator(stypy.reporting.localization.Localization(__file__, 1240, 21), '+', pad_6520, ljust_call_result_6527)
        
        
        # Call to ljust(...): (line 1241)
        # Processing the call arguments (line 1241)
        # Getting the type of 'col1_len' (line 1241)
        col1_len_6531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 41), 'col1_len', False)
        # Processing the call keyword arguments (line 1241)
        kwargs_6532 = {}
        unicode_6529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1241, 21), 'unicode', u'Description')
        # Obtaining the member 'ljust' of a type (line 1241)
        ljust_6530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1241, 21), unicode_6529, 'ljust')
        # Calling ljust(args, kwargs) (line 1241)
        ljust_call_result_6533 = invoke(stypy.reporting.localization.Localization(__file__, 1241, 21), ljust_6530, *[col1_len_6531], **kwargs_6532)
        
        # Applying the binary operator '+' (line 1240)
        result_add_6534 = python_operator(stypy.reporting.localization.Localization(__file__, 1240, 58), '+', result_add_6528, ljust_call_result_6533)
        
        # Processing the call keyword arguments (line 1240)
        kwargs_6535 = {}
        # Getting the type of 'lines' (line 1240)
        lines_6518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 1240)
        append_6519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 8), lines_6518, 'append')
        # Calling append(args, kwargs) (line 1240)
        append_call_result_6536 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 8), append_6519, *[result_add_6534], **kwargs_6535)
        
        
        # Call to append(...): (line 1242)
        # Processing the call arguments (line 1242)
        # Getting the type of 'table_formatstr' (line 1242)
        table_formatstr_6539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 21), 'table_formatstr', False)
        # Processing the call keyword arguments (line 1242)
        kwargs_6540 = {}
        # Getting the type of 'lines' (line 1242)
        lines_6537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 1242)
        append_6538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1242, 8), lines_6537, 'append')
        # Calling append(args, kwargs) (line 1242)
        append_call_result_6541 = invoke(stypy.reporting.localization.Localization(__file__, 1242, 8), append_6538, *[table_formatstr_6539], **kwargs_6540)
        
        
        # Call to extend(...): (line 1244)
        # Processing the call arguments (line 1244)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to zip(...): (line 1245)
        # Processing the call arguments (line 1245)
        # Getting the type of 'names' (line 1245)
        names_6560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 38), 'names', False)
        # Getting the type of 'accepts' (line 1245)
        accepts_6561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 45), 'accepts', False)
        # Processing the call keyword arguments (line 1245)
        kwargs_6562 = {}
        # Getting the type of 'zip' (line 1245)
        zip_6559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 34), 'zip', False)
        # Calling zip(args, kwargs) (line 1245)
        zip_call_result_6563 = invoke(stypy.reporting.localization.Localization(__file__, 1245, 34), zip_6559, *[names_6560, accepts_6561], **kwargs_6562)
        
        comprehension_6564 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1244, 22), zip_call_result_6563)
        # Assigning a type to the variable 'n' (line 1244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1244, 22), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1244, 22), comprehension_6564))
        # Assigning a type to the variable 'a' (line 1244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1244, 22), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1244, 22), comprehension_6564))
        # Getting the type of 'pad' (line 1244)
        pad_6544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 22), 'pad', False)
        
        # Call to ljust(...): (line 1244)
        # Processing the call arguments (line 1244)
        # Getting the type of 'col0_len' (line 1244)
        col0_len_6547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 36), 'col0_len', False)
        int_6548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1244, 47), 'int')
        # Applying the binary operator '+' (line 1244)
        result_add_6549 = python_operator(stypy.reporting.localization.Localization(__file__, 1244, 36), '+', col0_len_6547, int_6548)
        
        # Processing the call keyword arguments (line 1244)
        kwargs_6550 = {}
        # Getting the type of 'n' (line 1244)
        n_6545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 28), 'n', False)
        # Obtaining the member 'ljust' of a type (line 1244)
        ljust_6546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1244, 28), n_6545, 'ljust')
        # Calling ljust(args, kwargs) (line 1244)
        ljust_call_result_6551 = invoke(stypy.reporting.localization.Localization(__file__, 1244, 28), ljust_6546, *[result_add_6549], **kwargs_6550)
        
        # Applying the binary operator '+' (line 1244)
        result_add_6552 = python_operator(stypy.reporting.localization.Localization(__file__, 1244, 22), '+', pad_6544, ljust_call_result_6551)
        
        
        # Call to ljust(...): (line 1244)
        # Processing the call arguments (line 1244)
        # Getting the type of 'col1_len' (line 1244)
        col1_len_6555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 60), 'col1_len', False)
        # Processing the call keyword arguments (line 1244)
        kwargs_6556 = {}
        # Getting the type of 'a' (line 1244)
        a_6553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 52), 'a', False)
        # Obtaining the member 'ljust' of a type (line 1244)
        ljust_6554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1244, 52), a_6553, 'ljust')
        # Calling ljust(args, kwargs) (line 1244)
        ljust_call_result_6557 = invoke(stypy.reporting.localization.Localization(__file__, 1244, 52), ljust_6554, *[col1_len_6555], **kwargs_6556)
        
        # Applying the binary operator '+' (line 1244)
        result_add_6558 = python_operator(stypy.reporting.localization.Localization(__file__, 1244, 50), '+', result_add_6552, ljust_call_result_6557)
        
        list_6565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1244, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1244, 22), list_6565, result_add_6558)
        # Processing the call keyword arguments (line 1244)
        kwargs_6566 = {}
        # Getting the type of 'lines' (line 1244)
        lines_6542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 8), 'lines', False)
        # Obtaining the member 'extend' of a type (line 1244)
        extend_6543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1244, 8), lines_6542, 'extend')
        # Calling extend(args, kwargs) (line 1244)
        extend_call_result_6567 = invoke(stypy.reporting.localization.Localization(__file__, 1244, 8), extend_6543, *[list_6565], **kwargs_6566)
        
        
        # Call to append(...): (line 1247)
        # Processing the call arguments (line 1247)
        # Getting the type of 'table_formatstr' (line 1247)
        table_formatstr_6570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 21), 'table_formatstr', False)
        # Processing the call keyword arguments (line 1247)
        kwargs_6571 = {}
        # Getting the type of 'lines' (line 1247)
        lines_6568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 1247)
        append_6569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 8), lines_6568, 'append')
        # Calling append(args, kwargs) (line 1247)
        append_call_result_6572 = invoke(stypy.reporting.localization.Localization(__file__, 1247, 8), append_6569, *[table_formatstr_6570], **kwargs_6571)
        
        
        # Call to append(...): (line 1248)
        # Processing the call arguments (line 1248)
        unicode_6575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1248, 21), 'unicode', u'')
        # Processing the call keyword arguments (line 1248)
        kwargs_6576 = {}
        # Getting the type of 'lines' (line 1248)
        lines_6573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 1248)
        append_6574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1248, 8), lines_6573, 'append')
        # Calling append(args, kwargs) (line 1248)
        append_call_result_6577 = invoke(stypy.reporting.localization.Localization(__file__, 1248, 8), append_6574, *[unicode_6575], **kwargs_6576)
        
        # Getting the type of 'lines' (line 1249)
        lines_6578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1249, 15), 'lines')
        # Assigning a type to the variable 'stypy_return_type' (line 1249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1249, 8), 'stypy_return_type', lines_6578)
        
        # ################# End of 'pprint_setters_rest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pprint_setters_rest' in the type store
        # Getting the type of 'stypy_return_type' (line 1208)
        stypy_return_type_6579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6579)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pprint_setters_rest'
        return stypy_return_type_6579


    @norecursion
    def properties(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'properties'
        module_type_store = module_type_store.open_function_context('properties', 1251, 4, False)
        # Assigning a type to the variable 'self' (line 1252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1252, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArtistInspector.properties.__dict__.__setitem__('stypy_localization', localization)
        ArtistInspector.properties.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArtistInspector.properties.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArtistInspector.properties.__dict__.__setitem__('stypy_function_name', 'ArtistInspector.properties')
        ArtistInspector.properties.__dict__.__setitem__('stypy_param_names_list', [])
        ArtistInspector.properties.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArtistInspector.properties.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArtistInspector.properties.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArtistInspector.properties.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArtistInspector.properties.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArtistInspector.properties.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector.properties', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'properties', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'properties(...)' code ##################

        unicode_6580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1254, (-1)), 'unicode', u'\n        return a dictionary mapping property name -> value\n        ')
        
        # Assigning a Attribute to a Name (line 1255):
        
        # Assigning a Attribute to a Name (line 1255):
        # Getting the type of 'self' (line 1255)
        self_6581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1255, 12), 'self')
        # Obtaining the member 'oorig' of a type (line 1255)
        oorig_6582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1255, 12), self_6581, 'oorig')
        # Assigning a type to the variable 'o' (line 1255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1255, 8), 'o', oorig_6582)
        
        # Assigning a ListComp to a Name (line 1256):
        
        # Assigning a ListComp to a Name (line 1256):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to dir(...): (line 1256)
        # Processing the call arguments (line 1256)
        # Getting the type of 'o' (line 1256)
        o_6599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 40), 'o', False)
        # Processing the call keyword arguments (line 1256)
        kwargs_6600 = {}
        # Getting the type of 'dir' (line 1256)
        dir_6598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 36), 'dir', False)
        # Calling dir(args, kwargs) (line 1256)
        dir_call_result_6601 = invoke(stypy.reporting.localization.Localization(__file__, 1256, 36), dir_6598, *[o_6599], **kwargs_6600)
        
        comprehension_6602 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1256, 19), dir_call_result_6601)
        # Assigning a type to the variable 'name' (line 1256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1256, 19), 'name', comprehension_6602)
        
        # Evaluating a boolean operation
        
        # Call to startswith(...): (line 1257)
        # Processing the call arguments (line 1257)
        unicode_6586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1257, 38), 'unicode', u'get_')
        # Processing the call keyword arguments (line 1257)
        kwargs_6587 = {}
        # Getting the type of 'name' (line 1257)
        name_6584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 22), 'name', False)
        # Obtaining the member 'startswith' of a type (line 1257)
        startswith_6585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1257, 22), name_6584, 'startswith')
        # Calling startswith(args, kwargs) (line 1257)
        startswith_call_result_6588 = invoke(stypy.reporting.localization.Localization(__file__, 1257, 22), startswith_6585, *[unicode_6586], **kwargs_6587)
        
        
        # Call to callable(...): (line 1257)
        # Processing the call arguments (line 1257)
        
        # Call to getattr(...): (line 1257)
        # Processing the call arguments (line 1257)
        # Getting the type of 'o' (line 1257)
        o_6591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 67), 'o', False)
        # Getting the type of 'name' (line 1257)
        name_6592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 70), 'name', False)
        # Processing the call keyword arguments (line 1257)
        kwargs_6593 = {}
        # Getting the type of 'getattr' (line 1257)
        getattr_6590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 59), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1257)
        getattr_call_result_6594 = invoke(stypy.reporting.localization.Localization(__file__, 1257, 59), getattr_6590, *[o_6591, name_6592], **kwargs_6593)
        
        # Processing the call keyword arguments (line 1257)
        kwargs_6595 = {}
        # Getting the type of 'callable' (line 1257)
        callable_6589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 50), 'callable', False)
        # Calling callable(args, kwargs) (line 1257)
        callable_call_result_6596 = invoke(stypy.reporting.localization.Localization(__file__, 1257, 50), callable_6589, *[getattr_call_result_6594], **kwargs_6595)
        
        # Applying the binary operator 'and' (line 1257)
        result_and_keyword_6597 = python_operator(stypy.reporting.localization.Localization(__file__, 1257, 22), 'and', startswith_call_result_6588, callable_call_result_6596)
        
        # Getting the type of 'name' (line 1256)
        name_6583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 19), 'name')
        list_6603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1256, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1256, 19), list_6603, name_6583)
        # Assigning a type to the variable 'getters' (line 1256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1256, 8), 'getters', list_6603)
        
        # Call to sort(...): (line 1258)
        # Processing the call keyword arguments (line 1258)
        kwargs_6606 = {}
        # Getting the type of 'getters' (line 1258)
        getters_6604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'getters', False)
        # Obtaining the member 'sort' of a type (line 1258)
        sort_6605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1258, 8), getters_6604, 'sort')
        # Calling sort(args, kwargs) (line 1258)
        sort_call_result_6607 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 8), sort_6605, *[], **kwargs_6606)
        
        
        # Assigning a Call to a Name (line 1259):
        
        # Assigning a Call to a Name (line 1259):
        
        # Call to dict(...): (line 1259)
        # Processing the call keyword arguments (line 1259)
        kwargs_6609 = {}
        # Getting the type of 'dict' (line 1259)
        dict_6608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 12), 'dict', False)
        # Calling dict(args, kwargs) (line 1259)
        dict_call_result_6610 = invoke(stypy.reporting.localization.Localization(__file__, 1259, 12), dict_6608, *[], **kwargs_6609)
        
        # Assigning a type to the variable 'd' (line 1259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1259, 8), 'd', dict_call_result_6610)
        
        # Getting the type of 'getters' (line 1260)
        getters_6611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1260, 20), 'getters')
        # Testing the type of a for loop iterable (line 1260)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1260, 8), getters_6611)
        # Getting the type of the for loop variable (line 1260)
        for_loop_var_6612 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1260, 8), getters_6611)
        # Assigning a type to the variable 'name' (line 1260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1260, 8), 'name', for_loop_var_6612)
        # SSA begins for a for statement (line 1260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 1261):
        
        # Assigning a Call to a Name (line 1261):
        
        # Call to getattr(...): (line 1261)
        # Processing the call arguments (line 1261)
        # Getting the type of 'o' (line 1261)
        o_6614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 27), 'o', False)
        # Getting the type of 'name' (line 1261)
        name_6615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 30), 'name', False)
        # Processing the call keyword arguments (line 1261)
        kwargs_6616 = {}
        # Getting the type of 'getattr' (line 1261)
        getattr_6613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1261)
        getattr_call_result_6617 = invoke(stypy.reporting.localization.Localization(__file__, 1261, 19), getattr_6613, *[o_6614, name_6615], **kwargs_6616)
        
        # Assigning a type to the variable 'func' (line 1261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1261, 12), 'func', getattr_call_result_6617)
        
        
        # Call to is_alias(...): (line 1262)
        # Processing the call arguments (line 1262)
        # Getting the type of 'func' (line 1262)
        func_6620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1262, 29), 'func', False)
        # Processing the call keyword arguments (line 1262)
        kwargs_6621 = {}
        # Getting the type of 'self' (line 1262)
        self_6618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1262, 15), 'self', False)
        # Obtaining the member 'is_alias' of a type (line 1262)
        is_alias_6619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1262, 15), self_6618, 'is_alias')
        # Calling is_alias(args, kwargs) (line 1262)
        is_alias_call_result_6622 = invoke(stypy.reporting.localization.Localization(__file__, 1262, 15), is_alias_6619, *[func_6620], **kwargs_6621)
        
        # Testing the type of an if condition (line 1262)
        if_condition_6623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1262, 12), is_alias_call_result_6622)
        # Assigning a type to the variable 'if_condition_6623' (line 1262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1262, 12), 'if_condition_6623', if_condition_6623)
        # SSA begins for if statement (line 1262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 1262)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 1265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to catch_warnings(...): (line 1266)
        # Processing the call keyword arguments (line 1266)
        kwargs_6626 = {}
        # Getting the type of 'warnings' (line 1266)
        warnings_6624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 21), 'warnings', False)
        # Obtaining the member 'catch_warnings' of a type (line 1266)
        catch_warnings_6625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1266, 21), warnings_6624, 'catch_warnings')
        # Calling catch_warnings(args, kwargs) (line 1266)
        catch_warnings_call_result_6627 = invoke(stypy.reporting.localization.Localization(__file__, 1266, 21), catch_warnings_6625, *[], **kwargs_6626)
        
        with_6628 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 1266, 21), catch_warnings_call_result_6627, 'with parameter', '__enter__', '__exit__')

        if with_6628:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 1266)
            enter___6629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1266, 21), catch_warnings_call_result_6627, '__enter__')
            with_enter_6630 = invoke(stypy.reporting.localization.Localization(__file__, 1266, 21), enter___6629)
            
            # Call to simplefilter(...): (line 1267)
            # Processing the call arguments (line 1267)
            unicode_6633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1267, 42), 'unicode', u'ignore')
            # Processing the call keyword arguments (line 1267)
            kwargs_6634 = {}
            # Getting the type of 'warnings' (line 1267)
            warnings_6631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1267, 20), 'warnings', False)
            # Obtaining the member 'simplefilter' of a type (line 1267)
            simplefilter_6632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1267, 20), warnings_6631, 'simplefilter')
            # Calling simplefilter(args, kwargs) (line 1267)
            simplefilter_call_result_6635 = invoke(stypy.reporting.localization.Localization(__file__, 1267, 20), simplefilter_6632, *[unicode_6633], **kwargs_6634)
            
            
            # Assigning a Call to a Name (line 1268):
            
            # Assigning a Call to a Name (line 1268):
            
            # Call to func(...): (line 1268)
            # Processing the call keyword arguments (line 1268)
            kwargs_6637 = {}
            # Getting the type of 'func' (line 1268)
            func_6636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1268, 26), 'func', False)
            # Calling func(args, kwargs) (line 1268)
            func_call_result_6638 = invoke(stypy.reporting.localization.Localization(__file__, 1268, 26), func_6636, *[], **kwargs_6637)
            
            # Assigning a type to the variable 'val' (line 1268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1268, 20), 'val', func_call_result_6638)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 1266)
            exit___6639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1266, 21), catch_warnings_call_result_6627, '__exit__')
            with_exit_6640 = invoke(stypy.reporting.localization.Localization(__file__, 1266, 21), exit___6639, None, None, None)

        # SSA branch for the except part of a try statement (line 1265)
        # SSA branch for the except '<any exception>' branch of a try statement (line 1265)
        module_type_store.open_ssa_branch('except')
        # SSA branch for the else branch of a try statement (line 1265)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a Name to a Subscript (line 1272):
        
        # Assigning a Name to a Subscript (line 1272):
        # Getting the type of 'val' (line 1272)
        val_6641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1272, 30), 'val')
        # Getting the type of 'd' (line 1272)
        d_6642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1272, 16), 'd')
        
        # Obtaining the type of the subscript
        int_6643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1272, 23), 'int')
        slice_6644 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1272, 18), int_6643, None, None)
        # Getting the type of 'name' (line 1272)
        name_6645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1272, 18), 'name')
        # Obtaining the member '__getitem__' of a type (line 1272)
        getitem___6646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1272, 18), name_6645, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1272)
        subscript_call_result_6647 = invoke(stypy.reporting.localization.Localization(__file__, 1272, 18), getitem___6646, slice_6644)
        
        # Storing an element on a container (line 1272)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1272, 16), d_6642, (subscript_call_result_6647, val_6641))
        # SSA join for try-except statement (line 1265)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'd' (line 1274)
        d_6648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1274, 15), 'd')
        # Assigning a type to the variable 'stypy_return_type' (line 1274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1274, 8), 'stypy_return_type', d_6648)
        
        # ################# End of 'properties(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'properties' in the type store
        # Getting the type of 'stypy_return_type' (line 1251)
        stypy_return_type_6649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6649)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'properties'
        return stypy_return_type_6649


    @norecursion
    def pprint_getters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pprint_getters'
        module_type_store = module_type_store.open_function_context('pprint_getters', 1276, 4, False)
        # Assigning a type to the variable 'self' (line 1277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1277, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ArtistInspector.pprint_getters.__dict__.__setitem__('stypy_localization', localization)
        ArtistInspector.pprint_getters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ArtistInspector.pprint_getters.__dict__.__setitem__('stypy_type_store', module_type_store)
        ArtistInspector.pprint_getters.__dict__.__setitem__('stypy_function_name', 'ArtistInspector.pprint_getters')
        ArtistInspector.pprint_getters.__dict__.__setitem__('stypy_param_names_list', [])
        ArtistInspector.pprint_getters.__dict__.__setitem__('stypy_varargs_param_name', None)
        ArtistInspector.pprint_getters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ArtistInspector.pprint_getters.__dict__.__setitem__('stypy_call_defaults', defaults)
        ArtistInspector.pprint_getters.__dict__.__setitem__('stypy_call_varargs', varargs)
        ArtistInspector.pprint_getters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ArtistInspector.pprint_getters.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ArtistInspector.pprint_getters', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pprint_getters', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pprint_getters(...)' code ##################

        unicode_6650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1279, (-1)), 'unicode', u'\n        Return the getters and actual values as list of strings.\n        ')
        
        # Assigning a List to a Name (line 1281):
        
        # Assigning a List to a Name (line 1281):
        
        # Obtaining an instance of the builtin type 'list' (line 1281)
        list_6651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1281, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 1281)
        
        # Assigning a type to the variable 'lines' (line 1281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1281, 8), 'lines', list_6651)
        
        
        # Call to sorted(...): (line 1282)
        # Processing the call arguments (line 1282)
        
        # Call to iteritems(...): (line 1282)
        # Processing the call arguments (line 1282)
        
        # Call to properties(...): (line 1282)
        # Processing the call keyword arguments (line 1282)
        kwargs_6657 = {}
        # Getting the type of 'self' (line 1282)
        self_6655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1282, 46), 'self', False)
        # Obtaining the member 'properties' of a type (line 1282)
        properties_6656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1282, 46), self_6655, 'properties')
        # Calling properties(args, kwargs) (line 1282)
        properties_call_result_6658 = invoke(stypy.reporting.localization.Localization(__file__, 1282, 46), properties_6656, *[], **kwargs_6657)
        
        # Processing the call keyword arguments (line 1282)
        kwargs_6659 = {}
        # Getting the type of 'six' (line 1282)
        six_6653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1282, 32), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 1282)
        iteritems_6654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1282, 32), six_6653, 'iteritems')
        # Calling iteritems(args, kwargs) (line 1282)
        iteritems_call_result_6660 = invoke(stypy.reporting.localization.Localization(__file__, 1282, 32), iteritems_6654, *[properties_call_result_6658], **kwargs_6659)
        
        # Processing the call keyword arguments (line 1282)
        kwargs_6661 = {}
        # Getting the type of 'sorted' (line 1282)
        sorted_6652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1282, 25), 'sorted', False)
        # Calling sorted(args, kwargs) (line 1282)
        sorted_call_result_6662 = invoke(stypy.reporting.localization.Localization(__file__, 1282, 25), sorted_6652, *[iteritems_call_result_6660], **kwargs_6661)
        
        # Testing the type of a for loop iterable (line 1282)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1282, 8), sorted_call_result_6662)
        # Getting the type of the for loop variable (line 1282)
        for_loop_var_6663 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1282, 8), sorted_call_result_6662)
        # Assigning a type to the variable 'name' (line 1282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1282, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1282, 8), for_loop_var_6663))
        # Assigning a type to the variable 'val' (line 1282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1282, 8), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1282, 8), for_loop_var_6663))
        # SSA begins for a for statement (line 1282)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        
        # Call to getattr(...): (line 1283)
        # Processing the call arguments (line 1283)
        # Getting the type of 'val' (line 1283)
        val_6665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 23), 'val', False)
        unicode_6666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1283, 28), 'unicode', u'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1283)
        tuple_6667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1283, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1283)
        
        # Processing the call keyword arguments (line 1283)
        kwargs_6668 = {}
        # Getting the type of 'getattr' (line 1283)
        getattr_6664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1283)
        getattr_call_result_6669 = invoke(stypy.reporting.localization.Localization(__file__, 1283, 15), getattr_6664, *[val_6665, unicode_6666, tuple_6667], **kwargs_6668)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 1283)
        tuple_6670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1283, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1283)
        
        # Applying the binary operator '!=' (line 1283)
        result_ne_6671 = python_operator(stypy.reporting.localization.Localization(__file__, 1283, 15), '!=', getattr_call_result_6669, tuple_6670)
        
        
        
        # Call to len(...): (line 1283)
        # Processing the call arguments (line 1283)
        # Getting the type of 'val' (line 1283)
        val_6673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 55), 'val', False)
        # Processing the call keyword arguments (line 1283)
        kwargs_6674 = {}
        # Getting the type of 'len' (line 1283)
        len_6672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 51), 'len', False)
        # Calling len(args, kwargs) (line 1283)
        len_call_result_6675 = invoke(stypy.reporting.localization.Localization(__file__, 1283, 51), len_6672, *[val_6673], **kwargs_6674)
        
        int_6676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1283, 62), 'int')
        # Applying the binary operator '>' (line 1283)
        result_gt_6677 = python_operator(stypy.reporting.localization.Localization(__file__, 1283, 51), '>', len_call_result_6675, int_6676)
        
        # Applying the binary operator 'and' (line 1283)
        result_and_keyword_6678 = python_operator(stypy.reporting.localization.Localization(__file__, 1283, 15), 'and', result_ne_6671, result_gt_6677)
        
        # Testing the type of an if condition (line 1283)
        if_condition_6679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1283, 12), result_and_keyword_6678)
        # Assigning a type to the variable 'if_condition_6679' (line 1283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1283, 12), 'if_condition_6679', if_condition_6679)
        # SSA begins for if statement (line 1283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1284):
        
        # Assigning a BinOp to a Name (line 1284):
        
        # Call to str(...): (line 1284)
        # Processing the call arguments (line 1284)
        
        # Obtaining the type of the subscript
        int_6681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1284, 29), 'int')
        slice_6682 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1284, 24), None, int_6681, None)
        # Getting the type of 'val' (line 1284)
        val_6683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 24), 'val', False)
        # Obtaining the member '__getitem__' of a type (line 1284)
        getitem___6684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1284, 24), val_6683, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1284)
        subscript_call_result_6685 = invoke(stypy.reporting.localization.Localization(__file__, 1284, 24), getitem___6684, slice_6682)
        
        # Processing the call keyword arguments (line 1284)
        kwargs_6686 = {}
        # Getting the type of 'str' (line 1284)
        str_6680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 20), 'str', False)
        # Calling str(args, kwargs) (line 1284)
        str_call_result_6687 = invoke(stypy.reporting.localization.Localization(__file__, 1284, 20), str_6680, *[subscript_call_result_6685], **kwargs_6686)
        
        unicode_6688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1284, 35), 'unicode', u'...')
        # Applying the binary operator '+' (line 1284)
        result_add_6689 = python_operator(stypy.reporting.localization.Localization(__file__, 1284, 20), '+', str_call_result_6687, unicode_6688)
        
        # Assigning a type to the variable 's' (line 1284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1284, 16), 's', result_add_6689)
        # SSA branch for the else part of an if statement (line 1283)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1286):
        
        # Assigning a Call to a Name (line 1286):
        
        # Call to str(...): (line 1286)
        # Processing the call arguments (line 1286)
        # Getting the type of 'val' (line 1286)
        val_6691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 24), 'val', False)
        # Processing the call keyword arguments (line 1286)
        kwargs_6692 = {}
        # Getting the type of 'str' (line 1286)
        str_6690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 20), 'str', False)
        # Calling str(args, kwargs) (line 1286)
        str_call_result_6693 = invoke(stypy.reporting.localization.Localization(__file__, 1286, 20), str_6690, *[val_6691], **kwargs_6692)
        
        # Assigning a type to the variable 's' (line 1286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1286, 16), 's', str_call_result_6693)
        # SSA join for if statement (line 1283)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1287):
        
        # Assigning a Call to a Name (line 1287):
        
        # Call to replace(...): (line 1287)
        # Processing the call arguments (line 1287)
        unicode_6696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1287, 26), 'unicode', u'\n')
        unicode_6697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1287, 32), 'unicode', u' ')
        # Processing the call keyword arguments (line 1287)
        kwargs_6698 = {}
        # Getting the type of 's' (line 1287)
        s_6694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 16), 's', False)
        # Obtaining the member 'replace' of a type (line 1287)
        replace_6695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 16), s_6694, 'replace')
        # Calling replace(args, kwargs) (line 1287)
        replace_call_result_6699 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 16), replace_6695, *[unicode_6696, unicode_6697], **kwargs_6698)
        
        # Assigning a type to the variable 's' (line 1287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1287, 12), 's', replace_call_result_6699)
        
        
        
        # Call to len(...): (line 1288)
        # Processing the call arguments (line 1288)
        # Getting the type of 's' (line 1288)
        s_6701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 19), 's', False)
        # Processing the call keyword arguments (line 1288)
        kwargs_6702 = {}
        # Getting the type of 'len' (line 1288)
        len_6700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 15), 'len', False)
        # Calling len(args, kwargs) (line 1288)
        len_call_result_6703 = invoke(stypy.reporting.localization.Localization(__file__, 1288, 15), len_6700, *[s_6701], **kwargs_6702)
        
        int_6704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1288, 24), 'int')
        # Applying the binary operator '>' (line 1288)
        result_gt_6705 = python_operator(stypy.reporting.localization.Localization(__file__, 1288, 15), '>', len_call_result_6703, int_6704)
        
        # Testing the type of an if condition (line 1288)
        if_condition_6706 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1288, 12), result_gt_6705)
        # Assigning a type to the variable 'if_condition_6706' (line 1288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1288, 12), 'if_condition_6706', if_condition_6706)
        # SSA begins for if statement (line 1288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 1289):
        
        # Assigning a BinOp to a Name (line 1289):
        
        # Obtaining the type of the subscript
        int_6707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 23), 'int')
        slice_6708 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1289, 20), None, int_6707, None)
        # Getting the type of 's' (line 1289)
        s_6709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 20), 's')
        # Obtaining the member '__getitem__' of a type (line 1289)
        getitem___6710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1289, 20), s_6709, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1289)
        subscript_call_result_6711 = invoke(stypy.reporting.localization.Localization(__file__, 1289, 20), getitem___6710, slice_6708)
        
        unicode_6712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 29), 'unicode', u'...')
        # Applying the binary operator '+' (line 1289)
        result_add_6713 = python_operator(stypy.reporting.localization.Localization(__file__, 1289, 20), '+', subscript_call_result_6711, unicode_6712)
        
        # Assigning a type to the variable 's' (line 1289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 16), 's', result_add_6713)
        # SSA join for if statement (line 1288)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1290):
        
        # Assigning a Call to a Name (line 1290):
        
        # Call to aliased_name(...): (line 1290)
        # Processing the call arguments (line 1290)
        # Getting the type of 'name' (line 1290)
        name_6716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 37), 'name', False)
        # Processing the call keyword arguments (line 1290)
        kwargs_6717 = {}
        # Getting the type of 'self' (line 1290)
        self_6714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 19), 'self', False)
        # Obtaining the member 'aliased_name' of a type (line 1290)
        aliased_name_6715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1290, 19), self_6714, 'aliased_name')
        # Calling aliased_name(args, kwargs) (line 1290)
        aliased_name_call_result_6718 = invoke(stypy.reporting.localization.Localization(__file__, 1290, 19), aliased_name_6715, *[name_6716], **kwargs_6717)
        
        # Assigning a type to the variable 'name' (line 1290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1290, 12), 'name', aliased_name_call_result_6718)
        
        # Call to append(...): (line 1291)
        # Processing the call arguments (line 1291)
        unicode_6721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1291, 25), 'unicode', u'    %s = %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 1291)
        tuple_6722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1291, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1291)
        # Adding element type (line 1291)
        # Getting the type of 'name' (line 1291)
        name_6723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 42), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1291, 42), tuple_6722, name_6723)
        # Adding element type (line 1291)
        # Getting the type of 's' (line 1291)
        s_6724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 48), 's', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1291, 42), tuple_6722, s_6724)
        
        # Applying the binary operator '%' (line 1291)
        result_mod_6725 = python_operator(stypy.reporting.localization.Localization(__file__, 1291, 25), '%', unicode_6721, tuple_6722)
        
        # Processing the call keyword arguments (line 1291)
        kwargs_6726 = {}
        # Getting the type of 'lines' (line 1291)
        lines_6719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 12), 'lines', False)
        # Obtaining the member 'append' of a type (line 1291)
        append_6720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1291, 12), lines_6719, 'append')
        # Calling append(args, kwargs) (line 1291)
        append_call_result_6727 = invoke(stypy.reporting.localization.Localization(__file__, 1291, 12), append_6720, *[result_mod_6725], **kwargs_6726)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'lines' (line 1292)
        lines_6728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 15), 'lines')
        # Assigning a type to the variable 'stypy_return_type' (line 1292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1292, 8), 'stypy_return_type', lines_6728)
        
        # ################# End of 'pprint_getters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pprint_getters' in the type store
        # Getting the type of 'stypy_return_type' (line 1276)
        stypy_return_type_6729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1276, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6729)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pprint_getters'
        return stypy_return_type_6729


# Assigning a type to the variable 'ArtistInspector' (line 1017)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 0), 'ArtistInspector', ArtistInspector)

# Assigning a Call to a Name (line 1069):

# Call to compile(...): (line 1069)
# Processing the call arguments (line 1069)
unicode_6732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1070, 8), 'unicode', u'\\n\\s*ACCEPTS:\\s*((?:.|\\n)*?)(?:$|(?:\\n\\n))')
# Processing the call keyword arguments (line 1069)
kwargs_6733 = {}
# Getting the type of 're' (line 1069)
re_6730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 30), 're', False)
# Obtaining the member 'compile' of a type (line 1069)
compile_6731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 30), re_6730, 'compile')
# Calling compile(args, kwargs) (line 1069)
compile_call_result_6734 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 30), compile_6731, *[unicode_6732], **kwargs_6733)

# Getting the type of 'ArtistInspector'
ArtistInspector_6735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ArtistInspector')
# Setting the type of the member '_get_valid_values_regex' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ArtistInspector_6735, '_get_valid_values_regex', compile_call_result_6734)

@norecursion
def getp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1295)
    None_6736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 23), 'None')
    defaults = [None_6736]
    # Create a new context for function 'getp'
    module_type_store = module_type_store.open_function_context('getp', 1295, 0, False)
    
    # Passed parameters checking function
    getp.stypy_localization = localization
    getp.stypy_type_of_self = None
    getp.stypy_type_store = module_type_store
    getp.stypy_function_name = 'getp'
    getp.stypy_param_names_list = ['obj', 'property']
    getp.stypy_varargs_param_name = None
    getp.stypy_kwargs_param_name = None
    getp.stypy_call_defaults = defaults
    getp.stypy_call_varargs = varargs
    getp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getp', ['obj', 'property'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getp', localization, ['obj', 'property'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getp(...)' code ##################

    unicode_6737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1322, (-1)), 'unicode', u"\n    Return the value of object's property.  *property* is an optional string\n    for the property you want to return\n\n    Example usage::\n\n        getp(obj)  # get all the object properties\n        getp(obj, 'linestyle')  # get the linestyle property\n\n    *obj* is a :class:`Artist` instance, e.g.,\n    :class:`~matplotllib.lines.Line2D` or an instance of a\n    :class:`~matplotlib.axes.Axes` or :class:`matplotlib.text.Text`.\n    If the *property* is 'somename', this function returns\n\n      obj.get_somename()\n\n    :func:`getp` can be used to query all the gettable properties with\n    ``getp(obj)``. Many properties have aliases for shorter typing, e.g.\n    'lw' is an alias for 'linewidth'.  In the output, aliases and full\n    property names will be listed as:\n\n      property or alias = value\n\n    e.g.:\n\n      linewidth or lw = 2\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 1323)
    # Getting the type of 'property' (line 1323)
    property_6738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1323, 7), 'property')
    # Getting the type of 'None' (line 1323)
    None_6739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1323, 19), 'None')
    
    (may_be_6740, more_types_in_union_6741) = may_be_none(property_6738, None_6739)

    if may_be_6740:

        if more_types_in_union_6741:
            # Runtime conditional SSA (line 1323)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 1324):
        
        # Assigning a Call to a Name (line 1324):
        
        # Call to ArtistInspector(...): (line 1324)
        # Processing the call arguments (line 1324)
        # Getting the type of 'obj' (line 1324)
        obj_6743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1324, 31), 'obj', False)
        # Processing the call keyword arguments (line 1324)
        kwargs_6744 = {}
        # Getting the type of 'ArtistInspector' (line 1324)
        ArtistInspector_6742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1324, 15), 'ArtistInspector', False)
        # Calling ArtistInspector(args, kwargs) (line 1324)
        ArtistInspector_call_result_6745 = invoke(stypy.reporting.localization.Localization(__file__, 1324, 15), ArtistInspector_6742, *[obj_6743], **kwargs_6744)
        
        # Assigning a type to the variable 'insp' (line 1324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1324, 8), 'insp', ArtistInspector_call_result_6745)
        
        # Assigning a Call to a Name (line 1325):
        
        # Assigning a Call to a Name (line 1325):
        
        # Call to pprint_getters(...): (line 1325)
        # Processing the call keyword arguments (line 1325)
        kwargs_6748 = {}
        # Getting the type of 'insp' (line 1325)
        insp_6746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 14), 'insp', False)
        # Obtaining the member 'pprint_getters' of a type (line 1325)
        pprint_getters_6747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1325, 14), insp_6746, 'pprint_getters')
        # Calling pprint_getters(args, kwargs) (line 1325)
        pprint_getters_call_result_6749 = invoke(stypy.reporting.localization.Localization(__file__, 1325, 14), pprint_getters_6747, *[], **kwargs_6748)
        
        # Assigning a type to the variable 'ret' (line 1325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1325, 8), 'ret', pprint_getters_call_result_6749)
        
        # Call to print(...): (line 1326)
        # Processing the call arguments (line 1326)
        
        # Call to join(...): (line 1326)
        # Processing the call arguments (line 1326)
        # Getting the type of 'ret' (line 1326)
        ret_6753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1326, 24), 'ret', False)
        # Processing the call keyword arguments (line 1326)
        kwargs_6754 = {}
        unicode_6751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1326, 14), 'unicode', u'\n')
        # Obtaining the member 'join' of a type (line 1326)
        join_6752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1326, 14), unicode_6751, 'join')
        # Calling join(args, kwargs) (line 1326)
        join_call_result_6755 = invoke(stypy.reporting.localization.Localization(__file__, 1326, 14), join_6752, *[ret_6753], **kwargs_6754)
        
        # Processing the call keyword arguments (line 1326)
        kwargs_6756 = {}
        # Getting the type of 'print' (line 1326)
        print_6750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1326, 8), 'print', False)
        # Calling print(args, kwargs) (line 1326)
        print_call_result_6757 = invoke(stypy.reporting.localization.Localization(__file__, 1326, 8), print_6750, *[join_call_result_6755], **kwargs_6756)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1327, 8), 'stypy_return_type', types.NoneType)

        if more_types_in_union_6741:
            # SSA join for if statement (line 1323)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1329):
    
    # Assigning a Call to a Name (line 1329):
    
    # Call to getattr(...): (line 1329)
    # Processing the call arguments (line 1329)
    # Getting the type of 'obj' (line 1329)
    obj_6759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 19), 'obj', False)
    unicode_6760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1329, 24), 'unicode', u'get_')
    # Getting the type of 'property' (line 1329)
    property_6761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 33), 'property', False)
    # Applying the binary operator '+' (line 1329)
    result_add_6762 = python_operator(stypy.reporting.localization.Localization(__file__, 1329, 24), '+', unicode_6760, property_6761)
    
    # Processing the call keyword arguments (line 1329)
    kwargs_6763 = {}
    # Getting the type of 'getattr' (line 1329)
    getattr_6758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 11), 'getattr', False)
    # Calling getattr(args, kwargs) (line 1329)
    getattr_call_result_6764 = invoke(stypy.reporting.localization.Localization(__file__, 1329, 11), getattr_6758, *[obj_6759, result_add_6762], **kwargs_6763)
    
    # Assigning a type to the variable 'func' (line 1329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1329, 4), 'func', getattr_call_result_6764)
    
    # Call to func(...): (line 1330)
    # Processing the call keyword arguments (line 1330)
    kwargs_6766 = {}
    # Getting the type of 'func' (line 1330)
    func_6765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1330, 11), 'func', False)
    # Calling func(args, kwargs) (line 1330)
    func_call_result_6767 = invoke(stypy.reporting.localization.Localization(__file__, 1330, 11), func_6765, *[], **kwargs_6766)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1330, 4), 'stypy_return_type', func_call_result_6767)
    
    # ################# End of 'getp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getp' in the type store
    # Getting the type of 'stypy_return_type' (line 1295)
    stypy_return_type_6768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6768)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getp'
    return stypy_return_type_6768

# Assigning a type to the variable 'getp' (line 1295)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1295, 0), 'getp', getp)

# Assigning a Name to a Name (line 1333):

# Assigning a Name to a Name (line 1333):
# Getting the type of 'getp' (line 1333)
getp_6769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1333, 6), 'getp')
# Assigning a type to the variable 'get' (line 1333)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1333, 0), 'get', getp_6769)

@norecursion
def setp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setp'
    module_type_store = module_type_store.open_function_context('setp', 1336, 0, False)
    
    # Passed parameters checking function
    setp.stypy_localization = localization
    setp.stypy_type_of_self = None
    setp.stypy_type_store = module_type_store
    setp.stypy_function_name = 'setp'
    setp.stypy_param_names_list = ['obj']
    setp.stypy_varargs_param_name = 'args'
    setp.stypy_kwargs_param_name = 'kwargs'
    setp.stypy_call_defaults = defaults
    setp.stypy_call_varargs = varargs
    setp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setp', ['obj'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setp', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setp(...)' code ##################

    unicode_6770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1384, (-1)), 'unicode', u'\n    Set a property on an artist object.\n\n    matplotlib supports the use of :func:`setp` ("set property") and\n    :func:`getp` to set and get object properties, as well as to do\n    introspection on the object.  For example, to set the linestyle of a\n    line to be dashed, you can do::\n\n      >>> line, = plot([1,2,3])\n      >>> setp(line, linestyle=\'--\')\n\n    If you want to know the valid types of arguments, you can provide\n    the name of the property you want to set without a value::\n\n      >>> setp(line, \'linestyle\')\n          linestyle: [ \'-\' | \'--\' | \'-.\' | \':\' | \'steps\' | \'None\' ]\n\n    If you want to see all the properties that can be set, and their\n    possible values, you can do::\n\n      >>> setp(line)\n          ... long output listing omitted\n\n    You may specify another output file to `setp` if `sys.stdout` is not\n    acceptable for some reason using the `file` keyword-only argument::\n\n      >>> with fopen(\'output.log\') as f:\n      >>>     setp(line, file=f)\n\n    :func:`setp` operates on a single instance or a iterable of\n    instances. If you are in query mode introspecting the possible\n    values, only the first instance in the sequence is used. When\n    actually setting values, all the instances will be set.  e.g.,\n    suppose you have a list of two lines, the following will make both\n    lines thicker and red::\n\n      >>> x = arange(0,1.0,0.01)\n      >>> y1 = sin(2*pi*x)\n      >>> y2 = sin(4*pi*x)\n      >>> lines = plot(x, y1, x, y2)\n      >>> setp(lines, linewidth=2, color=\'r\')\n\n    :func:`setp` works with the MATLAB style string/value pairs or\n    with python kwargs.  For example, the following are equivalent::\n\n      >>> setp(lines, \'linewidth\', 2, \'color\', \'r\')  # MATLAB style\n      >>> setp(lines, linewidth=2, color=\'r\')        # python style\n    ')
    
    
    
    # Call to iterable(...): (line 1386)
    # Processing the call arguments (line 1386)
    # Getting the type of 'obj' (line 1386)
    obj_6773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 26), 'obj', False)
    # Processing the call keyword arguments (line 1386)
    kwargs_6774 = {}
    # Getting the type of 'cbook' (line 1386)
    cbook_6771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 11), 'cbook', False)
    # Obtaining the member 'iterable' of a type (line 1386)
    iterable_6772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1386, 11), cbook_6771, 'iterable')
    # Calling iterable(args, kwargs) (line 1386)
    iterable_call_result_6775 = invoke(stypy.reporting.localization.Localization(__file__, 1386, 11), iterable_6772, *[obj_6773], **kwargs_6774)
    
    # Applying the 'not' unary operator (line 1386)
    result_not__6776 = python_operator(stypy.reporting.localization.Localization(__file__, 1386, 7), 'not', iterable_call_result_6775)
    
    # Testing the type of an if condition (line 1386)
    if_condition_6777 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1386, 4), result_not__6776)
    # Assigning a type to the variable 'if_condition_6777' (line 1386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1386, 4), 'if_condition_6777', if_condition_6777)
    # SSA begins for if statement (line 1386)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 1387):
    
    # Assigning a List to a Name (line 1387):
    
    # Obtaining an instance of the builtin type 'list' (line 1387)
    list_6778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1387, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1387)
    # Adding element type (line 1387)
    # Getting the type of 'obj' (line 1387)
    obj_6779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 16), 'obj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1387, 15), list_6778, obj_6779)
    
    # Assigning a type to the variable 'objs' (line 1387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1387, 8), 'objs', list_6778)
    # SSA branch for the else part of an if statement (line 1386)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1389):
    
    # Assigning a Call to a Name (line 1389):
    
    # Call to list(...): (line 1389)
    # Processing the call arguments (line 1389)
    
    # Call to flatten(...): (line 1389)
    # Processing the call arguments (line 1389)
    # Getting the type of 'obj' (line 1389)
    obj_6783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 34), 'obj', False)
    # Processing the call keyword arguments (line 1389)
    kwargs_6784 = {}
    # Getting the type of 'cbook' (line 1389)
    cbook_6781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 20), 'cbook', False)
    # Obtaining the member 'flatten' of a type (line 1389)
    flatten_6782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 20), cbook_6781, 'flatten')
    # Calling flatten(args, kwargs) (line 1389)
    flatten_call_result_6785 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 20), flatten_6782, *[obj_6783], **kwargs_6784)
    
    # Processing the call keyword arguments (line 1389)
    kwargs_6786 = {}
    # Getting the type of 'list' (line 1389)
    list_6780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 15), 'list', False)
    # Calling list(args, kwargs) (line 1389)
    list_call_result_6787 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 15), list_6780, *[flatten_call_result_6785], **kwargs_6786)
    
    # Assigning a type to the variable 'objs' (line 1389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1389, 8), 'objs', list_call_result_6787)
    # SSA join for if statement (line 1386)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'objs' (line 1391)
    objs_6788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 11), 'objs')
    # Applying the 'not' unary operator (line 1391)
    result_not__6789 = python_operator(stypy.reporting.localization.Localization(__file__, 1391, 7), 'not', objs_6788)
    
    # Testing the type of an if condition (line 1391)
    if_condition_6790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1391, 4), result_not__6789)
    # Assigning a type to the variable 'if_condition_6790' (line 1391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1391, 4), 'if_condition_6790', if_condition_6790)
    # SSA begins for if statement (line 1391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 1391)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1394):
    
    # Assigning a Call to a Name (line 1394):
    
    # Call to ArtistInspector(...): (line 1394)
    # Processing the call arguments (line 1394)
    
    # Obtaining the type of the subscript
    int_6792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1394, 32), 'int')
    # Getting the type of 'objs' (line 1394)
    objs_6793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 27), 'objs', False)
    # Obtaining the member '__getitem__' of a type (line 1394)
    getitem___6794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1394, 27), objs_6793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1394)
    subscript_call_result_6795 = invoke(stypy.reporting.localization.Localization(__file__, 1394, 27), getitem___6794, int_6792)
    
    # Processing the call keyword arguments (line 1394)
    kwargs_6796 = {}
    # Getting the type of 'ArtistInspector' (line 1394)
    ArtistInspector_6791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 11), 'ArtistInspector', False)
    # Calling ArtistInspector(args, kwargs) (line 1394)
    ArtistInspector_call_result_6797 = invoke(stypy.reporting.localization.Localization(__file__, 1394, 11), ArtistInspector_6791, *[subscript_call_result_6795], **kwargs_6796)
    
    # Assigning a type to the variable 'insp' (line 1394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1394, 4), 'insp', ArtistInspector_call_result_6797)
    
    # Assigning a Dict to a Name (line 1397):
    
    # Assigning a Dict to a Name (line 1397):
    
    # Obtaining an instance of the builtin type 'dict' (line 1397)
    dict_6798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1397, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 1397)
    
    # Assigning a type to the variable 'printArgs' (line 1397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1397, 4), 'printArgs', dict_6798)
    
    
    unicode_6799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1398, 7), 'unicode', u'file')
    # Getting the type of 'kwargs' (line 1398)
    kwargs_6800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 17), 'kwargs')
    # Applying the binary operator 'in' (line 1398)
    result_contains_6801 = python_operator(stypy.reporting.localization.Localization(__file__, 1398, 7), 'in', unicode_6799, kwargs_6800)
    
    # Testing the type of an if condition (line 1398)
    if_condition_6802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1398, 4), result_contains_6801)
    # Assigning a type to the variable 'if_condition_6802' (line 1398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1398, 4), 'if_condition_6802', if_condition_6802)
    # SSA begins for if statement (line 1398)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 1399):
    
    # Assigning a Call to a Subscript (line 1399):
    
    # Call to pop(...): (line 1399)
    # Processing the call arguments (line 1399)
    unicode_6805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1399, 39), 'unicode', u'file')
    # Processing the call keyword arguments (line 1399)
    kwargs_6806 = {}
    # Getting the type of 'kwargs' (line 1399)
    kwargs_6803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 28), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 1399)
    pop_6804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1399, 28), kwargs_6803, 'pop')
    # Calling pop(args, kwargs) (line 1399)
    pop_call_result_6807 = invoke(stypy.reporting.localization.Localization(__file__, 1399, 28), pop_6804, *[unicode_6805], **kwargs_6806)
    
    # Getting the type of 'printArgs' (line 1399)
    printArgs_6808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 8), 'printArgs')
    unicode_6809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1399, 18), 'unicode', u'file')
    # Storing an element on a container (line 1399)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1399, 8), printArgs_6808, (unicode_6809, pop_call_result_6807))
    # SSA join for if statement (line 1398)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'kwargs' (line 1401)
    kwargs_6810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1401, 11), 'kwargs')
    # Applying the 'not' unary operator (line 1401)
    result_not__6811 = python_operator(stypy.reporting.localization.Localization(__file__, 1401, 7), 'not', kwargs_6810)
    
    
    
    # Call to len(...): (line 1401)
    # Processing the call arguments (line 1401)
    # Getting the type of 'args' (line 1401)
    args_6813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1401, 26), 'args', False)
    # Processing the call keyword arguments (line 1401)
    kwargs_6814 = {}
    # Getting the type of 'len' (line 1401)
    len_6812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1401, 22), 'len', False)
    # Calling len(args, kwargs) (line 1401)
    len_call_result_6815 = invoke(stypy.reporting.localization.Localization(__file__, 1401, 22), len_6812, *[args_6813], **kwargs_6814)
    
    int_6816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1401, 34), 'int')
    # Applying the binary operator '<' (line 1401)
    result_lt_6817 = python_operator(stypy.reporting.localization.Localization(__file__, 1401, 22), '<', len_call_result_6815, int_6816)
    
    # Applying the binary operator 'and' (line 1401)
    result_and_keyword_6818 = python_operator(stypy.reporting.localization.Localization(__file__, 1401, 7), 'and', result_not__6811, result_lt_6817)
    
    # Testing the type of an if condition (line 1401)
    if_condition_6819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1401, 4), result_and_keyword_6818)
    # Assigning a type to the variable 'if_condition_6819' (line 1401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1401, 4), 'if_condition_6819', if_condition_6819)
    # SSA begins for if statement (line 1401)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'args' (line 1402)
    args_6820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1402, 11), 'args')
    # Testing the type of an if condition (line 1402)
    if_condition_6821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1402, 8), args_6820)
    # Assigning a type to the variable 'if_condition_6821' (line 1402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1402, 8), 'if_condition_6821', if_condition_6821)
    # SSA begins for if statement (line 1402)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 1403)
    # Processing the call arguments (line 1403)
    
    # Call to pprint_setters(...): (line 1403)
    # Processing the call keyword arguments (line 1403)
    
    # Obtaining the type of the subscript
    int_6825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1403, 48), 'int')
    # Getting the type of 'args' (line 1403)
    args_6826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1403, 43), 'args', False)
    # Obtaining the member '__getitem__' of a type (line 1403)
    getitem___6827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1403, 43), args_6826, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1403)
    subscript_call_result_6828 = invoke(stypy.reporting.localization.Localization(__file__, 1403, 43), getitem___6827, int_6825)
    
    keyword_6829 = subscript_call_result_6828
    kwargs_6830 = {'prop': keyword_6829}
    # Getting the type of 'insp' (line 1403)
    insp_6823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1403, 18), 'insp', False)
    # Obtaining the member 'pprint_setters' of a type (line 1403)
    pprint_setters_6824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1403, 18), insp_6823, 'pprint_setters')
    # Calling pprint_setters(args, kwargs) (line 1403)
    pprint_setters_call_result_6831 = invoke(stypy.reporting.localization.Localization(__file__, 1403, 18), pprint_setters_6824, *[], **kwargs_6830)
    
    # Processing the call keyword arguments (line 1403)
    # Getting the type of 'printArgs' (line 1403)
    printArgs_6832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1403, 55), 'printArgs', False)
    kwargs_6833 = {'printArgs_6832': printArgs_6832}
    # Getting the type of 'print' (line 1403)
    print_6822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1403, 12), 'print', False)
    # Calling print(args, kwargs) (line 1403)
    print_call_result_6834 = invoke(stypy.reporting.localization.Localization(__file__, 1403, 12), print_6822, *[pprint_setters_call_result_6831], **kwargs_6833)
    
    # SSA branch for the else part of an if statement (line 1402)
    module_type_store.open_ssa_branch('else')
    
    # Call to print(...): (line 1405)
    # Processing the call arguments (line 1405)
    
    # Call to join(...): (line 1405)
    # Processing the call arguments (line 1405)
    
    # Call to pprint_setters(...): (line 1405)
    # Processing the call keyword arguments (line 1405)
    kwargs_6840 = {}
    # Getting the type of 'insp' (line 1405)
    insp_6838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 28), 'insp', False)
    # Obtaining the member 'pprint_setters' of a type (line 1405)
    pprint_setters_6839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1405, 28), insp_6838, 'pprint_setters')
    # Calling pprint_setters(args, kwargs) (line 1405)
    pprint_setters_call_result_6841 = invoke(stypy.reporting.localization.Localization(__file__, 1405, 28), pprint_setters_6839, *[], **kwargs_6840)
    
    # Processing the call keyword arguments (line 1405)
    kwargs_6842 = {}
    unicode_6836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1405, 18), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 1405)
    join_6837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1405, 18), unicode_6836, 'join')
    # Calling join(args, kwargs) (line 1405)
    join_call_result_6843 = invoke(stypy.reporting.localization.Localization(__file__, 1405, 18), join_6837, *[pprint_setters_call_result_6841], **kwargs_6842)
    
    # Processing the call keyword arguments (line 1405)
    # Getting the type of 'printArgs' (line 1405)
    printArgs_6844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 54), 'printArgs', False)
    kwargs_6845 = {'printArgs_6844': printArgs_6844}
    # Getting the type of 'print' (line 1405)
    print_6835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 12), 'print', False)
    # Calling print(args, kwargs) (line 1405)
    print_call_result_6846 = invoke(stypy.reporting.localization.Localization(__file__, 1405, 12), print_6835, *[join_call_result_6843], **kwargs_6845)
    
    # SSA join for if statement (line 1402)
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 1406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1406, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 1401)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to len(...): (line 1408)
    # Processing the call arguments (line 1408)
    # Getting the type of 'args' (line 1408)
    args_6848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 11), 'args', False)
    # Processing the call keyword arguments (line 1408)
    kwargs_6849 = {}
    # Getting the type of 'len' (line 1408)
    len_6847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 7), 'len', False)
    # Calling len(args, kwargs) (line 1408)
    len_call_result_6850 = invoke(stypy.reporting.localization.Localization(__file__, 1408, 7), len_6847, *[args_6848], **kwargs_6849)
    
    int_6851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1408, 19), 'int')
    # Applying the binary operator '%' (line 1408)
    result_mod_6852 = python_operator(stypy.reporting.localization.Localization(__file__, 1408, 7), '%', len_call_result_6850, int_6851)
    
    # Testing the type of an if condition (line 1408)
    if_condition_6853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1408, 4), result_mod_6852)
    # Assigning a type to the variable 'if_condition_6853' (line 1408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1408, 4), 'if_condition_6853', if_condition_6853)
    # SSA begins for if statement (line 1408)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1409)
    # Processing the call arguments (line 1409)
    unicode_6855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1409, 25), 'unicode', u'The set args must be string, value pairs')
    # Processing the call keyword arguments (line 1409)
    kwargs_6856 = {}
    # Getting the type of 'ValueError' (line 1409)
    ValueError_6854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1409, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1409)
    ValueError_call_result_6857 = invoke(stypy.reporting.localization.Localization(__file__, 1409, 14), ValueError_6854, *[unicode_6855], **kwargs_6856)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1409, 8), ValueError_call_result_6857, 'raise parameter', BaseException)
    # SSA join for if statement (line 1408)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1412):
    
    # Assigning a Call to a Name (line 1412):
    
    # Call to OrderedDict(...): (line 1412)
    # Processing the call keyword arguments (line 1412)
    kwargs_6859 = {}
    # Getting the type of 'OrderedDict' (line 1412)
    OrderedDict_6858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1412, 15), 'OrderedDict', False)
    # Calling OrderedDict(args, kwargs) (line 1412)
    OrderedDict_call_result_6860 = invoke(stypy.reporting.localization.Localization(__file__, 1412, 15), OrderedDict_6858, *[], **kwargs_6859)
    
    # Assigning a type to the variable 'funcvals' (line 1412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1412, 4), 'funcvals', OrderedDict_call_result_6860)
    
    
    # Call to range(...): (line 1413)
    # Processing the call arguments (line 1413)
    int_6862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1413, 19), 'int')
    
    # Call to len(...): (line 1413)
    # Processing the call arguments (line 1413)
    # Getting the type of 'args' (line 1413)
    args_6864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1413, 26), 'args', False)
    # Processing the call keyword arguments (line 1413)
    kwargs_6865 = {}
    # Getting the type of 'len' (line 1413)
    len_6863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1413, 22), 'len', False)
    # Calling len(args, kwargs) (line 1413)
    len_call_result_6866 = invoke(stypy.reporting.localization.Localization(__file__, 1413, 22), len_6863, *[args_6864], **kwargs_6865)
    
    int_6867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1413, 34), 'int')
    # Applying the binary operator '-' (line 1413)
    result_sub_6868 = python_operator(stypy.reporting.localization.Localization(__file__, 1413, 22), '-', len_call_result_6866, int_6867)
    
    int_6869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1413, 37), 'int')
    # Processing the call keyword arguments (line 1413)
    kwargs_6870 = {}
    # Getting the type of 'range' (line 1413)
    range_6861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1413, 13), 'range', False)
    # Calling range(args, kwargs) (line 1413)
    range_call_result_6871 = invoke(stypy.reporting.localization.Localization(__file__, 1413, 13), range_6861, *[int_6862, result_sub_6868, int_6869], **kwargs_6870)
    
    # Testing the type of a for loop iterable (line 1413)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1413, 4), range_call_result_6871)
    # Getting the type of the for loop variable (line 1413)
    for_loop_var_6872 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1413, 4), range_call_result_6871)
    # Assigning a type to the variable 'i' (line 1413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1413, 4), 'i', for_loop_var_6872)
    # SSA begins for a for statement (line 1413)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 1414):
    
    # Assigning a Subscript to a Subscript (line 1414):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1414)
    i_6873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 33), 'i')
    int_6874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1414, 37), 'int')
    # Applying the binary operator '+' (line 1414)
    result_add_6875 = python_operator(stypy.reporting.localization.Localization(__file__, 1414, 33), '+', i_6873, int_6874)
    
    # Getting the type of 'args' (line 1414)
    args_6876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 28), 'args')
    # Obtaining the member '__getitem__' of a type (line 1414)
    getitem___6877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 28), args_6876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1414)
    subscript_call_result_6878 = invoke(stypy.reporting.localization.Localization(__file__, 1414, 28), getitem___6877, result_add_6875)
    
    # Getting the type of 'funcvals' (line 1414)
    funcvals_6879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 8), 'funcvals')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1414)
    i_6880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 22), 'i')
    # Getting the type of 'args' (line 1414)
    args_6881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 17), 'args')
    # Obtaining the member '__getitem__' of a type (line 1414)
    getitem___6882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 17), args_6881, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1414)
    subscript_call_result_6883 = invoke(stypy.reporting.localization.Localization(__file__, 1414, 17), getitem___6882, i_6880)
    
    # Storing an element on a container (line 1414)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1414, 8), funcvals_6879, (subscript_call_result_6883, subscript_call_result_6878))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 1416):
    
    # Assigning a ListComp to a Name (line 1416):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'objs' (line 1416)
    objs_6889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 39), 'objs')
    comprehension_6890 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1416, 11), objs_6889)
    # Assigning a type to the variable 'o' (line 1416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1416, 11), 'o', comprehension_6890)
    
    # Call to update(...): (line 1416)
    # Processing the call arguments (line 1416)
    # Getting the type of 'funcvals' (line 1416)
    funcvals_6886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 20), 'funcvals', False)
    # Processing the call keyword arguments (line 1416)
    kwargs_6887 = {}
    # Getting the type of 'o' (line 1416)
    o_6884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 11), 'o', False)
    # Obtaining the member 'update' of a type (line 1416)
    update_6885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1416, 11), o_6884, 'update')
    # Calling update(args, kwargs) (line 1416)
    update_call_result_6888 = invoke(stypy.reporting.localization.Localization(__file__, 1416, 11), update_6885, *[funcvals_6886], **kwargs_6887)
    
    list_6891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1416, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1416, 11), list_6891, update_call_result_6888)
    # Assigning a type to the variable 'ret' (line 1416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1416, 4), 'ret', list_6891)
    
    # Call to extend(...): (line 1417)
    # Processing the call arguments (line 1417)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'objs' (line 1417)
    objs_6899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 41), 'objs', False)
    comprehension_6900 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1417, 16), objs_6899)
    # Assigning a type to the variable 'o' (line 1417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1417, 16), 'o', comprehension_6900)
    
    # Call to set(...): (line 1417)
    # Processing the call keyword arguments (line 1417)
    # Getting the type of 'kwargs' (line 1417)
    kwargs_6896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 24), 'kwargs', False)
    kwargs_6897 = {'kwargs_6896': kwargs_6896}
    # Getting the type of 'o' (line 1417)
    o_6894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 16), 'o', False)
    # Obtaining the member 'set' of a type (line 1417)
    set_6895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1417, 16), o_6894, 'set')
    # Calling set(args, kwargs) (line 1417)
    set_call_result_6898 = invoke(stypy.reporting.localization.Localization(__file__, 1417, 16), set_6895, *[], **kwargs_6897)
    
    list_6901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1417, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1417, 16), list_6901, set_call_result_6898)
    # Processing the call keyword arguments (line 1417)
    kwargs_6902 = {}
    # Getting the type of 'ret' (line 1417)
    ret_6892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 4), 'ret', False)
    # Obtaining the member 'extend' of a type (line 1417)
    extend_6893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1417, 4), ret_6892, 'extend')
    # Calling extend(args, kwargs) (line 1417)
    extend_call_result_6903 = invoke(stypy.reporting.localization.Localization(__file__, 1417, 4), extend_6893, *[list_6901], **kwargs_6902)
    
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to flatten(...): (line 1418)
    # Processing the call arguments (line 1418)
    # Getting the type of 'ret' (line 1418)
    ret_6907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1418, 37), 'ret', False)
    # Processing the call keyword arguments (line 1418)
    kwargs_6908 = {}
    # Getting the type of 'cbook' (line 1418)
    cbook_6905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1418, 23), 'cbook', False)
    # Obtaining the member 'flatten' of a type (line 1418)
    flatten_6906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1418, 23), cbook_6905, 'flatten')
    # Calling flatten(args, kwargs) (line 1418)
    flatten_call_result_6909 = invoke(stypy.reporting.localization.Localization(__file__, 1418, 23), flatten_6906, *[ret_6907], **kwargs_6908)
    
    comprehension_6910 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1418, 12), flatten_call_result_6909)
    # Assigning a type to the variable 'x' (line 1418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1418, 12), 'x', comprehension_6910)
    # Getting the type of 'x' (line 1418)
    x_6904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1418, 12), 'x')
    list_6911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1418, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1418, 12), list_6911, x_6904)
    # Assigning a type to the variable 'stypy_return_type' (line 1418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1418, 4), 'stypy_return_type', list_6911)
    
    # ################# End of 'setp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setp' in the type store
    # Getting the type of 'stypy_return_type' (line 1336)
    stypy_return_type_6912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1336, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6912)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setp'
    return stypy_return_type_6912

# Assigning a type to the variable 'setp' (line 1336)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1336, 0), 'setp', setp)

@norecursion
def kwdoc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'kwdoc'
    module_type_store = module_type_store.open_function_context('kwdoc', 1421, 0, False)
    
    # Passed parameters checking function
    kwdoc.stypy_localization = localization
    kwdoc.stypy_type_of_self = None
    kwdoc.stypy_type_store = module_type_store
    kwdoc.stypy_function_name = 'kwdoc'
    kwdoc.stypy_param_names_list = ['a']
    kwdoc.stypy_varargs_param_name = None
    kwdoc.stypy_kwargs_param_name = None
    kwdoc.stypy_call_defaults = defaults
    kwdoc.stypy_call_varargs = varargs
    kwdoc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'kwdoc', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'kwdoc', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'kwdoc(...)' code ##################

    
    # Assigning a Subscript to a Name (line 1422):
    
    # Assigning a Subscript to a Name (line 1422):
    
    # Obtaining the type of the subscript
    unicode_6913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1422, 35), 'unicode', u'docstring.hardcopy')
    # Getting the type of 'matplotlib' (line 1422)
    matplotlib_6914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1422, 15), 'matplotlib')
    # Obtaining the member 'rcParams' of a type (line 1422)
    rcParams_6915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1422, 15), matplotlib_6914, 'rcParams')
    # Obtaining the member '__getitem__' of a type (line 1422)
    getitem___6916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1422, 15), rcParams_6915, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1422)
    subscript_call_result_6917 = invoke(stypy.reporting.localization.Localization(__file__, 1422, 15), getitem___6916, unicode_6913)
    
    # Assigning a type to the variable 'hardcopy' (line 1422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1422, 4), 'hardcopy', subscript_call_result_6917)
    
    # Getting the type of 'hardcopy' (line 1423)
    hardcopy_6918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1423, 7), 'hardcopy')
    # Testing the type of an if condition (line 1423)
    if_condition_6919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1423, 4), hardcopy_6918)
    # Assigning a type to the variable 'if_condition_6919' (line 1423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1423, 4), 'if_condition_6919', if_condition_6919)
    # SSA begins for if statement (line 1423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to join(...): (line 1424)
    # Processing the call arguments (line 1424)
    
    # Call to pprint_setters_rest(...): (line 1424)
    # Processing the call keyword arguments (line 1424)
    int_6927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1425, 38), 'int')
    keyword_6928 = int_6927
    kwargs_6929 = {'leadingspace': keyword_6928}
    
    # Call to ArtistInspector(...): (line 1424)
    # Processing the call arguments (line 1424)
    # Getting the type of 'a' (line 1424)
    a_6923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1424, 41), 'a', False)
    # Processing the call keyword arguments (line 1424)
    kwargs_6924 = {}
    # Getting the type of 'ArtistInspector' (line 1424)
    ArtistInspector_6922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1424, 25), 'ArtistInspector', False)
    # Calling ArtistInspector(args, kwargs) (line 1424)
    ArtistInspector_call_result_6925 = invoke(stypy.reporting.localization.Localization(__file__, 1424, 25), ArtistInspector_6922, *[a_6923], **kwargs_6924)
    
    # Obtaining the member 'pprint_setters_rest' of a type (line 1424)
    pprint_setters_rest_6926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1424, 25), ArtistInspector_call_result_6925, 'pprint_setters_rest')
    # Calling pprint_setters_rest(args, kwargs) (line 1424)
    pprint_setters_rest_call_result_6930 = invoke(stypy.reporting.localization.Localization(__file__, 1424, 25), pprint_setters_rest_6926, *[], **kwargs_6929)
    
    # Processing the call keyword arguments (line 1424)
    kwargs_6931 = {}
    unicode_6920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1424, 15), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 1424)
    join_6921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1424, 15), unicode_6920, 'join')
    # Calling join(args, kwargs) (line 1424)
    join_call_result_6932 = invoke(stypy.reporting.localization.Localization(__file__, 1424, 15), join_6921, *[pprint_setters_rest_call_result_6930], **kwargs_6931)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1424, 8), 'stypy_return_type', join_call_result_6932)
    # SSA branch for the else part of an if statement (line 1423)
    module_type_store.open_ssa_branch('else')
    
    # Call to join(...): (line 1427)
    # Processing the call arguments (line 1427)
    
    # Call to pprint_setters(...): (line 1427)
    # Processing the call keyword arguments (line 1427)
    int_6940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1427, 72), 'int')
    keyword_6941 = int_6940
    kwargs_6942 = {'leadingspace': keyword_6941}
    
    # Call to ArtistInspector(...): (line 1427)
    # Processing the call arguments (line 1427)
    # Getting the type of 'a' (line 1427)
    a_6936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1427, 41), 'a', False)
    # Processing the call keyword arguments (line 1427)
    kwargs_6937 = {}
    # Getting the type of 'ArtistInspector' (line 1427)
    ArtistInspector_6935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1427, 25), 'ArtistInspector', False)
    # Calling ArtistInspector(args, kwargs) (line 1427)
    ArtistInspector_call_result_6938 = invoke(stypy.reporting.localization.Localization(__file__, 1427, 25), ArtistInspector_6935, *[a_6936], **kwargs_6937)
    
    # Obtaining the member 'pprint_setters' of a type (line 1427)
    pprint_setters_6939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1427, 25), ArtistInspector_call_result_6938, 'pprint_setters')
    # Calling pprint_setters(args, kwargs) (line 1427)
    pprint_setters_call_result_6943 = invoke(stypy.reporting.localization.Localization(__file__, 1427, 25), pprint_setters_6939, *[], **kwargs_6942)
    
    # Processing the call keyword arguments (line 1427)
    kwargs_6944 = {}
    unicode_6933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1427, 15), 'unicode', u'\n')
    # Obtaining the member 'join' of a type (line 1427)
    join_6934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1427, 15), unicode_6933, 'join')
    # Calling join(args, kwargs) (line 1427)
    join_call_result_6945 = invoke(stypy.reporting.localization.Localization(__file__, 1427, 15), join_6934, *[pprint_setters_call_result_6943], **kwargs_6944)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1427, 8), 'stypy_return_type', join_call_result_6945)
    # SSA join for if statement (line 1423)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'kwdoc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kwdoc' in the type store
    # Getting the type of 'stypy_return_type' (line 1421)
    stypy_return_type_6946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6946)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kwdoc'
    return stypy_return_type_6946

# Assigning a type to the variable 'kwdoc' (line 1421)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1421, 0), 'kwdoc', kwdoc)

# Call to update(...): (line 1429)
# Processing the call keyword arguments (line 1429)

# Call to kwdoc(...): (line 1429)
# Processing the call arguments (line 1429)
# Getting the type of 'Artist' (line 1429)
Artist_6951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 38), 'Artist', False)
# Processing the call keyword arguments (line 1429)
kwargs_6952 = {}
# Getting the type of 'kwdoc' (line 1429)
kwdoc_6950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 32), 'kwdoc', False)
# Calling kwdoc(args, kwargs) (line 1429)
kwdoc_call_result_6953 = invoke(stypy.reporting.localization.Localization(__file__, 1429, 32), kwdoc_6950, *[Artist_6951], **kwargs_6952)

keyword_6954 = kwdoc_call_result_6953
kwargs_6955 = {'Artist': keyword_6954}
# Getting the type of 'docstring' (line 1429)
docstring_6947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 0), 'docstring', False)
# Obtaining the member 'interpd' of a type (line 1429)
interpd_6948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1429, 0), docstring_6947, 'interpd')
# Obtaining the member 'update' of a type (line 1429)
update_6949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1429, 0), interpd_6948, 'update')
# Calling update(args, kwargs) (line 1429)
update_call_result_6956 = invoke(stypy.reporting.localization.Localization(__file__, 1429, 0), update_6949, *[], **kwargs_6955)


# Assigning a Str to a Name (line 1431):

# Assigning a Str to a Name (line 1431):
unicode_6957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1432, (-1)), 'unicode', u'{0} has been deprecated in mpl 1.5, please use the\naxes property.  A removal date has not been set.')
# Assigning a type to the variable '_get_axes_msg' (line 1431)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1431, 0), '_get_axes_msg', unicode_6957)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
