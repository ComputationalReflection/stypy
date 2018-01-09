
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: The legend module defines the Legend class, which is responsible for
3: drawing legends associated with axes and/or figures.
4: 
5: .. important::
6: 
7:     It is unlikely that you would ever create a Legend instance manually.
8:     Most users would normally create a legend via the
9:     :meth:`~matplotlib.axes.Axes.legend` function. For more details on legends
10:     there is also a :ref:`legend guide
11:     <sphx_glr_tutorials_intermediate_legend_guide.py>`.
12: 
13: The Legend class can be considered as a container of legend handles
14: and legend texts. Creation of corresponding legend handles from the
15: plot elements in the axes or figures (e.g., lines, patches, etc.) are
16: specified by the handler map, which defines the mapping between the
17: plot elements and the legend handlers to be used (the default legend
18: handlers are defined in the :mod:`~matplotlib.legend_handler` module).
19: Note that not all kinds of artist are supported by the legend yet by default
20: but it is possible to extend the legend handler's capabilities to support
21: arbitrary objects. See the :ref:`legend guide
22: <sphx_glr_tutorials_intermediate_legend_guide.py>` for more information.
23: 
24: '''
25: from __future__ import (absolute_import, division, print_function,
26:                         unicode_literals)
27: 
28: import six
29: from six.moves import xrange
30: 
31: import warnings
32: 
33: import numpy as np
34: 
35: from matplotlib import rcParams
36: from matplotlib.artist import Artist, allow_rasterization
37: from matplotlib.cbook import silent_list, is_hashable
38: from matplotlib.font_manager import FontProperties
39: from matplotlib.lines import Line2D
40: from matplotlib.patches import Patch, Rectangle, Shadow, FancyBboxPatch
41: from matplotlib.collections import (LineCollection, RegularPolyCollection,
42:                                     CircleCollection, PathCollection,
43:                                     PolyCollection)
44: from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
45: from matplotlib.transforms import BboxTransformTo, BboxTransformFrom
46: 
47: from matplotlib.offsetbox import HPacker, VPacker, TextArea, DrawingArea
48: from matplotlib.offsetbox import DraggableOffsetBox
49: 
50: from matplotlib.container import ErrorbarContainer, BarContainer, StemContainer
51: from . import legend_handler
52: 
53: 
54: class DraggableLegend(DraggableOffsetBox):
55:     def __init__(self, legend, use_blit=False, update="loc"):
56:         '''
57:         update : If "loc", update *loc* parameter of
58:                  legend upon finalizing. If "bbox", update
59:                  *bbox_to_anchor* parameter.
60:         '''
61:         self.legend = legend
62: 
63:         if update in ["loc", "bbox"]:
64:             self._update = update
65:         else:
66:             raise ValueError("update parameter '%s' is not supported." %
67:                              update)
68: 
69:         DraggableOffsetBox.__init__(self, legend, legend._legend_box,
70:                                     use_blit=use_blit)
71: 
72:     def artist_picker(self, legend, evt):
73:         return self.legend.contains(evt)
74: 
75:     def finalize_offset(self):
76:         loc_in_canvas = self.get_loc_in_canvas()
77: 
78:         if self._update == "loc":
79:             self._update_loc(loc_in_canvas)
80:         elif self._update == "bbox":
81:             self._update_bbox_to_anchor(loc_in_canvas)
82:         else:
83:             raise RuntimeError("update parameter '%s' is not supported." %
84:                                self.update)
85: 
86:     def _update_loc(self, loc_in_canvas):
87:         bbox = self.legend.get_bbox_to_anchor()
88: 
89:         # if bbox has zero width or height, the transformation is
90:         # ill-defined. Fall back to the defaul bbox_to_anchor.
91:         if bbox.width == 0 or bbox.height == 0:
92:             self.legend.set_bbox_to_anchor(None)
93:             bbox = self.legend.get_bbox_to_anchor()
94: 
95:         _bbox_transform = BboxTransformFrom(bbox)
96:         self.legend._loc = tuple(
97:             _bbox_transform.transform_point(loc_in_canvas)
98:         )
99: 
100:     def _update_bbox_to_anchor(self, loc_in_canvas):
101: 
102:         tr = self.legend.axes.transAxes
103:         loc_in_bbox = tr.transform_point(loc_in_canvas)
104: 
105:         self.legend.set_bbox_to_anchor(loc_in_bbox)
106: 
107: 
108: class Legend(Artist):
109:     '''
110:     Place a legend on the axes at location loc.  Labels are a
111:     sequence of strings and loc can be a string or an integer
112:     specifying the legend location
113: 
114:     The location codes are::
115: 
116:       'best'         : 0, (only implemented for axes legends)
117:       'upper right'  : 1,
118:       'upper left'   : 2,
119:       'lower left'   : 3,
120:       'lower right'  : 4,
121:       'right'        : 5, (same as 'center right', for back-compatibility)
122:       'center left'  : 6,
123:       'center right' : 7,
124:       'lower center' : 8,
125:       'upper center' : 9,
126:       'center'       : 10,
127: 
128:     loc can be a tuple of the normalized coordinate values with
129:     respect its parent.
130: 
131:     '''
132:     codes = {'best':         0,  # only implemented for axes legends
133:              'upper right':  1,
134:              'upper left':   2,
135:              'lower left':   3,
136:              'lower right':  4,
137:              'right':        5,
138:              'center left':  6,
139:              'center right': 7,
140:              'lower center': 8,
141:              'upper center': 9,
142:              'center':       10,
143:              }
144: 
145:     zorder = 5
146: 
147:     def __str__(self):
148:         return "Legend"
149: 
150:     def __init__(self, parent, handles, labels,
151:                  loc=None,
152:                  numpoints=None,    # the number of points in the legend line
153:                  markerscale=None,  # the relative size of legend markers
154:                                     # vs. original
155:                  markerfirst=True,  # controls ordering (left-to-right) of
156:                                     # legend marker and label
157:                  scatterpoints=None,    # number of scatter points
158:                  scatteryoffsets=None,
159:                  prop=None,          # properties for the legend texts
160:                  fontsize=None,        # keyword to set font size directly
161: 
162:                  # spacing & pad defined as a fraction of the font-size
163:                  borderpad=None,      # the whitespace inside the legend border
164:                  labelspacing=None,   # the vertical space between the legend
165:                                       # entries
166:                  handlelength=None,   # the length of the legend handles
167:                  handleheight=None,   # the height of the legend handles
168:                  handletextpad=None,  # the pad between the legend handle
169:                                       # and text
170:                  borderaxespad=None,  # the pad between the axes and legend
171:                                       # border
172:                  columnspacing=None,  # spacing between columns
173: 
174:                  ncol=1,     # number of columns
175:                  mode=None,  # mode for horizontal distribution of columns.
176:                              # None, "expand"
177: 
178:                  fancybox=None,  # True use a fancy box, false use a rounded
179:                                  # box, none use rc
180:                  shadow=None,
181:                  title=None,  # set a title for the legend
182: 
183:                  framealpha=None,  # set frame alpha
184:                  edgecolor=None,  # frame patch edgecolor
185:                  facecolor=None,  # frame patch facecolor
186: 
187:                  bbox_to_anchor=None,  # bbox that the legend will be anchored.
188:                  bbox_transform=None,  # transform for the bbox
189:                  frameon=None,  # draw frame
190:                  handler_map=None,
191:                  ):
192:         '''
193:         - *parent*: the artist that contains the legend
194:         - *handles*: a list of artists (lines, patches) to be added to the
195:                       legend
196:         - *labels*: a list of strings to label the legend
197: 
198:         Optional keyword arguments:
199: 
200:         ================   ====================================================
201:         Keyword            Description
202:         ================   ====================================================
203:         loc                Location code string, or tuple (see below).
204:         prop               the font property
205:         fontsize           the font size (used only if prop is not specified)
206:         markerscale        the relative size of legend markers vs. original
207:         markerfirst        If True (default), marker is to left of the label.
208:         numpoints          the number of points in the legend for line
209:         scatterpoints      the number of points in the legend for scatter plot
210:         scatteryoffsets    a list of yoffsets for scatter symbols in legend
211:         frameon            If True, draw the legend on a patch (frame).
212:         fancybox           If True, draw the frame with a round fancybox.
213:         shadow             If True, draw a shadow behind legend.
214:         framealpha         Transparency of the frame.
215:         edgecolor          Frame edgecolor.
216:         facecolor          Frame facecolor.
217:         ncol               number of columns
218:         borderpad          the fractional whitespace inside the legend border
219:         labelspacing       the vertical space between the legend entries
220:         handlelength       the length of the legend handles
221:         handleheight       the height of the legend handles
222:         handletextpad      the pad between the legend handle and text
223:         borderaxespad      the pad between the axes and legend border
224:         columnspacing      the spacing between columns
225:         title              the legend title
226:         bbox_to_anchor     the bbox that the legend will be anchored.
227:         bbox_transform     the transform for the bbox. transAxes if None.
228:         ================   ====================================================
229: 
230: 
231:         The pad and spacing parameters are measured in font-size units.  e.g.,
232:         a fontsize of 10 points and a handlelength=5 implies a handlelength of
233:         50 points.  Values from rcParams will be used if None.
234: 
235:         Users can specify any arbitrary location for the legend using the
236:         *bbox_to_anchor* keyword argument. bbox_to_anchor can be an instance
237:         of BboxBase(or its derivatives) or a tuple of 2 or 4 floats.
238:         See :meth:`set_bbox_to_anchor` for more detail.
239: 
240:         The legend location can be specified by setting *loc* with a tuple of
241:         2 floats, which is interpreted as the lower-left corner of the legend
242:         in the normalized axes coordinate.
243:         '''
244:         # local import only to avoid circularity
245:         from matplotlib.axes import Axes
246:         from matplotlib.figure import Figure
247: 
248:         Artist.__init__(self)
249: 
250:         if prop is None:
251:             if fontsize is not None:
252:                 self.prop = FontProperties(size=fontsize)
253:             else:
254:                 self.prop = FontProperties(size=rcParams["legend.fontsize"])
255:         elif isinstance(prop, dict):
256:             self.prop = FontProperties(**prop)
257:             if "size" not in prop:
258:                 self.prop.set_size(rcParams["legend.fontsize"])
259:         else:
260:             self.prop = prop
261: 
262:         self._fontsize = self.prop.get_size_in_points()
263: 
264:         self.texts = []
265:         self.legendHandles = []
266:         self._legend_title_box = None
267: 
268:         #: A dictionary with the extra handler mappings for this Legend
269:         #: instance.
270:         self._custom_handler_map = handler_map
271: 
272:         locals_view = locals()
273:         for name in ["numpoints", "markerscale", "shadow", "columnspacing",
274:                      "scatterpoints", "handleheight", 'borderpad',
275:                      'labelspacing', 'handlelength', 'handletextpad',
276:                      'borderaxespad']:
277:             if locals_view[name] is None:
278:                 value = rcParams["legend." + name]
279:             else:
280:                 value = locals_view[name]
281:             setattr(self, name, value)
282:         del locals_view
283: 
284:         handles = list(handles)
285:         if len(handles) < 2:
286:             ncol = 1
287:         self._ncol = ncol
288: 
289:         if self.numpoints <= 0:
290:             raise ValueError("numpoints must be > 0; it was %d" % numpoints)
291: 
292:         # introduce y-offset for handles of the scatter plot
293:         if scatteryoffsets is None:
294:             self._scatteryoffsets = np.array([3. / 8., 4. / 8., 2.5 / 8.])
295:         else:
296:             self._scatteryoffsets = np.asarray(scatteryoffsets)
297:         reps = self.scatterpoints // len(self._scatteryoffsets) + 1
298:         self._scatteryoffsets = np.tile(self._scatteryoffsets,
299:                                         reps)[:self.scatterpoints]
300: 
301:         # _legend_box is an OffsetBox instance that contains all
302:         # legend items and will be initialized from _init_legend_box()
303:         # method.
304:         self._legend_box = None
305: 
306:         if isinstance(parent, Axes):
307:             self.isaxes = True
308:             self.axes = parent
309:             self.set_figure(parent.figure)
310:         elif isinstance(parent, Figure):
311:             self.isaxes = False
312:             self.set_figure(parent)
313:         else:
314:             raise TypeError("Legend needs either Axes or Figure as parent")
315:         self.parent = parent
316: 
317:         if loc is None:
318:             loc = rcParams["legend.loc"]
319:             if not self.isaxes and loc in [0, 'best']:
320:                 loc = 'upper right'
321:         if isinstance(loc, six.string_types):
322:             if loc not in self.codes:
323:                 if self.isaxes:
324:                     warnings.warn('Unrecognized location "%s". Falling back '
325:                                   'on "best"; valid locations are\n\t%s\n'
326:                                   % (loc, '\n\t'.join(self.codes)))
327:                     loc = 0
328:                 else:
329:                     warnings.warn('Unrecognized location "%s". Falling back '
330:                                   'on "upper right"; '
331:                                   'valid locations are\n\t%s\n'
332:                                   % (loc, '\n\t'.join(self.codes)))
333:                     loc = 1
334:             else:
335:                 loc = self.codes[loc]
336:         if not self.isaxes and loc == 0:
337:             warnings.warn('Automatic legend placement (loc="best") not '
338:                           'implemented for figure legend. '
339:                           'Falling back on "upper right".')
340:             loc = 1
341: 
342:         self._mode = mode
343:         self.set_bbox_to_anchor(bbox_to_anchor, bbox_transform)
344: 
345:         # We use FancyBboxPatch to draw a legend frame. The location
346:         # and size of the box will be updated during the drawing time.
347: 
348:         if facecolor is None:
349:             facecolor = rcParams["legend.facecolor"]
350:         if facecolor == 'inherit':
351:             facecolor = rcParams["axes.facecolor"]
352: 
353:         if edgecolor is None:
354:             edgecolor = rcParams["legend.edgecolor"]
355:         if edgecolor == 'inherit':
356:             edgecolor = rcParams["axes.edgecolor"]
357: 
358:         self.legendPatch = FancyBboxPatch(
359:             xy=(0.0, 0.0), width=1., height=1.,
360:             facecolor=facecolor,
361:             edgecolor=edgecolor,
362:             mutation_scale=self._fontsize,
363:             snap=True
364:             )
365: 
366:         # The width and height of the legendPatch will be set (in the
367:         # draw()) to the length that includes the padding. Thus we set
368:         # pad=0 here.
369:         if fancybox is None:
370:             fancybox = rcParams["legend.fancybox"]
371: 
372:         if fancybox:
373:             self.legendPatch.set_boxstyle("round", pad=0,
374:                                           rounding_size=0.2)
375:         else:
376:             self.legendPatch.set_boxstyle("square", pad=0)
377: 
378:         self._set_artist_props(self.legendPatch)
379: 
380:         self._drawFrame = frameon
381:         if frameon is None:
382:             self._drawFrame = rcParams["legend.frameon"]
383: 
384:         # init with null renderer
385:         self._init_legend_box(handles, labels, markerfirst)
386: 
387:         # If shadow is activated use framealpha if not
388:         # explicitly passed. See Issue 8943
389:         if framealpha is None:
390:             if shadow:
391:                 self.get_frame().set_alpha(1)
392:             else:
393:                 self.get_frame().set_alpha(rcParams["legend.framealpha"])
394:         else:
395:             self.get_frame().set_alpha(framealpha)
396: 
397:         self._loc = loc
398:         self.set_title(title)
399:         self._last_fontsize_points = self._fontsize
400:         self._draggable = None
401: 
402:     def _set_artist_props(self, a):
403:         '''
404:         set the boilerplate props for artists added to axes
405:         '''
406:         a.set_figure(self.figure)
407:         if self.isaxes:
408:             # a.set_axes(self.axes)
409:             a.axes = self.axes
410: 
411:         a.set_transform(self.get_transform())
412: 
413:     def _set_loc(self, loc):
414:         # find_offset function will be provided to _legend_box and
415:         # _legend_box will draw itself at the location of the return
416:         # value of the find_offset.
417:         self._loc_real = loc
418:         self.stale = True
419: 
420:     def _get_loc(self):
421:         return self._loc_real
422: 
423:     _loc = property(_get_loc, _set_loc)
424: 
425:     def _findoffset(self, width, height, xdescent, ydescent, renderer):
426:         "Helper function to locate the legend"
427: 
428:         if self._loc == 0:  # "best".
429:             x, y = self._find_best_position(width, height, renderer)
430:         elif self._loc in Legend.codes.values():  # Fixed location.
431:             bbox = Bbox.from_bounds(0, 0, width, height)
432:             x, y = self._get_anchored_bbox(self._loc, bbox,
433:                                            self.get_bbox_to_anchor(),
434:                                            renderer)
435:         else:  # Axes or figure coordinates.
436:             fx, fy = self._loc
437:             bbox = self.get_bbox_to_anchor()
438:             x, y = bbox.x0 + bbox.width * fx, bbox.y0 + bbox.height * fy
439: 
440:         return x + xdescent, y + ydescent
441: 
442:     @allow_rasterization
443:     def draw(self, renderer):
444:         "Draw everything that belongs to the legend"
445:         if not self.get_visible():
446:             return
447: 
448:         renderer.open_group('legend')
449: 
450:         fontsize = renderer.points_to_pixels(self._fontsize)
451: 
452:         # if mode == fill, set the width of the legend_box to the
453:         # width of the paret (minus pads)
454:         if self._mode in ["expand"]:
455:             pad = 2 * (self.borderaxespad + self.borderpad) * fontsize
456:             self._legend_box.set_width(self.get_bbox_to_anchor().width - pad)
457: 
458:         # update the location and size of the legend. This needs to
459:         # be done in any case to clip the figure right.
460:         bbox = self._legend_box.get_window_extent(renderer)
461:         self.legendPatch.set_bounds(bbox.x0, bbox.y0,
462:                                     bbox.width, bbox.height)
463:         self.legendPatch.set_mutation_scale(fontsize)
464: 
465:         if self._drawFrame:
466:             if self.shadow:
467:                 shadow = Shadow(self.legendPatch, 2, -2)
468:                 shadow.draw(renderer)
469: 
470:             self.legendPatch.draw(renderer)
471: 
472:         self._legend_box.draw(renderer)
473: 
474:         renderer.close_group('legend')
475:         self.stale = False
476: 
477:     def _approx_text_height(self, renderer=None):
478:         '''
479:         Return the approximate height of the text. This is used to place
480:         the legend handle.
481:         '''
482:         if renderer is None:
483:             return self._fontsize
484:         else:
485:             return renderer.points_to_pixels(self._fontsize)
486: 
487:     # _default_handler_map defines the default mapping between plot
488:     # elements and the legend handlers.
489: 
490:     _default_handler_map = {
491:         StemContainer: legend_handler.HandlerStem(),
492:         ErrorbarContainer: legend_handler.HandlerErrorbar(),
493:         Line2D: legend_handler.HandlerLine2D(),
494:         Patch: legend_handler.HandlerPatch(),
495:         LineCollection: legend_handler.HandlerLineCollection(),
496:         RegularPolyCollection: legend_handler.HandlerRegularPolyCollection(),
497:         CircleCollection: legend_handler.HandlerCircleCollection(),
498:         BarContainer: legend_handler.HandlerPatch(
499:             update_func=legend_handler.update_from_first_child),
500:         tuple: legend_handler.HandlerTuple(),
501:         PathCollection: legend_handler.HandlerPathCollection(),
502:         PolyCollection: legend_handler.HandlerPolyCollection()
503:         }
504: 
505:     # (get|set|update)_default_handler_maps are public interfaces to
506:     # modify the default handler map.
507: 
508:     @classmethod
509:     def get_default_handler_map(cls):
510:         '''
511:         A class method that returns the default handler map.
512:         '''
513:         return cls._default_handler_map
514: 
515:     @classmethod
516:     def set_default_handler_map(cls, handler_map):
517:         '''
518:         A class method to set the default handler map.
519:         '''
520:         cls._default_handler_map = handler_map
521: 
522:     @classmethod
523:     def update_default_handler_map(cls, handler_map):
524:         '''
525:         A class method to update the default handler map.
526:         '''
527:         cls._default_handler_map.update(handler_map)
528: 
529:     def get_legend_handler_map(self):
530:         '''
531:         return the handler map.
532:         '''
533: 
534:         default_handler_map = self.get_default_handler_map()
535: 
536:         if self._custom_handler_map:
537:             hm = default_handler_map.copy()
538:             hm.update(self._custom_handler_map)
539:             return hm
540:         else:
541:             return default_handler_map
542: 
543:     @staticmethod
544:     def get_legend_handler(legend_handler_map, orig_handle):
545:         '''
546:         return a legend handler from *legend_handler_map* that
547:         corresponds to *orig_handler*.
548: 
549:         *legend_handler_map* should be a dictionary object (that is
550:         returned by the get_legend_handler_map method).
551: 
552:         It first checks if the *orig_handle* itself is a key in the
553:         *legend_hanler_map* and return the associated value.
554:         Otherwise, it checks for each of the classes in its
555:         method-resolution-order. If no matching key is found, it
556:         returns None.
557:         '''
558:         if is_hashable(orig_handle):
559:             try:
560:                 return legend_handler_map[orig_handle]
561:             except KeyError:
562:                 pass
563: 
564:         for handle_type in type(orig_handle).mro():
565:             try:
566:                 return legend_handler_map[handle_type]
567:             except KeyError:
568:                 pass
569: 
570:         return None
571: 
572:     def _init_legend_box(self, handles, labels, markerfirst=True):
573:         '''
574:         Initialize the legend_box. The legend_box is an instance of
575:         the OffsetBox, which is packed with legend handles and
576:         texts. Once packed, their location is calculated during the
577:         drawing time.
578:         '''
579: 
580:         fontsize = self._fontsize
581: 
582:         # legend_box is a HPacker, horizontally packed with
583:         # columns. Each column is a VPacker, vertically packed with
584:         # legend items. Each legend item is HPacker packed with
585:         # legend handleBox and labelBox. handleBox is an instance of
586:         # offsetbox.DrawingArea which contains legend handle. labelBox
587:         # is an instance of offsetbox.TextArea which contains legend
588:         # text.
589: 
590:         text_list = []  # the list of text instances
591:         handle_list = []  # the list of text instances
592: 
593:         label_prop = dict(verticalalignment='baseline',
594:                           horizontalalignment='left',
595:                           fontproperties=self.prop,
596:                           )
597: 
598:         labelboxes = []
599:         handleboxes = []
600: 
601:         # The approximate height and descent of text. These values are
602:         # only used for plotting the legend handle.
603:         descent = 0.35 * self._approx_text_height() * (self.handleheight - 0.7)
604:         # 0.35 and 0.7 are just heuristic numbers and may need to be improved.
605:         height = self._approx_text_height() * self.handleheight - descent
606:         # each handle needs to be drawn inside a box of (x, y, w, h) =
607:         # (0, -descent, width, height).  And their coordinates should
608:         # be given in the display coordinates.
609: 
610:         # The transformation of each handle will be automatically set
611:         # to self.get_trasnform(). If the artist does not use its
612:         # default transform (e.g., Collections), you need to
613:         # manually set their transform to the self.get_transform().
614:         legend_handler_map = self.get_legend_handler_map()
615: 
616:         for orig_handle, lab in zip(handles, labels):
617:             handler = self.get_legend_handler(legend_handler_map, orig_handle)
618:             if handler is None:
619:                 warnings.warn(
620:                     "Legend does not support {!r} instances.\nA proxy artist "
621:                     "may be used instead.\nSee: "
622:                     "http://matplotlib.org/users/legend_guide.html"
623:                     "#using-proxy-artist".format(orig_handle)
624:                 )
625:                 # We don't have a handle for this artist, so we just defer
626:                 # to None.
627:                 handle_list.append(None)
628:             else:
629:                 textbox = TextArea(lab, textprops=label_prop,
630:                                    multilinebaseline=True,
631:                                    minimumdescent=True)
632:                 text_list.append(textbox._text)
633: 
634:                 labelboxes.append(textbox)
635: 
636:                 handlebox = DrawingArea(width=self.handlelength * fontsize,
637:                                         height=height,
638:                                         xdescent=0., ydescent=descent)
639:                 handleboxes.append(handlebox)
640: 
641:                 # Create the artist for the legend which represents the
642:                 # original artist/handle.
643:                 handle_list.append(handler.legend_artist(self, orig_handle,
644:                                                          fontsize, handlebox))
645: 
646:         if handleboxes:
647:             # We calculate number of rows in each column. The first
648:             # (num_largecol) columns will have (nrows+1) rows, and remaining
649:             # (num_smallcol) columns will have (nrows) rows.
650:             ncol = min(self._ncol, len(handleboxes))
651:             nrows, num_largecol = divmod(len(handleboxes), ncol)
652:             num_smallcol = ncol - num_largecol
653:             # starting index of each column and number of rows in it.
654:             rows_per_col = [nrows + 1] * num_largecol + [nrows] * num_smallcol
655:             start_idxs = np.concatenate([[0], np.cumsum(rows_per_col)[:-1]])
656:             cols = zip(start_idxs, rows_per_col)
657:         else:
658:             cols = []
659: 
660:         handle_label = list(zip(handleboxes, labelboxes))
661:         columnbox = []
662:         for i0, di in cols:
663:             # pack handleBox and labelBox into itemBox
664:             itemBoxes = [HPacker(pad=0,
665:                                  sep=self.handletextpad * fontsize,
666:                                  children=[h, t] if markerfirst else [t, h],
667:                                  align="baseline")
668:                          for h, t in handle_label[i0:i0 + di]]
669:             # minimumdescent=False for the text of the last row of the column
670:             if markerfirst:
671:                 itemBoxes[-1].get_children()[1].set_minimumdescent(False)
672:             else:
673:                 itemBoxes[-1].get_children()[0].set_minimumdescent(False)
674: 
675:             # pack columnBox
676:             alignment = "baseline" if markerfirst else "right"
677:             columnbox.append(VPacker(pad=0,
678:                                      sep=self.labelspacing * fontsize,
679:                                      align=alignment,
680:                                      children=itemBoxes))
681: 
682:         mode = "expand" if self._mode == "expand" else "fixed"
683:         sep = self.columnspacing * fontsize
684:         self._legend_handle_box = HPacker(pad=0,
685:                                           sep=sep, align="baseline",
686:                                           mode=mode,
687:                                           children=columnbox)
688:         self._legend_title_box = TextArea("")
689:         self._legend_box = VPacker(pad=self.borderpad * fontsize,
690:                                    sep=self.labelspacing * fontsize,
691:                                    align="center",
692:                                    children=[self._legend_title_box,
693:                                              self._legend_handle_box])
694:         self._legend_box.set_figure(self.figure)
695:         self._legend_box.set_offset(self._findoffset)
696:         self.texts = text_list
697:         self.legendHandles = handle_list
698: 
699:     def _auto_legend_data(self):
700:         '''
701:         Returns list of vertices and extents covered by the plot.
702: 
703:         Returns a two long list.
704: 
705:         First element is a list of (x, y) vertices (in
706:         display-coordinates) covered by all the lines and line
707:         collections, in the legend's handles.
708: 
709:         Second element is a list of bounding boxes for all the patches in
710:         the legend's handles.
711:         '''
712:         # should always hold because function is only called internally
713:         assert self.isaxes
714: 
715:         ax = self.parent
716:         bboxes = []
717:         lines = []
718:         offsets = []
719: 
720:         for handle in ax.lines:
721:             assert isinstance(handle, Line2D)
722:             path = handle.get_path()
723:             trans = handle.get_transform()
724:             tpath = trans.transform_path(path)
725:             lines.append(tpath)
726: 
727:         for handle in ax.patches:
728:             assert isinstance(handle, Patch)
729: 
730:             if isinstance(handle, Rectangle):
731:                 transform = handle.get_data_transform()
732:                 bboxes.append(handle.get_bbox().transformed(transform))
733:             else:
734:                 transform = handle.get_transform()
735:                 bboxes.append(handle.get_path().get_extents(transform))
736: 
737:         for handle in ax.collections:
738:             transform, transOffset, hoffsets, paths = handle._prepare_points()
739: 
740:             if len(hoffsets):
741:                 for offset in transOffset.transform(hoffsets):
742:                     offsets.append(offset)
743: 
744:         try:
745:             vertices = np.concatenate([l.vertices for l in lines])
746:         except ValueError:
747:             vertices = np.array([])
748: 
749:         return [vertices, bboxes, lines, offsets]
750: 
751:     def draw_frame(self, b):
752:         'b is a boolean.  Set draw frame to b'
753:         self.set_frame_on(b)
754: 
755:     def get_children(self):
756:         'return a list of child artists'
757:         children = []
758:         if self._legend_box:
759:             children.append(self._legend_box)
760:         children.append(self.get_frame())
761: 
762:         return children
763: 
764:     def get_frame(self):
765:         'return the Rectangle instance used to frame the legend'
766:         return self.legendPatch
767: 
768:     def get_lines(self):
769:         'return a list of lines.Line2D instances in the legend'
770:         return [h for h in self.legendHandles if isinstance(h, Line2D)]
771: 
772:     def get_patches(self):
773:         'return a list of patch instances in the legend'
774:         return silent_list('Patch',
775:                            [h for h in self.legendHandles
776:                             if isinstance(h, Patch)])
777: 
778:     def get_texts(self):
779:         'return a list of text.Text instance in the legend'
780:         return silent_list('Text', self.texts)
781: 
782:     def set_title(self, title, prop=None):
783:         '''
784:         set the legend title. Fontproperties can be optionally set
785:         with *prop* parameter.
786:         '''
787:         self._legend_title_box._text.set_text(title)
788: 
789:         if prop is not None:
790:             if isinstance(prop, dict):
791:                 prop = FontProperties(**prop)
792:             self._legend_title_box._text.set_fontproperties(prop)
793: 
794:         if title:
795:             self._legend_title_box.set_visible(True)
796:         else:
797:             self._legend_title_box.set_visible(False)
798:         self.stale = True
799: 
800:     def get_title(self):
801:         'return Text instance for the legend title'
802:         return self._legend_title_box._text
803: 
804:     def get_window_extent(self, *args, **kwargs):
805:         'return a extent of the legend'
806:         return self.legendPatch.get_window_extent(*args, **kwargs)
807: 
808:     def get_frame_on(self):
809:         '''
810:         Get whether the legend box patch is drawn
811:         '''
812:         return self._drawFrame
813: 
814:     def set_frame_on(self, b):
815:         '''
816:         Set whether the legend box patch is drawn
817: 
818:         ACCEPTS: [ *True* | *False* ]
819:         '''
820:         self._drawFrame = b
821:         self.stale = True
822: 
823:     def get_bbox_to_anchor(self):
824:         '''
825:         return the bbox that the legend will be anchored
826:         '''
827:         if self._bbox_to_anchor is None:
828:             return self.parent.bbox
829:         else:
830:             return self._bbox_to_anchor
831: 
832:     def set_bbox_to_anchor(self, bbox, transform=None):
833:         '''
834:         set the bbox that the legend will be anchored.
835: 
836:         *bbox* can be a BboxBase instance, a tuple of [left, bottom,
837:         width, height] in the given transform (normalized axes
838:         coordinate if None), or a tuple of [left, bottom] where the
839:         width and height will be assumed to be zero.
840:         '''
841:         if bbox is None:
842:             self._bbox_to_anchor = None
843:             return
844:         elif isinstance(bbox, BboxBase):
845:             self._bbox_to_anchor = bbox
846:         else:
847:             try:
848:                 l = len(bbox)
849:             except TypeError:
850:                 raise ValueError("Invalid argument for bbox : %s" % str(bbox))
851: 
852:             if l == 2:
853:                 bbox = [bbox[0], bbox[1], 0, 0]
854: 
855:             self._bbox_to_anchor = Bbox.from_bounds(*bbox)
856: 
857:         if transform is None:
858:             transform = BboxTransformTo(self.parent.bbox)
859: 
860:         self._bbox_to_anchor = TransformedBbox(self._bbox_to_anchor,
861:                                                transform)
862:         self.stale = True
863: 
864:     def _get_anchored_bbox(self, loc, bbox, parentbbox, renderer):
865:         '''
866:         Place the *bbox* inside the *parentbbox* according to a given
867:         location code. Return the (x,y) coordinate of the bbox.
868: 
869:         - loc: a location code in range(1, 11).
870:           This corresponds to the possible values for self._loc, excluding
871:           "best".
872: 
873:         - bbox: bbox to be placed, display coodinate units.
874:         - parentbbox: a parent box which will contain the bbox. In
875:             display coordinates.
876:         '''
877:         assert loc in range(1, 11)  # called only internally
878: 
879:         BEST, UR, UL, LL, LR, R, CL, CR, LC, UC, C = list(xrange(11))
880: 
881:         anchor_coefs = {UR: "NE",
882:                         UL: "NW",
883:                         LL: "SW",
884:                         LR: "SE",
885:                         R: "E",
886:                         CL: "W",
887:                         CR: "E",
888:                         LC: "S",
889:                         UC: "N",
890:                         C: "C"}
891: 
892:         c = anchor_coefs[loc]
893: 
894:         fontsize = renderer.points_to_pixels(self._fontsize)
895:         container = parentbbox.padded(-(self.borderaxespad) * fontsize)
896:         anchored_box = bbox.anchored(c, container=container)
897:         return anchored_box.x0, anchored_box.y0
898: 
899:     def _find_best_position(self, width, height, renderer, consider=None):
900:         '''
901:         Determine the best location to place the legend.
902: 
903:         `consider` is a list of (x, y) pairs to consider as a potential
904:         lower-left corner of the legend. All are display coords.
905:         '''
906:         # should always hold because function is only called internally
907:         assert self.isaxes
908: 
909:         verts, bboxes, lines, offsets = self._auto_legend_data()
910: 
911:         bbox = Bbox.from_bounds(0, 0, width, height)
912:         if consider is None:
913:             consider = [self._get_anchored_bbox(x, bbox,
914:                                                 self.get_bbox_to_anchor(),
915:                                                 renderer)
916:                         for x in range(1, len(self.codes))]
917: 
918:         candidates = []
919:         for idx, (l, b) in enumerate(consider):
920:             legendBox = Bbox.from_bounds(l, b, width, height)
921:             badness = 0
922:             # XXX TODO: If markers are present, it would be good to
923:             # take them into account when checking vertex overlaps in
924:             # the next line.
925:             badness = (legendBox.count_contains(verts)
926:                        + legendBox.count_contains(offsets)
927:                        + legendBox.count_overlaps(bboxes)
928:                        + sum(line.intersects_bbox(legendBox, filled=False)
929:                              for line in lines))
930:             if badness == 0:
931:                 return l, b
932:             # Include the index to favor lower codes in case of a tie.
933:             candidates.append((badness, idx, (l, b)))
934: 
935:         _, _, (l, b) = min(candidates)
936:         return l, b
937: 
938:     def contains(self, event):
939:         return self.legendPatch.contains(event)
940: 
941:     def draggable(self, state=None, use_blit=False, update="loc"):
942:         '''
943:         Set the draggable state -- if state is
944: 
945:           * None : toggle the current state
946: 
947:           * True : turn draggable on
948: 
949:           * False : turn draggable off
950: 
951:         If draggable is on, you can drag the legend on the canvas with
952:         the mouse.  The DraggableLegend helper instance is returned if
953:         draggable is on.
954: 
955:         The update parameter control which parameter of the legend changes
956:         when dragged. If update is "loc", the *loc* parameter of the legend
957:         is changed. If "bbox", the *bbox_to_anchor* parameter is changed.
958:         '''
959:         is_draggable = self._draggable is not None
960: 
961:         # if state is None we'll toggle
962:         if state is None:
963:             state = not is_draggable
964: 
965:         if state:
966:             if self._draggable is None:
967:                 self._draggable = DraggableLegend(self,
968:                                                   use_blit,
969:                                                   update=update)
970:         else:
971:             if self._draggable is not None:
972:                 self._draggable.disconnect()
973:             self._draggable = None
974: 
975:         return self._draggable
976: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_66679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, (-1)), 'unicode', u"\nThe legend module defines the Legend class, which is responsible for\ndrawing legends associated with axes and/or figures.\n\n.. important::\n\n    It is unlikely that you would ever create a Legend instance manually.\n    Most users would normally create a legend via the\n    :meth:`~matplotlib.axes.Axes.legend` function. For more details on legends\n    there is also a :ref:`legend guide\n    <sphx_glr_tutorials_intermediate_legend_guide.py>`.\n\nThe Legend class can be considered as a container of legend handles\nand legend texts. Creation of corresponding legend handles from the\nplot elements in the axes or figures (e.g., lines, patches, etc.) are\nspecified by the handler map, which defines the mapping between the\nplot elements and the legend handlers to be used (the default legend\nhandlers are defined in the :mod:`~matplotlib.legend_handler` module).\nNote that not all kinds of artist are supported by the legend yet by default\nbut it is possible to extend the legend handler's capabilities to support\narbitrary objects. See the :ref:`legend guide\n<sphx_glr_tutorials_intermediate_legend_guide.py>` for more information.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import six' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66680 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'six')

if (type(import_66680) is not StypyTypeError):

    if (import_66680 != 'pyd_module'):
        __import__(import_66680)
        sys_modules_66681 = sys.modules[import_66680]
        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'six', sys_modules_66681.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'six', import_66680)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from six.moves import xrange' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66682 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'six.moves')

if (type(import_66682) is not StypyTypeError):

    if (import_66682 != 'pyd_module'):
        __import__(import_66682)
        sys_modules_66683 = sys.modules[import_66682]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'six.moves', sys_modules_66683.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_66683, sys_modules_66683.module_type_store, module_type_store)
    else:
        from six.moves import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'six.moves', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'six.moves' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'six.moves', import_66682)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'import warnings' statement (line 31)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'import numpy' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy')

if (type(import_66684) is not StypyTypeError):

    if (import_66684 != 'pyd_module'):
        __import__(import_66684)
        sys_modules_66685 = sys.modules[import_66684]
        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'np', sys_modules_66685.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', import_66684)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from matplotlib import rcParams' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66686 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib')

if (type(import_66686) is not StypyTypeError):

    if (import_66686 != 'pyd_module'):
        __import__(import_66686)
        sys_modules_66687 = sys.modules[import_66686]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib', sys_modules_66687.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_66687, sys_modules_66687.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib', import_66686)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from matplotlib.artist import Artist, allow_rasterization' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66688 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.artist')

if (type(import_66688) is not StypyTypeError):

    if (import_66688 != 'pyd_module'):
        __import__(import_66688)
        sys_modules_66689 = sys.modules[import_66688]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.artist', sys_modules_66689.module_type_store, module_type_store, ['Artist', 'allow_rasterization'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_66689, sys_modules_66689.module_type_store, module_type_store)
    else:
        from matplotlib.artist import Artist, allow_rasterization

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.artist', None, module_type_store, ['Artist', 'allow_rasterization'], [Artist, allow_rasterization])

else:
    # Assigning a type to the variable 'matplotlib.artist' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.artist', import_66688)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from matplotlib.cbook import silent_list, is_hashable' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66690 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.cbook')

if (type(import_66690) is not StypyTypeError):

    if (import_66690 != 'pyd_module'):
        __import__(import_66690)
        sys_modules_66691 = sys.modules[import_66690]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.cbook', sys_modules_66691.module_type_store, module_type_store, ['silent_list', 'is_hashable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_66691, sys_modules_66691.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import silent_list, is_hashable

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.cbook', None, module_type_store, ['silent_list', 'is_hashable'], [silent_list, is_hashable])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.cbook', import_66690)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'from matplotlib.font_manager import FontProperties' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66692 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'matplotlib.font_manager')

if (type(import_66692) is not StypyTypeError):

    if (import_66692 != 'pyd_module'):
        __import__(import_66692)
        sys_modules_66693 = sys.modules[import_66692]
        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'matplotlib.font_manager', sys_modules_66693.module_type_store, module_type_store, ['FontProperties'])
        nest_module(stypy.reporting.localization.Localization(__file__, 38, 0), __file__, sys_modules_66693, sys_modules_66693.module_type_store, module_type_store)
    else:
        from matplotlib.font_manager import FontProperties

        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'matplotlib.font_manager', None, module_type_store, ['FontProperties'], [FontProperties])

else:
    # Assigning a type to the variable 'matplotlib.font_manager' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'matplotlib.font_manager', import_66692)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'from matplotlib.lines import Line2D' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66694 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.lines')

if (type(import_66694) is not StypyTypeError):

    if (import_66694 != 'pyd_module'):
        __import__(import_66694)
        sys_modules_66695 = sys.modules[import_66694]
        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.lines', sys_modules_66695.module_type_store, module_type_store, ['Line2D'])
        nest_module(stypy.reporting.localization.Localization(__file__, 39, 0), __file__, sys_modules_66695, sys_modules_66695.module_type_store, module_type_store)
    else:
        from matplotlib.lines import Line2D

        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.lines', None, module_type_store, ['Line2D'], [Line2D])

else:
    # Assigning a type to the variable 'matplotlib.lines' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.lines', import_66694)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'from matplotlib.patches import Patch, Rectangle, Shadow, FancyBboxPatch' statement (line 40)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66696 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.patches')

if (type(import_66696) is not StypyTypeError):

    if (import_66696 != 'pyd_module'):
        __import__(import_66696)
        sys_modules_66697 = sys.modules[import_66696]
        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.patches', sys_modules_66697.module_type_store, module_type_store, ['Patch', 'Rectangle', 'Shadow', 'FancyBboxPatch'])
        nest_module(stypy.reporting.localization.Localization(__file__, 40, 0), __file__, sys_modules_66697, sys_modules_66697.module_type_store, module_type_store)
    else:
        from matplotlib.patches import Patch, Rectangle, Shadow, FancyBboxPatch

        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.patches', None, module_type_store, ['Patch', 'Rectangle', 'Shadow', 'FancyBboxPatch'], [Patch, Rectangle, Shadow, FancyBboxPatch])

else:
    # Assigning a type to the variable 'matplotlib.patches' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.patches', import_66696)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 0))

# 'from matplotlib.collections import LineCollection, RegularPolyCollection, CircleCollection, PathCollection, PolyCollection' statement (line 41)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66698 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.collections')

if (type(import_66698) is not StypyTypeError):

    if (import_66698 != 'pyd_module'):
        __import__(import_66698)
        sys_modules_66699 = sys.modules[import_66698]
        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.collections', sys_modules_66699.module_type_store, module_type_store, ['LineCollection', 'RegularPolyCollection', 'CircleCollection', 'PathCollection', 'PolyCollection'])
        nest_module(stypy.reporting.localization.Localization(__file__, 41, 0), __file__, sys_modules_66699, sys_modules_66699.module_type_store, module_type_store)
    else:
        from matplotlib.collections import LineCollection, RegularPolyCollection, CircleCollection, PathCollection, PolyCollection

        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.collections', None, module_type_store, ['LineCollection', 'RegularPolyCollection', 'CircleCollection', 'PathCollection', 'PolyCollection'], [LineCollection, RegularPolyCollection, CircleCollection, PathCollection, PolyCollection])

else:
    # Assigning a type to the variable 'matplotlib.collections' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.collections', import_66698)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'from matplotlib.transforms import Bbox, BboxBase, TransformedBbox' statement (line 44)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66700 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.transforms')

if (type(import_66700) is not StypyTypeError):

    if (import_66700 != 'pyd_module'):
        __import__(import_66700)
        sys_modules_66701 = sys.modules[import_66700]
        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.transforms', sys_modules_66701.module_type_store, module_type_store, ['Bbox', 'BboxBase', 'TransformedBbox'])
        nest_module(stypy.reporting.localization.Localization(__file__, 44, 0), __file__, sys_modules_66701, sys_modules_66701.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Bbox, BboxBase, TransformedBbox

        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.transforms', None, module_type_store, ['Bbox', 'BboxBase', 'TransformedBbox'], [Bbox, BboxBase, TransformedBbox])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.transforms', import_66700)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 0))

# 'from matplotlib.transforms import BboxTransformTo, BboxTransformFrom' statement (line 45)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66702 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib.transforms')

if (type(import_66702) is not StypyTypeError):

    if (import_66702 != 'pyd_module'):
        __import__(import_66702)
        sys_modules_66703 = sys.modules[import_66702]
        import_from_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib.transforms', sys_modules_66703.module_type_store, module_type_store, ['BboxTransformTo', 'BboxTransformFrom'])
        nest_module(stypy.reporting.localization.Localization(__file__, 45, 0), __file__, sys_modules_66703, sys_modules_66703.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import BboxTransformTo, BboxTransformFrom

        import_from_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib.transforms', None, module_type_store, ['BboxTransformTo', 'BboxTransformFrom'], [BboxTransformTo, BboxTransformFrom])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'matplotlib.transforms', import_66702)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 47, 0))

# 'from matplotlib.offsetbox import HPacker, VPacker, TextArea, DrawingArea' statement (line 47)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66704 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'matplotlib.offsetbox')

if (type(import_66704) is not StypyTypeError):

    if (import_66704 != 'pyd_module'):
        __import__(import_66704)
        sys_modules_66705 = sys.modules[import_66704]
        import_from_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'matplotlib.offsetbox', sys_modules_66705.module_type_store, module_type_store, ['HPacker', 'VPacker', 'TextArea', 'DrawingArea'])
        nest_module(stypy.reporting.localization.Localization(__file__, 47, 0), __file__, sys_modules_66705, sys_modules_66705.module_type_store, module_type_store)
    else:
        from matplotlib.offsetbox import HPacker, VPacker, TextArea, DrawingArea

        import_from_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'matplotlib.offsetbox', None, module_type_store, ['HPacker', 'VPacker', 'TextArea', 'DrawingArea'], [HPacker, VPacker, TextArea, DrawingArea])

else:
    # Assigning a type to the variable 'matplotlib.offsetbox' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'matplotlib.offsetbox', import_66704)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 48, 0))

# 'from matplotlib.offsetbox import DraggableOffsetBox' statement (line 48)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66706 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'matplotlib.offsetbox')

if (type(import_66706) is not StypyTypeError):

    if (import_66706 != 'pyd_module'):
        __import__(import_66706)
        sys_modules_66707 = sys.modules[import_66706]
        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'matplotlib.offsetbox', sys_modules_66707.module_type_store, module_type_store, ['DraggableOffsetBox'])
        nest_module(stypy.reporting.localization.Localization(__file__, 48, 0), __file__, sys_modules_66707, sys_modules_66707.module_type_store, module_type_store)
    else:
        from matplotlib.offsetbox import DraggableOffsetBox

        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'matplotlib.offsetbox', None, module_type_store, ['DraggableOffsetBox'], [DraggableOffsetBox])

else:
    # Assigning a type to the variable 'matplotlib.offsetbox' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'matplotlib.offsetbox', import_66706)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 50, 0))

# 'from matplotlib.container import ErrorbarContainer, BarContainer, StemContainer' statement (line 50)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66708 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'matplotlib.container')

if (type(import_66708) is not StypyTypeError):

    if (import_66708 != 'pyd_module'):
        __import__(import_66708)
        sys_modules_66709 = sys.modules[import_66708]
        import_from_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'matplotlib.container', sys_modules_66709.module_type_store, module_type_store, ['ErrorbarContainer', 'BarContainer', 'StemContainer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 50, 0), __file__, sys_modules_66709, sys_modules_66709.module_type_store, module_type_store)
    else:
        from matplotlib.container import ErrorbarContainer, BarContainer, StemContainer

        import_from_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'matplotlib.container', None, module_type_store, ['ErrorbarContainer', 'BarContainer', 'StemContainer'], [ErrorbarContainer, BarContainer, StemContainer])

else:
    # Assigning a type to the variable 'matplotlib.container' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'matplotlib.container', import_66708)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 51, 0))

# 'from matplotlib import legend_handler' statement (line 51)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_66710 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'matplotlib')

if (type(import_66710) is not StypyTypeError):

    if (import_66710 != 'pyd_module'):
        __import__(import_66710)
        sys_modules_66711 = sys.modules[import_66710]
        import_from_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'matplotlib', sys_modules_66711.module_type_store, module_type_store, ['legend_handler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 51, 0), __file__, sys_modules_66711, sys_modules_66711.module_type_store, module_type_store)
    else:
        from matplotlib import legend_handler

        import_from_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'matplotlib', None, module_type_store, ['legend_handler'], [legend_handler])

else:
    # Assigning a type to the variable 'matplotlib' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'matplotlib', import_66710)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'DraggableLegend' class
# Getting the type of 'DraggableOffsetBox' (line 54)
DraggableOffsetBox_66712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'DraggableOffsetBox')

class DraggableLegend(DraggableOffsetBox_66712, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 55)
        False_66713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'False')
        unicode_66714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 54), 'unicode', u'loc')
        defaults = [False_66713, unicode_66714]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DraggableLegend.__init__', ['legend', 'use_blit', 'update'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['legend', 'use_blit', 'update'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_66715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, (-1)), 'unicode', u'\n        update : If "loc", update *loc* parameter of\n                 legend upon finalizing. If "bbox", update\n                 *bbox_to_anchor* parameter.\n        ')
        
        # Assigning a Name to a Attribute (line 61):
        
        # Assigning a Name to a Attribute (line 61):
        
        # Assigning a Name to a Attribute (line 61):
        # Getting the type of 'legend' (line 61)
        legend_66716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 22), 'legend')
        # Getting the type of 'self' (line 61)
        self_66717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'legend' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_66717, 'legend', legend_66716)
        
        
        # Getting the type of 'update' (line 63)
        update_66718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'update')
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_66719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        unicode_66720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 22), 'unicode', u'loc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), list_66719, unicode_66720)
        # Adding element type (line 63)
        unicode_66721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'unicode', u'bbox')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), list_66719, unicode_66721)
        
        # Applying the binary operator 'in' (line 63)
        result_contains_66722 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 11), 'in', update_66718, list_66719)
        
        # Testing the type of an if condition (line 63)
        if_condition_66723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 8), result_contains_66722)
        # Assigning a type to the variable 'if_condition_66723' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'if_condition_66723', if_condition_66723)
        # SSA begins for if statement (line 63)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 64):
        
        # Assigning a Name to a Attribute (line 64):
        
        # Assigning a Name to a Attribute (line 64):
        # Getting the type of 'update' (line 64)
        update_66724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'update')
        # Getting the type of 'self' (line 64)
        self_66725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self')
        # Setting the type of the member '_update' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_66725, '_update', update_66724)
        # SSA branch for the else part of an if statement (line 63)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 66)
        # Processing the call arguments (line 66)
        unicode_66727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 29), 'unicode', u"update parameter '%s' is not supported.")
        # Getting the type of 'update' (line 67)
        update_66728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'update', False)
        # Applying the binary operator '%' (line 66)
        result_mod_66729 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 29), '%', unicode_66727, update_66728)
        
        # Processing the call keyword arguments (line 66)
        kwargs_66730 = {}
        # Getting the type of 'ValueError' (line 66)
        ValueError_66726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 66)
        ValueError_call_result_66731 = invoke(stypy.reporting.localization.Localization(__file__, 66, 18), ValueError_66726, *[result_mod_66729], **kwargs_66730)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 66, 12), ValueError_call_result_66731, 'raise parameter', BaseException)
        # SSA join for if statement (line 63)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __init__(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'self' (line 69)
        self_66734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 36), 'self', False)
        # Getting the type of 'legend' (line 69)
        legend_66735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 42), 'legend', False)
        # Getting the type of 'legend' (line 69)
        legend_66736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 50), 'legend', False)
        # Obtaining the member '_legend_box' of a type (line 69)
        _legend_box_66737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 50), legend_66736, '_legend_box')
        # Processing the call keyword arguments (line 69)
        # Getting the type of 'use_blit' (line 70)
        use_blit_66738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 45), 'use_blit', False)
        keyword_66739 = use_blit_66738
        kwargs_66740 = {'use_blit': keyword_66739}
        # Getting the type of 'DraggableOffsetBox' (line 69)
        DraggableOffsetBox_66732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'DraggableOffsetBox', False)
        # Obtaining the member '__init__' of a type (line 69)
        init___66733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), DraggableOffsetBox_66732, '__init__')
        # Calling __init__(args, kwargs) (line 69)
        init___call_result_66741 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), init___66733, *[self_66734, legend_66735, _legend_box_66737], **kwargs_66740)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def artist_picker(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'artist_picker'
        module_type_store = module_type_store.open_function_context('artist_picker', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DraggableLegend.artist_picker.__dict__.__setitem__('stypy_localization', localization)
        DraggableLegend.artist_picker.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DraggableLegend.artist_picker.__dict__.__setitem__('stypy_type_store', module_type_store)
        DraggableLegend.artist_picker.__dict__.__setitem__('stypy_function_name', 'DraggableLegend.artist_picker')
        DraggableLegend.artist_picker.__dict__.__setitem__('stypy_param_names_list', ['legend', 'evt'])
        DraggableLegend.artist_picker.__dict__.__setitem__('stypy_varargs_param_name', None)
        DraggableLegend.artist_picker.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DraggableLegend.artist_picker.__dict__.__setitem__('stypy_call_defaults', defaults)
        DraggableLegend.artist_picker.__dict__.__setitem__('stypy_call_varargs', varargs)
        DraggableLegend.artist_picker.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DraggableLegend.artist_picker.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DraggableLegend.artist_picker', ['legend', 'evt'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'artist_picker', localization, ['legend', 'evt'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'artist_picker(...)' code ##################

        
        # Call to contains(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'evt' (line 73)
        evt_66745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 36), 'evt', False)
        # Processing the call keyword arguments (line 73)
        kwargs_66746 = {}
        # Getting the type of 'self' (line 73)
        self_66742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'self', False)
        # Obtaining the member 'legend' of a type (line 73)
        legend_66743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 15), self_66742, 'legend')
        # Obtaining the member 'contains' of a type (line 73)
        contains_66744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 15), legend_66743, 'contains')
        # Calling contains(args, kwargs) (line 73)
        contains_call_result_66747 = invoke(stypy.reporting.localization.Localization(__file__, 73, 15), contains_66744, *[evt_66745], **kwargs_66746)
        
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', contains_call_result_66747)
        
        # ################# End of 'artist_picker(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'artist_picker' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_66748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_66748)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'artist_picker'
        return stypy_return_type_66748


    @norecursion
    def finalize_offset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_offset'
        module_type_store = module_type_store.open_function_context('finalize_offset', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DraggableLegend.finalize_offset.__dict__.__setitem__('stypy_localization', localization)
        DraggableLegend.finalize_offset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DraggableLegend.finalize_offset.__dict__.__setitem__('stypy_type_store', module_type_store)
        DraggableLegend.finalize_offset.__dict__.__setitem__('stypy_function_name', 'DraggableLegend.finalize_offset')
        DraggableLegend.finalize_offset.__dict__.__setitem__('stypy_param_names_list', [])
        DraggableLegend.finalize_offset.__dict__.__setitem__('stypy_varargs_param_name', None)
        DraggableLegend.finalize_offset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DraggableLegend.finalize_offset.__dict__.__setitem__('stypy_call_defaults', defaults)
        DraggableLegend.finalize_offset.__dict__.__setitem__('stypy_call_varargs', varargs)
        DraggableLegend.finalize_offset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DraggableLegend.finalize_offset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DraggableLegend.finalize_offset', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'finalize_offset', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'finalize_offset(...)' code ##################

        
        # Assigning a Call to a Name (line 76):
        
        # Assigning a Call to a Name (line 76):
        
        # Assigning a Call to a Name (line 76):
        
        # Call to get_loc_in_canvas(...): (line 76)
        # Processing the call keyword arguments (line 76)
        kwargs_66751 = {}
        # Getting the type of 'self' (line 76)
        self_66749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'self', False)
        # Obtaining the member 'get_loc_in_canvas' of a type (line 76)
        get_loc_in_canvas_66750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 24), self_66749, 'get_loc_in_canvas')
        # Calling get_loc_in_canvas(args, kwargs) (line 76)
        get_loc_in_canvas_call_result_66752 = invoke(stypy.reporting.localization.Localization(__file__, 76, 24), get_loc_in_canvas_66750, *[], **kwargs_66751)
        
        # Assigning a type to the variable 'loc_in_canvas' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'loc_in_canvas', get_loc_in_canvas_call_result_66752)
        
        
        # Getting the type of 'self' (line 78)
        self_66753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'self')
        # Obtaining the member '_update' of a type (line 78)
        _update_66754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 11), self_66753, '_update')
        unicode_66755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 27), 'unicode', u'loc')
        # Applying the binary operator '==' (line 78)
        result_eq_66756 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), '==', _update_66754, unicode_66755)
        
        # Testing the type of an if condition (line 78)
        if_condition_66757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_eq_66756)
        # Assigning a type to the variable 'if_condition_66757' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_66757', if_condition_66757)
        # SSA begins for if statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _update_loc(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'loc_in_canvas' (line 79)
        loc_in_canvas_66760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'loc_in_canvas', False)
        # Processing the call keyword arguments (line 79)
        kwargs_66761 = {}
        # Getting the type of 'self' (line 79)
        self_66758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'self', False)
        # Obtaining the member '_update_loc' of a type (line 79)
        _update_loc_66759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), self_66758, '_update_loc')
        # Calling _update_loc(args, kwargs) (line 79)
        _update_loc_call_result_66762 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), _update_loc_66759, *[loc_in_canvas_66760], **kwargs_66761)
        
        # SSA branch for the else part of an if statement (line 78)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 80)
        self_66763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'self')
        # Obtaining the member '_update' of a type (line 80)
        _update_66764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 13), self_66763, '_update')
        unicode_66765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 29), 'unicode', u'bbox')
        # Applying the binary operator '==' (line 80)
        result_eq_66766 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '==', _update_66764, unicode_66765)
        
        # Testing the type of an if condition (line 80)
        if_condition_66767 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 13), result_eq_66766)
        # Assigning a type to the variable 'if_condition_66767' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'if_condition_66767', if_condition_66767)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _update_bbox_to_anchor(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'loc_in_canvas' (line 81)
        loc_in_canvas_66770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 40), 'loc_in_canvas', False)
        # Processing the call keyword arguments (line 81)
        kwargs_66771 = {}
        # Getting the type of 'self' (line 81)
        self_66768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'self', False)
        # Obtaining the member '_update_bbox_to_anchor' of a type (line 81)
        _update_bbox_to_anchor_66769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), self_66768, '_update_bbox_to_anchor')
        # Calling _update_bbox_to_anchor(args, kwargs) (line 81)
        _update_bbox_to_anchor_call_result_66772 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), _update_bbox_to_anchor_66769, *[loc_in_canvas_66770], **kwargs_66771)
        
        # SSA branch for the else part of an if statement (line 80)
        module_type_store.open_ssa_branch('else')
        
        # Call to RuntimeError(...): (line 83)
        # Processing the call arguments (line 83)
        unicode_66774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 31), 'unicode', u"update parameter '%s' is not supported.")
        # Getting the type of 'self' (line 84)
        self_66775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'self', False)
        # Obtaining the member 'update' of a type (line 84)
        update_66776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 31), self_66775, 'update')
        # Applying the binary operator '%' (line 83)
        result_mod_66777 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 31), '%', unicode_66774, update_66776)
        
        # Processing the call keyword arguments (line 83)
        kwargs_66778 = {}
        # Getting the type of 'RuntimeError' (line 83)
        RuntimeError_66773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 83)
        RuntimeError_call_result_66779 = invoke(stypy.reporting.localization.Localization(__file__, 83, 18), RuntimeError_66773, *[result_mod_66777], **kwargs_66778)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 83, 12), RuntimeError_call_result_66779, 'raise parameter', BaseException)
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 78)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'finalize_offset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_offset' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_66780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_66780)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_offset'
        return stypy_return_type_66780


    @norecursion
    def _update_loc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update_loc'
        module_type_store = module_type_store.open_function_context('_update_loc', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DraggableLegend._update_loc.__dict__.__setitem__('stypy_localization', localization)
        DraggableLegend._update_loc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DraggableLegend._update_loc.__dict__.__setitem__('stypy_type_store', module_type_store)
        DraggableLegend._update_loc.__dict__.__setitem__('stypy_function_name', 'DraggableLegend._update_loc')
        DraggableLegend._update_loc.__dict__.__setitem__('stypy_param_names_list', ['loc_in_canvas'])
        DraggableLegend._update_loc.__dict__.__setitem__('stypy_varargs_param_name', None)
        DraggableLegend._update_loc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DraggableLegend._update_loc.__dict__.__setitem__('stypy_call_defaults', defaults)
        DraggableLegend._update_loc.__dict__.__setitem__('stypy_call_varargs', varargs)
        DraggableLegend._update_loc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DraggableLegend._update_loc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DraggableLegend._update_loc', ['loc_in_canvas'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update_loc', localization, ['loc_in_canvas'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update_loc(...)' code ##################

        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to get_bbox_to_anchor(...): (line 87)
        # Processing the call keyword arguments (line 87)
        kwargs_66784 = {}
        # Getting the type of 'self' (line 87)
        self_66781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'self', False)
        # Obtaining the member 'legend' of a type (line 87)
        legend_66782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), self_66781, 'legend')
        # Obtaining the member 'get_bbox_to_anchor' of a type (line 87)
        get_bbox_to_anchor_66783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), legend_66782, 'get_bbox_to_anchor')
        # Calling get_bbox_to_anchor(args, kwargs) (line 87)
        get_bbox_to_anchor_call_result_66785 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), get_bbox_to_anchor_66783, *[], **kwargs_66784)
        
        # Assigning a type to the variable 'bbox' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'bbox', get_bbox_to_anchor_call_result_66785)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'bbox' (line 91)
        bbox_66786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'bbox')
        # Obtaining the member 'width' of a type (line 91)
        width_66787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 11), bbox_66786, 'width')
        int_66788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 25), 'int')
        # Applying the binary operator '==' (line 91)
        result_eq_66789 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), '==', width_66787, int_66788)
        
        
        # Getting the type of 'bbox' (line 91)
        bbox_66790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), 'bbox')
        # Obtaining the member 'height' of a type (line 91)
        height_66791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 30), bbox_66790, 'height')
        int_66792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 45), 'int')
        # Applying the binary operator '==' (line 91)
        result_eq_66793 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 30), '==', height_66791, int_66792)
        
        # Applying the binary operator 'or' (line 91)
        result_or_keyword_66794 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), 'or', result_eq_66789, result_eq_66793)
        
        # Testing the type of an if condition (line 91)
        if_condition_66795 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), result_or_keyword_66794)
        # Assigning a type to the variable 'if_condition_66795' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_66795', if_condition_66795)
        # SSA begins for if statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_bbox_to_anchor(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'None' (line 92)
        None_66799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 43), 'None', False)
        # Processing the call keyword arguments (line 92)
        kwargs_66800 = {}
        # Getting the type of 'self' (line 92)
        self_66796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self', False)
        # Obtaining the member 'legend' of a type (line 92)
        legend_66797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_66796, 'legend')
        # Obtaining the member 'set_bbox_to_anchor' of a type (line 92)
        set_bbox_to_anchor_66798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), legend_66797, 'set_bbox_to_anchor')
        # Calling set_bbox_to_anchor(args, kwargs) (line 92)
        set_bbox_to_anchor_call_result_66801 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), set_bbox_to_anchor_66798, *[None_66799], **kwargs_66800)
        
        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to get_bbox_to_anchor(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_66805 = {}
        # Getting the type of 'self' (line 93)
        self_66802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'self', False)
        # Obtaining the member 'legend' of a type (line 93)
        legend_66803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 19), self_66802, 'legend')
        # Obtaining the member 'get_bbox_to_anchor' of a type (line 93)
        get_bbox_to_anchor_66804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 19), legend_66803, 'get_bbox_to_anchor')
        # Calling get_bbox_to_anchor(args, kwargs) (line 93)
        get_bbox_to_anchor_call_result_66806 = invoke(stypy.reporting.localization.Localization(__file__, 93, 19), get_bbox_to_anchor_66804, *[], **kwargs_66805)
        
        # Assigning a type to the variable 'bbox' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'bbox', get_bbox_to_anchor_call_result_66806)
        # SSA join for if statement (line 91)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to BboxTransformFrom(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'bbox' (line 95)
        bbox_66808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 44), 'bbox', False)
        # Processing the call keyword arguments (line 95)
        kwargs_66809 = {}
        # Getting the type of 'BboxTransformFrom' (line 95)
        BboxTransformFrom_66807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'BboxTransformFrom', False)
        # Calling BboxTransformFrom(args, kwargs) (line 95)
        BboxTransformFrom_call_result_66810 = invoke(stypy.reporting.localization.Localization(__file__, 95, 26), BboxTransformFrom_66807, *[bbox_66808], **kwargs_66809)
        
        # Assigning a type to the variable '_bbox_transform' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), '_bbox_transform', BboxTransformFrom_call_result_66810)
        
        # Assigning a Call to a Attribute (line 96):
        
        # Assigning a Call to a Attribute (line 96):
        
        # Assigning a Call to a Attribute (line 96):
        
        # Call to tuple(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to transform_point(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'loc_in_canvas' (line 97)
        loc_in_canvas_66814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 44), 'loc_in_canvas', False)
        # Processing the call keyword arguments (line 97)
        kwargs_66815 = {}
        # Getting the type of '_bbox_transform' (line 97)
        _bbox_transform_66812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), '_bbox_transform', False)
        # Obtaining the member 'transform_point' of a type (line 97)
        transform_point_66813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), _bbox_transform_66812, 'transform_point')
        # Calling transform_point(args, kwargs) (line 97)
        transform_point_call_result_66816 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), transform_point_66813, *[loc_in_canvas_66814], **kwargs_66815)
        
        # Processing the call keyword arguments (line 96)
        kwargs_66817 = {}
        # Getting the type of 'tuple' (line 96)
        tuple_66811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'tuple', False)
        # Calling tuple(args, kwargs) (line 96)
        tuple_call_result_66818 = invoke(stypy.reporting.localization.Localization(__file__, 96, 27), tuple_66811, *[transform_point_call_result_66816], **kwargs_66817)
        
        # Getting the type of 'self' (line 96)
        self_66819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Obtaining the member 'legend' of a type (line 96)
        legend_66820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_66819, 'legend')
        # Setting the type of the member '_loc' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), legend_66820, '_loc', tuple_call_result_66818)
        
        # ################# End of '_update_loc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update_loc' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_66821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_66821)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update_loc'
        return stypy_return_type_66821


    @norecursion
    def _update_bbox_to_anchor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update_bbox_to_anchor'
        module_type_store = module_type_store.open_function_context('_update_bbox_to_anchor', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DraggableLegend._update_bbox_to_anchor.__dict__.__setitem__('stypy_localization', localization)
        DraggableLegend._update_bbox_to_anchor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DraggableLegend._update_bbox_to_anchor.__dict__.__setitem__('stypy_type_store', module_type_store)
        DraggableLegend._update_bbox_to_anchor.__dict__.__setitem__('stypy_function_name', 'DraggableLegend._update_bbox_to_anchor')
        DraggableLegend._update_bbox_to_anchor.__dict__.__setitem__('stypy_param_names_list', ['loc_in_canvas'])
        DraggableLegend._update_bbox_to_anchor.__dict__.__setitem__('stypy_varargs_param_name', None)
        DraggableLegend._update_bbox_to_anchor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DraggableLegend._update_bbox_to_anchor.__dict__.__setitem__('stypy_call_defaults', defaults)
        DraggableLegend._update_bbox_to_anchor.__dict__.__setitem__('stypy_call_varargs', varargs)
        DraggableLegend._update_bbox_to_anchor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DraggableLegend._update_bbox_to_anchor.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DraggableLegend._update_bbox_to_anchor', ['loc_in_canvas'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update_bbox_to_anchor', localization, ['loc_in_canvas'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update_bbox_to_anchor(...)' code ##################

        
        # Assigning a Attribute to a Name (line 102):
        
        # Assigning a Attribute to a Name (line 102):
        
        # Assigning a Attribute to a Name (line 102):
        # Getting the type of 'self' (line 102)
        self_66822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'self')
        # Obtaining the member 'legend' of a type (line 102)
        legend_66823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 13), self_66822, 'legend')
        # Obtaining the member 'axes' of a type (line 102)
        axes_66824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 13), legend_66823, 'axes')
        # Obtaining the member 'transAxes' of a type (line 102)
        transAxes_66825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 13), axes_66824, 'transAxes')
        # Assigning a type to the variable 'tr' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'tr', transAxes_66825)
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to transform_point(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'loc_in_canvas' (line 103)
        loc_in_canvas_66828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 41), 'loc_in_canvas', False)
        # Processing the call keyword arguments (line 103)
        kwargs_66829 = {}
        # Getting the type of 'tr' (line 103)
        tr_66826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'tr', False)
        # Obtaining the member 'transform_point' of a type (line 103)
        transform_point_66827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 22), tr_66826, 'transform_point')
        # Calling transform_point(args, kwargs) (line 103)
        transform_point_call_result_66830 = invoke(stypy.reporting.localization.Localization(__file__, 103, 22), transform_point_66827, *[loc_in_canvas_66828], **kwargs_66829)
        
        # Assigning a type to the variable 'loc_in_bbox' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'loc_in_bbox', transform_point_call_result_66830)
        
        # Call to set_bbox_to_anchor(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'loc_in_bbox' (line 105)
        loc_in_bbox_66834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 39), 'loc_in_bbox', False)
        # Processing the call keyword arguments (line 105)
        kwargs_66835 = {}
        # Getting the type of 'self' (line 105)
        self_66831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self', False)
        # Obtaining the member 'legend' of a type (line 105)
        legend_66832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_66831, 'legend')
        # Obtaining the member 'set_bbox_to_anchor' of a type (line 105)
        set_bbox_to_anchor_66833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), legend_66832, 'set_bbox_to_anchor')
        # Calling set_bbox_to_anchor(args, kwargs) (line 105)
        set_bbox_to_anchor_call_result_66836 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), set_bbox_to_anchor_66833, *[loc_in_bbox_66834], **kwargs_66835)
        
        
        # ################# End of '_update_bbox_to_anchor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update_bbox_to_anchor' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_66837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_66837)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update_bbox_to_anchor'
        return stypy_return_type_66837


# Assigning a type to the variable 'DraggableLegend' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'DraggableLegend', DraggableLegend)
# Declaration of the 'Legend' class
# Getting the type of 'Artist' (line 108)
Artist_66838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'Artist')

class Legend(Artist_66838, ):
    unicode_66839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, (-1)), 'unicode', u"\n    Place a legend on the axes at location loc.  Labels are a\n    sequence of strings and loc can be a string or an integer\n    specifying the legend location\n\n    The location codes are::\n\n      'best'         : 0, (only implemented for axes legends)\n      'upper right'  : 1,\n      'upper left'   : 2,\n      'lower left'   : 3,\n      'lower right'  : 4,\n      'right'        : 5, (same as 'center right', for back-compatibility)\n      'center left'  : 6,\n      'center right' : 7,\n      'lower center' : 8,\n      'upper center' : 9,\n      'center'       : 10,\n\n    loc can be a tuple of the normalized coordinate values with\n    respect its parent.\n\n    ")
    
    # Assigning a Dict to a Name (line 132):
    
    # Assigning a Dict to a Name (line 132):
    
    # Assigning a Num to a Name (line 145):
    
    # Assigning a Num to a Name (line 145):

    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Legend.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Legend.stypy__str__')
        Legend.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        unicode_66840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 15), 'unicode', u'Legend')
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type', unicode_66840)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_66841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_66841)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_66841


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 151)
        None_66842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'None')
        # Getting the type of 'None' (line 152)
        None_66843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 27), 'None')
        # Getting the type of 'None' (line 153)
        None_66844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 29), 'None')
        # Getting the type of 'True' (line 155)
        True_66845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 29), 'True')
        # Getting the type of 'None' (line 157)
        None_66846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 31), 'None')
        # Getting the type of 'None' (line 158)
        None_66847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'None')
        # Getting the type of 'None' (line 159)
        None_66848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'None')
        # Getting the type of 'None' (line 160)
        None_66849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'None')
        # Getting the type of 'None' (line 163)
        None_66850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'None')
        # Getting the type of 'None' (line 164)
        None_66851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 30), 'None')
        # Getting the type of 'None' (line 166)
        None_66852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 30), 'None')
        # Getting the type of 'None' (line 167)
        None_66853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'None')
        # Getting the type of 'None' (line 168)
        None_66854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'None')
        # Getting the type of 'None' (line 170)
        None_66855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 31), 'None')
        # Getting the type of 'None' (line 172)
        None_66856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 31), 'None')
        int_66857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 22), 'int')
        # Getting the type of 'None' (line 175)
        None_66858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 22), 'None')
        # Getting the type of 'None' (line 178)
        None_66859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 26), 'None')
        # Getting the type of 'None' (line 180)
        None_66860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'None')
        # Getting the type of 'None' (line 181)
        None_66861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 23), 'None')
        # Getting the type of 'None' (line 183)
        None_66862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'None')
        # Getting the type of 'None' (line 184)
        None_66863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'None')
        # Getting the type of 'None' (line 185)
        None_66864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 27), 'None')
        # Getting the type of 'None' (line 187)
        None_66865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 32), 'None')
        # Getting the type of 'None' (line 188)
        None_66866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 32), 'None')
        # Getting the type of 'None' (line 189)
        None_66867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'None')
        # Getting the type of 'None' (line 190)
        None_66868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 29), 'None')
        defaults = [None_66842, None_66843, None_66844, True_66845, None_66846, None_66847, None_66848, None_66849, None_66850, None_66851, None_66852, None_66853, None_66854, None_66855, None_66856, int_66857, None_66858, None_66859, None_66860, None_66861, None_66862, None_66863, None_66864, None_66865, None_66866, None_66867, None_66868]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.__init__', ['parent', 'handles', 'labels', 'loc', 'numpoints', 'markerscale', 'markerfirst', 'scatterpoints', 'scatteryoffsets', 'prop', 'fontsize', 'borderpad', 'labelspacing', 'handlelength', 'handleheight', 'handletextpad', 'borderaxespad', 'columnspacing', 'ncol', 'mode', 'fancybox', 'shadow', 'title', 'framealpha', 'edgecolor', 'facecolor', 'bbox_to_anchor', 'bbox_transform', 'frameon', 'handler_map'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['parent', 'handles', 'labels', 'loc', 'numpoints', 'markerscale', 'markerfirst', 'scatterpoints', 'scatteryoffsets', 'prop', 'fontsize', 'borderpad', 'labelspacing', 'handlelength', 'handleheight', 'handletextpad', 'borderaxespad', 'columnspacing', 'ncol', 'mode', 'fancybox', 'shadow', 'title', 'framealpha', 'edgecolor', 'facecolor', 'bbox_to_anchor', 'bbox_transform', 'frameon', 'handler_map'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_66869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, (-1)), 'unicode', u'\n        - *parent*: the artist that contains the legend\n        - *handles*: a list of artists (lines, patches) to be added to the\n                      legend\n        - *labels*: a list of strings to label the legend\n\n        Optional keyword arguments:\n\n        ================   ====================================================\n        Keyword            Description\n        ================   ====================================================\n        loc                Location code string, or tuple (see below).\n        prop               the font property\n        fontsize           the font size (used only if prop is not specified)\n        markerscale        the relative size of legend markers vs. original\n        markerfirst        If True (default), marker is to left of the label.\n        numpoints          the number of points in the legend for line\n        scatterpoints      the number of points in the legend for scatter plot\n        scatteryoffsets    a list of yoffsets for scatter symbols in legend\n        frameon            If True, draw the legend on a patch (frame).\n        fancybox           If True, draw the frame with a round fancybox.\n        shadow             If True, draw a shadow behind legend.\n        framealpha         Transparency of the frame.\n        edgecolor          Frame edgecolor.\n        facecolor          Frame facecolor.\n        ncol               number of columns\n        borderpad          the fractional whitespace inside the legend border\n        labelspacing       the vertical space between the legend entries\n        handlelength       the length of the legend handles\n        handleheight       the height of the legend handles\n        handletextpad      the pad between the legend handle and text\n        borderaxespad      the pad between the axes and legend border\n        columnspacing      the spacing between columns\n        title              the legend title\n        bbox_to_anchor     the bbox that the legend will be anchored.\n        bbox_transform     the transform for the bbox. transAxes if None.\n        ================   ====================================================\n\n\n        The pad and spacing parameters are measured in font-size units.  e.g.,\n        a fontsize of 10 points and a handlelength=5 implies a handlelength of\n        50 points.  Values from rcParams will be used if None.\n\n        Users can specify any arbitrary location for the legend using the\n        *bbox_to_anchor* keyword argument. bbox_to_anchor can be an instance\n        of BboxBase(or its derivatives) or a tuple of 2 or 4 floats.\n        See :meth:`set_bbox_to_anchor` for more detail.\n\n        The legend location can be specified by setting *loc* with a tuple of\n        2 floats, which is interpreted as the lower-left corner of the legend\n        in the normalized axes coordinate.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 245, 8))
        
        # 'from matplotlib.axes import Axes' statement (line 245)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
        import_66870 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 245, 8), 'matplotlib.axes')

        if (type(import_66870) is not StypyTypeError):

            if (import_66870 != 'pyd_module'):
                __import__(import_66870)
                sys_modules_66871 = sys.modules[import_66870]
                import_from_module(stypy.reporting.localization.Localization(__file__, 245, 8), 'matplotlib.axes', sys_modules_66871.module_type_store, module_type_store, ['Axes'])
                nest_module(stypy.reporting.localization.Localization(__file__, 245, 8), __file__, sys_modules_66871, sys_modules_66871.module_type_store, module_type_store)
            else:
                from matplotlib.axes import Axes

                import_from_module(stypy.reporting.localization.Localization(__file__, 245, 8), 'matplotlib.axes', None, module_type_store, ['Axes'], [Axes])

        else:
            # Assigning a type to the variable 'matplotlib.axes' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'matplotlib.axes', import_66870)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 246, 8))
        
        # 'from matplotlib.figure import Figure' statement (line 246)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
        import_66872 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 246, 8), 'matplotlib.figure')

        if (type(import_66872) is not StypyTypeError):

            if (import_66872 != 'pyd_module'):
                __import__(import_66872)
                sys_modules_66873 = sys.modules[import_66872]
                import_from_module(stypy.reporting.localization.Localization(__file__, 246, 8), 'matplotlib.figure', sys_modules_66873.module_type_store, module_type_store, ['Figure'])
                nest_module(stypy.reporting.localization.Localization(__file__, 246, 8), __file__, sys_modules_66873, sys_modules_66873.module_type_store, module_type_store)
            else:
                from matplotlib.figure import Figure

                import_from_module(stypy.reporting.localization.Localization(__file__, 246, 8), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

        else:
            # Assigning a type to the variable 'matplotlib.figure' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'matplotlib.figure', import_66872)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')
        
        
        # Call to __init__(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'self' (line 248)
        self_66876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'self', False)
        # Processing the call keyword arguments (line 248)
        kwargs_66877 = {}
        # Getting the type of 'Artist' (line 248)
        Artist_66874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'Artist', False)
        # Obtaining the member '__init__' of a type (line 248)
        init___66875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), Artist_66874, '__init__')
        # Calling __init__(args, kwargs) (line 248)
        init___call_result_66878 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), init___66875, *[self_66876], **kwargs_66877)
        
        
        # Type idiom detected: calculating its left and rigth part (line 250)
        # Getting the type of 'prop' (line 250)
        prop_66879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 11), 'prop')
        # Getting the type of 'None' (line 250)
        None_66880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 19), 'None')
        
        (may_be_66881, more_types_in_union_66882) = may_be_none(prop_66879, None_66880)

        if may_be_66881:

            if more_types_in_union_66882:
                # Runtime conditional SSA (line 250)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 251)
            # Getting the type of 'fontsize' (line 251)
            fontsize_66883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'fontsize')
            # Getting the type of 'None' (line 251)
            None_66884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 31), 'None')
            
            (may_be_66885, more_types_in_union_66886) = may_not_be_none(fontsize_66883, None_66884)

            if may_be_66885:

                if more_types_in_union_66886:
                    # Runtime conditional SSA (line 251)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Attribute (line 252):
                
                # Assigning a Call to a Attribute (line 252):
                
                # Assigning a Call to a Attribute (line 252):
                
                # Call to FontProperties(...): (line 252)
                # Processing the call keyword arguments (line 252)
                # Getting the type of 'fontsize' (line 252)
                fontsize_66888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 48), 'fontsize', False)
                keyword_66889 = fontsize_66888
                kwargs_66890 = {'size': keyword_66889}
                # Getting the type of 'FontProperties' (line 252)
                FontProperties_66887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 28), 'FontProperties', False)
                # Calling FontProperties(args, kwargs) (line 252)
                FontProperties_call_result_66891 = invoke(stypy.reporting.localization.Localization(__file__, 252, 28), FontProperties_66887, *[], **kwargs_66890)
                
                # Getting the type of 'self' (line 252)
                self_66892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'self')
                # Setting the type of the member 'prop' of a type (line 252)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), self_66892, 'prop', FontProperties_call_result_66891)

                if more_types_in_union_66886:
                    # Runtime conditional SSA for else branch (line 251)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_66885) or more_types_in_union_66886):
                
                # Assigning a Call to a Attribute (line 254):
                
                # Assigning a Call to a Attribute (line 254):
                
                # Assigning a Call to a Attribute (line 254):
                
                # Call to FontProperties(...): (line 254)
                # Processing the call keyword arguments (line 254)
                
                # Obtaining the type of the subscript
                unicode_66894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 57), 'unicode', u'legend.fontsize')
                # Getting the type of 'rcParams' (line 254)
                rcParams_66895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 48), 'rcParams', False)
                # Obtaining the member '__getitem__' of a type (line 254)
                getitem___66896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 48), rcParams_66895, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 254)
                subscript_call_result_66897 = invoke(stypy.reporting.localization.Localization(__file__, 254, 48), getitem___66896, unicode_66894)
                
                keyword_66898 = subscript_call_result_66897
                kwargs_66899 = {'size': keyword_66898}
                # Getting the type of 'FontProperties' (line 254)
                FontProperties_66893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 28), 'FontProperties', False)
                # Calling FontProperties(args, kwargs) (line 254)
                FontProperties_call_result_66900 = invoke(stypy.reporting.localization.Localization(__file__, 254, 28), FontProperties_66893, *[], **kwargs_66899)
                
                # Getting the type of 'self' (line 254)
                self_66901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'self')
                # Setting the type of the member 'prop' of a type (line 254)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 16), self_66901, 'prop', FontProperties_call_result_66900)

                if (may_be_66885 and more_types_in_union_66886):
                    # SSA join for if statement (line 251)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_66882:
                # Runtime conditional SSA for else branch (line 250)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_66881) or more_types_in_union_66882):
            
            # Type idiom detected: calculating its left and rigth part (line 255)
            # Getting the type of 'dict' (line 255)
            dict_66902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 30), 'dict')
            # Getting the type of 'prop' (line 255)
            prop_66903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'prop')
            
            (may_be_66904, more_types_in_union_66905) = may_be_subtype(dict_66902, prop_66903)

            if may_be_66904:

                if more_types_in_union_66905:
                    # Runtime conditional SSA (line 255)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'prop' (line 255)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 13), 'prop', remove_not_subtype_from_union(prop_66903, dict))
                
                # Assigning a Call to a Attribute (line 256):
                
                # Assigning a Call to a Attribute (line 256):
                
                # Assigning a Call to a Attribute (line 256):
                
                # Call to FontProperties(...): (line 256)
                # Processing the call keyword arguments (line 256)
                # Getting the type of 'prop' (line 256)
                prop_66907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 41), 'prop', False)
                kwargs_66908 = {'prop_66907': prop_66907}
                # Getting the type of 'FontProperties' (line 256)
                FontProperties_66906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 24), 'FontProperties', False)
                # Calling FontProperties(args, kwargs) (line 256)
                FontProperties_call_result_66909 = invoke(stypy.reporting.localization.Localization(__file__, 256, 24), FontProperties_66906, *[], **kwargs_66908)
                
                # Getting the type of 'self' (line 256)
                self_66910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'self')
                # Setting the type of the member 'prop' of a type (line 256)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), self_66910, 'prop', FontProperties_call_result_66909)
                
                
                unicode_66911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 15), 'unicode', u'size')
                # Getting the type of 'prop' (line 257)
                prop_66912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 29), 'prop')
                # Applying the binary operator 'notin' (line 257)
                result_contains_66913 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 15), 'notin', unicode_66911, prop_66912)
                
                # Testing the type of an if condition (line 257)
                if_condition_66914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 12), result_contains_66913)
                # Assigning a type to the variable 'if_condition_66914' (line 257)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'if_condition_66914', if_condition_66914)
                # SSA begins for if statement (line 257)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to set_size(...): (line 258)
                # Processing the call arguments (line 258)
                
                # Obtaining the type of the subscript
                unicode_66918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 44), 'unicode', u'legend.fontsize')
                # Getting the type of 'rcParams' (line 258)
                rcParams_66919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 35), 'rcParams', False)
                # Obtaining the member '__getitem__' of a type (line 258)
                getitem___66920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 35), rcParams_66919, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 258)
                subscript_call_result_66921 = invoke(stypy.reporting.localization.Localization(__file__, 258, 35), getitem___66920, unicode_66918)
                
                # Processing the call keyword arguments (line 258)
                kwargs_66922 = {}
                # Getting the type of 'self' (line 258)
                self_66915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'self', False)
                # Obtaining the member 'prop' of a type (line 258)
                prop_66916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 16), self_66915, 'prop')
                # Obtaining the member 'set_size' of a type (line 258)
                set_size_66917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 16), prop_66916, 'set_size')
                # Calling set_size(args, kwargs) (line 258)
                set_size_call_result_66923 = invoke(stypy.reporting.localization.Localization(__file__, 258, 16), set_size_66917, *[subscript_call_result_66921], **kwargs_66922)
                
                # SSA join for if statement (line 257)
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_66905:
                    # Runtime conditional SSA for else branch (line 255)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_66904) or more_types_in_union_66905):
                # Assigning a type to the variable 'prop' (line 255)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 13), 'prop', remove_subtype_from_union(prop_66903, dict))
                
                # Assigning a Name to a Attribute (line 260):
                
                # Assigning a Name to a Attribute (line 260):
                
                # Assigning a Name to a Attribute (line 260):
                # Getting the type of 'prop' (line 260)
                prop_66924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), 'prop')
                # Getting the type of 'self' (line 260)
                self_66925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'self')
                # Setting the type of the member 'prop' of a type (line 260)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), self_66925, 'prop', prop_66924)

                if (may_be_66904 and more_types_in_union_66905):
                    # SSA join for if statement (line 255)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_66881 and more_types_in_union_66882):
                # SSA join for if statement (line 250)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 262):
        
        # Assigning a Call to a Attribute (line 262):
        
        # Assigning a Call to a Attribute (line 262):
        
        # Call to get_size_in_points(...): (line 262)
        # Processing the call keyword arguments (line 262)
        kwargs_66929 = {}
        # Getting the type of 'self' (line 262)
        self_66926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'self', False)
        # Obtaining the member 'prop' of a type (line 262)
        prop_66927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 25), self_66926, 'prop')
        # Obtaining the member 'get_size_in_points' of a type (line 262)
        get_size_in_points_66928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 25), prop_66927, 'get_size_in_points')
        # Calling get_size_in_points(args, kwargs) (line 262)
        get_size_in_points_call_result_66930 = invoke(stypy.reporting.localization.Localization(__file__, 262, 25), get_size_in_points_66928, *[], **kwargs_66929)
        
        # Getting the type of 'self' (line 262)
        self_66931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self')
        # Setting the type of the member '_fontsize' of a type (line 262)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_66931, '_fontsize', get_size_in_points_call_result_66930)
        
        # Assigning a List to a Attribute (line 264):
        
        # Assigning a List to a Attribute (line 264):
        
        # Assigning a List to a Attribute (line 264):
        
        # Obtaining an instance of the builtin type 'list' (line 264)
        list_66932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 264)
        
        # Getting the type of 'self' (line 264)
        self_66933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'self')
        # Setting the type of the member 'texts' of a type (line 264)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_66933, 'texts', list_66932)
        
        # Assigning a List to a Attribute (line 265):
        
        # Assigning a List to a Attribute (line 265):
        
        # Assigning a List to a Attribute (line 265):
        
        # Obtaining an instance of the builtin type 'list' (line 265)
        list_66934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 265)
        
        # Getting the type of 'self' (line 265)
        self_66935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'self')
        # Setting the type of the member 'legendHandles' of a type (line 265)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), self_66935, 'legendHandles', list_66934)
        
        # Assigning a Name to a Attribute (line 266):
        
        # Assigning a Name to a Attribute (line 266):
        
        # Assigning a Name to a Attribute (line 266):
        # Getting the type of 'None' (line 266)
        None_66936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 33), 'None')
        # Getting the type of 'self' (line 266)
        self_66937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'self')
        # Setting the type of the member '_legend_title_box' of a type (line 266)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), self_66937, '_legend_title_box', None_66936)
        
        # Assigning a Name to a Attribute (line 270):
        
        # Assigning a Name to a Attribute (line 270):
        
        # Assigning a Name to a Attribute (line 270):
        # Getting the type of 'handler_map' (line 270)
        handler_map_66938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 35), 'handler_map')
        # Getting the type of 'self' (line 270)
        self_66939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self')
        # Setting the type of the member '_custom_handler_map' of a type (line 270)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_66939, '_custom_handler_map', handler_map_66938)
        
        # Assigning a Call to a Name (line 272):
        
        # Assigning a Call to a Name (line 272):
        
        # Assigning a Call to a Name (line 272):
        
        # Call to locals(...): (line 272)
        # Processing the call keyword arguments (line 272)
        kwargs_66941 = {}
        # Getting the type of 'locals' (line 272)
        locals_66940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'locals', False)
        # Calling locals(args, kwargs) (line 272)
        locals_call_result_66942 = invoke(stypy.reporting.localization.Localization(__file__, 272, 22), locals_66940, *[], **kwargs_66941)
        
        # Assigning a type to the variable 'locals_view' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'locals_view', locals_call_result_66942)
        
        
        # Obtaining an instance of the builtin type 'list' (line 273)
        list_66943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 273)
        # Adding element type (line 273)
        unicode_66944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 21), 'unicode', u'numpoints')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_66943, unicode_66944)
        # Adding element type (line 273)
        unicode_66945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 34), 'unicode', u'markerscale')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_66943, unicode_66945)
        # Adding element type (line 273)
        unicode_66946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 49), 'unicode', u'shadow')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_66943, unicode_66946)
        # Adding element type (line 273)
        unicode_66947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 59), 'unicode', u'columnspacing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_66943, unicode_66947)
        # Adding element type (line 273)
        unicode_66948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 21), 'unicode', u'scatterpoints')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_66943, unicode_66948)
        # Adding element type (line 273)
        unicode_66949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 38), 'unicode', u'handleheight')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_66943, unicode_66949)
        # Adding element type (line 273)
        unicode_66950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 54), 'unicode', u'borderpad')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_66943, unicode_66950)
        # Adding element type (line 273)
        unicode_66951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 21), 'unicode', u'labelspacing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_66943, unicode_66951)
        # Adding element type (line 273)
        unicode_66952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 37), 'unicode', u'handlelength')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_66943, unicode_66952)
        # Adding element type (line 273)
        unicode_66953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 53), 'unicode', u'handletextpad')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_66943, unicode_66953)
        # Adding element type (line 273)
        unicode_66954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 21), 'unicode', u'borderaxespad')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 20), list_66943, unicode_66954)
        
        # Testing the type of a for loop iterable (line 273)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 273, 8), list_66943)
        # Getting the type of the for loop variable (line 273)
        for_loop_var_66955 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 273, 8), list_66943)
        # Assigning a type to the variable 'name' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'name', for_loop_var_66955)
        # SSA begins for a for statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 277)
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 277)
        name_66956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 27), 'name')
        # Getting the type of 'locals_view' (line 277)
        locals_view_66957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 15), 'locals_view')
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___66958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 15), locals_view_66957, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_66959 = invoke(stypy.reporting.localization.Localization(__file__, 277, 15), getitem___66958, name_66956)
        
        # Getting the type of 'None' (line 277)
        None_66960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 36), 'None')
        
        (may_be_66961, more_types_in_union_66962) = may_be_none(subscript_call_result_66959, None_66960)

        if may_be_66961:

            if more_types_in_union_66962:
                # Runtime conditional SSA (line 277)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 278):
            
            # Assigning a Subscript to a Name (line 278):
            
            # Assigning a Subscript to a Name (line 278):
            
            # Obtaining the type of the subscript
            unicode_66963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 33), 'unicode', u'legend.')
            # Getting the type of 'name' (line 278)
            name_66964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 45), 'name')
            # Applying the binary operator '+' (line 278)
            result_add_66965 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 33), '+', unicode_66963, name_66964)
            
            # Getting the type of 'rcParams' (line 278)
            rcParams_66966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 278)
            getitem___66967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 24), rcParams_66966, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 278)
            subscript_call_result_66968 = invoke(stypy.reporting.localization.Localization(__file__, 278, 24), getitem___66967, result_add_66965)
            
            # Assigning a type to the variable 'value' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'value', subscript_call_result_66968)

            if more_types_in_union_66962:
                # Runtime conditional SSA for else branch (line 277)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_66961) or more_types_in_union_66962):
            
            # Assigning a Subscript to a Name (line 280):
            
            # Assigning a Subscript to a Name (line 280):
            
            # Assigning a Subscript to a Name (line 280):
            
            # Obtaining the type of the subscript
            # Getting the type of 'name' (line 280)
            name_66969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 36), 'name')
            # Getting the type of 'locals_view' (line 280)
            locals_view_66970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'locals_view')
            # Obtaining the member '__getitem__' of a type (line 280)
            getitem___66971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 24), locals_view_66970, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 280)
            subscript_call_result_66972 = invoke(stypy.reporting.localization.Localization(__file__, 280, 24), getitem___66971, name_66969)
            
            # Assigning a type to the variable 'value' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'value', subscript_call_result_66972)

            if (may_be_66961 and more_types_in_union_66962):
                # SSA join for if statement (line 277)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to setattr(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'self' (line 281)
        self_66974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 'self', False)
        # Getting the type of 'name' (line 281)
        name_66975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 26), 'name', False)
        # Getting the type of 'value' (line 281)
        value_66976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 32), 'value', False)
        # Processing the call keyword arguments (line 281)
        kwargs_66977 = {}
        # Getting the type of 'setattr' (line 281)
        setattr_66973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 281)
        setattr_call_result_66978 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), setattr_66973, *[self_66974, name_66975, value_66976], **kwargs_66977)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 282, 8), module_type_store, 'locals_view')
        
        # Assigning a Call to a Name (line 284):
        
        # Assigning a Call to a Name (line 284):
        
        # Assigning a Call to a Name (line 284):
        
        # Call to list(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'handles' (line 284)
        handles_66980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'handles', False)
        # Processing the call keyword arguments (line 284)
        kwargs_66981 = {}
        # Getting the type of 'list' (line 284)
        list_66979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 18), 'list', False)
        # Calling list(args, kwargs) (line 284)
        list_call_result_66982 = invoke(stypy.reporting.localization.Localization(__file__, 284, 18), list_66979, *[handles_66980], **kwargs_66981)
        
        # Assigning a type to the variable 'handles' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'handles', list_call_result_66982)
        
        
        
        # Call to len(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'handles' (line 285)
        handles_66984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'handles', False)
        # Processing the call keyword arguments (line 285)
        kwargs_66985 = {}
        # Getting the type of 'len' (line 285)
        len_66983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'len', False)
        # Calling len(args, kwargs) (line 285)
        len_call_result_66986 = invoke(stypy.reporting.localization.Localization(__file__, 285, 11), len_66983, *[handles_66984], **kwargs_66985)
        
        int_66987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 26), 'int')
        # Applying the binary operator '<' (line 285)
        result_lt_66988 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 11), '<', len_call_result_66986, int_66987)
        
        # Testing the type of an if condition (line 285)
        if_condition_66989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 8), result_lt_66988)
        # Assigning a type to the variable 'if_condition_66989' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'if_condition_66989', if_condition_66989)
        # SSA begins for if statement (line 285)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 286):
        
        # Assigning a Num to a Name (line 286):
        
        # Assigning a Num to a Name (line 286):
        int_66990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 19), 'int')
        # Assigning a type to the variable 'ncol' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'ncol', int_66990)
        # SSA join for if statement (line 285)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 287):
        
        # Assigning a Name to a Attribute (line 287):
        
        # Assigning a Name to a Attribute (line 287):
        # Getting the type of 'ncol' (line 287)
        ncol_66991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 21), 'ncol')
        # Getting the type of 'self' (line 287)
        self_66992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self')
        # Setting the type of the member '_ncol' of a type (line 287)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_66992, '_ncol', ncol_66991)
        
        
        # Getting the type of 'self' (line 289)
        self_66993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 11), 'self')
        # Obtaining the member 'numpoints' of a type (line 289)
        numpoints_66994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 11), self_66993, 'numpoints')
        int_66995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 29), 'int')
        # Applying the binary operator '<=' (line 289)
        result_le_66996 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 11), '<=', numpoints_66994, int_66995)
        
        # Testing the type of an if condition (line 289)
        if_condition_66997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 8), result_le_66996)
        # Assigning a type to the variable 'if_condition_66997' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'if_condition_66997', if_condition_66997)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 290)
        # Processing the call arguments (line 290)
        unicode_66999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 29), 'unicode', u'numpoints must be > 0; it was %d')
        # Getting the type of 'numpoints' (line 290)
        numpoints_67000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 66), 'numpoints', False)
        # Applying the binary operator '%' (line 290)
        result_mod_67001 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 29), '%', unicode_66999, numpoints_67000)
        
        # Processing the call keyword arguments (line 290)
        kwargs_67002 = {}
        # Getting the type of 'ValueError' (line 290)
        ValueError_66998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 290)
        ValueError_call_result_67003 = invoke(stypy.reporting.localization.Localization(__file__, 290, 18), ValueError_66998, *[result_mod_67001], **kwargs_67002)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 290, 12), ValueError_call_result_67003, 'raise parameter', BaseException)
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 293)
        # Getting the type of 'scatteryoffsets' (line 293)
        scatteryoffsets_67004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 11), 'scatteryoffsets')
        # Getting the type of 'None' (line 293)
        None_67005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), 'None')
        
        (may_be_67006, more_types_in_union_67007) = may_be_none(scatteryoffsets_67004, None_67005)

        if may_be_67006:

            if more_types_in_union_67007:
                # Runtime conditional SSA (line 293)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 294):
            
            # Assigning a Call to a Attribute (line 294):
            
            # Assigning a Call to a Attribute (line 294):
            
            # Call to array(...): (line 294)
            # Processing the call arguments (line 294)
            
            # Obtaining an instance of the builtin type 'list' (line 294)
            list_67010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 45), 'list')
            # Adding type elements to the builtin type 'list' instance (line 294)
            # Adding element type (line 294)
            float_67011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 46), 'float')
            float_67012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 51), 'float')
            # Applying the binary operator 'div' (line 294)
            result_div_67013 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 46), 'div', float_67011, float_67012)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 45), list_67010, result_div_67013)
            # Adding element type (line 294)
            float_67014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 55), 'float')
            float_67015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 60), 'float')
            # Applying the binary operator 'div' (line 294)
            result_div_67016 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 55), 'div', float_67014, float_67015)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 45), list_67010, result_div_67016)
            # Adding element type (line 294)
            float_67017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 64), 'float')
            float_67018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 70), 'float')
            # Applying the binary operator 'div' (line 294)
            result_div_67019 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 64), 'div', float_67017, float_67018)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 45), list_67010, result_div_67019)
            
            # Processing the call keyword arguments (line 294)
            kwargs_67020 = {}
            # Getting the type of 'np' (line 294)
            np_67008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 36), 'np', False)
            # Obtaining the member 'array' of a type (line 294)
            array_67009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 36), np_67008, 'array')
            # Calling array(args, kwargs) (line 294)
            array_call_result_67021 = invoke(stypy.reporting.localization.Localization(__file__, 294, 36), array_67009, *[list_67010], **kwargs_67020)
            
            # Getting the type of 'self' (line 294)
            self_67022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'self')
            # Setting the type of the member '_scatteryoffsets' of a type (line 294)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), self_67022, '_scatteryoffsets', array_call_result_67021)

            if more_types_in_union_67007:
                # Runtime conditional SSA for else branch (line 293)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_67006) or more_types_in_union_67007):
            
            # Assigning a Call to a Attribute (line 296):
            
            # Assigning a Call to a Attribute (line 296):
            
            # Assigning a Call to a Attribute (line 296):
            
            # Call to asarray(...): (line 296)
            # Processing the call arguments (line 296)
            # Getting the type of 'scatteryoffsets' (line 296)
            scatteryoffsets_67025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 47), 'scatteryoffsets', False)
            # Processing the call keyword arguments (line 296)
            kwargs_67026 = {}
            # Getting the type of 'np' (line 296)
            np_67023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 36), 'np', False)
            # Obtaining the member 'asarray' of a type (line 296)
            asarray_67024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 36), np_67023, 'asarray')
            # Calling asarray(args, kwargs) (line 296)
            asarray_call_result_67027 = invoke(stypy.reporting.localization.Localization(__file__, 296, 36), asarray_67024, *[scatteryoffsets_67025], **kwargs_67026)
            
            # Getting the type of 'self' (line 296)
            self_67028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'self')
            # Setting the type of the member '_scatteryoffsets' of a type (line 296)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), self_67028, '_scatteryoffsets', asarray_call_result_67027)

            if (may_be_67006 and more_types_in_union_67007):
                # SSA join for if statement (line 293)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 297):
        
        # Assigning a BinOp to a Name (line 297):
        
        # Assigning a BinOp to a Name (line 297):
        # Getting the type of 'self' (line 297)
        self_67029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 15), 'self')
        # Obtaining the member 'scatterpoints' of a type (line 297)
        scatterpoints_67030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 15), self_67029, 'scatterpoints')
        
        # Call to len(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'self' (line 297)
        self_67032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 41), 'self', False)
        # Obtaining the member '_scatteryoffsets' of a type (line 297)
        _scatteryoffsets_67033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 41), self_67032, '_scatteryoffsets')
        # Processing the call keyword arguments (line 297)
        kwargs_67034 = {}
        # Getting the type of 'len' (line 297)
        len_67031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 37), 'len', False)
        # Calling len(args, kwargs) (line 297)
        len_call_result_67035 = invoke(stypy.reporting.localization.Localization(__file__, 297, 37), len_67031, *[_scatteryoffsets_67033], **kwargs_67034)
        
        # Applying the binary operator '//' (line 297)
        result_floordiv_67036 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 15), '//', scatterpoints_67030, len_call_result_67035)
        
        int_67037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 66), 'int')
        # Applying the binary operator '+' (line 297)
        result_add_67038 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 15), '+', result_floordiv_67036, int_67037)
        
        # Assigning a type to the variable 'reps' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'reps', result_add_67038)
        
        # Assigning a Subscript to a Attribute (line 298):
        
        # Assigning a Subscript to a Attribute (line 298):
        
        # Assigning a Subscript to a Attribute (line 298):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 299)
        self_67039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 47), 'self')
        # Obtaining the member 'scatterpoints' of a type (line 299)
        scatterpoints_67040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 47), self_67039, 'scatterpoints')
        slice_67041 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 298, 32), None, scatterpoints_67040, None)
        
        # Call to tile(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'self' (line 298)
        self_67044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 40), 'self', False)
        # Obtaining the member '_scatteryoffsets' of a type (line 298)
        _scatteryoffsets_67045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 40), self_67044, '_scatteryoffsets')
        # Getting the type of 'reps' (line 299)
        reps_67046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 40), 'reps', False)
        # Processing the call keyword arguments (line 298)
        kwargs_67047 = {}
        # Getting the type of 'np' (line 298)
        np_67042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 32), 'np', False)
        # Obtaining the member 'tile' of a type (line 298)
        tile_67043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 32), np_67042, 'tile')
        # Calling tile(args, kwargs) (line 298)
        tile_call_result_67048 = invoke(stypy.reporting.localization.Localization(__file__, 298, 32), tile_67043, *[_scatteryoffsets_67045, reps_67046], **kwargs_67047)
        
        # Obtaining the member '__getitem__' of a type (line 298)
        getitem___67049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 32), tile_call_result_67048, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 298)
        subscript_call_result_67050 = invoke(stypy.reporting.localization.Localization(__file__, 298, 32), getitem___67049, slice_67041)
        
        # Getting the type of 'self' (line 298)
        self_67051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'self')
        # Setting the type of the member '_scatteryoffsets' of a type (line 298)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), self_67051, '_scatteryoffsets', subscript_call_result_67050)
        
        # Assigning a Name to a Attribute (line 304):
        
        # Assigning a Name to a Attribute (line 304):
        
        # Assigning a Name to a Attribute (line 304):
        # Getting the type of 'None' (line 304)
        None_67052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'None')
        # Getting the type of 'self' (line 304)
        self_67053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'self')
        # Setting the type of the member '_legend_box' of a type (line 304)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), self_67053, '_legend_box', None_67052)
        
        
        # Call to isinstance(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'parent' (line 306)
        parent_67055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 22), 'parent', False)
        # Getting the type of 'Axes' (line 306)
        Axes_67056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'Axes', False)
        # Processing the call keyword arguments (line 306)
        kwargs_67057 = {}
        # Getting the type of 'isinstance' (line 306)
        isinstance_67054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 306)
        isinstance_call_result_67058 = invoke(stypy.reporting.localization.Localization(__file__, 306, 11), isinstance_67054, *[parent_67055, Axes_67056], **kwargs_67057)
        
        # Testing the type of an if condition (line 306)
        if_condition_67059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 8), isinstance_call_result_67058)
        # Assigning a type to the variable 'if_condition_67059' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'if_condition_67059', if_condition_67059)
        # SSA begins for if statement (line 306)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 307):
        
        # Assigning a Name to a Attribute (line 307):
        
        # Assigning a Name to a Attribute (line 307):
        # Getting the type of 'True' (line 307)
        True_67060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 26), 'True')
        # Getting the type of 'self' (line 307)
        self_67061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'self')
        # Setting the type of the member 'isaxes' of a type (line 307)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), self_67061, 'isaxes', True_67060)
        
        # Assigning a Name to a Attribute (line 308):
        
        # Assigning a Name to a Attribute (line 308):
        
        # Assigning a Name to a Attribute (line 308):
        # Getting the type of 'parent' (line 308)
        parent_67062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 24), 'parent')
        # Getting the type of 'self' (line 308)
        self_67063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'self')
        # Setting the type of the member 'axes' of a type (line 308)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), self_67063, 'axes', parent_67062)
        
        # Call to set_figure(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'parent' (line 309)
        parent_67066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'parent', False)
        # Obtaining the member 'figure' of a type (line 309)
        figure_67067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 28), parent_67066, 'figure')
        # Processing the call keyword arguments (line 309)
        kwargs_67068 = {}
        # Getting the type of 'self' (line 309)
        self_67064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'self', False)
        # Obtaining the member 'set_figure' of a type (line 309)
        set_figure_67065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), self_67064, 'set_figure')
        # Calling set_figure(args, kwargs) (line 309)
        set_figure_call_result_67069 = invoke(stypy.reporting.localization.Localization(__file__, 309, 12), set_figure_67065, *[figure_67067], **kwargs_67068)
        
        # SSA branch for the else part of an if statement (line 306)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'parent' (line 310)
        parent_67071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 24), 'parent', False)
        # Getting the type of 'Figure' (line 310)
        Figure_67072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 32), 'Figure', False)
        # Processing the call keyword arguments (line 310)
        kwargs_67073 = {}
        # Getting the type of 'isinstance' (line 310)
        isinstance_67070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 310)
        isinstance_call_result_67074 = invoke(stypy.reporting.localization.Localization(__file__, 310, 13), isinstance_67070, *[parent_67071, Figure_67072], **kwargs_67073)
        
        # Testing the type of an if condition (line 310)
        if_condition_67075 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 13), isinstance_call_result_67074)
        # Assigning a type to the variable 'if_condition_67075' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 13), 'if_condition_67075', if_condition_67075)
        # SSA begins for if statement (line 310)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 311):
        
        # Assigning a Name to a Attribute (line 311):
        
        # Assigning a Name to a Attribute (line 311):
        # Getting the type of 'False' (line 311)
        False_67076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 26), 'False')
        # Getting the type of 'self' (line 311)
        self_67077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'self')
        # Setting the type of the member 'isaxes' of a type (line 311)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), self_67077, 'isaxes', False_67076)
        
        # Call to set_figure(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'parent' (line 312)
        parent_67080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 28), 'parent', False)
        # Processing the call keyword arguments (line 312)
        kwargs_67081 = {}
        # Getting the type of 'self' (line 312)
        self_67078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'self', False)
        # Obtaining the member 'set_figure' of a type (line 312)
        set_figure_67079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), self_67078, 'set_figure')
        # Calling set_figure(args, kwargs) (line 312)
        set_figure_call_result_67082 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), set_figure_67079, *[parent_67080], **kwargs_67081)
        
        # SSA branch for the else part of an if statement (line 310)
        module_type_store.open_ssa_branch('else')
        
        # Call to TypeError(...): (line 314)
        # Processing the call arguments (line 314)
        unicode_67084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 28), 'unicode', u'Legend needs either Axes or Figure as parent')
        # Processing the call keyword arguments (line 314)
        kwargs_67085 = {}
        # Getting the type of 'TypeError' (line 314)
        TypeError_67083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 314)
        TypeError_call_result_67086 = invoke(stypy.reporting.localization.Localization(__file__, 314, 18), TypeError_67083, *[unicode_67084], **kwargs_67085)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 314, 12), TypeError_call_result_67086, 'raise parameter', BaseException)
        # SSA join for if statement (line 310)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 306)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 315):
        
        # Assigning a Name to a Attribute (line 315):
        
        # Assigning a Name to a Attribute (line 315):
        # Getting the type of 'parent' (line 315)
        parent_67087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 22), 'parent')
        # Getting the type of 'self' (line 315)
        self_67088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self')
        # Setting the type of the member 'parent' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_67088, 'parent', parent_67087)
        
        # Type idiom detected: calculating its left and rigth part (line 317)
        # Getting the type of 'loc' (line 317)
        loc_67089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'loc')
        # Getting the type of 'None' (line 317)
        None_67090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 18), 'None')
        
        (may_be_67091, more_types_in_union_67092) = may_be_none(loc_67089, None_67090)

        if may_be_67091:

            if more_types_in_union_67092:
                # Runtime conditional SSA (line 317)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 318):
            
            # Assigning a Subscript to a Name (line 318):
            
            # Assigning a Subscript to a Name (line 318):
            
            # Obtaining the type of the subscript
            unicode_67093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 27), 'unicode', u'legend.loc')
            # Getting the type of 'rcParams' (line 318)
            rcParams_67094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 18), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 318)
            getitem___67095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 18), rcParams_67094, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 318)
            subscript_call_result_67096 = invoke(stypy.reporting.localization.Localization(__file__, 318, 18), getitem___67095, unicode_67093)
            
            # Assigning a type to the variable 'loc' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'loc', subscript_call_result_67096)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'self' (line 319)
            self_67097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'self')
            # Obtaining the member 'isaxes' of a type (line 319)
            isaxes_67098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 19), self_67097, 'isaxes')
            # Applying the 'not' unary operator (line 319)
            result_not__67099 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 15), 'not', isaxes_67098)
            
            
            # Getting the type of 'loc' (line 319)
            loc_67100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 35), 'loc')
            
            # Obtaining an instance of the builtin type 'list' (line 319)
            list_67101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 42), 'list')
            # Adding type elements to the builtin type 'list' instance (line 319)
            # Adding element type (line 319)
            int_67102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 43), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 42), list_67101, int_67102)
            # Adding element type (line 319)
            unicode_67103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 46), 'unicode', u'best')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 42), list_67101, unicode_67103)
            
            # Applying the binary operator 'in' (line 319)
            result_contains_67104 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 35), 'in', loc_67100, list_67101)
            
            # Applying the binary operator 'and' (line 319)
            result_and_keyword_67105 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 15), 'and', result_not__67099, result_contains_67104)
            
            # Testing the type of an if condition (line 319)
            if_condition_67106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 12), result_and_keyword_67105)
            # Assigning a type to the variable 'if_condition_67106' (line 319)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'if_condition_67106', if_condition_67106)
            # SSA begins for if statement (line 319)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 320):
            
            # Assigning a Str to a Name (line 320):
            
            # Assigning a Str to a Name (line 320):
            unicode_67107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 22), 'unicode', u'upper right')
            # Assigning a type to the variable 'loc' (line 320)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'loc', unicode_67107)
            # SSA join for if statement (line 319)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_67092:
                # SSA join for if statement (line 317)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to isinstance(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'loc' (line 321)
        loc_67109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 22), 'loc', False)
        # Getting the type of 'six' (line 321)
        six_67110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 27), 'six', False)
        # Obtaining the member 'string_types' of a type (line 321)
        string_types_67111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 27), six_67110, 'string_types')
        # Processing the call keyword arguments (line 321)
        kwargs_67112 = {}
        # Getting the type of 'isinstance' (line 321)
        isinstance_67108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 321)
        isinstance_call_result_67113 = invoke(stypy.reporting.localization.Localization(__file__, 321, 11), isinstance_67108, *[loc_67109, string_types_67111], **kwargs_67112)
        
        # Testing the type of an if condition (line 321)
        if_condition_67114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 8), isinstance_call_result_67113)
        # Assigning a type to the variable 'if_condition_67114' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'if_condition_67114', if_condition_67114)
        # SSA begins for if statement (line 321)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'loc' (line 322)
        loc_67115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'loc')
        # Getting the type of 'self' (line 322)
        self_67116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 26), 'self')
        # Obtaining the member 'codes' of a type (line 322)
        codes_67117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 26), self_67116, 'codes')
        # Applying the binary operator 'notin' (line 322)
        result_contains_67118 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 15), 'notin', loc_67115, codes_67117)
        
        # Testing the type of an if condition (line 322)
        if_condition_67119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 12), result_contains_67118)
        # Assigning a type to the variable 'if_condition_67119' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'if_condition_67119', if_condition_67119)
        # SSA begins for if statement (line 322)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 323)
        self_67120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 'self')
        # Obtaining the member 'isaxes' of a type (line 323)
        isaxes_67121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 19), self_67120, 'isaxes')
        # Testing the type of an if condition (line 323)
        if_condition_67122 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 16), isaxes_67121)
        # Assigning a type to the variable 'if_condition_67122' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'if_condition_67122', if_condition_67122)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 324)
        # Processing the call arguments (line 324)
        unicode_67125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 34), 'unicode', u'Unrecognized location "%s". Falling back on "best"; valid locations are\n\t%s\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 326)
        tuple_67126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 326)
        # Adding element type (line 326)
        # Getting the type of 'loc' (line 326)
        loc_67127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 37), 'loc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 37), tuple_67126, loc_67127)
        # Adding element type (line 326)
        
        # Call to join(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'self' (line 326)
        self_67130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 54), 'self', False)
        # Obtaining the member 'codes' of a type (line 326)
        codes_67131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 54), self_67130, 'codes')
        # Processing the call keyword arguments (line 326)
        kwargs_67132 = {}
        unicode_67128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 42), 'unicode', u'\n\t')
        # Obtaining the member 'join' of a type (line 326)
        join_67129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 42), unicode_67128, 'join')
        # Calling join(args, kwargs) (line 326)
        join_call_result_67133 = invoke(stypy.reporting.localization.Localization(__file__, 326, 42), join_67129, *[codes_67131], **kwargs_67132)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 37), tuple_67126, join_call_result_67133)
        
        # Applying the binary operator '%' (line 324)
        result_mod_67134 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 34), '%', unicode_67125, tuple_67126)
        
        # Processing the call keyword arguments (line 324)
        kwargs_67135 = {}
        # Getting the type of 'warnings' (line 324)
        warnings_67123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 324)
        warn_67124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 20), warnings_67123, 'warn')
        # Calling warn(args, kwargs) (line 324)
        warn_call_result_67136 = invoke(stypy.reporting.localization.Localization(__file__, 324, 20), warn_67124, *[result_mod_67134], **kwargs_67135)
        
        
        # Assigning a Num to a Name (line 327):
        
        # Assigning a Num to a Name (line 327):
        
        # Assigning a Num to a Name (line 327):
        int_67137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 26), 'int')
        # Assigning a type to the variable 'loc' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 20), 'loc', int_67137)
        # SSA branch for the else part of an if statement (line 323)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 329)
        # Processing the call arguments (line 329)
        unicode_67140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 34), 'unicode', u'Unrecognized location "%s". Falling back on "upper right"; valid locations are\n\t%s\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 332)
        tuple_67141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 332)
        # Adding element type (line 332)
        # Getting the type of 'loc' (line 332)
        loc_67142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 37), 'loc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 37), tuple_67141, loc_67142)
        # Adding element type (line 332)
        
        # Call to join(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'self' (line 332)
        self_67145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 54), 'self', False)
        # Obtaining the member 'codes' of a type (line 332)
        codes_67146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 54), self_67145, 'codes')
        # Processing the call keyword arguments (line 332)
        kwargs_67147 = {}
        unicode_67143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 42), 'unicode', u'\n\t')
        # Obtaining the member 'join' of a type (line 332)
        join_67144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 42), unicode_67143, 'join')
        # Calling join(args, kwargs) (line 332)
        join_call_result_67148 = invoke(stypy.reporting.localization.Localization(__file__, 332, 42), join_67144, *[codes_67146], **kwargs_67147)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 37), tuple_67141, join_call_result_67148)
        
        # Applying the binary operator '%' (line 329)
        result_mod_67149 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 34), '%', unicode_67140, tuple_67141)
        
        # Processing the call keyword arguments (line 329)
        kwargs_67150 = {}
        # Getting the type of 'warnings' (line 329)
        warnings_67138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 329)
        warn_67139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 20), warnings_67138, 'warn')
        # Calling warn(args, kwargs) (line 329)
        warn_call_result_67151 = invoke(stypy.reporting.localization.Localization(__file__, 329, 20), warn_67139, *[result_mod_67149], **kwargs_67150)
        
        
        # Assigning a Num to a Name (line 333):
        
        # Assigning a Num to a Name (line 333):
        
        # Assigning a Num to a Name (line 333):
        int_67152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 26), 'int')
        # Assigning a type to the variable 'loc' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 20), 'loc', int_67152)
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 322)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 335):
        
        # Assigning a Subscript to a Name (line 335):
        
        # Assigning a Subscript to a Name (line 335):
        
        # Obtaining the type of the subscript
        # Getting the type of 'loc' (line 335)
        loc_67153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 33), 'loc')
        # Getting the type of 'self' (line 335)
        self_67154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'self')
        # Obtaining the member 'codes' of a type (line 335)
        codes_67155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 22), self_67154, 'codes')
        # Obtaining the member '__getitem__' of a type (line 335)
        getitem___67156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 22), codes_67155, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 335)
        subscript_call_result_67157 = invoke(stypy.reporting.localization.Localization(__file__, 335, 22), getitem___67156, loc_67153)
        
        # Assigning a type to the variable 'loc' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'loc', subscript_call_result_67157)
        # SSA join for if statement (line 322)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 321)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 336)
        self_67158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 15), 'self')
        # Obtaining the member 'isaxes' of a type (line 336)
        isaxes_67159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 15), self_67158, 'isaxes')
        # Applying the 'not' unary operator (line 336)
        result_not__67160 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 11), 'not', isaxes_67159)
        
        
        # Getting the type of 'loc' (line 336)
        loc_67161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 31), 'loc')
        int_67162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 38), 'int')
        # Applying the binary operator '==' (line 336)
        result_eq_67163 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 31), '==', loc_67161, int_67162)
        
        # Applying the binary operator 'and' (line 336)
        result_and_keyword_67164 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 11), 'and', result_not__67160, result_eq_67163)
        
        # Testing the type of an if condition (line 336)
        if_condition_67165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 8), result_and_keyword_67164)
        # Assigning a type to the variable 'if_condition_67165' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'if_condition_67165', if_condition_67165)
        # SSA begins for if statement (line 336)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 337)
        # Processing the call arguments (line 337)
        unicode_67168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 26), 'unicode', u'Automatic legend placement (loc="best") not implemented for figure legend. Falling back on "upper right".')
        # Processing the call keyword arguments (line 337)
        kwargs_67169 = {}
        # Getting the type of 'warnings' (line 337)
        warnings_67166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 337)
        warn_67167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), warnings_67166, 'warn')
        # Calling warn(args, kwargs) (line 337)
        warn_call_result_67170 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), warn_67167, *[unicode_67168], **kwargs_67169)
        
        
        # Assigning a Num to a Name (line 340):
        
        # Assigning a Num to a Name (line 340):
        
        # Assigning a Num to a Name (line 340):
        int_67171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 18), 'int')
        # Assigning a type to the variable 'loc' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'loc', int_67171)
        # SSA join for if statement (line 336)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 342):
        
        # Assigning a Name to a Attribute (line 342):
        
        # Assigning a Name to a Attribute (line 342):
        # Getting the type of 'mode' (line 342)
        mode_67172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 21), 'mode')
        # Getting the type of 'self' (line 342)
        self_67173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'self')
        # Setting the type of the member '_mode' of a type (line 342)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), self_67173, '_mode', mode_67172)
        
        # Call to set_bbox_to_anchor(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'bbox_to_anchor' (line 343)
        bbox_to_anchor_67176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 32), 'bbox_to_anchor', False)
        # Getting the type of 'bbox_transform' (line 343)
        bbox_transform_67177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 48), 'bbox_transform', False)
        # Processing the call keyword arguments (line 343)
        kwargs_67178 = {}
        # Getting the type of 'self' (line 343)
        self_67174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'self', False)
        # Obtaining the member 'set_bbox_to_anchor' of a type (line 343)
        set_bbox_to_anchor_67175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), self_67174, 'set_bbox_to_anchor')
        # Calling set_bbox_to_anchor(args, kwargs) (line 343)
        set_bbox_to_anchor_call_result_67179 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), set_bbox_to_anchor_67175, *[bbox_to_anchor_67176, bbox_transform_67177], **kwargs_67178)
        
        
        # Type idiom detected: calculating its left and rigth part (line 348)
        # Getting the type of 'facecolor' (line 348)
        facecolor_67180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 11), 'facecolor')
        # Getting the type of 'None' (line 348)
        None_67181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 24), 'None')
        
        (may_be_67182, more_types_in_union_67183) = may_be_none(facecolor_67180, None_67181)

        if may_be_67182:

            if more_types_in_union_67183:
                # Runtime conditional SSA (line 348)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 349):
            
            # Assigning a Subscript to a Name (line 349):
            
            # Assigning a Subscript to a Name (line 349):
            
            # Obtaining the type of the subscript
            unicode_67184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 33), 'unicode', u'legend.facecolor')
            # Getting the type of 'rcParams' (line 349)
            rcParams_67185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 24), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 349)
            getitem___67186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 24), rcParams_67185, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 349)
            subscript_call_result_67187 = invoke(stypy.reporting.localization.Localization(__file__, 349, 24), getitem___67186, unicode_67184)
            
            # Assigning a type to the variable 'facecolor' (line 349)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'facecolor', subscript_call_result_67187)

            if more_types_in_union_67183:
                # SSA join for if statement (line 348)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'facecolor' (line 350)
        facecolor_67188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 11), 'facecolor')
        unicode_67189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 24), 'unicode', u'inherit')
        # Applying the binary operator '==' (line 350)
        result_eq_67190 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 11), '==', facecolor_67188, unicode_67189)
        
        # Testing the type of an if condition (line 350)
        if_condition_67191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 8), result_eq_67190)
        # Assigning a type to the variable 'if_condition_67191' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'if_condition_67191', if_condition_67191)
        # SSA begins for if statement (line 350)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 351):
        
        # Assigning a Subscript to a Name (line 351):
        
        # Assigning a Subscript to a Name (line 351):
        
        # Obtaining the type of the subscript
        unicode_67192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 33), 'unicode', u'axes.facecolor')
        # Getting the type of 'rcParams' (line 351)
        rcParams_67193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 351)
        getitem___67194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 24), rcParams_67193, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 351)
        subscript_call_result_67195 = invoke(stypy.reporting.localization.Localization(__file__, 351, 24), getitem___67194, unicode_67192)
        
        # Assigning a type to the variable 'facecolor' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'facecolor', subscript_call_result_67195)
        # SSA join for if statement (line 350)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 353)
        # Getting the type of 'edgecolor' (line 353)
        edgecolor_67196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 11), 'edgecolor')
        # Getting the type of 'None' (line 353)
        None_67197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 24), 'None')
        
        (may_be_67198, more_types_in_union_67199) = may_be_none(edgecolor_67196, None_67197)

        if may_be_67198:

            if more_types_in_union_67199:
                # Runtime conditional SSA (line 353)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 354):
            
            # Assigning a Subscript to a Name (line 354):
            
            # Assigning a Subscript to a Name (line 354):
            
            # Obtaining the type of the subscript
            unicode_67200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 33), 'unicode', u'legend.edgecolor')
            # Getting the type of 'rcParams' (line 354)
            rcParams_67201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 354)
            getitem___67202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 24), rcParams_67201, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 354)
            subscript_call_result_67203 = invoke(stypy.reporting.localization.Localization(__file__, 354, 24), getitem___67202, unicode_67200)
            
            # Assigning a type to the variable 'edgecolor' (line 354)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'edgecolor', subscript_call_result_67203)

            if more_types_in_union_67199:
                # SSA join for if statement (line 353)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'edgecolor' (line 355)
        edgecolor_67204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 11), 'edgecolor')
        unicode_67205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 24), 'unicode', u'inherit')
        # Applying the binary operator '==' (line 355)
        result_eq_67206 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 11), '==', edgecolor_67204, unicode_67205)
        
        # Testing the type of an if condition (line 355)
        if_condition_67207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 8), result_eq_67206)
        # Assigning a type to the variable 'if_condition_67207' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'if_condition_67207', if_condition_67207)
        # SSA begins for if statement (line 355)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 356):
        
        # Assigning a Subscript to a Name (line 356):
        
        # Assigning a Subscript to a Name (line 356):
        
        # Obtaining the type of the subscript
        unicode_67208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 33), 'unicode', u'axes.edgecolor')
        # Getting the type of 'rcParams' (line 356)
        rcParams_67209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 24), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 356)
        getitem___67210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 24), rcParams_67209, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 356)
        subscript_call_result_67211 = invoke(stypy.reporting.localization.Localization(__file__, 356, 24), getitem___67210, unicode_67208)
        
        # Assigning a type to the variable 'edgecolor' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'edgecolor', subscript_call_result_67211)
        # SSA join for if statement (line 355)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 358):
        
        # Assigning a Call to a Attribute (line 358):
        
        # Assigning a Call to a Attribute (line 358):
        
        # Call to FancyBboxPatch(...): (line 358)
        # Processing the call keyword arguments (line 358)
        
        # Obtaining an instance of the builtin type 'tuple' (line 359)
        tuple_67213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 359)
        # Adding element type (line 359)
        float_67214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 16), tuple_67213, float_67214)
        # Adding element type (line 359)
        float_67215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 16), tuple_67213, float_67215)
        
        keyword_67216 = tuple_67213
        float_67217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 33), 'float')
        keyword_67218 = float_67217
        float_67219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 44), 'float')
        keyword_67220 = float_67219
        # Getting the type of 'facecolor' (line 360)
        facecolor_67221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 22), 'facecolor', False)
        keyword_67222 = facecolor_67221
        # Getting the type of 'edgecolor' (line 361)
        edgecolor_67223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 22), 'edgecolor', False)
        keyword_67224 = edgecolor_67223
        # Getting the type of 'self' (line 362)
        self_67225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 27), 'self', False)
        # Obtaining the member '_fontsize' of a type (line 362)
        _fontsize_67226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 27), self_67225, '_fontsize')
        keyword_67227 = _fontsize_67226
        # Getting the type of 'True' (line 363)
        True_67228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 17), 'True', False)
        keyword_67229 = True_67228
        kwargs_67230 = {'mutation_scale': keyword_67227, 'edgecolor': keyword_67224, 'facecolor': keyword_67222, 'height': keyword_67220, 'width': keyword_67218, 'xy': keyword_67216, 'snap': keyword_67229}
        # Getting the type of 'FancyBboxPatch' (line 358)
        FancyBboxPatch_67212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'FancyBboxPatch', False)
        # Calling FancyBboxPatch(args, kwargs) (line 358)
        FancyBboxPatch_call_result_67231 = invoke(stypy.reporting.localization.Localization(__file__, 358, 27), FancyBboxPatch_67212, *[], **kwargs_67230)
        
        # Getting the type of 'self' (line 358)
        self_67232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'self')
        # Setting the type of the member 'legendPatch' of a type (line 358)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), self_67232, 'legendPatch', FancyBboxPatch_call_result_67231)
        
        # Type idiom detected: calculating its left and rigth part (line 369)
        # Getting the type of 'fancybox' (line 369)
        fancybox_67233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 11), 'fancybox')
        # Getting the type of 'None' (line 369)
        None_67234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 23), 'None')
        
        (may_be_67235, more_types_in_union_67236) = may_be_none(fancybox_67233, None_67234)

        if may_be_67235:

            if more_types_in_union_67236:
                # Runtime conditional SSA (line 369)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 370):
            
            # Assigning a Subscript to a Name (line 370):
            
            # Assigning a Subscript to a Name (line 370):
            
            # Obtaining the type of the subscript
            unicode_67237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 32), 'unicode', u'legend.fancybox')
            # Getting the type of 'rcParams' (line 370)
            rcParams_67238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 23), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 370)
            getitem___67239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 23), rcParams_67238, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 370)
            subscript_call_result_67240 = invoke(stypy.reporting.localization.Localization(__file__, 370, 23), getitem___67239, unicode_67237)
            
            # Assigning a type to the variable 'fancybox' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'fancybox', subscript_call_result_67240)

            if more_types_in_union_67236:
                # SSA join for if statement (line 369)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'fancybox' (line 372)
        fancybox_67241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 11), 'fancybox')
        # Testing the type of an if condition (line 372)
        if_condition_67242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 8), fancybox_67241)
        # Assigning a type to the variable 'if_condition_67242' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'if_condition_67242', if_condition_67242)
        # SSA begins for if statement (line 372)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_boxstyle(...): (line 373)
        # Processing the call arguments (line 373)
        unicode_67246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 42), 'unicode', u'round')
        # Processing the call keyword arguments (line 373)
        int_67247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 55), 'int')
        keyword_67248 = int_67247
        float_67249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 56), 'float')
        keyword_67250 = float_67249
        kwargs_67251 = {'rounding_size': keyword_67250, 'pad': keyword_67248}
        # Getting the type of 'self' (line 373)
        self_67243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'self', False)
        # Obtaining the member 'legendPatch' of a type (line 373)
        legendPatch_67244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 12), self_67243, 'legendPatch')
        # Obtaining the member 'set_boxstyle' of a type (line 373)
        set_boxstyle_67245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 12), legendPatch_67244, 'set_boxstyle')
        # Calling set_boxstyle(args, kwargs) (line 373)
        set_boxstyle_call_result_67252 = invoke(stypy.reporting.localization.Localization(__file__, 373, 12), set_boxstyle_67245, *[unicode_67246], **kwargs_67251)
        
        # SSA branch for the else part of an if statement (line 372)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_boxstyle(...): (line 376)
        # Processing the call arguments (line 376)
        unicode_67256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 42), 'unicode', u'square')
        # Processing the call keyword arguments (line 376)
        int_67257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 56), 'int')
        keyword_67258 = int_67257
        kwargs_67259 = {'pad': keyword_67258}
        # Getting the type of 'self' (line 376)
        self_67253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'self', False)
        # Obtaining the member 'legendPatch' of a type (line 376)
        legendPatch_67254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 12), self_67253, 'legendPatch')
        # Obtaining the member 'set_boxstyle' of a type (line 376)
        set_boxstyle_67255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 12), legendPatch_67254, 'set_boxstyle')
        # Calling set_boxstyle(args, kwargs) (line 376)
        set_boxstyle_call_result_67260 = invoke(stypy.reporting.localization.Localization(__file__, 376, 12), set_boxstyle_67255, *[unicode_67256], **kwargs_67259)
        
        # SSA join for if statement (line 372)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _set_artist_props(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'self' (line 378)
        self_67263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 31), 'self', False)
        # Obtaining the member 'legendPatch' of a type (line 378)
        legendPatch_67264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 31), self_67263, 'legendPatch')
        # Processing the call keyword arguments (line 378)
        kwargs_67265 = {}
        # Getting the type of 'self' (line 378)
        self_67261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self', False)
        # Obtaining the member '_set_artist_props' of a type (line 378)
        _set_artist_props_67262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_67261, '_set_artist_props')
        # Calling _set_artist_props(args, kwargs) (line 378)
        _set_artist_props_call_result_67266 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), _set_artist_props_67262, *[legendPatch_67264], **kwargs_67265)
        
        
        # Assigning a Name to a Attribute (line 380):
        
        # Assigning a Name to a Attribute (line 380):
        
        # Assigning a Name to a Attribute (line 380):
        # Getting the type of 'frameon' (line 380)
        frameon_67267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 26), 'frameon')
        # Getting the type of 'self' (line 380)
        self_67268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'self')
        # Setting the type of the member '_drawFrame' of a type (line 380)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), self_67268, '_drawFrame', frameon_67267)
        
        # Type idiom detected: calculating its left and rigth part (line 381)
        # Getting the type of 'frameon' (line 381)
        frameon_67269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 11), 'frameon')
        # Getting the type of 'None' (line 381)
        None_67270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 22), 'None')
        
        (may_be_67271, more_types_in_union_67272) = may_be_none(frameon_67269, None_67270)

        if may_be_67271:

            if more_types_in_union_67272:
                # Runtime conditional SSA (line 381)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Attribute (line 382):
            
            # Assigning a Subscript to a Attribute (line 382):
            
            # Assigning a Subscript to a Attribute (line 382):
            
            # Obtaining the type of the subscript
            unicode_67273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 39), 'unicode', u'legend.frameon')
            # Getting the type of 'rcParams' (line 382)
            rcParams_67274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 30), 'rcParams')
            # Obtaining the member '__getitem__' of a type (line 382)
            getitem___67275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 30), rcParams_67274, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 382)
            subscript_call_result_67276 = invoke(stypy.reporting.localization.Localization(__file__, 382, 30), getitem___67275, unicode_67273)
            
            # Getting the type of 'self' (line 382)
            self_67277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'self')
            # Setting the type of the member '_drawFrame' of a type (line 382)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 12), self_67277, '_drawFrame', subscript_call_result_67276)

            if more_types_in_union_67272:
                # SSA join for if statement (line 381)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _init_legend_box(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'handles' (line 385)
        handles_67280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 30), 'handles', False)
        # Getting the type of 'labels' (line 385)
        labels_67281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 39), 'labels', False)
        # Getting the type of 'markerfirst' (line 385)
        markerfirst_67282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 47), 'markerfirst', False)
        # Processing the call keyword arguments (line 385)
        kwargs_67283 = {}
        # Getting the type of 'self' (line 385)
        self_67278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'self', False)
        # Obtaining the member '_init_legend_box' of a type (line 385)
        _init_legend_box_67279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 8), self_67278, '_init_legend_box')
        # Calling _init_legend_box(args, kwargs) (line 385)
        _init_legend_box_call_result_67284 = invoke(stypy.reporting.localization.Localization(__file__, 385, 8), _init_legend_box_67279, *[handles_67280, labels_67281, markerfirst_67282], **kwargs_67283)
        
        
        # Type idiom detected: calculating its left and rigth part (line 389)
        # Getting the type of 'framealpha' (line 389)
        framealpha_67285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), 'framealpha')
        # Getting the type of 'None' (line 389)
        None_67286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 25), 'None')
        
        (may_be_67287, more_types_in_union_67288) = may_be_none(framealpha_67285, None_67286)

        if may_be_67287:

            if more_types_in_union_67288:
                # Runtime conditional SSA (line 389)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'shadow' (line 390)
            shadow_67289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'shadow')
            # Testing the type of an if condition (line 390)
            if_condition_67290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 390, 12), shadow_67289)
            # Assigning a type to the variable 'if_condition_67290' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'if_condition_67290', if_condition_67290)
            # SSA begins for if statement (line 390)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to set_alpha(...): (line 391)
            # Processing the call arguments (line 391)
            int_67296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 43), 'int')
            # Processing the call keyword arguments (line 391)
            kwargs_67297 = {}
            
            # Call to get_frame(...): (line 391)
            # Processing the call keyword arguments (line 391)
            kwargs_67293 = {}
            # Getting the type of 'self' (line 391)
            self_67291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'self', False)
            # Obtaining the member 'get_frame' of a type (line 391)
            get_frame_67292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 16), self_67291, 'get_frame')
            # Calling get_frame(args, kwargs) (line 391)
            get_frame_call_result_67294 = invoke(stypy.reporting.localization.Localization(__file__, 391, 16), get_frame_67292, *[], **kwargs_67293)
            
            # Obtaining the member 'set_alpha' of a type (line 391)
            set_alpha_67295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 16), get_frame_call_result_67294, 'set_alpha')
            # Calling set_alpha(args, kwargs) (line 391)
            set_alpha_call_result_67298 = invoke(stypy.reporting.localization.Localization(__file__, 391, 16), set_alpha_67295, *[int_67296], **kwargs_67297)
            
            # SSA branch for the else part of an if statement (line 390)
            module_type_store.open_ssa_branch('else')
            
            # Call to set_alpha(...): (line 393)
            # Processing the call arguments (line 393)
            
            # Obtaining the type of the subscript
            unicode_67304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 52), 'unicode', u'legend.framealpha')
            # Getting the type of 'rcParams' (line 393)
            rcParams_67305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 43), 'rcParams', False)
            # Obtaining the member '__getitem__' of a type (line 393)
            getitem___67306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 43), rcParams_67305, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 393)
            subscript_call_result_67307 = invoke(stypy.reporting.localization.Localization(__file__, 393, 43), getitem___67306, unicode_67304)
            
            # Processing the call keyword arguments (line 393)
            kwargs_67308 = {}
            
            # Call to get_frame(...): (line 393)
            # Processing the call keyword arguments (line 393)
            kwargs_67301 = {}
            # Getting the type of 'self' (line 393)
            self_67299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 16), 'self', False)
            # Obtaining the member 'get_frame' of a type (line 393)
            get_frame_67300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 16), self_67299, 'get_frame')
            # Calling get_frame(args, kwargs) (line 393)
            get_frame_call_result_67302 = invoke(stypy.reporting.localization.Localization(__file__, 393, 16), get_frame_67300, *[], **kwargs_67301)
            
            # Obtaining the member 'set_alpha' of a type (line 393)
            set_alpha_67303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 16), get_frame_call_result_67302, 'set_alpha')
            # Calling set_alpha(args, kwargs) (line 393)
            set_alpha_call_result_67309 = invoke(stypy.reporting.localization.Localization(__file__, 393, 16), set_alpha_67303, *[subscript_call_result_67307], **kwargs_67308)
            
            # SSA join for if statement (line 390)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_67288:
                # Runtime conditional SSA for else branch (line 389)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_67287) or more_types_in_union_67288):
            
            # Call to set_alpha(...): (line 395)
            # Processing the call arguments (line 395)
            # Getting the type of 'framealpha' (line 395)
            framealpha_67315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 39), 'framealpha', False)
            # Processing the call keyword arguments (line 395)
            kwargs_67316 = {}
            
            # Call to get_frame(...): (line 395)
            # Processing the call keyword arguments (line 395)
            kwargs_67312 = {}
            # Getting the type of 'self' (line 395)
            self_67310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'self', False)
            # Obtaining the member 'get_frame' of a type (line 395)
            get_frame_67311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), self_67310, 'get_frame')
            # Calling get_frame(args, kwargs) (line 395)
            get_frame_call_result_67313 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), get_frame_67311, *[], **kwargs_67312)
            
            # Obtaining the member 'set_alpha' of a type (line 395)
            set_alpha_67314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), get_frame_call_result_67313, 'set_alpha')
            # Calling set_alpha(args, kwargs) (line 395)
            set_alpha_call_result_67317 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), set_alpha_67314, *[framealpha_67315], **kwargs_67316)
            

            if (may_be_67287 and more_types_in_union_67288):
                # SSA join for if statement (line 389)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 397):
        
        # Assigning a Name to a Attribute (line 397):
        
        # Assigning a Name to a Attribute (line 397):
        # Getting the type of 'loc' (line 397)
        loc_67318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 20), 'loc')
        # Getting the type of 'self' (line 397)
        self_67319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self')
        # Setting the type of the member '_loc' of a type (line 397)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_67319, '_loc', loc_67318)
        
        # Call to set_title(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'title' (line 398)
        title_67322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 23), 'title', False)
        # Processing the call keyword arguments (line 398)
        kwargs_67323 = {}
        # Getting the type of 'self' (line 398)
        self_67320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'self', False)
        # Obtaining the member 'set_title' of a type (line 398)
        set_title_67321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), self_67320, 'set_title')
        # Calling set_title(args, kwargs) (line 398)
        set_title_call_result_67324 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), set_title_67321, *[title_67322], **kwargs_67323)
        
        
        # Assigning a Attribute to a Attribute (line 399):
        
        # Assigning a Attribute to a Attribute (line 399):
        
        # Assigning a Attribute to a Attribute (line 399):
        # Getting the type of 'self' (line 399)
        self_67325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 37), 'self')
        # Obtaining the member '_fontsize' of a type (line 399)
        _fontsize_67326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 37), self_67325, '_fontsize')
        # Getting the type of 'self' (line 399)
        self_67327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'self')
        # Setting the type of the member '_last_fontsize_points' of a type (line 399)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), self_67327, '_last_fontsize_points', _fontsize_67326)
        
        # Assigning a Name to a Attribute (line 400):
        
        # Assigning a Name to a Attribute (line 400):
        
        # Assigning a Name to a Attribute (line 400):
        # Getting the type of 'None' (line 400)
        None_67328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 26), 'None')
        # Getting the type of 'self' (line 400)
        self_67329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'self')
        # Setting the type of the member '_draggable' of a type (line 400)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 8), self_67329, '_draggable', None_67328)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _set_artist_props(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_artist_props'
        module_type_store = module_type_store.open_function_context('_set_artist_props', 402, 4, False)
        # Assigning a type to the variable 'self' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend._set_artist_props.__dict__.__setitem__('stypy_localization', localization)
        Legend._set_artist_props.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend._set_artist_props.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend._set_artist_props.__dict__.__setitem__('stypy_function_name', 'Legend._set_artist_props')
        Legend._set_artist_props.__dict__.__setitem__('stypy_param_names_list', ['a'])
        Legend._set_artist_props.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend._set_artist_props.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend._set_artist_props.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend._set_artist_props.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend._set_artist_props.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend._set_artist_props.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend._set_artist_props', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_artist_props', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_artist_props(...)' code ##################

        unicode_67330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, (-1)), 'unicode', u'\n        set the boilerplate props for artists added to axes\n        ')
        
        # Call to set_figure(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'self' (line 406)
        self_67333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 21), 'self', False)
        # Obtaining the member 'figure' of a type (line 406)
        figure_67334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 21), self_67333, 'figure')
        # Processing the call keyword arguments (line 406)
        kwargs_67335 = {}
        # Getting the type of 'a' (line 406)
        a_67331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'a', False)
        # Obtaining the member 'set_figure' of a type (line 406)
        set_figure_67332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), a_67331, 'set_figure')
        # Calling set_figure(args, kwargs) (line 406)
        set_figure_call_result_67336 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), set_figure_67332, *[figure_67334], **kwargs_67335)
        
        
        # Getting the type of 'self' (line 407)
        self_67337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 11), 'self')
        # Obtaining the member 'isaxes' of a type (line 407)
        isaxes_67338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 11), self_67337, 'isaxes')
        # Testing the type of an if condition (line 407)
        if_condition_67339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 407, 8), isaxes_67338)
        # Assigning a type to the variable 'if_condition_67339' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'if_condition_67339', if_condition_67339)
        # SSA begins for if statement (line 407)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 409):
        
        # Assigning a Attribute to a Attribute (line 409):
        
        # Assigning a Attribute to a Attribute (line 409):
        # Getting the type of 'self' (line 409)
        self_67340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 21), 'self')
        # Obtaining the member 'axes' of a type (line 409)
        axes_67341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 21), self_67340, 'axes')
        # Getting the type of 'a' (line 409)
        a_67342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'a')
        # Setting the type of the member 'axes' of a type (line 409)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 12), a_67342, 'axes', axes_67341)
        # SSA join for if statement (line 407)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_transform(...): (line 411)
        # Processing the call arguments (line 411)
        
        # Call to get_transform(...): (line 411)
        # Processing the call keyword arguments (line 411)
        kwargs_67347 = {}
        # Getting the type of 'self' (line 411)
        self_67345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 24), 'self', False)
        # Obtaining the member 'get_transform' of a type (line 411)
        get_transform_67346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 24), self_67345, 'get_transform')
        # Calling get_transform(args, kwargs) (line 411)
        get_transform_call_result_67348 = invoke(stypy.reporting.localization.Localization(__file__, 411, 24), get_transform_67346, *[], **kwargs_67347)
        
        # Processing the call keyword arguments (line 411)
        kwargs_67349 = {}
        # Getting the type of 'a' (line 411)
        a_67343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'a', False)
        # Obtaining the member 'set_transform' of a type (line 411)
        set_transform_67344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), a_67343, 'set_transform')
        # Calling set_transform(args, kwargs) (line 411)
        set_transform_call_result_67350 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), set_transform_67344, *[get_transform_call_result_67348], **kwargs_67349)
        
        
        # ################# End of '_set_artist_props(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_artist_props' in the type store
        # Getting the type of 'stypy_return_type' (line 402)
        stypy_return_type_67351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_67351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_artist_props'
        return stypy_return_type_67351


    @norecursion
    def _set_loc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_loc'
        module_type_store = module_type_store.open_function_context('_set_loc', 413, 4, False)
        # Assigning a type to the variable 'self' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend._set_loc.__dict__.__setitem__('stypy_localization', localization)
        Legend._set_loc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend._set_loc.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend._set_loc.__dict__.__setitem__('stypy_function_name', 'Legend._set_loc')
        Legend._set_loc.__dict__.__setitem__('stypy_param_names_list', ['loc'])
        Legend._set_loc.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend._set_loc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend._set_loc.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend._set_loc.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend._set_loc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend._set_loc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend._set_loc', ['loc'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_loc', localization, ['loc'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_loc(...)' code ##################

        
        # Assigning a Name to a Attribute (line 417):
        
        # Assigning a Name to a Attribute (line 417):
        
        # Assigning a Name to a Attribute (line 417):
        # Getting the type of 'loc' (line 417)
        loc_67352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 25), 'loc')
        # Getting the type of 'self' (line 417)
        self_67353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'self')
        # Setting the type of the member '_loc_real' of a type (line 417)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 8), self_67353, '_loc_real', loc_67352)
        
        # Assigning a Name to a Attribute (line 418):
        
        # Assigning a Name to a Attribute (line 418):
        
        # Assigning a Name to a Attribute (line 418):
        # Getting the type of 'True' (line 418)
        True_67354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 21), 'True')
        # Getting the type of 'self' (line 418)
        self_67355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 418)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), self_67355, 'stale', True_67354)
        
        # ################# End of '_set_loc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_loc' in the type store
        # Getting the type of 'stypy_return_type' (line 413)
        stypy_return_type_67356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_67356)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_loc'
        return stypy_return_type_67356


    @norecursion
    def _get_loc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_loc'
        module_type_store = module_type_store.open_function_context('_get_loc', 420, 4, False)
        # Assigning a type to the variable 'self' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend._get_loc.__dict__.__setitem__('stypy_localization', localization)
        Legend._get_loc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend._get_loc.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend._get_loc.__dict__.__setitem__('stypy_function_name', 'Legend._get_loc')
        Legend._get_loc.__dict__.__setitem__('stypy_param_names_list', [])
        Legend._get_loc.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend._get_loc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend._get_loc.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend._get_loc.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend._get_loc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend._get_loc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend._get_loc', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_loc', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_loc(...)' code ##################

        # Getting the type of 'self' (line 421)
        self_67357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), 'self')
        # Obtaining the member '_loc_real' of a type (line 421)
        _loc_real_67358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 15), self_67357, '_loc_real')
        # Assigning a type to the variable 'stypy_return_type' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'stypy_return_type', _loc_real_67358)
        
        # ################# End of '_get_loc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_loc' in the type store
        # Getting the type of 'stypy_return_type' (line 420)
        stypy_return_type_67359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_67359)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_loc'
        return stypy_return_type_67359

    
    # Assigning a Call to a Name (line 423):
    
    # Assigning a Call to a Name (line 423):

    @norecursion
    def _findoffset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_findoffset'
        module_type_store = module_type_store.open_function_context('_findoffset', 425, 4, False)
        # Assigning a type to the variable 'self' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend._findoffset.__dict__.__setitem__('stypy_localization', localization)
        Legend._findoffset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend._findoffset.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend._findoffset.__dict__.__setitem__('stypy_function_name', 'Legend._findoffset')
        Legend._findoffset.__dict__.__setitem__('stypy_param_names_list', ['width', 'height', 'xdescent', 'ydescent', 'renderer'])
        Legend._findoffset.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend._findoffset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend._findoffset.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend._findoffset.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend._findoffset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend._findoffset.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend._findoffset', ['width', 'height', 'xdescent', 'ydescent', 'renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_findoffset', localization, ['width', 'height', 'xdescent', 'ydescent', 'renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_findoffset(...)' code ##################

        unicode_67360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 8), 'unicode', u'Helper function to locate the legend')
        
        
        # Getting the type of 'self' (line 428)
        self_67361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 11), 'self')
        # Obtaining the member '_loc' of a type (line 428)
        _loc_67362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 11), self_67361, '_loc')
        int_67363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 24), 'int')
        # Applying the binary operator '==' (line 428)
        result_eq_67364 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 11), '==', _loc_67362, int_67363)
        
        # Testing the type of an if condition (line 428)
        if_condition_67365 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 8), result_eq_67364)
        # Assigning a type to the variable 'if_condition_67365' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'if_condition_67365', if_condition_67365)
        # SSA begins for if statement (line 428)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 429):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to _find_best_position(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'width' (line 429)
        width_67368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 44), 'width', False)
        # Getting the type of 'height' (line 429)
        height_67369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 51), 'height', False)
        # Getting the type of 'renderer' (line 429)
        renderer_67370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 59), 'renderer', False)
        # Processing the call keyword arguments (line 429)
        kwargs_67371 = {}
        # Getting the type of 'self' (line 429)
        self_67366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 19), 'self', False)
        # Obtaining the member '_find_best_position' of a type (line 429)
        _find_best_position_67367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 19), self_67366, '_find_best_position')
        # Calling _find_best_position(args, kwargs) (line 429)
        _find_best_position_call_result_67372 = invoke(stypy.reporting.localization.Localization(__file__, 429, 19), _find_best_position_67367, *[width_67368, height_67369, renderer_67370], **kwargs_67371)
        
        # Assigning a type to the variable 'call_assignment_66638' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'call_assignment_66638', _find_best_position_call_result_67372)
        
        # Assigning a Call to a Name (line 429):
        
        # Assigning a Call to a Name (line 429):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_67375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 12), 'int')
        # Processing the call keyword arguments
        kwargs_67376 = {}
        # Getting the type of 'call_assignment_66638' (line 429)
        call_assignment_66638_67373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'call_assignment_66638', False)
        # Obtaining the member '__getitem__' of a type (line 429)
        getitem___67374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 12), call_assignment_66638_67373, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_67377 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___67374, *[int_67375], **kwargs_67376)
        
        # Assigning a type to the variable 'call_assignment_66639' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'call_assignment_66639', getitem___call_result_67377)
        
        # Assigning a Name to a Name (line 429):
        
        # Assigning a Name to a Name (line 429):
        # Getting the type of 'call_assignment_66639' (line 429)
        call_assignment_66639_67378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'call_assignment_66639')
        # Assigning a type to the variable 'x' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'x', call_assignment_66639_67378)
        
        # Assigning a Call to a Name (line 429):
        
        # Assigning a Call to a Name (line 429):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_67381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 12), 'int')
        # Processing the call keyword arguments
        kwargs_67382 = {}
        # Getting the type of 'call_assignment_66638' (line 429)
        call_assignment_66638_67379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'call_assignment_66638', False)
        # Obtaining the member '__getitem__' of a type (line 429)
        getitem___67380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 12), call_assignment_66638_67379, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_67383 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___67380, *[int_67381], **kwargs_67382)
        
        # Assigning a type to the variable 'call_assignment_66640' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'call_assignment_66640', getitem___call_result_67383)
        
        # Assigning a Name to a Name (line 429):
        
        # Assigning a Name to a Name (line 429):
        # Getting the type of 'call_assignment_66640' (line 429)
        call_assignment_66640_67384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'call_assignment_66640')
        # Assigning a type to the variable 'y' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 15), 'y', call_assignment_66640_67384)
        # SSA branch for the else part of an if statement (line 428)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 430)
        self_67385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 13), 'self')
        # Obtaining the member '_loc' of a type (line 430)
        _loc_67386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 13), self_67385, '_loc')
        
        # Call to values(...): (line 430)
        # Processing the call keyword arguments (line 430)
        kwargs_67390 = {}
        # Getting the type of 'Legend' (line 430)
        Legend_67387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 26), 'Legend', False)
        # Obtaining the member 'codes' of a type (line 430)
        codes_67388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 26), Legend_67387, 'codes')
        # Obtaining the member 'values' of a type (line 430)
        values_67389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 26), codes_67388, 'values')
        # Calling values(args, kwargs) (line 430)
        values_call_result_67391 = invoke(stypy.reporting.localization.Localization(__file__, 430, 26), values_67389, *[], **kwargs_67390)
        
        # Applying the binary operator 'in' (line 430)
        result_contains_67392 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 13), 'in', _loc_67386, values_call_result_67391)
        
        # Testing the type of an if condition (line 430)
        if_condition_67393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 430, 13), result_contains_67392)
        # Assigning a type to the variable 'if_condition_67393' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 13), 'if_condition_67393', if_condition_67393)
        # SSA begins for if statement (line 430)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Call to from_bounds(...): (line 431)
        # Processing the call arguments (line 431)
        int_67396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 36), 'int')
        int_67397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 39), 'int')
        # Getting the type of 'width' (line 431)
        width_67398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 42), 'width', False)
        # Getting the type of 'height' (line 431)
        height_67399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 49), 'height', False)
        # Processing the call keyword arguments (line 431)
        kwargs_67400 = {}
        # Getting the type of 'Bbox' (line 431)
        Bbox_67394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 19), 'Bbox', False)
        # Obtaining the member 'from_bounds' of a type (line 431)
        from_bounds_67395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 19), Bbox_67394, 'from_bounds')
        # Calling from_bounds(args, kwargs) (line 431)
        from_bounds_call_result_67401 = invoke(stypy.reporting.localization.Localization(__file__, 431, 19), from_bounds_67395, *[int_67396, int_67397, width_67398, height_67399], **kwargs_67400)
        
        # Assigning a type to the variable 'bbox' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'bbox', from_bounds_call_result_67401)
        
        # Assigning a Call to a Tuple (line 432):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to _get_anchored_bbox(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'self' (line 432)
        self_67404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 43), 'self', False)
        # Obtaining the member '_loc' of a type (line 432)
        _loc_67405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 43), self_67404, '_loc')
        # Getting the type of 'bbox' (line 432)
        bbox_67406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 54), 'bbox', False)
        
        # Call to get_bbox_to_anchor(...): (line 433)
        # Processing the call keyword arguments (line 433)
        kwargs_67409 = {}
        # Getting the type of 'self' (line 433)
        self_67407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 43), 'self', False)
        # Obtaining the member 'get_bbox_to_anchor' of a type (line 433)
        get_bbox_to_anchor_67408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 43), self_67407, 'get_bbox_to_anchor')
        # Calling get_bbox_to_anchor(args, kwargs) (line 433)
        get_bbox_to_anchor_call_result_67410 = invoke(stypy.reporting.localization.Localization(__file__, 433, 43), get_bbox_to_anchor_67408, *[], **kwargs_67409)
        
        # Getting the type of 'renderer' (line 434)
        renderer_67411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 43), 'renderer', False)
        # Processing the call keyword arguments (line 432)
        kwargs_67412 = {}
        # Getting the type of 'self' (line 432)
        self_67402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 19), 'self', False)
        # Obtaining the member '_get_anchored_bbox' of a type (line 432)
        _get_anchored_bbox_67403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 19), self_67402, '_get_anchored_bbox')
        # Calling _get_anchored_bbox(args, kwargs) (line 432)
        _get_anchored_bbox_call_result_67413 = invoke(stypy.reporting.localization.Localization(__file__, 432, 19), _get_anchored_bbox_67403, *[_loc_67405, bbox_67406, get_bbox_to_anchor_call_result_67410, renderer_67411], **kwargs_67412)
        
        # Assigning a type to the variable 'call_assignment_66641' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'call_assignment_66641', _get_anchored_bbox_call_result_67413)
        
        # Assigning a Call to a Name (line 432):
        
        # Assigning a Call to a Name (line 432):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_67416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 12), 'int')
        # Processing the call keyword arguments
        kwargs_67417 = {}
        # Getting the type of 'call_assignment_66641' (line 432)
        call_assignment_66641_67414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'call_assignment_66641', False)
        # Obtaining the member '__getitem__' of a type (line 432)
        getitem___67415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 12), call_assignment_66641_67414, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_67418 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___67415, *[int_67416], **kwargs_67417)
        
        # Assigning a type to the variable 'call_assignment_66642' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'call_assignment_66642', getitem___call_result_67418)
        
        # Assigning a Name to a Name (line 432):
        
        # Assigning a Name to a Name (line 432):
        # Getting the type of 'call_assignment_66642' (line 432)
        call_assignment_66642_67419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'call_assignment_66642')
        # Assigning a type to the variable 'x' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'x', call_assignment_66642_67419)
        
        # Assigning a Call to a Name (line 432):
        
        # Assigning a Call to a Name (line 432):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_67422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 12), 'int')
        # Processing the call keyword arguments
        kwargs_67423 = {}
        # Getting the type of 'call_assignment_66641' (line 432)
        call_assignment_66641_67420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'call_assignment_66641', False)
        # Obtaining the member '__getitem__' of a type (line 432)
        getitem___67421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 12), call_assignment_66641_67420, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_67424 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___67421, *[int_67422], **kwargs_67423)
        
        # Assigning a type to the variable 'call_assignment_66643' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'call_assignment_66643', getitem___call_result_67424)
        
        # Assigning a Name to a Name (line 432):
        
        # Assigning a Name to a Name (line 432):
        # Getting the type of 'call_assignment_66643' (line 432)
        call_assignment_66643_67425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'call_assignment_66643')
        # Assigning a type to the variable 'y' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 15), 'y', call_assignment_66643_67425)
        # SSA branch for the else part of an if statement (line 430)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Tuple (line 436):
        
        # Assigning a Subscript to a Name (line 436):
        
        # Assigning a Subscript to a Name (line 436):
        
        # Obtaining the type of the subscript
        int_67426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 12), 'int')
        # Getting the type of 'self' (line 436)
        self_67427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 21), 'self')
        # Obtaining the member '_loc' of a type (line 436)
        _loc_67428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 21), self_67427, '_loc')
        # Obtaining the member '__getitem__' of a type (line 436)
        getitem___67429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 12), _loc_67428, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 436)
        subscript_call_result_67430 = invoke(stypy.reporting.localization.Localization(__file__, 436, 12), getitem___67429, int_67426)
        
        # Assigning a type to the variable 'tuple_var_assignment_66644' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'tuple_var_assignment_66644', subscript_call_result_67430)
        
        # Assigning a Subscript to a Name (line 436):
        
        # Assigning a Subscript to a Name (line 436):
        
        # Obtaining the type of the subscript
        int_67431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 12), 'int')
        # Getting the type of 'self' (line 436)
        self_67432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 21), 'self')
        # Obtaining the member '_loc' of a type (line 436)
        _loc_67433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 21), self_67432, '_loc')
        # Obtaining the member '__getitem__' of a type (line 436)
        getitem___67434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 12), _loc_67433, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 436)
        subscript_call_result_67435 = invoke(stypy.reporting.localization.Localization(__file__, 436, 12), getitem___67434, int_67431)
        
        # Assigning a type to the variable 'tuple_var_assignment_66645' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'tuple_var_assignment_66645', subscript_call_result_67435)
        
        # Assigning a Name to a Name (line 436):
        
        # Assigning a Name to a Name (line 436):
        # Getting the type of 'tuple_var_assignment_66644' (line 436)
        tuple_var_assignment_66644_67436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'tuple_var_assignment_66644')
        # Assigning a type to the variable 'fx' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'fx', tuple_var_assignment_66644_67436)
        
        # Assigning a Name to a Name (line 436):
        
        # Assigning a Name to a Name (line 436):
        # Getting the type of 'tuple_var_assignment_66645' (line 436)
        tuple_var_assignment_66645_67437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'tuple_var_assignment_66645')
        # Assigning a type to the variable 'fy' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'fy', tuple_var_assignment_66645_67437)
        
        # Assigning a Call to a Name (line 437):
        
        # Assigning a Call to a Name (line 437):
        
        # Assigning a Call to a Name (line 437):
        
        # Call to get_bbox_to_anchor(...): (line 437)
        # Processing the call keyword arguments (line 437)
        kwargs_67440 = {}
        # Getting the type of 'self' (line 437)
        self_67438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 19), 'self', False)
        # Obtaining the member 'get_bbox_to_anchor' of a type (line 437)
        get_bbox_to_anchor_67439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 19), self_67438, 'get_bbox_to_anchor')
        # Calling get_bbox_to_anchor(args, kwargs) (line 437)
        get_bbox_to_anchor_call_result_67441 = invoke(stypy.reporting.localization.Localization(__file__, 437, 19), get_bbox_to_anchor_67439, *[], **kwargs_67440)
        
        # Assigning a type to the variable 'bbox' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'bbox', get_bbox_to_anchor_call_result_67441)
        
        # Assigning a Tuple to a Tuple (line 438):
        
        # Assigning a BinOp to a Name (line 438):
        
        # Assigning a BinOp to a Name (line 438):
        # Getting the type of 'bbox' (line 438)
        bbox_67442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'bbox')
        # Obtaining the member 'x0' of a type (line 438)
        x0_67443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 19), bbox_67442, 'x0')
        # Getting the type of 'bbox' (line 438)
        bbox_67444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 29), 'bbox')
        # Obtaining the member 'width' of a type (line 438)
        width_67445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 29), bbox_67444, 'width')
        # Getting the type of 'fx' (line 438)
        fx_67446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 42), 'fx')
        # Applying the binary operator '*' (line 438)
        result_mul_67447 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 29), '*', width_67445, fx_67446)
        
        # Applying the binary operator '+' (line 438)
        result_add_67448 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 19), '+', x0_67443, result_mul_67447)
        
        # Assigning a type to the variable 'tuple_assignment_66646' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'tuple_assignment_66646', result_add_67448)
        
        # Assigning a BinOp to a Name (line 438):
        
        # Assigning a BinOp to a Name (line 438):
        # Getting the type of 'bbox' (line 438)
        bbox_67449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 46), 'bbox')
        # Obtaining the member 'y0' of a type (line 438)
        y0_67450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 46), bbox_67449, 'y0')
        # Getting the type of 'bbox' (line 438)
        bbox_67451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 56), 'bbox')
        # Obtaining the member 'height' of a type (line 438)
        height_67452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 56), bbox_67451, 'height')
        # Getting the type of 'fy' (line 438)
        fy_67453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 70), 'fy')
        # Applying the binary operator '*' (line 438)
        result_mul_67454 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 56), '*', height_67452, fy_67453)
        
        # Applying the binary operator '+' (line 438)
        result_add_67455 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 46), '+', y0_67450, result_mul_67454)
        
        # Assigning a type to the variable 'tuple_assignment_66647' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'tuple_assignment_66647', result_add_67455)
        
        # Assigning a Name to a Name (line 438):
        
        # Assigning a Name to a Name (line 438):
        # Getting the type of 'tuple_assignment_66646' (line 438)
        tuple_assignment_66646_67456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'tuple_assignment_66646')
        # Assigning a type to the variable 'x' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'x', tuple_assignment_66646_67456)
        
        # Assigning a Name to a Name (line 438):
        
        # Assigning a Name to a Name (line 438):
        # Getting the type of 'tuple_assignment_66647' (line 438)
        tuple_assignment_66647_67457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'tuple_assignment_66647')
        # Assigning a type to the variable 'y' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'y', tuple_assignment_66647_67457)
        # SSA join for if statement (line 430)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 428)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 440)
        tuple_67458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 440)
        # Adding element type (line 440)
        # Getting the type of 'x' (line 440)
        x_67459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'x')
        # Getting the type of 'xdescent' (line 440)
        xdescent_67460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 19), 'xdescent')
        # Applying the binary operator '+' (line 440)
        result_add_67461 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 15), '+', x_67459, xdescent_67460)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 15), tuple_67458, result_add_67461)
        # Adding element type (line 440)
        # Getting the type of 'y' (line 440)
        y_67462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 29), 'y')
        # Getting the type of 'ydescent' (line 440)
        ydescent_67463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 33), 'ydescent')
        # Applying the binary operator '+' (line 440)
        result_add_67464 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 29), '+', y_67462, ydescent_67463)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 15), tuple_67458, result_add_67464)
        
        # Assigning a type to the variable 'stypy_return_type' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'stypy_return_type', tuple_67458)
        
        # ################# End of '_findoffset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_findoffset' in the type store
        # Getting the type of 'stypy_return_type' (line 425)
        stypy_return_type_67465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_67465)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_findoffset'
        return stypy_return_type_67465


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.draw.__dict__.__setitem__('stypy_localization', localization)
        Legend.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.draw.__dict__.__setitem__('stypy_function_name', 'Legend.draw')
        Legend.draw.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Legend.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.draw.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.draw', ['renderer'], None, None, defaults, varargs, kwargs)

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

        unicode_67466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 8), 'unicode', u'Draw everything that belongs to the legend')
        
        
        
        # Call to get_visible(...): (line 445)
        # Processing the call keyword arguments (line 445)
        kwargs_67469 = {}
        # Getting the type of 'self' (line 445)
        self_67467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 15), 'self', False)
        # Obtaining the member 'get_visible' of a type (line 445)
        get_visible_67468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 15), self_67467, 'get_visible')
        # Calling get_visible(args, kwargs) (line 445)
        get_visible_call_result_67470 = invoke(stypy.reporting.localization.Localization(__file__, 445, 15), get_visible_67468, *[], **kwargs_67469)
        
        # Applying the 'not' unary operator (line 445)
        result_not__67471 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 11), 'not', get_visible_call_result_67470)
        
        # Testing the type of an if condition (line 445)
        if_condition_67472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 8), result_not__67471)
        # Assigning a type to the variable 'if_condition_67472' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'if_condition_67472', if_condition_67472)
        # SSA begins for if statement (line 445)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 445)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to open_group(...): (line 448)
        # Processing the call arguments (line 448)
        unicode_67475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 28), 'unicode', u'legend')
        # Processing the call keyword arguments (line 448)
        kwargs_67476 = {}
        # Getting the type of 'renderer' (line 448)
        renderer_67473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'renderer', False)
        # Obtaining the member 'open_group' of a type (line 448)
        open_group_67474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), renderer_67473, 'open_group')
        # Calling open_group(args, kwargs) (line 448)
        open_group_call_result_67477 = invoke(stypy.reporting.localization.Localization(__file__, 448, 8), open_group_67474, *[unicode_67475], **kwargs_67476)
        
        
        # Assigning a Call to a Name (line 450):
        
        # Assigning a Call to a Name (line 450):
        
        # Assigning a Call to a Name (line 450):
        
        # Call to points_to_pixels(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'self' (line 450)
        self_67480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 45), 'self', False)
        # Obtaining the member '_fontsize' of a type (line 450)
        _fontsize_67481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 45), self_67480, '_fontsize')
        # Processing the call keyword arguments (line 450)
        kwargs_67482 = {}
        # Getting the type of 'renderer' (line 450)
        renderer_67478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), 'renderer', False)
        # Obtaining the member 'points_to_pixels' of a type (line 450)
        points_to_pixels_67479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 19), renderer_67478, 'points_to_pixels')
        # Calling points_to_pixels(args, kwargs) (line 450)
        points_to_pixels_call_result_67483 = invoke(stypy.reporting.localization.Localization(__file__, 450, 19), points_to_pixels_67479, *[_fontsize_67481], **kwargs_67482)
        
        # Assigning a type to the variable 'fontsize' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'fontsize', points_to_pixels_call_result_67483)
        
        
        # Getting the type of 'self' (line 454)
        self_67484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 11), 'self')
        # Obtaining the member '_mode' of a type (line 454)
        _mode_67485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 11), self_67484, '_mode')
        
        # Obtaining an instance of the builtin type 'list' (line 454)
        list_67486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 454)
        # Adding element type (line 454)
        unicode_67487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 26), 'unicode', u'expand')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 25), list_67486, unicode_67487)
        
        # Applying the binary operator 'in' (line 454)
        result_contains_67488 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 11), 'in', _mode_67485, list_67486)
        
        # Testing the type of an if condition (line 454)
        if_condition_67489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 8), result_contains_67488)
        # Assigning a type to the variable 'if_condition_67489' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'if_condition_67489', if_condition_67489)
        # SSA begins for if statement (line 454)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 455):
        
        # Assigning a BinOp to a Name (line 455):
        
        # Assigning a BinOp to a Name (line 455):
        int_67490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 18), 'int')
        # Getting the type of 'self' (line 455)
        self_67491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 23), 'self')
        # Obtaining the member 'borderaxespad' of a type (line 455)
        borderaxespad_67492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 23), self_67491, 'borderaxespad')
        # Getting the type of 'self' (line 455)
        self_67493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 44), 'self')
        # Obtaining the member 'borderpad' of a type (line 455)
        borderpad_67494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 44), self_67493, 'borderpad')
        # Applying the binary operator '+' (line 455)
        result_add_67495 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 23), '+', borderaxespad_67492, borderpad_67494)
        
        # Applying the binary operator '*' (line 455)
        result_mul_67496 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 18), '*', int_67490, result_add_67495)
        
        # Getting the type of 'fontsize' (line 455)
        fontsize_67497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 62), 'fontsize')
        # Applying the binary operator '*' (line 455)
        result_mul_67498 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 60), '*', result_mul_67496, fontsize_67497)
        
        # Assigning a type to the variable 'pad' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'pad', result_mul_67498)
        
        # Call to set_width(...): (line 456)
        # Processing the call arguments (line 456)
        
        # Call to get_bbox_to_anchor(...): (line 456)
        # Processing the call keyword arguments (line 456)
        kwargs_67504 = {}
        # Getting the type of 'self' (line 456)
        self_67502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 39), 'self', False)
        # Obtaining the member 'get_bbox_to_anchor' of a type (line 456)
        get_bbox_to_anchor_67503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 39), self_67502, 'get_bbox_to_anchor')
        # Calling get_bbox_to_anchor(args, kwargs) (line 456)
        get_bbox_to_anchor_call_result_67505 = invoke(stypy.reporting.localization.Localization(__file__, 456, 39), get_bbox_to_anchor_67503, *[], **kwargs_67504)
        
        # Obtaining the member 'width' of a type (line 456)
        width_67506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 39), get_bbox_to_anchor_call_result_67505, 'width')
        # Getting the type of 'pad' (line 456)
        pad_67507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 73), 'pad', False)
        # Applying the binary operator '-' (line 456)
        result_sub_67508 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 39), '-', width_67506, pad_67507)
        
        # Processing the call keyword arguments (line 456)
        kwargs_67509 = {}
        # Getting the type of 'self' (line 456)
        self_67499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'self', False)
        # Obtaining the member '_legend_box' of a type (line 456)
        _legend_box_67500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 12), self_67499, '_legend_box')
        # Obtaining the member 'set_width' of a type (line 456)
        set_width_67501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 12), _legend_box_67500, 'set_width')
        # Calling set_width(args, kwargs) (line 456)
        set_width_call_result_67510 = invoke(stypy.reporting.localization.Localization(__file__, 456, 12), set_width_67501, *[result_sub_67508], **kwargs_67509)
        
        # SSA join for if statement (line 454)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 460):
        
        # Assigning a Call to a Name (line 460):
        
        # Assigning a Call to a Name (line 460):
        
        # Call to get_window_extent(...): (line 460)
        # Processing the call arguments (line 460)
        # Getting the type of 'renderer' (line 460)
        renderer_67514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 50), 'renderer', False)
        # Processing the call keyword arguments (line 460)
        kwargs_67515 = {}
        # Getting the type of 'self' (line 460)
        self_67511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 15), 'self', False)
        # Obtaining the member '_legend_box' of a type (line 460)
        _legend_box_67512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 15), self_67511, '_legend_box')
        # Obtaining the member 'get_window_extent' of a type (line 460)
        get_window_extent_67513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 15), _legend_box_67512, 'get_window_extent')
        # Calling get_window_extent(args, kwargs) (line 460)
        get_window_extent_call_result_67516 = invoke(stypy.reporting.localization.Localization(__file__, 460, 15), get_window_extent_67513, *[renderer_67514], **kwargs_67515)
        
        # Assigning a type to the variable 'bbox' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'bbox', get_window_extent_call_result_67516)
        
        # Call to set_bounds(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'bbox' (line 461)
        bbox_67520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 36), 'bbox', False)
        # Obtaining the member 'x0' of a type (line 461)
        x0_67521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 36), bbox_67520, 'x0')
        # Getting the type of 'bbox' (line 461)
        bbox_67522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 45), 'bbox', False)
        # Obtaining the member 'y0' of a type (line 461)
        y0_67523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 45), bbox_67522, 'y0')
        # Getting the type of 'bbox' (line 462)
        bbox_67524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 36), 'bbox', False)
        # Obtaining the member 'width' of a type (line 462)
        width_67525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 36), bbox_67524, 'width')
        # Getting the type of 'bbox' (line 462)
        bbox_67526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 48), 'bbox', False)
        # Obtaining the member 'height' of a type (line 462)
        height_67527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 48), bbox_67526, 'height')
        # Processing the call keyword arguments (line 461)
        kwargs_67528 = {}
        # Getting the type of 'self' (line 461)
        self_67517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'self', False)
        # Obtaining the member 'legendPatch' of a type (line 461)
        legendPatch_67518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 8), self_67517, 'legendPatch')
        # Obtaining the member 'set_bounds' of a type (line 461)
        set_bounds_67519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 8), legendPatch_67518, 'set_bounds')
        # Calling set_bounds(args, kwargs) (line 461)
        set_bounds_call_result_67529 = invoke(stypy.reporting.localization.Localization(__file__, 461, 8), set_bounds_67519, *[x0_67521, y0_67523, width_67525, height_67527], **kwargs_67528)
        
        
        # Call to set_mutation_scale(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'fontsize' (line 463)
        fontsize_67533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 44), 'fontsize', False)
        # Processing the call keyword arguments (line 463)
        kwargs_67534 = {}
        # Getting the type of 'self' (line 463)
        self_67530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'self', False)
        # Obtaining the member 'legendPatch' of a type (line 463)
        legendPatch_67531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 8), self_67530, 'legendPatch')
        # Obtaining the member 'set_mutation_scale' of a type (line 463)
        set_mutation_scale_67532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 8), legendPatch_67531, 'set_mutation_scale')
        # Calling set_mutation_scale(args, kwargs) (line 463)
        set_mutation_scale_call_result_67535 = invoke(stypy.reporting.localization.Localization(__file__, 463, 8), set_mutation_scale_67532, *[fontsize_67533], **kwargs_67534)
        
        
        # Getting the type of 'self' (line 465)
        self_67536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 11), 'self')
        # Obtaining the member '_drawFrame' of a type (line 465)
        _drawFrame_67537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 11), self_67536, '_drawFrame')
        # Testing the type of an if condition (line 465)
        if_condition_67538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 465, 8), _drawFrame_67537)
        # Assigning a type to the variable 'if_condition_67538' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'if_condition_67538', if_condition_67538)
        # SSA begins for if statement (line 465)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 466)
        self_67539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 15), 'self')
        # Obtaining the member 'shadow' of a type (line 466)
        shadow_67540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 15), self_67539, 'shadow')
        # Testing the type of an if condition (line 466)
        if_condition_67541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 12), shadow_67540)
        # Assigning a type to the variable 'if_condition_67541' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'if_condition_67541', if_condition_67541)
        # SSA begins for if statement (line 466)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 467):
        
        # Assigning a Call to a Name (line 467):
        
        # Assigning a Call to a Name (line 467):
        
        # Call to Shadow(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'self' (line 467)
        self_67543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 32), 'self', False)
        # Obtaining the member 'legendPatch' of a type (line 467)
        legendPatch_67544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 32), self_67543, 'legendPatch')
        int_67545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 50), 'int')
        int_67546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 53), 'int')
        # Processing the call keyword arguments (line 467)
        kwargs_67547 = {}
        # Getting the type of 'Shadow' (line 467)
        Shadow_67542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 25), 'Shadow', False)
        # Calling Shadow(args, kwargs) (line 467)
        Shadow_call_result_67548 = invoke(stypy.reporting.localization.Localization(__file__, 467, 25), Shadow_67542, *[legendPatch_67544, int_67545, int_67546], **kwargs_67547)
        
        # Assigning a type to the variable 'shadow' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'shadow', Shadow_call_result_67548)
        
        # Call to draw(...): (line 468)
        # Processing the call arguments (line 468)
        # Getting the type of 'renderer' (line 468)
        renderer_67551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 28), 'renderer', False)
        # Processing the call keyword arguments (line 468)
        kwargs_67552 = {}
        # Getting the type of 'shadow' (line 468)
        shadow_67549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 16), 'shadow', False)
        # Obtaining the member 'draw' of a type (line 468)
        draw_67550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 16), shadow_67549, 'draw')
        # Calling draw(args, kwargs) (line 468)
        draw_call_result_67553 = invoke(stypy.reporting.localization.Localization(__file__, 468, 16), draw_67550, *[renderer_67551], **kwargs_67552)
        
        # SSA join for if statement (line 466)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'renderer' (line 470)
        renderer_67557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 34), 'renderer', False)
        # Processing the call keyword arguments (line 470)
        kwargs_67558 = {}
        # Getting the type of 'self' (line 470)
        self_67554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'self', False)
        # Obtaining the member 'legendPatch' of a type (line 470)
        legendPatch_67555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 12), self_67554, 'legendPatch')
        # Obtaining the member 'draw' of a type (line 470)
        draw_67556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 12), legendPatch_67555, 'draw')
        # Calling draw(args, kwargs) (line 470)
        draw_call_result_67559 = invoke(stypy.reporting.localization.Localization(__file__, 470, 12), draw_67556, *[renderer_67557], **kwargs_67558)
        
        # SSA join for if statement (line 465)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw(...): (line 472)
        # Processing the call arguments (line 472)
        # Getting the type of 'renderer' (line 472)
        renderer_67563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 30), 'renderer', False)
        # Processing the call keyword arguments (line 472)
        kwargs_67564 = {}
        # Getting the type of 'self' (line 472)
        self_67560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'self', False)
        # Obtaining the member '_legend_box' of a type (line 472)
        _legend_box_67561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), self_67560, '_legend_box')
        # Obtaining the member 'draw' of a type (line 472)
        draw_67562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), _legend_box_67561, 'draw')
        # Calling draw(args, kwargs) (line 472)
        draw_call_result_67565 = invoke(stypy.reporting.localization.Localization(__file__, 472, 8), draw_67562, *[renderer_67563], **kwargs_67564)
        
        
        # Call to close_group(...): (line 474)
        # Processing the call arguments (line 474)
        unicode_67568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 29), 'unicode', u'legend')
        # Processing the call keyword arguments (line 474)
        kwargs_67569 = {}
        # Getting the type of 'renderer' (line 474)
        renderer_67566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'renderer', False)
        # Obtaining the member 'close_group' of a type (line 474)
        close_group_67567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), renderer_67566, 'close_group')
        # Calling close_group(args, kwargs) (line 474)
        close_group_call_result_67570 = invoke(stypy.reporting.localization.Localization(__file__, 474, 8), close_group_67567, *[unicode_67568], **kwargs_67569)
        
        
        # Assigning a Name to a Attribute (line 475):
        
        # Assigning a Name to a Attribute (line 475):
        
        # Assigning a Name to a Attribute (line 475):
        # Getting the type of 'False' (line 475)
        False_67571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 21), 'False')
        # Getting the type of 'self' (line 475)
        self_67572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 475)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), self_67572, 'stale', False_67571)
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 442)
        stypy_return_type_67573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_67573)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_67573


    @norecursion
    def _approx_text_height(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 477)
        None_67574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 43), 'None')
        defaults = [None_67574]
        # Create a new context for function '_approx_text_height'
        module_type_store = module_type_store.open_function_context('_approx_text_height', 477, 4, False)
        # Assigning a type to the variable 'self' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend._approx_text_height.__dict__.__setitem__('stypy_localization', localization)
        Legend._approx_text_height.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend._approx_text_height.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend._approx_text_height.__dict__.__setitem__('stypy_function_name', 'Legend._approx_text_height')
        Legend._approx_text_height.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Legend._approx_text_height.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend._approx_text_height.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend._approx_text_height.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend._approx_text_height.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend._approx_text_height.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend._approx_text_height.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend._approx_text_height', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_approx_text_height', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_approx_text_height(...)' code ##################

        unicode_67575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, (-1)), 'unicode', u'\n        Return the approximate height of the text. This is used to place\n        the legend handle.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 482)
        # Getting the type of 'renderer' (line 482)
        renderer_67576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 11), 'renderer')
        # Getting the type of 'None' (line 482)
        None_67577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 23), 'None')
        
        (may_be_67578, more_types_in_union_67579) = may_be_none(renderer_67576, None_67577)

        if may_be_67578:

            if more_types_in_union_67579:
                # Runtime conditional SSA (line 482)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 483)
            self_67580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 19), 'self')
            # Obtaining the member '_fontsize' of a type (line 483)
            _fontsize_67581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 19), self_67580, '_fontsize')
            # Assigning a type to the variable 'stypy_return_type' (line 483)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'stypy_return_type', _fontsize_67581)

            if more_types_in_union_67579:
                # Runtime conditional SSA for else branch (line 482)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_67578) or more_types_in_union_67579):
            
            # Call to points_to_pixels(...): (line 485)
            # Processing the call arguments (line 485)
            # Getting the type of 'self' (line 485)
            self_67584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 45), 'self', False)
            # Obtaining the member '_fontsize' of a type (line 485)
            _fontsize_67585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 45), self_67584, '_fontsize')
            # Processing the call keyword arguments (line 485)
            kwargs_67586 = {}
            # Getting the type of 'renderer' (line 485)
            renderer_67582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 19), 'renderer', False)
            # Obtaining the member 'points_to_pixels' of a type (line 485)
            points_to_pixels_67583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 19), renderer_67582, 'points_to_pixels')
            # Calling points_to_pixels(args, kwargs) (line 485)
            points_to_pixels_call_result_67587 = invoke(stypy.reporting.localization.Localization(__file__, 485, 19), points_to_pixels_67583, *[_fontsize_67585], **kwargs_67586)
            
            # Assigning a type to the variable 'stypy_return_type' (line 485)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'stypy_return_type', points_to_pixels_call_result_67587)

            if (may_be_67578 and more_types_in_union_67579):
                # SSA join for if statement (line 482)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_approx_text_height(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_approx_text_height' in the type store
        # Getting the type of 'stypy_return_type' (line 477)
        stypy_return_type_67588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_67588)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_approx_text_height'
        return stypy_return_type_67588

    
    # Assigning a Dict to a Name (line 490):
    
    # Assigning a Dict to a Name (line 490):

    @norecursion
    def get_default_handler_map(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_default_handler_map'
        module_type_store = module_type_store.open_function_context('get_default_handler_map', 508, 4, False)
        # Assigning a type to the variable 'self' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.get_default_handler_map.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_default_handler_map.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.get_default_handler_map.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_default_handler_map.__dict__.__setitem__('stypy_function_name', 'Legend.get_default_handler_map')
        Legend.get_default_handler_map.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.get_default_handler_map.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.get_default_handler_map.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.get_default_handler_map.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_default_handler_map.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_default_handler_map.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_default_handler_map.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.get_default_handler_map', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_default_handler_map', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_default_handler_map(...)' code ##################

        unicode_67589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, (-1)), 'unicode', u'\n        A class method that returns the default handler map.\n        ')
        # Getting the type of 'cls' (line 513)
        cls_67590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 15), 'cls')
        # Obtaining the member '_default_handler_map' of a type (line 513)
        _default_handler_map_67591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 15), cls_67590, '_default_handler_map')
        # Assigning a type to the variable 'stypy_return_type' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'stypy_return_type', _default_handler_map_67591)
        
        # ################# End of 'get_default_handler_map(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_default_handler_map' in the type store
        # Getting the type of 'stypy_return_type' (line 508)
        stypy_return_type_67592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_67592)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_default_handler_map'
        return stypy_return_type_67592


    @norecursion
    def set_default_handler_map(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_default_handler_map'
        module_type_store = module_type_store.open_function_context('set_default_handler_map', 515, 4, False)
        # Assigning a type to the variable 'self' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.set_default_handler_map.__dict__.__setitem__('stypy_localization', localization)
        Legend.set_default_handler_map.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.set_default_handler_map.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.set_default_handler_map.__dict__.__setitem__('stypy_function_name', 'Legend.set_default_handler_map')
        Legend.set_default_handler_map.__dict__.__setitem__('stypy_param_names_list', ['handler_map'])
        Legend.set_default_handler_map.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.set_default_handler_map.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.set_default_handler_map.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.set_default_handler_map.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.set_default_handler_map.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.set_default_handler_map.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.set_default_handler_map', ['handler_map'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_default_handler_map', localization, ['handler_map'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_default_handler_map(...)' code ##################

        unicode_67593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, (-1)), 'unicode', u'\n        A class method to set the default handler map.\n        ')
        
        # Assigning a Name to a Attribute (line 520):
        
        # Assigning a Name to a Attribute (line 520):
        
        # Assigning a Name to a Attribute (line 520):
        # Getting the type of 'handler_map' (line 520)
        handler_map_67594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 35), 'handler_map')
        # Getting the type of 'cls' (line 520)
        cls_67595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'cls')
        # Setting the type of the member '_default_handler_map' of a type (line 520)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 8), cls_67595, '_default_handler_map', handler_map_67594)
        
        # ################# End of 'set_default_handler_map(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_default_handler_map' in the type store
        # Getting the type of 'stypy_return_type' (line 515)
        stypy_return_type_67596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_67596)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_default_handler_map'
        return stypy_return_type_67596


    @norecursion
    def update_default_handler_map(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_default_handler_map'
        module_type_store = module_type_store.open_function_context('update_default_handler_map', 522, 4, False)
        # Assigning a type to the variable 'self' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.update_default_handler_map.__dict__.__setitem__('stypy_localization', localization)
        Legend.update_default_handler_map.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.update_default_handler_map.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.update_default_handler_map.__dict__.__setitem__('stypy_function_name', 'Legend.update_default_handler_map')
        Legend.update_default_handler_map.__dict__.__setitem__('stypy_param_names_list', ['handler_map'])
        Legend.update_default_handler_map.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.update_default_handler_map.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.update_default_handler_map.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.update_default_handler_map.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.update_default_handler_map.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.update_default_handler_map.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.update_default_handler_map', ['handler_map'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_default_handler_map', localization, ['handler_map'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_default_handler_map(...)' code ##################

        unicode_67597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, (-1)), 'unicode', u'\n        A class method to update the default handler map.\n        ')
        
        # Call to update(...): (line 527)
        # Processing the call arguments (line 527)
        # Getting the type of 'handler_map' (line 527)
        handler_map_67601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 40), 'handler_map', False)
        # Processing the call keyword arguments (line 527)
        kwargs_67602 = {}
        # Getting the type of 'cls' (line 527)
        cls_67598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'cls', False)
        # Obtaining the member '_default_handler_map' of a type (line 527)
        _default_handler_map_67599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 8), cls_67598, '_default_handler_map')
        # Obtaining the member 'update' of a type (line 527)
        update_67600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 8), _default_handler_map_67599, 'update')
        # Calling update(args, kwargs) (line 527)
        update_call_result_67603 = invoke(stypy.reporting.localization.Localization(__file__, 527, 8), update_67600, *[handler_map_67601], **kwargs_67602)
        
        
        # ################# End of 'update_default_handler_map(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_default_handler_map' in the type store
        # Getting the type of 'stypy_return_type' (line 522)
        stypy_return_type_67604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_67604)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_default_handler_map'
        return stypy_return_type_67604


    @norecursion
    def get_legend_handler_map(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_legend_handler_map'
        module_type_store = module_type_store.open_function_context('get_legend_handler_map', 529, 4, False)
        # Assigning a type to the variable 'self' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.get_legend_handler_map.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_legend_handler_map.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.get_legend_handler_map.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_legend_handler_map.__dict__.__setitem__('stypy_function_name', 'Legend.get_legend_handler_map')
        Legend.get_legend_handler_map.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.get_legend_handler_map.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.get_legend_handler_map.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.get_legend_handler_map.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_legend_handler_map.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_legend_handler_map.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_legend_handler_map.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.get_legend_handler_map', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_legend_handler_map', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_legend_handler_map(...)' code ##################

        unicode_67605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, (-1)), 'unicode', u'\n        return the handler map.\n        ')
        
        # Assigning a Call to a Name (line 534):
        
        # Assigning a Call to a Name (line 534):
        
        # Assigning a Call to a Name (line 534):
        
        # Call to get_default_handler_map(...): (line 534)
        # Processing the call keyword arguments (line 534)
        kwargs_67608 = {}
        # Getting the type of 'self' (line 534)
        self_67606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 30), 'self', False)
        # Obtaining the member 'get_default_handler_map' of a type (line 534)
        get_default_handler_map_67607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 30), self_67606, 'get_default_handler_map')
        # Calling get_default_handler_map(args, kwargs) (line 534)
        get_default_handler_map_call_result_67609 = invoke(stypy.reporting.localization.Localization(__file__, 534, 30), get_default_handler_map_67607, *[], **kwargs_67608)
        
        # Assigning a type to the variable 'default_handler_map' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'default_handler_map', get_default_handler_map_call_result_67609)
        
        # Getting the type of 'self' (line 536)
        self_67610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 11), 'self')
        # Obtaining the member '_custom_handler_map' of a type (line 536)
        _custom_handler_map_67611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 11), self_67610, '_custom_handler_map')
        # Testing the type of an if condition (line 536)
        if_condition_67612 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 8), _custom_handler_map_67611)
        # Assigning a type to the variable 'if_condition_67612' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'if_condition_67612', if_condition_67612)
        # SSA begins for if statement (line 536)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 537):
        
        # Assigning a Call to a Name (line 537):
        
        # Assigning a Call to a Name (line 537):
        
        # Call to copy(...): (line 537)
        # Processing the call keyword arguments (line 537)
        kwargs_67615 = {}
        # Getting the type of 'default_handler_map' (line 537)
        default_handler_map_67613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 17), 'default_handler_map', False)
        # Obtaining the member 'copy' of a type (line 537)
        copy_67614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 17), default_handler_map_67613, 'copy')
        # Calling copy(args, kwargs) (line 537)
        copy_call_result_67616 = invoke(stypy.reporting.localization.Localization(__file__, 537, 17), copy_67614, *[], **kwargs_67615)
        
        # Assigning a type to the variable 'hm' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'hm', copy_call_result_67616)
        
        # Call to update(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'self' (line 538)
        self_67619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 22), 'self', False)
        # Obtaining the member '_custom_handler_map' of a type (line 538)
        _custom_handler_map_67620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 22), self_67619, '_custom_handler_map')
        # Processing the call keyword arguments (line 538)
        kwargs_67621 = {}
        # Getting the type of 'hm' (line 538)
        hm_67617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'hm', False)
        # Obtaining the member 'update' of a type (line 538)
        update_67618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 12), hm_67617, 'update')
        # Calling update(args, kwargs) (line 538)
        update_call_result_67622 = invoke(stypy.reporting.localization.Localization(__file__, 538, 12), update_67618, *[_custom_handler_map_67620], **kwargs_67621)
        
        # Getting the type of 'hm' (line 539)
        hm_67623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 19), 'hm')
        # Assigning a type to the variable 'stypy_return_type' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'stypy_return_type', hm_67623)
        # SSA branch for the else part of an if statement (line 536)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'default_handler_map' (line 541)
        default_handler_map_67624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 19), 'default_handler_map')
        # Assigning a type to the variable 'stypy_return_type' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'stypy_return_type', default_handler_map_67624)
        # SSA join for if statement (line 536)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_legend_handler_map(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_legend_handler_map' in the type store
        # Getting the type of 'stypy_return_type' (line 529)
        stypy_return_type_67625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_67625)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_legend_handler_map'
        return stypy_return_type_67625


    @staticmethod
    @norecursion
    def get_legend_handler(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_legend_handler'
        module_type_store = module_type_store.open_function_context('get_legend_handler', 543, 4, False)
        
        # Passed parameters checking function
        Legend.get_legend_handler.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_legend_handler.__dict__.__setitem__('stypy_type_of_self', None)
        Legend.get_legend_handler.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_legend_handler.__dict__.__setitem__('stypy_function_name', 'get_legend_handler')
        Legend.get_legend_handler.__dict__.__setitem__('stypy_param_names_list', ['legend_handler_map', 'orig_handle'])
        Legend.get_legend_handler.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.get_legend_handler.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.get_legend_handler.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_legend_handler.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_legend_handler.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_legend_handler.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'get_legend_handler', ['legend_handler_map', 'orig_handle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_legend_handler', localization, ['orig_handle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_legend_handler(...)' code ##################

        unicode_67626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, (-1)), 'unicode', u'\n        return a legend handler from *legend_handler_map* that\n        corresponds to *orig_handler*.\n\n        *legend_handler_map* should be a dictionary object (that is\n        returned by the get_legend_handler_map method).\n\n        It first checks if the *orig_handle* itself is a key in the\n        *legend_hanler_map* and return the associated value.\n        Otherwise, it checks for each of the classes in its\n        method-resolution-order. If no matching key is found, it\n        returns None.\n        ')
        
        
        # Call to is_hashable(...): (line 558)
        # Processing the call arguments (line 558)
        # Getting the type of 'orig_handle' (line 558)
        orig_handle_67628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 23), 'orig_handle', False)
        # Processing the call keyword arguments (line 558)
        kwargs_67629 = {}
        # Getting the type of 'is_hashable' (line 558)
        is_hashable_67627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 11), 'is_hashable', False)
        # Calling is_hashable(args, kwargs) (line 558)
        is_hashable_call_result_67630 = invoke(stypy.reporting.localization.Localization(__file__, 558, 11), is_hashable_67627, *[orig_handle_67628], **kwargs_67629)
        
        # Testing the type of an if condition (line 558)
        if_condition_67631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 558, 8), is_hashable_call_result_67630)
        # Assigning a type to the variable 'if_condition_67631' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'if_condition_67631', if_condition_67631)
        # SSA begins for if statement (line 558)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 559)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        # Getting the type of 'orig_handle' (line 560)
        orig_handle_67632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 42), 'orig_handle')
        # Getting the type of 'legend_handler_map' (line 560)
        legend_handler_map_67633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 23), 'legend_handler_map')
        # Obtaining the member '__getitem__' of a type (line 560)
        getitem___67634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 23), legend_handler_map_67633, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 560)
        subscript_call_result_67635 = invoke(stypy.reporting.localization.Localization(__file__, 560, 23), getitem___67634, orig_handle_67632)
        
        # Assigning a type to the variable 'stypy_return_type' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 16), 'stypy_return_type', subscript_call_result_67635)
        # SSA branch for the except part of a try statement (line 559)
        # SSA branch for the except 'KeyError' branch of a try statement (line 559)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 559)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 558)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to mro(...): (line 564)
        # Processing the call keyword arguments (line 564)
        kwargs_67641 = {}
        
        # Call to type(...): (line 564)
        # Processing the call arguments (line 564)
        # Getting the type of 'orig_handle' (line 564)
        orig_handle_67637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 32), 'orig_handle', False)
        # Processing the call keyword arguments (line 564)
        kwargs_67638 = {}
        # Getting the type of 'type' (line 564)
        type_67636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 27), 'type', False)
        # Calling type(args, kwargs) (line 564)
        type_call_result_67639 = invoke(stypy.reporting.localization.Localization(__file__, 564, 27), type_67636, *[orig_handle_67637], **kwargs_67638)
        
        # Obtaining the member 'mro' of a type (line 564)
        mro_67640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 27), type_call_result_67639, 'mro')
        # Calling mro(args, kwargs) (line 564)
        mro_call_result_67642 = invoke(stypy.reporting.localization.Localization(__file__, 564, 27), mro_67640, *[], **kwargs_67641)
        
        # Testing the type of a for loop iterable (line 564)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 564, 8), mro_call_result_67642)
        # Getting the type of the for loop variable (line 564)
        for_loop_var_67643 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 564, 8), mro_call_result_67642)
        # Assigning a type to the variable 'handle_type' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'handle_type', for_loop_var_67643)
        # SSA begins for a for statement (line 564)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 565)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        # Getting the type of 'handle_type' (line 566)
        handle_type_67644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 42), 'handle_type')
        # Getting the type of 'legend_handler_map' (line 566)
        legend_handler_map_67645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 23), 'legend_handler_map')
        # Obtaining the member '__getitem__' of a type (line 566)
        getitem___67646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 23), legend_handler_map_67645, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 566)
        subscript_call_result_67647 = invoke(stypy.reporting.localization.Localization(__file__, 566, 23), getitem___67646, handle_type_67644)
        
        # Assigning a type to the variable 'stypy_return_type' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 16), 'stypy_return_type', subscript_call_result_67647)
        # SSA branch for the except part of a try statement (line 565)
        # SSA branch for the except 'KeyError' branch of a try statement (line 565)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 565)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 570)
        None_67648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'stypy_return_type', None_67648)
        
        # ################# End of 'get_legend_handler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_legend_handler' in the type store
        # Getting the type of 'stypy_return_type' (line 543)
        stypy_return_type_67649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_67649)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_legend_handler'
        return stypy_return_type_67649


    @norecursion
    def _init_legend_box(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 572)
        True_67650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 60), 'True')
        defaults = [True_67650]
        # Create a new context for function '_init_legend_box'
        module_type_store = module_type_store.open_function_context('_init_legend_box', 572, 4, False)
        # Assigning a type to the variable 'self' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend._init_legend_box.__dict__.__setitem__('stypy_localization', localization)
        Legend._init_legend_box.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend._init_legend_box.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend._init_legend_box.__dict__.__setitem__('stypy_function_name', 'Legend._init_legend_box')
        Legend._init_legend_box.__dict__.__setitem__('stypy_param_names_list', ['handles', 'labels', 'markerfirst'])
        Legend._init_legend_box.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend._init_legend_box.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend._init_legend_box.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend._init_legend_box.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend._init_legend_box.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend._init_legend_box.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend._init_legend_box', ['handles', 'labels', 'markerfirst'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init_legend_box', localization, ['handles', 'labels', 'markerfirst'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init_legend_box(...)' code ##################

        unicode_67651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, (-1)), 'unicode', u'\n        Initialize the legend_box. The legend_box is an instance of\n        the OffsetBox, which is packed with legend handles and\n        texts. Once packed, their location is calculated during the\n        drawing time.\n        ')
        
        # Assigning a Attribute to a Name (line 580):
        
        # Assigning a Attribute to a Name (line 580):
        
        # Assigning a Attribute to a Name (line 580):
        # Getting the type of 'self' (line 580)
        self_67652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 19), 'self')
        # Obtaining the member '_fontsize' of a type (line 580)
        _fontsize_67653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 19), self_67652, '_fontsize')
        # Assigning a type to the variable 'fontsize' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'fontsize', _fontsize_67653)
        
        # Assigning a List to a Name (line 590):
        
        # Assigning a List to a Name (line 590):
        
        # Assigning a List to a Name (line 590):
        
        # Obtaining an instance of the builtin type 'list' (line 590)
        list_67654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 590)
        
        # Assigning a type to the variable 'text_list' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'text_list', list_67654)
        
        # Assigning a List to a Name (line 591):
        
        # Assigning a List to a Name (line 591):
        
        # Assigning a List to a Name (line 591):
        
        # Obtaining an instance of the builtin type 'list' (line 591)
        list_67655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 591)
        
        # Assigning a type to the variable 'handle_list' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'handle_list', list_67655)
        
        # Assigning a Call to a Name (line 593):
        
        # Assigning a Call to a Name (line 593):
        
        # Assigning a Call to a Name (line 593):
        
        # Call to dict(...): (line 593)
        # Processing the call keyword arguments (line 593)
        unicode_67657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 44), 'unicode', u'baseline')
        keyword_67658 = unicode_67657
        unicode_67659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 46), 'unicode', u'left')
        keyword_67660 = unicode_67659
        # Getting the type of 'self' (line 595)
        self_67661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 41), 'self', False)
        # Obtaining the member 'prop' of a type (line 595)
        prop_67662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 41), self_67661, 'prop')
        keyword_67663 = prop_67662
        kwargs_67664 = {'horizontalalignment': keyword_67660, 'fontproperties': keyword_67663, 'verticalalignment': keyword_67658}
        # Getting the type of 'dict' (line 593)
        dict_67656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 21), 'dict', False)
        # Calling dict(args, kwargs) (line 593)
        dict_call_result_67665 = invoke(stypy.reporting.localization.Localization(__file__, 593, 21), dict_67656, *[], **kwargs_67664)
        
        # Assigning a type to the variable 'label_prop' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'label_prop', dict_call_result_67665)
        
        # Assigning a List to a Name (line 598):
        
        # Assigning a List to a Name (line 598):
        
        # Assigning a List to a Name (line 598):
        
        # Obtaining an instance of the builtin type 'list' (line 598)
        list_67666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 598)
        
        # Assigning a type to the variable 'labelboxes' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'labelboxes', list_67666)
        
        # Assigning a List to a Name (line 599):
        
        # Assigning a List to a Name (line 599):
        
        # Assigning a List to a Name (line 599):
        
        # Obtaining an instance of the builtin type 'list' (line 599)
        list_67667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 599)
        
        # Assigning a type to the variable 'handleboxes' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'handleboxes', list_67667)
        
        # Assigning a BinOp to a Name (line 603):
        
        # Assigning a BinOp to a Name (line 603):
        
        # Assigning a BinOp to a Name (line 603):
        float_67668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 18), 'float')
        
        # Call to _approx_text_height(...): (line 603)
        # Processing the call keyword arguments (line 603)
        kwargs_67671 = {}
        # Getting the type of 'self' (line 603)
        self_67669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 25), 'self', False)
        # Obtaining the member '_approx_text_height' of a type (line 603)
        _approx_text_height_67670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 25), self_67669, '_approx_text_height')
        # Calling _approx_text_height(args, kwargs) (line 603)
        _approx_text_height_call_result_67672 = invoke(stypy.reporting.localization.Localization(__file__, 603, 25), _approx_text_height_67670, *[], **kwargs_67671)
        
        # Applying the binary operator '*' (line 603)
        result_mul_67673 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 18), '*', float_67668, _approx_text_height_call_result_67672)
        
        # Getting the type of 'self' (line 603)
        self_67674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 55), 'self')
        # Obtaining the member 'handleheight' of a type (line 603)
        handleheight_67675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 55), self_67674, 'handleheight')
        float_67676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 75), 'float')
        # Applying the binary operator '-' (line 603)
        result_sub_67677 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 55), '-', handleheight_67675, float_67676)
        
        # Applying the binary operator '*' (line 603)
        result_mul_67678 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 52), '*', result_mul_67673, result_sub_67677)
        
        # Assigning a type to the variable 'descent' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'descent', result_mul_67678)
        
        # Assigning a BinOp to a Name (line 605):
        
        # Assigning a BinOp to a Name (line 605):
        
        # Assigning a BinOp to a Name (line 605):
        
        # Call to _approx_text_height(...): (line 605)
        # Processing the call keyword arguments (line 605)
        kwargs_67681 = {}
        # Getting the type of 'self' (line 605)
        self_67679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 17), 'self', False)
        # Obtaining the member '_approx_text_height' of a type (line 605)
        _approx_text_height_67680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 17), self_67679, '_approx_text_height')
        # Calling _approx_text_height(args, kwargs) (line 605)
        _approx_text_height_call_result_67682 = invoke(stypy.reporting.localization.Localization(__file__, 605, 17), _approx_text_height_67680, *[], **kwargs_67681)
        
        # Getting the type of 'self' (line 605)
        self_67683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 46), 'self')
        # Obtaining the member 'handleheight' of a type (line 605)
        handleheight_67684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 46), self_67683, 'handleheight')
        # Applying the binary operator '*' (line 605)
        result_mul_67685 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 17), '*', _approx_text_height_call_result_67682, handleheight_67684)
        
        # Getting the type of 'descent' (line 605)
        descent_67686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 66), 'descent')
        # Applying the binary operator '-' (line 605)
        result_sub_67687 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 17), '-', result_mul_67685, descent_67686)
        
        # Assigning a type to the variable 'height' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'height', result_sub_67687)
        
        # Assigning a Call to a Name (line 614):
        
        # Assigning a Call to a Name (line 614):
        
        # Assigning a Call to a Name (line 614):
        
        # Call to get_legend_handler_map(...): (line 614)
        # Processing the call keyword arguments (line 614)
        kwargs_67690 = {}
        # Getting the type of 'self' (line 614)
        self_67688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 29), 'self', False)
        # Obtaining the member 'get_legend_handler_map' of a type (line 614)
        get_legend_handler_map_67689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 29), self_67688, 'get_legend_handler_map')
        # Calling get_legend_handler_map(args, kwargs) (line 614)
        get_legend_handler_map_call_result_67691 = invoke(stypy.reporting.localization.Localization(__file__, 614, 29), get_legend_handler_map_67689, *[], **kwargs_67690)
        
        # Assigning a type to the variable 'legend_handler_map' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'legend_handler_map', get_legend_handler_map_call_result_67691)
        
        
        # Call to zip(...): (line 616)
        # Processing the call arguments (line 616)
        # Getting the type of 'handles' (line 616)
        handles_67693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 36), 'handles', False)
        # Getting the type of 'labels' (line 616)
        labels_67694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 45), 'labels', False)
        # Processing the call keyword arguments (line 616)
        kwargs_67695 = {}
        # Getting the type of 'zip' (line 616)
        zip_67692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 32), 'zip', False)
        # Calling zip(args, kwargs) (line 616)
        zip_call_result_67696 = invoke(stypy.reporting.localization.Localization(__file__, 616, 32), zip_67692, *[handles_67693, labels_67694], **kwargs_67695)
        
        # Testing the type of a for loop iterable (line 616)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 616, 8), zip_call_result_67696)
        # Getting the type of the for loop variable (line 616)
        for_loop_var_67697 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 616, 8), zip_call_result_67696)
        # Assigning a type to the variable 'orig_handle' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'orig_handle', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 8), for_loop_var_67697))
        # Assigning a type to the variable 'lab' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'lab', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 8), for_loop_var_67697))
        # SSA begins for a for statement (line 616)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 617):
        
        # Assigning a Call to a Name (line 617):
        
        # Assigning a Call to a Name (line 617):
        
        # Call to get_legend_handler(...): (line 617)
        # Processing the call arguments (line 617)
        # Getting the type of 'legend_handler_map' (line 617)
        legend_handler_map_67700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 46), 'legend_handler_map', False)
        # Getting the type of 'orig_handle' (line 617)
        orig_handle_67701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 66), 'orig_handle', False)
        # Processing the call keyword arguments (line 617)
        kwargs_67702 = {}
        # Getting the type of 'self' (line 617)
        self_67698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 22), 'self', False)
        # Obtaining the member 'get_legend_handler' of a type (line 617)
        get_legend_handler_67699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 22), self_67698, 'get_legend_handler')
        # Calling get_legend_handler(args, kwargs) (line 617)
        get_legend_handler_call_result_67703 = invoke(stypy.reporting.localization.Localization(__file__, 617, 22), get_legend_handler_67699, *[legend_handler_map_67700, orig_handle_67701], **kwargs_67702)
        
        # Assigning a type to the variable 'handler' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'handler', get_legend_handler_call_result_67703)
        
        # Type idiom detected: calculating its left and rigth part (line 618)
        # Getting the type of 'handler' (line 618)
        handler_67704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 15), 'handler')
        # Getting the type of 'None' (line 618)
        None_67705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 26), 'None')
        
        (may_be_67706, more_types_in_union_67707) = may_be_none(handler_67704, None_67705)

        if may_be_67706:

            if more_types_in_union_67707:
                # Runtime conditional SSA (line 618)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to warn(...): (line 619)
            # Processing the call arguments (line 619)
            
            # Call to format(...): (line 620)
            # Processing the call arguments (line 620)
            # Getting the type of 'orig_handle' (line 623)
            orig_handle_67712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 49), 'orig_handle', False)
            # Processing the call keyword arguments (line 620)
            kwargs_67713 = {}
            unicode_67710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 20), 'unicode', u'Legend does not support {!r} instances.\nA proxy artist may be used instead.\nSee: http://matplotlib.org/users/legend_guide.html#using-proxy-artist')
            # Obtaining the member 'format' of a type (line 620)
            format_67711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 20), unicode_67710, 'format')
            # Calling format(args, kwargs) (line 620)
            format_call_result_67714 = invoke(stypy.reporting.localization.Localization(__file__, 620, 20), format_67711, *[orig_handle_67712], **kwargs_67713)
            
            # Processing the call keyword arguments (line 619)
            kwargs_67715 = {}
            # Getting the type of 'warnings' (line 619)
            warnings_67708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 619)
            warn_67709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 16), warnings_67708, 'warn')
            # Calling warn(args, kwargs) (line 619)
            warn_call_result_67716 = invoke(stypy.reporting.localization.Localization(__file__, 619, 16), warn_67709, *[format_call_result_67714], **kwargs_67715)
            
            
            # Call to append(...): (line 627)
            # Processing the call arguments (line 627)
            # Getting the type of 'None' (line 627)
            None_67719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 35), 'None', False)
            # Processing the call keyword arguments (line 627)
            kwargs_67720 = {}
            # Getting the type of 'handle_list' (line 627)
            handle_list_67717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 16), 'handle_list', False)
            # Obtaining the member 'append' of a type (line 627)
            append_67718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 16), handle_list_67717, 'append')
            # Calling append(args, kwargs) (line 627)
            append_call_result_67721 = invoke(stypy.reporting.localization.Localization(__file__, 627, 16), append_67718, *[None_67719], **kwargs_67720)
            

            if more_types_in_union_67707:
                # Runtime conditional SSA for else branch (line 618)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_67706) or more_types_in_union_67707):
            
            # Assigning a Call to a Name (line 629):
            
            # Assigning a Call to a Name (line 629):
            
            # Assigning a Call to a Name (line 629):
            
            # Call to TextArea(...): (line 629)
            # Processing the call arguments (line 629)
            # Getting the type of 'lab' (line 629)
            lab_67723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 35), 'lab', False)
            # Processing the call keyword arguments (line 629)
            # Getting the type of 'label_prop' (line 629)
            label_prop_67724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 50), 'label_prop', False)
            keyword_67725 = label_prop_67724
            # Getting the type of 'True' (line 630)
            True_67726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 53), 'True', False)
            keyword_67727 = True_67726
            # Getting the type of 'True' (line 631)
            True_67728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 50), 'True', False)
            keyword_67729 = True_67728
            kwargs_67730 = {'textprops': keyword_67725, 'multilinebaseline': keyword_67727, 'minimumdescent': keyword_67729}
            # Getting the type of 'TextArea' (line 629)
            TextArea_67722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 26), 'TextArea', False)
            # Calling TextArea(args, kwargs) (line 629)
            TextArea_call_result_67731 = invoke(stypy.reporting.localization.Localization(__file__, 629, 26), TextArea_67722, *[lab_67723], **kwargs_67730)
            
            # Assigning a type to the variable 'textbox' (line 629)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 16), 'textbox', TextArea_call_result_67731)
            
            # Call to append(...): (line 632)
            # Processing the call arguments (line 632)
            # Getting the type of 'textbox' (line 632)
            textbox_67734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 33), 'textbox', False)
            # Obtaining the member '_text' of a type (line 632)
            _text_67735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 33), textbox_67734, '_text')
            # Processing the call keyword arguments (line 632)
            kwargs_67736 = {}
            # Getting the type of 'text_list' (line 632)
            text_list_67732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 16), 'text_list', False)
            # Obtaining the member 'append' of a type (line 632)
            append_67733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 16), text_list_67732, 'append')
            # Calling append(args, kwargs) (line 632)
            append_call_result_67737 = invoke(stypy.reporting.localization.Localization(__file__, 632, 16), append_67733, *[_text_67735], **kwargs_67736)
            
            
            # Call to append(...): (line 634)
            # Processing the call arguments (line 634)
            # Getting the type of 'textbox' (line 634)
            textbox_67740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 34), 'textbox', False)
            # Processing the call keyword arguments (line 634)
            kwargs_67741 = {}
            # Getting the type of 'labelboxes' (line 634)
            labelboxes_67738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 16), 'labelboxes', False)
            # Obtaining the member 'append' of a type (line 634)
            append_67739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 16), labelboxes_67738, 'append')
            # Calling append(args, kwargs) (line 634)
            append_call_result_67742 = invoke(stypy.reporting.localization.Localization(__file__, 634, 16), append_67739, *[textbox_67740], **kwargs_67741)
            
            
            # Assigning a Call to a Name (line 636):
            
            # Assigning a Call to a Name (line 636):
            
            # Assigning a Call to a Name (line 636):
            
            # Call to DrawingArea(...): (line 636)
            # Processing the call keyword arguments (line 636)
            # Getting the type of 'self' (line 636)
            self_67744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 46), 'self', False)
            # Obtaining the member 'handlelength' of a type (line 636)
            handlelength_67745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 46), self_67744, 'handlelength')
            # Getting the type of 'fontsize' (line 636)
            fontsize_67746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 66), 'fontsize', False)
            # Applying the binary operator '*' (line 636)
            result_mul_67747 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 46), '*', handlelength_67745, fontsize_67746)
            
            keyword_67748 = result_mul_67747
            # Getting the type of 'height' (line 637)
            height_67749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 47), 'height', False)
            keyword_67750 = height_67749
            float_67751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 49), 'float')
            keyword_67752 = float_67751
            # Getting the type of 'descent' (line 638)
            descent_67753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 62), 'descent', False)
            keyword_67754 = descent_67753
            kwargs_67755 = {'width': keyword_67748, 'xdescent': keyword_67752, 'ydescent': keyword_67754, 'height': keyword_67750}
            # Getting the type of 'DrawingArea' (line 636)
            DrawingArea_67743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 28), 'DrawingArea', False)
            # Calling DrawingArea(args, kwargs) (line 636)
            DrawingArea_call_result_67756 = invoke(stypy.reporting.localization.Localization(__file__, 636, 28), DrawingArea_67743, *[], **kwargs_67755)
            
            # Assigning a type to the variable 'handlebox' (line 636)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'handlebox', DrawingArea_call_result_67756)
            
            # Call to append(...): (line 639)
            # Processing the call arguments (line 639)
            # Getting the type of 'handlebox' (line 639)
            handlebox_67759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 35), 'handlebox', False)
            # Processing the call keyword arguments (line 639)
            kwargs_67760 = {}
            # Getting the type of 'handleboxes' (line 639)
            handleboxes_67757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 16), 'handleboxes', False)
            # Obtaining the member 'append' of a type (line 639)
            append_67758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 16), handleboxes_67757, 'append')
            # Calling append(args, kwargs) (line 639)
            append_call_result_67761 = invoke(stypy.reporting.localization.Localization(__file__, 639, 16), append_67758, *[handlebox_67759], **kwargs_67760)
            
            
            # Call to append(...): (line 643)
            # Processing the call arguments (line 643)
            
            # Call to legend_artist(...): (line 643)
            # Processing the call arguments (line 643)
            # Getting the type of 'self' (line 643)
            self_67766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 57), 'self', False)
            # Getting the type of 'orig_handle' (line 643)
            orig_handle_67767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 63), 'orig_handle', False)
            # Getting the type of 'fontsize' (line 644)
            fontsize_67768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 57), 'fontsize', False)
            # Getting the type of 'handlebox' (line 644)
            handlebox_67769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 67), 'handlebox', False)
            # Processing the call keyword arguments (line 643)
            kwargs_67770 = {}
            # Getting the type of 'handler' (line 643)
            handler_67764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 35), 'handler', False)
            # Obtaining the member 'legend_artist' of a type (line 643)
            legend_artist_67765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 35), handler_67764, 'legend_artist')
            # Calling legend_artist(args, kwargs) (line 643)
            legend_artist_call_result_67771 = invoke(stypy.reporting.localization.Localization(__file__, 643, 35), legend_artist_67765, *[self_67766, orig_handle_67767, fontsize_67768, handlebox_67769], **kwargs_67770)
            
            # Processing the call keyword arguments (line 643)
            kwargs_67772 = {}
            # Getting the type of 'handle_list' (line 643)
            handle_list_67762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 16), 'handle_list', False)
            # Obtaining the member 'append' of a type (line 643)
            append_67763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 16), handle_list_67762, 'append')
            # Calling append(args, kwargs) (line 643)
            append_call_result_67773 = invoke(stypy.reporting.localization.Localization(__file__, 643, 16), append_67763, *[legend_artist_call_result_67771], **kwargs_67772)
            

            if (may_be_67706 and more_types_in_union_67707):
                # SSA join for if statement (line 618)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'handleboxes' (line 646)
        handleboxes_67774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 11), 'handleboxes')
        # Testing the type of an if condition (line 646)
        if_condition_67775 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 646, 8), handleboxes_67774)
        # Assigning a type to the variable 'if_condition_67775' (line 646)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'if_condition_67775', if_condition_67775)
        # SSA begins for if statement (line 646)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 650):
        
        # Assigning a Call to a Name (line 650):
        
        # Assigning a Call to a Name (line 650):
        
        # Call to min(...): (line 650)
        # Processing the call arguments (line 650)
        # Getting the type of 'self' (line 650)
        self_67777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 23), 'self', False)
        # Obtaining the member '_ncol' of a type (line 650)
        _ncol_67778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 23), self_67777, '_ncol')
        
        # Call to len(...): (line 650)
        # Processing the call arguments (line 650)
        # Getting the type of 'handleboxes' (line 650)
        handleboxes_67780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 39), 'handleboxes', False)
        # Processing the call keyword arguments (line 650)
        kwargs_67781 = {}
        # Getting the type of 'len' (line 650)
        len_67779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 35), 'len', False)
        # Calling len(args, kwargs) (line 650)
        len_call_result_67782 = invoke(stypy.reporting.localization.Localization(__file__, 650, 35), len_67779, *[handleboxes_67780], **kwargs_67781)
        
        # Processing the call keyword arguments (line 650)
        kwargs_67783 = {}
        # Getting the type of 'min' (line 650)
        min_67776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 19), 'min', False)
        # Calling min(args, kwargs) (line 650)
        min_call_result_67784 = invoke(stypy.reporting.localization.Localization(__file__, 650, 19), min_67776, *[_ncol_67778, len_call_result_67782], **kwargs_67783)
        
        # Assigning a type to the variable 'ncol' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'ncol', min_call_result_67784)
        
        # Assigning a Call to a Tuple (line 651):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to divmod(...): (line 651)
        # Processing the call arguments (line 651)
        
        # Call to len(...): (line 651)
        # Processing the call arguments (line 651)
        # Getting the type of 'handleboxes' (line 651)
        handleboxes_67787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 45), 'handleboxes', False)
        # Processing the call keyword arguments (line 651)
        kwargs_67788 = {}
        # Getting the type of 'len' (line 651)
        len_67786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 41), 'len', False)
        # Calling len(args, kwargs) (line 651)
        len_call_result_67789 = invoke(stypy.reporting.localization.Localization(__file__, 651, 41), len_67786, *[handleboxes_67787], **kwargs_67788)
        
        # Getting the type of 'ncol' (line 651)
        ncol_67790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 59), 'ncol', False)
        # Processing the call keyword arguments (line 651)
        kwargs_67791 = {}
        # Getting the type of 'divmod' (line 651)
        divmod_67785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 34), 'divmod', False)
        # Calling divmod(args, kwargs) (line 651)
        divmod_call_result_67792 = invoke(stypy.reporting.localization.Localization(__file__, 651, 34), divmod_67785, *[len_call_result_67789, ncol_67790], **kwargs_67791)
        
        # Assigning a type to the variable 'call_assignment_66648' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'call_assignment_66648', divmod_call_result_67792)
        
        # Assigning a Call to a Name (line 651):
        
        # Assigning a Call to a Name (line 651):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_67795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 12), 'int')
        # Processing the call keyword arguments
        kwargs_67796 = {}
        # Getting the type of 'call_assignment_66648' (line 651)
        call_assignment_66648_67793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'call_assignment_66648', False)
        # Obtaining the member '__getitem__' of a type (line 651)
        getitem___67794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 12), call_assignment_66648_67793, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_67797 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___67794, *[int_67795], **kwargs_67796)
        
        # Assigning a type to the variable 'call_assignment_66649' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'call_assignment_66649', getitem___call_result_67797)
        
        # Assigning a Name to a Name (line 651):
        
        # Assigning a Name to a Name (line 651):
        # Getting the type of 'call_assignment_66649' (line 651)
        call_assignment_66649_67798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'call_assignment_66649')
        # Assigning a type to the variable 'nrows' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'nrows', call_assignment_66649_67798)
        
        # Assigning a Call to a Name (line 651):
        
        # Assigning a Call to a Name (line 651):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_67801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 12), 'int')
        # Processing the call keyword arguments
        kwargs_67802 = {}
        # Getting the type of 'call_assignment_66648' (line 651)
        call_assignment_66648_67799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'call_assignment_66648', False)
        # Obtaining the member '__getitem__' of a type (line 651)
        getitem___67800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 12), call_assignment_66648_67799, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_67803 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___67800, *[int_67801], **kwargs_67802)
        
        # Assigning a type to the variable 'call_assignment_66650' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'call_assignment_66650', getitem___call_result_67803)
        
        # Assigning a Name to a Name (line 651):
        
        # Assigning a Name to a Name (line 651):
        # Getting the type of 'call_assignment_66650' (line 651)
        call_assignment_66650_67804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'call_assignment_66650')
        # Assigning a type to the variable 'num_largecol' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 19), 'num_largecol', call_assignment_66650_67804)
        
        # Assigning a BinOp to a Name (line 652):
        
        # Assigning a BinOp to a Name (line 652):
        
        # Assigning a BinOp to a Name (line 652):
        # Getting the type of 'ncol' (line 652)
        ncol_67805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 27), 'ncol')
        # Getting the type of 'num_largecol' (line 652)
        num_largecol_67806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 34), 'num_largecol')
        # Applying the binary operator '-' (line 652)
        result_sub_67807 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 27), '-', ncol_67805, num_largecol_67806)
        
        # Assigning a type to the variable 'num_smallcol' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'num_smallcol', result_sub_67807)
        
        # Assigning a BinOp to a Name (line 654):
        
        # Assigning a BinOp to a Name (line 654):
        
        # Assigning a BinOp to a Name (line 654):
        
        # Obtaining an instance of the builtin type 'list' (line 654)
        list_67808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 654)
        # Adding element type (line 654)
        # Getting the type of 'nrows' (line 654)
        nrows_67809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 28), 'nrows')
        int_67810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 36), 'int')
        # Applying the binary operator '+' (line 654)
        result_add_67811 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 28), '+', nrows_67809, int_67810)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 27), list_67808, result_add_67811)
        
        # Getting the type of 'num_largecol' (line 654)
        num_largecol_67812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 41), 'num_largecol')
        # Applying the binary operator '*' (line 654)
        result_mul_67813 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 27), '*', list_67808, num_largecol_67812)
        
        
        # Obtaining an instance of the builtin type 'list' (line 654)
        list_67814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 654)
        # Adding element type (line 654)
        # Getting the type of 'nrows' (line 654)
        nrows_67815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 57), 'nrows')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 654, 56), list_67814, nrows_67815)
        
        # Getting the type of 'num_smallcol' (line 654)
        num_smallcol_67816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 66), 'num_smallcol')
        # Applying the binary operator '*' (line 654)
        result_mul_67817 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 56), '*', list_67814, num_smallcol_67816)
        
        # Applying the binary operator '+' (line 654)
        result_add_67818 = python_operator(stypy.reporting.localization.Localization(__file__, 654, 27), '+', result_mul_67813, result_mul_67817)
        
        # Assigning a type to the variable 'rows_per_col' (line 654)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'rows_per_col', result_add_67818)
        
        # Assigning a Call to a Name (line 655):
        
        # Assigning a Call to a Name (line 655):
        
        # Assigning a Call to a Name (line 655):
        
        # Call to concatenate(...): (line 655)
        # Processing the call arguments (line 655)
        
        # Obtaining an instance of the builtin type 'list' (line 655)
        list_67821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 655)
        # Adding element type (line 655)
        
        # Obtaining an instance of the builtin type 'list' (line 655)
        list_67822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 655)
        # Adding element type (line 655)
        int_67823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 41), list_67822, int_67823)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 40), list_67821, list_67822)
        # Adding element type (line 655)
        
        # Obtaining the type of the subscript
        int_67824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 71), 'int')
        slice_67825 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 655, 46), None, int_67824, None)
        
        # Call to cumsum(...): (line 655)
        # Processing the call arguments (line 655)
        # Getting the type of 'rows_per_col' (line 655)
        rows_per_col_67828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 56), 'rows_per_col', False)
        # Processing the call keyword arguments (line 655)
        kwargs_67829 = {}
        # Getting the type of 'np' (line 655)
        np_67826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 46), 'np', False)
        # Obtaining the member 'cumsum' of a type (line 655)
        cumsum_67827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 46), np_67826, 'cumsum')
        # Calling cumsum(args, kwargs) (line 655)
        cumsum_call_result_67830 = invoke(stypy.reporting.localization.Localization(__file__, 655, 46), cumsum_67827, *[rows_per_col_67828], **kwargs_67829)
        
        # Obtaining the member '__getitem__' of a type (line 655)
        getitem___67831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 46), cumsum_call_result_67830, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 655)
        subscript_call_result_67832 = invoke(stypy.reporting.localization.Localization(__file__, 655, 46), getitem___67831, slice_67825)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 40), list_67821, subscript_call_result_67832)
        
        # Processing the call keyword arguments (line 655)
        kwargs_67833 = {}
        # Getting the type of 'np' (line 655)
        np_67819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 25), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 655)
        concatenate_67820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 25), np_67819, 'concatenate')
        # Calling concatenate(args, kwargs) (line 655)
        concatenate_call_result_67834 = invoke(stypy.reporting.localization.Localization(__file__, 655, 25), concatenate_67820, *[list_67821], **kwargs_67833)
        
        # Assigning a type to the variable 'start_idxs' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 12), 'start_idxs', concatenate_call_result_67834)
        
        # Assigning a Call to a Name (line 656):
        
        # Assigning a Call to a Name (line 656):
        
        # Assigning a Call to a Name (line 656):
        
        # Call to zip(...): (line 656)
        # Processing the call arguments (line 656)
        # Getting the type of 'start_idxs' (line 656)
        start_idxs_67836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 23), 'start_idxs', False)
        # Getting the type of 'rows_per_col' (line 656)
        rows_per_col_67837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 35), 'rows_per_col', False)
        # Processing the call keyword arguments (line 656)
        kwargs_67838 = {}
        # Getting the type of 'zip' (line 656)
        zip_67835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 19), 'zip', False)
        # Calling zip(args, kwargs) (line 656)
        zip_call_result_67839 = invoke(stypy.reporting.localization.Localization(__file__, 656, 19), zip_67835, *[start_idxs_67836, rows_per_col_67837], **kwargs_67838)
        
        # Assigning a type to the variable 'cols' (line 656)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 12), 'cols', zip_call_result_67839)
        # SSA branch for the else part of an if statement (line 646)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 658):
        
        # Assigning a List to a Name (line 658):
        
        # Assigning a List to a Name (line 658):
        
        # Obtaining an instance of the builtin type 'list' (line 658)
        list_67840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 658)
        
        # Assigning a type to the variable 'cols' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'cols', list_67840)
        # SSA join for if statement (line 646)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 660):
        
        # Assigning a Call to a Name (line 660):
        
        # Assigning a Call to a Name (line 660):
        
        # Call to list(...): (line 660)
        # Processing the call arguments (line 660)
        
        # Call to zip(...): (line 660)
        # Processing the call arguments (line 660)
        # Getting the type of 'handleboxes' (line 660)
        handleboxes_67843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 32), 'handleboxes', False)
        # Getting the type of 'labelboxes' (line 660)
        labelboxes_67844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 45), 'labelboxes', False)
        # Processing the call keyword arguments (line 660)
        kwargs_67845 = {}
        # Getting the type of 'zip' (line 660)
        zip_67842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 28), 'zip', False)
        # Calling zip(args, kwargs) (line 660)
        zip_call_result_67846 = invoke(stypy.reporting.localization.Localization(__file__, 660, 28), zip_67842, *[handleboxes_67843, labelboxes_67844], **kwargs_67845)
        
        # Processing the call keyword arguments (line 660)
        kwargs_67847 = {}
        # Getting the type of 'list' (line 660)
        list_67841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 23), 'list', False)
        # Calling list(args, kwargs) (line 660)
        list_call_result_67848 = invoke(stypy.reporting.localization.Localization(__file__, 660, 23), list_67841, *[zip_call_result_67846], **kwargs_67847)
        
        # Assigning a type to the variable 'handle_label' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'handle_label', list_call_result_67848)
        
        # Assigning a List to a Name (line 661):
        
        # Assigning a List to a Name (line 661):
        
        # Assigning a List to a Name (line 661):
        
        # Obtaining an instance of the builtin type 'list' (line 661)
        list_67849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 661)
        
        # Assigning a type to the variable 'columnbox' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 8), 'columnbox', list_67849)
        
        # Getting the type of 'cols' (line 662)
        cols_67850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 22), 'cols')
        # Testing the type of a for loop iterable (line 662)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 662, 8), cols_67850)
        # Getting the type of the for loop variable (line 662)
        for_loop_var_67851 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 662, 8), cols_67850)
        # Assigning a type to the variable 'i0' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'i0', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 8), for_loop_var_67851))
        # Assigning a type to the variable 'di' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'di', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 8), for_loop_var_67851))
        # SSA begins for a for statement (line 662)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a ListComp to a Name (line 664):
        
        # Assigning a ListComp to a Name (line 664):
        
        # Assigning a ListComp to a Name (line 664):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining the type of the subscript
        # Getting the type of 'i0' (line 668)
        i0_67873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 50), 'i0')
        # Getting the type of 'i0' (line 668)
        i0_67874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 53), 'i0')
        # Getting the type of 'di' (line 668)
        di_67875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 58), 'di')
        # Applying the binary operator '+' (line 668)
        result_add_67876 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 53), '+', i0_67874, di_67875)
        
        slice_67877 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 668, 37), i0_67873, result_add_67876, None)
        # Getting the type of 'handle_label' (line 668)
        handle_label_67878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 37), 'handle_label')
        # Obtaining the member '__getitem__' of a type (line 668)
        getitem___67879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 37), handle_label_67878, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 668)
        subscript_call_result_67880 = invoke(stypy.reporting.localization.Localization(__file__, 668, 37), getitem___67879, slice_67877)
        
        comprehension_67881 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 25), subscript_call_result_67880)
        # Assigning a type to the variable 'h' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 25), 'h', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 25), comprehension_67881))
        # Assigning a type to the variable 't' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 25), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 25), comprehension_67881))
        
        # Call to HPacker(...): (line 664)
        # Processing the call keyword arguments (line 664)
        int_67853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 37), 'int')
        keyword_67854 = int_67853
        # Getting the type of 'self' (line 665)
        self_67855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 37), 'self', False)
        # Obtaining the member 'handletextpad' of a type (line 665)
        handletextpad_67856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 37), self_67855, 'handletextpad')
        # Getting the type of 'fontsize' (line 665)
        fontsize_67857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 58), 'fontsize', False)
        # Applying the binary operator '*' (line 665)
        result_mul_67858 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 37), '*', handletextpad_67856, fontsize_67857)
        
        keyword_67859 = result_mul_67858
        
        # Getting the type of 'markerfirst' (line 666)
        markerfirst_67860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 52), 'markerfirst', False)
        # Testing the type of an if expression (line 666)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 42), markerfirst_67860)
        # SSA begins for if expression (line 666)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Obtaining an instance of the builtin type 'list' (line 666)
        list_67861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 666)
        # Adding element type (line 666)
        # Getting the type of 'h' (line 666)
        h_67862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 43), 'h', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 42), list_67861, h_67862)
        # Adding element type (line 666)
        # Getting the type of 't' (line 666)
        t_67863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 46), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 42), list_67861, t_67863)
        
        # SSA branch for the else part of an if expression (line 666)
        module_type_store.open_ssa_branch('if expression else')
        
        # Obtaining an instance of the builtin type 'list' (line 666)
        list_67864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 69), 'list')
        # Adding type elements to the builtin type 'list' instance (line 666)
        # Adding element type (line 666)
        # Getting the type of 't' (line 666)
        t_67865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 70), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 69), list_67864, t_67865)
        # Adding element type (line 666)
        # Getting the type of 'h' (line 666)
        h_67866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 73), 'h', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 69), list_67864, h_67866)
        
        # SSA join for if expression (line 666)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_67867 = union_type.UnionType.add(list_67861, list_67864)
        
        keyword_67868 = if_exp_67867
        unicode_67869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 39), 'unicode', u'baseline')
        keyword_67870 = unicode_67869
        kwargs_67871 = {'align': keyword_67870, 'pad': keyword_67854, 'children': keyword_67868, 'sep': keyword_67859}
        # Getting the type of 'HPacker' (line 664)
        HPacker_67852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 25), 'HPacker', False)
        # Calling HPacker(args, kwargs) (line 664)
        HPacker_call_result_67872 = invoke(stypy.reporting.localization.Localization(__file__, 664, 25), HPacker_67852, *[], **kwargs_67871)
        
        list_67882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 25), list_67882, HPacker_call_result_67872)
        # Assigning a type to the variable 'itemBoxes' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'itemBoxes', list_67882)
        
        # Getting the type of 'markerfirst' (line 670)
        markerfirst_67883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 15), 'markerfirst')
        # Testing the type of an if condition (line 670)
        if_condition_67884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 670, 12), markerfirst_67883)
        # Assigning a type to the variable 'if_condition_67884' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), 'if_condition_67884', if_condition_67884)
        # SSA begins for if statement (line 670)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_minimumdescent(...): (line 671)
        # Processing the call arguments (line 671)
        # Getting the type of 'False' (line 671)
        False_67896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 67), 'False', False)
        # Processing the call keyword arguments (line 671)
        kwargs_67897 = {}
        
        # Obtaining the type of the subscript
        int_67885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 45), 'int')
        
        # Call to get_children(...): (line 671)
        # Processing the call keyword arguments (line 671)
        kwargs_67891 = {}
        
        # Obtaining the type of the subscript
        int_67886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 26), 'int')
        # Getting the type of 'itemBoxes' (line 671)
        itemBoxes_67887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 16), 'itemBoxes', False)
        # Obtaining the member '__getitem__' of a type (line 671)
        getitem___67888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 16), itemBoxes_67887, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 671)
        subscript_call_result_67889 = invoke(stypy.reporting.localization.Localization(__file__, 671, 16), getitem___67888, int_67886)
        
        # Obtaining the member 'get_children' of a type (line 671)
        get_children_67890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 16), subscript_call_result_67889, 'get_children')
        # Calling get_children(args, kwargs) (line 671)
        get_children_call_result_67892 = invoke(stypy.reporting.localization.Localization(__file__, 671, 16), get_children_67890, *[], **kwargs_67891)
        
        # Obtaining the member '__getitem__' of a type (line 671)
        getitem___67893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 16), get_children_call_result_67892, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 671)
        subscript_call_result_67894 = invoke(stypy.reporting.localization.Localization(__file__, 671, 16), getitem___67893, int_67885)
        
        # Obtaining the member 'set_minimumdescent' of a type (line 671)
        set_minimumdescent_67895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 16), subscript_call_result_67894, 'set_minimumdescent')
        # Calling set_minimumdescent(args, kwargs) (line 671)
        set_minimumdescent_call_result_67898 = invoke(stypy.reporting.localization.Localization(__file__, 671, 16), set_minimumdescent_67895, *[False_67896], **kwargs_67897)
        
        # SSA branch for the else part of an if statement (line 670)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_minimumdescent(...): (line 673)
        # Processing the call arguments (line 673)
        # Getting the type of 'False' (line 673)
        False_67910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 67), 'False', False)
        # Processing the call keyword arguments (line 673)
        kwargs_67911 = {}
        
        # Obtaining the type of the subscript
        int_67899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 45), 'int')
        
        # Call to get_children(...): (line 673)
        # Processing the call keyword arguments (line 673)
        kwargs_67905 = {}
        
        # Obtaining the type of the subscript
        int_67900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 26), 'int')
        # Getting the type of 'itemBoxes' (line 673)
        itemBoxes_67901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 16), 'itemBoxes', False)
        # Obtaining the member '__getitem__' of a type (line 673)
        getitem___67902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 16), itemBoxes_67901, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 673)
        subscript_call_result_67903 = invoke(stypy.reporting.localization.Localization(__file__, 673, 16), getitem___67902, int_67900)
        
        # Obtaining the member 'get_children' of a type (line 673)
        get_children_67904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 16), subscript_call_result_67903, 'get_children')
        # Calling get_children(args, kwargs) (line 673)
        get_children_call_result_67906 = invoke(stypy.reporting.localization.Localization(__file__, 673, 16), get_children_67904, *[], **kwargs_67905)
        
        # Obtaining the member '__getitem__' of a type (line 673)
        getitem___67907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 16), get_children_call_result_67906, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 673)
        subscript_call_result_67908 = invoke(stypy.reporting.localization.Localization(__file__, 673, 16), getitem___67907, int_67899)
        
        # Obtaining the member 'set_minimumdescent' of a type (line 673)
        set_minimumdescent_67909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 16), subscript_call_result_67908, 'set_minimumdescent')
        # Calling set_minimumdescent(args, kwargs) (line 673)
        set_minimumdescent_call_result_67912 = invoke(stypy.reporting.localization.Localization(__file__, 673, 16), set_minimumdescent_67909, *[False_67910], **kwargs_67911)
        
        # SSA join for if statement (line 670)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a IfExp to a Name (line 676):
        
        # Assigning a IfExp to a Name (line 676):
        
        # Assigning a IfExp to a Name (line 676):
        
        # Getting the type of 'markerfirst' (line 676)
        markerfirst_67913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 38), 'markerfirst')
        # Testing the type of an if expression (line 676)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 676, 24), markerfirst_67913)
        # SSA begins for if expression (line 676)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        unicode_67914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 24), 'unicode', u'baseline')
        # SSA branch for the else part of an if expression (line 676)
        module_type_store.open_ssa_branch('if expression else')
        unicode_67915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 55), 'unicode', u'right')
        # SSA join for if expression (line 676)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_67916 = union_type.UnionType.add(unicode_67914, unicode_67915)
        
        # Assigning a type to the variable 'alignment' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 12), 'alignment', if_exp_67916)
        
        # Call to append(...): (line 677)
        # Processing the call arguments (line 677)
        
        # Call to VPacker(...): (line 677)
        # Processing the call keyword arguments (line 677)
        int_67920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 41), 'int')
        keyword_67921 = int_67920
        # Getting the type of 'self' (line 678)
        self_67922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 41), 'self', False)
        # Obtaining the member 'labelspacing' of a type (line 678)
        labelspacing_67923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 41), self_67922, 'labelspacing')
        # Getting the type of 'fontsize' (line 678)
        fontsize_67924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 61), 'fontsize', False)
        # Applying the binary operator '*' (line 678)
        result_mul_67925 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 41), '*', labelspacing_67923, fontsize_67924)
        
        keyword_67926 = result_mul_67925
        # Getting the type of 'alignment' (line 679)
        alignment_67927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 43), 'alignment', False)
        keyword_67928 = alignment_67927
        # Getting the type of 'itemBoxes' (line 680)
        itemBoxes_67929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 46), 'itemBoxes', False)
        keyword_67930 = itemBoxes_67929
        kwargs_67931 = {'align': keyword_67928, 'pad': keyword_67921, 'children': keyword_67930, 'sep': keyword_67926}
        # Getting the type of 'VPacker' (line 677)
        VPacker_67919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 29), 'VPacker', False)
        # Calling VPacker(args, kwargs) (line 677)
        VPacker_call_result_67932 = invoke(stypy.reporting.localization.Localization(__file__, 677, 29), VPacker_67919, *[], **kwargs_67931)
        
        # Processing the call keyword arguments (line 677)
        kwargs_67933 = {}
        # Getting the type of 'columnbox' (line 677)
        columnbox_67917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 12), 'columnbox', False)
        # Obtaining the member 'append' of a type (line 677)
        append_67918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 12), columnbox_67917, 'append')
        # Calling append(args, kwargs) (line 677)
        append_call_result_67934 = invoke(stypy.reporting.localization.Localization(__file__, 677, 12), append_67918, *[VPacker_call_result_67932], **kwargs_67933)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a IfExp to a Name (line 682):
        
        # Assigning a IfExp to a Name (line 682):
        
        # Assigning a IfExp to a Name (line 682):
        
        
        # Getting the type of 'self' (line 682)
        self_67935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 27), 'self')
        # Obtaining the member '_mode' of a type (line 682)
        _mode_67936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 27), self_67935, '_mode')
        unicode_67937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 41), 'unicode', u'expand')
        # Applying the binary operator '==' (line 682)
        result_eq_67938 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 27), '==', _mode_67936, unicode_67937)
        
        # Testing the type of an if expression (line 682)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 682, 15), result_eq_67938)
        # SSA begins for if expression (line 682)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        unicode_67939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 15), 'unicode', u'expand')
        # SSA branch for the else part of an if expression (line 682)
        module_type_store.open_ssa_branch('if expression else')
        unicode_67940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 55), 'unicode', u'fixed')
        # SSA join for if expression (line 682)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_67941 = union_type.UnionType.add(unicode_67939, unicode_67940)
        
        # Assigning a type to the variable 'mode' (line 682)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 8), 'mode', if_exp_67941)
        
        # Assigning a BinOp to a Name (line 683):
        
        # Assigning a BinOp to a Name (line 683):
        
        # Assigning a BinOp to a Name (line 683):
        # Getting the type of 'self' (line 683)
        self_67942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 14), 'self')
        # Obtaining the member 'columnspacing' of a type (line 683)
        columnspacing_67943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 14), self_67942, 'columnspacing')
        # Getting the type of 'fontsize' (line 683)
        fontsize_67944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 35), 'fontsize')
        # Applying the binary operator '*' (line 683)
        result_mul_67945 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 14), '*', columnspacing_67943, fontsize_67944)
        
        # Assigning a type to the variable 'sep' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'sep', result_mul_67945)
        
        # Assigning a Call to a Attribute (line 684):
        
        # Assigning a Call to a Attribute (line 684):
        
        # Assigning a Call to a Attribute (line 684):
        
        # Call to HPacker(...): (line 684)
        # Processing the call keyword arguments (line 684)
        int_67947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 46), 'int')
        keyword_67948 = int_67947
        # Getting the type of 'sep' (line 685)
        sep_67949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 46), 'sep', False)
        keyword_67950 = sep_67949
        unicode_67951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 57), 'unicode', u'baseline')
        keyword_67952 = unicode_67951
        # Getting the type of 'mode' (line 686)
        mode_67953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 47), 'mode', False)
        keyword_67954 = mode_67953
        # Getting the type of 'columnbox' (line 687)
        columnbox_67955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 51), 'columnbox', False)
        keyword_67956 = columnbox_67955
        kwargs_67957 = {'children': keyword_67956, 'align': keyword_67952, 'pad': keyword_67948, 'mode': keyword_67954, 'sep': keyword_67950}
        # Getting the type of 'HPacker' (line 684)
        HPacker_67946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 34), 'HPacker', False)
        # Calling HPacker(args, kwargs) (line 684)
        HPacker_call_result_67958 = invoke(stypy.reporting.localization.Localization(__file__, 684, 34), HPacker_67946, *[], **kwargs_67957)
        
        # Getting the type of 'self' (line 684)
        self_67959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'self')
        # Setting the type of the member '_legend_handle_box' of a type (line 684)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 8), self_67959, '_legend_handle_box', HPacker_call_result_67958)
        
        # Assigning a Call to a Attribute (line 688):
        
        # Assigning a Call to a Attribute (line 688):
        
        # Assigning a Call to a Attribute (line 688):
        
        # Call to TextArea(...): (line 688)
        # Processing the call arguments (line 688)
        unicode_67961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 42), 'unicode', u'')
        # Processing the call keyword arguments (line 688)
        kwargs_67962 = {}
        # Getting the type of 'TextArea' (line 688)
        TextArea_67960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 33), 'TextArea', False)
        # Calling TextArea(args, kwargs) (line 688)
        TextArea_call_result_67963 = invoke(stypy.reporting.localization.Localization(__file__, 688, 33), TextArea_67960, *[unicode_67961], **kwargs_67962)
        
        # Getting the type of 'self' (line 688)
        self_67964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 8), 'self')
        # Setting the type of the member '_legend_title_box' of a type (line 688)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 8), self_67964, '_legend_title_box', TextArea_call_result_67963)
        
        # Assigning a Call to a Attribute (line 689):
        
        # Assigning a Call to a Attribute (line 689):
        
        # Assigning a Call to a Attribute (line 689):
        
        # Call to VPacker(...): (line 689)
        # Processing the call keyword arguments (line 689)
        # Getting the type of 'self' (line 689)
        self_67966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 39), 'self', False)
        # Obtaining the member 'borderpad' of a type (line 689)
        borderpad_67967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 39), self_67966, 'borderpad')
        # Getting the type of 'fontsize' (line 689)
        fontsize_67968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 56), 'fontsize', False)
        # Applying the binary operator '*' (line 689)
        result_mul_67969 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 39), '*', borderpad_67967, fontsize_67968)
        
        keyword_67970 = result_mul_67969
        # Getting the type of 'self' (line 690)
        self_67971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 39), 'self', False)
        # Obtaining the member 'labelspacing' of a type (line 690)
        labelspacing_67972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 39), self_67971, 'labelspacing')
        # Getting the type of 'fontsize' (line 690)
        fontsize_67973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 59), 'fontsize', False)
        # Applying the binary operator '*' (line 690)
        result_mul_67974 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 39), '*', labelspacing_67972, fontsize_67973)
        
        keyword_67975 = result_mul_67974
        unicode_67976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 41), 'unicode', u'center')
        keyword_67977 = unicode_67976
        
        # Obtaining an instance of the builtin type 'list' (line 692)
        list_67978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 692)
        # Adding element type (line 692)
        # Getting the type of 'self' (line 692)
        self_67979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 45), 'self', False)
        # Obtaining the member '_legend_title_box' of a type (line 692)
        _legend_title_box_67980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 45), self_67979, '_legend_title_box')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 44), list_67978, _legend_title_box_67980)
        # Adding element type (line 692)
        # Getting the type of 'self' (line 693)
        self_67981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 45), 'self', False)
        # Obtaining the member '_legend_handle_box' of a type (line 693)
        _legend_handle_box_67982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 45), self_67981, '_legend_handle_box')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 44), list_67978, _legend_handle_box_67982)
        
        keyword_67983 = list_67978
        kwargs_67984 = {'align': keyword_67977, 'pad': keyword_67970, 'children': keyword_67983, 'sep': keyword_67975}
        # Getting the type of 'VPacker' (line 689)
        VPacker_67965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 27), 'VPacker', False)
        # Calling VPacker(args, kwargs) (line 689)
        VPacker_call_result_67985 = invoke(stypy.reporting.localization.Localization(__file__, 689, 27), VPacker_67965, *[], **kwargs_67984)
        
        # Getting the type of 'self' (line 689)
        self_67986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'self')
        # Setting the type of the member '_legend_box' of a type (line 689)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), self_67986, '_legend_box', VPacker_call_result_67985)
        
        # Call to set_figure(...): (line 694)
        # Processing the call arguments (line 694)
        # Getting the type of 'self' (line 694)
        self_67990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 36), 'self', False)
        # Obtaining the member 'figure' of a type (line 694)
        figure_67991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 36), self_67990, 'figure')
        # Processing the call keyword arguments (line 694)
        kwargs_67992 = {}
        # Getting the type of 'self' (line 694)
        self_67987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'self', False)
        # Obtaining the member '_legend_box' of a type (line 694)
        _legend_box_67988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 8), self_67987, '_legend_box')
        # Obtaining the member 'set_figure' of a type (line 694)
        set_figure_67989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 8), _legend_box_67988, 'set_figure')
        # Calling set_figure(args, kwargs) (line 694)
        set_figure_call_result_67993 = invoke(stypy.reporting.localization.Localization(__file__, 694, 8), set_figure_67989, *[figure_67991], **kwargs_67992)
        
        
        # Call to set_offset(...): (line 695)
        # Processing the call arguments (line 695)
        # Getting the type of 'self' (line 695)
        self_67997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 36), 'self', False)
        # Obtaining the member '_findoffset' of a type (line 695)
        _findoffset_67998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 36), self_67997, '_findoffset')
        # Processing the call keyword arguments (line 695)
        kwargs_67999 = {}
        # Getting the type of 'self' (line 695)
        self_67994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'self', False)
        # Obtaining the member '_legend_box' of a type (line 695)
        _legend_box_67995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 8), self_67994, '_legend_box')
        # Obtaining the member 'set_offset' of a type (line 695)
        set_offset_67996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 8), _legend_box_67995, 'set_offset')
        # Calling set_offset(args, kwargs) (line 695)
        set_offset_call_result_68000 = invoke(stypy.reporting.localization.Localization(__file__, 695, 8), set_offset_67996, *[_findoffset_67998], **kwargs_67999)
        
        
        # Assigning a Name to a Attribute (line 696):
        
        # Assigning a Name to a Attribute (line 696):
        
        # Assigning a Name to a Attribute (line 696):
        # Getting the type of 'text_list' (line 696)
        text_list_68001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 21), 'text_list')
        # Getting the type of 'self' (line 696)
        self_68002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'self')
        # Setting the type of the member 'texts' of a type (line 696)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 8), self_68002, 'texts', text_list_68001)
        
        # Assigning a Name to a Attribute (line 697):
        
        # Assigning a Name to a Attribute (line 697):
        
        # Assigning a Name to a Attribute (line 697):
        # Getting the type of 'handle_list' (line 697)
        handle_list_68003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 29), 'handle_list')
        # Getting the type of 'self' (line 697)
        self_68004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'self')
        # Setting the type of the member 'legendHandles' of a type (line 697)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), self_68004, 'legendHandles', handle_list_68003)
        
        # ################# End of '_init_legend_box(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_legend_box' in the type store
        # Getting the type of 'stypy_return_type' (line 572)
        stypy_return_type_68005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68005)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_legend_box'
        return stypy_return_type_68005


    @norecursion
    def _auto_legend_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_auto_legend_data'
        module_type_store = module_type_store.open_function_context('_auto_legend_data', 699, 4, False)
        # Assigning a type to the variable 'self' (line 700)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend._auto_legend_data.__dict__.__setitem__('stypy_localization', localization)
        Legend._auto_legend_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend._auto_legend_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend._auto_legend_data.__dict__.__setitem__('stypy_function_name', 'Legend._auto_legend_data')
        Legend._auto_legend_data.__dict__.__setitem__('stypy_param_names_list', [])
        Legend._auto_legend_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend._auto_legend_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend._auto_legend_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend._auto_legend_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend._auto_legend_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend._auto_legend_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend._auto_legend_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_auto_legend_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_auto_legend_data(...)' code ##################

        unicode_68006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, (-1)), 'unicode', u"\n        Returns list of vertices and extents covered by the plot.\n\n        Returns a two long list.\n\n        First element is a list of (x, y) vertices (in\n        display-coordinates) covered by all the lines and line\n        collections, in the legend's handles.\n\n        Second element is a list of bounding boxes for all the patches in\n        the legend's handles.\n        ")
        # Evaluating assert statement condition
        # Getting the type of 'self' (line 713)
        self_68007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 15), 'self')
        # Obtaining the member 'isaxes' of a type (line 713)
        isaxes_68008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 15), self_68007, 'isaxes')
        
        # Assigning a Attribute to a Name (line 715):
        
        # Assigning a Attribute to a Name (line 715):
        
        # Assigning a Attribute to a Name (line 715):
        # Getting the type of 'self' (line 715)
        self_68009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 13), 'self')
        # Obtaining the member 'parent' of a type (line 715)
        parent_68010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 13), self_68009, 'parent')
        # Assigning a type to the variable 'ax' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'ax', parent_68010)
        
        # Assigning a List to a Name (line 716):
        
        # Assigning a List to a Name (line 716):
        
        # Assigning a List to a Name (line 716):
        
        # Obtaining an instance of the builtin type 'list' (line 716)
        list_68011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 716)
        
        # Assigning a type to the variable 'bboxes' (line 716)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'bboxes', list_68011)
        
        # Assigning a List to a Name (line 717):
        
        # Assigning a List to a Name (line 717):
        
        # Assigning a List to a Name (line 717):
        
        # Obtaining an instance of the builtin type 'list' (line 717)
        list_68012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 717)
        
        # Assigning a type to the variable 'lines' (line 717)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'lines', list_68012)
        
        # Assigning a List to a Name (line 718):
        
        # Assigning a List to a Name (line 718):
        
        # Assigning a List to a Name (line 718):
        
        # Obtaining an instance of the builtin type 'list' (line 718)
        list_68013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 718)
        
        # Assigning a type to the variable 'offsets' (line 718)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'offsets', list_68013)
        
        # Getting the type of 'ax' (line 720)
        ax_68014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 22), 'ax')
        # Obtaining the member 'lines' of a type (line 720)
        lines_68015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 22), ax_68014, 'lines')
        # Testing the type of a for loop iterable (line 720)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 720, 8), lines_68015)
        # Getting the type of the for loop variable (line 720)
        for_loop_var_68016 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 720, 8), lines_68015)
        # Assigning a type to the variable 'handle' (line 720)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'handle', for_loop_var_68016)
        # SSA begins for a for statement (line 720)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 721)
        # Processing the call arguments (line 721)
        # Getting the type of 'handle' (line 721)
        handle_68018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 30), 'handle', False)
        # Getting the type of 'Line2D' (line 721)
        Line2D_68019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 38), 'Line2D', False)
        # Processing the call keyword arguments (line 721)
        kwargs_68020 = {}
        # Getting the type of 'isinstance' (line 721)
        isinstance_68017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 721)
        isinstance_call_result_68021 = invoke(stypy.reporting.localization.Localization(__file__, 721, 19), isinstance_68017, *[handle_68018, Line2D_68019], **kwargs_68020)
        
        
        # Assigning a Call to a Name (line 722):
        
        # Assigning a Call to a Name (line 722):
        
        # Assigning a Call to a Name (line 722):
        
        # Call to get_path(...): (line 722)
        # Processing the call keyword arguments (line 722)
        kwargs_68024 = {}
        # Getting the type of 'handle' (line 722)
        handle_68022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 19), 'handle', False)
        # Obtaining the member 'get_path' of a type (line 722)
        get_path_68023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 19), handle_68022, 'get_path')
        # Calling get_path(args, kwargs) (line 722)
        get_path_call_result_68025 = invoke(stypy.reporting.localization.Localization(__file__, 722, 19), get_path_68023, *[], **kwargs_68024)
        
        # Assigning a type to the variable 'path' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 12), 'path', get_path_call_result_68025)
        
        # Assigning a Call to a Name (line 723):
        
        # Assigning a Call to a Name (line 723):
        
        # Assigning a Call to a Name (line 723):
        
        # Call to get_transform(...): (line 723)
        # Processing the call keyword arguments (line 723)
        kwargs_68028 = {}
        # Getting the type of 'handle' (line 723)
        handle_68026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 20), 'handle', False)
        # Obtaining the member 'get_transform' of a type (line 723)
        get_transform_68027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 20), handle_68026, 'get_transform')
        # Calling get_transform(args, kwargs) (line 723)
        get_transform_call_result_68029 = invoke(stypy.reporting.localization.Localization(__file__, 723, 20), get_transform_68027, *[], **kwargs_68028)
        
        # Assigning a type to the variable 'trans' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 12), 'trans', get_transform_call_result_68029)
        
        # Assigning a Call to a Name (line 724):
        
        # Assigning a Call to a Name (line 724):
        
        # Assigning a Call to a Name (line 724):
        
        # Call to transform_path(...): (line 724)
        # Processing the call arguments (line 724)
        # Getting the type of 'path' (line 724)
        path_68032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 41), 'path', False)
        # Processing the call keyword arguments (line 724)
        kwargs_68033 = {}
        # Getting the type of 'trans' (line 724)
        trans_68030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 20), 'trans', False)
        # Obtaining the member 'transform_path' of a type (line 724)
        transform_path_68031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 20), trans_68030, 'transform_path')
        # Calling transform_path(args, kwargs) (line 724)
        transform_path_call_result_68034 = invoke(stypy.reporting.localization.Localization(__file__, 724, 20), transform_path_68031, *[path_68032], **kwargs_68033)
        
        # Assigning a type to the variable 'tpath' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 12), 'tpath', transform_path_call_result_68034)
        
        # Call to append(...): (line 725)
        # Processing the call arguments (line 725)
        # Getting the type of 'tpath' (line 725)
        tpath_68037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 25), 'tpath', False)
        # Processing the call keyword arguments (line 725)
        kwargs_68038 = {}
        # Getting the type of 'lines' (line 725)
        lines_68035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 12), 'lines', False)
        # Obtaining the member 'append' of a type (line 725)
        append_68036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 12), lines_68035, 'append')
        # Calling append(args, kwargs) (line 725)
        append_call_result_68039 = invoke(stypy.reporting.localization.Localization(__file__, 725, 12), append_68036, *[tpath_68037], **kwargs_68038)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'ax' (line 727)
        ax_68040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 22), 'ax')
        # Obtaining the member 'patches' of a type (line 727)
        patches_68041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 22), ax_68040, 'patches')
        # Testing the type of a for loop iterable (line 727)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 727, 8), patches_68041)
        # Getting the type of the for loop variable (line 727)
        for_loop_var_68042 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 727, 8), patches_68041)
        # Assigning a type to the variable 'handle' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'handle', for_loop_var_68042)
        # SSA begins for a for statement (line 727)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 728)
        # Processing the call arguments (line 728)
        # Getting the type of 'handle' (line 728)
        handle_68044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 30), 'handle', False)
        # Getting the type of 'Patch' (line 728)
        Patch_68045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 38), 'Patch', False)
        # Processing the call keyword arguments (line 728)
        kwargs_68046 = {}
        # Getting the type of 'isinstance' (line 728)
        isinstance_68043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 728)
        isinstance_call_result_68047 = invoke(stypy.reporting.localization.Localization(__file__, 728, 19), isinstance_68043, *[handle_68044, Patch_68045], **kwargs_68046)
        
        
        
        # Call to isinstance(...): (line 730)
        # Processing the call arguments (line 730)
        # Getting the type of 'handle' (line 730)
        handle_68049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 26), 'handle', False)
        # Getting the type of 'Rectangle' (line 730)
        Rectangle_68050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 34), 'Rectangle', False)
        # Processing the call keyword arguments (line 730)
        kwargs_68051 = {}
        # Getting the type of 'isinstance' (line 730)
        isinstance_68048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 730)
        isinstance_call_result_68052 = invoke(stypy.reporting.localization.Localization(__file__, 730, 15), isinstance_68048, *[handle_68049, Rectangle_68050], **kwargs_68051)
        
        # Testing the type of an if condition (line 730)
        if_condition_68053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 730, 12), isinstance_call_result_68052)
        # Assigning a type to the variable 'if_condition_68053' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'if_condition_68053', if_condition_68053)
        # SSA begins for if statement (line 730)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 731):
        
        # Assigning a Call to a Name (line 731):
        
        # Assigning a Call to a Name (line 731):
        
        # Call to get_data_transform(...): (line 731)
        # Processing the call keyword arguments (line 731)
        kwargs_68056 = {}
        # Getting the type of 'handle' (line 731)
        handle_68054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 28), 'handle', False)
        # Obtaining the member 'get_data_transform' of a type (line 731)
        get_data_transform_68055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 28), handle_68054, 'get_data_transform')
        # Calling get_data_transform(args, kwargs) (line 731)
        get_data_transform_call_result_68057 = invoke(stypy.reporting.localization.Localization(__file__, 731, 28), get_data_transform_68055, *[], **kwargs_68056)
        
        # Assigning a type to the variable 'transform' (line 731)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 16), 'transform', get_data_transform_call_result_68057)
        
        # Call to append(...): (line 732)
        # Processing the call arguments (line 732)
        
        # Call to transformed(...): (line 732)
        # Processing the call arguments (line 732)
        # Getting the type of 'transform' (line 732)
        transform_68065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 60), 'transform', False)
        # Processing the call keyword arguments (line 732)
        kwargs_68066 = {}
        
        # Call to get_bbox(...): (line 732)
        # Processing the call keyword arguments (line 732)
        kwargs_68062 = {}
        # Getting the type of 'handle' (line 732)
        handle_68060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 30), 'handle', False)
        # Obtaining the member 'get_bbox' of a type (line 732)
        get_bbox_68061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 30), handle_68060, 'get_bbox')
        # Calling get_bbox(args, kwargs) (line 732)
        get_bbox_call_result_68063 = invoke(stypy.reporting.localization.Localization(__file__, 732, 30), get_bbox_68061, *[], **kwargs_68062)
        
        # Obtaining the member 'transformed' of a type (line 732)
        transformed_68064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 30), get_bbox_call_result_68063, 'transformed')
        # Calling transformed(args, kwargs) (line 732)
        transformed_call_result_68067 = invoke(stypy.reporting.localization.Localization(__file__, 732, 30), transformed_68064, *[transform_68065], **kwargs_68066)
        
        # Processing the call keyword arguments (line 732)
        kwargs_68068 = {}
        # Getting the type of 'bboxes' (line 732)
        bboxes_68058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 16), 'bboxes', False)
        # Obtaining the member 'append' of a type (line 732)
        append_68059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 16), bboxes_68058, 'append')
        # Calling append(args, kwargs) (line 732)
        append_call_result_68069 = invoke(stypy.reporting.localization.Localization(__file__, 732, 16), append_68059, *[transformed_call_result_68067], **kwargs_68068)
        
        # SSA branch for the else part of an if statement (line 730)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 734):
        
        # Assigning a Call to a Name (line 734):
        
        # Assigning a Call to a Name (line 734):
        
        # Call to get_transform(...): (line 734)
        # Processing the call keyword arguments (line 734)
        kwargs_68072 = {}
        # Getting the type of 'handle' (line 734)
        handle_68070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 28), 'handle', False)
        # Obtaining the member 'get_transform' of a type (line 734)
        get_transform_68071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 28), handle_68070, 'get_transform')
        # Calling get_transform(args, kwargs) (line 734)
        get_transform_call_result_68073 = invoke(stypy.reporting.localization.Localization(__file__, 734, 28), get_transform_68071, *[], **kwargs_68072)
        
        # Assigning a type to the variable 'transform' (line 734)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 16), 'transform', get_transform_call_result_68073)
        
        # Call to append(...): (line 735)
        # Processing the call arguments (line 735)
        
        # Call to get_extents(...): (line 735)
        # Processing the call arguments (line 735)
        # Getting the type of 'transform' (line 735)
        transform_68081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 60), 'transform', False)
        # Processing the call keyword arguments (line 735)
        kwargs_68082 = {}
        
        # Call to get_path(...): (line 735)
        # Processing the call keyword arguments (line 735)
        kwargs_68078 = {}
        # Getting the type of 'handle' (line 735)
        handle_68076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 30), 'handle', False)
        # Obtaining the member 'get_path' of a type (line 735)
        get_path_68077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 30), handle_68076, 'get_path')
        # Calling get_path(args, kwargs) (line 735)
        get_path_call_result_68079 = invoke(stypy.reporting.localization.Localization(__file__, 735, 30), get_path_68077, *[], **kwargs_68078)
        
        # Obtaining the member 'get_extents' of a type (line 735)
        get_extents_68080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 30), get_path_call_result_68079, 'get_extents')
        # Calling get_extents(args, kwargs) (line 735)
        get_extents_call_result_68083 = invoke(stypy.reporting.localization.Localization(__file__, 735, 30), get_extents_68080, *[transform_68081], **kwargs_68082)
        
        # Processing the call keyword arguments (line 735)
        kwargs_68084 = {}
        # Getting the type of 'bboxes' (line 735)
        bboxes_68074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 16), 'bboxes', False)
        # Obtaining the member 'append' of a type (line 735)
        append_68075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 16), bboxes_68074, 'append')
        # Calling append(args, kwargs) (line 735)
        append_call_result_68085 = invoke(stypy.reporting.localization.Localization(__file__, 735, 16), append_68075, *[get_extents_call_result_68083], **kwargs_68084)
        
        # SSA join for if statement (line 730)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'ax' (line 737)
        ax_68086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 22), 'ax')
        # Obtaining the member 'collections' of a type (line 737)
        collections_68087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 22), ax_68086, 'collections')
        # Testing the type of a for loop iterable (line 737)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 737, 8), collections_68087)
        # Getting the type of the for loop variable (line 737)
        for_loop_var_68088 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 737, 8), collections_68087)
        # Assigning a type to the variable 'handle' (line 737)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 8), 'handle', for_loop_var_68088)
        # SSA begins for a for statement (line 737)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 738):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to _prepare_points(...): (line 738)
        # Processing the call keyword arguments (line 738)
        kwargs_68091 = {}
        # Getting the type of 'handle' (line 738)
        handle_68089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 54), 'handle', False)
        # Obtaining the member '_prepare_points' of a type (line 738)
        _prepare_points_68090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 54), handle_68089, '_prepare_points')
        # Calling _prepare_points(args, kwargs) (line 738)
        _prepare_points_call_result_68092 = invoke(stypy.reporting.localization.Localization(__file__, 738, 54), _prepare_points_68090, *[], **kwargs_68091)
        
        # Assigning a type to the variable 'call_assignment_66651' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66651', _prepare_points_call_result_68092)
        
        # Assigning a Call to a Name (line 738):
        
        # Assigning a Call to a Name (line 738):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 12), 'int')
        # Processing the call keyword arguments
        kwargs_68096 = {}
        # Getting the type of 'call_assignment_66651' (line 738)
        call_assignment_66651_68093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66651', False)
        # Obtaining the member '__getitem__' of a type (line 738)
        getitem___68094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 12), call_assignment_66651_68093, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68097 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68094, *[int_68095], **kwargs_68096)
        
        # Assigning a type to the variable 'call_assignment_66652' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66652', getitem___call_result_68097)
        
        # Assigning a Name to a Name (line 738):
        
        # Assigning a Name to a Name (line 738):
        # Getting the type of 'call_assignment_66652' (line 738)
        call_assignment_66652_68098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66652')
        # Assigning a type to the variable 'transform' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'transform', call_assignment_66652_68098)
        
        # Assigning a Call to a Name (line 738):
        
        # Assigning a Call to a Name (line 738):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 12), 'int')
        # Processing the call keyword arguments
        kwargs_68102 = {}
        # Getting the type of 'call_assignment_66651' (line 738)
        call_assignment_66651_68099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66651', False)
        # Obtaining the member '__getitem__' of a type (line 738)
        getitem___68100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 12), call_assignment_66651_68099, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68103 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68100, *[int_68101], **kwargs_68102)
        
        # Assigning a type to the variable 'call_assignment_66653' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66653', getitem___call_result_68103)
        
        # Assigning a Name to a Name (line 738):
        
        # Assigning a Name to a Name (line 738):
        # Getting the type of 'call_assignment_66653' (line 738)
        call_assignment_66653_68104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66653')
        # Assigning a type to the variable 'transOffset' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 23), 'transOffset', call_assignment_66653_68104)
        
        # Assigning a Call to a Name (line 738):
        
        # Assigning a Call to a Name (line 738):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 12), 'int')
        # Processing the call keyword arguments
        kwargs_68108 = {}
        # Getting the type of 'call_assignment_66651' (line 738)
        call_assignment_66651_68105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66651', False)
        # Obtaining the member '__getitem__' of a type (line 738)
        getitem___68106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 12), call_assignment_66651_68105, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68109 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68106, *[int_68107], **kwargs_68108)
        
        # Assigning a type to the variable 'call_assignment_66654' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66654', getitem___call_result_68109)
        
        # Assigning a Name to a Name (line 738):
        
        # Assigning a Name to a Name (line 738):
        # Getting the type of 'call_assignment_66654' (line 738)
        call_assignment_66654_68110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66654')
        # Assigning a type to the variable 'hoffsets' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 36), 'hoffsets', call_assignment_66654_68110)
        
        # Assigning a Call to a Name (line 738):
        
        # Assigning a Call to a Name (line 738):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 12), 'int')
        # Processing the call keyword arguments
        kwargs_68114 = {}
        # Getting the type of 'call_assignment_66651' (line 738)
        call_assignment_66651_68111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66651', False)
        # Obtaining the member '__getitem__' of a type (line 738)
        getitem___68112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 12), call_assignment_66651_68111, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68115 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68112, *[int_68113], **kwargs_68114)
        
        # Assigning a type to the variable 'call_assignment_66655' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66655', getitem___call_result_68115)
        
        # Assigning a Name to a Name (line 738):
        
        # Assigning a Name to a Name (line 738):
        # Getting the type of 'call_assignment_66655' (line 738)
        call_assignment_66655_68116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'call_assignment_66655')
        # Assigning a type to the variable 'paths' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 46), 'paths', call_assignment_66655_68116)
        
        
        # Call to len(...): (line 740)
        # Processing the call arguments (line 740)
        # Getting the type of 'hoffsets' (line 740)
        hoffsets_68118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 19), 'hoffsets', False)
        # Processing the call keyword arguments (line 740)
        kwargs_68119 = {}
        # Getting the type of 'len' (line 740)
        len_68117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 15), 'len', False)
        # Calling len(args, kwargs) (line 740)
        len_call_result_68120 = invoke(stypy.reporting.localization.Localization(__file__, 740, 15), len_68117, *[hoffsets_68118], **kwargs_68119)
        
        # Testing the type of an if condition (line 740)
        if_condition_68121 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 740, 12), len_call_result_68120)
        # Assigning a type to the variable 'if_condition_68121' (line 740)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 'if_condition_68121', if_condition_68121)
        # SSA begins for if statement (line 740)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to transform(...): (line 741)
        # Processing the call arguments (line 741)
        # Getting the type of 'hoffsets' (line 741)
        hoffsets_68124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 52), 'hoffsets', False)
        # Processing the call keyword arguments (line 741)
        kwargs_68125 = {}
        # Getting the type of 'transOffset' (line 741)
        transOffset_68122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 30), 'transOffset', False)
        # Obtaining the member 'transform' of a type (line 741)
        transform_68123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 30), transOffset_68122, 'transform')
        # Calling transform(args, kwargs) (line 741)
        transform_call_result_68126 = invoke(stypy.reporting.localization.Localization(__file__, 741, 30), transform_68123, *[hoffsets_68124], **kwargs_68125)
        
        # Testing the type of a for loop iterable (line 741)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 741, 16), transform_call_result_68126)
        # Getting the type of the for loop variable (line 741)
        for_loop_var_68127 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 741, 16), transform_call_result_68126)
        # Assigning a type to the variable 'offset' (line 741)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 16), 'offset', for_loop_var_68127)
        # SSA begins for a for statement (line 741)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 742)
        # Processing the call arguments (line 742)
        # Getting the type of 'offset' (line 742)
        offset_68130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 35), 'offset', False)
        # Processing the call keyword arguments (line 742)
        kwargs_68131 = {}
        # Getting the type of 'offsets' (line 742)
        offsets_68128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 20), 'offsets', False)
        # Obtaining the member 'append' of a type (line 742)
        append_68129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 20), offsets_68128, 'append')
        # Calling append(args, kwargs) (line 742)
        append_call_result_68132 = invoke(stypy.reporting.localization.Localization(__file__, 742, 20), append_68129, *[offset_68130], **kwargs_68131)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 740)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 744)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 745):
        
        # Assigning a Call to a Name (line 745):
        
        # Assigning a Call to a Name (line 745):
        
        # Call to concatenate(...): (line 745)
        # Processing the call arguments (line 745)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'lines' (line 745)
        lines_68137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 59), 'lines', False)
        comprehension_68138 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 39), lines_68137)
        # Assigning a type to the variable 'l' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 39), 'l', comprehension_68138)
        # Getting the type of 'l' (line 745)
        l_68135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 39), 'l', False)
        # Obtaining the member 'vertices' of a type (line 745)
        vertices_68136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 39), l_68135, 'vertices')
        list_68139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 39), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 39), list_68139, vertices_68136)
        # Processing the call keyword arguments (line 745)
        kwargs_68140 = {}
        # Getting the type of 'np' (line 745)
        np_68133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 23), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 745)
        concatenate_68134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 23), np_68133, 'concatenate')
        # Calling concatenate(args, kwargs) (line 745)
        concatenate_call_result_68141 = invoke(stypy.reporting.localization.Localization(__file__, 745, 23), concatenate_68134, *[list_68139], **kwargs_68140)
        
        # Assigning a type to the variable 'vertices' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 12), 'vertices', concatenate_call_result_68141)
        # SSA branch for the except part of a try statement (line 744)
        # SSA branch for the except 'ValueError' branch of a try statement (line 744)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 747):
        
        # Assigning a Call to a Name (line 747):
        
        # Assigning a Call to a Name (line 747):
        
        # Call to array(...): (line 747)
        # Processing the call arguments (line 747)
        
        # Obtaining an instance of the builtin type 'list' (line 747)
        list_68144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 747)
        
        # Processing the call keyword arguments (line 747)
        kwargs_68145 = {}
        # Getting the type of 'np' (line 747)
        np_68142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 747)
        array_68143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 23), np_68142, 'array')
        # Calling array(args, kwargs) (line 747)
        array_call_result_68146 = invoke(stypy.reporting.localization.Localization(__file__, 747, 23), array_68143, *[list_68144], **kwargs_68145)
        
        # Assigning a type to the variable 'vertices' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 12), 'vertices', array_call_result_68146)
        # SSA join for try-except statement (line 744)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'list' (line 749)
        list_68147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 749)
        # Adding element type (line 749)
        # Getting the type of 'vertices' (line 749)
        vertices_68148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 16), 'vertices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 15), list_68147, vertices_68148)
        # Adding element type (line 749)
        # Getting the type of 'bboxes' (line 749)
        bboxes_68149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 26), 'bboxes')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 15), list_68147, bboxes_68149)
        # Adding element type (line 749)
        # Getting the type of 'lines' (line 749)
        lines_68150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 34), 'lines')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 15), list_68147, lines_68150)
        # Adding element type (line 749)
        # Getting the type of 'offsets' (line 749)
        offsets_68151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 41), 'offsets')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 15), list_68147, offsets_68151)
        
        # Assigning a type to the variable 'stypy_return_type' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'stypy_return_type', list_68147)
        
        # ################# End of '_auto_legend_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_auto_legend_data' in the type store
        # Getting the type of 'stypy_return_type' (line 699)
        stypy_return_type_68152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68152)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_auto_legend_data'
        return stypy_return_type_68152


    @norecursion
    def draw_frame(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_frame'
        module_type_store = module_type_store.open_function_context('draw_frame', 751, 4, False)
        # Assigning a type to the variable 'self' (line 752)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.draw_frame.__dict__.__setitem__('stypy_localization', localization)
        Legend.draw_frame.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.draw_frame.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.draw_frame.__dict__.__setitem__('stypy_function_name', 'Legend.draw_frame')
        Legend.draw_frame.__dict__.__setitem__('stypy_param_names_list', ['b'])
        Legend.draw_frame.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.draw_frame.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.draw_frame.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.draw_frame.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.draw_frame.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.draw_frame.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.draw_frame', ['b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_frame', localization, ['b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_frame(...)' code ##################

        unicode_68153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 8), 'unicode', u'b is a boolean.  Set draw frame to b')
        
        # Call to set_frame_on(...): (line 753)
        # Processing the call arguments (line 753)
        # Getting the type of 'b' (line 753)
        b_68156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 26), 'b', False)
        # Processing the call keyword arguments (line 753)
        kwargs_68157 = {}
        # Getting the type of 'self' (line 753)
        self_68154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 8), 'self', False)
        # Obtaining the member 'set_frame_on' of a type (line 753)
        set_frame_on_68155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 8), self_68154, 'set_frame_on')
        # Calling set_frame_on(args, kwargs) (line 753)
        set_frame_on_call_result_68158 = invoke(stypy.reporting.localization.Localization(__file__, 753, 8), set_frame_on_68155, *[b_68156], **kwargs_68157)
        
        
        # ################# End of 'draw_frame(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_frame' in the type store
        # Getting the type of 'stypy_return_type' (line 751)
        stypy_return_type_68159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68159)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_frame'
        return stypy_return_type_68159


    @norecursion
    def get_children(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_children'
        module_type_store = module_type_store.open_function_context('get_children', 755, 4, False)
        # Assigning a type to the variable 'self' (line 756)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.get_children.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_children.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.get_children.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_children.__dict__.__setitem__('stypy_function_name', 'Legend.get_children')
        Legend.get_children.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.get_children.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.get_children.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.get_children.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_children.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_children.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_children.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.get_children', [], None, None, defaults, varargs, kwargs)

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

        unicode_68160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 8), 'unicode', u'return a list of child artists')
        
        # Assigning a List to a Name (line 757):
        
        # Assigning a List to a Name (line 757):
        
        # Assigning a List to a Name (line 757):
        
        # Obtaining an instance of the builtin type 'list' (line 757)
        list_68161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 757)
        
        # Assigning a type to the variable 'children' (line 757)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 8), 'children', list_68161)
        
        # Getting the type of 'self' (line 758)
        self_68162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 11), 'self')
        # Obtaining the member '_legend_box' of a type (line 758)
        _legend_box_68163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 11), self_68162, '_legend_box')
        # Testing the type of an if condition (line 758)
        if_condition_68164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 758, 8), _legend_box_68163)
        # Assigning a type to the variable 'if_condition_68164' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'if_condition_68164', if_condition_68164)
        # SSA begins for if statement (line 758)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 759)
        # Processing the call arguments (line 759)
        # Getting the type of 'self' (line 759)
        self_68167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 28), 'self', False)
        # Obtaining the member '_legend_box' of a type (line 759)
        _legend_box_68168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 28), self_68167, '_legend_box')
        # Processing the call keyword arguments (line 759)
        kwargs_68169 = {}
        # Getting the type of 'children' (line 759)
        children_68165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 12), 'children', False)
        # Obtaining the member 'append' of a type (line 759)
        append_68166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 12), children_68165, 'append')
        # Calling append(args, kwargs) (line 759)
        append_call_result_68170 = invoke(stypy.reporting.localization.Localization(__file__, 759, 12), append_68166, *[_legend_box_68168], **kwargs_68169)
        
        # SSA join for if statement (line 758)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 760)
        # Processing the call arguments (line 760)
        
        # Call to get_frame(...): (line 760)
        # Processing the call keyword arguments (line 760)
        kwargs_68175 = {}
        # Getting the type of 'self' (line 760)
        self_68173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 24), 'self', False)
        # Obtaining the member 'get_frame' of a type (line 760)
        get_frame_68174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 24), self_68173, 'get_frame')
        # Calling get_frame(args, kwargs) (line 760)
        get_frame_call_result_68176 = invoke(stypy.reporting.localization.Localization(__file__, 760, 24), get_frame_68174, *[], **kwargs_68175)
        
        # Processing the call keyword arguments (line 760)
        kwargs_68177 = {}
        # Getting the type of 'children' (line 760)
        children_68171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'children', False)
        # Obtaining the member 'append' of a type (line 760)
        append_68172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 8), children_68171, 'append')
        # Calling append(args, kwargs) (line 760)
        append_call_result_68178 = invoke(stypy.reporting.localization.Localization(__file__, 760, 8), append_68172, *[get_frame_call_result_68176], **kwargs_68177)
        
        # Getting the type of 'children' (line 762)
        children_68179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 15), 'children')
        # Assigning a type to the variable 'stypy_return_type' (line 762)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'stypy_return_type', children_68179)
        
        # ################# End of 'get_children(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_children' in the type store
        # Getting the type of 'stypy_return_type' (line 755)
        stypy_return_type_68180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68180)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_children'
        return stypy_return_type_68180


    @norecursion
    def get_frame(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_frame'
        module_type_store = module_type_store.open_function_context('get_frame', 764, 4, False)
        # Assigning a type to the variable 'self' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.get_frame.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_frame.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.get_frame.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_frame.__dict__.__setitem__('stypy_function_name', 'Legend.get_frame')
        Legend.get_frame.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.get_frame.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.get_frame.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.get_frame.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_frame.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_frame.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_frame.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.get_frame', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_frame', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_frame(...)' code ##################

        unicode_68181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 8), 'unicode', u'return the Rectangle instance used to frame the legend')
        # Getting the type of 'self' (line 766)
        self_68182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 15), 'self')
        # Obtaining the member 'legendPatch' of a type (line 766)
        legendPatch_68183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 15), self_68182, 'legendPatch')
        # Assigning a type to the variable 'stypy_return_type' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'stypy_return_type', legendPatch_68183)
        
        # ################# End of 'get_frame(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_frame' in the type store
        # Getting the type of 'stypy_return_type' (line 764)
        stypy_return_type_68184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68184)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_frame'
        return stypy_return_type_68184


    @norecursion
    def get_lines(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_lines'
        module_type_store = module_type_store.open_function_context('get_lines', 768, 4, False)
        # Assigning a type to the variable 'self' (line 769)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.get_lines.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_lines.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.get_lines.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_lines.__dict__.__setitem__('stypy_function_name', 'Legend.get_lines')
        Legend.get_lines.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.get_lines.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.get_lines.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.get_lines.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_lines.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_lines.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_lines.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.get_lines', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_lines', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_lines(...)' code ##################

        unicode_68185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 8), 'unicode', u'return a list of lines.Line2D instances in the legend')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 770)
        self_68192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 27), 'self')
        # Obtaining the member 'legendHandles' of a type (line 770)
        legendHandles_68193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 27), self_68192, 'legendHandles')
        comprehension_68194 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 770, 16), legendHandles_68193)
        # Assigning a type to the variable 'h' (line 770)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 16), 'h', comprehension_68194)
        
        # Call to isinstance(...): (line 770)
        # Processing the call arguments (line 770)
        # Getting the type of 'h' (line 770)
        h_68188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 60), 'h', False)
        # Getting the type of 'Line2D' (line 770)
        Line2D_68189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 63), 'Line2D', False)
        # Processing the call keyword arguments (line 770)
        kwargs_68190 = {}
        # Getting the type of 'isinstance' (line 770)
        isinstance_68187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 49), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 770)
        isinstance_call_result_68191 = invoke(stypy.reporting.localization.Localization(__file__, 770, 49), isinstance_68187, *[h_68188, Line2D_68189], **kwargs_68190)
        
        # Getting the type of 'h' (line 770)
        h_68186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 16), 'h')
        list_68195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 770, 16), list_68195, h_68186)
        # Assigning a type to the variable 'stypy_return_type' (line 770)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'stypy_return_type', list_68195)
        
        # ################# End of 'get_lines(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_lines' in the type store
        # Getting the type of 'stypy_return_type' (line 768)
        stypy_return_type_68196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_lines'
        return stypy_return_type_68196


    @norecursion
    def get_patches(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_patches'
        module_type_store = module_type_store.open_function_context('get_patches', 772, 4, False)
        # Assigning a type to the variable 'self' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.get_patches.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_patches.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.get_patches.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_patches.__dict__.__setitem__('stypy_function_name', 'Legend.get_patches')
        Legend.get_patches.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.get_patches.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.get_patches.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.get_patches.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_patches.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_patches.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_patches.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.get_patches', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_patches', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_patches(...)' code ##################

        unicode_68197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 8), 'unicode', u'return a list of patch instances in the legend')
        
        # Call to silent_list(...): (line 774)
        # Processing the call arguments (line 774)
        unicode_68199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 27), 'unicode', u'Patch')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 775)
        self_68206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 39), 'self', False)
        # Obtaining the member 'legendHandles' of a type (line 775)
        legendHandles_68207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 39), self_68206, 'legendHandles')
        comprehension_68208 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 28), legendHandles_68207)
        # Assigning a type to the variable 'h' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 28), 'h', comprehension_68208)
        
        # Call to isinstance(...): (line 776)
        # Processing the call arguments (line 776)
        # Getting the type of 'h' (line 776)
        h_68202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 42), 'h', False)
        # Getting the type of 'Patch' (line 776)
        Patch_68203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 45), 'Patch', False)
        # Processing the call keyword arguments (line 776)
        kwargs_68204 = {}
        # Getting the type of 'isinstance' (line 776)
        isinstance_68201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 31), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 776)
        isinstance_call_result_68205 = invoke(stypy.reporting.localization.Localization(__file__, 776, 31), isinstance_68201, *[h_68202, Patch_68203], **kwargs_68204)
        
        # Getting the type of 'h' (line 775)
        h_68200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 28), 'h', False)
        list_68209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 28), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 775, 28), list_68209, h_68200)
        # Processing the call keyword arguments (line 774)
        kwargs_68210 = {}
        # Getting the type of 'silent_list' (line 774)
        silent_list_68198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 15), 'silent_list', False)
        # Calling silent_list(args, kwargs) (line 774)
        silent_list_call_result_68211 = invoke(stypy.reporting.localization.Localization(__file__, 774, 15), silent_list_68198, *[unicode_68199, list_68209], **kwargs_68210)
        
        # Assigning a type to the variable 'stypy_return_type' (line 774)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 8), 'stypy_return_type', silent_list_call_result_68211)
        
        # ################# End of 'get_patches(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_patches' in the type store
        # Getting the type of 'stypy_return_type' (line 772)
        stypy_return_type_68212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68212)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_patches'
        return stypy_return_type_68212


    @norecursion
    def get_texts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_texts'
        module_type_store = module_type_store.open_function_context('get_texts', 778, 4, False)
        # Assigning a type to the variable 'self' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.get_texts.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_texts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.get_texts.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_texts.__dict__.__setitem__('stypy_function_name', 'Legend.get_texts')
        Legend.get_texts.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.get_texts.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.get_texts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.get_texts.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_texts.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_texts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_texts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.get_texts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_texts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_texts(...)' code ##################

        unicode_68213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 8), 'unicode', u'return a list of text.Text instance in the legend')
        
        # Call to silent_list(...): (line 780)
        # Processing the call arguments (line 780)
        unicode_68215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 27), 'unicode', u'Text')
        # Getting the type of 'self' (line 780)
        self_68216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 35), 'self', False)
        # Obtaining the member 'texts' of a type (line 780)
        texts_68217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 35), self_68216, 'texts')
        # Processing the call keyword arguments (line 780)
        kwargs_68218 = {}
        # Getting the type of 'silent_list' (line 780)
        silent_list_68214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 15), 'silent_list', False)
        # Calling silent_list(args, kwargs) (line 780)
        silent_list_call_result_68219 = invoke(stypy.reporting.localization.Localization(__file__, 780, 15), silent_list_68214, *[unicode_68215, texts_68217], **kwargs_68218)
        
        # Assigning a type to the variable 'stypy_return_type' (line 780)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'stypy_return_type', silent_list_call_result_68219)
        
        # ################# End of 'get_texts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_texts' in the type store
        # Getting the type of 'stypy_return_type' (line 778)
        stypy_return_type_68220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68220)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_texts'
        return stypy_return_type_68220


    @norecursion
    def set_title(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 782)
        None_68221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 36), 'None')
        defaults = [None_68221]
        # Create a new context for function 'set_title'
        module_type_store = module_type_store.open_function_context('set_title', 782, 4, False)
        # Assigning a type to the variable 'self' (line 783)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.set_title.__dict__.__setitem__('stypy_localization', localization)
        Legend.set_title.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.set_title.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.set_title.__dict__.__setitem__('stypy_function_name', 'Legend.set_title')
        Legend.set_title.__dict__.__setitem__('stypy_param_names_list', ['title', 'prop'])
        Legend.set_title.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.set_title.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.set_title.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.set_title.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.set_title.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.set_title.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.set_title', ['title', 'prop'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_title', localization, ['title', 'prop'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_title(...)' code ##################

        unicode_68222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, (-1)), 'unicode', u'\n        set the legend title. Fontproperties can be optionally set\n        with *prop* parameter.\n        ')
        
        # Call to set_text(...): (line 787)
        # Processing the call arguments (line 787)
        # Getting the type of 'title' (line 787)
        title_68227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 46), 'title', False)
        # Processing the call keyword arguments (line 787)
        kwargs_68228 = {}
        # Getting the type of 'self' (line 787)
        self_68223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'self', False)
        # Obtaining the member '_legend_title_box' of a type (line 787)
        _legend_title_box_68224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 8), self_68223, '_legend_title_box')
        # Obtaining the member '_text' of a type (line 787)
        _text_68225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 8), _legend_title_box_68224, '_text')
        # Obtaining the member 'set_text' of a type (line 787)
        set_text_68226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 8), _text_68225, 'set_text')
        # Calling set_text(args, kwargs) (line 787)
        set_text_call_result_68229 = invoke(stypy.reporting.localization.Localization(__file__, 787, 8), set_text_68226, *[title_68227], **kwargs_68228)
        
        
        # Type idiom detected: calculating its left and rigth part (line 789)
        # Getting the type of 'prop' (line 789)
        prop_68230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 8), 'prop')
        # Getting the type of 'None' (line 789)
        None_68231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 23), 'None')
        
        (may_be_68232, more_types_in_union_68233) = may_not_be_none(prop_68230, None_68231)

        if may_be_68232:

            if more_types_in_union_68233:
                # Runtime conditional SSA (line 789)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 790)
            # Getting the type of 'dict' (line 790)
            dict_68234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 32), 'dict')
            # Getting the type of 'prop' (line 790)
            prop_68235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 26), 'prop')
            
            (may_be_68236, more_types_in_union_68237) = may_be_subtype(dict_68234, prop_68235)

            if may_be_68236:

                if more_types_in_union_68237:
                    # Runtime conditional SSA (line 790)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'prop' (line 790)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 12), 'prop', remove_not_subtype_from_union(prop_68235, dict))
                
                # Assigning a Call to a Name (line 791):
                
                # Assigning a Call to a Name (line 791):
                
                # Assigning a Call to a Name (line 791):
                
                # Call to FontProperties(...): (line 791)
                # Processing the call keyword arguments (line 791)
                # Getting the type of 'prop' (line 791)
                prop_68239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 40), 'prop', False)
                kwargs_68240 = {'prop_68239': prop_68239}
                # Getting the type of 'FontProperties' (line 791)
                FontProperties_68238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 23), 'FontProperties', False)
                # Calling FontProperties(args, kwargs) (line 791)
                FontProperties_call_result_68241 = invoke(stypy.reporting.localization.Localization(__file__, 791, 23), FontProperties_68238, *[], **kwargs_68240)
                
                # Assigning a type to the variable 'prop' (line 791)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 16), 'prop', FontProperties_call_result_68241)

                if more_types_in_union_68237:
                    # SSA join for if statement (line 790)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to set_fontproperties(...): (line 792)
            # Processing the call arguments (line 792)
            # Getting the type of 'prop' (line 792)
            prop_68246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 60), 'prop', False)
            # Processing the call keyword arguments (line 792)
            kwargs_68247 = {}
            # Getting the type of 'self' (line 792)
            self_68242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 12), 'self', False)
            # Obtaining the member '_legend_title_box' of a type (line 792)
            _legend_title_box_68243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 12), self_68242, '_legend_title_box')
            # Obtaining the member '_text' of a type (line 792)
            _text_68244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 12), _legend_title_box_68243, '_text')
            # Obtaining the member 'set_fontproperties' of a type (line 792)
            set_fontproperties_68245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 12), _text_68244, 'set_fontproperties')
            # Calling set_fontproperties(args, kwargs) (line 792)
            set_fontproperties_call_result_68248 = invoke(stypy.reporting.localization.Localization(__file__, 792, 12), set_fontproperties_68245, *[prop_68246], **kwargs_68247)
            

            if more_types_in_union_68233:
                # SSA join for if statement (line 789)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'title' (line 794)
        title_68249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 11), 'title')
        # Testing the type of an if condition (line 794)
        if_condition_68250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 794, 8), title_68249)
        # Assigning a type to the variable 'if_condition_68250' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'if_condition_68250', if_condition_68250)
        # SSA begins for if statement (line 794)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_visible(...): (line 795)
        # Processing the call arguments (line 795)
        # Getting the type of 'True' (line 795)
        True_68254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 47), 'True', False)
        # Processing the call keyword arguments (line 795)
        kwargs_68255 = {}
        # Getting the type of 'self' (line 795)
        self_68251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 12), 'self', False)
        # Obtaining the member '_legend_title_box' of a type (line 795)
        _legend_title_box_68252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 12), self_68251, '_legend_title_box')
        # Obtaining the member 'set_visible' of a type (line 795)
        set_visible_68253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 12), _legend_title_box_68252, 'set_visible')
        # Calling set_visible(args, kwargs) (line 795)
        set_visible_call_result_68256 = invoke(stypy.reporting.localization.Localization(__file__, 795, 12), set_visible_68253, *[True_68254], **kwargs_68255)
        
        # SSA branch for the else part of an if statement (line 794)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_visible(...): (line 797)
        # Processing the call arguments (line 797)
        # Getting the type of 'False' (line 797)
        False_68260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 47), 'False', False)
        # Processing the call keyword arguments (line 797)
        kwargs_68261 = {}
        # Getting the type of 'self' (line 797)
        self_68257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 12), 'self', False)
        # Obtaining the member '_legend_title_box' of a type (line 797)
        _legend_title_box_68258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 12), self_68257, '_legend_title_box')
        # Obtaining the member 'set_visible' of a type (line 797)
        set_visible_68259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 12), _legend_title_box_68258, 'set_visible')
        # Calling set_visible(args, kwargs) (line 797)
        set_visible_call_result_68262 = invoke(stypy.reporting.localization.Localization(__file__, 797, 12), set_visible_68259, *[False_68260], **kwargs_68261)
        
        # SSA join for if statement (line 794)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 798):
        
        # Assigning a Name to a Attribute (line 798):
        
        # Assigning a Name to a Attribute (line 798):
        # Getting the type of 'True' (line 798)
        True_68263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 21), 'True')
        # Getting the type of 'self' (line 798)
        self_68264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 798)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 8), self_68264, 'stale', True_68263)
        
        # ################# End of 'set_title(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_title' in the type store
        # Getting the type of 'stypy_return_type' (line 782)
        stypy_return_type_68265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68265)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_title'
        return stypy_return_type_68265


    @norecursion
    def get_title(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_title'
        module_type_store = module_type_store.open_function_context('get_title', 800, 4, False)
        # Assigning a type to the variable 'self' (line 801)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.get_title.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_title.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.get_title.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_title.__dict__.__setitem__('stypy_function_name', 'Legend.get_title')
        Legend.get_title.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.get_title.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.get_title.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.get_title.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_title.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_title.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_title.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.get_title', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_title', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_title(...)' code ##################

        unicode_68266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 8), 'unicode', u'return Text instance for the legend title')
        # Getting the type of 'self' (line 802)
        self_68267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 15), 'self')
        # Obtaining the member '_legend_title_box' of a type (line 802)
        _legend_title_box_68268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 15), self_68267, '_legend_title_box')
        # Obtaining the member '_text' of a type (line 802)
        _text_68269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 15), _legend_title_box_68268, '_text')
        # Assigning a type to the variable 'stypy_return_type' (line 802)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 8), 'stypy_return_type', _text_68269)
        
        # ################# End of 'get_title(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_title' in the type store
        # Getting the type of 'stypy_return_type' (line 800)
        stypy_return_type_68270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68270)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_title'
        return stypy_return_type_68270


    @norecursion
    def get_window_extent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_window_extent'
        module_type_store = module_type_store.open_function_context('get_window_extent', 804, 4, False)
        # Assigning a type to the variable 'self' (line 805)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.get_window_extent.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_window_extent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.get_window_extent.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_window_extent.__dict__.__setitem__('stypy_function_name', 'Legend.get_window_extent')
        Legend.get_window_extent.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.get_window_extent.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Legend.get_window_extent.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Legend.get_window_extent.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_window_extent.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_window_extent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_window_extent.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.get_window_extent', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_window_extent', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_window_extent(...)' code ##################

        unicode_68271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 8), 'unicode', u'return a extent of the legend')
        
        # Call to get_window_extent(...): (line 806)
        # Getting the type of 'args' (line 806)
        args_68275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 51), 'args', False)
        # Processing the call keyword arguments (line 806)
        # Getting the type of 'kwargs' (line 806)
        kwargs_68276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 59), 'kwargs', False)
        kwargs_68277 = {'kwargs_68276': kwargs_68276}
        # Getting the type of 'self' (line 806)
        self_68272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 15), 'self', False)
        # Obtaining the member 'legendPatch' of a type (line 806)
        legendPatch_68273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 15), self_68272, 'legendPatch')
        # Obtaining the member 'get_window_extent' of a type (line 806)
        get_window_extent_68274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 15), legendPatch_68273, 'get_window_extent')
        # Calling get_window_extent(args, kwargs) (line 806)
        get_window_extent_call_result_68278 = invoke(stypy.reporting.localization.Localization(__file__, 806, 15), get_window_extent_68274, *[args_68275], **kwargs_68277)
        
        # Assigning a type to the variable 'stypy_return_type' (line 806)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'stypy_return_type', get_window_extent_call_result_68278)
        
        # ################# End of 'get_window_extent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_window_extent' in the type store
        # Getting the type of 'stypy_return_type' (line 804)
        stypy_return_type_68279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68279)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_window_extent'
        return stypy_return_type_68279


    @norecursion
    def get_frame_on(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_frame_on'
        module_type_store = module_type_store.open_function_context('get_frame_on', 808, 4, False)
        # Assigning a type to the variable 'self' (line 809)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.get_frame_on.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_frame_on.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.get_frame_on.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_frame_on.__dict__.__setitem__('stypy_function_name', 'Legend.get_frame_on')
        Legend.get_frame_on.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.get_frame_on.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.get_frame_on.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.get_frame_on.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_frame_on.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_frame_on.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_frame_on.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.get_frame_on', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_frame_on', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_frame_on(...)' code ##################

        unicode_68280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, (-1)), 'unicode', u'\n        Get whether the legend box patch is drawn\n        ')
        # Getting the type of 'self' (line 812)
        self_68281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 15), 'self')
        # Obtaining the member '_drawFrame' of a type (line 812)
        _drawFrame_68282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 15), self_68281, '_drawFrame')
        # Assigning a type to the variable 'stypy_return_type' (line 812)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'stypy_return_type', _drawFrame_68282)
        
        # ################# End of 'get_frame_on(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_frame_on' in the type store
        # Getting the type of 'stypy_return_type' (line 808)
        stypy_return_type_68283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68283)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_frame_on'
        return stypy_return_type_68283


    @norecursion
    def set_frame_on(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_frame_on'
        module_type_store = module_type_store.open_function_context('set_frame_on', 814, 4, False)
        # Assigning a type to the variable 'self' (line 815)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.set_frame_on.__dict__.__setitem__('stypy_localization', localization)
        Legend.set_frame_on.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.set_frame_on.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.set_frame_on.__dict__.__setitem__('stypy_function_name', 'Legend.set_frame_on')
        Legend.set_frame_on.__dict__.__setitem__('stypy_param_names_list', ['b'])
        Legend.set_frame_on.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.set_frame_on.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.set_frame_on.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.set_frame_on.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.set_frame_on.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.set_frame_on.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.set_frame_on', ['b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_frame_on', localization, ['b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_frame_on(...)' code ##################

        unicode_68284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, (-1)), 'unicode', u'\n        Set whether the legend box patch is drawn\n\n        ACCEPTS: [ *True* | *False* ]\n        ')
        
        # Assigning a Name to a Attribute (line 820):
        
        # Assigning a Name to a Attribute (line 820):
        
        # Assigning a Name to a Attribute (line 820):
        # Getting the type of 'b' (line 820)
        b_68285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 26), 'b')
        # Getting the type of 'self' (line 820)
        self_68286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 8), 'self')
        # Setting the type of the member '_drawFrame' of a type (line 820)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 8), self_68286, '_drawFrame', b_68285)
        
        # Assigning a Name to a Attribute (line 821):
        
        # Assigning a Name to a Attribute (line 821):
        
        # Assigning a Name to a Attribute (line 821):
        # Getting the type of 'True' (line 821)
        True_68287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 21), 'True')
        # Getting the type of 'self' (line 821)
        self_68288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 821)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 8), self_68288, 'stale', True_68287)
        
        # ################# End of 'set_frame_on(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_frame_on' in the type store
        # Getting the type of 'stypy_return_type' (line 814)
        stypy_return_type_68289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68289)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_frame_on'
        return stypy_return_type_68289


    @norecursion
    def get_bbox_to_anchor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_bbox_to_anchor'
        module_type_store = module_type_store.open_function_context('get_bbox_to_anchor', 823, 4, False)
        # Assigning a type to the variable 'self' (line 824)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.get_bbox_to_anchor.__dict__.__setitem__('stypy_localization', localization)
        Legend.get_bbox_to_anchor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.get_bbox_to_anchor.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.get_bbox_to_anchor.__dict__.__setitem__('stypy_function_name', 'Legend.get_bbox_to_anchor')
        Legend.get_bbox_to_anchor.__dict__.__setitem__('stypy_param_names_list', [])
        Legend.get_bbox_to_anchor.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.get_bbox_to_anchor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.get_bbox_to_anchor.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.get_bbox_to_anchor.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.get_bbox_to_anchor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.get_bbox_to_anchor.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.get_bbox_to_anchor', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_bbox_to_anchor', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_bbox_to_anchor(...)' code ##################

        unicode_68290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, (-1)), 'unicode', u'\n        return the bbox that the legend will be anchored\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 827)
        # Getting the type of 'self' (line 827)
        self_68291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 11), 'self')
        # Obtaining the member '_bbox_to_anchor' of a type (line 827)
        _bbox_to_anchor_68292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 11), self_68291, '_bbox_to_anchor')
        # Getting the type of 'None' (line 827)
        None_68293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 35), 'None')
        
        (may_be_68294, more_types_in_union_68295) = may_be_none(_bbox_to_anchor_68292, None_68293)

        if may_be_68294:

            if more_types_in_union_68295:
                # Runtime conditional SSA (line 827)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 828)
            self_68296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 19), 'self')
            # Obtaining the member 'parent' of a type (line 828)
            parent_68297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 19), self_68296, 'parent')
            # Obtaining the member 'bbox' of a type (line 828)
            bbox_68298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 19), parent_68297, 'bbox')
            # Assigning a type to the variable 'stypy_return_type' (line 828)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 12), 'stypy_return_type', bbox_68298)

            if more_types_in_union_68295:
                # Runtime conditional SSA for else branch (line 827)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_68294) or more_types_in_union_68295):
            # Getting the type of 'self' (line 830)
            self_68299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 19), 'self')
            # Obtaining the member '_bbox_to_anchor' of a type (line 830)
            _bbox_to_anchor_68300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 19), self_68299, '_bbox_to_anchor')
            # Assigning a type to the variable 'stypy_return_type' (line 830)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 12), 'stypy_return_type', _bbox_to_anchor_68300)

            if (may_be_68294 and more_types_in_union_68295):
                # SSA join for if statement (line 827)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'get_bbox_to_anchor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_bbox_to_anchor' in the type store
        # Getting the type of 'stypy_return_type' (line 823)
        stypy_return_type_68301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68301)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_bbox_to_anchor'
        return stypy_return_type_68301


    @norecursion
    def set_bbox_to_anchor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 832)
        None_68302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 49), 'None')
        defaults = [None_68302]
        # Create a new context for function 'set_bbox_to_anchor'
        module_type_store = module_type_store.open_function_context('set_bbox_to_anchor', 832, 4, False)
        # Assigning a type to the variable 'self' (line 833)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.set_bbox_to_anchor.__dict__.__setitem__('stypy_localization', localization)
        Legend.set_bbox_to_anchor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.set_bbox_to_anchor.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.set_bbox_to_anchor.__dict__.__setitem__('stypy_function_name', 'Legend.set_bbox_to_anchor')
        Legend.set_bbox_to_anchor.__dict__.__setitem__('stypy_param_names_list', ['bbox', 'transform'])
        Legend.set_bbox_to_anchor.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.set_bbox_to_anchor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.set_bbox_to_anchor.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.set_bbox_to_anchor.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.set_bbox_to_anchor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.set_bbox_to_anchor.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.set_bbox_to_anchor', ['bbox', 'transform'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_bbox_to_anchor', localization, ['bbox', 'transform'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_bbox_to_anchor(...)' code ##################

        unicode_68303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, (-1)), 'unicode', u'\n        set the bbox that the legend will be anchored.\n\n        *bbox* can be a BboxBase instance, a tuple of [left, bottom,\n        width, height] in the given transform (normalized axes\n        coordinate if None), or a tuple of [left, bottom] where the\n        width and height will be assumed to be zero.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 841)
        # Getting the type of 'bbox' (line 841)
        bbox_68304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 11), 'bbox')
        # Getting the type of 'None' (line 841)
        None_68305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 19), 'None')
        
        (may_be_68306, more_types_in_union_68307) = may_be_none(bbox_68304, None_68305)

        if may_be_68306:

            if more_types_in_union_68307:
                # Runtime conditional SSA (line 841)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 842):
            
            # Assigning a Name to a Attribute (line 842):
            
            # Assigning a Name to a Attribute (line 842):
            # Getting the type of 'None' (line 842)
            None_68308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 35), 'None')
            # Getting the type of 'self' (line 842)
            self_68309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 12), 'self')
            # Setting the type of the member '_bbox_to_anchor' of a type (line 842)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 12), self_68309, '_bbox_to_anchor', None_68308)
            # Assigning a type to the variable 'stypy_return_type' (line 843)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_68307:
                # Runtime conditional SSA for else branch (line 841)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_68306) or more_types_in_union_68307):
            
            
            # Call to isinstance(...): (line 844)
            # Processing the call arguments (line 844)
            # Getting the type of 'bbox' (line 844)
            bbox_68311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 24), 'bbox', False)
            # Getting the type of 'BboxBase' (line 844)
            BboxBase_68312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 30), 'BboxBase', False)
            # Processing the call keyword arguments (line 844)
            kwargs_68313 = {}
            # Getting the type of 'isinstance' (line 844)
            isinstance_68310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 844)
            isinstance_call_result_68314 = invoke(stypy.reporting.localization.Localization(__file__, 844, 13), isinstance_68310, *[bbox_68311, BboxBase_68312], **kwargs_68313)
            
            # Testing the type of an if condition (line 844)
            if_condition_68315 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 844, 13), isinstance_call_result_68314)
            # Assigning a type to the variable 'if_condition_68315' (line 844)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 13), 'if_condition_68315', if_condition_68315)
            # SSA begins for if statement (line 844)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 845):
            
            # Assigning a Name to a Attribute (line 845):
            
            # Assigning a Name to a Attribute (line 845):
            # Getting the type of 'bbox' (line 845)
            bbox_68316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 35), 'bbox')
            # Getting the type of 'self' (line 845)
            self_68317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 12), 'self')
            # Setting the type of the member '_bbox_to_anchor' of a type (line 845)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 12), self_68317, '_bbox_to_anchor', bbox_68316)
            # SSA branch for the else part of an if statement (line 844)
            module_type_store.open_ssa_branch('else')
            
            
            # SSA begins for try-except statement (line 847)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 848):
            
            # Assigning a Call to a Name (line 848):
            
            # Assigning a Call to a Name (line 848):
            
            # Call to len(...): (line 848)
            # Processing the call arguments (line 848)
            # Getting the type of 'bbox' (line 848)
            bbox_68319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 24), 'bbox', False)
            # Processing the call keyword arguments (line 848)
            kwargs_68320 = {}
            # Getting the type of 'len' (line 848)
            len_68318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 20), 'len', False)
            # Calling len(args, kwargs) (line 848)
            len_call_result_68321 = invoke(stypy.reporting.localization.Localization(__file__, 848, 20), len_68318, *[bbox_68319], **kwargs_68320)
            
            # Assigning a type to the variable 'l' (line 848)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 16), 'l', len_call_result_68321)
            # SSA branch for the except part of a try statement (line 847)
            # SSA branch for the except 'TypeError' branch of a try statement (line 847)
            module_type_store.open_ssa_branch('except')
            
            # Call to ValueError(...): (line 850)
            # Processing the call arguments (line 850)
            unicode_68323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 33), 'unicode', u'Invalid argument for bbox : %s')
            
            # Call to str(...): (line 850)
            # Processing the call arguments (line 850)
            # Getting the type of 'bbox' (line 850)
            bbox_68325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 72), 'bbox', False)
            # Processing the call keyword arguments (line 850)
            kwargs_68326 = {}
            # Getting the type of 'str' (line 850)
            str_68324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 68), 'str', False)
            # Calling str(args, kwargs) (line 850)
            str_call_result_68327 = invoke(stypy.reporting.localization.Localization(__file__, 850, 68), str_68324, *[bbox_68325], **kwargs_68326)
            
            # Applying the binary operator '%' (line 850)
            result_mod_68328 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 33), '%', unicode_68323, str_call_result_68327)
            
            # Processing the call keyword arguments (line 850)
            kwargs_68329 = {}
            # Getting the type of 'ValueError' (line 850)
            ValueError_68322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 850)
            ValueError_call_result_68330 = invoke(stypy.reporting.localization.Localization(__file__, 850, 22), ValueError_68322, *[result_mod_68328], **kwargs_68329)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 850, 16), ValueError_call_result_68330, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 847)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'l' (line 852)
            l_68331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 15), 'l')
            int_68332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 20), 'int')
            # Applying the binary operator '==' (line 852)
            result_eq_68333 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 15), '==', l_68331, int_68332)
            
            # Testing the type of an if condition (line 852)
            if_condition_68334 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 852, 12), result_eq_68333)
            # Assigning a type to the variable 'if_condition_68334' (line 852)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 12), 'if_condition_68334', if_condition_68334)
            # SSA begins for if statement (line 852)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a List to a Name (line 853):
            
            # Assigning a List to a Name (line 853):
            
            # Assigning a List to a Name (line 853):
            
            # Obtaining an instance of the builtin type 'list' (line 853)
            list_68335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 853)
            # Adding element type (line 853)
            
            # Obtaining the type of the subscript
            int_68336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 29), 'int')
            # Getting the type of 'bbox' (line 853)
            bbox_68337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 24), 'bbox')
            # Obtaining the member '__getitem__' of a type (line 853)
            getitem___68338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 24), bbox_68337, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 853)
            subscript_call_result_68339 = invoke(stypy.reporting.localization.Localization(__file__, 853, 24), getitem___68338, int_68336)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 853, 23), list_68335, subscript_call_result_68339)
            # Adding element type (line 853)
            
            # Obtaining the type of the subscript
            int_68340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 38), 'int')
            # Getting the type of 'bbox' (line 853)
            bbox_68341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 33), 'bbox')
            # Obtaining the member '__getitem__' of a type (line 853)
            getitem___68342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 33), bbox_68341, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 853)
            subscript_call_result_68343 = invoke(stypy.reporting.localization.Localization(__file__, 853, 33), getitem___68342, int_68340)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 853, 23), list_68335, subscript_call_result_68343)
            # Adding element type (line 853)
            int_68344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 42), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 853, 23), list_68335, int_68344)
            # Adding element type (line 853)
            int_68345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 45), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 853, 23), list_68335, int_68345)
            
            # Assigning a type to the variable 'bbox' (line 853)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 16), 'bbox', list_68335)
            # SSA join for if statement (line 852)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Attribute (line 855):
            
            # Assigning a Call to a Attribute (line 855):
            
            # Assigning a Call to a Attribute (line 855):
            
            # Call to from_bounds(...): (line 855)
            # Getting the type of 'bbox' (line 855)
            bbox_68348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 53), 'bbox', False)
            # Processing the call keyword arguments (line 855)
            kwargs_68349 = {}
            # Getting the type of 'Bbox' (line 855)
            Bbox_68346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 35), 'Bbox', False)
            # Obtaining the member 'from_bounds' of a type (line 855)
            from_bounds_68347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 35), Bbox_68346, 'from_bounds')
            # Calling from_bounds(args, kwargs) (line 855)
            from_bounds_call_result_68350 = invoke(stypy.reporting.localization.Localization(__file__, 855, 35), from_bounds_68347, *[bbox_68348], **kwargs_68349)
            
            # Getting the type of 'self' (line 855)
            self_68351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 12), 'self')
            # Setting the type of the member '_bbox_to_anchor' of a type (line 855)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 12), self_68351, '_bbox_to_anchor', from_bounds_call_result_68350)
            # SSA join for if statement (line 844)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_68306 and more_types_in_union_68307):
                # SSA join for if statement (line 841)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 857)
        # Getting the type of 'transform' (line 857)
        transform_68352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 11), 'transform')
        # Getting the type of 'None' (line 857)
        None_68353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 24), 'None')
        
        (may_be_68354, more_types_in_union_68355) = may_be_none(transform_68352, None_68353)

        if may_be_68354:

            if more_types_in_union_68355:
                # Runtime conditional SSA (line 857)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 858):
            
            # Assigning a Call to a Name (line 858):
            
            # Assigning a Call to a Name (line 858):
            
            # Call to BboxTransformTo(...): (line 858)
            # Processing the call arguments (line 858)
            # Getting the type of 'self' (line 858)
            self_68357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 40), 'self', False)
            # Obtaining the member 'parent' of a type (line 858)
            parent_68358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 858, 40), self_68357, 'parent')
            # Obtaining the member 'bbox' of a type (line 858)
            bbox_68359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 858, 40), parent_68358, 'bbox')
            # Processing the call keyword arguments (line 858)
            kwargs_68360 = {}
            # Getting the type of 'BboxTransformTo' (line 858)
            BboxTransformTo_68356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 24), 'BboxTransformTo', False)
            # Calling BboxTransformTo(args, kwargs) (line 858)
            BboxTransformTo_call_result_68361 = invoke(stypy.reporting.localization.Localization(__file__, 858, 24), BboxTransformTo_68356, *[bbox_68359], **kwargs_68360)
            
            # Assigning a type to the variable 'transform' (line 858)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 858, 12), 'transform', BboxTransformTo_call_result_68361)

            if more_types_in_union_68355:
                # SSA join for if statement (line 857)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 860):
        
        # Assigning a Call to a Attribute (line 860):
        
        # Assigning a Call to a Attribute (line 860):
        
        # Call to TransformedBbox(...): (line 860)
        # Processing the call arguments (line 860)
        # Getting the type of 'self' (line 860)
        self_68363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 47), 'self', False)
        # Obtaining the member '_bbox_to_anchor' of a type (line 860)
        _bbox_to_anchor_68364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 47), self_68363, '_bbox_to_anchor')
        # Getting the type of 'transform' (line 861)
        transform_68365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 47), 'transform', False)
        # Processing the call keyword arguments (line 860)
        kwargs_68366 = {}
        # Getting the type of 'TransformedBbox' (line 860)
        TransformedBbox_68362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 31), 'TransformedBbox', False)
        # Calling TransformedBbox(args, kwargs) (line 860)
        TransformedBbox_call_result_68367 = invoke(stypy.reporting.localization.Localization(__file__, 860, 31), TransformedBbox_68362, *[_bbox_to_anchor_68364, transform_68365], **kwargs_68366)
        
        # Getting the type of 'self' (line 860)
        self_68368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 8), 'self')
        # Setting the type of the member '_bbox_to_anchor' of a type (line 860)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 8), self_68368, '_bbox_to_anchor', TransformedBbox_call_result_68367)
        
        # Assigning a Name to a Attribute (line 862):
        
        # Assigning a Name to a Attribute (line 862):
        
        # Assigning a Name to a Attribute (line 862):
        # Getting the type of 'True' (line 862)
        True_68369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 21), 'True')
        # Getting the type of 'self' (line 862)
        self_68370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 862)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 8), self_68370, 'stale', True_68369)
        
        # ################# End of 'set_bbox_to_anchor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_bbox_to_anchor' in the type store
        # Getting the type of 'stypy_return_type' (line 832)
        stypy_return_type_68371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68371)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_bbox_to_anchor'
        return stypy_return_type_68371


    @norecursion
    def _get_anchored_bbox(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_anchored_bbox'
        module_type_store = module_type_store.open_function_context('_get_anchored_bbox', 864, 4, False)
        # Assigning a type to the variable 'self' (line 865)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend._get_anchored_bbox.__dict__.__setitem__('stypy_localization', localization)
        Legend._get_anchored_bbox.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend._get_anchored_bbox.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend._get_anchored_bbox.__dict__.__setitem__('stypy_function_name', 'Legend._get_anchored_bbox')
        Legend._get_anchored_bbox.__dict__.__setitem__('stypy_param_names_list', ['loc', 'bbox', 'parentbbox', 'renderer'])
        Legend._get_anchored_bbox.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend._get_anchored_bbox.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend._get_anchored_bbox.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend._get_anchored_bbox.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend._get_anchored_bbox.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend._get_anchored_bbox.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend._get_anchored_bbox', ['loc', 'bbox', 'parentbbox', 'renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_anchored_bbox', localization, ['loc', 'bbox', 'parentbbox', 'renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_anchored_bbox(...)' code ##################

        unicode_68372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, (-1)), 'unicode', u'\n        Place the *bbox* inside the *parentbbox* according to a given\n        location code. Return the (x,y) coordinate of the bbox.\n\n        - loc: a location code in range(1, 11).\n          This corresponds to the possible values for self._loc, excluding\n          "best".\n\n        - bbox: bbox to be placed, display coodinate units.\n        - parentbbox: a parent box which will contain the bbox. In\n            display coordinates.\n        ')
        # Evaluating assert statement condition
        
        # Getting the type of 'loc' (line 877)
        loc_68373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 15), 'loc')
        
        # Call to range(...): (line 877)
        # Processing the call arguments (line 877)
        int_68375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 28), 'int')
        int_68376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 31), 'int')
        # Processing the call keyword arguments (line 877)
        kwargs_68377 = {}
        # Getting the type of 'range' (line 877)
        range_68374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 22), 'range', False)
        # Calling range(args, kwargs) (line 877)
        range_call_result_68378 = invoke(stypy.reporting.localization.Localization(__file__, 877, 22), range_68374, *[int_68375, int_68376], **kwargs_68377)
        
        # Applying the binary operator 'in' (line 877)
        result_contains_68379 = python_operator(stypy.reporting.localization.Localization(__file__, 877, 15), 'in', loc_68373, range_call_result_68378)
        
        
        # Assigning a Call to a Tuple (line 879):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to list(...): (line 879)
        # Processing the call arguments (line 879)
        
        # Call to xrange(...): (line 879)
        # Processing the call arguments (line 879)
        int_68382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 65), 'int')
        # Processing the call keyword arguments (line 879)
        kwargs_68383 = {}
        # Getting the type of 'xrange' (line 879)
        xrange_68381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 58), 'xrange', False)
        # Calling xrange(args, kwargs) (line 879)
        xrange_call_result_68384 = invoke(stypy.reporting.localization.Localization(__file__, 879, 58), xrange_68381, *[int_68382], **kwargs_68383)
        
        # Processing the call keyword arguments (line 879)
        kwargs_68385 = {}
        # Getting the type of 'list' (line 879)
        list_68380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 53), 'list', False)
        # Calling list(args, kwargs) (line 879)
        list_call_result_68386 = invoke(stypy.reporting.localization.Localization(__file__, 879, 53), list_68380, *[xrange_call_result_68384], **kwargs_68385)
        
        # Assigning a type to the variable 'call_assignment_66656' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', list_call_result_68386)
        
        # Assigning a Call to a Name (line 879):
        
        # Assigning a Call to a Name (line 879):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68390 = {}
        # Getting the type of 'call_assignment_66656' (line 879)
        call_assignment_66656_68387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', False)
        # Obtaining the member '__getitem__' of a type (line 879)
        getitem___68388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), call_assignment_66656_68387, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68391 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68388, *[int_68389], **kwargs_68390)
        
        # Assigning a type to the variable 'call_assignment_66657' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66657', getitem___call_result_68391)
        
        # Assigning a Name to a Name (line 879):
        
        # Assigning a Name to a Name (line 879):
        # Getting the type of 'call_assignment_66657' (line 879)
        call_assignment_66657_68392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66657')
        # Assigning a type to the variable 'BEST' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'BEST', call_assignment_66657_68392)
        
        # Assigning a Call to a Name (line 879):
        
        # Assigning a Call to a Name (line 879):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68396 = {}
        # Getting the type of 'call_assignment_66656' (line 879)
        call_assignment_66656_68393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', False)
        # Obtaining the member '__getitem__' of a type (line 879)
        getitem___68394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), call_assignment_66656_68393, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68397 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68394, *[int_68395], **kwargs_68396)
        
        # Assigning a type to the variable 'call_assignment_66658' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66658', getitem___call_result_68397)
        
        # Assigning a Name to a Name (line 879):
        
        # Assigning a Name to a Name (line 879):
        # Getting the type of 'call_assignment_66658' (line 879)
        call_assignment_66658_68398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66658')
        # Assigning a type to the variable 'UR' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 14), 'UR', call_assignment_66658_68398)
        
        # Assigning a Call to a Name (line 879):
        
        # Assigning a Call to a Name (line 879):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68402 = {}
        # Getting the type of 'call_assignment_66656' (line 879)
        call_assignment_66656_68399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', False)
        # Obtaining the member '__getitem__' of a type (line 879)
        getitem___68400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), call_assignment_66656_68399, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68403 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68400, *[int_68401], **kwargs_68402)
        
        # Assigning a type to the variable 'call_assignment_66659' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66659', getitem___call_result_68403)
        
        # Assigning a Name to a Name (line 879):
        
        # Assigning a Name to a Name (line 879):
        # Getting the type of 'call_assignment_66659' (line 879)
        call_assignment_66659_68404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66659')
        # Assigning a type to the variable 'UL' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 18), 'UL', call_assignment_66659_68404)
        
        # Assigning a Call to a Name (line 879):
        
        # Assigning a Call to a Name (line 879):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68408 = {}
        # Getting the type of 'call_assignment_66656' (line 879)
        call_assignment_66656_68405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', False)
        # Obtaining the member '__getitem__' of a type (line 879)
        getitem___68406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), call_assignment_66656_68405, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68409 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68406, *[int_68407], **kwargs_68408)
        
        # Assigning a type to the variable 'call_assignment_66660' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66660', getitem___call_result_68409)
        
        # Assigning a Name to a Name (line 879):
        
        # Assigning a Name to a Name (line 879):
        # Getting the type of 'call_assignment_66660' (line 879)
        call_assignment_66660_68410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66660')
        # Assigning a type to the variable 'LL' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 22), 'LL', call_assignment_66660_68410)
        
        # Assigning a Call to a Name (line 879):
        
        # Assigning a Call to a Name (line 879):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68414 = {}
        # Getting the type of 'call_assignment_66656' (line 879)
        call_assignment_66656_68411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', False)
        # Obtaining the member '__getitem__' of a type (line 879)
        getitem___68412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), call_assignment_66656_68411, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68415 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68412, *[int_68413], **kwargs_68414)
        
        # Assigning a type to the variable 'call_assignment_66661' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66661', getitem___call_result_68415)
        
        # Assigning a Name to a Name (line 879):
        
        # Assigning a Name to a Name (line 879):
        # Getting the type of 'call_assignment_66661' (line 879)
        call_assignment_66661_68416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66661')
        # Assigning a type to the variable 'LR' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 26), 'LR', call_assignment_66661_68416)
        
        # Assigning a Call to a Name (line 879):
        
        # Assigning a Call to a Name (line 879):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68420 = {}
        # Getting the type of 'call_assignment_66656' (line 879)
        call_assignment_66656_68417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', False)
        # Obtaining the member '__getitem__' of a type (line 879)
        getitem___68418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), call_assignment_66656_68417, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68421 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68418, *[int_68419], **kwargs_68420)
        
        # Assigning a type to the variable 'call_assignment_66662' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66662', getitem___call_result_68421)
        
        # Assigning a Name to a Name (line 879):
        
        # Assigning a Name to a Name (line 879):
        # Getting the type of 'call_assignment_66662' (line 879)
        call_assignment_66662_68422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66662')
        # Assigning a type to the variable 'R' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 30), 'R', call_assignment_66662_68422)
        
        # Assigning a Call to a Name (line 879):
        
        # Assigning a Call to a Name (line 879):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68426 = {}
        # Getting the type of 'call_assignment_66656' (line 879)
        call_assignment_66656_68423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', False)
        # Obtaining the member '__getitem__' of a type (line 879)
        getitem___68424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), call_assignment_66656_68423, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68427 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68424, *[int_68425], **kwargs_68426)
        
        # Assigning a type to the variable 'call_assignment_66663' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66663', getitem___call_result_68427)
        
        # Assigning a Name to a Name (line 879):
        
        # Assigning a Name to a Name (line 879):
        # Getting the type of 'call_assignment_66663' (line 879)
        call_assignment_66663_68428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66663')
        # Assigning a type to the variable 'CL' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 33), 'CL', call_assignment_66663_68428)
        
        # Assigning a Call to a Name (line 879):
        
        # Assigning a Call to a Name (line 879):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68432 = {}
        # Getting the type of 'call_assignment_66656' (line 879)
        call_assignment_66656_68429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', False)
        # Obtaining the member '__getitem__' of a type (line 879)
        getitem___68430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), call_assignment_66656_68429, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68433 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68430, *[int_68431], **kwargs_68432)
        
        # Assigning a type to the variable 'call_assignment_66664' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66664', getitem___call_result_68433)
        
        # Assigning a Name to a Name (line 879):
        
        # Assigning a Name to a Name (line 879):
        # Getting the type of 'call_assignment_66664' (line 879)
        call_assignment_66664_68434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66664')
        # Assigning a type to the variable 'CR' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 37), 'CR', call_assignment_66664_68434)
        
        # Assigning a Call to a Name (line 879):
        
        # Assigning a Call to a Name (line 879):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68438 = {}
        # Getting the type of 'call_assignment_66656' (line 879)
        call_assignment_66656_68435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', False)
        # Obtaining the member '__getitem__' of a type (line 879)
        getitem___68436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), call_assignment_66656_68435, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68439 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68436, *[int_68437], **kwargs_68438)
        
        # Assigning a type to the variable 'call_assignment_66665' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66665', getitem___call_result_68439)
        
        # Assigning a Name to a Name (line 879):
        
        # Assigning a Name to a Name (line 879):
        # Getting the type of 'call_assignment_66665' (line 879)
        call_assignment_66665_68440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66665')
        # Assigning a type to the variable 'LC' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 41), 'LC', call_assignment_66665_68440)
        
        # Assigning a Call to a Name (line 879):
        
        # Assigning a Call to a Name (line 879):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68444 = {}
        # Getting the type of 'call_assignment_66656' (line 879)
        call_assignment_66656_68441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', False)
        # Obtaining the member '__getitem__' of a type (line 879)
        getitem___68442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), call_assignment_66656_68441, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68445 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68442, *[int_68443], **kwargs_68444)
        
        # Assigning a type to the variable 'call_assignment_66666' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66666', getitem___call_result_68445)
        
        # Assigning a Name to a Name (line 879):
        
        # Assigning a Name to a Name (line 879):
        # Getting the type of 'call_assignment_66666' (line 879)
        call_assignment_66666_68446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66666')
        # Assigning a type to the variable 'UC' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 45), 'UC', call_assignment_66666_68446)
        
        # Assigning a Call to a Name (line 879):
        
        # Assigning a Call to a Name (line 879):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68450 = {}
        # Getting the type of 'call_assignment_66656' (line 879)
        call_assignment_66656_68447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66656', False)
        # Obtaining the member '__getitem__' of a type (line 879)
        getitem___68448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), call_assignment_66656_68447, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68451 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68448, *[int_68449], **kwargs_68450)
        
        # Assigning a type to the variable 'call_assignment_66667' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66667', getitem___call_result_68451)
        
        # Assigning a Name to a Name (line 879):
        
        # Assigning a Name to a Name (line 879):
        # Getting the type of 'call_assignment_66667' (line 879)
        call_assignment_66667_68452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'call_assignment_66667')
        # Assigning a type to the variable 'C' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 49), 'C', call_assignment_66667_68452)
        
        # Assigning a Dict to a Name (line 881):
        
        # Assigning a Dict to a Name (line 881):
        
        # Assigning a Dict to a Name (line 881):
        
        # Obtaining an instance of the builtin type 'dict' (line 881)
        dict_68453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 881)
        # Adding element type (key, value) (line 881)
        # Getting the type of 'UR' (line 881)
        UR_68454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 24), 'UR')
        unicode_68455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 28), 'unicode', u'NE')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 23), dict_68453, (UR_68454, unicode_68455))
        # Adding element type (key, value) (line 881)
        # Getting the type of 'UL' (line 882)
        UL_68456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 24), 'UL')
        unicode_68457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 28), 'unicode', u'NW')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 23), dict_68453, (UL_68456, unicode_68457))
        # Adding element type (key, value) (line 881)
        # Getting the type of 'LL' (line 883)
        LL_68458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 24), 'LL')
        unicode_68459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 28), 'unicode', u'SW')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 23), dict_68453, (LL_68458, unicode_68459))
        # Adding element type (key, value) (line 881)
        # Getting the type of 'LR' (line 884)
        LR_68460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 24), 'LR')
        unicode_68461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 28), 'unicode', u'SE')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 23), dict_68453, (LR_68460, unicode_68461))
        # Adding element type (key, value) (line 881)
        # Getting the type of 'R' (line 885)
        R_68462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 24), 'R')
        unicode_68463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 27), 'unicode', u'E')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 23), dict_68453, (R_68462, unicode_68463))
        # Adding element type (key, value) (line 881)
        # Getting the type of 'CL' (line 886)
        CL_68464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 24), 'CL')
        unicode_68465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 28), 'unicode', u'W')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 23), dict_68453, (CL_68464, unicode_68465))
        # Adding element type (key, value) (line 881)
        # Getting the type of 'CR' (line 887)
        CR_68466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 24), 'CR')
        unicode_68467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 28), 'unicode', u'E')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 23), dict_68453, (CR_68466, unicode_68467))
        # Adding element type (key, value) (line 881)
        # Getting the type of 'LC' (line 888)
        LC_68468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 24), 'LC')
        unicode_68469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 28), 'unicode', u'S')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 23), dict_68453, (LC_68468, unicode_68469))
        # Adding element type (key, value) (line 881)
        # Getting the type of 'UC' (line 889)
        UC_68470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 24), 'UC')
        unicode_68471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 28), 'unicode', u'N')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 23), dict_68453, (UC_68470, unicode_68471))
        # Adding element type (key, value) (line 881)
        # Getting the type of 'C' (line 890)
        C_68472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 24), 'C')
        unicode_68473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 27), 'unicode', u'C')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 23), dict_68453, (C_68472, unicode_68473))
        
        # Assigning a type to the variable 'anchor_coefs' (line 881)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 8), 'anchor_coefs', dict_68453)
        
        # Assigning a Subscript to a Name (line 892):
        
        # Assigning a Subscript to a Name (line 892):
        
        # Assigning a Subscript to a Name (line 892):
        
        # Obtaining the type of the subscript
        # Getting the type of 'loc' (line 892)
        loc_68474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 25), 'loc')
        # Getting the type of 'anchor_coefs' (line 892)
        anchor_coefs_68475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 12), 'anchor_coefs')
        # Obtaining the member '__getitem__' of a type (line 892)
        getitem___68476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 12), anchor_coefs_68475, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 892)
        subscript_call_result_68477 = invoke(stypy.reporting.localization.Localization(__file__, 892, 12), getitem___68476, loc_68474)
        
        # Assigning a type to the variable 'c' (line 892)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 8), 'c', subscript_call_result_68477)
        
        # Assigning a Call to a Name (line 894):
        
        # Assigning a Call to a Name (line 894):
        
        # Assigning a Call to a Name (line 894):
        
        # Call to points_to_pixels(...): (line 894)
        # Processing the call arguments (line 894)
        # Getting the type of 'self' (line 894)
        self_68480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 45), 'self', False)
        # Obtaining the member '_fontsize' of a type (line 894)
        _fontsize_68481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 45), self_68480, '_fontsize')
        # Processing the call keyword arguments (line 894)
        kwargs_68482 = {}
        # Getting the type of 'renderer' (line 894)
        renderer_68478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 19), 'renderer', False)
        # Obtaining the member 'points_to_pixels' of a type (line 894)
        points_to_pixels_68479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 19), renderer_68478, 'points_to_pixels')
        # Calling points_to_pixels(args, kwargs) (line 894)
        points_to_pixels_call_result_68483 = invoke(stypy.reporting.localization.Localization(__file__, 894, 19), points_to_pixels_68479, *[_fontsize_68481], **kwargs_68482)
        
        # Assigning a type to the variable 'fontsize' (line 894)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 8), 'fontsize', points_to_pixels_call_result_68483)
        
        # Assigning a Call to a Name (line 895):
        
        # Assigning a Call to a Name (line 895):
        
        # Assigning a Call to a Name (line 895):
        
        # Call to padded(...): (line 895)
        # Processing the call arguments (line 895)
        
        # Getting the type of 'self' (line 895)
        self_68486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 40), 'self', False)
        # Obtaining the member 'borderaxespad' of a type (line 895)
        borderaxespad_68487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 40), self_68486, 'borderaxespad')
        # Applying the 'usub' unary operator (line 895)
        result___neg___68488 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 38), 'usub', borderaxespad_68487)
        
        # Getting the type of 'fontsize' (line 895)
        fontsize_68489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 62), 'fontsize', False)
        # Applying the binary operator '*' (line 895)
        result_mul_68490 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 38), '*', result___neg___68488, fontsize_68489)
        
        # Processing the call keyword arguments (line 895)
        kwargs_68491 = {}
        # Getting the type of 'parentbbox' (line 895)
        parentbbox_68484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 20), 'parentbbox', False)
        # Obtaining the member 'padded' of a type (line 895)
        padded_68485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 20), parentbbox_68484, 'padded')
        # Calling padded(args, kwargs) (line 895)
        padded_call_result_68492 = invoke(stypy.reporting.localization.Localization(__file__, 895, 20), padded_68485, *[result_mul_68490], **kwargs_68491)
        
        # Assigning a type to the variable 'container' (line 895)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 8), 'container', padded_call_result_68492)
        
        # Assigning a Call to a Name (line 896):
        
        # Assigning a Call to a Name (line 896):
        
        # Assigning a Call to a Name (line 896):
        
        # Call to anchored(...): (line 896)
        # Processing the call arguments (line 896)
        # Getting the type of 'c' (line 896)
        c_68495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 37), 'c', False)
        # Processing the call keyword arguments (line 896)
        # Getting the type of 'container' (line 896)
        container_68496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 50), 'container', False)
        keyword_68497 = container_68496
        kwargs_68498 = {'container': keyword_68497}
        # Getting the type of 'bbox' (line 896)
        bbox_68493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 23), 'bbox', False)
        # Obtaining the member 'anchored' of a type (line 896)
        anchored_68494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 23), bbox_68493, 'anchored')
        # Calling anchored(args, kwargs) (line 896)
        anchored_call_result_68499 = invoke(stypy.reporting.localization.Localization(__file__, 896, 23), anchored_68494, *[c_68495], **kwargs_68498)
        
        # Assigning a type to the variable 'anchored_box' (line 896)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 8), 'anchored_box', anchored_call_result_68499)
        
        # Obtaining an instance of the builtin type 'tuple' (line 897)
        tuple_68500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 897)
        # Adding element type (line 897)
        # Getting the type of 'anchored_box' (line 897)
        anchored_box_68501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 15), 'anchored_box')
        # Obtaining the member 'x0' of a type (line 897)
        x0_68502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 15), anchored_box_68501, 'x0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 15), tuple_68500, x0_68502)
        # Adding element type (line 897)
        # Getting the type of 'anchored_box' (line 897)
        anchored_box_68503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 32), 'anchored_box')
        # Obtaining the member 'y0' of a type (line 897)
        y0_68504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 32), anchored_box_68503, 'y0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 15), tuple_68500, y0_68504)
        
        # Assigning a type to the variable 'stypy_return_type' (line 897)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 8), 'stypy_return_type', tuple_68500)
        
        # ################# End of '_get_anchored_bbox(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_anchored_bbox' in the type store
        # Getting the type of 'stypy_return_type' (line 864)
        stypy_return_type_68505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68505)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_anchored_bbox'
        return stypy_return_type_68505


    @norecursion
    def _find_best_position(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 899)
        None_68506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 68), 'None')
        defaults = [None_68506]
        # Create a new context for function '_find_best_position'
        module_type_store = module_type_store.open_function_context('_find_best_position', 899, 4, False)
        # Assigning a type to the variable 'self' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend._find_best_position.__dict__.__setitem__('stypy_localization', localization)
        Legend._find_best_position.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend._find_best_position.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend._find_best_position.__dict__.__setitem__('stypy_function_name', 'Legend._find_best_position')
        Legend._find_best_position.__dict__.__setitem__('stypy_param_names_list', ['width', 'height', 'renderer', 'consider'])
        Legend._find_best_position.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend._find_best_position.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend._find_best_position.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend._find_best_position.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend._find_best_position.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend._find_best_position.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend._find_best_position', ['width', 'height', 'renderer', 'consider'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_find_best_position', localization, ['width', 'height', 'renderer', 'consider'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_find_best_position(...)' code ##################

        unicode_68507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, (-1)), 'unicode', u'\n        Determine the best location to place the legend.\n\n        `consider` is a list of (x, y) pairs to consider as a potential\n        lower-left corner of the legend. All are display coords.\n        ')
        # Evaluating assert statement condition
        # Getting the type of 'self' (line 907)
        self_68508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 15), 'self')
        # Obtaining the member 'isaxes' of a type (line 907)
        isaxes_68509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 15), self_68508, 'isaxes')
        
        # Assigning a Call to a Tuple (line 909):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to _auto_legend_data(...): (line 909)
        # Processing the call keyword arguments (line 909)
        kwargs_68512 = {}
        # Getting the type of 'self' (line 909)
        self_68510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 40), 'self', False)
        # Obtaining the member '_auto_legend_data' of a type (line 909)
        _auto_legend_data_68511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 40), self_68510, '_auto_legend_data')
        # Calling _auto_legend_data(args, kwargs) (line 909)
        _auto_legend_data_call_result_68513 = invoke(stypy.reporting.localization.Localization(__file__, 909, 40), _auto_legend_data_68511, *[], **kwargs_68512)
        
        # Assigning a type to the variable 'call_assignment_66668' (line 909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66668', _auto_legend_data_call_result_68513)
        
        # Assigning a Call to a Name (line 909):
        
        # Assigning a Call to a Name (line 909):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68517 = {}
        # Getting the type of 'call_assignment_66668' (line 909)
        call_assignment_66668_68514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66668', False)
        # Obtaining the member '__getitem__' of a type (line 909)
        getitem___68515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 8), call_assignment_66668_68514, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68518 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68515, *[int_68516], **kwargs_68517)
        
        # Assigning a type to the variable 'call_assignment_66669' (line 909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66669', getitem___call_result_68518)
        
        # Assigning a Name to a Name (line 909):
        
        # Assigning a Name to a Name (line 909):
        # Getting the type of 'call_assignment_66669' (line 909)
        call_assignment_66669_68519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66669')
        # Assigning a type to the variable 'verts' (line 909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'verts', call_assignment_66669_68519)
        
        # Assigning a Call to a Name (line 909):
        
        # Assigning a Call to a Name (line 909):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68523 = {}
        # Getting the type of 'call_assignment_66668' (line 909)
        call_assignment_66668_68520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66668', False)
        # Obtaining the member '__getitem__' of a type (line 909)
        getitem___68521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 8), call_assignment_66668_68520, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68524 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68521, *[int_68522], **kwargs_68523)
        
        # Assigning a type to the variable 'call_assignment_66670' (line 909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66670', getitem___call_result_68524)
        
        # Assigning a Name to a Name (line 909):
        
        # Assigning a Name to a Name (line 909):
        # Getting the type of 'call_assignment_66670' (line 909)
        call_assignment_66670_68525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66670')
        # Assigning a type to the variable 'bboxes' (line 909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 15), 'bboxes', call_assignment_66670_68525)
        
        # Assigning a Call to a Name (line 909):
        
        # Assigning a Call to a Name (line 909):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68529 = {}
        # Getting the type of 'call_assignment_66668' (line 909)
        call_assignment_66668_68526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66668', False)
        # Obtaining the member '__getitem__' of a type (line 909)
        getitem___68527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 8), call_assignment_66668_68526, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68530 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68527, *[int_68528], **kwargs_68529)
        
        # Assigning a type to the variable 'call_assignment_66671' (line 909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66671', getitem___call_result_68530)
        
        # Assigning a Name to a Name (line 909):
        
        # Assigning a Name to a Name (line 909):
        # Getting the type of 'call_assignment_66671' (line 909)
        call_assignment_66671_68531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66671')
        # Assigning a type to the variable 'lines' (line 909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 23), 'lines', call_assignment_66671_68531)
        
        # Assigning a Call to a Name (line 909):
        
        # Assigning a Call to a Name (line 909):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68535 = {}
        # Getting the type of 'call_assignment_66668' (line 909)
        call_assignment_66668_68532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66668', False)
        # Obtaining the member '__getitem__' of a type (line 909)
        getitem___68533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 8), call_assignment_66668_68532, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68536 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68533, *[int_68534], **kwargs_68535)
        
        # Assigning a type to the variable 'call_assignment_66672' (line 909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66672', getitem___call_result_68536)
        
        # Assigning a Name to a Name (line 909):
        
        # Assigning a Name to a Name (line 909):
        # Getting the type of 'call_assignment_66672' (line 909)
        call_assignment_66672_68537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 8), 'call_assignment_66672')
        # Assigning a type to the variable 'offsets' (line 909)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 30), 'offsets', call_assignment_66672_68537)
        
        # Assigning a Call to a Name (line 911):
        
        # Assigning a Call to a Name (line 911):
        
        # Assigning a Call to a Name (line 911):
        
        # Call to from_bounds(...): (line 911)
        # Processing the call arguments (line 911)
        int_68540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 911, 32), 'int')
        int_68541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 911, 35), 'int')
        # Getting the type of 'width' (line 911)
        width_68542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 38), 'width', False)
        # Getting the type of 'height' (line 911)
        height_68543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 45), 'height', False)
        # Processing the call keyword arguments (line 911)
        kwargs_68544 = {}
        # Getting the type of 'Bbox' (line 911)
        Bbox_68538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 15), 'Bbox', False)
        # Obtaining the member 'from_bounds' of a type (line 911)
        from_bounds_68539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 911, 15), Bbox_68538, 'from_bounds')
        # Calling from_bounds(args, kwargs) (line 911)
        from_bounds_call_result_68545 = invoke(stypy.reporting.localization.Localization(__file__, 911, 15), from_bounds_68539, *[int_68540, int_68541, width_68542, height_68543], **kwargs_68544)
        
        # Assigning a type to the variable 'bbox' (line 911)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 8), 'bbox', from_bounds_call_result_68545)
        
        # Type idiom detected: calculating its left and rigth part (line 912)
        # Getting the type of 'consider' (line 912)
        consider_68546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 11), 'consider')
        # Getting the type of 'None' (line 912)
        None_68547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 23), 'None')
        
        (may_be_68548, more_types_in_union_68549) = may_be_none(consider_68546, None_68547)

        if may_be_68548:

            if more_types_in_union_68549:
                # Runtime conditional SSA (line 912)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a ListComp to a Name (line 913):
            
            # Assigning a ListComp to a Name (line 913):
            
            # Assigning a ListComp to a Name (line 913):
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to range(...): (line 916)
            # Processing the call arguments (line 916)
            int_68562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 39), 'int')
            
            # Call to len(...): (line 916)
            # Processing the call arguments (line 916)
            # Getting the type of 'self' (line 916)
            self_68564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 46), 'self', False)
            # Obtaining the member 'codes' of a type (line 916)
            codes_68565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 46), self_68564, 'codes')
            # Processing the call keyword arguments (line 916)
            kwargs_68566 = {}
            # Getting the type of 'len' (line 916)
            len_68563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 42), 'len', False)
            # Calling len(args, kwargs) (line 916)
            len_call_result_68567 = invoke(stypy.reporting.localization.Localization(__file__, 916, 42), len_68563, *[codes_68565], **kwargs_68566)
            
            # Processing the call keyword arguments (line 916)
            kwargs_68568 = {}
            # Getting the type of 'range' (line 916)
            range_68561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 33), 'range', False)
            # Calling range(args, kwargs) (line 916)
            range_call_result_68569 = invoke(stypy.reporting.localization.Localization(__file__, 916, 33), range_68561, *[int_68562, len_call_result_68567], **kwargs_68568)
            
            comprehension_68570 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 913, 24), range_call_result_68569)
            # Assigning a type to the variable 'x' (line 913)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 24), 'x', comprehension_68570)
            
            # Call to _get_anchored_bbox(...): (line 913)
            # Processing the call arguments (line 913)
            # Getting the type of 'x' (line 913)
            x_68552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 48), 'x', False)
            # Getting the type of 'bbox' (line 913)
            bbox_68553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 51), 'bbox', False)
            
            # Call to get_bbox_to_anchor(...): (line 914)
            # Processing the call keyword arguments (line 914)
            kwargs_68556 = {}
            # Getting the type of 'self' (line 914)
            self_68554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 48), 'self', False)
            # Obtaining the member 'get_bbox_to_anchor' of a type (line 914)
            get_bbox_to_anchor_68555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 48), self_68554, 'get_bbox_to_anchor')
            # Calling get_bbox_to_anchor(args, kwargs) (line 914)
            get_bbox_to_anchor_call_result_68557 = invoke(stypy.reporting.localization.Localization(__file__, 914, 48), get_bbox_to_anchor_68555, *[], **kwargs_68556)
            
            # Getting the type of 'renderer' (line 915)
            renderer_68558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 48), 'renderer', False)
            # Processing the call keyword arguments (line 913)
            kwargs_68559 = {}
            # Getting the type of 'self' (line 913)
            self_68550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 24), 'self', False)
            # Obtaining the member '_get_anchored_bbox' of a type (line 913)
            _get_anchored_bbox_68551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 24), self_68550, '_get_anchored_bbox')
            # Calling _get_anchored_bbox(args, kwargs) (line 913)
            _get_anchored_bbox_call_result_68560 = invoke(stypy.reporting.localization.Localization(__file__, 913, 24), _get_anchored_bbox_68551, *[x_68552, bbox_68553, get_bbox_to_anchor_call_result_68557, renderer_68558], **kwargs_68559)
            
            list_68571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 24), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 913, 24), list_68571, _get_anchored_bbox_call_result_68560)
            # Assigning a type to the variable 'consider' (line 913)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 12), 'consider', list_68571)

            if more_types_in_union_68549:
                # SSA join for if statement (line 912)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 918):
        
        # Assigning a List to a Name (line 918):
        
        # Assigning a List to a Name (line 918):
        
        # Obtaining an instance of the builtin type 'list' (line 918)
        list_68572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 918)
        
        # Assigning a type to the variable 'candidates' (line 918)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 8), 'candidates', list_68572)
        
        
        # Call to enumerate(...): (line 919)
        # Processing the call arguments (line 919)
        # Getting the type of 'consider' (line 919)
        consider_68574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 37), 'consider', False)
        # Processing the call keyword arguments (line 919)
        kwargs_68575 = {}
        # Getting the type of 'enumerate' (line 919)
        enumerate_68573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 27), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 919)
        enumerate_call_result_68576 = invoke(stypy.reporting.localization.Localization(__file__, 919, 27), enumerate_68573, *[consider_68574], **kwargs_68575)
        
        # Testing the type of a for loop iterable (line 919)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 919, 8), enumerate_call_result_68576)
        # Getting the type of the for loop variable (line 919)
        for_loop_var_68577 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 919, 8), enumerate_call_result_68576)
        # Assigning a type to the variable 'idx' (line 919)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 8), 'idx', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 919, 8), for_loop_var_68577))
        # Assigning a type to the variable 'l' (line 919)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 8), 'l', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 919, 8), for_loop_var_68577))
        # Assigning a type to the variable 'b' (line 919)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 8), 'b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 919, 8), for_loop_var_68577))
        # SSA begins for a for statement (line 919)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 920):
        
        # Assigning a Call to a Name (line 920):
        
        # Assigning a Call to a Name (line 920):
        
        # Call to from_bounds(...): (line 920)
        # Processing the call arguments (line 920)
        # Getting the type of 'l' (line 920)
        l_68580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 41), 'l', False)
        # Getting the type of 'b' (line 920)
        b_68581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 44), 'b', False)
        # Getting the type of 'width' (line 920)
        width_68582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 47), 'width', False)
        # Getting the type of 'height' (line 920)
        height_68583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 54), 'height', False)
        # Processing the call keyword arguments (line 920)
        kwargs_68584 = {}
        # Getting the type of 'Bbox' (line 920)
        Bbox_68578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 24), 'Bbox', False)
        # Obtaining the member 'from_bounds' of a type (line 920)
        from_bounds_68579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 24), Bbox_68578, 'from_bounds')
        # Calling from_bounds(args, kwargs) (line 920)
        from_bounds_call_result_68585 = invoke(stypy.reporting.localization.Localization(__file__, 920, 24), from_bounds_68579, *[l_68580, b_68581, width_68582, height_68583], **kwargs_68584)
        
        # Assigning a type to the variable 'legendBox' (line 920)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 12), 'legendBox', from_bounds_call_result_68585)
        
        # Assigning a Num to a Name (line 921):
        
        # Assigning a Num to a Name (line 921):
        
        # Assigning a Num to a Name (line 921):
        int_68586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 22), 'int')
        # Assigning a type to the variable 'badness' (line 921)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 12), 'badness', int_68586)
        
        # Assigning a BinOp to a Name (line 925):
        
        # Assigning a BinOp to a Name (line 925):
        
        # Assigning a BinOp to a Name (line 925):
        
        # Call to count_contains(...): (line 925)
        # Processing the call arguments (line 925)
        # Getting the type of 'verts' (line 925)
        verts_68589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 48), 'verts', False)
        # Processing the call keyword arguments (line 925)
        kwargs_68590 = {}
        # Getting the type of 'legendBox' (line 925)
        legendBox_68587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 23), 'legendBox', False)
        # Obtaining the member 'count_contains' of a type (line 925)
        count_contains_68588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 23), legendBox_68587, 'count_contains')
        # Calling count_contains(args, kwargs) (line 925)
        count_contains_call_result_68591 = invoke(stypy.reporting.localization.Localization(__file__, 925, 23), count_contains_68588, *[verts_68589], **kwargs_68590)
        
        
        # Call to count_contains(...): (line 926)
        # Processing the call arguments (line 926)
        # Getting the type of 'offsets' (line 926)
        offsets_68594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 50), 'offsets', False)
        # Processing the call keyword arguments (line 926)
        kwargs_68595 = {}
        # Getting the type of 'legendBox' (line 926)
        legendBox_68592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 25), 'legendBox', False)
        # Obtaining the member 'count_contains' of a type (line 926)
        count_contains_68593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 25), legendBox_68592, 'count_contains')
        # Calling count_contains(args, kwargs) (line 926)
        count_contains_call_result_68596 = invoke(stypy.reporting.localization.Localization(__file__, 926, 25), count_contains_68593, *[offsets_68594], **kwargs_68595)
        
        # Applying the binary operator '+' (line 925)
        result_add_68597 = python_operator(stypy.reporting.localization.Localization(__file__, 925, 23), '+', count_contains_call_result_68591, count_contains_call_result_68596)
        
        
        # Call to count_overlaps(...): (line 927)
        # Processing the call arguments (line 927)
        # Getting the type of 'bboxes' (line 927)
        bboxes_68600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 50), 'bboxes', False)
        # Processing the call keyword arguments (line 927)
        kwargs_68601 = {}
        # Getting the type of 'legendBox' (line 927)
        legendBox_68598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 25), 'legendBox', False)
        # Obtaining the member 'count_overlaps' of a type (line 927)
        count_overlaps_68599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 25), legendBox_68598, 'count_overlaps')
        # Calling count_overlaps(args, kwargs) (line 927)
        count_overlaps_call_result_68602 = invoke(stypy.reporting.localization.Localization(__file__, 927, 25), count_overlaps_68599, *[bboxes_68600], **kwargs_68601)
        
        # Applying the binary operator '+' (line 927)
        result_add_68603 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 23), '+', result_add_68597, count_overlaps_call_result_68602)
        
        
        # Call to sum(...): (line 928)
        # Processing the call arguments (line 928)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 928, 29, True)
        # Calculating comprehension expression
        # Getting the type of 'lines' (line 929)
        lines_68612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 41), 'lines', False)
        comprehension_68613 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 928, 29), lines_68612)
        # Assigning a type to the variable 'line' (line 928)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 29), 'line', comprehension_68613)
        
        # Call to intersects_bbox(...): (line 928)
        # Processing the call arguments (line 928)
        # Getting the type of 'legendBox' (line 928)
        legendBox_68607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 50), 'legendBox', False)
        # Processing the call keyword arguments (line 928)
        # Getting the type of 'False' (line 928)
        False_68608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 68), 'False', False)
        keyword_68609 = False_68608
        kwargs_68610 = {'filled': keyword_68609}
        # Getting the type of 'line' (line 928)
        line_68605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 29), 'line', False)
        # Obtaining the member 'intersects_bbox' of a type (line 928)
        intersects_bbox_68606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 29), line_68605, 'intersects_bbox')
        # Calling intersects_bbox(args, kwargs) (line 928)
        intersects_bbox_call_result_68611 = invoke(stypy.reporting.localization.Localization(__file__, 928, 29), intersects_bbox_68606, *[legendBox_68607], **kwargs_68610)
        
        list_68614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 29), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 928, 29), list_68614, intersects_bbox_call_result_68611)
        # Processing the call keyword arguments (line 928)
        kwargs_68615 = {}
        # Getting the type of 'sum' (line 928)
        sum_68604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 25), 'sum', False)
        # Calling sum(args, kwargs) (line 928)
        sum_call_result_68616 = invoke(stypy.reporting.localization.Localization(__file__, 928, 25), sum_68604, *[list_68614], **kwargs_68615)
        
        # Applying the binary operator '+' (line 928)
        result_add_68617 = python_operator(stypy.reporting.localization.Localization(__file__, 928, 23), '+', result_add_68603, sum_call_result_68616)
        
        # Assigning a type to the variable 'badness' (line 925)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 12), 'badness', result_add_68617)
        
        
        # Getting the type of 'badness' (line 930)
        badness_68618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 15), 'badness')
        int_68619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 26), 'int')
        # Applying the binary operator '==' (line 930)
        result_eq_68620 = python_operator(stypy.reporting.localization.Localization(__file__, 930, 15), '==', badness_68618, int_68619)
        
        # Testing the type of an if condition (line 930)
        if_condition_68621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 930, 12), result_eq_68620)
        # Assigning a type to the variable 'if_condition_68621' (line 930)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 12), 'if_condition_68621', if_condition_68621)
        # SSA begins for if statement (line 930)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 931)
        tuple_68622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 931)
        # Adding element type (line 931)
        # Getting the type of 'l' (line 931)
        l_68623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 23), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 931, 23), tuple_68622, l_68623)
        # Adding element type (line 931)
        # Getting the type of 'b' (line 931)
        b_68624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 26), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 931, 23), tuple_68622, b_68624)
        
        # Assigning a type to the variable 'stypy_return_type' (line 931)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 16), 'stypy_return_type', tuple_68622)
        # SSA join for if statement (line 930)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 933)
        # Processing the call arguments (line 933)
        
        # Obtaining an instance of the builtin type 'tuple' (line 933)
        tuple_68627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 933)
        # Adding element type (line 933)
        # Getting the type of 'badness' (line 933)
        badness_68628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 31), 'badness', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 933, 31), tuple_68627, badness_68628)
        # Adding element type (line 933)
        # Getting the type of 'idx' (line 933)
        idx_68629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 40), 'idx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 933, 31), tuple_68627, idx_68629)
        # Adding element type (line 933)
        
        # Obtaining an instance of the builtin type 'tuple' (line 933)
        tuple_68630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 933)
        # Adding element type (line 933)
        # Getting the type of 'l' (line 933)
        l_68631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 46), 'l', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 933, 46), tuple_68630, l_68631)
        # Adding element type (line 933)
        # Getting the type of 'b' (line 933)
        b_68632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 49), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 933, 46), tuple_68630, b_68632)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 933, 31), tuple_68627, tuple_68630)
        
        # Processing the call keyword arguments (line 933)
        kwargs_68633 = {}
        # Getting the type of 'candidates' (line 933)
        candidates_68625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 12), 'candidates', False)
        # Obtaining the member 'append' of a type (line 933)
        append_68626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 933, 12), candidates_68625, 'append')
        # Calling append(args, kwargs) (line 933)
        append_call_result_68634 = invoke(stypy.reporting.localization.Localization(__file__, 933, 12), append_68626, *[tuple_68627], **kwargs_68633)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 935):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to min(...): (line 935)
        # Processing the call arguments (line 935)
        # Getting the type of 'candidates' (line 935)
        candidates_68636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 27), 'candidates', False)
        # Processing the call keyword arguments (line 935)
        kwargs_68637 = {}
        # Getting the type of 'min' (line 935)
        min_68635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 23), 'min', False)
        # Calling min(args, kwargs) (line 935)
        min_call_result_68638 = invoke(stypy.reporting.localization.Localization(__file__, 935, 23), min_68635, *[candidates_68636], **kwargs_68637)
        
        # Assigning a type to the variable 'call_assignment_66673' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'call_assignment_66673', min_call_result_68638)
        
        # Assigning a Call to a Name (line 935):
        
        # Assigning a Call to a Name (line 935):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68642 = {}
        # Getting the type of 'call_assignment_66673' (line 935)
        call_assignment_66673_68639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'call_assignment_66673', False)
        # Obtaining the member '__getitem__' of a type (line 935)
        getitem___68640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 8), call_assignment_66673_68639, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68643 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68640, *[int_68641], **kwargs_68642)
        
        # Assigning a type to the variable 'call_assignment_66674' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'call_assignment_66674', getitem___call_result_68643)
        
        # Assigning a Name to a Name (line 935):
        
        # Assigning a Name to a Name (line 935):
        # Getting the type of 'call_assignment_66674' (line 935)
        call_assignment_66674_68644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'call_assignment_66674')
        # Assigning a type to the variable '_' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), '_', call_assignment_66674_68644)
        
        # Assigning a Call to a Name (line 935):
        
        # Assigning a Call to a Name (line 935):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68648 = {}
        # Getting the type of 'call_assignment_66673' (line 935)
        call_assignment_66673_68645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'call_assignment_66673', False)
        # Obtaining the member '__getitem__' of a type (line 935)
        getitem___68646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 8), call_assignment_66673_68645, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68649 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68646, *[int_68647], **kwargs_68648)
        
        # Assigning a type to the variable 'call_assignment_66675' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'call_assignment_66675', getitem___call_result_68649)
        
        # Assigning a Name to a Name (line 935):
        
        # Assigning a Name to a Name (line 935):
        # Getting the type of 'call_assignment_66675' (line 935)
        call_assignment_66675_68650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'call_assignment_66675')
        # Assigning a type to the variable '_' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 11), '_', call_assignment_66675_68650)
        
        # Assigning a Call to a Name (line 935):
        
        # Assigning a Call to a Name (line 935):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68654 = {}
        # Getting the type of 'call_assignment_66673' (line 935)
        call_assignment_66673_68651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'call_assignment_66673', False)
        # Obtaining the member '__getitem__' of a type (line 935)
        getitem___68652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 8), call_assignment_66673_68651, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68655 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68652, *[int_68653], **kwargs_68654)
        
        # Assigning a type to the variable 'call_assignment_66676' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'call_assignment_66676', getitem___call_result_68655)
        
        # Assigning a Name to a Tuple (line 935):
        
        # Assigning a Subscript to a Name (line 935):
        
        # Obtaining the type of the subscript
        int_68656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 8), 'int')
        # Getting the type of 'call_assignment_66676' (line 935)
        call_assignment_66676_68657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'call_assignment_66676')
        # Obtaining the member '__getitem__' of a type (line 935)
        getitem___68658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 8), call_assignment_66676_68657, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 935)
        subscript_call_result_68659 = invoke(stypy.reporting.localization.Localization(__file__, 935, 8), getitem___68658, int_68656)
        
        # Assigning a type to the variable 'tuple_var_assignment_66677' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'tuple_var_assignment_66677', subscript_call_result_68659)
        
        # Assigning a Subscript to a Name (line 935):
        
        # Obtaining the type of the subscript
        int_68660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 8), 'int')
        # Getting the type of 'call_assignment_66676' (line 935)
        call_assignment_66676_68661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'call_assignment_66676')
        # Obtaining the member '__getitem__' of a type (line 935)
        getitem___68662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 8), call_assignment_66676_68661, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 935)
        subscript_call_result_68663 = invoke(stypy.reporting.localization.Localization(__file__, 935, 8), getitem___68662, int_68660)
        
        # Assigning a type to the variable 'tuple_var_assignment_66678' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'tuple_var_assignment_66678', subscript_call_result_68663)
        
        # Assigning a Name to a Name (line 935):
        # Getting the type of 'tuple_var_assignment_66677' (line 935)
        tuple_var_assignment_66677_68664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'tuple_var_assignment_66677')
        # Assigning a type to the variable 'l' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 15), 'l', tuple_var_assignment_66677_68664)
        
        # Assigning a Name to a Name (line 935):
        # Getting the type of 'tuple_var_assignment_66678' (line 935)
        tuple_var_assignment_66678_68665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'tuple_var_assignment_66678')
        # Assigning a type to the variable 'b' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 18), 'b', tuple_var_assignment_66678_68665)
        
        # Obtaining an instance of the builtin type 'tuple' (line 936)
        tuple_68666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 936)
        # Adding element type (line 936)
        # Getting the type of 'l' (line 936)
        l_68667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 15), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 936, 15), tuple_68666, l_68667)
        # Adding element type (line 936)
        # Getting the type of 'b' (line 936)
        b_68668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 18), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 936, 15), tuple_68666, b_68668)
        
        # Assigning a type to the variable 'stypy_return_type' (line 936)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 8), 'stypy_return_type', tuple_68666)
        
        # ################# End of '_find_best_position(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_find_best_position' in the type store
        # Getting the type of 'stypy_return_type' (line 899)
        stypy_return_type_68669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68669)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_find_best_position'
        return stypy_return_type_68669


    @norecursion
    def contains(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'contains'
        module_type_store = module_type_store.open_function_context('contains', 938, 4, False)
        # Assigning a type to the variable 'self' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.contains.__dict__.__setitem__('stypy_localization', localization)
        Legend.contains.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.contains.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.contains.__dict__.__setitem__('stypy_function_name', 'Legend.contains')
        Legend.contains.__dict__.__setitem__('stypy_param_names_list', ['event'])
        Legend.contains.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.contains.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.contains.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.contains.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.contains.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.contains.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.contains', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'contains', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'contains(...)' code ##################

        
        # Call to contains(...): (line 939)
        # Processing the call arguments (line 939)
        # Getting the type of 'event' (line 939)
        event_68673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 41), 'event', False)
        # Processing the call keyword arguments (line 939)
        kwargs_68674 = {}
        # Getting the type of 'self' (line 939)
        self_68670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 15), 'self', False)
        # Obtaining the member 'legendPatch' of a type (line 939)
        legendPatch_68671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 15), self_68670, 'legendPatch')
        # Obtaining the member 'contains' of a type (line 939)
        contains_68672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 15), legendPatch_68671, 'contains')
        # Calling contains(args, kwargs) (line 939)
        contains_call_result_68675 = invoke(stypy.reporting.localization.Localization(__file__, 939, 15), contains_68672, *[event_68673], **kwargs_68674)
        
        # Assigning a type to the variable 'stypy_return_type' (line 939)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 8), 'stypy_return_type', contains_call_result_68675)
        
        # ################# End of 'contains(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'contains' in the type store
        # Getting the type of 'stypy_return_type' (line 938)
        stypy_return_type_68676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68676)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'contains'
        return stypy_return_type_68676


    @norecursion
    def draggable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 941)
        None_68677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 30), 'None')
        # Getting the type of 'False' (line 941)
        False_68678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 45), 'False')
        unicode_68679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 59), 'unicode', u'loc')
        defaults = [None_68677, False_68678, unicode_68679]
        # Create a new context for function 'draggable'
        module_type_store = module_type_store.open_function_context('draggable', 941, 4, False)
        # Assigning a type to the variable 'self' (line 942)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Legend.draggable.__dict__.__setitem__('stypy_localization', localization)
        Legend.draggable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Legend.draggable.__dict__.__setitem__('stypy_type_store', module_type_store)
        Legend.draggable.__dict__.__setitem__('stypy_function_name', 'Legend.draggable')
        Legend.draggable.__dict__.__setitem__('stypy_param_names_list', ['state', 'use_blit', 'update'])
        Legend.draggable.__dict__.__setitem__('stypy_varargs_param_name', None)
        Legend.draggable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Legend.draggable.__dict__.__setitem__('stypy_call_defaults', defaults)
        Legend.draggable.__dict__.__setitem__('stypy_call_varargs', varargs)
        Legend.draggable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Legend.draggable.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Legend.draggable', ['state', 'use_blit', 'update'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draggable', localization, ['state', 'use_blit', 'update'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draggable(...)' code ##################

        unicode_68680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, (-1)), 'unicode', u'\n        Set the draggable state -- if state is\n\n          * None : toggle the current state\n\n          * True : turn draggable on\n\n          * False : turn draggable off\n\n        If draggable is on, you can drag the legend on the canvas with\n        the mouse.  The DraggableLegend helper instance is returned if\n        draggable is on.\n\n        The update parameter control which parameter of the legend changes\n        when dragged. If update is "loc", the *loc* parameter of the legend\n        is changed. If "bbox", the *bbox_to_anchor* parameter is changed.\n        ')
        
        # Assigning a Compare to a Name (line 959):
        
        # Assigning a Compare to a Name (line 959):
        
        # Assigning a Compare to a Name (line 959):
        
        # Getting the type of 'self' (line 959)
        self_68681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 23), 'self')
        # Obtaining the member '_draggable' of a type (line 959)
        _draggable_68682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 959, 23), self_68681, '_draggable')
        # Getting the type of 'None' (line 959)
        None_68683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 46), 'None')
        # Applying the binary operator 'isnot' (line 959)
        result_is_not_68684 = python_operator(stypy.reporting.localization.Localization(__file__, 959, 23), 'isnot', _draggable_68682, None_68683)
        
        # Assigning a type to the variable 'is_draggable' (line 959)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 8), 'is_draggable', result_is_not_68684)
        
        # Type idiom detected: calculating its left and rigth part (line 962)
        # Getting the type of 'state' (line 962)
        state_68685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 11), 'state')
        # Getting the type of 'None' (line 962)
        None_68686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 20), 'None')
        
        (may_be_68687, more_types_in_union_68688) = may_be_none(state_68685, None_68686)

        if may_be_68687:

            if more_types_in_union_68688:
                # Runtime conditional SSA (line 962)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a UnaryOp to a Name (line 963):
            
            # Assigning a UnaryOp to a Name (line 963):
            
            # Assigning a UnaryOp to a Name (line 963):
            
            # Getting the type of 'is_draggable' (line 963)
            is_draggable_68689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 24), 'is_draggable')
            # Applying the 'not' unary operator (line 963)
            result_not__68690 = python_operator(stypy.reporting.localization.Localization(__file__, 963, 20), 'not', is_draggable_68689)
            
            # Assigning a type to the variable 'state' (line 963)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 12), 'state', result_not__68690)

            if more_types_in_union_68688:
                # SSA join for if statement (line 962)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'state' (line 965)
        state_68691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 11), 'state')
        # Testing the type of an if condition (line 965)
        if_condition_68692 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 965, 8), state_68691)
        # Assigning a type to the variable 'if_condition_68692' (line 965)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 8), 'if_condition_68692', if_condition_68692)
        # SSA begins for if statement (line 965)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 966)
        # Getting the type of 'self' (line 966)
        self_68693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 15), 'self')
        # Obtaining the member '_draggable' of a type (line 966)
        _draggable_68694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 15), self_68693, '_draggable')
        # Getting the type of 'None' (line 966)
        None_68695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 34), 'None')
        
        (may_be_68696, more_types_in_union_68697) = may_be_none(_draggable_68694, None_68695)

        if may_be_68696:

            if more_types_in_union_68697:
                # Runtime conditional SSA (line 966)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 967):
            
            # Assigning a Call to a Attribute (line 967):
            
            # Assigning a Call to a Attribute (line 967):
            
            # Call to DraggableLegend(...): (line 967)
            # Processing the call arguments (line 967)
            # Getting the type of 'self' (line 967)
            self_68699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 50), 'self', False)
            # Getting the type of 'use_blit' (line 968)
            use_blit_68700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 50), 'use_blit', False)
            # Processing the call keyword arguments (line 967)
            # Getting the type of 'update' (line 969)
            update_68701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 57), 'update', False)
            keyword_68702 = update_68701
            kwargs_68703 = {'update': keyword_68702}
            # Getting the type of 'DraggableLegend' (line 967)
            DraggableLegend_68698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 34), 'DraggableLegend', False)
            # Calling DraggableLegend(args, kwargs) (line 967)
            DraggableLegend_call_result_68704 = invoke(stypy.reporting.localization.Localization(__file__, 967, 34), DraggableLegend_68698, *[self_68699, use_blit_68700], **kwargs_68703)
            
            # Getting the type of 'self' (line 967)
            self_68705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 16), 'self')
            # Setting the type of the member '_draggable' of a type (line 967)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 967, 16), self_68705, '_draggable', DraggableLegend_call_result_68704)

            if more_types_in_union_68697:
                # SSA join for if statement (line 966)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 965)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 971)
        self_68706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 15), 'self')
        # Obtaining the member '_draggable' of a type (line 971)
        _draggable_68707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 15), self_68706, '_draggable')
        # Getting the type of 'None' (line 971)
        None_68708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 38), 'None')
        # Applying the binary operator 'isnot' (line 971)
        result_is_not_68709 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 15), 'isnot', _draggable_68707, None_68708)
        
        # Testing the type of an if condition (line 971)
        if_condition_68710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 971, 12), result_is_not_68709)
        # Assigning a type to the variable 'if_condition_68710' (line 971)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 12), 'if_condition_68710', if_condition_68710)
        # SSA begins for if statement (line 971)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to disconnect(...): (line 972)
        # Processing the call keyword arguments (line 972)
        kwargs_68714 = {}
        # Getting the type of 'self' (line 972)
        self_68711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 16), 'self', False)
        # Obtaining the member '_draggable' of a type (line 972)
        _draggable_68712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 16), self_68711, '_draggable')
        # Obtaining the member 'disconnect' of a type (line 972)
        disconnect_68713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 16), _draggable_68712, 'disconnect')
        # Calling disconnect(args, kwargs) (line 972)
        disconnect_call_result_68715 = invoke(stypy.reporting.localization.Localization(__file__, 972, 16), disconnect_68713, *[], **kwargs_68714)
        
        # SSA join for if statement (line 971)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 973):
        
        # Assigning a Name to a Attribute (line 973):
        
        # Assigning a Name to a Attribute (line 973):
        # Getting the type of 'None' (line 973)
        None_68716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 30), 'None')
        # Getting the type of 'self' (line 973)
        self_68717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 12), 'self')
        # Setting the type of the member '_draggable' of a type (line 973)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 12), self_68717, '_draggable', None_68716)
        # SSA join for if statement (line 965)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 975)
        self_68718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 15), 'self')
        # Obtaining the member '_draggable' of a type (line 975)
        _draggable_68719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 975, 15), self_68718, '_draggable')
        # Assigning a type to the variable 'stypy_return_type' (line 975)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 8), 'stypy_return_type', _draggable_68719)
        
        # ################# End of 'draggable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draggable' in the type store
        # Getting the type of 'stypy_return_type' (line 941)
        stypy_return_type_68720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68720)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draggable'
        return stypy_return_type_68720


# Assigning a type to the variable 'Legend' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'Legend', Legend)

# Assigning a Dict to a Name (line 132):

# Obtaining an instance of the builtin type 'dict' (line 132)
dict_68721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 132)
# Adding element type (key, value) (line 132)
unicode_68722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 13), 'unicode', u'best')
int_68723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), dict_68721, (unicode_68722, int_68723))
# Adding element type (key, value) (line 132)
unicode_68724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 13), 'unicode', u'upper right')
int_68725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), dict_68721, (unicode_68724, int_68725))
# Adding element type (key, value) (line 132)
unicode_68726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 13), 'unicode', u'upper left')
int_68727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), dict_68721, (unicode_68726, int_68727))
# Adding element type (key, value) (line 132)
unicode_68728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 13), 'unicode', u'lower left')
int_68729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), dict_68721, (unicode_68728, int_68729))
# Adding element type (key, value) (line 132)
unicode_68730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 13), 'unicode', u'lower right')
int_68731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), dict_68721, (unicode_68730, int_68731))
# Adding element type (key, value) (line 132)
unicode_68732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 13), 'unicode', u'right')
int_68733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), dict_68721, (unicode_68732, int_68733))
# Adding element type (key, value) (line 132)
unicode_68734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 13), 'unicode', u'center left')
int_68735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), dict_68721, (unicode_68734, int_68735))
# Adding element type (key, value) (line 132)
unicode_68736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 13), 'unicode', u'center right')
int_68737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), dict_68721, (unicode_68736, int_68737))
# Adding element type (key, value) (line 132)
unicode_68738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 13), 'unicode', u'lower center')
int_68739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), dict_68721, (unicode_68738, int_68739))
# Adding element type (key, value) (line 132)
unicode_68740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 13), 'unicode', u'upper center')
int_68741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), dict_68721, (unicode_68740, int_68741))
# Adding element type (key, value) (line 132)
unicode_68742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 13), 'unicode', u'center')
int_68743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 12), dict_68721, (unicode_68742, int_68743))

# Getting the type of 'Legend'
Legend_68744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legend')
# Setting the type of the member 'codes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legend_68744, 'codes', dict_68721)

# Assigning a Num to a Name (line 145):
int_68745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 13), 'int')
# Getting the type of 'Legend'
Legend_68746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legend')
# Setting the type of the member 'zorder' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legend_68746, 'zorder', int_68745)

# Assigning a Call to a Name (line 423):

# Call to property(...): (line 423)
# Processing the call arguments (line 423)
# Getting the type of 'Legend'
Legend_68748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legend', False)
# Obtaining the member '_get_loc' of a type
_get_loc_68749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legend_68748, '_get_loc')
# Getting the type of 'Legend'
Legend_68750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legend', False)
# Obtaining the member '_set_loc' of a type
_set_loc_68751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legend_68750, '_set_loc')
# Processing the call keyword arguments (line 423)
kwargs_68752 = {}
# Getting the type of 'property' (line 423)
property_68747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'property', False)
# Calling property(args, kwargs) (line 423)
property_call_result_68753 = invoke(stypy.reporting.localization.Localization(__file__, 423, 11), property_68747, *[_get_loc_68749, _set_loc_68751], **kwargs_68752)

# Getting the type of 'Legend'
Legend_68754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legend')
# Setting the type of the member '_loc' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legend_68754, '_loc', property_call_result_68753)

# Assigning a Dict to a Name (line 490):

# Obtaining an instance of the builtin type 'dict' (line 490)
dict_68755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 27), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 490)
# Adding element type (key, value) (line 490)
# Getting the type of 'StemContainer' (line 491)
StemContainer_68756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'StemContainer')

# Call to HandlerStem(...): (line 491)
# Processing the call keyword arguments (line 491)
kwargs_68759 = {}
# Getting the type of 'legend_handler' (line 491)
legend_handler_68757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 23), 'legend_handler', False)
# Obtaining the member 'HandlerStem' of a type (line 491)
HandlerStem_68758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 23), legend_handler_68757, 'HandlerStem')
# Calling HandlerStem(args, kwargs) (line 491)
HandlerStem_call_result_68760 = invoke(stypy.reporting.localization.Localization(__file__, 491, 23), HandlerStem_68758, *[], **kwargs_68759)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 27), dict_68755, (StemContainer_68756, HandlerStem_call_result_68760))
# Adding element type (key, value) (line 490)
# Getting the type of 'ErrorbarContainer' (line 492)
ErrorbarContainer_68761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'ErrorbarContainer')

# Call to HandlerErrorbar(...): (line 492)
# Processing the call keyword arguments (line 492)
kwargs_68764 = {}
# Getting the type of 'legend_handler' (line 492)
legend_handler_68762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 27), 'legend_handler', False)
# Obtaining the member 'HandlerErrorbar' of a type (line 492)
HandlerErrorbar_68763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 27), legend_handler_68762, 'HandlerErrorbar')
# Calling HandlerErrorbar(args, kwargs) (line 492)
HandlerErrorbar_call_result_68765 = invoke(stypy.reporting.localization.Localization(__file__, 492, 27), HandlerErrorbar_68763, *[], **kwargs_68764)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 27), dict_68755, (ErrorbarContainer_68761, HandlerErrorbar_call_result_68765))
# Adding element type (key, value) (line 490)
# Getting the type of 'Line2D' (line 493)
Line2D_68766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'Line2D')

# Call to HandlerLine2D(...): (line 493)
# Processing the call keyword arguments (line 493)
kwargs_68769 = {}
# Getting the type of 'legend_handler' (line 493)
legend_handler_68767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 16), 'legend_handler', False)
# Obtaining the member 'HandlerLine2D' of a type (line 493)
HandlerLine2D_68768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 16), legend_handler_68767, 'HandlerLine2D')
# Calling HandlerLine2D(args, kwargs) (line 493)
HandlerLine2D_call_result_68770 = invoke(stypy.reporting.localization.Localization(__file__, 493, 16), HandlerLine2D_68768, *[], **kwargs_68769)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 27), dict_68755, (Line2D_68766, HandlerLine2D_call_result_68770))
# Adding element type (key, value) (line 490)
# Getting the type of 'Patch' (line 494)
Patch_68771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'Patch')

# Call to HandlerPatch(...): (line 494)
# Processing the call keyword arguments (line 494)
kwargs_68774 = {}
# Getting the type of 'legend_handler' (line 494)
legend_handler_68772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 15), 'legend_handler', False)
# Obtaining the member 'HandlerPatch' of a type (line 494)
HandlerPatch_68773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 15), legend_handler_68772, 'HandlerPatch')
# Calling HandlerPatch(args, kwargs) (line 494)
HandlerPatch_call_result_68775 = invoke(stypy.reporting.localization.Localization(__file__, 494, 15), HandlerPatch_68773, *[], **kwargs_68774)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 27), dict_68755, (Patch_68771, HandlerPatch_call_result_68775))
# Adding element type (key, value) (line 490)
# Getting the type of 'LineCollection' (line 495)
LineCollection_68776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'LineCollection')

# Call to HandlerLineCollection(...): (line 495)
# Processing the call keyword arguments (line 495)
kwargs_68779 = {}
# Getting the type of 'legend_handler' (line 495)
legend_handler_68777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 24), 'legend_handler', False)
# Obtaining the member 'HandlerLineCollection' of a type (line 495)
HandlerLineCollection_68778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 24), legend_handler_68777, 'HandlerLineCollection')
# Calling HandlerLineCollection(args, kwargs) (line 495)
HandlerLineCollection_call_result_68780 = invoke(stypy.reporting.localization.Localization(__file__, 495, 24), HandlerLineCollection_68778, *[], **kwargs_68779)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 27), dict_68755, (LineCollection_68776, HandlerLineCollection_call_result_68780))
# Adding element type (key, value) (line 490)
# Getting the type of 'RegularPolyCollection' (line 496)
RegularPolyCollection_68781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'RegularPolyCollection')

# Call to HandlerRegularPolyCollection(...): (line 496)
# Processing the call keyword arguments (line 496)
kwargs_68784 = {}
# Getting the type of 'legend_handler' (line 496)
legend_handler_68782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 31), 'legend_handler', False)
# Obtaining the member 'HandlerRegularPolyCollection' of a type (line 496)
HandlerRegularPolyCollection_68783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 31), legend_handler_68782, 'HandlerRegularPolyCollection')
# Calling HandlerRegularPolyCollection(args, kwargs) (line 496)
HandlerRegularPolyCollection_call_result_68785 = invoke(stypy.reporting.localization.Localization(__file__, 496, 31), HandlerRegularPolyCollection_68783, *[], **kwargs_68784)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 27), dict_68755, (RegularPolyCollection_68781, HandlerRegularPolyCollection_call_result_68785))
# Adding element type (key, value) (line 490)
# Getting the type of 'CircleCollection' (line 497)
CircleCollection_68786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'CircleCollection')

# Call to HandlerCircleCollection(...): (line 497)
# Processing the call keyword arguments (line 497)
kwargs_68789 = {}
# Getting the type of 'legend_handler' (line 497)
legend_handler_68787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 26), 'legend_handler', False)
# Obtaining the member 'HandlerCircleCollection' of a type (line 497)
HandlerCircleCollection_68788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 26), legend_handler_68787, 'HandlerCircleCollection')
# Calling HandlerCircleCollection(args, kwargs) (line 497)
HandlerCircleCollection_call_result_68790 = invoke(stypy.reporting.localization.Localization(__file__, 497, 26), HandlerCircleCollection_68788, *[], **kwargs_68789)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 27), dict_68755, (CircleCollection_68786, HandlerCircleCollection_call_result_68790))
# Adding element type (key, value) (line 490)
# Getting the type of 'BarContainer' (line 498)
BarContainer_68791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'BarContainer')

# Call to HandlerPatch(...): (line 498)
# Processing the call keyword arguments (line 498)
# Getting the type of 'legend_handler' (line 499)
legend_handler_68794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 24), 'legend_handler', False)
# Obtaining the member 'update_from_first_child' of a type (line 499)
update_from_first_child_68795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 24), legend_handler_68794, 'update_from_first_child')
keyword_68796 = update_from_first_child_68795
kwargs_68797 = {'update_func': keyword_68796}
# Getting the type of 'legend_handler' (line 498)
legend_handler_68792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 22), 'legend_handler', False)
# Obtaining the member 'HandlerPatch' of a type (line 498)
HandlerPatch_68793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 22), legend_handler_68792, 'HandlerPatch')
# Calling HandlerPatch(args, kwargs) (line 498)
HandlerPatch_call_result_68798 = invoke(stypy.reporting.localization.Localization(__file__, 498, 22), HandlerPatch_68793, *[], **kwargs_68797)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 27), dict_68755, (BarContainer_68791, HandlerPatch_call_result_68798))
# Adding element type (key, value) (line 490)
# Getting the type of 'tuple' (line 500)
tuple_68799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'tuple')

# Call to HandlerTuple(...): (line 500)
# Processing the call keyword arguments (line 500)
kwargs_68802 = {}
# Getting the type of 'legend_handler' (line 500)
legend_handler_68800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 15), 'legend_handler', False)
# Obtaining the member 'HandlerTuple' of a type (line 500)
HandlerTuple_68801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 15), legend_handler_68800, 'HandlerTuple')
# Calling HandlerTuple(args, kwargs) (line 500)
HandlerTuple_call_result_68803 = invoke(stypy.reporting.localization.Localization(__file__, 500, 15), HandlerTuple_68801, *[], **kwargs_68802)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 27), dict_68755, (tuple_68799, HandlerTuple_call_result_68803))
# Adding element type (key, value) (line 490)
# Getting the type of 'PathCollection' (line 501)
PathCollection_68804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'PathCollection')

# Call to HandlerPathCollection(...): (line 501)
# Processing the call keyword arguments (line 501)
kwargs_68807 = {}
# Getting the type of 'legend_handler' (line 501)
legend_handler_68805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 24), 'legend_handler', False)
# Obtaining the member 'HandlerPathCollection' of a type (line 501)
HandlerPathCollection_68806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 24), legend_handler_68805, 'HandlerPathCollection')
# Calling HandlerPathCollection(args, kwargs) (line 501)
HandlerPathCollection_call_result_68808 = invoke(stypy.reporting.localization.Localization(__file__, 501, 24), HandlerPathCollection_68806, *[], **kwargs_68807)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 27), dict_68755, (PathCollection_68804, HandlerPathCollection_call_result_68808))
# Adding element type (key, value) (line 490)
# Getting the type of 'PolyCollection' (line 502)
PolyCollection_68809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'PolyCollection')

# Call to HandlerPolyCollection(...): (line 502)
# Processing the call keyword arguments (line 502)
kwargs_68812 = {}
# Getting the type of 'legend_handler' (line 502)
legend_handler_68810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 24), 'legend_handler', False)
# Obtaining the member 'HandlerPolyCollection' of a type (line 502)
HandlerPolyCollection_68811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 24), legend_handler_68810, 'HandlerPolyCollection')
# Calling HandlerPolyCollection(args, kwargs) (line 502)
HandlerPolyCollection_call_result_68813 = invoke(stypy.reporting.localization.Localization(__file__, 502, 24), HandlerPolyCollection_68811, *[], **kwargs_68812)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 27), dict_68755, (PolyCollection_68809, HandlerPolyCollection_call_result_68813))

# Getting the type of 'Legend'
Legend_68814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Legend')
# Setting the type of the member '_default_handler_map' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Legend_68814, '_default_handler_map', dict_68755)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
