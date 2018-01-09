
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This module defines default legend handlers.
3: 
4: It is strongly encouraged to have read the :ref:`legend guide
5: <sphx_glr_tutorials_intermediate_legend_guide.py>` before this documentation.
6: 
7: Legend handlers are expected to be a callable object with a following
8: signature. ::
9: 
10:     legend_handler(legend, orig_handle, fontsize, handlebox)
11: 
12: Where *legend* is the legend itself, *orig_handle* is the original
13: plot, *fontsize* is the fontsize in pixles, and *handlebox* is a
14: OffsetBox instance. Within the call, you should create relevant
15: artists (using relevant properties from the *legend* and/or
16: *orig_handle*) and add them into the handlebox. The artists needs to
17: be scaled according to the fontsize (note that the size is in pixel,
18: i.e., this is dpi-scaled value).
19: 
20: This module includes definition of several legend handler classes
21: derived from the base class (HandlerBase) with the following method.
22: 
23:     def legend_artist(self, legend, orig_handle, fontsize, handlebox):
24: 
25: 
26: '''
27: from __future__ import (absolute_import, division, print_function,
28:                         unicode_literals)
29: 
30: import six
31: from six.moves import zip
32: from itertools import cycle
33: 
34: import numpy as np
35: 
36: from matplotlib.lines import Line2D
37: from matplotlib.patches import Rectangle
38: import matplotlib.collections as mcoll
39: import matplotlib.colors as mcolors
40: 
41: 
42: def update_from_first_child(tgt, src):
43:     tgt.update_from(src.get_children()[0])
44: 
45: 
46: class HandlerBase(object):
47:     '''
48:     A Base class for default legend handlers.
49: 
50:     The derived classes are meant to override *create_artists* method, which
51:     has a following signature.::
52: 
53:       def create_artists(self, legend, orig_handle,
54:                          xdescent, ydescent, width, height, fontsize,
55:                          trans):
56: 
57:     The overridden method needs to create artists of the given
58:     transform that fits in the given dimension (xdescent, ydescent,
59:     width, height) that are scaled by fontsize if necessary.
60: 
61:     '''
62:     def __init__(self, xpad=0., ypad=0., update_func=None):
63:         self._xpad, self._ypad = xpad, ypad
64:         self._update_prop_func = update_func
65: 
66:     def _update_prop(self, legend_handle, orig_handle):
67:         if self._update_prop_func is None:
68:             self._default_update_prop(legend_handle, orig_handle)
69:         else:
70:             self._update_prop_func(legend_handle, orig_handle)
71: 
72:     def _default_update_prop(self, legend_handle, orig_handle):
73:         legend_handle.update_from(orig_handle)
74: 
75:     def update_prop(self, legend_handle, orig_handle, legend):
76: 
77:         self._update_prop(legend_handle, orig_handle)
78: 
79:         legend._set_artist_props(legend_handle)
80:         legend_handle.set_clip_box(None)
81:         legend_handle.set_clip_path(None)
82: 
83:     def adjust_drawing_area(self, legend, orig_handle,
84:                             xdescent, ydescent, width, height, fontsize,
85:                             ):
86:         xdescent = xdescent - self._xpad * fontsize
87:         ydescent = ydescent - self._ypad * fontsize
88:         width = width - self._xpad * fontsize
89:         height = height - self._ypad * fontsize
90:         return xdescent, ydescent, width, height
91: 
92:     def legend_artist(self, legend, orig_handle,
93:                        fontsize, handlebox):
94:         '''
95:         Return the artist that this HandlerBase generates for the given
96:         original artist/handle.
97: 
98:         Parameters
99:         ----------
100:         legend : :class:`matplotlib.legend.Legend` instance
101:             The legend for which these legend artists are being created.
102:         orig_handle : :class:`matplotlib.artist.Artist` or similar
103:             The object for which these legend artists are being created.
104:         fontsize : float or int
105:             The fontsize in pixels. The artists being created should
106:             be scaled according to the given fontsize.
107:         handlebox : :class:`matplotlib.offsetbox.OffsetBox` instance
108:             The box which has been created to hold this legend entry's
109:             artists. Artists created in the `legend_artist` method must
110:             be added to this handlebox inside this method.
111: 
112:         '''
113:         xdescent, ydescent, width, height = self.adjust_drawing_area(
114:                  legend, orig_handle,
115:                  handlebox.xdescent, handlebox.ydescent,
116:                  handlebox.width, handlebox.height,
117:                  fontsize)
118:         artists = self.create_artists(legend, orig_handle,
119:                                       xdescent, ydescent, width, height,
120:                                       fontsize, handlebox.get_transform())
121: 
122:         # create_artists will return a list of artists.
123:         for a in artists:
124:             handlebox.add_artist(a)
125: 
126:         # we only return the first artist
127:         return artists[0]
128: 
129:     def create_artists(self, legend, orig_handle,
130:                        xdescent, ydescent, width, height, fontsize,
131:                        trans):
132:         raise NotImplementedError('Derived must override')
133: 
134: 
135: class HandlerNpoints(HandlerBase):
136:     def __init__(self, marker_pad=0.3, numpoints=None, **kw):
137:         HandlerBase.__init__(self, **kw)
138: 
139:         self._numpoints = numpoints
140:         self._marker_pad = marker_pad
141: 
142:     def get_numpoints(self, legend):
143:         if self._numpoints is None:
144:             return legend.numpoints
145:         else:
146:             return self._numpoints
147: 
148:     def get_xdata(self, legend, xdescent, ydescent, width, height, fontsize):
149:         numpoints = self.get_numpoints(legend)
150:         if numpoints > 1:
151:             # we put some pad here to compensate the size of the marker
152:             pad = self._marker_pad * fontsize
153:             xdata = np.linspace(-xdescent + pad,
154:                                 -xdescent + width - pad,
155:                                 numpoints)
156:             xdata_marker = xdata
157:         else:
158:             xdata = np.linspace(-xdescent, -xdescent + width, 2)
159:             xdata_marker = [-xdescent + 0.5 * width]
160:         return xdata, xdata_marker
161: 
162: 
163: 
164: class HandlerNpointsYoffsets(HandlerNpoints):
165:     def __init__(self, numpoints=None, yoffsets=None, **kw):
166:         HandlerNpoints.__init__(self, numpoints=numpoints, **kw)
167:         self._yoffsets = yoffsets
168: 
169:     def get_ydata(self, legend, xdescent, ydescent, width, height, fontsize):
170:         if self._yoffsets is None:
171:             ydata = height * legend._scatteryoffsets
172:         else:
173:             ydata = height * np.asarray(self._yoffsets)
174: 
175:         return ydata
176: 
177: 
178: class HandlerLine2D(HandlerNpoints):
179:     '''
180:     Handler for Line2D instances.
181:     '''
182:     def __init__(self, marker_pad=0.3, numpoints=None, **kw):
183:         HandlerNpoints.__init__(self, marker_pad=marker_pad, numpoints=numpoints, **kw)
184: 
185:     def create_artists(self, legend, orig_handle,
186:                        xdescent, ydescent, width, height, fontsize,
187:                        trans):
188: 
189:         xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
190:                                              width, height, fontsize)
191: 
192:         ydata = ((height - ydescent) / 2.) * np.ones(xdata.shape, float)
193:         legline = Line2D(xdata, ydata)
194: 
195:         self.update_prop(legline, orig_handle, legend)
196:         legline.set_drawstyle('default')
197:         legline.set_marker("")
198: 
199:         legline_marker = Line2D(xdata_marker, ydata[:len(xdata_marker)])
200:         self.update_prop(legline_marker, orig_handle, legend)
201:         legline_marker.set_linestyle('None')
202:         if legend.markerscale != 1:
203:             newsz = legline_marker.get_markersize() * legend.markerscale
204:             legline_marker.set_markersize(newsz)
205:         # we don't want to add this to the return list because
206:         # the texts and handles are assumed to be in one-to-one
207:         # correspondence.
208:         legline._legmarker = legline_marker
209: 
210:         legline.set_transform(trans)
211:         legline_marker.set_transform(trans)
212: 
213:         return [legline, legline_marker]
214: 
215: 
216: class HandlerPatch(HandlerBase):
217:     '''
218:     Handler for Patch instances.
219:     '''
220:     def __init__(self, patch_func=None, **kw):
221:         '''
222:         The HandlerPatch class optionally takes a function ``patch_func``
223:         who's responsibility is to create the legend key artist. The
224:         ``patch_func`` should have the signature::
225: 
226:             def patch_func(legend=legend, orig_handle=orig_handle,
227:                            xdescent=xdescent, ydescent=ydescent,
228:                            width=width, height=height, fontsize=fontsize)
229: 
230:         Subsequently the created artist will have its ``update_prop`` method
231:         called and the appropriate transform will be applied.
232: 
233:         '''
234:         HandlerBase.__init__(self, **kw)
235:         self._patch_func = patch_func
236: 
237:     def _create_patch(self, legend, orig_handle,
238:                       xdescent, ydescent, width, height, fontsize):
239:         if self._patch_func is None:
240:             p = Rectangle(xy=(-xdescent, -ydescent),
241:                           width=width, height=height)
242:         else:
243:             p = self._patch_func(legend=legend, orig_handle=orig_handle,
244:                                  xdescent=xdescent, ydescent=ydescent,
245:                                  width=width, height=height, fontsize=fontsize)
246:         return p
247: 
248:     def create_artists(self, legend, orig_handle,
249:                        xdescent, ydescent, width, height, fontsize, trans):
250:         p = self._create_patch(legend, orig_handle,
251:                                xdescent, ydescent, width, height, fontsize)
252:         self.update_prop(p, orig_handle, legend)
253:         p.set_transform(trans)
254:         return [p]
255: 
256: 
257: class HandlerLineCollection(HandlerLine2D):
258:     '''
259:     Handler for LineCollection instances.
260:     '''
261:     def get_numpoints(self, legend):
262:         if self._numpoints is None:
263:             return legend.scatterpoints
264:         else:
265:             return self._numpoints
266: 
267:     def _default_update_prop(self, legend_handle, orig_handle):
268:         lw = orig_handle.get_linewidths()[0]
269:         dashes = orig_handle._us_linestyles[0]
270:         color = orig_handle.get_colors()[0]
271:         legend_handle.set_color(color)
272:         legend_handle.set_linestyle(dashes)
273:         legend_handle.set_linewidth(lw)
274: 
275:     def create_artists(self, legend, orig_handle,
276:                        xdescent, ydescent, width, height, fontsize, trans):
277: 
278:         xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
279:                                              width, height, fontsize)
280:         ydata = ((height - ydescent) / 2.) * np.ones(xdata.shape, float)
281:         legline = Line2D(xdata, ydata)
282: 
283:         self.update_prop(legline, orig_handle, legend)
284:         legline.set_transform(trans)
285: 
286:         return [legline]
287: 
288: 
289: class HandlerRegularPolyCollection(HandlerNpointsYoffsets):
290:     '''
291:     Handler for RegularPolyCollections.
292:     '''
293:     def __init__(self, yoffsets=None, sizes=None, **kw):
294:         HandlerNpointsYoffsets.__init__(self, yoffsets=yoffsets, **kw)
295: 
296:         self._sizes = sizes
297: 
298:     def get_numpoints(self, legend):
299:         if self._numpoints is None:
300:             return legend.scatterpoints
301:         else:
302:             return self._numpoints
303: 
304:     def get_sizes(self, legend, orig_handle,
305:                  xdescent, ydescent, width, height, fontsize):
306:         if self._sizes is None:
307:             handle_sizes = orig_handle.get_sizes()
308:             if not len(handle_sizes):
309:                 handle_sizes = [1]
310:             size_max = max(handle_sizes) * legend.markerscale ** 2
311:             size_min = min(handle_sizes) * legend.markerscale ** 2
312: 
313:             numpoints = self.get_numpoints(legend)
314:             if numpoints < 4:
315:                 sizes = [.5 * (size_max + size_min), size_max,
316:                          size_min][:numpoints]
317:             else:
318:                 rng = (size_max - size_min)
319:                 sizes = rng * np.linspace(0, 1, numpoints) + size_min
320:         else:
321:             sizes = self._sizes
322: 
323:         return sizes
324: 
325:     def update_prop(self, legend_handle, orig_handle, legend):
326: 
327:         self._update_prop(legend_handle, orig_handle)
328: 
329:         legend_handle.set_figure(legend.figure)
330:         #legend._set_artist_props(legend_handle)
331:         legend_handle.set_clip_box(None)
332:         legend_handle.set_clip_path(None)
333: 
334:     def create_collection(self, orig_handle, sizes, offsets, transOffset):
335:         p = type(orig_handle)(orig_handle.get_numsides(),
336:                               rotation=orig_handle.get_rotation(),
337:                               sizes=sizes,
338:                               offsets=offsets,
339:                               transOffset=transOffset,
340:                               )
341:         return p
342: 
343:     def create_artists(self, legend, orig_handle,
344:                        xdescent, ydescent, width, height, fontsize,
345:                        trans):
346:         xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
347:                                              width, height, fontsize)
348: 
349:         ydata = self.get_ydata(legend, xdescent, ydescent,
350:                                width, height, fontsize)
351: 
352:         sizes = self.get_sizes(legend, orig_handle, xdescent, ydescent,
353:                                width, height, fontsize)
354: 
355:         p = self.create_collection(orig_handle, sizes,
356:                                    offsets=list(zip(xdata_marker, ydata)),
357:                                    transOffset=trans)
358: 
359:         self.update_prop(p, orig_handle, legend)
360:         p._transOffset = trans
361:         return [p]
362: 
363: 
364: class HandlerPathCollection(HandlerRegularPolyCollection):
365:     '''
366:     Handler for PathCollections, which are used by scatter
367:     '''
368:     def create_collection(self, orig_handle, sizes, offsets, transOffset):
369:         p = type(orig_handle)([orig_handle.get_paths()[0]],
370:                               sizes=sizes,
371:                               offsets=offsets,
372:                               transOffset=transOffset,
373:                               )
374:         return p
375: 
376: 
377: class HandlerCircleCollection(HandlerRegularPolyCollection):
378:     '''
379:     Handler for CircleCollections
380:     '''
381:     def create_collection(self, orig_handle, sizes, offsets, transOffset):
382:         p = type(orig_handle)(sizes,
383:                               offsets=offsets,
384:                               transOffset=transOffset,
385:                               )
386:         return p
387: 
388: 
389: class HandlerErrorbar(HandlerLine2D):
390:     '''
391:     Handler for Errorbars
392:     '''
393:     def __init__(self, xerr_size=0.5, yerr_size=None,
394:                  marker_pad=0.3, numpoints=None, **kw):
395: 
396:         self._xerr_size = xerr_size
397:         self._yerr_size = yerr_size
398: 
399:         HandlerLine2D.__init__(self, marker_pad=marker_pad, numpoints=numpoints,
400:                                **kw)
401: 
402:     def get_err_size(self, legend, xdescent, ydescent, width, height, fontsize):
403:         xerr_size = self._xerr_size * fontsize
404: 
405:         if self._yerr_size is None:
406:             yerr_size = xerr_size
407:         else:
408:             yerr_size = self._yerr_size * fontsize
409: 
410:         return xerr_size, yerr_size
411: 
412:     def create_artists(self, legend, orig_handle,
413:                        xdescent, ydescent, width, height, fontsize,
414:                        trans):
415: 
416:         plotlines, caplines, barlinecols = orig_handle
417: 
418:         xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
419:                                              width, height, fontsize)
420: 
421:         ydata = ((height - ydescent) / 2.) * np.ones(xdata.shape, float)
422:         legline = Line2D(xdata, ydata)
423: 
424: 
425:         xdata_marker = np.asarray(xdata_marker)
426:         ydata_marker = np.asarray(ydata[:len(xdata_marker)])
427: 
428:         xerr_size, yerr_size = self.get_err_size(legend, xdescent, ydescent,
429:                                                  width, height, fontsize)
430: 
431:         legline_marker = Line2D(xdata_marker, ydata_marker)
432: 
433:         # when plotlines are None (only errorbars are drawn), we just
434:         # make legline invisible.
435:         if plotlines is None:
436:             legline.set_visible(False)
437:             legline_marker.set_visible(False)
438:         else:
439:             self.update_prop(legline, plotlines, legend)
440: 
441:             legline.set_drawstyle('default')
442:             legline.set_marker('None')
443: 
444:             self.update_prop(legline_marker, plotlines, legend)
445:             legline_marker.set_linestyle('None')
446: 
447:             if legend.markerscale != 1:
448:                 newsz = legline_marker.get_markersize() * legend.markerscale
449:                 legline_marker.set_markersize(newsz)
450: 
451:         handle_barlinecols = []
452:         handle_caplines = []
453: 
454:         if orig_handle.has_xerr:
455:             verts = [ ((x - xerr_size, y), (x + xerr_size, y))
456:                       for x, y in zip(xdata_marker, ydata_marker)]
457:             coll = mcoll.LineCollection(verts)
458:             self.update_prop(coll, barlinecols[0], legend)
459:             handle_barlinecols.append(coll)
460: 
461:             if caplines:
462:                 capline_left = Line2D(xdata_marker - xerr_size, ydata_marker)
463:                 capline_right = Line2D(xdata_marker + xerr_size, ydata_marker)
464:                 self.update_prop(capline_left, caplines[0], legend)
465:                 self.update_prop(capline_right, caplines[0], legend)
466:                 capline_left.set_marker("|")
467:                 capline_right.set_marker("|")
468: 
469:                 handle_caplines.append(capline_left)
470:                 handle_caplines.append(capline_right)
471: 
472:         if orig_handle.has_yerr:
473:             verts = [ ((x, y - yerr_size), (x, y + yerr_size))
474:                       for x, y in zip(xdata_marker, ydata_marker)]
475:             coll = mcoll.LineCollection(verts)
476:             self.update_prop(coll, barlinecols[0], legend)
477:             handle_barlinecols.append(coll)
478: 
479:             if caplines:
480:                 capline_left = Line2D(xdata_marker, ydata_marker - yerr_size)
481:                 capline_right = Line2D(xdata_marker, ydata_marker + yerr_size)
482:                 self.update_prop(capline_left, caplines[0], legend)
483:                 self.update_prop(capline_right, caplines[0], legend)
484:                 capline_left.set_marker("_")
485:                 capline_right.set_marker("_")
486: 
487:                 handle_caplines.append(capline_left)
488:                 handle_caplines.append(capline_right)
489: 
490:         artists = []
491:         artists.extend(handle_barlinecols)
492:         artists.extend(handle_caplines)
493:         artists.append(legline)
494:         artists.append(legline_marker)
495: 
496:         for artist in artists:
497:             artist.set_transform(trans)
498: 
499:         return artists
500: 
501: 
502: class HandlerStem(HandlerNpointsYoffsets):
503:     '''
504:     Handler for Errorbars
505:     '''
506:     def __init__(self, marker_pad=0.3, numpoints=None,
507:                  bottom=None, yoffsets=None, **kw):
508: 
509:         HandlerNpointsYoffsets.__init__(self, marker_pad=marker_pad,
510:                                         numpoints=numpoints,
511:                                         yoffsets=yoffsets,
512:                                         **kw)
513:         self._bottom = bottom
514: 
515:     def get_ydata(self, legend, xdescent, ydescent, width, height, fontsize):
516:         if self._yoffsets is None:
517:             ydata = height * (0.5 * legend._scatteryoffsets + 0.5)
518:         else:
519:             ydata = height * np.asarray(self._yoffsets)
520: 
521:         return ydata
522: 
523:     def create_artists(self, legend, orig_handle,
524:                        xdescent, ydescent, width, height, fontsize,
525:                        trans):
526: 
527:         markerline, stemlines, baseline = orig_handle
528: 
529:         xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
530:                                              width, height, fontsize)
531: 
532:         ydata = self.get_ydata(legend, xdescent, ydescent,
533:                                width, height, fontsize)
534: 
535:         if self._bottom is None:
536:             bottom = 0.
537:         else:
538:             bottom = self._bottom
539: 
540:         leg_markerline = Line2D(xdata_marker, ydata[:len(xdata_marker)])
541:         self.update_prop(leg_markerline, markerline, legend)
542: 
543:         leg_stemlines = []
544:         for thisx, thisy in zip(xdata_marker, ydata):
545:             l = Line2D([thisx, thisx], [bottom, thisy])
546:             leg_stemlines.append(l)
547: 
548:         for lm, m in zip(leg_stemlines, stemlines):
549:             self.update_prop(lm, m, legend)
550: 
551:         leg_baseline = Line2D([np.min(xdata), np.max(xdata)],
552:                               [bottom, bottom])
553: 
554:         self.update_prop(leg_baseline, baseline, legend)
555: 
556:         artists = [leg_markerline]
557:         artists.extend(leg_stemlines)
558:         artists.append(leg_baseline)
559: 
560:         for artist in artists:
561:             artist.set_transform(trans)
562: 
563:         return artists
564: 
565: 
566: class HandlerTuple(HandlerBase):
567:     '''
568:     Handler for Tuple.
569: 
570:     Additional kwargs are passed through to `HandlerBase`.
571: 
572:     Parameters
573:     ----------
574: 
575:     ndivide : int, optional
576:         The number of sections to divide the legend area into.  If None,
577:         use the length of the input tuple. Default is 1.
578: 
579: 
580:     pad : float, optional
581:         If None, fall back to `legend.borderpad` as the default.
582:         In units of fraction of font size. Default is None.
583: 
584: 
585: 
586:     '''
587:     def __init__(self, ndivide=1, pad=None, **kwargs):
588: 
589:         self._ndivide = ndivide
590:         self._pad = pad
591:         HandlerBase.__init__(self, **kwargs)
592: 
593:     def create_artists(self, legend, orig_handle,
594:                        xdescent, ydescent, width, height, fontsize,
595:                        trans):
596: 
597:         handler_map = legend.get_legend_handler_map()
598: 
599:         if self._ndivide is None:
600:             ndivide = len(orig_handle)
601:         else:
602:             ndivide = self._ndivide
603: 
604:         if self._pad is None:
605:             pad = legend.borderpad * fontsize
606:         else:
607:             pad = self._pad * fontsize
608: 
609:         if ndivide > 1:
610:             width = (width - pad*(ndivide - 1)) / ndivide
611: 
612:         xds = [xdescent - (width + pad) * i for i in range(ndivide)]
613:         xds_cycle = cycle(xds)
614: 
615:         a_list = []
616:         for handle1 in orig_handle:
617:             handler = legend.get_legend_handler(handler_map, handle1)
618:             _a_list = handler.create_artists(legend, handle1,
619:                                              six.next(xds_cycle),
620:                                              ydescent,
621:                                              width, height,
622:                                              fontsize,
623:                                              trans)
624:             a_list.extend(_a_list)
625: 
626:         return a_list
627: 
628: 
629: class HandlerPolyCollection(HandlerBase):
630:     '''
631:     Handler for PolyCollection used in fill_between and stackplot.
632:     '''
633:     def _update_prop(self, legend_handle, orig_handle):
634:         def first_color(colors):
635:             if colors is None:
636:                 return None
637:             colors = mcolors.to_rgba_array(colors)
638:             if len(colors):
639:                 return colors[0]
640:             else:
641:                 return "none"
642:         def get_first(prop_array):
643:             if len(prop_array):
644:                 return prop_array[0]
645:             else:
646:                 return None
647:         edgecolor = getattr(orig_handle, '_original_edgecolor',
648:                             orig_handle.get_edgecolor())
649:         legend_handle.set_edgecolor(first_color(edgecolor))
650:         facecolor = getattr(orig_handle, '_original_facecolor',
651:                             orig_handle.get_facecolor())
652:         legend_handle.set_facecolor(first_color(facecolor))
653:         legend_handle.set_fill(orig_handle.get_fill())
654:         legend_handle.set_hatch(orig_handle.get_hatch())
655:         legend_handle.set_linewidth(get_first(orig_handle.get_linewidths()))
656:         legend_handle.set_linestyle(get_first(orig_handle.get_linestyles()))
657:         legend_handle.set_transform(get_first(orig_handle.get_transforms()))
658:         legend_handle.set_figure(orig_handle.get_figure())
659:         legend_handle.set_alpha(orig_handle.get_alpha())
660: 
661:     def create_artists(self, legend, orig_handle,
662:                        xdescent, ydescent, width, height, fontsize, trans):
663:         p = Rectangle(xy=(-xdescent, -ydescent),
664:                       width=width, height=height)
665:         self.update_prop(p, orig_handle, legend)
666:         p.set_transform(trans)
667:         return [p]
668: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_68846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'unicode', u'\nThis module defines default legend handlers.\n\nIt is strongly encouraged to have read the :ref:`legend guide\n<sphx_glr_tutorials_intermediate_legend_guide.py>` before this documentation.\n\nLegend handlers are expected to be a callable object with a following\nsignature. ::\n\n    legend_handler(legend, orig_handle, fontsize, handlebox)\n\nWhere *legend* is the legend itself, *orig_handle* is the original\nplot, *fontsize* is the fontsize in pixles, and *handlebox* is a\nOffsetBox instance. Within the call, you should create relevant\nartists (using relevant properties from the *legend* and/or\n*orig_handle*) and add them into the handlebox. The artists needs to\nbe scaled according to the fontsize (note that the size is in pixel,\ni.e., this is dpi-scaled value).\n\nThis module includes definition of several legend handler classes\nderived from the base class (HandlerBase) with the following method.\n\n    def legend_artist(self, legend, orig_handle, fontsize, handlebox):\n\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'import six' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_68847 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'six')

if (type(import_68847) is not StypyTypeError):

    if (import_68847 != 'pyd_module'):
        __import__(import_68847)
        sys_modules_68848 = sys.modules[import_68847]
        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'six', sys_modules_68848.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'six', import_68847)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from six.moves import zip' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_68849 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'six.moves')

if (type(import_68849) is not StypyTypeError):

    if (import_68849 != 'pyd_module'):
        __import__(import_68849)
        sys_modules_68850 = sys.modules[import_68849]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'six.moves', sys_modules_68850.module_type_store, module_type_store, ['zip'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_68850, sys_modules_68850.module_type_store, module_type_store)
    else:
        from six.moves import zip

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'six.moves', None, module_type_store, ['zip'], [zip])

else:
    # Assigning a type to the variable 'six.moves' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'six.moves', import_68849)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from itertools import cycle' statement (line 32)
try:
    from itertools import cycle

except:
    cycle = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'itertools', None, module_type_store, ['cycle'], [cycle])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'import numpy' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_68851 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy')

if (type(import_68851) is not StypyTypeError):

    if (import_68851 != 'pyd_module'):
        __import__(import_68851)
        sys_modules_68852 = sys.modules[import_68851]
        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'np', sys_modules_68852.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy', import_68851)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from matplotlib.lines import Line2D' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_68853 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.lines')

if (type(import_68853) is not StypyTypeError):

    if (import_68853 != 'pyd_module'):
        __import__(import_68853)
        sys_modules_68854 = sys.modules[import_68853]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.lines', sys_modules_68854.module_type_store, module_type_store, ['Line2D'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_68854, sys_modules_68854.module_type_store, module_type_store)
    else:
        from matplotlib.lines import Line2D

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.lines', None, module_type_store, ['Line2D'], [Line2D])

else:
    # Assigning a type to the variable 'matplotlib.lines' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.lines', import_68853)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from matplotlib.patches import Rectangle' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_68855 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.patches')

if (type(import_68855) is not StypyTypeError):

    if (import_68855 != 'pyd_module'):
        __import__(import_68855)
        sys_modules_68856 = sys.modules[import_68855]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.patches', sys_modules_68856.module_type_store, module_type_store, ['Rectangle'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_68856, sys_modules_68856.module_type_store, module_type_store)
    else:
        from matplotlib.patches import Rectangle

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.patches', None, module_type_store, ['Rectangle'], [Rectangle])

else:
    # Assigning a type to the variable 'matplotlib.patches' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.patches', import_68855)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'import matplotlib.collections' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_68857 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'matplotlib.collections')

if (type(import_68857) is not StypyTypeError):

    if (import_68857 != 'pyd_module'):
        __import__(import_68857)
        sys_modules_68858 = sys.modules[import_68857]
        import_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'mcoll', sys_modules_68858.module_type_store, module_type_store)
    else:
        import matplotlib.collections as mcoll

        import_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'mcoll', matplotlib.collections, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.collections' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'matplotlib.collections', import_68857)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'import matplotlib.colors' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_68859 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.colors')

if (type(import_68859) is not StypyTypeError):

    if (import_68859 != 'pyd_module'):
        __import__(import_68859)
        sys_modules_68860 = sys.modules[import_68859]
        import_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'mcolors', sys_modules_68860.module_type_store, module_type_store)
    else:
        import matplotlib.colors as mcolors

        import_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'mcolors', matplotlib.colors, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.colors' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib.colors', import_68859)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


@norecursion
def update_from_first_child(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'update_from_first_child'
    module_type_store = module_type_store.open_function_context('update_from_first_child', 42, 0, False)
    
    # Passed parameters checking function
    update_from_first_child.stypy_localization = localization
    update_from_first_child.stypy_type_of_self = None
    update_from_first_child.stypy_type_store = module_type_store
    update_from_first_child.stypy_function_name = 'update_from_first_child'
    update_from_first_child.stypy_param_names_list = ['tgt', 'src']
    update_from_first_child.stypy_varargs_param_name = None
    update_from_first_child.stypy_kwargs_param_name = None
    update_from_first_child.stypy_call_defaults = defaults
    update_from_first_child.stypy_call_varargs = varargs
    update_from_first_child.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'update_from_first_child', ['tgt', 'src'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'update_from_first_child', localization, ['tgt', 'src'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'update_from_first_child(...)' code ##################

    
    # Call to update_from(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Obtaining the type of the subscript
    int_68863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 39), 'int')
    
    # Call to get_children(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_68866 = {}
    # Getting the type of 'src' (line 43)
    src_68864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'src', False)
    # Obtaining the member 'get_children' of a type (line 43)
    get_children_68865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 20), src_68864, 'get_children')
    # Calling get_children(args, kwargs) (line 43)
    get_children_call_result_68867 = invoke(stypy.reporting.localization.Localization(__file__, 43, 20), get_children_68865, *[], **kwargs_68866)
    
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___68868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 20), get_children_call_result_68867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_68869 = invoke(stypy.reporting.localization.Localization(__file__, 43, 20), getitem___68868, int_68863)
    
    # Processing the call keyword arguments (line 43)
    kwargs_68870 = {}
    # Getting the type of 'tgt' (line 43)
    tgt_68861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'tgt', False)
    # Obtaining the member 'update_from' of a type (line 43)
    update_from_68862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), tgt_68861, 'update_from')
    # Calling update_from(args, kwargs) (line 43)
    update_from_call_result_68871 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), update_from_68862, *[subscript_call_result_68869], **kwargs_68870)
    
    
    # ################# End of 'update_from_first_child(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'update_from_first_child' in the type store
    # Getting the type of 'stypy_return_type' (line 42)
    stypy_return_type_68872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68872)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'update_from_first_child'
    return stypy_return_type_68872

# Assigning a type to the variable 'update_from_first_child' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'update_from_first_child', update_from_first_child)
# Declaration of the 'HandlerBase' class

class HandlerBase(object, ):
    unicode_68873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'unicode', u'\n    A Base class for default legend handlers.\n\n    The derived classes are meant to override *create_artists* method, which\n    has a following signature.::\n\n      def create_artists(self, legend, orig_handle,\n                         xdescent, ydescent, width, height, fontsize,\n                         trans):\n\n    The overridden method needs to create artists of the given\n    transform that fits in the given dimension (xdescent, ydescent,\n    width, height) that are scaled by fontsize if necessary.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_68874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'float')
        float_68875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 37), 'float')
        # Getting the type of 'None' (line 62)
        None_68876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 53), 'None')
        defaults = [float_68874, float_68875, None_68876]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerBase.__init__', ['xpad', 'ypad', 'update_func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['xpad', 'ypad', 'update_func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 63):
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'xpad' (line 63)
        xpad_68877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'xpad')
        # Assigning a type to the variable 'tuple_assignment_68815' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_assignment_68815', xpad_68877)
        
        # Assigning a Name to a Name (line 63):
        # Getting the type of 'ypad' (line 63)
        ypad_68878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 39), 'ypad')
        # Assigning a type to the variable 'tuple_assignment_68816' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_assignment_68816', ypad_68878)
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'tuple_assignment_68815' (line 63)
        tuple_assignment_68815_68879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_assignment_68815')
        # Getting the type of 'self' (line 63)
        self_68880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member '_xpad' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_68880, '_xpad', tuple_assignment_68815_68879)
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'tuple_assignment_68816' (line 63)
        tuple_assignment_68816_68881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_assignment_68816')
        # Getting the type of 'self' (line 63)
        self_68882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'self')
        # Setting the type of the member '_ypad' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 20), self_68882, '_ypad', tuple_assignment_68816_68881)
        
        # Assigning a Name to a Attribute (line 64):
        
        # Assigning a Name to a Attribute (line 64):
        # Getting the type of 'update_func' (line 64)
        update_func_68883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'update_func')
        # Getting the type of 'self' (line 64)
        self_68884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member '_update_prop_func' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_68884, '_update_prop_func', update_func_68883)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _update_prop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update_prop'
        module_type_store = module_type_store.open_function_context('_update_prop', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerBase._update_prop.__dict__.__setitem__('stypy_localization', localization)
        HandlerBase._update_prop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerBase._update_prop.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerBase._update_prop.__dict__.__setitem__('stypy_function_name', 'HandlerBase._update_prop')
        HandlerBase._update_prop.__dict__.__setitem__('stypy_param_names_list', ['legend_handle', 'orig_handle'])
        HandlerBase._update_prop.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerBase._update_prop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerBase._update_prop.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerBase._update_prop.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerBase._update_prop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerBase._update_prop.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerBase._update_prop', ['legend_handle', 'orig_handle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update_prop', localization, ['legend_handle', 'orig_handle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update_prop(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 67)
        # Getting the type of 'self' (line 67)
        self_68885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'self')
        # Obtaining the member '_update_prop_func' of a type (line 67)
        _update_prop_func_68886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 11), self_68885, '_update_prop_func')
        # Getting the type of 'None' (line 67)
        None_68887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 37), 'None')
        
        (may_be_68888, more_types_in_union_68889) = may_be_none(_update_prop_func_68886, None_68887)

        if may_be_68888:

            if more_types_in_union_68889:
                # Runtime conditional SSA (line 67)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _default_update_prop(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'legend_handle' (line 68)
            legend_handle_68892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 38), 'legend_handle', False)
            # Getting the type of 'orig_handle' (line 68)
            orig_handle_68893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 53), 'orig_handle', False)
            # Processing the call keyword arguments (line 68)
            kwargs_68894 = {}
            # Getting the type of 'self' (line 68)
            self_68890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'self', False)
            # Obtaining the member '_default_update_prop' of a type (line 68)
            _default_update_prop_68891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), self_68890, '_default_update_prop')
            # Calling _default_update_prop(args, kwargs) (line 68)
            _default_update_prop_call_result_68895 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), _default_update_prop_68891, *[legend_handle_68892, orig_handle_68893], **kwargs_68894)
            

            if more_types_in_union_68889:
                # Runtime conditional SSA for else branch (line 67)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_68888) or more_types_in_union_68889):
            
            # Call to _update_prop_func(...): (line 70)
            # Processing the call arguments (line 70)
            # Getting the type of 'legend_handle' (line 70)
            legend_handle_68898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'legend_handle', False)
            # Getting the type of 'orig_handle' (line 70)
            orig_handle_68899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 50), 'orig_handle', False)
            # Processing the call keyword arguments (line 70)
            kwargs_68900 = {}
            # Getting the type of 'self' (line 70)
            self_68896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'self', False)
            # Obtaining the member '_update_prop_func' of a type (line 70)
            _update_prop_func_68897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), self_68896, '_update_prop_func')
            # Calling _update_prop_func(args, kwargs) (line 70)
            _update_prop_func_call_result_68901 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), _update_prop_func_68897, *[legend_handle_68898, orig_handle_68899], **kwargs_68900)
            

            if (may_be_68888 and more_types_in_union_68889):
                # SSA join for if statement (line 67)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_update_prop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update_prop' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_68902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68902)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update_prop'
        return stypy_return_type_68902


    @norecursion
    def _default_update_prop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_default_update_prop'
        module_type_store = module_type_store.open_function_context('_default_update_prop', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerBase._default_update_prop.__dict__.__setitem__('stypy_localization', localization)
        HandlerBase._default_update_prop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerBase._default_update_prop.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerBase._default_update_prop.__dict__.__setitem__('stypy_function_name', 'HandlerBase._default_update_prop')
        HandlerBase._default_update_prop.__dict__.__setitem__('stypy_param_names_list', ['legend_handle', 'orig_handle'])
        HandlerBase._default_update_prop.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerBase._default_update_prop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerBase._default_update_prop.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerBase._default_update_prop.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerBase._default_update_prop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerBase._default_update_prop.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerBase._default_update_prop', ['legend_handle', 'orig_handle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_default_update_prop', localization, ['legend_handle', 'orig_handle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_default_update_prop(...)' code ##################

        
        # Call to update_from(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'orig_handle' (line 73)
        orig_handle_68905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 34), 'orig_handle', False)
        # Processing the call keyword arguments (line 73)
        kwargs_68906 = {}
        # Getting the type of 'legend_handle' (line 73)
        legend_handle_68903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'legend_handle', False)
        # Obtaining the member 'update_from' of a type (line 73)
        update_from_68904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), legend_handle_68903, 'update_from')
        # Calling update_from(args, kwargs) (line 73)
        update_from_call_result_68907 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), update_from_68904, *[orig_handle_68905], **kwargs_68906)
        
        
        # ################# End of '_default_update_prop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_default_update_prop' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_68908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68908)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_default_update_prop'
        return stypy_return_type_68908


    @norecursion
    def update_prop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_prop'
        module_type_store = module_type_store.open_function_context('update_prop', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerBase.update_prop.__dict__.__setitem__('stypy_localization', localization)
        HandlerBase.update_prop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerBase.update_prop.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerBase.update_prop.__dict__.__setitem__('stypy_function_name', 'HandlerBase.update_prop')
        HandlerBase.update_prop.__dict__.__setitem__('stypy_param_names_list', ['legend_handle', 'orig_handle', 'legend'])
        HandlerBase.update_prop.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerBase.update_prop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerBase.update_prop.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerBase.update_prop.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerBase.update_prop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerBase.update_prop.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerBase.update_prop', ['legend_handle', 'orig_handle', 'legend'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_prop', localization, ['legend_handle', 'orig_handle', 'legend'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_prop(...)' code ##################

        
        # Call to _update_prop(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'legend_handle' (line 77)
        legend_handle_68911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'legend_handle', False)
        # Getting the type of 'orig_handle' (line 77)
        orig_handle_68912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 41), 'orig_handle', False)
        # Processing the call keyword arguments (line 77)
        kwargs_68913 = {}
        # Getting the type of 'self' (line 77)
        self_68909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self', False)
        # Obtaining the member '_update_prop' of a type (line 77)
        _update_prop_68910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_68909, '_update_prop')
        # Calling _update_prop(args, kwargs) (line 77)
        _update_prop_call_result_68914 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), _update_prop_68910, *[legend_handle_68911, orig_handle_68912], **kwargs_68913)
        
        
        # Call to _set_artist_props(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'legend_handle' (line 79)
        legend_handle_68917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'legend_handle', False)
        # Processing the call keyword arguments (line 79)
        kwargs_68918 = {}
        # Getting the type of 'legend' (line 79)
        legend_68915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'legend', False)
        # Obtaining the member '_set_artist_props' of a type (line 79)
        _set_artist_props_68916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), legend_68915, '_set_artist_props')
        # Calling _set_artist_props(args, kwargs) (line 79)
        _set_artist_props_call_result_68919 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), _set_artist_props_68916, *[legend_handle_68917], **kwargs_68918)
        
        
        # Call to set_clip_box(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'None' (line 80)
        None_68922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'None', False)
        # Processing the call keyword arguments (line 80)
        kwargs_68923 = {}
        # Getting the type of 'legend_handle' (line 80)
        legend_handle_68920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'legend_handle', False)
        # Obtaining the member 'set_clip_box' of a type (line 80)
        set_clip_box_68921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), legend_handle_68920, 'set_clip_box')
        # Calling set_clip_box(args, kwargs) (line 80)
        set_clip_box_call_result_68924 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), set_clip_box_68921, *[None_68922], **kwargs_68923)
        
        
        # Call to set_clip_path(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'None' (line 81)
        None_68927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 36), 'None', False)
        # Processing the call keyword arguments (line 81)
        kwargs_68928 = {}
        # Getting the type of 'legend_handle' (line 81)
        legend_handle_68925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'legend_handle', False)
        # Obtaining the member 'set_clip_path' of a type (line 81)
        set_clip_path_68926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), legend_handle_68925, 'set_clip_path')
        # Calling set_clip_path(args, kwargs) (line 81)
        set_clip_path_call_result_68929 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), set_clip_path_68926, *[None_68927], **kwargs_68928)
        
        
        # ################# End of 'update_prop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_prop' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_68930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68930)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_prop'
        return stypy_return_type_68930


    @norecursion
    def adjust_drawing_area(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'adjust_drawing_area'
        module_type_store = module_type_store.open_function_context('adjust_drawing_area', 83, 4, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerBase.adjust_drawing_area.__dict__.__setitem__('stypy_localization', localization)
        HandlerBase.adjust_drawing_area.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerBase.adjust_drawing_area.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerBase.adjust_drawing_area.__dict__.__setitem__('stypy_function_name', 'HandlerBase.adjust_drawing_area')
        HandlerBase.adjust_drawing_area.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'])
        HandlerBase.adjust_drawing_area.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerBase.adjust_drawing_area.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerBase.adjust_drawing_area.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerBase.adjust_drawing_area.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerBase.adjust_drawing_area.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerBase.adjust_drawing_area.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerBase.adjust_drawing_area', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'adjust_drawing_area', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'adjust_drawing_area(...)' code ##################

        
        # Assigning a BinOp to a Name (line 86):
        
        # Assigning a BinOp to a Name (line 86):
        # Getting the type of 'xdescent' (line 86)
        xdescent_68931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'xdescent')
        # Getting the type of 'self' (line 86)
        self_68932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'self')
        # Obtaining the member '_xpad' of a type (line 86)
        _xpad_68933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 30), self_68932, '_xpad')
        # Getting the type of 'fontsize' (line 86)
        fontsize_68934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 43), 'fontsize')
        # Applying the binary operator '*' (line 86)
        result_mul_68935 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 30), '*', _xpad_68933, fontsize_68934)
        
        # Applying the binary operator '-' (line 86)
        result_sub_68936 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 19), '-', xdescent_68931, result_mul_68935)
        
        # Assigning a type to the variable 'xdescent' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'xdescent', result_sub_68936)
        
        # Assigning a BinOp to a Name (line 87):
        
        # Assigning a BinOp to a Name (line 87):
        # Getting the type of 'ydescent' (line 87)
        ydescent_68937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'ydescent')
        # Getting the type of 'self' (line 87)
        self_68938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), 'self')
        # Obtaining the member '_ypad' of a type (line 87)
        _ypad_68939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 30), self_68938, '_ypad')
        # Getting the type of 'fontsize' (line 87)
        fontsize_68940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 43), 'fontsize')
        # Applying the binary operator '*' (line 87)
        result_mul_68941 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 30), '*', _ypad_68939, fontsize_68940)
        
        # Applying the binary operator '-' (line 87)
        result_sub_68942 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 19), '-', ydescent_68937, result_mul_68941)
        
        # Assigning a type to the variable 'ydescent' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'ydescent', result_sub_68942)
        
        # Assigning a BinOp to a Name (line 88):
        
        # Assigning a BinOp to a Name (line 88):
        # Getting the type of 'width' (line 88)
        width_68943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'width')
        # Getting the type of 'self' (line 88)
        self_68944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'self')
        # Obtaining the member '_xpad' of a type (line 88)
        _xpad_68945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), self_68944, '_xpad')
        # Getting the type of 'fontsize' (line 88)
        fontsize_68946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 37), 'fontsize')
        # Applying the binary operator '*' (line 88)
        result_mul_68947 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 24), '*', _xpad_68945, fontsize_68946)
        
        # Applying the binary operator '-' (line 88)
        result_sub_68948 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 16), '-', width_68943, result_mul_68947)
        
        # Assigning a type to the variable 'width' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'width', result_sub_68948)
        
        # Assigning a BinOp to a Name (line 89):
        
        # Assigning a BinOp to a Name (line 89):
        # Getting the type of 'height' (line 89)
        height_68949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'height')
        # Getting the type of 'self' (line 89)
        self_68950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'self')
        # Obtaining the member '_ypad' of a type (line 89)
        _ypad_68951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 26), self_68950, '_ypad')
        # Getting the type of 'fontsize' (line 89)
        fontsize_68952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 39), 'fontsize')
        # Applying the binary operator '*' (line 89)
        result_mul_68953 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 26), '*', _ypad_68951, fontsize_68952)
        
        # Applying the binary operator '-' (line 89)
        result_sub_68954 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 17), '-', height_68949, result_mul_68953)
        
        # Assigning a type to the variable 'height' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'height', result_sub_68954)
        
        # Obtaining an instance of the builtin type 'tuple' (line 90)
        tuple_68955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 90)
        # Adding element type (line 90)
        # Getting the type of 'xdescent' (line 90)
        xdescent_68956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'xdescent')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), tuple_68955, xdescent_68956)
        # Adding element type (line 90)
        # Getting the type of 'ydescent' (line 90)
        ydescent_68957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'ydescent')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), tuple_68955, ydescent_68957)
        # Adding element type (line 90)
        # Getting the type of 'width' (line 90)
        width_68958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 35), 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), tuple_68955, width_68958)
        # Adding element type (line 90)
        # Getting the type of 'height' (line 90)
        height_68959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 42), 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), tuple_68955, height_68959)
        
        # Assigning a type to the variable 'stypy_return_type' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'stypy_return_type', tuple_68955)
        
        # ################# End of 'adjust_drawing_area(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'adjust_drawing_area' in the type store
        # Getting the type of 'stypy_return_type' (line 83)
        stypy_return_type_68960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68960)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'adjust_drawing_area'
        return stypy_return_type_68960


    @norecursion
    def legend_artist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'legend_artist'
        module_type_store = module_type_store.open_function_context('legend_artist', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerBase.legend_artist.__dict__.__setitem__('stypy_localization', localization)
        HandlerBase.legend_artist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerBase.legend_artist.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerBase.legend_artist.__dict__.__setitem__('stypy_function_name', 'HandlerBase.legend_artist')
        HandlerBase.legend_artist.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'fontsize', 'handlebox'])
        HandlerBase.legend_artist.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerBase.legend_artist.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerBase.legend_artist.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerBase.legend_artist.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerBase.legend_artist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerBase.legend_artist.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerBase.legend_artist', ['legend', 'orig_handle', 'fontsize', 'handlebox'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'legend_artist', localization, ['legend', 'orig_handle', 'fontsize', 'handlebox'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'legend_artist(...)' code ##################

        unicode_68961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, (-1)), 'unicode', u"\n        Return the artist that this HandlerBase generates for the given\n        original artist/handle.\n\n        Parameters\n        ----------\n        legend : :class:`matplotlib.legend.Legend` instance\n            The legend for which these legend artists are being created.\n        orig_handle : :class:`matplotlib.artist.Artist` or similar\n            The object for which these legend artists are being created.\n        fontsize : float or int\n            The fontsize in pixels. The artists being created should\n            be scaled according to the given fontsize.\n        handlebox : :class:`matplotlib.offsetbox.OffsetBox` instance\n            The box which has been created to hold this legend entry's\n            artists. Artists created in the `legend_artist` method must\n            be added to this handlebox inside this method.\n\n        ")
        
        # Assigning a Call to a Tuple (line 113):
        
        # Assigning a Call to a Name:
        
        # Call to adjust_drawing_area(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'legend' (line 114)
        legend_68964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'legend', False)
        # Getting the type of 'orig_handle' (line 114)
        orig_handle_68965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 25), 'orig_handle', False)
        # Getting the type of 'handlebox' (line 115)
        handlebox_68966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'handlebox', False)
        # Obtaining the member 'xdescent' of a type (line 115)
        xdescent_68967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 17), handlebox_68966, 'xdescent')
        # Getting the type of 'handlebox' (line 115)
        handlebox_68968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 37), 'handlebox', False)
        # Obtaining the member 'ydescent' of a type (line 115)
        ydescent_68969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 37), handlebox_68968, 'ydescent')
        # Getting the type of 'handlebox' (line 116)
        handlebox_68970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'handlebox', False)
        # Obtaining the member 'width' of a type (line 116)
        width_68971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 17), handlebox_68970, 'width')
        # Getting the type of 'handlebox' (line 116)
        handlebox_68972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'handlebox', False)
        # Obtaining the member 'height' of a type (line 116)
        height_68973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 34), handlebox_68972, 'height')
        # Getting the type of 'fontsize' (line 117)
        fontsize_68974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'fontsize', False)
        # Processing the call keyword arguments (line 113)
        kwargs_68975 = {}
        # Getting the type of 'self' (line 113)
        self_68962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 44), 'self', False)
        # Obtaining the member 'adjust_drawing_area' of a type (line 113)
        adjust_drawing_area_68963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 44), self_68962, 'adjust_drawing_area')
        # Calling adjust_drawing_area(args, kwargs) (line 113)
        adjust_drawing_area_call_result_68976 = invoke(stypy.reporting.localization.Localization(__file__, 113, 44), adjust_drawing_area_68963, *[legend_68964, orig_handle_68965, xdescent_68967, ydescent_68969, width_68971, height_68973, fontsize_68974], **kwargs_68975)
        
        # Assigning a type to the variable 'call_assignment_68817' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68817', adjust_drawing_area_call_result_68976)
        
        # Assigning a Call to a Name (line 113):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68980 = {}
        # Getting the type of 'call_assignment_68817' (line 113)
        call_assignment_68817_68977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68817', False)
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___68978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), call_assignment_68817_68977, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68981 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68978, *[int_68979], **kwargs_68980)
        
        # Assigning a type to the variable 'call_assignment_68818' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68818', getitem___call_result_68981)
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'call_assignment_68818' (line 113)
        call_assignment_68818_68982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68818')
        # Assigning a type to the variable 'xdescent' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'xdescent', call_assignment_68818_68982)
        
        # Assigning a Call to a Name (line 113):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68986 = {}
        # Getting the type of 'call_assignment_68817' (line 113)
        call_assignment_68817_68983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68817', False)
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___68984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), call_assignment_68817_68983, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68987 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68984, *[int_68985], **kwargs_68986)
        
        # Assigning a type to the variable 'call_assignment_68819' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68819', getitem___call_result_68987)
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'call_assignment_68819' (line 113)
        call_assignment_68819_68988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68819')
        # Assigning a type to the variable 'ydescent' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'ydescent', call_assignment_68819_68988)
        
        # Assigning a Call to a Name (line 113):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68992 = {}
        # Getting the type of 'call_assignment_68817' (line 113)
        call_assignment_68817_68989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68817', False)
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___68990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), call_assignment_68817_68989, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68993 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68990, *[int_68991], **kwargs_68992)
        
        # Assigning a type to the variable 'call_assignment_68820' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68820', getitem___call_result_68993)
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'call_assignment_68820' (line 113)
        call_assignment_68820_68994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68820')
        # Assigning a type to the variable 'width' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'width', call_assignment_68820_68994)
        
        # Assigning a Call to a Name (line 113):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_68997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'int')
        # Processing the call keyword arguments
        kwargs_68998 = {}
        # Getting the type of 'call_assignment_68817' (line 113)
        call_assignment_68817_68995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68817', False)
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___68996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), call_assignment_68817_68995, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_68999 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___68996, *[int_68997], **kwargs_68998)
        
        # Assigning a type to the variable 'call_assignment_68821' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68821', getitem___call_result_68999)
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'call_assignment_68821' (line 113)
        call_assignment_68821_69000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'call_assignment_68821')
        # Assigning a type to the variable 'height' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'height', call_assignment_68821_69000)
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to create_artists(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'legend' (line 118)
        legend_69003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'legend', False)
        # Getting the type of 'orig_handle' (line 118)
        orig_handle_69004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 46), 'orig_handle', False)
        # Getting the type of 'xdescent' (line 119)
        xdescent_69005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'xdescent', False)
        # Getting the type of 'ydescent' (line 119)
        ydescent_69006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 48), 'ydescent', False)
        # Getting the type of 'width' (line 119)
        width_69007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 58), 'width', False)
        # Getting the type of 'height' (line 119)
        height_69008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 65), 'height', False)
        # Getting the type of 'fontsize' (line 120)
        fontsize_69009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 38), 'fontsize', False)
        
        # Call to get_transform(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_69012 = {}
        # Getting the type of 'handlebox' (line 120)
        handlebox_69010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 48), 'handlebox', False)
        # Obtaining the member 'get_transform' of a type (line 120)
        get_transform_69011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 48), handlebox_69010, 'get_transform')
        # Calling get_transform(args, kwargs) (line 120)
        get_transform_call_result_69013 = invoke(stypy.reporting.localization.Localization(__file__, 120, 48), get_transform_69011, *[], **kwargs_69012)
        
        # Processing the call keyword arguments (line 118)
        kwargs_69014 = {}
        # Getting the type of 'self' (line 118)
        self_69001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'self', False)
        # Obtaining the member 'create_artists' of a type (line 118)
        create_artists_69002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 18), self_69001, 'create_artists')
        # Calling create_artists(args, kwargs) (line 118)
        create_artists_call_result_69015 = invoke(stypy.reporting.localization.Localization(__file__, 118, 18), create_artists_69002, *[legend_69003, orig_handle_69004, xdescent_69005, ydescent_69006, width_69007, height_69008, fontsize_69009, get_transform_call_result_69013], **kwargs_69014)
        
        # Assigning a type to the variable 'artists' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'artists', create_artists_call_result_69015)
        
        # Getting the type of 'artists' (line 123)
        artists_69016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'artists')
        # Testing the type of a for loop iterable (line 123)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 123, 8), artists_69016)
        # Getting the type of the for loop variable (line 123)
        for_loop_var_69017 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 123, 8), artists_69016)
        # Assigning a type to the variable 'a' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'a', for_loop_var_69017)
        # SSA begins for a for statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to add_artist(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'a' (line 124)
        a_69020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 33), 'a', False)
        # Processing the call keyword arguments (line 124)
        kwargs_69021 = {}
        # Getting the type of 'handlebox' (line 124)
        handlebox_69018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'handlebox', False)
        # Obtaining the member 'add_artist' of a type (line 124)
        add_artist_69019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), handlebox_69018, 'add_artist')
        # Calling add_artist(args, kwargs) (line 124)
        add_artist_call_result_69022 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), add_artist_69019, *[a_69020], **kwargs_69021)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        int_69023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 23), 'int')
        # Getting the type of 'artists' (line 127)
        artists_69024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'artists')
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___69025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 15), artists_69024, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_69026 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), getitem___69025, int_69023)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', subscript_call_result_69026)
        
        # ################# End of 'legend_artist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'legend_artist' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_69027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69027)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'legend_artist'
        return stypy_return_type_69027


    @norecursion
    def create_artists(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_artists'
        module_type_store = module_type_store.open_function_context('create_artists', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerBase.create_artists.__dict__.__setitem__('stypy_localization', localization)
        HandlerBase.create_artists.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerBase.create_artists.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerBase.create_artists.__dict__.__setitem__('stypy_function_name', 'HandlerBase.create_artists')
        HandlerBase.create_artists.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'])
        HandlerBase.create_artists.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerBase.create_artists.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerBase.create_artists.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerBase.create_artists.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerBase.create_artists.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerBase.create_artists.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerBase.create_artists', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_artists', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_artists(...)' code ##################

        
        # Call to NotImplementedError(...): (line 132)
        # Processing the call arguments (line 132)
        unicode_69029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 34), 'unicode', u'Derived must override')
        # Processing the call keyword arguments (line 132)
        kwargs_69030 = {}
        # Getting the type of 'NotImplementedError' (line 132)
        NotImplementedError_69028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 132)
        NotImplementedError_call_result_69031 = invoke(stypy.reporting.localization.Localization(__file__, 132, 14), NotImplementedError_69028, *[unicode_69029], **kwargs_69030)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 132, 8), NotImplementedError_call_result_69031, 'raise parameter', BaseException)
        
        # ################# End of 'create_artists(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_artists' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_69032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69032)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_artists'
        return stypy_return_type_69032


# Assigning a type to the variable 'HandlerBase' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'HandlerBase', HandlerBase)
# Declaration of the 'HandlerNpoints' class
# Getting the type of 'HandlerBase' (line 135)
HandlerBase_69033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'HandlerBase')

class HandlerNpoints(HandlerBase_69033, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_69034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 34), 'float')
        # Getting the type of 'None' (line 136)
        None_69035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 49), 'None')
        defaults = [float_69034, None_69035]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerNpoints.__init__', ['marker_pad', 'numpoints'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['marker_pad', 'numpoints'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'self' (line 137)
        self_69038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'self', False)
        # Processing the call keyword arguments (line 137)
        # Getting the type of 'kw' (line 137)
        kw_69039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 37), 'kw', False)
        kwargs_69040 = {'kw_69039': kw_69039}
        # Getting the type of 'HandlerBase' (line 137)
        HandlerBase_69036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'HandlerBase', False)
        # Obtaining the member '__init__' of a type (line 137)
        init___69037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), HandlerBase_69036, '__init__')
        # Calling __init__(args, kwargs) (line 137)
        init___call_result_69041 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), init___69037, *[self_69038], **kwargs_69040)
        
        
        # Assigning a Name to a Attribute (line 139):
        
        # Assigning a Name to a Attribute (line 139):
        # Getting the type of 'numpoints' (line 139)
        numpoints_69042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 26), 'numpoints')
        # Getting the type of 'self' (line 139)
        self_69043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self')
        # Setting the type of the member '_numpoints' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_69043, '_numpoints', numpoints_69042)
        
        # Assigning a Name to a Attribute (line 140):
        
        # Assigning a Name to a Attribute (line 140):
        # Getting the type of 'marker_pad' (line 140)
        marker_pad_69044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 27), 'marker_pad')
        # Getting the type of 'self' (line 140)
        self_69045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self')
        # Setting the type of the member '_marker_pad' of a type (line 140)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_69045, '_marker_pad', marker_pad_69044)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_numpoints(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_numpoints'
        module_type_store = module_type_store.open_function_context('get_numpoints', 142, 4, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerNpoints.get_numpoints.__dict__.__setitem__('stypy_localization', localization)
        HandlerNpoints.get_numpoints.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerNpoints.get_numpoints.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerNpoints.get_numpoints.__dict__.__setitem__('stypy_function_name', 'HandlerNpoints.get_numpoints')
        HandlerNpoints.get_numpoints.__dict__.__setitem__('stypy_param_names_list', ['legend'])
        HandlerNpoints.get_numpoints.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerNpoints.get_numpoints.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerNpoints.get_numpoints.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerNpoints.get_numpoints.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerNpoints.get_numpoints.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerNpoints.get_numpoints.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerNpoints.get_numpoints', ['legend'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_numpoints', localization, ['legend'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_numpoints(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 143)
        # Getting the type of 'self' (line 143)
        self_69046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'self')
        # Obtaining the member '_numpoints' of a type (line 143)
        _numpoints_69047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 11), self_69046, '_numpoints')
        # Getting the type of 'None' (line 143)
        None_69048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 30), 'None')
        
        (may_be_69049, more_types_in_union_69050) = may_be_none(_numpoints_69047, None_69048)

        if may_be_69049:

            if more_types_in_union_69050:
                # Runtime conditional SSA (line 143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'legend' (line 144)
            legend_69051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 19), 'legend')
            # Obtaining the member 'numpoints' of a type (line 144)
            numpoints_69052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 19), legend_69051, 'numpoints')
            # Assigning a type to the variable 'stypy_return_type' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'stypy_return_type', numpoints_69052)

            if more_types_in_union_69050:
                # Runtime conditional SSA for else branch (line 143)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_69049) or more_types_in_union_69050):
            # Getting the type of 'self' (line 146)
            self_69053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'self')
            # Obtaining the member '_numpoints' of a type (line 146)
            _numpoints_69054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), self_69053, '_numpoints')
            # Assigning a type to the variable 'stypy_return_type' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'stypy_return_type', _numpoints_69054)

            if (may_be_69049 and more_types_in_union_69050):
                # SSA join for if statement (line 143)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'get_numpoints(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_numpoints' in the type store
        # Getting the type of 'stypy_return_type' (line 142)
        stypy_return_type_69055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69055)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_numpoints'
        return stypy_return_type_69055


    @norecursion
    def get_xdata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_xdata'
        module_type_store = module_type_store.open_function_context('get_xdata', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerNpoints.get_xdata.__dict__.__setitem__('stypy_localization', localization)
        HandlerNpoints.get_xdata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerNpoints.get_xdata.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerNpoints.get_xdata.__dict__.__setitem__('stypy_function_name', 'HandlerNpoints.get_xdata')
        HandlerNpoints.get_xdata.__dict__.__setitem__('stypy_param_names_list', ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'])
        HandlerNpoints.get_xdata.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerNpoints.get_xdata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerNpoints.get_xdata.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerNpoints.get_xdata.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerNpoints.get_xdata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerNpoints.get_xdata.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerNpoints.get_xdata', ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_xdata', localization, ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_xdata(...)' code ##################

        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to get_numpoints(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'legend' (line 149)
        legend_69058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 39), 'legend', False)
        # Processing the call keyword arguments (line 149)
        kwargs_69059 = {}
        # Getting the type of 'self' (line 149)
        self_69056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'self', False)
        # Obtaining the member 'get_numpoints' of a type (line 149)
        get_numpoints_69057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), self_69056, 'get_numpoints')
        # Calling get_numpoints(args, kwargs) (line 149)
        get_numpoints_call_result_69060 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), get_numpoints_69057, *[legend_69058], **kwargs_69059)
        
        # Assigning a type to the variable 'numpoints' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'numpoints', get_numpoints_call_result_69060)
        
        
        # Getting the type of 'numpoints' (line 150)
        numpoints_69061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'numpoints')
        int_69062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 23), 'int')
        # Applying the binary operator '>' (line 150)
        result_gt_69063 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 11), '>', numpoints_69061, int_69062)
        
        # Testing the type of an if condition (line 150)
        if_condition_69064 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), result_gt_69063)
        # Assigning a type to the variable 'if_condition_69064' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_69064', if_condition_69064)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 152):
        
        # Assigning a BinOp to a Name (line 152):
        # Getting the type of 'self' (line 152)
        self_69065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'self')
        # Obtaining the member '_marker_pad' of a type (line 152)
        _marker_pad_69066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 18), self_69065, '_marker_pad')
        # Getting the type of 'fontsize' (line 152)
        fontsize_69067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 37), 'fontsize')
        # Applying the binary operator '*' (line 152)
        result_mul_69068 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 18), '*', _marker_pad_69066, fontsize_69067)
        
        # Assigning a type to the variable 'pad' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'pad', result_mul_69068)
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to linspace(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Getting the type of 'xdescent' (line 153)
        xdescent_69071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'xdescent', False)
        # Applying the 'usub' unary operator (line 153)
        result___neg___69072 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 32), 'usub', xdescent_69071)
        
        # Getting the type of 'pad' (line 153)
        pad_69073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 44), 'pad', False)
        # Applying the binary operator '+' (line 153)
        result_add_69074 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 32), '+', result___neg___69072, pad_69073)
        
        
        # Getting the type of 'xdescent' (line 154)
        xdescent_69075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 33), 'xdescent', False)
        # Applying the 'usub' unary operator (line 154)
        result___neg___69076 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 32), 'usub', xdescent_69075)
        
        # Getting the type of 'width' (line 154)
        width_69077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 44), 'width', False)
        # Applying the binary operator '+' (line 154)
        result_add_69078 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 32), '+', result___neg___69076, width_69077)
        
        # Getting the type of 'pad' (line 154)
        pad_69079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 52), 'pad', False)
        # Applying the binary operator '-' (line 154)
        result_sub_69080 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 50), '-', result_add_69078, pad_69079)
        
        # Getting the type of 'numpoints' (line 155)
        numpoints_69081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'numpoints', False)
        # Processing the call keyword arguments (line 153)
        kwargs_69082 = {}
        # Getting the type of 'np' (line 153)
        np_69069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'np', False)
        # Obtaining the member 'linspace' of a type (line 153)
        linspace_69070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 20), np_69069, 'linspace')
        # Calling linspace(args, kwargs) (line 153)
        linspace_call_result_69083 = invoke(stypy.reporting.localization.Localization(__file__, 153, 20), linspace_69070, *[result_add_69074, result_sub_69080, numpoints_69081], **kwargs_69082)
        
        # Assigning a type to the variable 'xdata' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'xdata', linspace_call_result_69083)
        
        # Assigning a Name to a Name (line 156):
        
        # Assigning a Name to a Name (line 156):
        # Getting the type of 'xdata' (line 156)
        xdata_69084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'xdata')
        # Assigning a type to the variable 'xdata_marker' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'xdata_marker', xdata_69084)
        # SSA branch for the else part of an if statement (line 150)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to linspace(...): (line 158)
        # Processing the call arguments (line 158)
        
        # Getting the type of 'xdescent' (line 158)
        xdescent_69087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'xdescent', False)
        # Applying the 'usub' unary operator (line 158)
        result___neg___69088 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 32), 'usub', xdescent_69087)
        
        
        # Getting the type of 'xdescent' (line 158)
        xdescent_69089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 44), 'xdescent', False)
        # Applying the 'usub' unary operator (line 158)
        result___neg___69090 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 43), 'usub', xdescent_69089)
        
        # Getting the type of 'width' (line 158)
        width_69091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 55), 'width', False)
        # Applying the binary operator '+' (line 158)
        result_add_69092 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 43), '+', result___neg___69090, width_69091)
        
        int_69093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 62), 'int')
        # Processing the call keyword arguments (line 158)
        kwargs_69094 = {}
        # Getting the type of 'np' (line 158)
        np_69085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'np', False)
        # Obtaining the member 'linspace' of a type (line 158)
        linspace_69086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 20), np_69085, 'linspace')
        # Calling linspace(args, kwargs) (line 158)
        linspace_call_result_69095 = invoke(stypy.reporting.localization.Localization(__file__, 158, 20), linspace_69086, *[result___neg___69088, result_add_69092, int_69093], **kwargs_69094)
        
        # Assigning a type to the variable 'xdata' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'xdata', linspace_call_result_69095)
        
        # Assigning a List to a Name (line 159):
        
        # Assigning a List to a Name (line 159):
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_69096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        # Adding element type (line 159)
        
        # Getting the type of 'xdescent' (line 159)
        xdescent_69097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 29), 'xdescent')
        # Applying the 'usub' unary operator (line 159)
        result___neg___69098 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 28), 'usub', xdescent_69097)
        
        float_69099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 40), 'float')
        # Getting the type of 'width' (line 159)
        width_69100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), 'width')
        # Applying the binary operator '*' (line 159)
        result_mul_69101 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 40), '*', float_69099, width_69100)
        
        # Applying the binary operator '+' (line 159)
        result_add_69102 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 28), '+', result___neg___69098, result_mul_69101)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 27), list_69096, result_add_69102)
        
        # Assigning a type to the variable 'xdata_marker' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'xdata_marker', list_69096)
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 160)
        tuple_69103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 160)
        # Adding element type (line 160)
        # Getting the type of 'xdata' (line 160)
        xdata_69104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'xdata')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 15), tuple_69103, xdata_69104)
        # Adding element type (line 160)
        # Getting the type of 'xdata_marker' (line 160)
        xdata_marker_69105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 22), 'xdata_marker')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 15), tuple_69103, xdata_marker_69105)
        
        # Assigning a type to the variable 'stypy_return_type' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'stypy_return_type', tuple_69103)
        
        # ################# End of 'get_xdata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_xdata' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_69106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69106)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_xdata'
        return stypy_return_type_69106


# Assigning a type to the variable 'HandlerNpoints' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'HandlerNpoints', HandlerNpoints)
# Declaration of the 'HandlerNpointsYoffsets' class
# Getting the type of 'HandlerNpoints' (line 164)
HandlerNpoints_69107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 29), 'HandlerNpoints')

class HandlerNpointsYoffsets(HandlerNpoints_69107, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 165)
        None_69108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 33), 'None')
        # Getting the type of 'None' (line 165)
        None_69109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 48), 'None')
        defaults = [None_69108, None_69109]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerNpointsYoffsets.__init__', ['numpoints', 'yoffsets'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['numpoints', 'yoffsets'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'self' (line 166)
        self_69112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 32), 'self', False)
        # Processing the call keyword arguments (line 166)
        # Getting the type of 'numpoints' (line 166)
        numpoints_69113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 48), 'numpoints', False)
        keyword_69114 = numpoints_69113
        # Getting the type of 'kw' (line 166)
        kw_69115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 61), 'kw', False)
        kwargs_69116 = {'kw_69115': kw_69115, 'numpoints': keyword_69114}
        # Getting the type of 'HandlerNpoints' (line 166)
        HandlerNpoints_69110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'HandlerNpoints', False)
        # Obtaining the member '__init__' of a type (line 166)
        init___69111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), HandlerNpoints_69110, '__init__')
        # Calling __init__(args, kwargs) (line 166)
        init___call_result_69117 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), init___69111, *[self_69112], **kwargs_69116)
        
        
        # Assigning a Name to a Attribute (line 167):
        
        # Assigning a Name to a Attribute (line 167):
        # Getting the type of 'yoffsets' (line 167)
        yoffsets_69118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'yoffsets')
        # Getting the type of 'self' (line 167)
        self_69119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member '_yoffsets' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_69119, '_yoffsets', yoffsets_69118)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_ydata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_ydata'
        module_type_store = module_type_store.open_function_context('get_ydata', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerNpointsYoffsets.get_ydata.__dict__.__setitem__('stypy_localization', localization)
        HandlerNpointsYoffsets.get_ydata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerNpointsYoffsets.get_ydata.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerNpointsYoffsets.get_ydata.__dict__.__setitem__('stypy_function_name', 'HandlerNpointsYoffsets.get_ydata')
        HandlerNpointsYoffsets.get_ydata.__dict__.__setitem__('stypy_param_names_list', ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'])
        HandlerNpointsYoffsets.get_ydata.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerNpointsYoffsets.get_ydata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerNpointsYoffsets.get_ydata.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerNpointsYoffsets.get_ydata.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerNpointsYoffsets.get_ydata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerNpointsYoffsets.get_ydata.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerNpointsYoffsets.get_ydata', ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_ydata', localization, ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_ydata(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 170)
        # Getting the type of 'self' (line 170)
        self_69120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'self')
        # Obtaining the member '_yoffsets' of a type (line 170)
        _yoffsets_69121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 11), self_69120, '_yoffsets')
        # Getting the type of 'None' (line 170)
        None_69122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 29), 'None')
        
        (may_be_69123, more_types_in_union_69124) = may_be_none(_yoffsets_69121, None_69122)

        if may_be_69123:

            if more_types_in_union_69124:
                # Runtime conditional SSA (line 170)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 171):
            
            # Assigning a BinOp to a Name (line 171):
            # Getting the type of 'height' (line 171)
            height_69125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'height')
            # Getting the type of 'legend' (line 171)
            legend_69126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'legend')
            # Obtaining the member '_scatteryoffsets' of a type (line 171)
            _scatteryoffsets_69127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 29), legend_69126, '_scatteryoffsets')
            # Applying the binary operator '*' (line 171)
            result_mul_69128 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 20), '*', height_69125, _scatteryoffsets_69127)
            
            # Assigning a type to the variable 'ydata' (line 171)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'ydata', result_mul_69128)

            if more_types_in_union_69124:
                # Runtime conditional SSA for else branch (line 170)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_69123) or more_types_in_union_69124):
            
            # Assigning a BinOp to a Name (line 173):
            
            # Assigning a BinOp to a Name (line 173):
            # Getting the type of 'height' (line 173)
            height_69129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'height')
            
            # Call to asarray(...): (line 173)
            # Processing the call arguments (line 173)
            # Getting the type of 'self' (line 173)
            self_69132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 40), 'self', False)
            # Obtaining the member '_yoffsets' of a type (line 173)
            _yoffsets_69133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 40), self_69132, '_yoffsets')
            # Processing the call keyword arguments (line 173)
            kwargs_69134 = {}
            # Getting the type of 'np' (line 173)
            np_69130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 'np', False)
            # Obtaining the member 'asarray' of a type (line 173)
            asarray_69131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 29), np_69130, 'asarray')
            # Calling asarray(args, kwargs) (line 173)
            asarray_call_result_69135 = invoke(stypy.reporting.localization.Localization(__file__, 173, 29), asarray_69131, *[_yoffsets_69133], **kwargs_69134)
            
            # Applying the binary operator '*' (line 173)
            result_mul_69136 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 20), '*', height_69129, asarray_call_result_69135)
            
            # Assigning a type to the variable 'ydata' (line 173)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'ydata', result_mul_69136)

            if (may_be_69123 and more_types_in_union_69124):
                # SSA join for if statement (line 170)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'ydata' (line 175)
        ydata_69137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'ydata')
        # Assigning a type to the variable 'stypy_return_type' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', ydata_69137)
        
        # ################# End of 'get_ydata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_ydata' in the type store
        # Getting the type of 'stypy_return_type' (line 169)
        stypy_return_type_69138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69138)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_ydata'
        return stypy_return_type_69138


# Assigning a type to the variable 'HandlerNpointsYoffsets' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'HandlerNpointsYoffsets', HandlerNpointsYoffsets)
# Declaration of the 'HandlerLine2D' class
# Getting the type of 'HandlerNpoints' (line 178)
HandlerNpoints_69139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'HandlerNpoints')

class HandlerLine2D(HandlerNpoints_69139, ):
    unicode_69140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, (-1)), 'unicode', u'\n    Handler for Line2D instances.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_69141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 34), 'float')
        # Getting the type of 'None' (line 182)
        None_69142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 49), 'None')
        defaults = [float_69141, None_69142]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerLine2D.__init__', ['marker_pad', 'numpoints'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['marker_pad', 'numpoints'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'self' (line 183)
        self_69145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 'self', False)
        # Processing the call keyword arguments (line 183)
        # Getting the type of 'marker_pad' (line 183)
        marker_pad_69146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 49), 'marker_pad', False)
        keyword_69147 = marker_pad_69146
        # Getting the type of 'numpoints' (line 183)
        numpoints_69148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 71), 'numpoints', False)
        keyword_69149 = numpoints_69148
        # Getting the type of 'kw' (line 183)
        kw_69150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 84), 'kw', False)
        kwargs_69151 = {'kw_69150': kw_69150, 'marker_pad': keyword_69147, 'numpoints': keyword_69149}
        # Getting the type of 'HandlerNpoints' (line 183)
        HandlerNpoints_69143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'HandlerNpoints', False)
        # Obtaining the member '__init__' of a type (line 183)
        init___69144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), HandlerNpoints_69143, '__init__')
        # Calling __init__(args, kwargs) (line 183)
        init___call_result_69152 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), init___69144, *[self_69145], **kwargs_69151)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def create_artists(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_artists'
        module_type_store = module_type_store.open_function_context('create_artists', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerLine2D.create_artists.__dict__.__setitem__('stypy_localization', localization)
        HandlerLine2D.create_artists.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerLine2D.create_artists.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerLine2D.create_artists.__dict__.__setitem__('stypy_function_name', 'HandlerLine2D.create_artists')
        HandlerLine2D.create_artists.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'])
        HandlerLine2D.create_artists.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerLine2D.create_artists.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerLine2D.create_artists.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerLine2D.create_artists.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerLine2D.create_artists.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerLine2D.create_artists.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerLine2D.create_artists', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_artists', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_artists(...)' code ##################

        
        # Assigning a Call to a Tuple (line 189):
        
        # Assigning a Call to a Name:
        
        # Call to get_xdata(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'legend' (line 189)
        legend_69155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 45), 'legend', False)
        # Getting the type of 'xdescent' (line 189)
        xdescent_69156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 53), 'xdescent', False)
        # Getting the type of 'ydescent' (line 189)
        ydescent_69157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 63), 'ydescent', False)
        # Getting the type of 'width' (line 190)
        width_69158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 45), 'width', False)
        # Getting the type of 'height' (line 190)
        height_69159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 52), 'height', False)
        # Getting the type of 'fontsize' (line 190)
        fontsize_69160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 60), 'fontsize', False)
        # Processing the call keyword arguments (line 189)
        kwargs_69161 = {}
        # Getting the type of 'self' (line 189)
        self_69153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 30), 'self', False)
        # Obtaining the member 'get_xdata' of a type (line 189)
        get_xdata_69154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 30), self_69153, 'get_xdata')
        # Calling get_xdata(args, kwargs) (line 189)
        get_xdata_call_result_69162 = invoke(stypy.reporting.localization.Localization(__file__, 189, 30), get_xdata_69154, *[legend_69155, xdescent_69156, ydescent_69157, width_69158, height_69159, fontsize_69160], **kwargs_69161)
        
        # Assigning a type to the variable 'call_assignment_68822' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_68822', get_xdata_call_result_69162)
        
        # Assigning a Call to a Name (line 189):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_69165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 8), 'int')
        # Processing the call keyword arguments
        kwargs_69166 = {}
        # Getting the type of 'call_assignment_68822' (line 189)
        call_assignment_68822_69163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_68822', False)
        # Obtaining the member '__getitem__' of a type (line 189)
        getitem___69164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), call_assignment_68822_69163, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_69167 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___69164, *[int_69165], **kwargs_69166)
        
        # Assigning a type to the variable 'call_assignment_68823' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_68823', getitem___call_result_69167)
        
        # Assigning a Name to a Name (line 189):
        # Getting the type of 'call_assignment_68823' (line 189)
        call_assignment_68823_69168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_68823')
        # Assigning a type to the variable 'xdata' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'xdata', call_assignment_68823_69168)
        
        # Assigning a Call to a Name (line 189):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_69171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 8), 'int')
        # Processing the call keyword arguments
        kwargs_69172 = {}
        # Getting the type of 'call_assignment_68822' (line 189)
        call_assignment_68822_69169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_68822', False)
        # Obtaining the member '__getitem__' of a type (line 189)
        getitem___69170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), call_assignment_68822_69169, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_69173 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___69170, *[int_69171], **kwargs_69172)
        
        # Assigning a type to the variable 'call_assignment_68824' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_68824', getitem___call_result_69173)
        
        # Assigning a Name to a Name (line 189):
        # Getting the type of 'call_assignment_68824' (line 189)
        call_assignment_68824_69174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_68824')
        # Assigning a type to the variable 'xdata_marker' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'xdata_marker', call_assignment_68824_69174)
        
        # Assigning a BinOp to a Name (line 192):
        
        # Assigning a BinOp to a Name (line 192):
        # Getting the type of 'height' (line 192)
        height_69175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 18), 'height')
        # Getting the type of 'ydescent' (line 192)
        ydescent_69176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'ydescent')
        # Applying the binary operator '-' (line 192)
        result_sub_69177 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 18), '-', height_69175, ydescent_69176)
        
        float_69178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 39), 'float')
        # Applying the binary operator 'div' (line 192)
        result_div_69179 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 17), 'div', result_sub_69177, float_69178)
        
        
        # Call to ones(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'xdata' (line 192)
        xdata_69182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 53), 'xdata', False)
        # Obtaining the member 'shape' of a type (line 192)
        shape_69183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 53), xdata_69182, 'shape')
        # Getting the type of 'float' (line 192)
        float_69184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 66), 'float', False)
        # Processing the call keyword arguments (line 192)
        kwargs_69185 = {}
        # Getting the type of 'np' (line 192)
        np_69180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 45), 'np', False)
        # Obtaining the member 'ones' of a type (line 192)
        ones_69181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 45), np_69180, 'ones')
        # Calling ones(args, kwargs) (line 192)
        ones_call_result_69186 = invoke(stypy.reporting.localization.Localization(__file__, 192, 45), ones_69181, *[shape_69183, float_69184], **kwargs_69185)
        
        # Applying the binary operator '*' (line 192)
        result_mul_69187 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 16), '*', result_div_69179, ones_call_result_69186)
        
        # Assigning a type to the variable 'ydata' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'ydata', result_mul_69187)
        
        # Assigning a Call to a Name (line 193):
        
        # Assigning a Call to a Name (line 193):
        
        # Call to Line2D(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'xdata' (line 193)
        xdata_69189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 25), 'xdata', False)
        # Getting the type of 'ydata' (line 193)
        ydata_69190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 32), 'ydata', False)
        # Processing the call keyword arguments (line 193)
        kwargs_69191 = {}
        # Getting the type of 'Line2D' (line 193)
        Line2D_69188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 193)
        Line2D_call_result_69192 = invoke(stypy.reporting.localization.Localization(__file__, 193, 18), Line2D_69188, *[xdata_69189, ydata_69190], **kwargs_69191)
        
        # Assigning a type to the variable 'legline' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'legline', Line2D_call_result_69192)
        
        # Call to update_prop(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'legline' (line 195)
        legline_69195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 25), 'legline', False)
        # Getting the type of 'orig_handle' (line 195)
        orig_handle_69196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 34), 'orig_handle', False)
        # Getting the type of 'legend' (line 195)
        legend_69197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 47), 'legend', False)
        # Processing the call keyword arguments (line 195)
        kwargs_69198 = {}
        # Getting the type of 'self' (line 195)
        self_69193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 195)
        update_prop_69194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), self_69193, 'update_prop')
        # Calling update_prop(args, kwargs) (line 195)
        update_prop_call_result_69199 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), update_prop_69194, *[legline_69195, orig_handle_69196, legend_69197], **kwargs_69198)
        
        
        # Call to set_drawstyle(...): (line 196)
        # Processing the call arguments (line 196)
        unicode_69202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 30), 'unicode', u'default')
        # Processing the call keyword arguments (line 196)
        kwargs_69203 = {}
        # Getting the type of 'legline' (line 196)
        legline_69200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'legline', False)
        # Obtaining the member 'set_drawstyle' of a type (line 196)
        set_drawstyle_69201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), legline_69200, 'set_drawstyle')
        # Calling set_drawstyle(args, kwargs) (line 196)
        set_drawstyle_call_result_69204 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), set_drawstyle_69201, *[unicode_69202], **kwargs_69203)
        
        
        # Call to set_marker(...): (line 197)
        # Processing the call arguments (line 197)
        unicode_69207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 27), 'unicode', u'')
        # Processing the call keyword arguments (line 197)
        kwargs_69208 = {}
        # Getting the type of 'legline' (line 197)
        legline_69205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'legline', False)
        # Obtaining the member 'set_marker' of a type (line 197)
        set_marker_69206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), legline_69205, 'set_marker')
        # Calling set_marker(args, kwargs) (line 197)
        set_marker_call_result_69209 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), set_marker_69206, *[unicode_69207], **kwargs_69208)
        
        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Call to Line2D(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'xdata_marker' (line 199)
        xdata_marker_69211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 32), 'xdata_marker', False)
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'xdata_marker' (line 199)
        xdata_marker_69213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 57), 'xdata_marker', False)
        # Processing the call keyword arguments (line 199)
        kwargs_69214 = {}
        # Getting the type of 'len' (line 199)
        len_69212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 53), 'len', False)
        # Calling len(args, kwargs) (line 199)
        len_call_result_69215 = invoke(stypy.reporting.localization.Localization(__file__, 199, 53), len_69212, *[xdata_marker_69213], **kwargs_69214)
        
        slice_69216 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 46), None, len_call_result_69215, None)
        # Getting the type of 'ydata' (line 199)
        ydata_69217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 46), 'ydata', False)
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___69218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 46), ydata_69217, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_69219 = invoke(stypy.reporting.localization.Localization(__file__, 199, 46), getitem___69218, slice_69216)
        
        # Processing the call keyword arguments (line 199)
        kwargs_69220 = {}
        # Getting the type of 'Line2D' (line 199)
        Line2D_69210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 25), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 199)
        Line2D_call_result_69221 = invoke(stypy.reporting.localization.Localization(__file__, 199, 25), Line2D_69210, *[xdata_marker_69211, subscript_call_result_69219], **kwargs_69220)
        
        # Assigning a type to the variable 'legline_marker' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'legline_marker', Line2D_call_result_69221)
        
        # Call to update_prop(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'legline_marker' (line 200)
        legline_marker_69224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'legline_marker', False)
        # Getting the type of 'orig_handle' (line 200)
        orig_handle_69225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 41), 'orig_handle', False)
        # Getting the type of 'legend' (line 200)
        legend_69226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 54), 'legend', False)
        # Processing the call keyword arguments (line 200)
        kwargs_69227 = {}
        # Getting the type of 'self' (line 200)
        self_69222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 200)
        update_prop_69223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_69222, 'update_prop')
        # Calling update_prop(args, kwargs) (line 200)
        update_prop_call_result_69228 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), update_prop_69223, *[legline_marker_69224, orig_handle_69225, legend_69226], **kwargs_69227)
        
        
        # Call to set_linestyle(...): (line 201)
        # Processing the call arguments (line 201)
        unicode_69231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 37), 'unicode', u'None')
        # Processing the call keyword arguments (line 201)
        kwargs_69232 = {}
        # Getting the type of 'legline_marker' (line 201)
        legline_marker_69229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'legline_marker', False)
        # Obtaining the member 'set_linestyle' of a type (line 201)
        set_linestyle_69230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), legline_marker_69229, 'set_linestyle')
        # Calling set_linestyle(args, kwargs) (line 201)
        set_linestyle_call_result_69233 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), set_linestyle_69230, *[unicode_69231], **kwargs_69232)
        
        
        
        # Getting the type of 'legend' (line 202)
        legend_69234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'legend')
        # Obtaining the member 'markerscale' of a type (line 202)
        markerscale_69235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 11), legend_69234, 'markerscale')
        int_69236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 33), 'int')
        # Applying the binary operator '!=' (line 202)
        result_ne_69237 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 11), '!=', markerscale_69235, int_69236)
        
        # Testing the type of an if condition (line 202)
        if_condition_69238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 8), result_ne_69237)
        # Assigning a type to the variable 'if_condition_69238' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'if_condition_69238', if_condition_69238)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 203):
        
        # Assigning a BinOp to a Name (line 203):
        
        # Call to get_markersize(...): (line 203)
        # Processing the call keyword arguments (line 203)
        kwargs_69241 = {}
        # Getting the type of 'legline_marker' (line 203)
        legline_marker_69239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'legline_marker', False)
        # Obtaining the member 'get_markersize' of a type (line 203)
        get_markersize_69240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 20), legline_marker_69239, 'get_markersize')
        # Calling get_markersize(args, kwargs) (line 203)
        get_markersize_call_result_69242 = invoke(stypy.reporting.localization.Localization(__file__, 203, 20), get_markersize_69240, *[], **kwargs_69241)
        
        # Getting the type of 'legend' (line 203)
        legend_69243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 54), 'legend')
        # Obtaining the member 'markerscale' of a type (line 203)
        markerscale_69244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 54), legend_69243, 'markerscale')
        # Applying the binary operator '*' (line 203)
        result_mul_69245 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 20), '*', get_markersize_call_result_69242, markerscale_69244)
        
        # Assigning a type to the variable 'newsz' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'newsz', result_mul_69245)
        
        # Call to set_markersize(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'newsz' (line 204)
        newsz_69248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 42), 'newsz', False)
        # Processing the call keyword arguments (line 204)
        kwargs_69249 = {}
        # Getting the type of 'legline_marker' (line 204)
        legline_marker_69246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'legline_marker', False)
        # Obtaining the member 'set_markersize' of a type (line 204)
        set_markersize_69247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), legline_marker_69246, 'set_markersize')
        # Calling set_markersize(args, kwargs) (line 204)
        set_markersize_call_result_69250 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), set_markersize_69247, *[newsz_69248], **kwargs_69249)
        
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 208):
        
        # Assigning a Name to a Attribute (line 208):
        # Getting the type of 'legline_marker' (line 208)
        legline_marker_69251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 29), 'legline_marker')
        # Getting the type of 'legline' (line 208)
        legline_69252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'legline')
        # Setting the type of the member '_legmarker' of a type (line 208)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), legline_69252, '_legmarker', legline_marker_69251)
        
        # Call to set_transform(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'trans' (line 210)
        trans_69255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'trans', False)
        # Processing the call keyword arguments (line 210)
        kwargs_69256 = {}
        # Getting the type of 'legline' (line 210)
        legline_69253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'legline', False)
        # Obtaining the member 'set_transform' of a type (line 210)
        set_transform_69254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), legline_69253, 'set_transform')
        # Calling set_transform(args, kwargs) (line 210)
        set_transform_call_result_69257 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), set_transform_69254, *[trans_69255], **kwargs_69256)
        
        
        # Call to set_transform(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'trans' (line 211)
        trans_69260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 37), 'trans', False)
        # Processing the call keyword arguments (line 211)
        kwargs_69261 = {}
        # Getting the type of 'legline_marker' (line 211)
        legline_marker_69258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'legline_marker', False)
        # Obtaining the member 'set_transform' of a type (line 211)
        set_transform_69259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), legline_marker_69258, 'set_transform')
        # Calling set_transform(args, kwargs) (line 211)
        set_transform_call_result_69262 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), set_transform_69259, *[trans_69260], **kwargs_69261)
        
        
        # Obtaining an instance of the builtin type 'list' (line 213)
        list_69263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 213)
        # Adding element type (line 213)
        # Getting the type of 'legline' (line 213)
        legline_69264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'legline')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 15), list_69263, legline_69264)
        # Adding element type (line 213)
        # Getting the type of 'legline_marker' (line 213)
        legline_marker_69265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 25), 'legline_marker')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 15), list_69263, legline_marker_69265)
        
        # Assigning a type to the variable 'stypy_return_type' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'stypy_return_type', list_69263)
        
        # ################# End of 'create_artists(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_artists' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_69266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69266)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_artists'
        return stypy_return_type_69266


# Assigning a type to the variable 'HandlerLine2D' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'HandlerLine2D', HandlerLine2D)
# Declaration of the 'HandlerPatch' class
# Getting the type of 'HandlerBase' (line 216)
HandlerBase_69267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 19), 'HandlerBase')

class HandlerPatch(HandlerBase_69267, ):
    unicode_69268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'unicode', u'\n    Handler for Patch instances.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 220)
        None_69269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 34), 'None')
        defaults = [None_69269]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 220, 4, False)
        # Assigning a type to the variable 'self' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerPatch.__init__', ['patch_func'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['patch_func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_69270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, (-1)), 'unicode', u"\n        The HandlerPatch class optionally takes a function ``patch_func``\n        who's responsibility is to create the legend key artist. The\n        ``patch_func`` should have the signature::\n\n            def patch_func(legend=legend, orig_handle=orig_handle,\n                           xdescent=xdescent, ydescent=ydescent,\n                           width=width, height=height, fontsize=fontsize)\n\n        Subsequently the created artist will have its ``update_prop`` method\n        called and the appropriate transform will be applied.\n\n        ")
        
        # Call to __init__(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'self' (line 234)
        self_69273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 29), 'self', False)
        # Processing the call keyword arguments (line 234)
        # Getting the type of 'kw' (line 234)
        kw_69274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 37), 'kw', False)
        kwargs_69275 = {'kw_69274': kw_69274}
        # Getting the type of 'HandlerBase' (line 234)
        HandlerBase_69271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'HandlerBase', False)
        # Obtaining the member '__init__' of a type (line 234)
        init___69272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), HandlerBase_69271, '__init__')
        # Calling __init__(args, kwargs) (line 234)
        init___call_result_69276 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), init___69272, *[self_69273], **kwargs_69275)
        
        
        # Assigning a Name to a Attribute (line 235):
        
        # Assigning a Name to a Attribute (line 235):
        # Getting the type of 'patch_func' (line 235)
        patch_func_69277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 27), 'patch_func')
        # Getting the type of 'self' (line 235)
        self_69278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self')
        # Setting the type of the member '_patch_func' of a type (line 235)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_69278, '_patch_func', patch_func_69277)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _create_patch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_create_patch'
        module_type_store = module_type_store.open_function_context('_create_patch', 237, 4, False)
        # Assigning a type to the variable 'self' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerPatch._create_patch.__dict__.__setitem__('stypy_localization', localization)
        HandlerPatch._create_patch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerPatch._create_patch.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerPatch._create_patch.__dict__.__setitem__('stypy_function_name', 'HandlerPatch._create_patch')
        HandlerPatch._create_patch.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'])
        HandlerPatch._create_patch.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerPatch._create_patch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerPatch._create_patch.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerPatch._create_patch.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerPatch._create_patch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerPatch._create_patch.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerPatch._create_patch', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_create_patch', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_create_patch(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 239)
        # Getting the type of 'self' (line 239)
        self_69279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), 'self')
        # Obtaining the member '_patch_func' of a type (line 239)
        _patch_func_69280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 11), self_69279, '_patch_func')
        # Getting the type of 'None' (line 239)
        None_69281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 31), 'None')
        
        (may_be_69282, more_types_in_union_69283) = may_be_none(_patch_func_69280, None_69281)

        if may_be_69282:

            if more_types_in_union_69283:
                # Runtime conditional SSA (line 239)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 240):
            
            # Assigning a Call to a Name (line 240):
            
            # Call to Rectangle(...): (line 240)
            # Processing the call keyword arguments (line 240)
            
            # Obtaining an instance of the builtin type 'tuple' (line 240)
            tuple_69285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 240)
            # Adding element type (line 240)
            
            # Getting the type of 'xdescent' (line 240)
            xdescent_69286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'xdescent', False)
            # Applying the 'usub' unary operator (line 240)
            result___neg___69287 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 30), 'usub', xdescent_69286)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 30), tuple_69285, result___neg___69287)
            # Adding element type (line 240)
            
            # Getting the type of 'ydescent' (line 240)
            ydescent_69288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 42), 'ydescent', False)
            # Applying the 'usub' unary operator (line 240)
            result___neg___69289 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 41), 'usub', ydescent_69288)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 30), tuple_69285, result___neg___69289)
            
            keyword_69290 = tuple_69285
            # Getting the type of 'width' (line 241)
            width_69291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 32), 'width', False)
            keyword_69292 = width_69291
            # Getting the type of 'height' (line 241)
            height_69293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 46), 'height', False)
            keyword_69294 = height_69293
            kwargs_69295 = {'width': keyword_69292, 'xy': keyword_69290, 'height': keyword_69294}
            # Getting the type of 'Rectangle' (line 240)
            Rectangle_69284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'Rectangle', False)
            # Calling Rectangle(args, kwargs) (line 240)
            Rectangle_call_result_69296 = invoke(stypy.reporting.localization.Localization(__file__, 240, 16), Rectangle_69284, *[], **kwargs_69295)
            
            # Assigning a type to the variable 'p' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'p', Rectangle_call_result_69296)

            if more_types_in_union_69283:
                # Runtime conditional SSA for else branch (line 239)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_69282) or more_types_in_union_69283):
            
            # Assigning a Call to a Name (line 243):
            
            # Assigning a Call to a Name (line 243):
            
            # Call to _patch_func(...): (line 243)
            # Processing the call keyword arguments (line 243)
            # Getting the type of 'legend' (line 243)
            legend_69299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 40), 'legend', False)
            keyword_69300 = legend_69299
            # Getting the type of 'orig_handle' (line 243)
            orig_handle_69301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 60), 'orig_handle', False)
            keyword_69302 = orig_handle_69301
            # Getting the type of 'xdescent' (line 244)
            xdescent_69303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 42), 'xdescent', False)
            keyword_69304 = xdescent_69303
            # Getting the type of 'ydescent' (line 244)
            ydescent_69305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 61), 'ydescent', False)
            keyword_69306 = ydescent_69305
            # Getting the type of 'width' (line 245)
            width_69307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 39), 'width', False)
            keyword_69308 = width_69307
            # Getting the type of 'height' (line 245)
            height_69309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 53), 'height', False)
            keyword_69310 = height_69309
            # Getting the type of 'fontsize' (line 245)
            fontsize_69311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 70), 'fontsize', False)
            keyword_69312 = fontsize_69311
            kwargs_69313 = {'ydescent': keyword_69306, 'height': keyword_69310, 'width': keyword_69308, 'xdescent': keyword_69304, 'fontsize': keyword_69312, 'orig_handle': keyword_69302, 'legend': keyword_69300}
            # Getting the type of 'self' (line 243)
            self_69297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'self', False)
            # Obtaining the member '_patch_func' of a type (line 243)
            _patch_func_69298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 16), self_69297, '_patch_func')
            # Calling _patch_func(args, kwargs) (line 243)
            _patch_func_call_result_69314 = invoke(stypy.reporting.localization.Localization(__file__, 243, 16), _patch_func_69298, *[], **kwargs_69313)
            
            # Assigning a type to the variable 'p' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'p', _patch_func_call_result_69314)

            if (may_be_69282 and more_types_in_union_69283):
                # SSA join for if statement (line 239)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'p' (line 246)
        p_69315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'p')
        # Assigning a type to the variable 'stypy_return_type' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'stypy_return_type', p_69315)
        
        # ################# End of '_create_patch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_create_patch' in the type store
        # Getting the type of 'stypy_return_type' (line 237)
        stypy_return_type_69316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69316)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_create_patch'
        return stypy_return_type_69316


    @norecursion
    def create_artists(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_artists'
        module_type_store = module_type_store.open_function_context('create_artists', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerPatch.create_artists.__dict__.__setitem__('stypy_localization', localization)
        HandlerPatch.create_artists.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerPatch.create_artists.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerPatch.create_artists.__dict__.__setitem__('stypy_function_name', 'HandlerPatch.create_artists')
        HandlerPatch.create_artists.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'])
        HandlerPatch.create_artists.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerPatch.create_artists.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerPatch.create_artists.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerPatch.create_artists.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerPatch.create_artists.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerPatch.create_artists.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerPatch.create_artists', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_artists', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_artists(...)' code ##################

        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to _create_patch(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'legend' (line 250)
        legend_69319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 31), 'legend', False)
        # Getting the type of 'orig_handle' (line 250)
        orig_handle_69320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 39), 'orig_handle', False)
        # Getting the type of 'xdescent' (line 251)
        xdescent_69321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 31), 'xdescent', False)
        # Getting the type of 'ydescent' (line 251)
        ydescent_69322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 41), 'ydescent', False)
        # Getting the type of 'width' (line 251)
        width_69323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 51), 'width', False)
        # Getting the type of 'height' (line 251)
        height_69324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 58), 'height', False)
        # Getting the type of 'fontsize' (line 251)
        fontsize_69325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 66), 'fontsize', False)
        # Processing the call keyword arguments (line 250)
        kwargs_69326 = {}
        # Getting the type of 'self' (line 250)
        self_69317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'self', False)
        # Obtaining the member '_create_patch' of a type (line 250)
        _create_patch_69318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), self_69317, '_create_patch')
        # Calling _create_patch(args, kwargs) (line 250)
        _create_patch_call_result_69327 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), _create_patch_69318, *[legend_69319, orig_handle_69320, xdescent_69321, ydescent_69322, width_69323, height_69324, fontsize_69325], **kwargs_69326)
        
        # Assigning a type to the variable 'p' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'p', _create_patch_call_result_69327)
        
        # Call to update_prop(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'p' (line 252)
        p_69330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 25), 'p', False)
        # Getting the type of 'orig_handle' (line 252)
        orig_handle_69331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 28), 'orig_handle', False)
        # Getting the type of 'legend' (line 252)
        legend_69332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 41), 'legend', False)
        # Processing the call keyword arguments (line 252)
        kwargs_69333 = {}
        # Getting the type of 'self' (line 252)
        self_69328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 252)
        update_prop_69329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), self_69328, 'update_prop')
        # Calling update_prop(args, kwargs) (line 252)
        update_prop_call_result_69334 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), update_prop_69329, *[p_69330, orig_handle_69331, legend_69332], **kwargs_69333)
        
        
        # Call to set_transform(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'trans' (line 253)
        trans_69337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 24), 'trans', False)
        # Processing the call keyword arguments (line 253)
        kwargs_69338 = {}
        # Getting the type of 'p' (line 253)
        p_69335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'p', False)
        # Obtaining the member 'set_transform' of a type (line 253)
        set_transform_69336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), p_69335, 'set_transform')
        # Calling set_transform(args, kwargs) (line 253)
        set_transform_call_result_69339 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), set_transform_69336, *[trans_69337], **kwargs_69338)
        
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_69340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        # Getting the type of 'p' (line 254)
        p_69341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 15), list_69340, p_69341)
        
        # Assigning a type to the variable 'stypy_return_type' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'stypy_return_type', list_69340)
        
        # ################# End of 'create_artists(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_artists' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_69342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69342)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_artists'
        return stypy_return_type_69342


# Assigning a type to the variable 'HandlerPatch' (line 216)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 0), 'HandlerPatch', HandlerPatch)
# Declaration of the 'HandlerLineCollection' class
# Getting the type of 'HandlerLine2D' (line 257)
HandlerLine2D_69343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'HandlerLine2D')

class HandlerLineCollection(HandlerLine2D_69343, ):
    unicode_69344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, (-1)), 'unicode', u'\n    Handler for LineCollection instances.\n    ')

    @norecursion
    def get_numpoints(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_numpoints'
        module_type_store = module_type_store.open_function_context('get_numpoints', 261, 4, False)
        # Assigning a type to the variable 'self' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerLineCollection.get_numpoints.__dict__.__setitem__('stypy_localization', localization)
        HandlerLineCollection.get_numpoints.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerLineCollection.get_numpoints.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerLineCollection.get_numpoints.__dict__.__setitem__('stypy_function_name', 'HandlerLineCollection.get_numpoints')
        HandlerLineCollection.get_numpoints.__dict__.__setitem__('stypy_param_names_list', ['legend'])
        HandlerLineCollection.get_numpoints.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerLineCollection.get_numpoints.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerLineCollection.get_numpoints.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerLineCollection.get_numpoints.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerLineCollection.get_numpoints.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerLineCollection.get_numpoints.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerLineCollection.get_numpoints', ['legend'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_numpoints', localization, ['legend'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_numpoints(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 262)
        # Getting the type of 'self' (line 262)
        self_69345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 11), 'self')
        # Obtaining the member '_numpoints' of a type (line 262)
        _numpoints_69346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 11), self_69345, '_numpoints')
        # Getting the type of 'None' (line 262)
        None_69347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 30), 'None')
        
        (may_be_69348, more_types_in_union_69349) = may_be_none(_numpoints_69346, None_69347)

        if may_be_69348:

            if more_types_in_union_69349:
                # Runtime conditional SSA (line 262)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'legend' (line 263)
            legend_69350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'legend')
            # Obtaining the member 'scatterpoints' of a type (line 263)
            scatterpoints_69351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 19), legend_69350, 'scatterpoints')
            # Assigning a type to the variable 'stypy_return_type' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'stypy_return_type', scatterpoints_69351)

            if more_types_in_union_69349:
                # Runtime conditional SSA for else branch (line 262)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_69348) or more_types_in_union_69349):
            # Getting the type of 'self' (line 265)
            self_69352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 19), 'self')
            # Obtaining the member '_numpoints' of a type (line 265)
            _numpoints_69353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 19), self_69352, '_numpoints')
            # Assigning a type to the variable 'stypy_return_type' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'stypy_return_type', _numpoints_69353)

            if (may_be_69348 and more_types_in_union_69349):
                # SSA join for if statement (line 262)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'get_numpoints(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_numpoints' in the type store
        # Getting the type of 'stypy_return_type' (line 261)
        stypy_return_type_69354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69354)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_numpoints'
        return stypy_return_type_69354


    @norecursion
    def _default_update_prop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_default_update_prop'
        module_type_store = module_type_store.open_function_context('_default_update_prop', 267, 4, False)
        # Assigning a type to the variable 'self' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerLineCollection._default_update_prop.__dict__.__setitem__('stypy_localization', localization)
        HandlerLineCollection._default_update_prop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerLineCollection._default_update_prop.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerLineCollection._default_update_prop.__dict__.__setitem__('stypy_function_name', 'HandlerLineCollection._default_update_prop')
        HandlerLineCollection._default_update_prop.__dict__.__setitem__('stypy_param_names_list', ['legend_handle', 'orig_handle'])
        HandlerLineCollection._default_update_prop.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerLineCollection._default_update_prop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerLineCollection._default_update_prop.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerLineCollection._default_update_prop.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerLineCollection._default_update_prop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerLineCollection._default_update_prop.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerLineCollection._default_update_prop', ['legend_handle', 'orig_handle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_default_update_prop', localization, ['legend_handle', 'orig_handle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_default_update_prop(...)' code ##################

        
        # Assigning a Subscript to a Name (line 268):
        
        # Assigning a Subscript to a Name (line 268):
        
        # Obtaining the type of the subscript
        int_69355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 42), 'int')
        
        # Call to get_linewidths(...): (line 268)
        # Processing the call keyword arguments (line 268)
        kwargs_69358 = {}
        # Getting the type of 'orig_handle' (line 268)
        orig_handle_69356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 13), 'orig_handle', False)
        # Obtaining the member 'get_linewidths' of a type (line 268)
        get_linewidths_69357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 13), orig_handle_69356, 'get_linewidths')
        # Calling get_linewidths(args, kwargs) (line 268)
        get_linewidths_call_result_69359 = invoke(stypy.reporting.localization.Localization(__file__, 268, 13), get_linewidths_69357, *[], **kwargs_69358)
        
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___69360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 13), get_linewidths_call_result_69359, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_69361 = invoke(stypy.reporting.localization.Localization(__file__, 268, 13), getitem___69360, int_69355)
        
        # Assigning a type to the variable 'lw' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'lw', subscript_call_result_69361)
        
        # Assigning a Subscript to a Name (line 269):
        
        # Assigning a Subscript to a Name (line 269):
        
        # Obtaining the type of the subscript
        int_69362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 44), 'int')
        # Getting the type of 'orig_handle' (line 269)
        orig_handle_69363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 17), 'orig_handle')
        # Obtaining the member '_us_linestyles' of a type (line 269)
        _us_linestyles_69364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 17), orig_handle_69363, '_us_linestyles')
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___69365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 17), _us_linestyles_69364, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_69366 = invoke(stypy.reporting.localization.Localization(__file__, 269, 17), getitem___69365, int_69362)
        
        # Assigning a type to the variable 'dashes' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'dashes', subscript_call_result_69366)
        
        # Assigning a Subscript to a Name (line 270):
        
        # Assigning a Subscript to a Name (line 270):
        
        # Obtaining the type of the subscript
        int_69367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 41), 'int')
        
        # Call to get_colors(...): (line 270)
        # Processing the call keyword arguments (line 270)
        kwargs_69370 = {}
        # Getting the type of 'orig_handle' (line 270)
        orig_handle_69368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'orig_handle', False)
        # Obtaining the member 'get_colors' of a type (line 270)
        get_colors_69369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), orig_handle_69368, 'get_colors')
        # Calling get_colors(args, kwargs) (line 270)
        get_colors_call_result_69371 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), get_colors_69369, *[], **kwargs_69370)
        
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___69372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), get_colors_call_result_69371, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_69373 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), getitem___69372, int_69367)
        
        # Assigning a type to the variable 'color' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'color', subscript_call_result_69373)
        
        # Call to set_color(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'color' (line 271)
        color_69376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 32), 'color', False)
        # Processing the call keyword arguments (line 271)
        kwargs_69377 = {}
        # Getting the type of 'legend_handle' (line 271)
        legend_handle_69374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'legend_handle', False)
        # Obtaining the member 'set_color' of a type (line 271)
        set_color_69375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), legend_handle_69374, 'set_color')
        # Calling set_color(args, kwargs) (line 271)
        set_color_call_result_69378 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), set_color_69375, *[color_69376], **kwargs_69377)
        
        
        # Call to set_linestyle(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'dashes' (line 272)
        dashes_69381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 36), 'dashes', False)
        # Processing the call keyword arguments (line 272)
        kwargs_69382 = {}
        # Getting the type of 'legend_handle' (line 272)
        legend_handle_69379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'legend_handle', False)
        # Obtaining the member 'set_linestyle' of a type (line 272)
        set_linestyle_69380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), legend_handle_69379, 'set_linestyle')
        # Calling set_linestyle(args, kwargs) (line 272)
        set_linestyle_call_result_69383 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), set_linestyle_69380, *[dashes_69381], **kwargs_69382)
        
        
        # Call to set_linewidth(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'lw' (line 273)
        lw_69386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 36), 'lw', False)
        # Processing the call keyword arguments (line 273)
        kwargs_69387 = {}
        # Getting the type of 'legend_handle' (line 273)
        legend_handle_69384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'legend_handle', False)
        # Obtaining the member 'set_linewidth' of a type (line 273)
        set_linewidth_69385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), legend_handle_69384, 'set_linewidth')
        # Calling set_linewidth(args, kwargs) (line 273)
        set_linewidth_call_result_69388 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), set_linewidth_69385, *[lw_69386], **kwargs_69387)
        
        
        # ################# End of '_default_update_prop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_default_update_prop' in the type store
        # Getting the type of 'stypy_return_type' (line 267)
        stypy_return_type_69389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_default_update_prop'
        return stypy_return_type_69389


    @norecursion
    def create_artists(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_artists'
        module_type_store = module_type_store.open_function_context('create_artists', 275, 4, False)
        # Assigning a type to the variable 'self' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerLineCollection.create_artists.__dict__.__setitem__('stypy_localization', localization)
        HandlerLineCollection.create_artists.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerLineCollection.create_artists.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerLineCollection.create_artists.__dict__.__setitem__('stypy_function_name', 'HandlerLineCollection.create_artists')
        HandlerLineCollection.create_artists.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'])
        HandlerLineCollection.create_artists.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerLineCollection.create_artists.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerLineCollection.create_artists.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerLineCollection.create_artists.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerLineCollection.create_artists.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerLineCollection.create_artists.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerLineCollection.create_artists', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_artists', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_artists(...)' code ##################

        
        # Assigning a Call to a Tuple (line 278):
        
        # Assigning a Call to a Name:
        
        # Call to get_xdata(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'legend' (line 278)
        legend_69392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 45), 'legend', False)
        # Getting the type of 'xdescent' (line 278)
        xdescent_69393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 53), 'xdescent', False)
        # Getting the type of 'ydescent' (line 278)
        ydescent_69394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 63), 'ydescent', False)
        # Getting the type of 'width' (line 279)
        width_69395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 45), 'width', False)
        # Getting the type of 'height' (line 279)
        height_69396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 52), 'height', False)
        # Getting the type of 'fontsize' (line 279)
        fontsize_69397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 60), 'fontsize', False)
        # Processing the call keyword arguments (line 278)
        kwargs_69398 = {}
        # Getting the type of 'self' (line 278)
        self_69390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 30), 'self', False)
        # Obtaining the member 'get_xdata' of a type (line 278)
        get_xdata_69391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 30), self_69390, 'get_xdata')
        # Calling get_xdata(args, kwargs) (line 278)
        get_xdata_call_result_69399 = invoke(stypy.reporting.localization.Localization(__file__, 278, 30), get_xdata_69391, *[legend_69392, xdescent_69393, ydescent_69394, width_69395, height_69396, fontsize_69397], **kwargs_69398)
        
        # Assigning a type to the variable 'call_assignment_68825' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_68825', get_xdata_call_result_69399)
        
        # Assigning a Call to a Name (line 278):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_69402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 8), 'int')
        # Processing the call keyword arguments
        kwargs_69403 = {}
        # Getting the type of 'call_assignment_68825' (line 278)
        call_assignment_68825_69400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_68825', False)
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___69401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), call_assignment_68825_69400, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_69404 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___69401, *[int_69402], **kwargs_69403)
        
        # Assigning a type to the variable 'call_assignment_68826' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_68826', getitem___call_result_69404)
        
        # Assigning a Name to a Name (line 278):
        # Getting the type of 'call_assignment_68826' (line 278)
        call_assignment_68826_69405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_68826')
        # Assigning a type to the variable 'xdata' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'xdata', call_assignment_68826_69405)
        
        # Assigning a Call to a Name (line 278):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_69408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 8), 'int')
        # Processing the call keyword arguments
        kwargs_69409 = {}
        # Getting the type of 'call_assignment_68825' (line 278)
        call_assignment_68825_69406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_68825', False)
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___69407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), call_assignment_68825_69406, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_69410 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___69407, *[int_69408], **kwargs_69409)
        
        # Assigning a type to the variable 'call_assignment_68827' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_68827', getitem___call_result_69410)
        
        # Assigning a Name to a Name (line 278):
        # Getting the type of 'call_assignment_68827' (line 278)
        call_assignment_68827_69411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'call_assignment_68827')
        # Assigning a type to the variable 'xdata_marker' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 15), 'xdata_marker', call_assignment_68827_69411)
        
        # Assigning a BinOp to a Name (line 280):
        
        # Assigning a BinOp to a Name (line 280):
        # Getting the type of 'height' (line 280)
        height_69412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 18), 'height')
        # Getting the type of 'ydescent' (line 280)
        ydescent_69413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 27), 'ydescent')
        # Applying the binary operator '-' (line 280)
        result_sub_69414 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 18), '-', height_69412, ydescent_69413)
        
        float_69415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 39), 'float')
        # Applying the binary operator 'div' (line 280)
        result_div_69416 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 17), 'div', result_sub_69414, float_69415)
        
        
        # Call to ones(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'xdata' (line 280)
        xdata_69419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 53), 'xdata', False)
        # Obtaining the member 'shape' of a type (line 280)
        shape_69420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 53), xdata_69419, 'shape')
        # Getting the type of 'float' (line 280)
        float_69421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 66), 'float', False)
        # Processing the call keyword arguments (line 280)
        kwargs_69422 = {}
        # Getting the type of 'np' (line 280)
        np_69417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 45), 'np', False)
        # Obtaining the member 'ones' of a type (line 280)
        ones_69418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 45), np_69417, 'ones')
        # Calling ones(args, kwargs) (line 280)
        ones_call_result_69423 = invoke(stypy.reporting.localization.Localization(__file__, 280, 45), ones_69418, *[shape_69420, float_69421], **kwargs_69422)
        
        # Applying the binary operator '*' (line 280)
        result_mul_69424 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 16), '*', result_div_69416, ones_call_result_69423)
        
        # Assigning a type to the variable 'ydata' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'ydata', result_mul_69424)
        
        # Assigning a Call to a Name (line 281):
        
        # Assigning a Call to a Name (line 281):
        
        # Call to Line2D(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'xdata' (line 281)
        xdata_69426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 25), 'xdata', False)
        # Getting the type of 'ydata' (line 281)
        ydata_69427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 32), 'ydata', False)
        # Processing the call keyword arguments (line 281)
        kwargs_69428 = {}
        # Getting the type of 'Line2D' (line 281)
        Line2D_69425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 18), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 281)
        Line2D_call_result_69429 = invoke(stypy.reporting.localization.Localization(__file__, 281, 18), Line2D_69425, *[xdata_69426, ydata_69427], **kwargs_69428)
        
        # Assigning a type to the variable 'legline' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'legline', Line2D_call_result_69429)
        
        # Call to update_prop(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'legline' (line 283)
        legline_69432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 25), 'legline', False)
        # Getting the type of 'orig_handle' (line 283)
        orig_handle_69433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 34), 'orig_handle', False)
        # Getting the type of 'legend' (line 283)
        legend_69434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 47), 'legend', False)
        # Processing the call keyword arguments (line 283)
        kwargs_69435 = {}
        # Getting the type of 'self' (line 283)
        self_69430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 283)
        update_prop_69431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_69430, 'update_prop')
        # Calling update_prop(args, kwargs) (line 283)
        update_prop_call_result_69436 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), update_prop_69431, *[legline_69432, orig_handle_69433, legend_69434], **kwargs_69435)
        
        
        # Call to set_transform(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'trans' (line 284)
        trans_69439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 30), 'trans', False)
        # Processing the call keyword arguments (line 284)
        kwargs_69440 = {}
        # Getting the type of 'legline' (line 284)
        legline_69437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'legline', False)
        # Obtaining the member 'set_transform' of a type (line 284)
        set_transform_69438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), legline_69437, 'set_transform')
        # Calling set_transform(args, kwargs) (line 284)
        set_transform_call_result_69441 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), set_transform_69438, *[trans_69439], **kwargs_69440)
        
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_69442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        # Adding element type (line 286)
        # Getting the type of 'legline' (line 286)
        legline_69443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'legline')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 15), list_69442, legline_69443)
        
        # Assigning a type to the variable 'stypy_return_type' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'stypy_return_type', list_69442)
        
        # ################# End of 'create_artists(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_artists' in the type store
        # Getting the type of 'stypy_return_type' (line 275)
        stypy_return_type_69444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69444)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_artists'
        return stypy_return_type_69444


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 257, 0, False)
        # Assigning a type to the variable 'self' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerLineCollection.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'HandlerLineCollection' (line 257)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 0), 'HandlerLineCollection', HandlerLineCollection)
# Declaration of the 'HandlerRegularPolyCollection' class
# Getting the type of 'HandlerNpointsYoffsets' (line 289)
HandlerNpointsYoffsets_69445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 35), 'HandlerNpointsYoffsets')

class HandlerRegularPolyCollection(HandlerNpointsYoffsets_69445, ):
    unicode_69446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, (-1)), 'unicode', u'\n    Handler for RegularPolyCollections.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 293)
        None_69447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 32), 'None')
        # Getting the type of 'None' (line 293)
        None_69448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 44), 'None')
        defaults = [None_69447, None_69448]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerRegularPolyCollection.__init__', ['yoffsets', 'sizes'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['yoffsets', 'sizes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'self' (line 294)
        self_69451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 40), 'self', False)
        # Processing the call keyword arguments (line 294)
        # Getting the type of 'yoffsets' (line 294)
        yoffsets_69452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 55), 'yoffsets', False)
        keyword_69453 = yoffsets_69452
        # Getting the type of 'kw' (line 294)
        kw_69454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 67), 'kw', False)
        kwargs_69455 = {'yoffsets': keyword_69453, 'kw_69454': kw_69454}
        # Getting the type of 'HandlerNpointsYoffsets' (line 294)
        HandlerNpointsYoffsets_69449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'HandlerNpointsYoffsets', False)
        # Obtaining the member '__init__' of a type (line 294)
        init___69450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), HandlerNpointsYoffsets_69449, '__init__')
        # Calling __init__(args, kwargs) (line 294)
        init___call_result_69456 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), init___69450, *[self_69451], **kwargs_69455)
        
        
        # Assigning a Name to a Attribute (line 296):
        
        # Assigning a Name to a Attribute (line 296):
        # Getting the type of 'sizes' (line 296)
        sizes_69457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 22), 'sizes')
        # Getting the type of 'self' (line 296)
        self_69458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'self')
        # Setting the type of the member '_sizes' of a type (line 296)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), self_69458, '_sizes', sizes_69457)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_numpoints(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_numpoints'
        module_type_store = module_type_store.open_function_context('get_numpoints', 298, 4, False)
        # Assigning a type to the variable 'self' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerRegularPolyCollection.get_numpoints.__dict__.__setitem__('stypy_localization', localization)
        HandlerRegularPolyCollection.get_numpoints.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerRegularPolyCollection.get_numpoints.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerRegularPolyCollection.get_numpoints.__dict__.__setitem__('stypy_function_name', 'HandlerRegularPolyCollection.get_numpoints')
        HandlerRegularPolyCollection.get_numpoints.__dict__.__setitem__('stypy_param_names_list', ['legend'])
        HandlerRegularPolyCollection.get_numpoints.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerRegularPolyCollection.get_numpoints.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerRegularPolyCollection.get_numpoints.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerRegularPolyCollection.get_numpoints.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerRegularPolyCollection.get_numpoints.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerRegularPolyCollection.get_numpoints.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerRegularPolyCollection.get_numpoints', ['legend'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_numpoints', localization, ['legend'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_numpoints(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 299)
        # Getting the type of 'self' (line 299)
        self_69459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 11), 'self')
        # Obtaining the member '_numpoints' of a type (line 299)
        _numpoints_69460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 11), self_69459, '_numpoints')
        # Getting the type of 'None' (line 299)
        None_69461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 30), 'None')
        
        (may_be_69462, more_types_in_union_69463) = may_be_none(_numpoints_69460, None_69461)

        if may_be_69462:

            if more_types_in_union_69463:
                # Runtime conditional SSA (line 299)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'legend' (line 300)
            legend_69464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 19), 'legend')
            # Obtaining the member 'scatterpoints' of a type (line 300)
            scatterpoints_69465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 19), legend_69464, 'scatterpoints')
            # Assigning a type to the variable 'stypy_return_type' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'stypy_return_type', scatterpoints_69465)

            if more_types_in_union_69463:
                # Runtime conditional SSA for else branch (line 299)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_69462) or more_types_in_union_69463):
            # Getting the type of 'self' (line 302)
            self_69466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 19), 'self')
            # Obtaining the member '_numpoints' of a type (line 302)
            _numpoints_69467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 19), self_69466, '_numpoints')
            # Assigning a type to the variable 'stypy_return_type' (line 302)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'stypy_return_type', _numpoints_69467)

            if (may_be_69462 and more_types_in_union_69463):
                # SSA join for if statement (line 299)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'get_numpoints(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_numpoints' in the type store
        # Getting the type of 'stypy_return_type' (line 298)
        stypy_return_type_69468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69468)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_numpoints'
        return stypy_return_type_69468


    @norecursion
    def get_sizes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_sizes'
        module_type_store = module_type_store.open_function_context('get_sizes', 304, 4, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerRegularPolyCollection.get_sizes.__dict__.__setitem__('stypy_localization', localization)
        HandlerRegularPolyCollection.get_sizes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerRegularPolyCollection.get_sizes.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerRegularPolyCollection.get_sizes.__dict__.__setitem__('stypy_function_name', 'HandlerRegularPolyCollection.get_sizes')
        HandlerRegularPolyCollection.get_sizes.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'])
        HandlerRegularPolyCollection.get_sizes.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerRegularPolyCollection.get_sizes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerRegularPolyCollection.get_sizes.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerRegularPolyCollection.get_sizes.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerRegularPolyCollection.get_sizes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerRegularPolyCollection.get_sizes.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerRegularPolyCollection.get_sizes', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_sizes', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_sizes(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 306)
        # Getting the type of 'self' (line 306)
        self_69469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 11), 'self')
        # Obtaining the member '_sizes' of a type (line 306)
        _sizes_69470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 11), self_69469, '_sizes')
        # Getting the type of 'None' (line 306)
        None_69471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 26), 'None')
        
        (may_be_69472, more_types_in_union_69473) = may_be_none(_sizes_69470, None_69471)

        if may_be_69472:

            if more_types_in_union_69473:
                # Runtime conditional SSA (line 306)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 307):
            
            # Assigning a Call to a Name (line 307):
            
            # Call to get_sizes(...): (line 307)
            # Processing the call keyword arguments (line 307)
            kwargs_69476 = {}
            # Getting the type of 'orig_handle' (line 307)
            orig_handle_69474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 27), 'orig_handle', False)
            # Obtaining the member 'get_sizes' of a type (line 307)
            get_sizes_69475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 27), orig_handle_69474, 'get_sizes')
            # Calling get_sizes(args, kwargs) (line 307)
            get_sizes_call_result_69477 = invoke(stypy.reporting.localization.Localization(__file__, 307, 27), get_sizes_69475, *[], **kwargs_69476)
            
            # Assigning a type to the variable 'handle_sizes' (line 307)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'handle_sizes', get_sizes_call_result_69477)
            
            
            
            # Call to len(...): (line 308)
            # Processing the call arguments (line 308)
            # Getting the type of 'handle_sizes' (line 308)
            handle_sizes_69479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 23), 'handle_sizes', False)
            # Processing the call keyword arguments (line 308)
            kwargs_69480 = {}
            # Getting the type of 'len' (line 308)
            len_69478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 19), 'len', False)
            # Calling len(args, kwargs) (line 308)
            len_call_result_69481 = invoke(stypy.reporting.localization.Localization(__file__, 308, 19), len_69478, *[handle_sizes_69479], **kwargs_69480)
            
            # Applying the 'not' unary operator (line 308)
            result_not__69482 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 15), 'not', len_call_result_69481)
            
            # Testing the type of an if condition (line 308)
            if_condition_69483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 12), result_not__69482)
            # Assigning a type to the variable 'if_condition_69483' (line 308)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'if_condition_69483', if_condition_69483)
            # SSA begins for if statement (line 308)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a List to a Name (line 309):
            
            # Assigning a List to a Name (line 309):
            
            # Obtaining an instance of the builtin type 'list' (line 309)
            list_69484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 309)
            # Adding element type (line 309)
            int_69485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 32), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 31), list_69484, int_69485)
            
            # Assigning a type to the variable 'handle_sizes' (line 309)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'handle_sizes', list_69484)
            # SSA join for if statement (line 308)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 310):
            
            # Assigning a BinOp to a Name (line 310):
            
            # Call to max(...): (line 310)
            # Processing the call arguments (line 310)
            # Getting the type of 'handle_sizes' (line 310)
            handle_sizes_69487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 27), 'handle_sizes', False)
            # Processing the call keyword arguments (line 310)
            kwargs_69488 = {}
            # Getting the type of 'max' (line 310)
            max_69486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 23), 'max', False)
            # Calling max(args, kwargs) (line 310)
            max_call_result_69489 = invoke(stypy.reporting.localization.Localization(__file__, 310, 23), max_69486, *[handle_sizes_69487], **kwargs_69488)
            
            # Getting the type of 'legend' (line 310)
            legend_69490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 43), 'legend')
            # Obtaining the member 'markerscale' of a type (line 310)
            markerscale_69491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 43), legend_69490, 'markerscale')
            int_69492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 65), 'int')
            # Applying the binary operator '**' (line 310)
            result_pow_69493 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 43), '**', markerscale_69491, int_69492)
            
            # Applying the binary operator '*' (line 310)
            result_mul_69494 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 23), '*', max_call_result_69489, result_pow_69493)
            
            # Assigning a type to the variable 'size_max' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'size_max', result_mul_69494)
            
            # Assigning a BinOp to a Name (line 311):
            
            # Assigning a BinOp to a Name (line 311):
            
            # Call to min(...): (line 311)
            # Processing the call arguments (line 311)
            # Getting the type of 'handle_sizes' (line 311)
            handle_sizes_69496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 27), 'handle_sizes', False)
            # Processing the call keyword arguments (line 311)
            kwargs_69497 = {}
            # Getting the type of 'min' (line 311)
            min_69495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 23), 'min', False)
            # Calling min(args, kwargs) (line 311)
            min_call_result_69498 = invoke(stypy.reporting.localization.Localization(__file__, 311, 23), min_69495, *[handle_sizes_69496], **kwargs_69497)
            
            # Getting the type of 'legend' (line 311)
            legend_69499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 43), 'legend')
            # Obtaining the member 'markerscale' of a type (line 311)
            markerscale_69500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 43), legend_69499, 'markerscale')
            int_69501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 65), 'int')
            # Applying the binary operator '**' (line 311)
            result_pow_69502 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 43), '**', markerscale_69500, int_69501)
            
            # Applying the binary operator '*' (line 311)
            result_mul_69503 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 23), '*', min_call_result_69498, result_pow_69502)
            
            # Assigning a type to the variable 'size_min' (line 311)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'size_min', result_mul_69503)
            
            # Assigning a Call to a Name (line 313):
            
            # Assigning a Call to a Name (line 313):
            
            # Call to get_numpoints(...): (line 313)
            # Processing the call arguments (line 313)
            # Getting the type of 'legend' (line 313)
            legend_69506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 43), 'legend', False)
            # Processing the call keyword arguments (line 313)
            kwargs_69507 = {}
            # Getting the type of 'self' (line 313)
            self_69504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 24), 'self', False)
            # Obtaining the member 'get_numpoints' of a type (line 313)
            get_numpoints_69505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 24), self_69504, 'get_numpoints')
            # Calling get_numpoints(args, kwargs) (line 313)
            get_numpoints_call_result_69508 = invoke(stypy.reporting.localization.Localization(__file__, 313, 24), get_numpoints_69505, *[legend_69506], **kwargs_69507)
            
            # Assigning a type to the variable 'numpoints' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'numpoints', get_numpoints_call_result_69508)
            
            
            # Getting the type of 'numpoints' (line 314)
            numpoints_69509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), 'numpoints')
            int_69510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 27), 'int')
            # Applying the binary operator '<' (line 314)
            result_lt_69511 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 15), '<', numpoints_69509, int_69510)
            
            # Testing the type of an if condition (line 314)
            if_condition_69512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 12), result_lt_69511)
            # Assigning a type to the variable 'if_condition_69512' (line 314)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'if_condition_69512', if_condition_69512)
            # SSA begins for if statement (line 314)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 315):
            
            # Assigning a Subscript to a Name (line 315):
            
            # Obtaining the type of the subscript
            # Getting the type of 'numpoints' (line 316)
            numpoints_69513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 36), 'numpoints')
            slice_69514 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 315, 24), None, numpoints_69513, None)
            
            # Obtaining an instance of the builtin type 'list' (line 315)
            list_69515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 315)
            # Adding element type (line 315)
            float_69516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 25), 'float')
            # Getting the type of 'size_max' (line 315)
            size_max_69517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 31), 'size_max')
            # Getting the type of 'size_min' (line 315)
            size_min_69518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 42), 'size_min')
            # Applying the binary operator '+' (line 315)
            result_add_69519 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 31), '+', size_max_69517, size_min_69518)
            
            # Applying the binary operator '*' (line 315)
            result_mul_69520 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 25), '*', float_69516, result_add_69519)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 24), list_69515, result_mul_69520)
            # Adding element type (line 315)
            # Getting the type of 'size_max' (line 315)
            size_max_69521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 53), 'size_max')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 24), list_69515, size_max_69521)
            # Adding element type (line 315)
            # Getting the type of 'size_min' (line 316)
            size_min_69522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 25), 'size_min')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 24), list_69515, size_min_69522)
            
            # Obtaining the member '__getitem__' of a type (line 315)
            getitem___69523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 24), list_69515, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 315)
            subscript_call_result_69524 = invoke(stypy.reporting.localization.Localization(__file__, 315, 24), getitem___69523, slice_69514)
            
            # Assigning a type to the variable 'sizes' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'sizes', subscript_call_result_69524)
            # SSA branch for the else part of an if statement (line 314)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 318):
            
            # Assigning a BinOp to a Name (line 318):
            # Getting the type of 'size_max' (line 318)
            size_max_69525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 23), 'size_max')
            # Getting the type of 'size_min' (line 318)
            size_min_69526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 34), 'size_min')
            # Applying the binary operator '-' (line 318)
            result_sub_69527 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 23), '-', size_max_69525, size_min_69526)
            
            # Assigning a type to the variable 'rng' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'rng', result_sub_69527)
            
            # Assigning a BinOp to a Name (line 319):
            
            # Assigning a BinOp to a Name (line 319):
            # Getting the type of 'rng' (line 319)
            rng_69528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 24), 'rng')
            
            # Call to linspace(...): (line 319)
            # Processing the call arguments (line 319)
            int_69531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 42), 'int')
            int_69532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 45), 'int')
            # Getting the type of 'numpoints' (line 319)
            numpoints_69533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 48), 'numpoints', False)
            # Processing the call keyword arguments (line 319)
            kwargs_69534 = {}
            # Getting the type of 'np' (line 319)
            np_69529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 30), 'np', False)
            # Obtaining the member 'linspace' of a type (line 319)
            linspace_69530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 30), np_69529, 'linspace')
            # Calling linspace(args, kwargs) (line 319)
            linspace_call_result_69535 = invoke(stypy.reporting.localization.Localization(__file__, 319, 30), linspace_69530, *[int_69531, int_69532, numpoints_69533], **kwargs_69534)
            
            # Applying the binary operator '*' (line 319)
            result_mul_69536 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 24), '*', rng_69528, linspace_call_result_69535)
            
            # Getting the type of 'size_min' (line 319)
            size_min_69537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 61), 'size_min')
            # Applying the binary operator '+' (line 319)
            result_add_69538 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 24), '+', result_mul_69536, size_min_69537)
            
            # Assigning a type to the variable 'sizes' (line 319)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 16), 'sizes', result_add_69538)
            # SSA join for if statement (line 314)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_69473:
                # Runtime conditional SSA for else branch (line 306)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_69472) or more_types_in_union_69473):
            
            # Assigning a Attribute to a Name (line 321):
            
            # Assigning a Attribute to a Name (line 321):
            # Getting the type of 'self' (line 321)
            self_69539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 20), 'self')
            # Obtaining the member '_sizes' of a type (line 321)
            _sizes_69540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 20), self_69539, '_sizes')
            # Assigning a type to the variable 'sizes' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'sizes', _sizes_69540)

            if (may_be_69472 and more_types_in_union_69473):
                # SSA join for if statement (line 306)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'sizes' (line 323)
        sizes_69541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'sizes')
        # Assigning a type to the variable 'stypy_return_type' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'stypy_return_type', sizes_69541)
        
        # ################# End of 'get_sizes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_sizes' in the type store
        # Getting the type of 'stypy_return_type' (line 304)
        stypy_return_type_69542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69542)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_sizes'
        return stypy_return_type_69542


    @norecursion
    def update_prop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_prop'
        module_type_store = module_type_store.open_function_context('update_prop', 325, 4, False)
        # Assigning a type to the variable 'self' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerRegularPolyCollection.update_prop.__dict__.__setitem__('stypy_localization', localization)
        HandlerRegularPolyCollection.update_prop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerRegularPolyCollection.update_prop.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerRegularPolyCollection.update_prop.__dict__.__setitem__('stypy_function_name', 'HandlerRegularPolyCollection.update_prop')
        HandlerRegularPolyCollection.update_prop.__dict__.__setitem__('stypy_param_names_list', ['legend_handle', 'orig_handle', 'legend'])
        HandlerRegularPolyCollection.update_prop.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerRegularPolyCollection.update_prop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerRegularPolyCollection.update_prop.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerRegularPolyCollection.update_prop.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerRegularPolyCollection.update_prop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerRegularPolyCollection.update_prop.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerRegularPolyCollection.update_prop', ['legend_handle', 'orig_handle', 'legend'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_prop', localization, ['legend_handle', 'orig_handle', 'legend'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_prop(...)' code ##################

        
        # Call to _update_prop(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'legend_handle' (line 327)
        legend_handle_69545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 26), 'legend_handle', False)
        # Getting the type of 'orig_handle' (line 327)
        orig_handle_69546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 41), 'orig_handle', False)
        # Processing the call keyword arguments (line 327)
        kwargs_69547 = {}
        # Getting the type of 'self' (line 327)
        self_69543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'self', False)
        # Obtaining the member '_update_prop' of a type (line 327)
        _update_prop_69544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 8), self_69543, '_update_prop')
        # Calling _update_prop(args, kwargs) (line 327)
        _update_prop_call_result_69548 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), _update_prop_69544, *[legend_handle_69545, orig_handle_69546], **kwargs_69547)
        
        
        # Call to set_figure(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'legend' (line 329)
        legend_69551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 33), 'legend', False)
        # Obtaining the member 'figure' of a type (line 329)
        figure_69552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 33), legend_69551, 'figure')
        # Processing the call keyword arguments (line 329)
        kwargs_69553 = {}
        # Getting the type of 'legend_handle' (line 329)
        legend_handle_69549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'legend_handle', False)
        # Obtaining the member 'set_figure' of a type (line 329)
        set_figure_69550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), legend_handle_69549, 'set_figure')
        # Calling set_figure(args, kwargs) (line 329)
        set_figure_call_result_69554 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), set_figure_69550, *[figure_69552], **kwargs_69553)
        
        
        # Call to set_clip_box(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'None' (line 331)
        None_69557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 35), 'None', False)
        # Processing the call keyword arguments (line 331)
        kwargs_69558 = {}
        # Getting the type of 'legend_handle' (line 331)
        legend_handle_69555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'legend_handle', False)
        # Obtaining the member 'set_clip_box' of a type (line 331)
        set_clip_box_69556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), legend_handle_69555, 'set_clip_box')
        # Calling set_clip_box(args, kwargs) (line 331)
        set_clip_box_call_result_69559 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), set_clip_box_69556, *[None_69557], **kwargs_69558)
        
        
        # Call to set_clip_path(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'None' (line 332)
        None_69562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 36), 'None', False)
        # Processing the call keyword arguments (line 332)
        kwargs_69563 = {}
        # Getting the type of 'legend_handle' (line 332)
        legend_handle_69560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'legend_handle', False)
        # Obtaining the member 'set_clip_path' of a type (line 332)
        set_clip_path_69561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), legend_handle_69560, 'set_clip_path')
        # Calling set_clip_path(args, kwargs) (line 332)
        set_clip_path_call_result_69564 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), set_clip_path_69561, *[None_69562], **kwargs_69563)
        
        
        # ################# End of 'update_prop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_prop' in the type store
        # Getting the type of 'stypy_return_type' (line 325)
        stypy_return_type_69565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69565)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_prop'
        return stypy_return_type_69565


    @norecursion
    def create_collection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_collection'
        module_type_store = module_type_store.open_function_context('create_collection', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerRegularPolyCollection.create_collection.__dict__.__setitem__('stypy_localization', localization)
        HandlerRegularPolyCollection.create_collection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerRegularPolyCollection.create_collection.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerRegularPolyCollection.create_collection.__dict__.__setitem__('stypy_function_name', 'HandlerRegularPolyCollection.create_collection')
        HandlerRegularPolyCollection.create_collection.__dict__.__setitem__('stypy_param_names_list', ['orig_handle', 'sizes', 'offsets', 'transOffset'])
        HandlerRegularPolyCollection.create_collection.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerRegularPolyCollection.create_collection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerRegularPolyCollection.create_collection.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerRegularPolyCollection.create_collection.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerRegularPolyCollection.create_collection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerRegularPolyCollection.create_collection.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerRegularPolyCollection.create_collection', ['orig_handle', 'sizes', 'offsets', 'transOffset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_collection', localization, ['orig_handle', 'sizes', 'offsets', 'transOffset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_collection(...)' code ##################

        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to (...): (line 335)
        # Processing the call arguments (line 335)
        
        # Call to get_numsides(...): (line 335)
        # Processing the call keyword arguments (line 335)
        kwargs_69572 = {}
        # Getting the type of 'orig_handle' (line 335)
        orig_handle_69570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 30), 'orig_handle', False)
        # Obtaining the member 'get_numsides' of a type (line 335)
        get_numsides_69571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 30), orig_handle_69570, 'get_numsides')
        # Calling get_numsides(args, kwargs) (line 335)
        get_numsides_call_result_69573 = invoke(stypy.reporting.localization.Localization(__file__, 335, 30), get_numsides_69571, *[], **kwargs_69572)
        
        # Processing the call keyword arguments (line 335)
        
        # Call to get_rotation(...): (line 336)
        # Processing the call keyword arguments (line 336)
        kwargs_69576 = {}
        # Getting the type of 'orig_handle' (line 336)
        orig_handle_69574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 39), 'orig_handle', False)
        # Obtaining the member 'get_rotation' of a type (line 336)
        get_rotation_69575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 39), orig_handle_69574, 'get_rotation')
        # Calling get_rotation(args, kwargs) (line 336)
        get_rotation_call_result_69577 = invoke(stypy.reporting.localization.Localization(__file__, 336, 39), get_rotation_69575, *[], **kwargs_69576)
        
        keyword_69578 = get_rotation_call_result_69577
        # Getting the type of 'sizes' (line 337)
        sizes_69579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 36), 'sizes', False)
        keyword_69580 = sizes_69579
        # Getting the type of 'offsets' (line 338)
        offsets_69581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 38), 'offsets', False)
        keyword_69582 = offsets_69581
        # Getting the type of 'transOffset' (line 339)
        transOffset_69583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 42), 'transOffset', False)
        keyword_69584 = transOffset_69583
        kwargs_69585 = {'rotation': keyword_69578, 'offsets': keyword_69582, 'transOffset': keyword_69584, 'sizes': keyword_69580}
        
        # Call to type(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'orig_handle' (line 335)
        orig_handle_69567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), 'orig_handle', False)
        # Processing the call keyword arguments (line 335)
        kwargs_69568 = {}
        # Getting the type of 'type' (line 335)
        type_69566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'type', False)
        # Calling type(args, kwargs) (line 335)
        type_call_result_69569 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), type_69566, *[orig_handle_69567], **kwargs_69568)
        
        # Calling (args, kwargs) (line 335)
        _call_result_69586 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), type_call_result_69569, *[get_numsides_call_result_69573], **kwargs_69585)
        
        # Assigning a type to the variable 'p' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'p', _call_result_69586)
        # Getting the type of 'p' (line 341)
        p_69587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'p')
        # Assigning a type to the variable 'stypy_return_type' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'stypy_return_type', p_69587)
        
        # ################# End of 'create_collection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_collection' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_69588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69588)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_collection'
        return stypy_return_type_69588


    @norecursion
    def create_artists(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_artists'
        module_type_store = module_type_store.open_function_context('create_artists', 343, 4, False)
        # Assigning a type to the variable 'self' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerRegularPolyCollection.create_artists.__dict__.__setitem__('stypy_localization', localization)
        HandlerRegularPolyCollection.create_artists.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerRegularPolyCollection.create_artists.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerRegularPolyCollection.create_artists.__dict__.__setitem__('stypy_function_name', 'HandlerRegularPolyCollection.create_artists')
        HandlerRegularPolyCollection.create_artists.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'])
        HandlerRegularPolyCollection.create_artists.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerRegularPolyCollection.create_artists.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerRegularPolyCollection.create_artists.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerRegularPolyCollection.create_artists.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerRegularPolyCollection.create_artists.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerRegularPolyCollection.create_artists.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerRegularPolyCollection.create_artists', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_artists', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_artists(...)' code ##################

        
        # Assigning a Call to a Tuple (line 346):
        
        # Assigning a Call to a Name:
        
        # Call to get_xdata(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'legend' (line 346)
        legend_69591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 45), 'legend', False)
        # Getting the type of 'xdescent' (line 346)
        xdescent_69592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 53), 'xdescent', False)
        # Getting the type of 'ydescent' (line 346)
        ydescent_69593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 63), 'ydescent', False)
        # Getting the type of 'width' (line 347)
        width_69594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 45), 'width', False)
        # Getting the type of 'height' (line 347)
        height_69595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 52), 'height', False)
        # Getting the type of 'fontsize' (line 347)
        fontsize_69596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 60), 'fontsize', False)
        # Processing the call keyword arguments (line 346)
        kwargs_69597 = {}
        # Getting the type of 'self' (line 346)
        self_69589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 30), 'self', False)
        # Obtaining the member 'get_xdata' of a type (line 346)
        get_xdata_69590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 30), self_69589, 'get_xdata')
        # Calling get_xdata(args, kwargs) (line 346)
        get_xdata_call_result_69598 = invoke(stypy.reporting.localization.Localization(__file__, 346, 30), get_xdata_69590, *[legend_69591, xdescent_69592, ydescent_69593, width_69594, height_69595, fontsize_69596], **kwargs_69597)
        
        # Assigning a type to the variable 'call_assignment_68828' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'call_assignment_68828', get_xdata_call_result_69598)
        
        # Assigning a Call to a Name (line 346):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_69601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 8), 'int')
        # Processing the call keyword arguments
        kwargs_69602 = {}
        # Getting the type of 'call_assignment_68828' (line 346)
        call_assignment_68828_69599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'call_assignment_68828', False)
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___69600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), call_assignment_68828_69599, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_69603 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___69600, *[int_69601], **kwargs_69602)
        
        # Assigning a type to the variable 'call_assignment_68829' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'call_assignment_68829', getitem___call_result_69603)
        
        # Assigning a Name to a Name (line 346):
        # Getting the type of 'call_assignment_68829' (line 346)
        call_assignment_68829_69604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'call_assignment_68829')
        # Assigning a type to the variable 'xdata' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'xdata', call_assignment_68829_69604)
        
        # Assigning a Call to a Name (line 346):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_69607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 8), 'int')
        # Processing the call keyword arguments
        kwargs_69608 = {}
        # Getting the type of 'call_assignment_68828' (line 346)
        call_assignment_68828_69605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'call_assignment_68828', False)
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___69606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), call_assignment_68828_69605, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_69609 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___69606, *[int_69607], **kwargs_69608)
        
        # Assigning a type to the variable 'call_assignment_68830' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'call_assignment_68830', getitem___call_result_69609)
        
        # Assigning a Name to a Name (line 346):
        # Getting the type of 'call_assignment_68830' (line 346)
        call_assignment_68830_69610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'call_assignment_68830')
        # Assigning a type to the variable 'xdata_marker' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'xdata_marker', call_assignment_68830_69610)
        
        # Assigning a Call to a Name (line 349):
        
        # Assigning a Call to a Name (line 349):
        
        # Call to get_ydata(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'legend' (line 349)
        legend_69613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 31), 'legend', False)
        # Getting the type of 'xdescent' (line 349)
        xdescent_69614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 39), 'xdescent', False)
        # Getting the type of 'ydescent' (line 349)
        ydescent_69615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 49), 'ydescent', False)
        # Getting the type of 'width' (line 350)
        width_69616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 31), 'width', False)
        # Getting the type of 'height' (line 350)
        height_69617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 38), 'height', False)
        # Getting the type of 'fontsize' (line 350)
        fontsize_69618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 46), 'fontsize', False)
        # Processing the call keyword arguments (line 349)
        kwargs_69619 = {}
        # Getting the type of 'self' (line 349)
        self_69611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'self', False)
        # Obtaining the member 'get_ydata' of a type (line 349)
        get_ydata_69612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 16), self_69611, 'get_ydata')
        # Calling get_ydata(args, kwargs) (line 349)
        get_ydata_call_result_69620 = invoke(stypy.reporting.localization.Localization(__file__, 349, 16), get_ydata_69612, *[legend_69613, xdescent_69614, ydescent_69615, width_69616, height_69617, fontsize_69618], **kwargs_69619)
        
        # Assigning a type to the variable 'ydata' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'ydata', get_ydata_call_result_69620)
        
        # Assigning a Call to a Name (line 352):
        
        # Assigning a Call to a Name (line 352):
        
        # Call to get_sizes(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'legend' (line 352)
        legend_69623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 31), 'legend', False)
        # Getting the type of 'orig_handle' (line 352)
        orig_handle_69624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 39), 'orig_handle', False)
        # Getting the type of 'xdescent' (line 352)
        xdescent_69625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 52), 'xdescent', False)
        # Getting the type of 'ydescent' (line 352)
        ydescent_69626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 62), 'ydescent', False)
        # Getting the type of 'width' (line 353)
        width_69627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 31), 'width', False)
        # Getting the type of 'height' (line 353)
        height_69628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 38), 'height', False)
        # Getting the type of 'fontsize' (line 353)
        fontsize_69629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 46), 'fontsize', False)
        # Processing the call keyword arguments (line 352)
        kwargs_69630 = {}
        # Getting the type of 'self' (line 352)
        self_69621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'self', False)
        # Obtaining the member 'get_sizes' of a type (line 352)
        get_sizes_69622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 16), self_69621, 'get_sizes')
        # Calling get_sizes(args, kwargs) (line 352)
        get_sizes_call_result_69631 = invoke(stypy.reporting.localization.Localization(__file__, 352, 16), get_sizes_69622, *[legend_69623, orig_handle_69624, xdescent_69625, ydescent_69626, width_69627, height_69628, fontsize_69629], **kwargs_69630)
        
        # Assigning a type to the variable 'sizes' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'sizes', get_sizes_call_result_69631)
        
        # Assigning a Call to a Name (line 355):
        
        # Assigning a Call to a Name (line 355):
        
        # Call to create_collection(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'orig_handle' (line 355)
        orig_handle_69634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 35), 'orig_handle', False)
        # Getting the type of 'sizes' (line 355)
        sizes_69635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 48), 'sizes', False)
        # Processing the call keyword arguments (line 355)
        
        # Call to list(...): (line 356)
        # Processing the call arguments (line 356)
        
        # Call to zip(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'xdata_marker' (line 356)
        xdata_marker_69638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 52), 'xdata_marker', False)
        # Getting the type of 'ydata' (line 356)
        ydata_69639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 66), 'ydata', False)
        # Processing the call keyword arguments (line 356)
        kwargs_69640 = {}
        # Getting the type of 'zip' (line 356)
        zip_69637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 48), 'zip', False)
        # Calling zip(args, kwargs) (line 356)
        zip_call_result_69641 = invoke(stypy.reporting.localization.Localization(__file__, 356, 48), zip_69637, *[xdata_marker_69638, ydata_69639], **kwargs_69640)
        
        # Processing the call keyword arguments (line 356)
        kwargs_69642 = {}
        # Getting the type of 'list' (line 356)
        list_69636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 43), 'list', False)
        # Calling list(args, kwargs) (line 356)
        list_call_result_69643 = invoke(stypy.reporting.localization.Localization(__file__, 356, 43), list_69636, *[zip_call_result_69641], **kwargs_69642)
        
        keyword_69644 = list_call_result_69643
        # Getting the type of 'trans' (line 357)
        trans_69645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 47), 'trans', False)
        keyword_69646 = trans_69645
        kwargs_69647 = {'transOffset': keyword_69646, 'offsets': keyword_69644}
        # Getting the type of 'self' (line 355)
        self_69632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'self', False)
        # Obtaining the member 'create_collection' of a type (line 355)
        create_collection_69633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 12), self_69632, 'create_collection')
        # Calling create_collection(args, kwargs) (line 355)
        create_collection_call_result_69648 = invoke(stypy.reporting.localization.Localization(__file__, 355, 12), create_collection_69633, *[orig_handle_69634, sizes_69635], **kwargs_69647)
        
        # Assigning a type to the variable 'p' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'p', create_collection_call_result_69648)
        
        # Call to update_prop(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'p' (line 359)
        p_69651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 25), 'p', False)
        # Getting the type of 'orig_handle' (line 359)
        orig_handle_69652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 28), 'orig_handle', False)
        # Getting the type of 'legend' (line 359)
        legend_69653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 41), 'legend', False)
        # Processing the call keyword arguments (line 359)
        kwargs_69654 = {}
        # Getting the type of 'self' (line 359)
        self_69649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 359)
        update_prop_69650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), self_69649, 'update_prop')
        # Calling update_prop(args, kwargs) (line 359)
        update_prop_call_result_69655 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), update_prop_69650, *[p_69651, orig_handle_69652, legend_69653], **kwargs_69654)
        
        
        # Assigning a Name to a Attribute (line 360):
        
        # Assigning a Name to a Attribute (line 360):
        # Getting the type of 'trans' (line 360)
        trans_69656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 25), 'trans')
        # Getting the type of 'p' (line 360)
        p_69657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'p')
        # Setting the type of the member '_transOffset' of a type (line 360)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 8), p_69657, '_transOffset', trans_69656)
        
        # Obtaining an instance of the builtin type 'list' (line 361)
        list_69658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 361)
        # Adding element type (line 361)
        # Getting the type of 'p' (line 361)
        p_69659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 15), list_69658, p_69659)
        
        # Assigning a type to the variable 'stypy_return_type' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'stypy_return_type', list_69658)
        
        # ################# End of 'create_artists(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_artists' in the type store
        # Getting the type of 'stypy_return_type' (line 343)
        stypy_return_type_69660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69660)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_artists'
        return stypy_return_type_69660


# Assigning a type to the variable 'HandlerRegularPolyCollection' (line 289)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 0), 'HandlerRegularPolyCollection', HandlerRegularPolyCollection)
# Declaration of the 'HandlerPathCollection' class
# Getting the type of 'HandlerRegularPolyCollection' (line 364)
HandlerRegularPolyCollection_69661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 28), 'HandlerRegularPolyCollection')

class HandlerPathCollection(HandlerRegularPolyCollection_69661, ):
    unicode_69662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, (-1)), 'unicode', u'\n    Handler for PathCollections, which are used by scatter\n    ')

    @norecursion
    def create_collection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_collection'
        module_type_store = module_type_store.open_function_context('create_collection', 368, 4, False)
        # Assigning a type to the variable 'self' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerPathCollection.create_collection.__dict__.__setitem__('stypy_localization', localization)
        HandlerPathCollection.create_collection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerPathCollection.create_collection.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerPathCollection.create_collection.__dict__.__setitem__('stypy_function_name', 'HandlerPathCollection.create_collection')
        HandlerPathCollection.create_collection.__dict__.__setitem__('stypy_param_names_list', ['orig_handle', 'sizes', 'offsets', 'transOffset'])
        HandlerPathCollection.create_collection.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerPathCollection.create_collection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerPathCollection.create_collection.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerPathCollection.create_collection.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerPathCollection.create_collection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerPathCollection.create_collection.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerPathCollection.create_collection', ['orig_handle', 'sizes', 'offsets', 'transOffset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_collection', localization, ['orig_handle', 'sizes', 'offsets', 'transOffset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_collection(...)' code ##################

        
        # Assigning a Call to a Name (line 369):
        
        # Assigning a Call to a Name (line 369):
        
        # Call to (...): (line 369)
        # Processing the call arguments (line 369)
        
        # Obtaining an instance of the builtin type 'list' (line 369)
        list_69667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 369)
        # Adding element type (line 369)
        
        # Obtaining the type of the subscript
        int_69668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 55), 'int')
        
        # Call to get_paths(...): (line 369)
        # Processing the call keyword arguments (line 369)
        kwargs_69671 = {}
        # Getting the type of 'orig_handle' (line 369)
        orig_handle_69669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 31), 'orig_handle', False)
        # Obtaining the member 'get_paths' of a type (line 369)
        get_paths_69670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 31), orig_handle_69669, 'get_paths')
        # Calling get_paths(args, kwargs) (line 369)
        get_paths_call_result_69672 = invoke(stypy.reporting.localization.Localization(__file__, 369, 31), get_paths_69670, *[], **kwargs_69671)
        
        # Obtaining the member '__getitem__' of a type (line 369)
        getitem___69673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 31), get_paths_call_result_69672, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 369)
        subscript_call_result_69674 = invoke(stypy.reporting.localization.Localization(__file__, 369, 31), getitem___69673, int_69668)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 30), list_69667, subscript_call_result_69674)
        
        # Processing the call keyword arguments (line 369)
        # Getting the type of 'sizes' (line 370)
        sizes_69675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 36), 'sizes', False)
        keyword_69676 = sizes_69675
        # Getting the type of 'offsets' (line 371)
        offsets_69677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 38), 'offsets', False)
        keyword_69678 = offsets_69677
        # Getting the type of 'transOffset' (line 372)
        transOffset_69679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 42), 'transOffset', False)
        keyword_69680 = transOffset_69679
        kwargs_69681 = {'offsets': keyword_69678, 'transOffset': keyword_69680, 'sizes': keyword_69676}
        
        # Call to type(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'orig_handle' (line 369)
        orig_handle_69664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 17), 'orig_handle', False)
        # Processing the call keyword arguments (line 369)
        kwargs_69665 = {}
        # Getting the type of 'type' (line 369)
        type_69663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'type', False)
        # Calling type(args, kwargs) (line 369)
        type_call_result_69666 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), type_69663, *[orig_handle_69664], **kwargs_69665)
        
        # Calling (args, kwargs) (line 369)
        _call_result_69682 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), type_call_result_69666, *[list_69667], **kwargs_69681)
        
        # Assigning a type to the variable 'p' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'p', _call_result_69682)
        # Getting the type of 'p' (line 374)
        p_69683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 15), 'p')
        # Assigning a type to the variable 'stypy_return_type' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'stypy_return_type', p_69683)
        
        # ################# End of 'create_collection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_collection' in the type store
        # Getting the type of 'stypy_return_type' (line 368)
        stypy_return_type_69684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69684)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_collection'
        return stypy_return_type_69684


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 364, 0, False)
        # Assigning a type to the variable 'self' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerPathCollection.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'HandlerPathCollection' (line 364)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 0), 'HandlerPathCollection', HandlerPathCollection)
# Declaration of the 'HandlerCircleCollection' class
# Getting the type of 'HandlerRegularPolyCollection' (line 377)
HandlerRegularPolyCollection_69685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 30), 'HandlerRegularPolyCollection')

class HandlerCircleCollection(HandlerRegularPolyCollection_69685, ):
    unicode_69686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, (-1)), 'unicode', u'\n    Handler for CircleCollections\n    ')

    @norecursion
    def create_collection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_collection'
        module_type_store = module_type_store.open_function_context('create_collection', 381, 4, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerCircleCollection.create_collection.__dict__.__setitem__('stypy_localization', localization)
        HandlerCircleCollection.create_collection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerCircleCollection.create_collection.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerCircleCollection.create_collection.__dict__.__setitem__('stypy_function_name', 'HandlerCircleCollection.create_collection')
        HandlerCircleCollection.create_collection.__dict__.__setitem__('stypy_param_names_list', ['orig_handle', 'sizes', 'offsets', 'transOffset'])
        HandlerCircleCollection.create_collection.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerCircleCollection.create_collection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerCircleCollection.create_collection.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerCircleCollection.create_collection.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerCircleCollection.create_collection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerCircleCollection.create_collection.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerCircleCollection.create_collection', ['orig_handle', 'sizes', 'offsets', 'transOffset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_collection', localization, ['orig_handle', 'sizes', 'offsets', 'transOffset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_collection(...)' code ##################

        
        # Assigning a Call to a Name (line 382):
        
        # Assigning a Call to a Name (line 382):
        
        # Call to (...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'sizes' (line 382)
        sizes_69691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 30), 'sizes', False)
        # Processing the call keyword arguments (line 382)
        # Getting the type of 'offsets' (line 383)
        offsets_69692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 38), 'offsets', False)
        keyword_69693 = offsets_69692
        # Getting the type of 'transOffset' (line 384)
        transOffset_69694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 42), 'transOffset', False)
        keyword_69695 = transOffset_69694
        kwargs_69696 = {'transOffset': keyword_69695, 'offsets': keyword_69693}
        
        # Call to type(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'orig_handle' (line 382)
        orig_handle_69688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 17), 'orig_handle', False)
        # Processing the call keyword arguments (line 382)
        kwargs_69689 = {}
        # Getting the type of 'type' (line 382)
        type_69687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'type', False)
        # Calling type(args, kwargs) (line 382)
        type_call_result_69690 = invoke(stypy.reporting.localization.Localization(__file__, 382, 12), type_69687, *[orig_handle_69688], **kwargs_69689)
        
        # Calling (args, kwargs) (line 382)
        _call_result_69697 = invoke(stypy.reporting.localization.Localization(__file__, 382, 12), type_call_result_69690, *[sizes_69691], **kwargs_69696)
        
        # Assigning a type to the variable 'p' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'p', _call_result_69697)
        # Getting the type of 'p' (line 386)
        p_69698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'p')
        # Assigning a type to the variable 'stypy_return_type' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'stypy_return_type', p_69698)
        
        # ################# End of 'create_collection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_collection' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_69699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69699)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_collection'
        return stypy_return_type_69699


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 377, 0, False)
        # Assigning a type to the variable 'self' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerCircleCollection.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'HandlerCircleCollection' (line 377)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'HandlerCircleCollection', HandlerCircleCollection)
# Declaration of the 'HandlerErrorbar' class
# Getting the type of 'HandlerLine2D' (line 389)
HandlerLine2D_69700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 22), 'HandlerLine2D')

class HandlerErrorbar(HandlerLine2D_69700, ):
    unicode_69701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, (-1)), 'unicode', u'\n    Handler for Errorbars\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_69702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 33), 'float')
        # Getting the type of 'None' (line 393)
        None_69703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 48), 'None')
        float_69704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 28), 'float')
        # Getting the type of 'None' (line 394)
        None_69705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 43), 'None')
        defaults = [float_69702, None_69703, float_69704, None_69705]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 393, 4, False)
        # Assigning a type to the variable 'self' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerErrorbar.__init__', ['xerr_size', 'yerr_size', 'marker_pad', 'numpoints'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['xerr_size', 'yerr_size', 'marker_pad', 'numpoints'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 396):
        
        # Assigning a Name to a Attribute (line 396):
        # Getting the type of 'xerr_size' (line 396)
        xerr_size_69706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 26), 'xerr_size')
        # Getting the type of 'self' (line 396)
        self_69707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'self')
        # Setting the type of the member '_xerr_size' of a type (line 396)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), self_69707, '_xerr_size', xerr_size_69706)
        
        # Assigning a Name to a Attribute (line 397):
        
        # Assigning a Name to a Attribute (line 397):
        # Getting the type of 'yerr_size' (line 397)
        yerr_size_69708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 26), 'yerr_size')
        # Getting the type of 'self' (line 397)
        self_69709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self')
        # Setting the type of the member '_yerr_size' of a type (line 397)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_69709, '_yerr_size', yerr_size_69708)
        
        # Call to __init__(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'self' (line 399)
        self_69712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'self', False)
        # Processing the call keyword arguments (line 399)
        # Getting the type of 'marker_pad' (line 399)
        marker_pad_69713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 48), 'marker_pad', False)
        keyword_69714 = marker_pad_69713
        # Getting the type of 'numpoints' (line 399)
        numpoints_69715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 70), 'numpoints', False)
        keyword_69716 = numpoints_69715
        # Getting the type of 'kw' (line 400)
        kw_69717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 33), 'kw', False)
        kwargs_69718 = {'kw_69717': kw_69717, 'marker_pad': keyword_69714, 'numpoints': keyword_69716}
        # Getting the type of 'HandlerLine2D' (line 399)
        HandlerLine2D_69710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'HandlerLine2D', False)
        # Obtaining the member '__init__' of a type (line 399)
        init___69711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), HandlerLine2D_69710, '__init__')
        # Calling __init__(args, kwargs) (line 399)
        init___call_result_69719 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), init___69711, *[self_69712], **kwargs_69718)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_err_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_err_size'
        module_type_store = module_type_store.open_function_context('get_err_size', 402, 4, False)
        # Assigning a type to the variable 'self' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerErrorbar.get_err_size.__dict__.__setitem__('stypy_localization', localization)
        HandlerErrorbar.get_err_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerErrorbar.get_err_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerErrorbar.get_err_size.__dict__.__setitem__('stypy_function_name', 'HandlerErrorbar.get_err_size')
        HandlerErrorbar.get_err_size.__dict__.__setitem__('stypy_param_names_list', ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'])
        HandlerErrorbar.get_err_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerErrorbar.get_err_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerErrorbar.get_err_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerErrorbar.get_err_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerErrorbar.get_err_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerErrorbar.get_err_size.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerErrorbar.get_err_size', ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_err_size', localization, ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_err_size(...)' code ##################

        
        # Assigning a BinOp to a Name (line 403):
        
        # Assigning a BinOp to a Name (line 403):
        # Getting the type of 'self' (line 403)
        self_69720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 20), 'self')
        # Obtaining the member '_xerr_size' of a type (line 403)
        _xerr_size_69721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 20), self_69720, '_xerr_size')
        # Getting the type of 'fontsize' (line 403)
        fontsize_69722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 38), 'fontsize')
        # Applying the binary operator '*' (line 403)
        result_mul_69723 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 20), '*', _xerr_size_69721, fontsize_69722)
        
        # Assigning a type to the variable 'xerr_size' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'xerr_size', result_mul_69723)
        
        # Type idiom detected: calculating its left and rigth part (line 405)
        # Getting the type of 'self' (line 405)
        self_69724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 11), 'self')
        # Obtaining the member '_yerr_size' of a type (line 405)
        _yerr_size_69725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 11), self_69724, '_yerr_size')
        # Getting the type of 'None' (line 405)
        None_69726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 30), 'None')
        
        (may_be_69727, more_types_in_union_69728) = may_be_none(_yerr_size_69725, None_69726)

        if may_be_69727:

            if more_types_in_union_69728:
                # Runtime conditional SSA (line 405)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 406):
            
            # Assigning a Name to a Name (line 406):
            # Getting the type of 'xerr_size' (line 406)
            xerr_size_69729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 24), 'xerr_size')
            # Assigning a type to the variable 'yerr_size' (line 406)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'yerr_size', xerr_size_69729)

            if more_types_in_union_69728:
                # Runtime conditional SSA for else branch (line 405)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_69727) or more_types_in_union_69728):
            
            # Assigning a BinOp to a Name (line 408):
            
            # Assigning a BinOp to a Name (line 408):
            # Getting the type of 'self' (line 408)
            self_69730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 24), 'self')
            # Obtaining the member '_yerr_size' of a type (line 408)
            _yerr_size_69731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 24), self_69730, '_yerr_size')
            # Getting the type of 'fontsize' (line 408)
            fontsize_69732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 42), 'fontsize')
            # Applying the binary operator '*' (line 408)
            result_mul_69733 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 24), '*', _yerr_size_69731, fontsize_69732)
            
            # Assigning a type to the variable 'yerr_size' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'yerr_size', result_mul_69733)

            if (may_be_69727 and more_types_in_union_69728):
                # SSA join for if statement (line 405)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 410)
        tuple_69734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 410)
        # Adding element type (line 410)
        # Getting the type of 'xerr_size' (line 410)
        xerr_size_69735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 15), 'xerr_size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 15), tuple_69734, xerr_size_69735)
        # Adding element type (line 410)
        # Getting the type of 'yerr_size' (line 410)
        yerr_size_69736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 26), 'yerr_size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 15), tuple_69734, yerr_size_69736)
        
        # Assigning a type to the variable 'stypy_return_type' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'stypy_return_type', tuple_69734)
        
        # ################# End of 'get_err_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_err_size' in the type store
        # Getting the type of 'stypy_return_type' (line 402)
        stypy_return_type_69737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_69737)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_err_size'
        return stypy_return_type_69737


    @norecursion
    def create_artists(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_artists'
        module_type_store = module_type_store.open_function_context('create_artists', 412, 4, False)
        # Assigning a type to the variable 'self' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerErrorbar.create_artists.__dict__.__setitem__('stypy_localization', localization)
        HandlerErrorbar.create_artists.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerErrorbar.create_artists.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerErrorbar.create_artists.__dict__.__setitem__('stypy_function_name', 'HandlerErrorbar.create_artists')
        HandlerErrorbar.create_artists.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'])
        HandlerErrorbar.create_artists.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerErrorbar.create_artists.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerErrorbar.create_artists.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerErrorbar.create_artists.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerErrorbar.create_artists.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerErrorbar.create_artists.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerErrorbar.create_artists', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_artists', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_artists(...)' code ##################

        
        # Assigning a Name to a Tuple (line 416):
        
        # Assigning a Subscript to a Name (line 416):
        
        # Obtaining the type of the subscript
        int_69738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 8), 'int')
        # Getting the type of 'orig_handle' (line 416)
        orig_handle_69739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 43), 'orig_handle')
        # Obtaining the member '__getitem__' of a type (line 416)
        getitem___69740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), orig_handle_69739, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 416)
        subscript_call_result_69741 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), getitem___69740, int_69738)
        
        # Assigning a type to the variable 'tuple_var_assignment_68831' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'tuple_var_assignment_68831', subscript_call_result_69741)
        
        # Assigning a Subscript to a Name (line 416):
        
        # Obtaining the type of the subscript
        int_69742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 8), 'int')
        # Getting the type of 'orig_handle' (line 416)
        orig_handle_69743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 43), 'orig_handle')
        # Obtaining the member '__getitem__' of a type (line 416)
        getitem___69744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), orig_handle_69743, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 416)
        subscript_call_result_69745 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), getitem___69744, int_69742)
        
        # Assigning a type to the variable 'tuple_var_assignment_68832' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'tuple_var_assignment_68832', subscript_call_result_69745)
        
        # Assigning a Subscript to a Name (line 416):
        
        # Obtaining the type of the subscript
        int_69746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 8), 'int')
        # Getting the type of 'orig_handle' (line 416)
        orig_handle_69747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 43), 'orig_handle')
        # Obtaining the member '__getitem__' of a type (line 416)
        getitem___69748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), orig_handle_69747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 416)
        subscript_call_result_69749 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), getitem___69748, int_69746)
        
        # Assigning a type to the variable 'tuple_var_assignment_68833' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'tuple_var_assignment_68833', subscript_call_result_69749)
        
        # Assigning a Name to a Name (line 416):
        # Getting the type of 'tuple_var_assignment_68831' (line 416)
        tuple_var_assignment_68831_69750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'tuple_var_assignment_68831')
        # Assigning a type to the variable 'plotlines' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'plotlines', tuple_var_assignment_68831_69750)
        
        # Assigning a Name to a Name (line 416):
        # Getting the type of 'tuple_var_assignment_68832' (line 416)
        tuple_var_assignment_68832_69751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'tuple_var_assignment_68832')
        # Assigning a type to the variable 'caplines' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 19), 'caplines', tuple_var_assignment_68832_69751)
        
        # Assigning a Name to a Name (line 416):
        # Getting the type of 'tuple_var_assignment_68833' (line 416)
        tuple_var_assignment_68833_69752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'tuple_var_assignment_68833')
        # Assigning a type to the variable 'barlinecols' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 29), 'barlinecols', tuple_var_assignment_68833_69752)
        
        # Assigning a Call to a Tuple (line 418):
        
        # Assigning a Call to a Name:
        
        # Call to get_xdata(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'legend' (line 418)
        legend_69755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 45), 'legend', False)
        # Getting the type of 'xdescent' (line 418)
        xdescent_69756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 53), 'xdescent', False)
        # Getting the type of 'ydescent' (line 418)
        ydescent_69757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 63), 'ydescent', False)
        # Getting the type of 'width' (line 419)
        width_69758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 45), 'width', False)
        # Getting the type of 'height' (line 419)
        height_69759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 52), 'height', False)
        # Getting the type of 'fontsize' (line 419)
        fontsize_69760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 60), 'fontsize', False)
        # Processing the call keyword arguments (line 418)
        kwargs_69761 = {}
        # Getting the type of 'self' (line 418)
        self_69753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 30), 'self', False)
        # Obtaining the member 'get_xdata' of a type (line 418)
        get_xdata_69754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 30), self_69753, 'get_xdata')
        # Calling get_xdata(args, kwargs) (line 418)
        get_xdata_call_result_69762 = invoke(stypy.reporting.localization.Localization(__file__, 418, 30), get_xdata_69754, *[legend_69755, xdescent_69756, ydescent_69757, width_69758, height_69759, fontsize_69760], **kwargs_69761)
        
        # Assigning a type to the variable 'call_assignment_68834' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_68834', get_xdata_call_result_69762)
        
        # Assigning a Call to a Name (line 418):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_69765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 8), 'int')
        # Processing the call keyword arguments
        kwargs_69766 = {}
        # Getting the type of 'call_assignment_68834' (line 418)
        call_assignment_68834_69763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_68834', False)
        # Obtaining the member '__getitem__' of a type (line 418)
        getitem___69764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), call_assignment_68834_69763, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_69767 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___69764, *[int_69765], **kwargs_69766)
        
        # Assigning a type to the variable 'call_assignment_68835' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_68835', getitem___call_result_69767)
        
        # Assigning a Name to a Name (line 418):
        # Getting the type of 'call_assignment_68835' (line 418)
        call_assignment_68835_69768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_68835')
        # Assigning a type to the variable 'xdata' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'xdata', call_assignment_68835_69768)
        
        # Assigning a Call to a Name (line 418):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_69771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 8), 'int')
        # Processing the call keyword arguments
        kwargs_69772 = {}
        # Getting the type of 'call_assignment_68834' (line 418)
        call_assignment_68834_69769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_68834', False)
        # Obtaining the member '__getitem__' of a type (line 418)
        getitem___69770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), call_assignment_68834_69769, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_69773 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___69770, *[int_69771], **kwargs_69772)
        
        # Assigning a type to the variable 'call_assignment_68836' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_68836', getitem___call_result_69773)
        
        # Assigning a Name to a Name (line 418):
        # Getting the type of 'call_assignment_68836' (line 418)
        call_assignment_68836_69774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'call_assignment_68836')
        # Assigning a type to the variable 'xdata_marker' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'xdata_marker', call_assignment_68836_69774)
        
        # Assigning a BinOp to a Name (line 421):
        
        # Assigning a BinOp to a Name (line 421):
        # Getting the type of 'height' (line 421)
        height_69775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 18), 'height')
        # Getting the type of 'ydescent' (line 421)
        ydescent_69776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 27), 'ydescent')
        # Applying the binary operator '-' (line 421)
        result_sub_69777 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 18), '-', height_69775, ydescent_69776)
        
        float_69778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 39), 'float')
        # Applying the binary operator 'div' (line 421)
        result_div_69779 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 17), 'div', result_sub_69777, float_69778)
        
        
        # Call to ones(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'xdata' (line 421)
        xdata_69782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 53), 'xdata', False)
        # Obtaining the member 'shape' of a type (line 421)
        shape_69783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 53), xdata_69782, 'shape')
        # Getting the type of 'float' (line 421)
        float_69784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 66), 'float', False)
        # Processing the call keyword arguments (line 421)
        kwargs_69785 = {}
        # Getting the type of 'np' (line 421)
        np_69780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 45), 'np', False)
        # Obtaining the member 'ones' of a type (line 421)
        ones_69781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 45), np_69780, 'ones')
        # Calling ones(args, kwargs) (line 421)
        ones_call_result_69786 = invoke(stypy.reporting.localization.Localization(__file__, 421, 45), ones_69781, *[shape_69783, float_69784], **kwargs_69785)
        
        # Applying the binary operator '*' (line 421)
        result_mul_69787 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 16), '*', result_div_69779, ones_call_result_69786)
        
        # Assigning a type to the variable 'ydata' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'ydata', result_mul_69787)
        
        # Assigning a Call to a Name (line 422):
        
        # Assigning a Call to a Name (line 422):
        
        # Call to Line2D(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'xdata' (line 422)
        xdata_69789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 25), 'xdata', False)
        # Getting the type of 'ydata' (line 422)
        ydata_69790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 32), 'ydata', False)
        # Processing the call keyword arguments (line 422)
        kwargs_69791 = {}
        # Getting the type of 'Line2D' (line 422)
        Line2D_69788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 18), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 422)
        Line2D_call_result_69792 = invoke(stypy.reporting.localization.Localization(__file__, 422, 18), Line2D_69788, *[xdata_69789, ydata_69790], **kwargs_69791)
        
        # Assigning a type to the variable 'legline' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'legline', Line2D_call_result_69792)
        
        # Assigning a Call to a Name (line 425):
        
        # Assigning a Call to a Name (line 425):
        
        # Call to asarray(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'xdata_marker' (line 425)
        xdata_marker_69795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 34), 'xdata_marker', False)
        # Processing the call keyword arguments (line 425)
        kwargs_69796 = {}
        # Getting the type of 'np' (line 425)
        np_69793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 23), 'np', False)
        # Obtaining the member 'asarray' of a type (line 425)
        asarray_69794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 23), np_69793, 'asarray')
        # Calling asarray(args, kwargs) (line 425)
        asarray_call_result_69797 = invoke(stypy.reporting.localization.Localization(__file__, 425, 23), asarray_69794, *[xdata_marker_69795], **kwargs_69796)
        
        # Assigning a type to the variable 'xdata_marker' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'xdata_marker', asarray_call_result_69797)
        
        # Assigning a Call to a Name (line 426):
        
        # Assigning a Call to a Name (line 426):
        
        # Call to asarray(...): (line 426)
        # Processing the call arguments (line 426)
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'xdata_marker' (line 426)
        xdata_marker_69801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 45), 'xdata_marker', False)
        # Processing the call keyword arguments (line 426)
        kwargs_69802 = {}
        # Getting the type of 'len' (line 426)
        len_69800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 41), 'len', False)
        # Calling len(args, kwargs) (line 426)
        len_call_result_69803 = invoke(stypy.reporting.localization.Localization(__file__, 426, 41), len_69800, *[xdata_marker_69801], **kwargs_69802)
        
        slice_69804 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 426, 34), None, len_call_result_69803, None)
        # Getting the type of 'ydata' (line 426)
        ydata_69805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 34), 'ydata', False)
        # Obtaining the member '__getitem__' of a type (line 426)
        getitem___69806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 34), ydata_69805, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 426)
        subscript_call_result_69807 = invoke(stypy.reporting.localization.Localization(__file__, 426, 34), getitem___69806, slice_69804)
        
        # Processing the call keyword arguments (line 426)
        kwargs_69808 = {}
        # Getting the type of 'np' (line 426)
        np_69798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 23), 'np', False)
        # Obtaining the member 'asarray' of a type (line 426)
        asarray_69799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 23), np_69798, 'asarray')
        # Calling asarray(args, kwargs) (line 426)
        asarray_call_result_69809 = invoke(stypy.reporting.localization.Localization(__file__, 426, 23), asarray_69799, *[subscript_call_result_69807], **kwargs_69808)
        
        # Assigning a type to the variable 'ydata_marker' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'ydata_marker', asarray_call_result_69809)
        
        # Assigning a Call to a Tuple (line 428):
        
        # Assigning a Call to a Name:
        
        # Call to get_err_size(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 'legend' (line 428)
        legend_69812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 49), 'legend', False)
        # Getting the type of 'xdescent' (line 428)
        xdescent_69813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 57), 'xdescent', False)
        # Getting the type of 'ydescent' (line 428)
        ydescent_69814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 67), 'ydescent', False)
        # Getting the type of 'width' (line 429)
        width_69815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 49), 'width', False)
        # Getting the type of 'height' (line 429)
        height_69816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 56), 'height', False)
        # Getting the type of 'fontsize' (line 429)
        fontsize_69817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 64), 'fontsize', False)
        # Processing the call keyword arguments (line 428)
        kwargs_69818 = {}
        # Getting the type of 'self' (line 428)
        self_69810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 31), 'self', False)
        # Obtaining the member 'get_err_size' of a type (line 428)
        get_err_size_69811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 31), self_69810, 'get_err_size')
        # Calling get_err_size(args, kwargs) (line 428)
        get_err_size_call_result_69819 = invoke(stypy.reporting.localization.Localization(__file__, 428, 31), get_err_size_69811, *[legend_69812, xdescent_69813, ydescent_69814, width_69815, height_69816, fontsize_69817], **kwargs_69818)
        
        # Assigning a type to the variable 'call_assignment_68837' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'call_assignment_68837', get_err_size_call_result_69819)
        
        # Assigning a Call to a Name (line 428):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_69822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 8), 'int')
        # Processing the call keyword arguments
        kwargs_69823 = {}
        # Getting the type of 'call_assignment_68837' (line 428)
        call_assignment_68837_69820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'call_assignment_68837', False)
        # Obtaining the member '__getitem__' of a type (line 428)
        getitem___69821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), call_assignment_68837_69820, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_69824 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___69821, *[int_69822], **kwargs_69823)
        
        # Assigning a type to the variable 'call_assignment_68838' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'call_assignment_68838', getitem___call_result_69824)
        
        # Assigning a Name to a Name (line 428):
        # Getting the type of 'call_assignment_68838' (line 428)
        call_assignment_68838_69825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'call_assignment_68838')
        # Assigning a type to the variable 'xerr_size' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'xerr_size', call_assignment_68838_69825)
        
        # Assigning a Call to a Name (line 428):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_69828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 8), 'int')
        # Processing the call keyword arguments
        kwargs_69829 = {}
        # Getting the type of 'call_assignment_68837' (line 428)
        call_assignment_68837_69826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'call_assignment_68837', False)
        # Obtaining the member '__getitem__' of a type (line 428)
        getitem___69827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), call_assignment_68837_69826, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_69830 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___69827, *[int_69828], **kwargs_69829)
        
        # Assigning a type to the variable 'call_assignment_68839' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'call_assignment_68839', getitem___call_result_69830)
        
        # Assigning a Name to a Name (line 428):
        # Getting the type of 'call_assignment_68839' (line 428)
        call_assignment_68839_69831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'call_assignment_68839')
        # Assigning a type to the variable 'yerr_size' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 19), 'yerr_size', call_assignment_68839_69831)
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Call to Line2D(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'xdata_marker' (line 431)
        xdata_marker_69833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 32), 'xdata_marker', False)
        # Getting the type of 'ydata_marker' (line 431)
        ydata_marker_69834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 46), 'ydata_marker', False)
        # Processing the call keyword arguments (line 431)
        kwargs_69835 = {}
        # Getting the type of 'Line2D' (line 431)
        Line2D_69832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 25), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 431)
        Line2D_call_result_69836 = invoke(stypy.reporting.localization.Localization(__file__, 431, 25), Line2D_69832, *[xdata_marker_69833, ydata_marker_69834], **kwargs_69835)
        
        # Assigning a type to the variable 'legline_marker' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'legline_marker', Line2D_call_result_69836)
        
        # Type idiom detected: calculating its left and rigth part (line 435)
        # Getting the type of 'plotlines' (line 435)
        plotlines_69837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 11), 'plotlines')
        # Getting the type of 'None' (line 435)
        None_69838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 24), 'None')
        
        (may_be_69839, more_types_in_union_69840) = may_be_none(plotlines_69837, None_69838)

        if may_be_69839:

            if more_types_in_union_69840:
                # Runtime conditional SSA (line 435)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to set_visible(...): (line 436)
            # Processing the call arguments (line 436)
            # Getting the type of 'False' (line 436)
            False_69843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 32), 'False', False)
            # Processing the call keyword arguments (line 436)
            kwargs_69844 = {}
            # Getting the type of 'legline' (line 436)
            legline_69841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'legline', False)
            # Obtaining the member 'set_visible' of a type (line 436)
            set_visible_69842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 12), legline_69841, 'set_visible')
            # Calling set_visible(args, kwargs) (line 436)
            set_visible_call_result_69845 = invoke(stypy.reporting.localization.Localization(__file__, 436, 12), set_visible_69842, *[False_69843], **kwargs_69844)
            
            
            # Call to set_visible(...): (line 437)
            # Processing the call arguments (line 437)
            # Getting the type of 'False' (line 437)
            False_69848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 39), 'False', False)
            # Processing the call keyword arguments (line 437)
            kwargs_69849 = {}
            # Getting the type of 'legline_marker' (line 437)
            legline_marker_69846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'legline_marker', False)
            # Obtaining the member 'set_visible' of a type (line 437)
            set_visible_69847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 12), legline_marker_69846, 'set_visible')
            # Calling set_visible(args, kwargs) (line 437)
            set_visible_call_result_69850 = invoke(stypy.reporting.localization.Localization(__file__, 437, 12), set_visible_69847, *[False_69848], **kwargs_69849)
            

            if more_types_in_union_69840:
                # Runtime conditional SSA for else branch (line 435)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_69839) or more_types_in_union_69840):
            
            # Call to update_prop(...): (line 439)
            # Processing the call arguments (line 439)
            # Getting the type of 'legline' (line 439)
            legline_69853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 29), 'legline', False)
            # Getting the type of 'plotlines' (line 439)
            plotlines_69854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 38), 'plotlines', False)
            # Getting the type of 'legend' (line 439)
            legend_69855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 49), 'legend', False)
            # Processing the call keyword arguments (line 439)
            kwargs_69856 = {}
            # Getting the type of 'self' (line 439)
            self_69851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'self', False)
            # Obtaining the member 'update_prop' of a type (line 439)
            update_prop_69852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 12), self_69851, 'update_prop')
            # Calling update_prop(args, kwargs) (line 439)
            update_prop_call_result_69857 = invoke(stypy.reporting.localization.Localization(__file__, 439, 12), update_prop_69852, *[legline_69853, plotlines_69854, legend_69855], **kwargs_69856)
            
            
            # Call to set_drawstyle(...): (line 441)
            # Processing the call arguments (line 441)
            unicode_69860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 34), 'unicode', u'default')
            # Processing the call keyword arguments (line 441)
            kwargs_69861 = {}
            # Getting the type of 'legline' (line 441)
            legline_69858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 12), 'legline', False)
            # Obtaining the member 'set_drawstyle' of a type (line 441)
            set_drawstyle_69859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 12), legline_69858, 'set_drawstyle')
            # Calling set_drawstyle(args, kwargs) (line 441)
            set_drawstyle_call_result_69862 = invoke(stypy.reporting.localization.Localization(__file__, 441, 12), set_drawstyle_69859, *[unicode_69860], **kwargs_69861)
            
            
            # Call to set_marker(...): (line 442)
            # Processing the call arguments (line 442)
            unicode_69865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 31), 'unicode', u'None')
            # Processing the call keyword arguments (line 442)
            kwargs_69866 = {}
            # Getting the type of 'legline' (line 442)
            legline_69863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'legline', False)
            # Obtaining the member 'set_marker' of a type (line 442)
            set_marker_69864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 12), legline_69863, 'set_marker')
            # Calling set_marker(args, kwargs) (line 442)
            set_marker_call_result_69867 = invoke(stypy.reporting.localization.Localization(__file__, 442, 12), set_marker_69864, *[unicode_69865], **kwargs_69866)
            
            
            # Call to update_prop(...): (line 444)
            # Processing the call arguments (line 444)
            # Getting the type of 'legline_marker' (line 444)
            legline_marker_69870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 29), 'legline_marker', False)
            # Getting the type of 'plotlines' (line 444)
            plotlines_69871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 45), 'plotlines', False)
            # Getting the type of 'legend' (line 444)
            legend_69872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 56), 'legend', False)
            # Processing the call keyword arguments (line 444)
            kwargs_69873 = {}
            # Getting the type of 'self' (line 444)
            self_69868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'self', False)
            # Obtaining the member 'update_prop' of a type (line 444)
            update_prop_69869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), self_69868, 'update_prop')
            # Calling update_prop(args, kwargs) (line 444)
            update_prop_call_result_69874 = invoke(stypy.reporting.localization.Localization(__file__, 444, 12), update_prop_69869, *[legline_marker_69870, plotlines_69871, legend_69872], **kwargs_69873)
            
            
            # Call to set_linestyle(...): (line 445)
            # Processing the call arguments (line 445)
            unicode_69877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 41), 'unicode', u'None')
            # Processing the call keyword arguments (line 445)
            kwargs_69878 = {}
            # Getting the type of 'legline_marker' (line 445)
            legline_marker_69875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'legline_marker', False)
            # Obtaining the member 'set_linestyle' of a type (line 445)
            set_linestyle_69876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 12), legline_marker_69875, 'set_linestyle')
            # Calling set_linestyle(args, kwargs) (line 445)
            set_linestyle_call_result_69879 = invoke(stypy.reporting.localization.Localization(__file__, 445, 12), set_linestyle_69876, *[unicode_69877], **kwargs_69878)
            
            
            
            # Getting the type of 'legend' (line 447)
            legend_69880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 15), 'legend')
            # Obtaining the member 'markerscale' of a type (line 447)
            markerscale_69881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 15), legend_69880, 'markerscale')
            int_69882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 37), 'int')
            # Applying the binary operator '!=' (line 447)
            result_ne_69883 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 15), '!=', markerscale_69881, int_69882)
            
            # Testing the type of an if condition (line 447)
            if_condition_69884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 447, 12), result_ne_69883)
            # Assigning a type to the variable 'if_condition_69884' (line 447)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'if_condition_69884', if_condition_69884)
            # SSA begins for if statement (line 447)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 448):
            
            # Assigning a BinOp to a Name (line 448):
            
            # Call to get_markersize(...): (line 448)
            # Processing the call keyword arguments (line 448)
            kwargs_69887 = {}
            # Getting the type of 'legline_marker' (line 448)
            legline_marker_69885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 24), 'legline_marker', False)
            # Obtaining the member 'get_markersize' of a type (line 448)
            get_markersize_69886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 24), legline_marker_69885, 'get_markersize')
            # Calling get_markersize(args, kwargs) (line 448)
            get_markersize_call_result_69888 = invoke(stypy.reporting.localization.Localization(__file__, 448, 24), get_markersize_69886, *[], **kwargs_69887)
            
            # Getting the type of 'legend' (line 448)
            legend_69889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 58), 'legend')
            # Obtaining the member 'markerscale' of a type (line 448)
            markerscale_69890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 58), legend_69889, 'markerscale')
            # Applying the binary operator '*' (line 448)
            result_mul_69891 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 24), '*', get_markersize_call_result_69888, markerscale_69890)
            
            # Assigning a type to the variable 'newsz' (line 448)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 16), 'newsz', result_mul_69891)
            
            # Call to set_markersize(...): (line 449)
            # Processing the call arguments (line 449)
            # Getting the type of 'newsz' (line 449)
            newsz_69894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 46), 'newsz', False)
            # Processing the call keyword arguments (line 449)
            kwargs_69895 = {}
            # Getting the type of 'legline_marker' (line 449)
            legline_marker_69892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 16), 'legline_marker', False)
            # Obtaining the member 'set_markersize' of a type (line 449)
            set_markersize_69893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 16), legline_marker_69892, 'set_markersize')
            # Calling set_markersize(args, kwargs) (line 449)
            set_markersize_call_result_69896 = invoke(stypy.reporting.localization.Localization(__file__, 449, 16), set_markersize_69893, *[newsz_69894], **kwargs_69895)
            
            # SSA join for if statement (line 447)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_69839 and more_types_in_union_69840):
                # SSA join for if statement (line 435)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 451):
        
        # Assigning a List to a Name (line 451):
        
        # Obtaining an instance of the builtin type 'list' (line 451)
        list_69897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 451)
        
        # Assigning a type to the variable 'handle_barlinecols' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'handle_barlinecols', list_69897)
        
        # Assigning a List to a Name (line 452):
        
        # Assigning a List to a Name (line 452):
        
        # Obtaining an instance of the builtin type 'list' (line 452)
        list_69898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 452)
        
        # Assigning a type to the variable 'handle_caplines' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'handle_caplines', list_69898)
        
        # Getting the type of 'orig_handle' (line 454)
        orig_handle_69899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 11), 'orig_handle')
        # Obtaining the member 'has_xerr' of a type (line 454)
        has_xerr_69900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 11), orig_handle_69899, 'has_xerr')
        # Testing the type of an if condition (line 454)
        if_condition_69901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 8), has_xerr_69900)
        # Assigning a type to the variable 'if_condition_69901' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'if_condition_69901', if_condition_69901)
        # SSA begins for if statement (line 454)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Name (line 455):
        
        # Assigning a ListComp to a Name (line 455):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to zip(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'xdata_marker' (line 456)
        xdata_marker_69914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 38), 'xdata_marker', False)
        # Getting the type of 'ydata_marker' (line 456)
        ydata_marker_69915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 52), 'ydata_marker', False)
        # Processing the call keyword arguments (line 456)
        kwargs_69916 = {}
        # Getting the type of 'zip' (line 456)
        zip_69913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 34), 'zip', False)
        # Calling zip(args, kwargs) (line 456)
        zip_call_result_69917 = invoke(stypy.reporting.localization.Localization(__file__, 456, 34), zip_69913, *[xdata_marker_69914, ydata_marker_69915], **kwargs_69916)
        
        comprehension_69918 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 22), zip_call_result_69917)
        # Assigning a type to the variable 'x' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 22), comprehension_69918))
        # Assigning a type to the variable 'y' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 22), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 22), comprehension_69918))
        
        # Obtaining an instance of the builtin type 'tuple' (line 455)
        tuple_69902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 455)
        # Adding element type (line 455)
        
        # Obtaining an instance of the builtin type 'tuple' (line 455)
        tuple_69903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 455)
        # Adding element type (line 455)
        # Getting the type of 'x' (line 455)
        x_69904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 24), 'x')
        # Getting the type of 'xerr_size' (line 455)
        xerr_size_69905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 28), 'xerr_size')
        # Applying the binary operator '-' (line 455)
        result_sub_69906 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 24), '-', x_69904, xerr_size_69905)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 24), tuple_69903, result_sub_69906)
        # Adding element type (line 455)
        # Getting the type of 'y' (line 455)
        y_69907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 39), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 24), tuple_69903, y_69907)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 23), tuple_69902, tuple_69903)
        # Adding element type (line 455)
        
        # Obtaining an instance of the builtin type 'tuple' (line 455)
        tuple_69908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 455)
        # Adding element type (line 455)
        # Getting the type of 'x' (line 455)
        x_69909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 44), 'x')
        # Getting the type of 'xerr_size' (line 455)
        xerr_size_69910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 48), 'xerr_size')
        # Applying the binary operator '+' (line 455)
        result_add_69911 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 44), '+', x_69909, xerr_size_69910)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 44), tuple_69908, result_add_69911)
        # Adding element type (line 455)
        # Getting the type of 'y' (line 455)
        y_69912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 59), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 44), tuple_69908, y_69912)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 23), tuple_69902, tuple_69908)
        
        list_69919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 22), list_69919, tuple_69902)
        # Assigning a type to the variable 'verts' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'verts', list_69919)
        
        # Assigning a Call to a Name (line 457):
        
        # Assigning a Call to a Name (line 457):
        
        # Call to LineCollection(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'verts' (line 457)
        verts_69922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 40), 'verts', False)
        # Processing the call keyword arguments (line 457)
        kwargs_69923 = {}
        # Getting the type of 'mcoll' (line 457)
        mcoll_69920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 19), 'mcoll', False)
        # Obtaining the member 'LineCollection' of a type (line 457)
        LineCollection_69921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 19), mcoll_69920, 'LineCollection')
        # Calling LineCollection(args, kwargs) (line 457)
        LineCollection_call_result_69924 = invoke(stypy.reporting.localization.Localization(__file__, 457, 19), LineCollection_69921, *[verts_69922], **kwargs_69923)
        
        # Assigning a type to the variable 'coll' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'coll', LineCollection_call_result_69924)
        
        # Call to update_prop(...): (line 458)
        # Processing the call arguments (line 458)
        # Getting the type of 'coll' (line 458)
        coll_69927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 29), 'coll', False)
        
        # Obtaining the type of the subscript
        int_69928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 47), 'int')
        # Getting the type of 'barlinecols' (line 458)
        barlinecols_69929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 35), 'barlinecols', False)
        # Obtaining the member '__getitem__' of a type (line 458)
        getitem___69930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 35), barlinecols_69929, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 458)
        subscript_call_result_69931 = invoke(stypy.reporting.localization.Localization(__file__, 458, 35), getitem___69930, int_69928)
        
        # Getting the type of 'legend' (line 458)
        legend_69932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 51), 'legend', False)
        # Processing the call keyword arguments (line 458)
        kwargs_69933 = {}
        # Getting the type of 'self' (line 458)
        self_69925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 458)
        update_prop_69926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 12), self_69925, 'update_prop')
        # Calling update_prop(args, kwargs) (line 458)
        update_prop_call_result_69934 = invoke(stypy.reporting.localization.Localization(__file__, 458, 12), update_prop_69926, *[coll_69927, subscript_call_result_69931, legend_69932], **kwargs_69933)
        
        
        # Call to append(...): (line 459)
        # Processing the call arguments (line 459)
        # Getting the type of 'coll' (line 459)
        coll_69937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 38), 'coll', False)
        # Processing the call keyword arguments (line 459)
        kwargs_69938 = {}
        # Getting the type of 'handle_barlinecols' (line 459)
        handle_barlinecols_69935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'handle_barlinecols', False)
        # Obtaining the member 'append' of a type (line 459)
        append_69936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 12), handle_barlinecols_69935, 'append')
        # Calling append(args, kwargs) (line 459)
        append_call_result_69939 = invoke(stypy.reporting.localization.Localization(__file__, 459, 12), append_69936, *[coll_69937], **kwargs_69938)
        
        
        # Getting the type of 'caplines' (line 461)
        caplines_69940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 15), 'caplines')
        # Testing the type of an if condition (line 461)
        if_condition_69941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 461, 12), caplines_69940)
        # Assigning a type to the variable 'if_condition_69941' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'if_condition_69941', if_condition_69941)
        # SSA begins for if statement (line 461)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 462):
        
        # Assigning a Call to a Name (line 462):
        
        # Call to Line2D(...): (line 462)
        # Processing the call arguments (line 462)
        # Getting the type of 'xdata_marker' (line 462)
        xdata_marker_69943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 38), 'xdata_marker', False)
        # Getting the type of 'xerr_size' (line 462)
        xerr_size_69944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 53), 'xerr_size', False)
        # Applying the binary operator '-' (line 462)
        result_sub_69945 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 38), '-', xdata_marker_69943, xerr_size_69944)
        
        # Getting the type of 'ydata_marker' (line 462)
        ydata_marker_69946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 64), 'ydata_marker', False)
        # Processing the call keyword arguments (line 462)
        kwargs_69947 = {}
        # Getting the type of 'Line2D' (line 462)
        Line2D_69942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 31), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 462)
        Line2D_call_result_69948 = invoke(stypy.reporting.localization.Localization(__file__, 462, 31), Line2D_69942, *[result_sub_69945, ydata_marker_69946], **kwargs_69947)
        
        # Assigning a type to the variable 'capline_left' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 16), 'capline_left', Line2D_call_result_69948)
        
        # Assigning a Call to a Name (line 463):
        
        # Assigning a Call to a Name (line 463):
        
        # Call to Line2D(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'xdata_marker' (line 463)
        xdata_marker_69950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 39), 'xdata_marker', False)
        # Getting the type of 'xerr_size' (line 463)
        xerr_size_69951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 54), 'xerr_size', False)
        # Applying the binary operator '+' (line 463)
        result_add_69952 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 39), '+', xdata_marker_69950, xerr_size_69951)
        
        # Getting the type of 'ydata_marker' (line 463)
        ydata_marker_69953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 65), 'ydata_marker', False)
        # Processing the call keyword arguments (line 463)
        kwargs_69954 = {}
        # Getting the type of 'Line2D' (line 463)
        Line2D_69949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 32), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 463)
        Line2D_call_result_69955 = invoke(stypy.reporting.localization.Localization(__file__, 463, 32), Line2D_69949, *[result_add_69952, ydata_marker_69953], **kwargs_69954)
        
        # Assigning a type to the variable 'capline_right' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 16), 'capline_right', Line2D_call_result_69955)
        
        # Call to update_prop(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'capline_left' (line 464)
        capline_left_69958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 33), 'capline_left', False)
        
        # Obtaining the type of the subscript
        int_69959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 56), 'int')
        # Getting the type of 'caplines' (line 464)
        caplines_69960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 47), 'caplines', False)
        # Obtaining the member '__getitem__' of a type (line 464)
        getitem___69961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 47), caplines_69960, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 464)
        subscript_call_result_69962 = invoke(stypy.reporting.localization.Localization(__file__, 464, 47), getitem___69961, int_69959)
        
        # Getting the type of 'legend' (line 464)
        legend_69963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 60), 'legend', False)
        # Processing the call keyword arguments (line 464)
        kwargs_69964 = {}
        # Getting the type of 'self' (line 464)
        self_69956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 16), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 464)
        update_prop_69957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 16), self_69956, 'update_prop')
        # Calling update_prop(args, kwargs) (line 464)
        update_prop_call_result_69965 = invoke(stypy.reporting.localization.Localization(__file__, 464, 16), update_prop_69957, *[capline_left_69958, subscript_call_result_69962, legend_69963], **kwargs_69964)
        
        
        # Call to update_prop(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'capline_right' (line 465)
        capline_right_69968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 33), 'capline_right', False)
        
        # Obtaining the type of the subscript
        int_69969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 57), 'int')
        # Getting the type of 'caplines' (line 465)
        caplines_69970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 48), 'caplines', False)
        # Obtaining the member '__getitem__' of a type (line 465)
        getitem___69971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 48), caplines_69970, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 465)
        subscript_call_result_69972 = invoke(stypy.reporting.localization.Localization(__file__, 465, 48), getitem___69971, int_69969)
        
        # Getting the type of 'legend' (line 465)
        legend_69973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 61), 'legend', False)
        # Processing the call keyword arguments (line 465)
        kwargs_69974 = {}
        # Getting the type of 'self' (line 465)
        self_69966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 465)
        update_prop_69967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 16), self_69966, 'update_prop')
        # Calling update_prop(args, kwargs) (line 465)
        update_prop_call_result_69975 = invoke(stypy.reporting.localization.Localization(__file__, 465, 16), update_prop_69967, *[capline_right_69968, subscript_call_result_69972, legend_69973], **kwargs_69974)
        
        
        # Call to set_marker(...): (line 466)
        # Processing the call arguments (line 466)
        unicode_69978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 40), 'unicode', u'|')
        # Processing the call keyword arguments (line 466)
        kwargs_69979 = {}
        # Getting the type of 'capline_left' (line 466)
        capline_left_69976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'capline_left', False)
        # Obtaining the member 'set_marker' of a type (line 466)
        set_marker_69977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 16), capline_left_69976, 'set_marker')
        # Calling set_marker(args, kwargs) (line 466)
        set_marker_call_result_69980 = invoke(stypy.reporting.localization.Localization(__file__, 466, 16), set_marker_69977, *[unicode_69978], **kwargs_69979)
        
        
        # Call to set_marker(...): (line 467)
        # Processing the call arguments (line 467)
        unicode_69983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 41), 'unicode', u'|')
        # Processing the call keyword arguments (line 467)
        kwargs_69984 = {}
        # Getting the type of 'capline_right' (line 467)
        capline_right_69981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'capline_right', False)
        # Obtaining the member 'set_marker' of a type (line 467)
        set_marker_69982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 16), capline_right_69981, 'set_marker')
        # Calling set_marker(args, kwargs) (line 467)
        set_marker_call_result_69985 = invoke(stypy.reporting.localization.Localization(__file__, 467, 16), set_marker_69982, *[unicode_69983], **kwargs_69984)
        
        
        # Call to append(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'capline_left' (line 469)
        capline_left_69988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 39), 'capline_left', False)
        # Processing the call keyword arguments (line 469)
        kwargs_69989 = {}
        # Getting the type of 'handle_caplines' (line 469)
        handle_caplines_69986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 16), 'handle_caplines', False)
        # Obtaining the member 'append' of a type (line 469)
        append_69987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 16), handle_caplines_69986, 'append')
        # Calling append(args, kwargs) (line 469)
        append_call_result_69990 = invoke(stypy.reporting.localization.Localization(__file__, 469, 16), append_69987, *[capline_left_69988], **kwargs_69989)
        
        
        # Call to append(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of 'capline_right' (line 470)
        capline_right_69993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 39), 'capline_right', False)
        # Processing the call keyword arguments (line 470)
        kwargs_69994 = {}
        # Getting the type of 'handle_caplines' (line 470)
        handle_caplines_69991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 16), 'handle_caplines', False)
        # Obtaining the member 'append' of a type (line 470)
        append_69992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 16), handle_caplines_69991, 'append')
        # Calling append(args, kwargs) (line 470)
        append_call_result_69995 = invoke(stypy.reporting.localization.Localization(__file__, 470, 16), append_69992, *[capline_right_69993], **kwargs_69994)
        
        # SSA join for if statement (line 461)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 454)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'orig_handle' (line 472)
        orig_handle_69996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 11), 'orig_handle')
        # Obtaining the member 'has_yerr' of a type (line 472)
        has_yerr_69997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 11), orig_handle_69996, 'has_yerr')
        # Testing the type of an if condition (line 472)
        if_condition_69998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 472, 8), has_yerr_69997)
        # Assigning a type to the variable 'if_condition_69998' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'if_condition_69998', if_condition_69998)
        # SSA begins for if statement (line 472)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Name (line 473):
        
        # Assigning a ListComp to a Name (line 473):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to zip(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'xdata_marker' (line 474)
        xdata_marker_70011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 38), 'xdata_marker', False)
        # Getting the type of 'ydata_marker' (line 474)
        ydata_marker_70012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 52), 'ydata_marker', False)
        # Processing the call keyword arguments (line 474)
        kwargs_70013 = {}
        # Getting the type of 'zip' (line 474)
        zip_70010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 34), 'zip', False)
        # Calling zip(args, kwargs) (line 474)
        zip_call_result_70014 = invoke(stypy.reporting.localization.Localization(__file__, 474, 34), zip_70010, *[xdata_marker_70011, ydata_marker_70012], **kwargs_70013)
        
        comprehension_70015 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 22), zip_call_result_70014)
        # Assigning a type to the variable 'x' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 22), comprehension_70015))
        # Assigning a type to the variable 'y' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 22), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 22), comprehension_70015))
        
        # Obtaining an instance of the builtin type 'tuple' (line 473)
        tuple_69999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 473)
        # Adding element type (line 473)
        
        # Obtaining an instance of the builtin type 'tuple' (line 473)
        tuple_70000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 473)
        # Adding element type (line 473)
        # Getting the type of 'x' (line 473)
        x_70001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 24), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 24), tuple_70000, x_70001)
        # Adding element type (line 473)
        # Getting the type of 'y' (line 473)
        y_70002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 27), 'y')
        # Getting the type of 'yerr_size' (line 473)
        yerr_size_70003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 31), 'yerr_size')
        # Applying the binary operator '-' (line 473)
        result_sub_70004 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 27), '-', y_70002, yerr_size_70003)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 24), tuple_70000, result_sub_70004)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 23), tuple_69999, tuple_70000)
        # Adding element type (line 473)
        
        # Obtaining an instance of the builtin type 'tuple' (line 473)
        tuple_70005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 473)
        # Adding element type (line 473)
        # Getting the type of 'x' (line 473)
        x_70006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 44), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 44), tuple_70005, x_70006)
        # Adding element type (line 473)
        # Getting the type of 'y' (line 473)
        y_70007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 47), 'y')
        # Getting the type of 'yerr_size' (line 473)
        yerr_size_70008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 51), 'yerr_size')
        # Applying the binary operator '+' (line 473)
        result_add_70009 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 47), '+', y_70007, yerr_size_70008)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 44), tuple_70005, result_add_70009)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 23), tuple_69999, tuple_70005)
        
        list_70016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 22), list_70016, tuple_69999)
        # Assigning a type to the variable 'verts' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'verts', list_70016)
        
        # Assigning a Call to a Name (line 475):
        
        # Assigning a Call to a Name (line 475):
        
        # Call to LineCollection(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'verts' (line 475)
        verts_70019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 40), 'verts', False)
        # Processing the call keyword arguments (line 475)
        kwargs_70020 = {}
        # Getting the type of 'mcoll' (line 475)
        mcoll_70017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'mcoll', False)
        # Obtaining the member 'LineCollection' of a type (line 475)
        LineCollection_70018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 19), mcoll_70017, 'LineCollection')
        # Calling LineCollection(args, kwargs) (line 475)
        LineCollection_call_result_70021 = invoke(stypy.reporting.localization.Localization(__file__, 475, 19), LineCollection_70018, *[verts_70019], **kwargs_70020)
        
        # Assigning a type to the variable 'coll' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'coll', LineCollection_call_result_70021)
        
        # Call to update_prop(...): (line 476)
        # Processing the call arguments (line 476)
        # Getting the type of 'coll' (line 476)
        coll_70024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 29), 'coll', False)
        
        # Obtaining the type of the subscript
        int_70025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 47), 'int')
        # Getting the type of 'barlinecols' (line 476)
        barlinecols_70026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 35), 'barlinecols', False)
        # Obtaining the member '__getitem__' of a type (line 476)
        getitem___70027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 35), barlinecols_70026, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 476)
        subscript_call_result_70028 = invoke(stypy.reporting.localization.Localization(__file__, 476, 35), getitem___70027, int_70025)
        
        # Getting the type of 'legend' (line 476)
        legend_70029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 51), 'legend', False)
        # Processing the call keyword arguments (line 476)
        kwargs_70030 = {}
        # Getting the type of 'self' (line 476)
        self_70022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 476)
        update_prop_70023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 12), self_70022, 'update_prop')
        # Calling update_prop(args, kwargs) (line 476)
        update_prop_call_result_70031 = invoke(stypy.reporting.localization.Localization(__file__, 476, 12), update_prop_70023, *[coll_70024, subscript_call_result_70028, legend_70029], **kwargs_70030)
        
        
        # Call to append(...): (line 477)
        # Processing the call arguments (line 477)
        # Getting the type of 'coll' (line 477)
        coll_70034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 38), 'coll', False)
        # Processing the call keyword arguments (line 477)
        kwargs_70035 = {}
        # Getting the type of 'handle_barlinecols' (line 477)
        handle_barlinecols_70032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'handle_barlinecols', False)
        # Obtaining the member 'append' of a type (line 477)
        append_70033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 12), handle_barlinecols_70032, 'append')
        # Calling append(args, kwargs) (line 477)
        append_call_result_70036 = invoke(stypy.reporting.localization.Localization(__file__, 477, 12), append_70033, *[coll_70034], **kwargs_70035)
        
        
        # Getting the type of 'caplines' (line 479)
        caplines_70037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 15), 'caplines')
        # Testing the type of an if condition (line 479)
        if_condition_70038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 479, 12), caplines_70037)
        # Assigning a type to the variable 'if_condition_70038' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'if_condition_70038', if_condition_70038)
        # SSA begins for if statement (line 479)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 480):
        
        # Assigning a Call to a Name (line 480):
        
        # Call to Line2D(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'xdata_marker' (line 480)
        xdata_marker_70040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 38), 'xdata_marker', False)
        # Getting the type of 'ydata_marker' (line 480)
        ydata_marker_70041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 52), 'ydata_marker', False)
        # Getting the type of 'yerr_size' (line 480)
        yerr_size_70042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 67), 'yerr_size', False)
        # Applying the binary operator '-' (line 480)
        result_sub_70043 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 52), '-', ydata_marker_70041, yerr_size_70042)
        
        # Processing the call keyword arguments (line 480)
        kwargs_70044 = {}
        # Getting the type of 'Line2D' (line 480)
        Line2D_70039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 31), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 480)
        Line2D_call_result_70045 = invoke(stypy.reporting.localization.Localization(__file__, 480, 31), Line2D_70039, *[xdata_marker_70040, result_sub_70043], **kwargs_70044)
        
        # Assigning a type to the variable 'capline_left' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'capline_left', Line2D_call_result_70045)
        
        # Assigning a Call to a Name (line 481):
        
        # Assigning a Call to a Name (line 481):
        
        # Call to Line2D(...): (line 481)
        # Processing the call arguments (line 481)
        # Getting the type of 'xdata_marker' (line 481)
        xdata_marker_70047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 39), 'xdata_marker', False)
        # Getting the type of 'ydata_marker' (line 481)
        ydata_marker_70048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 53), 'ydata_marker', False)
        # Getting the type of 'yerr_size' (line 481)
        yerr_size_70049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 68), 'yerr_size', False)
        # Applying the binary operator '+' (line 481)
        result_add_70050 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 53), '+', ydata_marker_70048, yerr_size_70049)
        
        # Processing the call keyword arguments (line 481)
        kwargs_70051 = {}
        # Getting the type of 'Line2D' (line 481)
        Line2D_70046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 32), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 481)
        Line2D_call_result_70052 = invoke(stypy.reporting.localization.Localization(__file__, 481, 32), Line2D_70046, *[xdata_marker_70047, result_add_70050], **kwargs_70051)
        
        # Assigning a type to the variable 'capline_right' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 16), 'capline_right', Line2D_call_result_70052)
        
        # Call to update_prop(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'capline_left' (line 482)
        capline_left_70055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 33), 'capline_left', False)
        
        # Obtaining the type of the subscript
        int_70056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 56), 'int')
        # Getting the type of 'caplines' (line 482)
        caplines_70057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 47), 'caplines', False)
        # Obtaining the member '__getitem__' of a type (line 482)
        getitem___70058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 47), caplines_70057, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 482)
        subscript_call_result_70059 = invoke(stypy.reporting.localization.Localization(__file__, 482, 47), getitem___70058, int_70056)
        
        # Getting the type of 'legend' (line 482)
        legend_70060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 60), 'legend', False)
        # Processing the call keyword arguments (line 482)
        kwargs_70061 = {}
        # Getting the type of 'self' (line 482)
        self_70053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 482)
        update_prop_70054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 16), self_70053, 'update_prop')
        # Calling update_prop(args, kwargs) (line 482)
        update_prop_call_result_70062 = invoke(stypy.reporting.localization.Localization(__file__, 482, 16), update_prop_70054, *[capline_left_70055, subscript_call_result_70059, legend_70060], **kwargs_70061)
        
        
        # Call to update_prop(...): (line 483)
        # Processing the call arguments (line 483)
        # Getting the type of 'capline_right' (line 483)
        capline_right_70065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 33), 'capline_right', False)
        
        # Obtaining the type of the subscript
        int_70066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 57), 'int')
        # Getting the type of 'caplines' (line 483)
        caplines_70067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 48), 'caplines', False)
        # Obtaining the member '__getitem__' of a type (line 483)
        getitem___70068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 48), caplines_70067, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 483)
        subscript_call_result_70069 = invoke(stypy.reporting.localization.Localization(__file__, 483, 48), getitem___70068, int_70066)
        
        # Getting the type of 'legend' (line 483)
        legend_70070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 61), 'legend', False)
        # Processing the call keyword arguments (line 483)
        kwargs_70071 = {}
        # Getting the type of 'self' (line 483)
        self_70063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 16), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 483)
        update_prop_70064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 16), self_70063, 'update_prop')
        # Calling update_prop(args, kwargs) (line 483)
        update_prop_call_result_70072 = invoke(stypy.reporting.localization.Localization(__file__, 483, 16), update_prop_70064, *[capline_right_70065, subscript_call_result_70069, legend_70070], **kwargs_70071)
        
        
        # Call to set_marker(...): (line 484)
        # Processing the call arguments (line 484)
        unicode_70075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 40), 'unicode', u'_')
        # Processing the call keyword arguments (line 484)
        kwargs_70076 = {}
        # Getting the type of 'capline_left' (line 484)
        capline_left_70073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 16), 'capline_left', False)
        # Obtaining the member 'set_marker' of a type (line 484)
        set_marker_70074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 16), capline_left_70073, 'set_marker')
        # Calling set_marker(args, kwargs) (line 484)
        set_marker_call_result_70077 = invoke(stypy.reporting.localization.Localization(__file__, 484, 16), set_marker_70074, *[unicode_70075], **kwargs_70076)
        
        
        # Call to set_marker(...): (line 485)
        # Processing the call arguments (line 485)
        unicode_70080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 41), 'unicode', u'_')
        # Processing the call keyword arguments (line 485)
        kwargs_70081 = {}
        # Getting the type of 'capline_right' (line 485)
        capline_right_70078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'capline_right', False)
        # Obtaining the member 'set_marker' of a type (line 485)
        set_marker_70079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 16), capline_right_70078, 'set_marker')
        # Calling set_marker(args, kwargs) (line 485)
        set_marker_call_result_70082 = invoke(stypy.reporting.localization.Localization(__file__, 485, 16), set_marker_70079, *[unicode_70080], **kwargs_70081)
        
        
        # Call to append(...): (line 487)
        # Processing the call arguments (line 487)
        # Getting the type of 'capline_left' (line 487)
        capline_left_70085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 39), 'capline_left', False)
        # Processing the call keyword arguments (line 487)
        kwargs_70086 = {}
        # Getting the type of 'handle_caplines' (line 487)
        handle_caplines_70083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'handle_caplines', False)
        # Obtaining the member 'append' of a type (line 487)
        append_70084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 16), handle_caplines_70083, 'append')
        # Calling append(args, kwargs) (line 487)
        append_call_result_70087 = invoke(stypy.reporting.localization.Localization(__file__, 487, 16), append_70084, *[capline_left_70085], **kwargs_70086)
        
        
        # Call to append(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 'capline_right' (line 488)
        capline_right_70090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 39), 'capline_right', False)
        # Processing the call keyword arguments (line 488)
        kwargs_70091 = {}
        # Getting the type of 'handle_caplines' (line 488)
        handle_caplines_70088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 16), 'handle_caplines', False)
        # Obtaining the member 'append' of a type (line 488)
        append_70089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 16), handle_caplines_70088, 'append')
        # Calling append(args, kwargs) (line 488)
        append_call_result_70092 = invoke(stypy.reporting.localization.Localization(__file__, 488, 16), append_70089, *[capline_right_70090], **kwargs_70091)
        
        # SSA join for if statement (line 479)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 472)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 490):
        
        # Assigning a List to a Name (line 490):
        
        # Obtaining an instance of the builtin type 'list' (line 490)
        list_70093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 490)
        
        # Assigning a type to the variable 'artists' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'artists', list_70093)
        
        # Call to extend(...): (line 491)
        # Processing the call arguments (line 491)
        # Getting the type of 'handle_barlinecols' (line 491)
        handle_barlinecols_70096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 23), 'handle_barlinecols', False)
        # Processing the call keyword arguments (line 491)
        kwargs_70097 = {}
        # Getting the type of 'artists' (line 491)
        artists_70094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'artists', False)
        # Obtaining the member 'extend' of a type (line 491)
        extend_70095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 8), artists_70094, 'extend')
        # Calling extend(args, kwargs) (line 491)
        extend_call_result_70098 = invoke(stypy.reporting.localization.Localization(__file__, 491, 8), extend_70095, *[handle_barlinecols_70096], **kwargs_70097)
        
        
        # Call to extend(...): (line 492)
        # Processing the call arguments (line 492)
        # Getting the type of 'handle_caplines' (line 492)
        handle_caplines_70101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 23), 'handle_caplines', False)
        # Processing the call keyword arguments (line 492)
        kwargs_70102 = {}
        # Getting the type of 'artists' (line 492)
        artists_70099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'artists', False)
        # Obtaining the member 'extend' of a type (line 492)
        extend_70100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 8), artists_70099, 'extend')
        # Calling extend(args, kwargs) (line 492)
        extend_call_result_70103 = invoke(stypy.reporting.localization.Localization(__file__, 492, 8), extend_70100, *[handle_caplines_70101], **kwargs_70102)
        
        
        # Call to append(...): (line 493)
        # Processing the call arguments (line 493)
        # Getting the type of 'legline' (line 493)
        legline_70106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 23), 'legline', False)
        # Processing the call keyword arguments (line 493)
        kwargs_70107 = {}
        # Getting the type of 'artists' (line 493)
        artists_70104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'artists', False)
        # Obtaining the member 'append' of a type (line 493)
        append_70105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), artists_70104, 'append')
        # Calling append(args, kwargs) (line 493)
        append_call_result_70108 = invoke(stypy.reporting.localization.Localization(__file__, 493, 8), append_70105, *[legline_70106], **kwargs_70107)
        
        
        # Call to append(...): (line 494)
        # Processing the call arguments (line 494)
        # Getting the type of 'legline_marker' (line 494)
        legline_marker_70111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 23), 'legline_marker', False)
        # Processing the call keyword arguments (line 494)
        kwargs_70112 = {}
        # Getting the type of 'artists' (line 494)
        artists_70109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'artists', False)
        # Obtaining the member 'append' of a type (line 494)
        append_70110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 8), artists_70109, 'append')
        # Calling append(args, kwargs) (line 494)
        append_call_result_70113 = invoke(stypy.reporting.localization.Localization(__file__, 494, 8), append_70110, *[legline_marker_70111], **kwargs_70112)
        
        
        # Getting the type of 'artists' (line 496)
        artists_70114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 22), 'artists')
        # Testing the type of a for loop iterable (line 496)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 496, 8), artists_70114)
        # Getting the type of the for loop variable (line 496)
        for_loop_var_70115 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 496, 8), artists_70114)
        # Assigning a type to the variable 'artist' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'artist', for_loop_var_70115)
        # SSA begins for a for statement (line 496)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_transform(...): (line 497)
        # Processing the call arguments (line 497)
        # Getting the type of 'trans' (line 497)
        trans_70118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 33), 'trans', False)
        # Processing the call keyword arguments (line 497)
        kwargs_70119 = {}
        # Getting the type of 'artist' (line 497)
        artist_70116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'artist', False)
        # Obtaining the member 'set_transform' of a type (line 497)
        set_transform_70117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 12), artist_70116, 'set_transform')
        # Calling set_transform(args, kwargs) (line 497)
        set_transform_call_result_70120 = invoke(stypy.reporting.localization.Localization(__file__, 497, 12), set_transform_70117, *[trans_70118], **kwargs_70119)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'artists' (line 499)
        artists_70121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 15), 'artists')
        # Assigning a type to the variable 'stypy_return_type' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'stypy_return_type', artists_70121)
        
        # ################# End of 'create_artists(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_artists' in the type store
        # Getting the type of 'stypy_return_type' (line 412)
        stypy_return_type_70122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_70122)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_artists'
        return stypy_return_type_70122


# Assigning a type to the variable 'HandlerErrorbar' (line 389)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 0), 'HandlerErrorbar', HandlerErrorbar)
# Declaration of the 'HandlerStem' class
# Getting the type of 'HandlerNpointsYoffsets' (line 502)
HandlerNpointsYoffsets_70123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 18), 'HandlerNpointsYoffsets')

class HandlerStem(HandlerNpointsYoffsets_70123, ):
    unicode_70124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, (-1)), 'unicode', u'\n    Handler for Errorbars\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_70125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 34), 'float')
        # Getting the type of 'None' (line 506)
        None_70126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 49), 'None')
        # Getting the type of 'None' (line 507)
        None_70127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 24), 'None')
        # Getting the type of 'None' (line 507)
        None_70128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 39), 'None')
        defaults = [float_70125, None_70126, None_70127, None_70128]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 506, 4, False)
        # Assigning a type to the variable 'self' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerStem.__init__', ['marker_pad', 'numpoints', 'bottom', 'yoffsets'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['marker_pad', 'numpoints', 'bottom', 'yoffsets'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'self' (line 509)
        self_70131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 40), 'self', False)
        # Processing the call keyword arguments (line 509)
        # Getting the type of 'marker_pad' (line 509)
        marker_pad_70132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 57), 'marker_pad', False)
        keyword_70133 = marker_pad_70132
        # Getting the type of 'numpoints' (line 510)
        numpoints_70134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 50), 'numpoints', False)
        keyword_70135 = numpoints_70134
        # Getting the type of 'yoffsets' (line 511)
        yoffsets_70136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 49), 'yoffsets', False)
        keyword_70137 = yoffsets_70136
        # Getting the type of 'kw' (line 512)
        kw_70138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 42), 'kw', False)
        kwargs_70139 = {'kw_70138': kw_70138, 'marker_pad': keyword_70133, 'yoffsets': keyword_70137, 'numpoints': keyword_70135}
        # Getting the type of 'HandlerNpointsYoffsets' (line 509)
        HandlerNpointsYoffsets_70129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'HandlerNpointsYoffsets', False)
        # Obtaining the member '__init__' of a type (line 509)
        init___70130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), HandlerNpointsYoffsets_70129, '__init__')
        # Calling __init__(args, kwargs) (line 509)
        init___call_result_70140 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), init___70130, *[self_70131], **kwargs_70139)
        
        
        # Assigning a Name to a Attribute (line 513):
        
        # Assigning a Name to a Attribute (line 513):
        # Getting the type of 'bottom' (line 513)
        bottom_70141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 23), 'bottom')
        # Getting the type of 'self' (line 513)
        self_70142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'self')
        # Setting the type of the member '_bottom' of a type (line 513)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 8), self_70142, '_bottom', bottom_70141)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_ydata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_ydata'
        module_type_store = module_type_store.open_function_context('get_ydata', 515, 4, False)
        # Assigning a type to the variable 'self' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerStem.get_ydata.__dict__.__setitem__('stypy_localization', localization)
        HandlerStem.get_ydata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerStem.get_ydata.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerStem.get_ydata.__dict__.__setitem__('stypy_function_name', 'HandlerStem.get_ydata')
        HandlerStem.get_ydata.__dict__.__setitem__('stypy_param_names_list', ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'])
        HandlerStem.get_ydata.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerStem.get_ydata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerStem.get_ydata.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerStem.get_ydata.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerStem.get_ydata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerStem.get_ydata.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerStem.get_ydata', ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_ydata', localization, ['legend', 'xdescent', 'ydescent', 'width', 'height', 'fontsize'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_ydata(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 516)
        # Getting the type of 'self' (line 516)
        self_70143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'self')
        # Obtaining the member '_yoffsets' of a type (line 516)
        _yoffsets_70144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 11), self_70143, '_yoffsets')
        # Getting the type of 'None' (line 516)
        None_70145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 29), 'None')
        
        (may_be_70146, more_types_in_union_70147) = may_be_none(_yoffsets_70144, None_70145)

        if may_be_70146:

            if more_types_in_union_70147:
                # Runtime conditional SSA (line 516)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 517):
            
            # Assigning a BinOp to a Name (line 517):
            # Getting the type of 'height' (line 517)
            height_70148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 20), 'height')
            float_70149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 30), 'float')
            # Getting the type of 'legend' (line 517)
            legend_70150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 36), 'legend')
            # Obtaining the member '_scatteryoffsets' of a type (line 517)
            _scatteryoffsets_70151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 36), legend_70150, '_scatteryoffsets')
            # Applying the binary operator '*' (line 517)
            result_mul_70152 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 30), '*', float_70149, _scatteryoffsets_70151)
            
            float_70153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 62), 'float')
            # Applying the binary operator '+' (line 517)
            result_add_70154 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 30), '+', result_mul_70152, float_70153)
            
            # Applying the binary operator '*' (line 517)
            result_mul_70155 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 20), '*', height_70148, result_add_70154)
            
            # Assigning a type to the variable 'ydata' (line 517)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'ydata', result_mul_70155)

            if more_types_in_union_70147:
                # Runtime conditional SSA for else branch (line 516)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_70146) or more_types_in_union_70147):
            
            # Assigning a BinOp to a Name (line 519):
            
            # Assigning a BinOp to a Name (line 519):
            # Getting the type of 'height' (line 519)
            height_70156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 20), 'height')
            
            # Call to asarray(...): (line 519)
            # Processing the call arguments (line 519)
            # Getting the type of 'self' (line 519)
            self_70159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 40), 'self', False)
            # Obtaining the member '_yoffsets' of a type (line 519)
            _yoffsets_70160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 40), self_70159, '_yoffsets')
            # Processing the call keyword arguments (line 519)
            kwargs_70161 = {}
            # Getting the type of 'np' (line 519)
            np_70157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 29), 'np', False)
            # Obtaining the member 'asarray' of a type (line 519)
            asarray_70158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 29), np_70157, 'asarray')
            # Calling asarray(args, kwargs) (line 519)
            asarray_call_result_70162 = invoke(stypy.reporting.localization.Localization(__file__, 519, 29), asarray_70158, *[_yoffsets_70160], **kwargs_70161)
            
            # Applying the binary operator '*' (line 519)
            result_mul_70163 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 20), '*', height_70156, asarray_call_result_70162)
            
            # Assigning a type to the variable 'ydata' (line 519)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'ydata', result_mul_70163)

            if (may_be_70146 and more_types_in_union_70147):
                # SSA join for if statement (line 516)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'ydata' (line 521)
        ydata_70164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 15), 'ydata')
        # Assigning a type to the variable 'stypy_return_type' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'stypy_return_type', ydata_70164)
        
        # ################# End of 'get_ydata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_ydata' in the type store
        # Getting the type of 'stypy_return_type' (line 515)
        stypy_return_type_70165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_70165)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_ydata'
        return stypy_return_type_70165


    @norecursion
    def create_artists(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_artists'
        module_type_store = module_type_store.open_function_context('create_artists', 523, 4, False)
        # Assigning a type to the variable 'self' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerStem.create_artists.__dict__.__setitem__('stypy_localization', localization)
        HandlerStem.create_artists.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerStem.create_artists.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerStem.create_artists.__dict__.__setitem__('stypy_function_name', 'HandlerStem.create_artists')
        HandlerStem.create_artists.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'])
        HandlerStem.create_artists.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerStem.create_artists.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerStem.create_artists.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerStem.create_artists.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerStem.create_artists.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerStem.create_artists.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerStem.create_artists', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_artists', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_artists(...)' code ##################

        
        # Assigning a Name to a Tuple (line 527):
        
        # Assigning a Subscript to a Name (line 527):
        
        # Obtaining the type of the subscript
        int_70166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 8), 'int')
        # Getting the type of 'orig_handle' (line 527)
        orig_handle_70167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 42), 'orig_handle')
        # Obtaining the member '__getitem__' of a type (line 527)
        getitem___70168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 8), orig_handle_70167, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 527)
        subscript_call_result_70169 = invoke(stypy.reporting.localization.Localization(__file__, 527, 8), getitem___70168, int_70166)
        
        # Assigning a type to the variable 'tuple_var_assignment_68840' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'tuple_var_assignment_68840', subscript_call_result_70169)
        
        # Assigning a Subscript to a Name (line 527):
        
        # Obtaining the type of the subscript
        int_70170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 8), 'int')
        # Getting the type of 'orig_handle' (line 527)
        orig_handle_70171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 42), 'orig_handle')
        # Obtaining the member '__getitem__' of a type (line 527)
        getitem___70172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 8), orig_handle_70171, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 527)
        subscript_call_result_70173 = invoke(stypy.reporting.localization.Localization(__file__, 527, 8), getitem___70172, int_70170)
        
        # Assigning a type to the variable 'tuple_var_assignment_68841' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'tuple_var_assignment_68841', subscript_call_result_70173)
        
        # Assigning a Subscript to a Name (line 527):
        
        # Obtaining the type of the subscript
        int_70174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 8), 'int')
        # Getting the type of 'orig_handle' (line 527)
        orig_handle_70175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 42), 'orig_handle')
        # Obtaining the member '__getitem__' of a type (line 527)
        getitem___70176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 8), orig_handle_70175, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 527)
        subscript_call_result_70177 = invoke(stypy.reporting.localization.Localization(__file__, 527, 8), getitem___70176, int_70174)
        
        # Assigning a type to the variable 'tuple_var_assignment_68842' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'tuple_var_assignment_68842', subscript_call_result_70177)
        
        # Assigning a Name to a Name (line 527):
        # Getting the type of 'tuple_var_assignment_68840' (line 527)
        tuple_var_assignment_68840_70178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'tuple_var_assignment_68840')
        # Assigning a type to the variable 'markerline' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'markerline', tuple_var_assignment_68840_70178)
        
        # Assigning a Name to a Name (line 527):
        # Getting the type of 'tuple_var_assignment_68841' (line 527)
        tuple_var_assignment_68841_70179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'tuple_var_assignment_68841')
        # Assigning a type to the variable 'stemlines' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 20), 'stemlines', tuple_var_assignment_68841_70179)
        
        # Assigning a Name to a Name (line 527):
        # Getting the type of 'tuple_var_assignment_68842' (line 527)
        tuple_var_assignment_68842_70180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'tuple_var_assignment_68842')
        # Assigning a type to the variable 'baseline' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 31), 'baseline', tuple_var_assignment_68842_70180)
        
        # Assigning a Call to a Tuple (line 529):
        
        # Assigning a Call to a Name:
        
        # Call to get_xdata(...): (line 529)
        # Processing the call arguments (line 529)
        # Getting the type of 'legend' (line 529)
        legend_70183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 45), 'legend', False)
        # Getting the type of 'xdescent' (line 529)
        xdescent_70184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 53), 'xdescent', False)
        # Getting the type of 'ydescent' (line 529)
        ydescent_70185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 63), 'ydescent', False)
        # Getting the type of 'width' (line 530)
        width_70186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 45), 'width', False)
        # Getting the type of 'height' (line 530)
        height_70187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 52), 'height', False)
        # Getting the type of 'fontsize' (line 530)
        fontsize_70188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 60), 'fontsize', False)
        # Processing the call keyword arguments (line 529)
        kwargs_70189 = {}
        # Getting the type of 'self' (line 529)
        self_70181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 30), 'self', False)
        # Obtaining the member 'get_xdata' of a type (line 529)
        get_xdata_70182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 30), self_70181, 'get_xdata')
        # Calling get_xdata(args, kwargs) (line 529)
        get_xdata_call_result_70190 = invoke(stypy.reporting.localization.Localization(__file__, 529, 30), get_xdata_70182, *[legend_70183, xdescent_70184, ydescent_70185, width_70186, height_70187, fontsize_70188], **kwargs_70189)
        
        # Assigning a type to the variable 'call_assignment_68843' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'call_assignment_68843', get_xdata_call_result_70190)
        
        # Assigning a Call to a Name (line 529):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_70193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 8), 'int')
        # Processing the call keyword arguments
        kwargs_70194 = {}
        # Getting the type of 'call_assignment_68843' (line 529)
        call_assignment_68843_70191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'call_assignment_68843', False)
        # Obtaining the member '__getitem__' of a type (line 529)
        getitem___70192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 8), call_assignment_68843_70191, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_70195 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___70192, *[int_70193], **kwargs_70194)
        
        # Assigning a type to the variable 'call_assignment_68844' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'call_assignment_68844', getitem___call_result_70195)
        
        # Assigning a Name to a Name (line 529):
        # Getting the type of 'call_assignment_68844' (line 529)
        call_assignment_68844_70196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'call_assignment_68844')
        # Assigning a type to the variable 'xdata' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'xdata', call_assignment_68844_70196)
        
        # Assigning a Call to a Name (line 529):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_70199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 8), 'int')
        # Processing the call keyword arguments
        kwargs_70200 = {}
        # Getting the type of 'call_assignment_68843' (line 529)
        call_assignment_68843_70197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'call_assignment_68843', False)
        # Obtaining the member '__getitem__' of a type (line 529)
        getitem___70198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 8), call_assignment_68843_70197, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_70201 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___70198, *[int_70199], **kwargs_70200)
        
        # Assigning a type to the variable 'call_assignment_68845' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'call_assignment_68845', getitem___call_result_70201)
        
        # Assigning a Name to a Name (line 529):
        # Getting the type of 'call_assignment_68845' (line 529)
        call_assignment_68845_70202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'call_assignment_68845')
        # Assigning a type to the variable 'xdata_marker' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 15), 'xdata_marker', call_assignment_68845_70202)
        
        # Assigning a Call to a Name (line 532):
        
        # Assigning a Call to a Name (line 532):
        
        # Call to get_ydata(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 'legend' (line 532)
        legend_70205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 31), 'legend', False)
        # Getting the type of 'xdescent' (line 532)
        xdescent_70206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 39), 'xdescent', False)
        # Getting the type of 'ydescent' (line 532)
        ydescent_70207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 49), 'ydescent', False)
        # Getting the type of 'width' (line 533)
        width_70208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 31), 'width', False)
        # Getting the type of 'height' (line 533)
        height_70209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 38), 'height', False)
        # Getting the type of 'fontsize' (line 533)
        fontsize_70210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 46), 'fontsize', False)
        # Processing the call keyword arguments (line 532)
        kwargs_70211 = {}
        # Getting the type of 'self' (line 532)
        self_70203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 16), 'self', False)
        # Obtaining the member 'get_ydata' of a type (line 532)
        get_ydata_70204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 16), self_70203, 'get_ydata')
        # Calling get_ydata(args, kwargs) (line 532)
        get_ydata_call_result_70212 = invoke(stypy.reporting.localization.Localization(__file__, 532, 16), get_ydata_70204, *[legend_70205, xdescent_70206, ydescent_70207, width_70208, height_70209, fontsize_70210], **kwargs_70211)
        
        # Assigning a type to the variable 'ydata' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'ydata', get_ydata_call_result_70212)
        
        # Type idiom detected: calculating its left and rigth part (line 535)
        # Getting the type of 'self' (line 535)
        self_70213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 11), 'self')
        # Obtaining the member '_bottom' of a type (line 535)
        _bottom_70214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 11), self_70213, '_bottom')
        # Getting the type of 'None' (line 535)
        None_70215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 27), 'None')
        
        (may_be_70216, more_types_in_union_70217) = may_be_none(_bottom_70214, None_70215)

        if may_be_70216:

            if more_types_in_union_70217:
                # Runtime conditional SSA (line 535)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 536):
            
            # Assigning a Num to a Name (line 536):
            float_70218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 21), 'float')
            # Assigning a type to the variable 'bottom' (line 536)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'bottom', float_70218)

            if more_types_in_union_70217:
                # Runtime conditional SSA for else branch (line 535)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_70216) or more_types_in_union_70217):
            
            # Assigning a Attribute to a Name (line 538):
            
            # Assigning a Attribute to a Name (line 538):
            # Getting the type of 'self' (line 538)
            self_70219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 21), 'self')
            # Obtaining the member '_bottom' of a type (line 538)
            _bottom_70220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 21), self_70219, '_bottom')
            # Assigning a type to the variable 'bottom' (line 538)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'bottom', _bottom_70220)

            if (may_be_70216 and more_types_in_union_70217):
                # SSA join for if statement (line 535)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 540):
        
        # Assigning a Call to a Name (line 540):
        
        # Call to Line2D(...): (line 540)
        # Processing the call arguments (line 540)
        # Getting the type of 'xdata_marker' (line 540)
        xdata_marker_70222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 32), 'xdata_marker', False)
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 540)
        # Processing the call arguments (line 540)
        # Getting the type of 'xdata_marker' (line 540)
        xdata_marker_70224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 57), 'xdata_marker', False)
        # Processing the call keyword arguments (line 540)
        kwargs_70225 = {}
        # Getting the type of 'len' (line 540)
        len_70223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 53), 'len', False)
        # Calling len(args, kwargs) (line 540)
        len_call_result_70226 = invoke(stypy.reporting.localization.Localization(__file__, 540, 53), len_70223, *[xdata_marker_70224], **kwargs_70225)
        
        slice_70227 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 540, 46), None, len_call_result_70226, None)
        # Getting the type of 'ydata' (line 540)
        ydata_70228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 46), 'ydata', False)
        # Obtaining the member '__getitem__' of a type (line 540)
        getitem___70229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 46), ydata_70228, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 540)
        subscript_call_result_70230 = invoke(stypy.reporting.localization.Localization(__file__, 540, 46), getitem___70229, slice_70227)
        
        # Processing the call keyword arguments (line 540)
        kwargs_70231 = {}
        # Getting the type of 'Line2D' (line 540)
        Line2D_70221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 25), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 540)
        Line2D_call_result_70232 = invoke(stypy.reporting.localization.Localization(__file__, 540, 25), Line2D_70221, *[xdata_marker_70222, subscript_call_result_70230], **kwargs_70231)
        
        # Assigning a type to the variable 'leg_markerline' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'leg_markerline', Line2D_call_result_70232)
        
        # Call to update_prop(...): (line 541)
        # Processing the call arguments (line 541)
        # Getting the type of 'leg_markerline' (line 541)
        leg_markerline_70235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 25), 'leg_markerline', False)
        # Getting the type of 'markerline' (line 541)
        markerline_70236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 41), 'markerline', False)
        # Getting the type of 'legend' (line 541)
        legend_70237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 53), 'legend', False)
        # Processing the call keyword arguments (line 541)
        kwargs_70238 = {}
        # Getting the type of 'self' (line 541)
        self_70233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 541)
        update_prop_70234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 8), self_70233, 'update_prop')
        # Calling update_prop(args, kwargs) (line 541)
        update_prop_call_result_70239 = invoke(stypy.reporting.localization.Localization(__file__, 541, 8), update_prop_70234, *[leg_markerline_70235, markerline_70236, legend_70237], **kwargs_70238)
        
        
        # Assigning a List to a Name (line 543):
        
        # Assigning a List to a Name (line 543):
        
        # Obtaining an instance of the builtin type 'list' (line 543)
        list_70240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 543)
        
        # Assigning a type to the variable 'leg_stemlines' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'leg_stemlines', list_70240)
        
        
        # Call to zip(...): (line 544)
        # Processing the call arguments (line 544)
        # Getting the type of 'xdata_marker' (line 544)
        xdata_marker_70242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 32), 'xdata_marker', False)
        # Getting the type of 'ydata' (line 544)
        ydata_70243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 46), 'ydata', False)
        # Processing the call keyword arguments (line 544)
        kwargs_70244 = {}
        # Getting the type of 'zip' (line 544)
        zip_70241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 28), 'zip', False)
        # Calling zip(args, kwargs) (line 544)
        zip_call_result_70245 = invoke(stypy.reporting.localization.Localization(__file__, 544, 28), zip_70241, *[xdata_marker_70242, ydata_70243], **kwargs_70244)
        
        # Testing the type of a for loop iterable (line 544)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 544, 8), zip_call_result_70245)
        # Getting the type of the for loop variable (line 544)
        for_loop_var_70246 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 544, 8), zip_call_result_70245)
        # Assigning a type to the variable 'thisx' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'thisx', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 8), for_loop_var_70246))
        # Assigning a type to the variable 'thisy' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'thisy', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 8), for_loop_var_70246))
        # SSA begins for a for statement (line 544)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 545):
        
        # Assigning a Call to a Name (line 545):
        
        # Call to Line2D(...): (line 545)
        # Processing the call arguments (line 545)
        
        # Obtaining an instance of the builtin type 'list' (line 545)
        list_70248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 545)
        # Adding element type (line 545)
        # Getting the type of 'thisx' (line 545)
        thisx_70249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 24), 'thisx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 23), list_70248, thisx_70249)
        # Adding element type (line 545)
        # Getting the type of 'thisx' (line 545)
        thisx_70250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 31), 'thisx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 23), list_70248, thisx_70250)
        
        
        # Obtaining an instance of the builtin type 'list' (line 545)
        list_70251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 545)
        # Adding element type (line 545)
        # Getting the type of 'bottom' (line 545)
        bottom_70252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 40), 'bottom', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 39), list_70251, bottom_70252)
        # Adding element type (line 545)
        # Getting the type of 'thisy' (line 545)
        thisy_70253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 48), 'thisy', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 39), list_70251, thisy_70253)
        
        # Processing the call keyword arguments (line 545)
        kwargs_70254 = {}
        # Getting the type of 'Line2D' (line 545)
        Line2D_70247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 545)
        Line2D_call_result_70255 = invoke(stypy.reporting.localization.Localization(__file__, 545, 16), Line2D_70247, *[list_70248, list_70251], **kwargs_70254)
        
        # Assigning a type to the variable 'l' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'l', Line2D_call_result_70255)
        
        # Call to append(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'l' (line 546)
        l_70258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 33), 'l', False)
        # Processing the call keyword arguments (line 546)
        kwargs_70259 = {}
        # Getting the type of 'leg_stemlines' (line 546)
        leg_stemlines_70256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'leg_stemlines', False)
        # Obtaining the member 'append' of a type (line 546)
        append_70257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 12), leg_stemlines_70256, 'append')
        # Calling append(args, kwargs) (line 546)
        append_call_result_70260 = invoke(stypy.reporting.localization.Localization(__file__, 546, 12), append_70257, *[l_70258], **kwargs_70259)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to zip(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'leg_stemlines' (line 548)
        leg_stemlines_70262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 25), 'leg_stemlines', False)
        # Getting the type of 'stemlines' (line 548)
        stemlines_70263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 40), 'stemlines', False)
        # Processing the call keyword arguments (line 548)
        kwargs_70264 = {}
        # Getting the type of 'zip' (line 548)
        zip_70261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 21), 'zip', False)
        # Calling zip(args, kwargs) (line 548)
        zip_call_result_70265 = invoke(stypy.reporting.localization.Localization(__file__, 548, 21), zip_70261, *[leg_stemlines_70262, stemlines_70263], **kwargs_70264)
        
        # Testing the type of a for loop iterable (line 548)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 548, 8), zip_call_result_70265)
        # Getting the type of the for loop variable (line 548)
        for_loop_var_70266 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 548, 8), zip_call_result_70265)
        # Assigning a type to the variable 'lm' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'lm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 8), for_loop_var_70266))
        # Assigning a type to the variable 'm' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 8), for_loop_var_70266))
        # SSA begins for a for statement (line 548)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to update_prop(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'lm' (line 549)
        lm_70269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 29), 'lm', False)
        # Getting the type of 'm' (line 549)
        m_70270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 33), 'm', False)
        # Getting the type of 'legend' (line 549)
        legend_70271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 36), 'legend', False)
        # Processing the call keyword arguments (line 549)
        kwargs_70272 = {}
        # Getting the type of 'self' (line 549)
        self_70267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 549)
        update_prop_70268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 12), self_70267, 'update_prop')
        # Calling update_prop(args, kwargs) (line 549)
        update_prop_call_result_70273 = invoke(stypy.reporting.localization.Localization(__file__, 549, 12), update_prop_70268, *[lm_70269, m_70270, legend_70271], **kwargs_70272)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 551):
        
        # Assigning a Call to a Name (line 551):
        
        # Call to Line2D(...): (line 551)
        # Processing the call arguments (line 551)
        
        # Obtaining an instance of the builtin type 'list' (line 551)
        list_70275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 551)
        # Adding element type (line 551)
        
        # Call to min(...): (line 551)
        # Processing the call arguments (line 551)
        # Getting the type of 'xdata' (line 551)
        xdata_70278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 38), 'xdata', False)
        # Processing the call keyword arguments (line 551)
        kwargs_70279 = {}
        # Getting the type of 'np' (line 551)
        np_70276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 31), 'np', False)
        # Obtaining the member 'min' of a type (line 551)
        min_70277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 31), np_70276, 'min')
        # Calling min(args, kwargs) (line 551)
        min_call_result_70280 = invoke(stypy.reporting.localization.Localization(__file__, 551, 31), min_70277, *[xdata_70278], **kwargs_70279)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 30), list_70275, min_call_result_70280)
        # Adding element type (line 551)
        
        # Call to max(...): (line 551)
        # Processing the call arguments (line 551)
        # Getting the type of 'xdata' (line 551)
        xdata_70283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 53), 'xdata', False)
        # Processing the call keyword arguments (line 551)
        kwargs_70284 = {}
        # Getting the type of 'np' (line 551)
        np_70281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 46), 'np', False)
        # Obtaining the member 'max' of a type (line 551)
        max_70282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 46), np_70281, 'max')
        # Calling max(args, kwargs) (line 551)
        max_call_result_70285 = invoke(stypy.reporting.localization.Localization(__file__, 551, 46), max_70282, *[xdata_70283], **kwargs_70284)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 30), list_70275, max_call_result_70285)
        
        
        # Obtaining an instance of the builtin type 'list' (line 552)
        list_70286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 552)
        # Adding element type (line 552)
        # Getting the type of 'bottom' (line 552)
        bottom_70287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 31), 'bottom', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 30), list_70286, bottom_70287)
        # Adding element type (line 552)
        # Getting the type of 'bottom' (line 552)
        bottom_70288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 39), 'bottom', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 30), list_70286, bottom_70288)
        
        # Processing the call keyword arguments (line 551)
        kwargs_70289 = {}
        # Getting the type of 'Line2D' (line 551)
        Line2D_70274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 23), 'Line2D', False)
        # Calling Line2D(args, kwargs) (line 551)
        Line2D_call_result_70290 = invoke(stypy.reporting.localization.Localization(__file__, 551, 23), Line2D_70274, *[list_70275, list_70286], **kwargs_70289)
        
        # Assigning a type to the variable 'leg_baseline' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'leg_baseline', Line2D_call_result_70290)
        
        # Call to update_prop(...): (line 554)
        # Processing the call arguments (line 554)
        # Getting the type of 'leg_baseline' (line 554)
        leg_baseline_70293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 25), 'leg_baseline', False)
        # Getting the type of 'baseline' (line 554)
        baseline_70294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 39), 'baseline', False)
        # Getting the type of 'legend' (line 554)
        legend_70295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 49), 'legend', False)
        # Processing the call keyword arguments (line 554)
        kwargs_70296 = {}
        # Getting the type of 'self' (line 554)
        self_70291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 554)
        update_prop_70292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 8), self_70291, 'update_prop')
        # Calling update_prop(args, kwargs) (line 554)
        update_prop_call_result_70297 = invoke(stypy.reporting.localization.Localization(__file__, 554, 8), update_prop_70292, *[leg_baseline_70293, baseline_70294, legend_70295], **kwargs_70296)
        
        
        # Assigning a List to a Name (line 556):
        
        # Assigning a List to a Name (line 556):
        
        # Obtaining an instance of the builtin type 'list' (line 556)
        list_70298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 556)
        # Adding element type (line 556)
        # Getting the type of 'leg_markerline' (line 556)
        leg_markerline_70299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 19), 'leg_markerline')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 18), list_70298, leg_markerline_70299)
        
        # Assigning a type to the variable 'artists' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'artists', list_70298)
        
        # Call to extend(...): (line 557)
        # Processing the call arguments (line 557)
        # Getting the type of 'leg_stemlines' (line 557)
        leg_stemlines_70302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 23), 'leg_stemlines', False)
        # Processing the call keyword arguments (line 557)
        kwargs_70303 = {}
        # Getting the type of 'artists' (line 557)
        artists_70300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'artists', False)
        # Obtaining the member 'extend' of a type (line 557)
        extend_70301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 8), artists_70300, 'extend')
        # Calling extend(args, kwargs) (line 557)
        extend_call_result_70304 = invoke(stypy.reporting.localization.Localization(__file__, 557, 8), extend_70301, *[leg_stemlines_70302], **kwargs_70303)
        
        
        # Call to append(...): (line 558)
        # Processing the call arguments (line 558)
        # Getting the type of 'leg_baseline' (line 558)
        leg_baseline_70307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 23), 'leg_baseline', False)
        # Processing the call keyword arguments (line 558)
        kwargs_70308 = {}
        # Getting the type of 'artists' (line 558)
        artists_70305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'artists', False)
        # Obtaining the member 'append' of a type (line 558)
        append_70306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 8), artists_70305, 'append')
        # Calling append(args, kwargs) (line 558)
        append_call_result_70309 = invoke(stypy.reporting.localization.Localization(__file__, 558, 8), append_70306, *[leg_baseline_70307], **kwargs_70308)
        
        
        # Getting the type of 'artists' (line 560)
        artists_70310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 22), 'artists')
        # Testing the type of a for loop iterable (line 560)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 560, 8), artists_70310)
        # Getting the type of the for loop variable (line 560)
        for_loop_var_70311 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 560, 8), artists_70310)
        # Assigning a type to the variable 'artist' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'artist', for_loop_var_70311)
        # SSA begins for a for statement (line 560)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_transform(...): (line 561)
        # Processing the call arguments (line 561)
        # Getting the type of 'trans' (line 561)
        trans_70314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 33), 'trans', False)
        # Processing the call keyword arguments (line 561)
        kwargs_70315 = {}
        # Getting the type of 'artist' (line 561)
        artist_70312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 12), 'artist', False)
        # Obtaining the member 'set_transform' of a type (line 561)
        set_transform_70313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 12), artist_70312, 'set_transform')
        # Calling set_transform(args, kwargs) (line 561)
        set_transform_call_result_70316 = invoke(stypy.reporting.localization.Localization(__file__, 561, 12), set_transform_70313, *[trans_70314], **kwargs_70315)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'artists' (line 563)
        artists_70317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 15), 'artists')
        # Assigning a type to the variable 'stypy_return_type' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'stypy_return_type', artists_70317)
        
        # ################# End of 'create_artists(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_artists' in the type store
        # Getting the type of 'stypy_return_type' (line 523)
        stypy_return_type_70318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_70318)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_artists'
        return stypy_return_type_70318


# Assigning a type to the variable 'HandlerStem' (line 502)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 0), 'HandlerStem', HandlerStem)
# Declaration of the 'HandlerTuple' class
# Getting the type of 'HandlerBase' (line 566)
HandlerBase_70319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 19), 'HandlerBase')

class HandlerTuple(HandlerBase_70319, ):
    unicode_70320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, (-1)), 'unicode', u'\n    Handler for Tuple.\n\n    Additional kwargs are passed through to `HandlerBase`.\n\n    Parameters\n    ----------\n\n    ndivide : int, optional\n        The number of sections to divide the legend area into.  If None,\n        use the length of the input tuple. Default is 1.\n\n\n    pad : float, optional\n        If None, fall back to `legend.borderpad` as the default.\n        In units of fraction of font size. Default is None.\n\n\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_70321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 31), 'int')
        # Getting the type of 'None' (line 587)
        None_70322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 38), 'None')
        defaults = [int_70321, None_70322]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 587, 4, False)
        # Assigning a type to the variable 'self' (line 588)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerTuple.__init__', ['ndivide', 'pad'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['ndivide', 'pad'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 589):
        
        # Assigning a Name to a Attribute (line 589):
        # Getting the type of 'ndivide' (line 589)
        ndivide_70323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 24), 'ndivide')
        # Getting the type of 'self' (line 589)
        self_70324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'self')
        # Setting the type of the member '_ndivide' of a type (line 589)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 8), self_70324, '_ndivide', ndivide_70323)
        
        # Assigning a Name to a Attribute (line 590):
        
        # Assigning a Name to a Attribute (line 590):
        # Getting the type of 'pad' (line 590)
        pad_70325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 20), 'pad')
        # Getting the type of 'self' (line 590)
        self_70326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'self')
        # Setting the type of the member '_pad' of a type (line 590)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 8), self_70326, '_pad', pad_70325)
        
        # Call to __init__(...): (line 591)
        # Processing the call arguments (line 591)
        # Getting the type of 'self' (line 591)
        self_70329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 29), 'self', False)
        # Processing the call keyword arguments (line 591)
        # Getting the type of 'kwargs' (line 591)
        kwargs_70330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 37), 'kwargs', False)
        kwargs_70331 = {'kwargs_70330': kwargs_70330}
        # Getting the type of 'HandlerBase' (line 591)
        HandlerBase_70327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'HandlerBase', False)
        # Obtaining the member '__init__' of a type (line 591)
        init___70328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 8), HandlerBase_70327, '__init__')
        # Calling __init__(args, kwargs) (line 591)
        init___call_result_70332 = invoke(stypy.reporting.localization.Localization(__file__, 591, 8), init___70328, *[self_70329], **kwargs_70331)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def create_artists(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_artists'
        module_type_store = module_type_store.open_function_context('create_artists', 593, 4, False)
        # Assigning a type to the variable 'self' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerTuple.create_artists.__dict__.__setitem__('stypy_localization', localization)
        HandlerTuple.create_artists.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerTuple.create_artists.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerTuple.create_artists.__dict__.__setitem__('stypy_function_name', 'HandlerTuple.create_artists')
        HandlerTuple.create_artists.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'])
        HandlerTuple.create_artists.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerTuple.create_artists.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerTuple.create_artists.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerTuple.create_artists.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerTuple.create_artists.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerTuple.create_artists.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerTuple.create_artists', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_artists', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_artists(...)' code ##################

        
        # Assigning a Call to a Name (line 597):
        
        # Assigning a Call to a Name (line 597):
        
        # Call to get_legend_handler_map(...): (line 597)
        # Processing the call keyword arguments (line 597)
        kwargs_70335 = {}
        # Getting the type of 'legend' (line 597)
        legend_70333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 22), 'legend', False)
        # Obtaining the member 'get_legend_handler_map' of a type (line 597)
        get_legend_handler_map_70334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 22), legend_70333, 'get_legend_handler_map')
        # Calling get_legend_handler_map(args, kwargs) (line 597)
        get_legend_handler_map_call_result_70336 = invoke(stypy.reporting.localization.Localization(__file__, 597, 22), get_legend_handler_map_70334, *[], **kwargs_70335)
        
        # Assigning a type to the variable 'handler_map' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'handler_map', get_legend_handler_map_call_result_70336)
        
        # Type idiom detected: calculating its left and rigth part (line 599)
        # Getting the type of 'self' (line 599)
        self_70337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 11), 'self')
        # Obtaining the member '_ndivide' of a type (line 599)
        _ndivide_70338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 11), self_70337, '_ndivide')
        # Getting the type of 'None' (line 599)
        None_70339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 28), 'None')
        
        (may_be_70340, more_types_in_union_70341) = may_be_none(_ndivide_70338, None_70339)

        if may_be_70340:

            if more_types_in_union_70341:
                # Runtime conditional SSA (line 599)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 600):
            
            # Assigning a Call to a Name (line 600):
            
            # Call to len(...): (line 600)
            # Processing the call arguments (line 600)
            # Getting the type of 'orig_handle' (line 600)
            orig_handle_70343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 26), 'orig_handle', False)
            # Processing the call keyword arguments (line 600)
            kwargs_70344 = {}
            # Getting the type of 'len' (line 600)
            len_70342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 22), 'len', False)
            # Calling len(args, kwargs) (line 600)
            len_call_result_70345 = invoke(stypy.reporting.localization.Localization(__file__, 600, 22), len_70342, *[orig_handle_70343], **kwargs_70344)
            
            # Assigning a type to the variable 'ndivide' (line 600)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'ndivide', len_call_result_70345)

            if more_types_in_union_70341:
                # Runtime conditional SSA for else branch (line 599)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_70340) or more_types_in_union_70341):
            
            # Assigning a Attribute to a Name (line 602):
            
            # Assigning a Attribute to a Name (line 602):
            # Getting the type of 'self' (line 602)
            self_70346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 22), 'self')
            # Obtaining the member '_ndivide' of a type (line 602)
            _ndivide_70347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 22), self_70346, '_ndivide')
            # Assigning a type to the variable 'ndivide' (line 602)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'ndivide', _ndivide_70347)

            if (may_be_70340 and more_types_in_union_70341):
                # SSA join for if statement (line 599)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 604)
        # Getting the type of 'self' (line 604)
        self_70348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 11), 'self')
        # Obtaining the member '_pad' of a type (line 604)
        _pad_70349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 11), self_70348, '_pad')
        # Getting the type of 'None' (line 604)
        None_70350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 24), 'None')
        
        (may_be_70351, more_types_in_union_70352) = may_be_none(_pad_70349, None_70350)

        if may_be_70351:

            if more_types_in_union_70352:
                # Runtime conditional SSA (line 604)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 605):
            
            # Assigning a BinOp to a Name (line 605):
            # Getting the type of 'legend' (line 605)
            legend_70353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 18), 'legend')
            # Obtaining the member 'borderpad' of a type (line 605)
            borderpad_70354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 18), legend_70353, 'borderpad')
            # Getting the type of 'fontsize' (line 605)
            fontsize_70355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 37), 'fontsize')
            # Applying the binary operator '*' (line 605)
            result_mul_70356 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 18), '*', borderpad_70354, fontsize_70355)
            
            # Assigning a type to the variable 'pad' (line 605)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'pad', result_mul_70356)

            if more_types_in_union_70352:
                # Runtime conditional SSA for else branch (line 604)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_70351) or more_types_in_union_70352):
            
            # Assigning a BinOp to a Name (line 607):
            
            # Assigning a BinOp to a Name (line 607):
            # Getting the type of 'self' (line 607)
            self_70357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 18), 'self')
            # Obtaining the member '_pad' of a type (line 607)
            _pad_70358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 18), self_70357, '_pad')
            # Getting the type of 'fontsize' (line 607)
            fontsize_70359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 30), 'fontsize')
            # Applying the binary operator '*' (line 607)
            result_mul_70360 = python_operator(stypy.reporting.localization.Localization(__file__, 607, 18), '*', _pad_70358, fontsize_70359)
            
            # Assigning a type to the variable 'pad' (line 607)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'pad', result_mul_70360)

            if (may_be_70351 and more_types_in_union_70352):
                # SSA join for if statement (line 604)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'ndivide' (line 609)
        ndivide_70361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 11), 'ndivide')
        int_70362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 21), 'int')
        # Applying the binary operator '>' (line 609)
        result_gt_70363 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 11), '>', ndivide_70361, int_70362)
        
        # Testing the type of an if condition (line 609)
        if_condition_70364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 609, 8), result_gt_70363)
        # Assigning a type to the variable 'if_condition_70364' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'if_condition_70364', if_condition_70364)
        # SSA begins for if statement (line 609)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 610):
        
        # Assigning a BinOp to a Name (line 610):
        # Getting the type of 'width' (line 610)
        width_70365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 21), 'width')
        # Getting the type of 'pad' (line 610)
        pad_70366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 29), 'pad')
        # Getting the type of 'ndivide' (line 610)
        ndivide_70367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 34), 'ndivide')
        int_70368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 44), 'int')
        # Applying the binary operator '-' (line 610)
        result_sub_70369 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 34), '-', ndivide_70367, int_70368)
        
        # Applying the binary operator '*' (line 610)
        result_mul_70370 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 29), '*', pad_70366, result_sub_70369)
        
        # Applying the binary operator '-' (line 610)
        result_sub_70371 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 21), '-', width_70365, result_mul_70370)
        
        # Getting the type of 'ndivide' (line 610)
        ndivide_70372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 50), 'ndivide')
        # Applying the binary operator 'div' (line 610)
        result_div_70373 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 20), 'div', result_sub_70371, ndivide_70372)
        
        # Assigning a type to the variable 'width' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'width', result_div_70373)
        # SSA join for if statement (line 609)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a ListComp to a Name (line 612):
        
        # Assigning a ListComp to a Name (line 612):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 612)
        # Processing the call arguments (line 612)
        # Getting the type of 'ndivide' (line 612)
        ndivide_70382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 59), 'ndivide', False)
        # Processing the call keyword arguments (line 612)
        kwargs_70383 = {}
        # Getting the type of 'range' (line 612)
        range_70381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 53), 'range', False)
        # Calling range(args, kwargs) (line 612)
        range_call_result_70384 = invoke(stypy.reporting.localization.Localization(__file__, 612, 53), range_70381, *[ndivide_70382], **kwargs_70383)
        
        comprehension_70385 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 15), range_call_result_70384)
        # Assigning a type to the variable 'i' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 15), 'i', comprehension_70385)
        # Getting the type of 'xdescent' (line 612)
        xdescent_70374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 15), 'xdescent')
        # Getting the type of 'width' (line 612)
        width_70375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 27), 'width')
        # Getting the type of 'pad' (line 612)
        pad_70376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 35), 'pad')
        # Applying the binary operator '+' (line 612)
        result_add_70377 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 27), '+', width_70375, pad_70376)
        
        # Getting the type of 'i' (line 612)
        i_70378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 42), 'i')
        # Applying the binary operator '*' (line 612)
        result_mul_70379 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 26), '*', result_add_70377, i_70378)
        
        # Applying the binary operator '-' (line 612)
        result_sub_70380 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 15), '-', xdescent_70374, result_mul_70379)
        
        list_70386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 15), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 612, 15), list_70386, result_sub_70380)
        # Assigning a type to the variable 'xds' (line 612)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'xds', list_70386)
        
        # Assigning a Call to a Name (line 613):
        
        # Assigning a Call to a Name (line 613):
        
        # Call to cycle(...): (line 613)
        # Processing the call arguments (line 613)
        # Getting the type of 'xds' (line 613)
        xds_70388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 26), 'xds', False)
        # Processing the call keyword arguments (line 613)
        kwargs_70389 = {}
        # Getting the type of 'cycle' (line 613)
        cycle_70387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 20), 'cycle', False)
        # Calling cycle(args, kwargs) (line 613)
        cycle_call_result_70390 = invoke(stypy.reporting.localization.Localization(__file__, 613, 20), cycle_70387, *[xds_70388], **kwargs_70389)
        
        # Assigning a type to the variable 'xds_cycle' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'xds_cycle', cycle_call_result_70390)
        
        # Assigning a List to a Name (line 615):
        
        # Assigning a List to a Name (line 615):
        
        # Obtaining an instance of the builtin type 'list' (line 615)
        list_70391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 615)
        
        # Assigning a type to the variable 'a_list' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'a_list', list_70391)
        
        # Getting the type of 'orig_handle' (line 616)
        orig_handle_70392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 23), 'orig_handle')
        # Testing the type of a for loop iterable (line 616)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 616, 8), orig_handle_70392)
        # Getting the type of the for loop variable (line 616)
        for_loop_var_70393 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 616, 8), orig_handle_70392)
        # Assigning a type to the variable 'handle1' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'handle1', for_loop_var_70393)
        # SSA begins for a for statement (line 616)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 617):
        
        # Assigning a Call to a Name (line 617):
        
        # Call to get_legend_handler(...): (line 617)
        # Processing the call arguments (line 617)
        # Getting the type of 'handler_map' (line 617)
        handler_map_70396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 48), 'handler_map', False)
        # Getting the type of 'handle1' (line 617)
        handle1_70397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 61), 'handle1', False)
        # Processing the call keyword arguments (line 617)
        kwargs_70398 = {}
        # Getting the type of 'legend' (line 617)
        legend_70394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 22), 'legend', False)
        # Obtaining the member 'get_legend_handler' of a type (line 617)
        get_legend_handler_70395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 22), legend_70394, 'get_legend_handler')
        # Calling get_legend_handler(args, kwargs) (line 617)
        get_legend_handler_call_result_70399 = invoke(stypy.reporting.localization.Localization(__file__, 617, 22), get_legend_handler_70395, *[handler_map_70396, handle1_70397], **kwargs_70398)
        
        # Assigning a type to the variable 'handler' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'handler', get_legend_handler_call_result_70399)
        
        # Assigning a Call to a Name (line 618):
        
        # Assigning a Call to a Name (line 618):
        
        # Call to create_artists(...): (line 618)
        # Processing the call arguments (line 618)
        # Getting the type of 'legend' (line 618)
        legend_70402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 45), 'legend', False)
        # Getting the type of 'handle1' (line 618)
        handle1_70403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 53), 'handle1', False)
        
        # Call to next(...): (line 619)
        # Processing the call arguments (line 619)
        # Getting the type of 'xds_cycle' (line 619)
        xds_cycle_70406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 54), 'xds_cycle', False)
        # Processing the call keyword arguments (line 619)
        kwargs_70407 = {}
        # Getting the type of 'six' (line 619)
        six_70404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 45), 'six', False)
        # Obtaining the member 'next' of a type (line 619)
        next_70405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 45), six_70404, 'next')
        # Calling next(args, kwargs) (line 619)
        next_call_result_70408 = invoke(stypy.reporting.localization.Localization(__file__, 619, 45), next_70405, *[xds_cycle_70406], **kwargs_70407)
        
        # Getting the type of 'ydescent' (line 620)
        ydescent_70409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 45), 'ydescent', False)
        # Getting the type of 'width' (line 621)
        width_70410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 45), 'width', False)
        # Getting the type of 'height' (line 621)
        height_70411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 52), 'height', False)
        # Getting the type of 'fontsize' (line 622)
        fontsize_70412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 45), 'fontsize', False)
        # Getting the type of 'trans' (line 623)
        trans_70413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 45), 'trans', False)
        # Processing the call keyword arguments (line 618)
        kwargs_70414 = {}
        # Getting the type of 'handler' (line 618)
        handler_70400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 22), 'handler', False)
        # Obtaining the member 'create_artists' of a type (line 618)
        create_artists_70401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 22), handler_70400, 'create_artists')
        # Calling create_artists(args, kwargs) (line 618)
        create_artists_call_result_70415 = invoke(stypy.reporting.localization.Localization(__file__, 618, 22), create_artists_70401, *[legend_70402, handle1_70403, next_call_result_70408, ydescent_70409, width_70410, height_70411, fontsize_70412, trans_70413], **kwargs_70414)
        
        # Assigning a type to the variable '_a_list' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), '_a_list', create_artists_call_result_70415)
        
        # Call to extend(...): (line 624)
        # Processing the call arguments (line 624)
        # Getting the type of '_a_list' (line 624)
        _a_list_70418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 26), '_a_list', False)
        # Processing the call keyword arguments (line 624)
        kwargs_70419 = {}
        # Getting the type of 'a_list' (line 624)
        a_list_70416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 12), 'a_list', False)
        # Obtaining the member 'extend' of a type (line 624)
        extend_70417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 12), a_list_70416, 'extend')
        # Calling extend(args, kwargs) (line 624)
        extend_call_result_70420 = invoke(stypy.reporting.localization.Localization(__file__, 624, 12), extend_70417, *[_a_list_70418], **kwargs_70419)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'a_list' (line 626)
        a_list_70421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 15), 'a_list')
        # Assigning a type to the variable 'stypy_return_type' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'stypy_return_type', a_list_70421)
        
        # ################# End of 'create_artists(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_artists' in the type store
        # Getting the type of 'stypy_return_type' (line 593)
        stypy_return_type_70422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_70422)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_artists'
        return stypy_return_type_70422


# Assigning a type to the variable 'HandlerTuple' (line 566)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 0), 'HandlerTuple', HandlerTuple)
# Declaration of the 'HandlerPolyCollection' class
# Getting the type of 'HandlerBase' (line 629)
HandlerBase_70423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 28), 'HandlerBase')

class HandlerPolyCollection(HandlerBase_70423, ):
    unicode_70424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, (-1)), 'unicode', u'\n    Handler for PolyCollection used in fill_between and stackplot.\n    ')

    @norecursion
    def _update_prop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update_prop'
        module_type_store = module_type_store.open_function_context('_update_prop', 633, 4, False)
        # Assigning a type to the variable 'self' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerPolyCollection._update_prop.__dict__.__setitem__('stypy_localization', localization)
        HandlerPolyCollection._update_prop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerPolyCollection._update_prop.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerPolyCollection._update_prop.__dict__.__setitem__('stypy_function_name', 'HandlerPolyCollection._update_prop')
        HandlerPolyCollection._update_prop.__dict__.__setitem__('stypy_param_names_list', ['legend_handle', 'orig_handle'])
        HandlerPolyCollection._update_prop.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerPolyCollection._update_prop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerPolyCollection._update_prop.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerPolyCollection._update_prop.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerPolyCollection._update_prop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerPolyCollection._update_prop.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerPolyCollection._update_prop', ['legend_handle', 'orig_handle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update_prop', localization, ['legend_handle', 'orig_handle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update_prop(...)' code ##################


        @norecursion
        def first_color(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'first_color'
            module_type_store = module_type_store.open_function_context('first_color', 634, 8, False)
            
            # Passed parameters checking function
            first_color.stypy_localization = localization
            first_color.stypy_type_of_self = None
            first_color.stypy_type_store = module_type_store
            first_color.stypy_function_name = 'first_color'
            first_color.stypy_param_names_list = ['colors']
            first_color.stypy_varargs_param_name = None
            first_color.stypy_kwargs_param_name = None
            first_color.stypy_call_defaults = defaults
            first_color.stypy_call_varargs = varargs
            first_color.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'first_color', ['colors'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'first_color', localization, ['colors'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'first_color(...)' code ##################

            
            # Type idiom detected: calculating its left and rigth part (line 635)
            # Getting the type of 'colors' (line 635)
            colors_70425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 15), 'colors')
            # Getting the type of 'None' (line 635)
            None_70426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 25), 'None')
            
            (may_be_70427, more_types_in_union_70428) = may_be_none(colors_70425, None_70426)

            if may_be_70427:

                if more_types_in_union_70428:
                    # Runtime conditional SSA (line 635)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'None' (line 636)
                None_70429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 23), 'None')
                # Assigning a type to the variable 'stypy_return_type' (line 636)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'stypy_return_type', None_70429)

                if more_types_in_union_70428:
                    # SSA join for if statement (line 635)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Name (line 637):
            
            # Assigning a Call to a Name (line 637):
            
            # Call to to_rgba_array(...): (line 637)
            # Processing the call arguments (line 637)
            # Getting the type of 'colors' (line 637)
            colors_70432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 43), 'colors', False)
            # Processing the call keyword arguments (line 637)
            kwargs_70433 = {}
            # Getting the type of 'mcolors' (line 637)
            mcolors_70430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 21), 'mcolors', False)
            # Obtaining the member 'to_rgba_array' of a type (line 637)
            to_rgba_array_70431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 21), mcolors_70430, 'to_rgba_array')
            # Calling to_rgba_array(args, kwargs) (line 637)
            to_rgba_array_call_result_70434 = invoke(stypy.reporting.localization.Localization(__file__, 637, 21), to_rgba_array_70431, *[colors_70432], **kwargs_70433)
            
            # Assigning a type to the variable 'colors' (line 637)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 12), 'colors', to_rgba_array_call_result_70434)
            
            
            # Call to len(...): (line 638)
            # Processing the call arguments (line 638)
            # Getting the type of 'colors' (line 638)
            colors_70436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 19), 'colors', False)
            # Processing the call keyword arguments (line 638)
            kwargs_70437 = {}
            # Getting the type of 'len' (line 638)
            len_70435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 15), 'len', False)
            # Calling len(args, kwargs) (line 638)
            len_call_result_70438 = invoke(stypy.reporting.localization.Localization(__file__, 638, 15), len_70435, *[colors_70436], **kwargs_70437)
            
            # Testing the type of an if condition (line 638)
            if_condition_70439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 638, 12), len_call_result_70438)
            # Assigning a type to the variable 'if_condition_70439' (line 638)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 12), 'if_condition_70439', if_condition_70439)
            # SSA begins for if statement (line 638)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_70440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 30), 'int')
            # Getting the type of 'colors' (line 639)
            colors_70441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 23), 'colors')
            # Obtaining the member '__getitem__' of a type (line 639)
            getitem___70442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 23), colors_70441, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 639)
            subscript_call_result_70443 = invoke(stypy.reporting.localization.Localization(__file__, 639, 23), getitem___70442, int_70440)
            
            # Assigning a type to the variable 'stypy_return_type' (line 639)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 16), 'stypy_return_type', subscript_call_result_70443)
            # SSA branch for the else part of an if statement (line 638)
            module_type_store.open_ssa_branch('else')
            unicode_70444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 23), 'unicode', u'none')
            # Assigning a type to the variable 'stypy_return_type' (line 641)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 16), 'stypy_return_type', unicode_70444)
            # SSA join for if statement (line 638)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'first_color(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'first_color' in the type store
            # Getting the type of 'stypy_return_type' (line 634)
            stypy_return_type_70445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_70445)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'first_color'
            return stypy_return_type_70445

        # Assigning a type to the variable 'first_color' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'first_color', first_color)

        @norecursion
        def get_first(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'get_first'
            module_type_store = module_type_store.open_function_context('get_first', 642, 8, False)
            
            # Passed parameters checking function
            get_first.stypy_localization = localization
            get_first.stypy_type_of_self = None
            get_first.stypy_type_store = module_type_store
            get_first.stypy_function_name = 'get_first'
            get_first.stypy_param_names_list = ['prop_array']
            get_first.stypy_varargs_param_name = None
            get_first.stypy_kwargs_param_name = None
            get_first.stypy_call_defaults = defaults
            get_first.stypy_call_varargs = varargs
            get_first.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'get_first', ['prop_array'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'get_first', localization, ['prop_array'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'get_first(...)' code ##################

            
            
            # Call to len(...): (line 643)
            # Processing the call arguments (line 643)
            # Getting the type of 'prop_array' (line 643)
            prop_array_70447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 19), 'prop_array', False)
            # Processing the call keyword arguments (line 643)
            kwargs_70448 = {}
            # Getting the type of 'len' (line 643)
            len_70446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 15), 'len', False)
            # Calling len(args, kwargs) (line 643)
            len_call_result_70449 = invoke(stypy.reporting.localization.Localization(__file__, 643, 15), len_70446, *[prop_array_70447], **kwargs_70448)
            
            # Testing the type of an if condition (line 643)
            if_condition_70450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 643, 12), len_call_result_70449)
            # Assigning a type to the variable 'if_condition_70450' (line 643)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 12), 'if_condition_70450', if_condition_70450)
            # SSA begins for if statement (line 643)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_70451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 34), 'int')
            # Getting the type of 'prop_array' (line 644)
            prop_array_70452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 23), 'prop_array')
            # Obtaining the member '__getitem__' of a type (line 644)
            getitem___70453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 23), prop_array_70452, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 644)
            subscript_call_result_70454 = invoke(stypy.reporting.localization.Localization(__file__, 644, 23), getitem___70453, int_70451)
            
            # Assigning a type to the variable 'stypy_return_type' (line 644)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 16), 'stypy_return_type', subscript_call_result_70454)
            # SSA branch for the else part of an if statement (line 643)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'None' (line 646)
            None_70455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 646)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 16), 'stypy_return_type', None_70455)
            # SSA join for if statement (line 643)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'get_first(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'get_first' in the type store
            # Getting the type of 'stypy_return_type' (line 642)
            stypy_return_type_70456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_70456)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'get_first'
            return stypy_return_type_70456

        # Assigning a type to the variable 'get_first' (line 642)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'get_first', get_first)
        
        # Assigning a Call to a Name (line 647):
        
        # Assigning a Call to a Name (line 647):
        
        # Call to getattr(...): (line 647)
        # Processing the call arguments (line 647)
        # Getting the type of 'orig_handle' (line 647)
        orig_handle_70458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 28), 'orig_handle', False)
        unicode_70459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 41), 'unicode', u'_original_edgecolor')
        
        # Call to get_edgecolor(...): (line 648)
        # Processing the call keyword arguments (line 648)
        kwargs_70462 = {}
        # Getting the type of 'orig_handle' (line 648)
        orig_handle_70460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 28), 'orig_handle', False)
        # Obtaining the member 'get_edgecolor' of a type (line 648)
        get_edgecolor_70461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 28), orig_handle_70460, 'get_edgecolor')
        # Calling get_edgecolor(args, kwargs) (line 648)
        get_edgecolor_call_result_70463 = invoke(stypy.reporting.localization.Localization(__file__, 648, 28), get_edgecolor_70461, *[], **kwargs_70462)
        
        # Processing the call keyword arguments (line 647)
        kwargs_70464 = {}
        # Getting the type of 'getattr' (line 647)
        getattr_70457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 647)
        getattr_call_result_70465 = invoke(stypy.reporting.localization.Localization(__file__, 647, 20), getattr_70457, *[orig_handle_70458, unicode_70459, get_edgecolor_call_result_70463], **kwargs_70464)
        
        # Assigning a type to the variable 'edgecolor' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'edgecolor', getattr_call_result_70465)
        
        # Call to set_edgecolor(...): (line 649)
        # Processing the call arguments (line 649)
        
        # Call to first_color(...): (line 649)
        # Processing the call arguments (line 649)
        # Getting the type of 'edgecolor' (line 649)
        edgecolor_70469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 48), 'edgecolor', False)
        # Processing the call keyword arguments (line 649)
        kwargs_70470 = {}
        # Getting the type of 'first_color' (line 649)
        first_color_70468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 36), 'first_color', False)
        # Calling first_color(args, kwargs) (line 649)
        first_color_call_result_70471 = invoke(stypy.reporting.localization.Localization(__file__, 649, 36), first_color_70468, *[edgecolor_70469], **kwargs_70470)
        
        # Processing the call keyword arguments (line 649)
        kwargs_70472 = {}
        # Getting the type of 'legend_handle' (line 649)
        legend_handle_70466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'legend_handle', False)
        # Obtaining the member 'set_edgecolor' of a type (line 649)
        set_edgecolor_70467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 8), legend_handle_70466, 'set_edgecolor')
        # Calling set_edgecolor(args, kwargs) (line 649)
        set_edgecolor_call_result_70473 = invoke(stypy.reporting.localization.Localization(__file__, 649, 8), set_edgecolor_70467, *[first_color_call_result_70471], **kwargs_70472)
        
        
        # Assigning a Call to a Name (line 650):
        
        # Assigning a Call to a Name (line 650):
        
        # Call to getattr(...): (line 650)
        # Processing the call arguments (line 650)
        # Getting the type of 'orig_handle' (line 650)
        orig_handle_70475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 28), 'orig_handle', False)
        unicode_70476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 41), 'unicode', u'_original_facecolor')
        
        # Call to get_facecolor(...): (line 651)
        # Processing the call keyword arguments (line 651)
        kwargs_70479 = {}
        # Getting the type of 'orig_handle' (line 651)
        orig_handle_70477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 28), 'orig_handle', False)
        # Obtaining the member 'get_facecolor' of a type (line 651)
        get_facecolor_70478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 28), orig_handle_70477, 'get_facecolor')
        # Calling get_facecolor(args, kwargs) (line 651)
        get_facecolor_call_result_70480 = invoke(stypy.reporting.localization.Localization(__file__, 651, 28), get_facecolor_70478, *[], **kwargs_70479)
        
        # Processing the call keyword arguments (line 650)
        kwargs_70481 = {}
        # Getting the type of 'getattr' (line 650)
        getattr_70474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 650)
        getattr_call_result_70482 = invoke(stypy.reporting.localization.Localization(__file__, 650, 20), getattr_70474, *[orig_handle_70475, unicode_70476, get_facecolor_call_result_70480], **kwargs_70481)
        
        # Assigning a type to the variable 'facecolor' (line 650)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'facecolor', getattr_call_result_70482)
        
        # Call to set_facecolor(...): (line 652)
        # Processing the call arguments (line 652)
        
        # Call to first_color(...): (line 652)
        # Processing the call arguments (line 652)
        # Getting the type of 'facecolor' (line 652)
        facecolor_70486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 48), 'facecolor', False)
        # Processing the call keyword arguments (line 652)
        kwargs_70487 = {}
        # Getting the type of 'first_color' (line 652)
        first_color_70485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 36), 'first_color', False)
        # Calling first_color(args, kwargs) (line 652)
        first_color_call_result_70488 = invoke(stypy.reporting.localization.Localization(__file__, 652, 36), first_color_70485, *[facecolor_70486], **kwargs_70487)
        
        # Processing the call keyword arguments (line 652)
        kwargs_70489 = {}
        # Getting the type of 'legend_handle' (line 652)
        legend_handle_70483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'legend_handle', False)
        # Obtaining the member 'set_facecolor' of a type (line 652)
        set_facecolor_70484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 8), legend_handle_70483, 'set_facecolor')
        # Calling set_facecolor(args, kwargs) (line 652)
        set_facecolor_call_result_70490 = invoke(stypy.reporting.localization.Localization(__file__, 652, 8), set_facecolor_70484, *[first_color_call_result_70488], **kwargs_70489)
        
        
        # Call to set_fill(...): (line 653)
        # Processing the call arguments (line 653)
        
        # Call to get_fill(...): (line 653)
        # Processing the call keyword arguments (line 653)
        kwargs_70495 = {}
        # Getting the type of 'orig_handle' (line 653)
        orig_handle_70493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 31), 'orig_handle', False)
        # Obtaining the member 'get_fill' of a type (line 653)
        get_fill_70494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 31), orig_handle_70493, 'get_fill')
        # Calling get_fill(args, kwargs) (line 653)
        get_fill_call_result_70496 = invoke(stypy.reporting.localization.Localization(__file__, 653, 31), get_fill_70494, *[], **kwargs_70495)
        
        # Processing the call keyword arguments (line 653)
        kwargs_70497 = {}
        # Getting the type of 'legend_handle' (line 653)
        legend_handle_70491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'legend_handle', False)
        # Obtaining the member 'set_fill' of a type (line 653)
        set_fill_70492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 8), legend_handle_70491, 'set_fill')
        # Calling set_fill(args, kwargs) (line 653)
        set_fill_call_result_70498 = invoke(stypy.reporting.localization.Localization(__file__, 653, 8), set_fill_70492, *[get_fill_call_result_70496], **kwargs_70497)
        
        
        # Call to set_hatch(...): (line 654)
        # Processing the call arguments (line 654)
        
        # Call to get_hatch(...): (line 654)
        # Processing the call keyword arguments (line 654)
        kwargs_70503 = {}
        # Getting the type of 'orig_handle' (line 654)
        orig_handle_70501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 32), 'orig_handle', False)
        # Obtaining the member 'get_hatch' of a type (line 654)
        get_hatch_70502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 32), orig_handle_70501, 'get_hatch')
        # Calling get_hatch(args, kwargs) (line 654)
        get_hatch_call_result_70504 = invoke(stypy.reporting.localization.Localization(__file__, 654, 32), get_hatch_70502, *[], **kwargs_70503)
        
        # Processing the call keyword arguments (line 654)
        kwargs_70505 = {}
        # Getting the type of 'legend_handle' (line 654)
        legend_handle_70499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'legend_handle', False)
        # Obtaining the member 'set_hatch' of a type (line 654)
        set_hatch_70500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 8), legend_handle_70499, 'set_hatch')
        # Calling set_hatch(args, kwargs) (line 654)
        set_hatch_call_result_70506 = invoke(stypy.reporting.localization.Localization(__file__, 654, 8), set_hatch_70500, *[get_hatch_call_result_70504], **kwargs_70505)
        
        
        # Call to set_linewidth(...): (line 655)
        # Processing the call arguments (line 655)
        
        # Call to get_first(...): (line 655)
        # Processing the call arguments (line 655)
        
        # Call to get_linewidths(...): (line 655)
        # Processing the call keyword arguments (line 655)
        kwargs_70512 = {}
        # Getting the type of 'orig_handle' (line 655)
        orig_handle_70510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 46), 'orig_handle', False)
        # Obtaining the member 'get_linewidths' of a type (line 655)
        get_linewidths_70511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 46), orig_handle_70510, 'get_linewidths')
        # Calling get_linewidths(args, kwargs) (line 655)
        get_linewidths_call_result_70513 = invoke(stypy.reporting.localization.Localization(__file__, 655, 46), get_linewidths_70511, *[], **kwargs_70512)
        
        # Processing the call keyword arguments (line 655)
        kwargs_70514 = {}
        # Getting the type of 'get_first' (line 655)
        get_first_70509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 36), 'get_first', False)
        # Calling get_first(args, kwargs) (line 655)
        get_first_call_result_70515 = invoke(stypy.reporting.localization.Localization(__file__, 655, 36), get_first_70509, *[get_linewidths_call_result_70513], **kwargs_70514)
        
        # Processing the call keyword arguments (line 655)
        kwargs_70516 = {}
        # Getting the type of 'legend_handle' (line 655)
        legend_handle_70507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 8), 'legend_handle', False)
        # Obtaining the member 'set_linewidth' of a type (line 655)
        set_linewidth_70508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 8), legend_handle_70507, 'set_linewidth')
        # Calling set_linewidth(args, kwargs) (line 655)
        set_linewidth_call_result_70517 = invoke(stypy.reporting.localization.Localization(__file__, 655, 8), set_linewidth_70508, *[get_first_call_result_70515], **kwargs_70516)
        
        
        # Call to set_linestyle(...): (line 656)
        # Processing the call arguments (line 656)
        
        # Call to get_first(...): (line 656)
        # Processing the call arguments (line 656)
        
        # Call to get_linestyles(...): (line 656)
        # Processing the call keyword arguments (line 656)
        kwargs_70523 = {}
        # Getting the type of 'orig_handle' (line 656)
        orig_handle_70521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 46), 'orig_handle', False)
        # Obtaining the member 'get_linestyles' of a type (line 656)
        get_linestyles_70522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 46), orig_handle_70521, 'get_linestyles')
        # Calling get_linestyles(args, kwargs) (line 656)
        get_linestyles_call_result_70524 = invoke(stypy.reporting.localization.Localization(__file__, 656, 46), get_linestyles_70522, *[], **kwargs_70523)
        
        # Processing the call keyword arguments (line 656)
        kwargs_70525 = {}
        # Getting the type of 'get_first' (line 656)
        get_first_70520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 36), 'get_first', False)
        # Calling get_first(args, kwargs) (line 656)
        get_first_call_result_70526 = invoke(stypy.reporting.localization.Localization(__file__, 656, 36), get_first_70520, *[get_linestyles_call_result_70524], **kwargs_70525)
        
        # Processing the call keyword arguments (line 656)
        kwargs_70527 = {}
        # Getting the type of 'legend_handle' (line 656)
        legend_handle_70518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'legend_handle', False)
        # Obtaining the member 'set_linestyle' of a type (line 656)
        set_linestyle_70519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 8), legend_handle_70518, 'set_linestyle')
        # Calling set_linestyle(args, kwargs) (line 656)
        set_linestyle_call_result_70528 = invoke(stypy.reporting.localization.Localization(__file__, 656, 8), set_linestyle_70519, *[get_first_call_result_70526], **kwargs_70527)
        
        
        # Call to set_transform(...): (line 657)
        # Processing the call arguments (line 657)
        
        # Call to get_first(...): (line 657)
        # Processing the call arguments (line 657)
        
        # Call to get_transforms(...): (line 657)
        # Processing the call keyword arguments (line 657)
        kwargs_70534 = {}
        # Getting the type of 'orig_handle' (line 657)
        orig_handle_70532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 46), 'orig_handle', False)
        # Obtaining the member 'get_transforms' of a type (line 657)
        get_transforms_70533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 46), orig_handle_70532, 'get_transforms')
        # Calling get_transforms(args, kwargs) (line 657)
        get_transforms_call_result_70535 = invoke(stypy.reporting.localization.Localization(__file__, 657, 46), get_transforms_70533, *[], **kwargs_70534)
        
        # Processing the call keyword arguments (line 657)
        kwargs_70536 = {}
        # Getting the type of 'get_first' (line 657)
        get_first_70531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 36), 'get_first', False)
        # Calling get_first(args, kwargs) (line 657)
        get_first_call_result_70537 = invoke(stypy.reporting.localization.Localization(__file__, 657, 36), get_first_70531, *[get_transforms_call_result_70535], **kwargs_70536)
        
        # Processing the call keyword arguments (line 657)
        kwargs_70538 = {}
        # Getting the type of 'legend_handle' (line 657)
        legend_handle_70529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'legend_handle', False)
        # Obtaining the member 'set_transform' of a type (line 657)
        set_transform_70530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 8), legend_handle_70529, 'set_transform')
        # Calling set_transform(args, kwargs) (line 657)
        set_transform_call_result_70539 = invoke(stypy.reporting.localization.Localization(__file__, 657, 8), set_transform_70530, *[get_first_call_result_70537], **kwargs_70538)
        
        
        # Call to set_figure(...): (line 658)
        # Processing the call arguments (line 658)
        
        # Call to get_figure(...): (line 658)
        # Processing the call keyword arguments (line 658)
        kwargs_70544 = {}
        # Getting the type of 'orig_handle' (line 658)
        orig_handle_70542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 33), 'orig_handle', False)
        # Obtaining the member 'get_figure' of a type (line 658)
        get_figure_70543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 33), orig_handle_70542, 'get_figure')
        # Calling get_figure(args, kwargs) (line 658)
        get_figure_call_result_70545 = invoke(stypy.reporting.localization.Localization(__file__, 658, 33), get_figure_70543, *[], **kwargs_70544)
        
        # Processing the call keyword arguments (line 658)
        kwargs_70546 = {}
        # Getting the type of 'legend_handle' (line 658)
        legend_handle_70540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'legend_handle', False)
        # Obtaining the member 'set_figure' of a type (line 658)
        set_figure_70541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 8), legend_handle_70540, 'set_figure')
        # Calling set_figure(args, kwargs) (line 658)
        set_figure_call_result_70547 = invoke(stypy.reporting.localization.Localization(__file__, 658, 8), set_figure_70541, *[get_figure_call_result_70545], **kwargs_70546)
        
        
        # Call to set_alpha(...): (line 659)
        # Processing the call arguments (line 659)
        
        # Call to get_alpha(...): (line 659)
        # Processing the call keyword arguments (line 659)
        kwargs_70552 = {}
        # Getting the type of 'orig_handle' (line 659)
        orig_handle_70550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 32), 'orig_handle', False)
        # Obtaining the member 'get_alpha' of a type (line 659)
        get_alpha_70551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 32), orig_handle_70550, 'get_alpha')
        # Calling get_alpha(args, kwargs) (line 659)
        get_alpha_call_result_70553 = invoke(stypy.reporting.localization.Localization(__file__, 659, 32), get_alpha_70551, *[], **kwargs_70552)
        
        # Processing the call keyword arguments (line 659)
        kwargs_70554 = {}
        # Getting the type of 'legend_handle' (line 659)
        legend_handle_70548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 8), 'legend_handle', False)
        # Obtaining the member 'set_alpha' of a type (line 659)
        set_alpha_70549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 8), legend_handle_70548, 'set_alpha')
        # Calling set_alpha(args, kwargs) (line 659)
        set_alpha_call_result_70555 = invoke(stypy.reporting.localization.Localization(__file__, 659, 8), set_alpha_70549, *[get_alpha_call_result_70553], **kwargs_70554)
        
        
        # ################# End of '_update_prop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update_prop' in the type store
        # Getting the type of 'stypy_return_type' (line 633)
        stypy_return_type_70556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_70556)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update_prop'
        return stypy_return_type_70556


    @norecursion
    def create_artists(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_artists'
        module_type_store = module_type_store.open_function_context('create_artists', 661, 4, False)
        # Assigning a type to the variable 'self' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HandlerPolyCollection.create_artists.__dict__.__setitem__('stypy_localization', localization)
        HandlerPolyCollection.create_artists.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HandlerPolyCollection.create_artists.__dict__.__setitem__('stypy_type_store', module_type_store)
        HandlerPolyCollection.create_artists.__dict__.__setitem__('stypy_function_name', 'HandlerPolyCollection.create_artists')
        HandlerPolyCollection.create_artists.__dict__.__setitem__('stypy_param_names_list', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'])
        HandlerPolyCollection.create_artists.__dict__.__setitem__('stypy_varargs_param_name', None)
        HandlerPolyCollection.create_artists.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HandlerPolyCollection.create_artists.__dict__.__setitem__('stypy_call_defaults', defaults)
        HandlerPolyCollection.create_artists.__dict__.__setitem__('stypy_call_varargs', varargs)
        HandlerPolyCollection.create_artists.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HandlerPolyCollection.create_artists.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerPolyCollection.create_artists', ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_artists', localization, ['legend', 'orig_handle', 'xdescent', 'ydescent', 'width', 'height', 'fontsize', 'trans'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_artists(...)' code ##################

        
        # Assigning a Call to a Name (line 663):
        
        # Assigning a Call to a Name (line 663):
        
        # Call to Rectangle(...): (line 663)
        # Processing the call keyword arguments (line 663)
        
        # Obtaining an instance of the builtin type 'tuple' (line 663)
        tuple_70558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 663)
        # Adding element type (line 663)
        
        # Getting the type of 'xdescent' (line 663)
        xdescent_70559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 27), 'xdescent', False)
        # Applying the 'usub' unary operator (line 663)
        result___neg___70560 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 26), 'usub', xdescent_70559)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 26), tuple_70558, result___neg___70560)
        # Adding element type (line 663)
        
        # Getting the type of 'ydescent' (line 663)
        ydescent_70561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 38), 'ydescent', False)
        # Applying the 'usub' unary operator (line 663)
        result___neg___70562 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 37), 'usub', ydescent_70561)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 26), tuple_70558, result___neg___70562)
        
        keyword_70563 = tuple_70558
        # Getting the type of 'width' (line 664)
        width_70564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 28), 'width', False)
        keyword_70565 = width_70564
        # Getting the type of 'height' (line 664)
        height_70566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 42), 'height', False)
        keyword_70567 = height_70566
        kwargs_70568 = {'width': keyword_70565, 'xy': keyword_70563, 'height': keyword_70567}
        # Getting the type of 'Rectangle' (line 663)
        Rectangle_70557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), 'Rectangle', False)
        # Calling Rectangle(args, kwargs) (line 663)
        Rectangle_call_result_70569 = invoke(stypy.reporting.localization.Localization(__file__, 663, 12), Rectangle_70557, *[], **kwargs_70568)
        
        # Assigning a type to the variable 'p' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'p', Rectangle_call_result_70569)
        
        # Call to update_prop(...): (line 665)
        # Processing the call arguments (line 665)
        # Getting the type of 'p' (line 665)
        p_70572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 25), 'p', False)
        # Getting the type of 'orig_handle' (line 665)
        orig_handle_70573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 28), 'orig_handle', False)
        # Getting the type of 'legend' (line 665)
        legend_70574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 41), 'legend', False)
        # Processing the call keyword arguments (line 665)
        kwargs_70575 = {}
        # Getting the type of 'self' (line 665)
        self_70570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 8), 'self', False)
        # Obtaining the member 'update_prop' of a type (line 665)
        update_prop_70571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 8), self_70570, 'update_prop')
        # Calling update_prop(args, kwargs) (line 665)
        update_prop_call_result_70576 = invoke(stypy.reporting.localization.Localization(__file__, 665, 8), update_prop_70571, *[p_70572, orig_handle_70573, legend_70574], **kwargs_70575)
        
        
        # Call to set_transform(...): (line 666)
        # Processing the call arguments (line 666)
        # Getting the type of 'trans' (line 666)
        trans_70579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 24), 'trans', False)
        # Processing the call keyword arguments (line 666)
        kwargs_70580 = {}
        # Getting the type of 'p' (line 666)
        p_70577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'p', False)
        # Obtaining the member 'set_transform' of a type (line 666)
        set_transform_70578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 8), p_70577, 'set_transform')
        # Calling set_transform(args, kwargs) (line 666)
        set_transform_call_result_70581 = invoke(stypy.reporting.localization.Localization(__file__, 666, 8), set_transform_70578, *[trans_70579], **kwargs_70580)
        
        
        # Obtaining an instance of the builtin type 'list' (line 667)
        list_70582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 667)
        # Adding element type (line 667)
        # Getting the type of 'p' (line 667)
        p_70583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 15), list_70582, p_70583)
        
        # Assigning a type to the variable 'stypy_return_type' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'stypy_return_type', list_70582)
        
        # ################# End of 'create_artists(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_artists' in the type store
        # Getting the type of 'stypy_return_type' (line 661)
        stypy_return_type_70584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_70584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_artists'
        return stypy_return_type_70584


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 629, 0, False)
        # Assigning a type to the variable 'self' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HandlerPolyCollection.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'HandlerPolyCollection' (line 629)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 0), 'HandlerPolyCollection', HandlerPolyCollection)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
