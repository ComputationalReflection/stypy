
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Place a table below the x-axis at location loc.
3: 
4: The table consists of a grid of cells.
5: 
6: The grid need not be rectangular and can have holes.
7: 
8: Cells are added by specifying their row and column.
9: 
10: For the purposes of positioning the cell at (0, 0) is
11: assumed to be at the top left and the cell at (max_row, max_col)
12: is assumed to be at bottom right.
13: 
14: You can add additional cells outside this range to have convenient
15: ways of positioning more interesting grids.
16: 
17: Author    : John Gill <jng@europe.renre.com>
18: Copyright : 2004 John Gill and John Hunter
19: License   : matplotlib license
20: 
21: '''
22: from __future__ import (absolute_import, division, print_function,
23:                         unicode_literals)
24: 
25: import six
26: from six.moves import xrange
27: 
28: import warnings
29: 
30: from . import artist
31: from .artist import Artist, allow_rasterization
32: from .patches import Rectangle
33: from matplotlib import docstring
34: from .text import Text
35: from .transforms import Bbox
36: from matplotlib.path import Path
37: 
38: 
39: class Cell(Rectangle):
40:     '''
41:     A cell is a Rectangle with some associated text.
42: 
43:     '''
44:     PAD = 0.1  # padding between text and rectangle
45: 
46:     def __init__(self, xy, width, height,
47:                  edgecolor='k', facecolor='w',
48:                  fill=True,
49:                  text='',
50:                  loc=None,
51:                  fontproperties=None
52:                  ):
53: 
54:         # Call base
55:         Rectangle.__init__(self, xy, width=width, height=height,
56:                            edgecolor=edgecolor, facecolor=facecolor)
57:         self.set_clip_on(False)
58: 
59:         # Create text object
60:         if loc is None:
61:             loc = 'right'
62:         self._loc = loc
63:         self._text = Text(x=xy[0], y=xy[1], text=text,
64:                           fontproperties=fontproperties)
65:         self._text.set_clip_on(False)
66: 
67:     def set_transform(self, trans):
68:         Rectangle.set_transform(self, trans)
69:         # the text does not get the transform!
70:         self.stale = True
71: 
72:     def set_figure(self, fig):
73:         Rectangle.set_figure(self, fig)
74:         self._text.set_figure(fig)
75: 
76:     def get_text(self):
77:         'Return the cell Text intance'
78:         return self._text
79: 
80:     def set_fontsize(self, size):
81:         self._text.set_fontsize(size)
82:         self.stale = True
83: 
84:     def get_fontsize(self):
85:         'Return the cell fontsize'
86:         return self._text.get_fontsize()
87: 
88:     def auto_set_font_size(self, renderer):
89:         ''' Shrink font size until text fits. '''
90:         fontsize = self.get_fontsize()
91:         required = self.get_required_width(renderer)
92:         while fontsize > 1 and required > self.get_width():
93:             fontsize -= 1
94:             self.set_fontsize(fontsize)
95:             required = self.get_required_width(renderer)
96: 
97:         return fontsize
98: 
99:     @allow_rasterization
100:     def draw(self, renderer):
101:         if not self.get_visible():
102:             return
103:         # draw the rectangle
104:         Rectangle.draw(self, renderer)
105: 
106:         # position the text
107:         self._set_text_position(renderer)
108:         self._text.draw(renderer)
109:         self.stale = False
110: 
111:     def _set_text_position(self, renderer):
112:         ''' Set text up so it draws in the right place.
113: 
114:         Currently support 'left', 'center' and 'right'
115:         '''
116:         bbox = self.get_window_extent(renderer)
117:         l, b, w, h = bbox.bounds
118: 
119:         # draw in center vertically
120:         self._text.set_verticalalignment('center')
121:         y = b + (h / 2.0)
122: 
123:         # now position horizontally
124:         if self._loc == 'center':
125:             self._text.set_horizontalalignment('center')
126:             x = l + (w / 2.0)
127:         elif self._loc == 'left':
128:             self._text.set_horizontalalignment('left')
129:             x = l + (w * self.PAD)
130:         else:
131:             self._text.set_horizontalalignment('right')
132:             x = l + (w * (1.0 - self.PAD))
133: 
134:         self._text.set_position((x, y))
135: 
136:     def get_text_bounds(self, renderer):
137:         ''' Get text bounds in axes co-ordinates. '''
138:         bbox = self._text.get_window_extent(renderer)
139:         bboxa = bbox.inverse_transformed(self.get_data_transform())
140:         return bboxa.bounds
141: 
142:     def get_required_width(self, renderer):
143:         ''' Get width required for this cell. '''
144:         l, b, w, h = self.get_text_bounds(renderer)
145:         return w * (1.0 + (2.0 * self.PAD))
146: 
147:     def set_text_props(self, **kwargs):
148:         'update the text properties with kwargs'
149:         self._text.update(kwargs)
150:         self.stale = True
151: 
152: 
153: class CustomCell(Cell):
154:     '''
155:     A subclass of Cell where the sides may be visibly toggled.
156: 
157:     '''
158: 
159:     _edges = 'BRTL'
160:     _edge_aliases = {'open':         '',
161:                      'closed':       _edges,  # default
162:                      'horizontal':   'BT',
163:                      'vertical':     'RL'
164:                      }
165: 
166:     def __init__(self, *args, **kwargs):
167:         visible_edges = kwargs.pop('visible_edges')
168:         Cell.__init__(self, *args, **kwargs)
169:         self.visible_edges = visible_edges
170: 
171:     @property
172:     def visible_edges(self):
173:         return self._visible_edges
174: 
175:     @visible_edges.setter
176:     def visible_edges(self, value):
177:         if value is None:
178:             self._visible_edges = self._edges
179:         elif value in self._edge_aliases:
180:             self._visible_edges = self._edge_aliases[value]
181:         else:
182:             for edge in value:
183:                 if edge not in self._edges:
184:                     msg = ('Invalid edge param {0}, must only be one of'
185:                            ' {1} or string of {2}.').format(
186:                                    value,
187:                                    ", ".join(self._edge_aliases),
188:                                    ", ".join(self._edges),
189:                                    )
190:                     raise ValueError(msg)
191:             self._visible_edges = value
192:         self.stale = True
193: 
194:     def get_path(self):
195:         'Return a path where the edges specificed by _visible_edges are drawn'
196: 
197:         codes = [Path.MOVETO]
198: 
199:         for edge in self._edges:
200:             if edge in self._visible_edges:
201:                 codes.append(Path.LINETO)
202:             else:
203:                 codes.append(Path.MOVETO)
204: 
205:         if Path.MOVETO not in codes[1:]:  # All sides are visible
206:             codes[-1] = Path.CLOSEPOLY
207: 
208:         return Path(
209:             [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
210:             codes,
211:             readonly=True
212:             )
213: 
214: 
215: class Table(Artist):
216:     '''
217:     Create a table of cells.
218: 
219:     Table can have (optional) row and column headers.
220: 
221:     Each entry in the table can be either text or patches.
222: 
223:     Column widths and row heights for the table can be specified.
224: 
225:     Return value is a sequence of text, line and patch instances that make
226:     up the table
227:     '''
228:     codes = {'best': 0,
229:              'upper right':  1,  # default
230:              'upper left':   2,
231:              'lower left':   3,
232:              'lower right':  4,
233:              'center left':  5,
234:              'center right': 6,
235:              'lower center': 7,
236:              'upper center': 8,
237:              'center':       9,
238:              'top right':    10,
239:              'top left':     11,
240:              'bottom left':  12,
241:              'bottom right': 13,
242:              'right':        14,
243:              'left':         15,
244:              'top':          16,
245:              'bottom':       17,
246:              }
247: 
248:     FONTSIZE = 10
249:     AXESPAD = 0.02    # the border between the axes and table edge
250: 
251:     def __init__(self, ax, loc=None, bbox=None, **kwargs):
252: 
253:         Artist.__init__(self)
254: 
255:         if isinstance(loc, six.string_types) and loc not in self.codes:
256:             warnings.warn('Unrecognized location %s. Falling back on '
257:                           'bottom; valid locations are\n%s\t' %
258:                           (loc, '\n\t'.join(self.codes)))
259:             loc = 'bottom'
260:         if isinstance(loc, six.string_types):
261:             loc = self.codes.get(loc, 1)
262:         self.set_figure(ax.figure)
263:         self._axes = ax
264:         self._loc = loc
265:         self._bbox = bbox
266: 
267:         # use axes coords
268:         self.set_transform(ax.transAxes)
269: 
270:         self._texts = []
271:         self._cells = {}
272:         self._edges = None
273:         self._autoRows = []
274:         self._autoColumns = []
275:         self._autoFontsize = True
276:         self.update(kwargs)
277: 
278:         self.set_clip_on(False)
279: 
280:     def add_cell(self, row, col, *args, **kwargs):
281:         ''' Add a cell to the table. '''
282:         xy = (0, 0)
283: 
284:         cell = CustomCell(xy, visible_edges=self.edges, *args, **kwargs)
285:         cell.set_figure(self.figure)
286:         cell.set_transform(self.get_transform())
287: 
288:         cell.set_clip_on(False)
289:         self._cells[row, col] = cell
290:         self.stale = True
291: 
292:     @property
293:     def edges(self):
294:         return self._edges
295: 
296:     @edges.setter
297:     def edges(self, value):
298:         self._edges = value
299:         self.stale = True
300: 
301:     def _approx_text_height(self):
302:         return (self.FONTSIZE / 72.0 * self.figure.dpi /
303:                 self._axes.bbox.height * 1.2)
304: 
305:     @allow_rasterization
306:     def draw(self, renderer):
307:         # Need a renderer to do hit tests on mouseevent; assume the last one
308:         # will do
309:         if renderer is None:
310:             renderer = self.figure._cachedRenderer
311:         if renderer is None:
312:             raise RuntimeError('No renderer defined')
313: 
314:         if not self.get_visible():
315:             return
316:         renderer.open_group('table')
317:         self._update_positions(renderer)
318: 
319:         for key in sorted(self._cells):
320:             self._cells[key].draw(renderer)
321: 
322:         renderer.close_group('table')
323:         self.stale = False
324: 
325:     def _get_grid_bbox(self, renderer):
326:         '''Get a bbox, in axes co-ordinates for the cells.
327: 
328:         Only include those in the range (0,0) to (maxRow, maxCol)'''
329:         boxes = [cell.get_window_extent(renderer)
330:                  for (row, col), cell in six.iteritems(self._cells)
331:                  if row >= 0 and col >= 0]
332:         bbox = Bbox.union(boxes)
333:         return bbox.inverse_transformed(self.get_transform())
334: 
335:     def contains(self, mouseevent):
336:         '''Test whether the mouse event occurred in the table.
337: 
338:         Returns T/F, {}
339:         '''
340:         if callable(self._contains):
341:             return self._contains(self, mouseevent)
342: 
343:         # TODO: Return index of the cell containing the cursor so that the user
344:         # doesn't have to bind to each one individually.
345:         renderer = self.figure._cachedRenderer
346:         if renderer is not None:
347:             boxes = [cell.get_window_extent(renderer)
348:                      for (row, col), cell in six.iteritems(self._cells)
349:                      if row >= 0 and col >= 0]
350:             bbox = Bbox.union(boxes)
351:             return bbox.contains(mouseevent.x, mouseevent.y), {}
352:         else:
353:             return False, {}
354: 
355:     def get_children(self):
356:         'Return the Artists contained by the table'
357:         return list(six.itervalues(self._cells))
358:     get_child_artists = get_children  # backward compatibility
359: 
360:     def get_window_extent(self, renderer):
361:         'Return the bounding box of the table in window coords'
362:         boxes = [cell.get_window_extent(renderer)
363:                  for cell in six.itervalues(self._cells)]
364:         return Bbox.union(boxes)
365: 
366:     def _do_cell_alignment(self):
367:         ''' Calculate row heights and column widths.
368: 
369:         Position cells accordingly.
370:         '''
371:         # Calculate row/column widths
372:         widths = {}
373:         heights = {}
374:         for (row, col), cell in six.iteritems(self._cells):
375:             height = heights.setdefault(row, 0.0)
376:             heights[row] = max(height, cell.get_height())
377:             width = widths.setdefault(col, 0.0)
378:             widths[col] = max(width, cell.get_width())
379: 
380:         # work out left position for each column
381:         xpos = 0
382:         lefts = {}
383:         for col in sorted(widths):
384:             lefts[col] = xpos
385:             xpos += widths[col]
386: 
387:         ypos = 0
388:         bottoms = {}
389:         for row in sorted(heights, reverse=True):
390:             bottoms[row] = ypos
391:             ypos += heights[row]
392: 
393:         # set cell positions
394:         for (row, col), cell in six.iteritems(self._cells):
395:             cell.set_x(lefts[col])
396:             cell.set_y(bottoms[row])
397: 
398:     def auto_set_column_width(self, col):
399:         ''' Given column indexs in either List, Tuple or int. Will be able to
400:         automatically set the columns into optimal sizes.
401: 
402:         Here is the example of the input, which triger automatic adjustment on
403:         columns to optimal size by given index numbers.
404:         -1: the row labling
405:         0: the 1st column
406:         1: the 2nd column
407: 
408:         Args:
409:             col(List): list of indexs
410:             >>>table.auto_set_column_width([-1,0,1])
411: 
412:             col(Tuple): tuple of indexs
413:             >>>table.auto_set_column_width((-1,0,1))
414: 
415:             col(int): index integer
416:             >>>table.auto_set_column_width(-1)
417:             >>>table.auto_set_column_width(0)
418:             >>>table.auto_set_column_width(1)
419:         '''
420:         # check for col possibility on iteration
421:         try:
422:             iter(col)
423:         except (TypeError, AttributeError):
424:             self._autoColumns.append(col)
425:         else:
426:             for cell in col:
427:                 self._autoColumns.append(cell)
428: 
429:         self.stale = True
430: 
431:     def _auto_set_column_width(self, col, renderer):
432:         ''' Automagically set width for column.
433:         '''
434:         cells = [key for key in self._cells if key[1] == col]
435: 
436:         # find max width
437:         width = 0
438:         for cell in cells:
439:             c = self._cells[cell]
440:             width = max(c.get_required_width(renderer), width)
441: 
442:         # Now set the widths
443:         for cell in cells:
444:             self._cells[cell].set_width(width)
445: 
446:     def auto_set_font_size(self, value=True):
447:         ''' Automatically set font size. '''
448:         self._autoFontsize = value
449:         self.stale = True
450: 
451:     def _auto_set_font_size(self, renderer):
452: 
453:         if len(self._cells) == 0:
454:             return
455:         fontsize = list(six.itervalues(self._cells))[0].get_fontsize()
456:         cells = []
457:         for key, cell in six.iteritems(self._cells):
458:             # ignore auto-sized columns
459:             if key[1] in self._autoColumns:
460:                 continue
461:             size = cell.auto_set_font_size(renderer)
462:             fontsize = min(fontsize, size)
463:             cells.append(cell)
464: 
465:         # now set all fontsizes equal
466:         for cell in six.itervalues(self._cells):
467:             cell.set_fontsize(fontsize)
468: 
469:     def scale(self, xscale, yscale):
470:         ''' Scale column widths by xscale and row heights by yscale. '''
471:         for c in six.itervalues(self._cells):
472:             c.set_width(c.get_width() * xscale)
473:             c.set_height(c.get_height() * yscale)
474: 
475:     def set_fontsize(self, size):
476:         '''
477:         Set the fontsize of the cell text
478: 
479:         ACCEPTS: a float in points
480:         '''
481: 
482:         for cell in six.itervalues(self._cells):
483:             cell.set_fontsize(size)
484:         self.stale = True
485: 
486:     def _offset(self, ox, oy):
487:         'Move all the artists by ox,oy (axes coords)'
488: 
489:         for c in six.itervalues(self._cells):
490:             x, y = c.get_x(), c.get_y()
491:             c.set_x(x + ox)
492:             c.set_y(y + oy)
493: 
494:     def _update_positions(self, renderer):
495:         # called from renderer to allow more precise estimates of
496:         # widths and heights with get_window_extent
497: 
498:         # Do any auto width setting
499:         for col in self._autoColumns:
500:             self._auto_set_column_width(col, renderer)
501: 
502:         if self._autoFontsize:
503:             self._auto_set_font_size(renderer)
504: 
505:         # Align all the cells
506:         self._do_cell_alignment()
507: 
508:         bbox = self._get_grid_bbox(renderer)
509:         l, b, w, h = bbox.bounds
510: 
511:         if self._bbox is not None:
512:             # Position according to bbox
513:             rl, rb, rw, rh = self._bbox
514:             self.scale(rw / w, rh / h)
515:             ox = rl - l
516:             oy = rb - b
517:             self._do_cell_alignment()
518:         else:
519:             # Position using loc
520:             (BEST, UR, UL, LL, LR, CL, CR, LC, UC, C,
521:              TR, TL, BL, BR, R, L, T, B) = xrange(len(self.codes))
522:             # defaults for center
523:             ox = (0.5 - w / 2) - l
524:             oy = (0.5 - h / 2) - b
525:             if self._loc in (UL, LL, CL):   # left
526:                 ox = self.AXESPAD - l
527:             if self._loc in (BEST, UR, LR, R, CR):  # right
528:                 ox = 1 - (l + w + self.AXESPAD)
529:             if self._loc in (BEST, UR, UL, UC):     # upper
530:                 oy = 1 - (b + h + self.AXESPAD)
531:             if self._loc in (LL, LR, LC):           # lower
532:                 oy = self.AXESPAD - b
533:             if self._loc in (LC, UC, C):            # center x
534:                 ox = (0.5 - w / 2) - l
535:             if self._loc in (CL, CR, C):            # center y
536:                 oy = (0.5 - h / 2) - b
537: 
538:             if self._loc in (TL, BL, L):            # out left
539:                 ox = - (l + w)
540:             if self._loc in (TR, BR, R):            # out right
541:                 ox = 1.0 - l
542:             if self._loc in (TR, TL, T):            # out top
543:                 oy = 1.0 - b
544:             if self._loc in (BL, BR, B):           # out bottom
545:                 oy = - (b + h)
546: 
547:         self._offset(ox, oy)
548: 
549:     def get_celld(self):
550:         'return a dict of cells in the table'
551:         return self._cells
552: 
553: 
554: def table(ax,
555:           cellText=None, cellColours=None,
556:           cellLoc='right', colWidths=None,
557:           rowLabels=None, rowColours=None, rowLoc='left',
558:           colLabels=None, colColours=None, colLoc='center',
559:           loc='bottom', bbox=None, edges='closed',
560:           **kwargs):
561:     '''
562:     TABLE(cellText=None, cellColours=None,
563:           cellLoc='right', colWidths=None,
564:           rowLabels=None, rowColours=None, rowLoc='left',
565:           colLabels=None, colColours=None, colLoc='center',
566:           loc='bottom', bbox=None, edges='closed')
567: 
568:     Factory function to generate a Table instance.
569: 
570:     Thanks to John Gill for providing the class and table.
571:     '''
572: 
573:     if cellColours is None and cellText is None:
574:         raise ValueError('At least one argument from "cellColours" or '
575:                          '"cellText" must be provided to create a table.')
576: 
577:     # Check we have some cellText
578:     if cellText is None:
579:         # assume just colours are needed
580:         rows = len(cellColours)
581:         cols = len(cellColours[0])
582:         cellText = [[''] * cols] * rows
583: 
584:     rows = len(cellText)
585:     cols = len(cellText[0])
586:     for row in cellText:
587:         if len(row) != cols:
588:             msg = "Each row in 'cellText' must have {0} columns"
589:             raise ValueError(msg.format(cols))
590: 
591:     if cellColours is not None:
592:         if len(cellColours) != rows:
593:             raise ValueError("'cellColours' must have {0} rows".format(rows))
594:         for row in cellColours:
595:             if len(row) != cols:
596:                 msg = "Each row in 'cellColours' must have {0} columns"
597:                 raise ValueError(msg.format(cols))
598:     else:
599:         cellColours = ['w' * cols] * rows
600: 
601:     # Set colwidths if not given
602:     if colWidths is None:
603:         colWidths = [1.0 / cols] * cols
604: 
605:     # Fill in missing information for column
606:     # and row labels
607:     rowLabelWidth = 0
608:     if rowLabels is None:
609:         if rowColours is not None:
610:             rowLabels = [''] * rows
611:             rowLabelWidth = colWidths[0]
612:     elif rowColours is None:
613:         rowColours = 'w' * rows
614: 
615:     if rowLabels is not None:
616:         if len(rowLabels) != rows:
617:             raise ValueError("'rowLabels' must be of length {0}".format(rows))
618: 
619:     # If we have column labels, need to shift
620:     # the text and colour arrays down 1 row
621:     offset = 1
622:     if colLabels is None:
623:         if colColours is not None:
624:             colLabels = [''] * cols
625:         else:
626:             offset = 0
627:     elif colColours is None:
628:         colColours = 'w' * cols
629: 
630:     # Set up cell colours if not given
631:     if cellColours is None:
632:         cellColours = ['w' * cols] * rows
633: 
634:     # Now create the table
635:     table = Table(ax, loc, bbox, **kwargs)
636:     table.edges = edges
637:     height = table._approx_text_height()
638: 
639:     # Add the cells
640:     for row in xrange(rows):
641:         for col in xrange(cols):
642:             table.add_cell(row + offset, col,
643:                            width=colWidths[col], height=height,
644:                            text=cellText[row][col],
645:                            facecolor=cellColours[row][col],
646:                            loc=cellLoc)
647:     # Do column labels
648:     if colLabels is not None:
649:         for col in xrange(cols):
650:             table.add_cell(0, col,
651:                            width=colWidths[col], height=height,
652:                            text=colLabels[col], facecolor=colColours[col],
653:                            loc=colLoc)
654: 
655:     # Do row labels
656:     if rowLabels is not None:
657:         for row in xrange(rows):
658:             table.add_cell(row + offset, -1,
659:                            width=rowLabelWidth or 1e-15, height=height,
660:                            text=rowLabels[row], facecolor=rowColours[row],
661:                            loc=rowLoc)
662:         if rowLabelWidth == 0:
663:             table.auto_set_column_width(-1)
664: 
665:     ax.add_table(table)
666:     return table
667: 
668: 
669: docstring.interpd.update(Table=artist.kwdoc(Table))
670: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_135281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, (-1)), 'unicode', u'\nPlace a table below the x-axis at location loc.\n\nThe table consists of a grid of cells.\n\nThe grid need not be rectangular and can have holes.\n\nCells are added by specifying their row and column.\n\nFor the purposes of positioning the cell at (0, 0) is\nassumed to be at the top left and the cell at (max_row, max_col)\nis assumed to be at bottom right.\n\nYou can add additional cells outside this range to have convenient\nways of positioning more interesting grids.\n\nAuthor    : John Gill <jng@europe.renre.com>\nCopyright : 2004 John Gill and John Hunter\nLicense   : matplotlib license\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'import six' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_135282 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'six')

if (type(import_135282) is not StypyTypeError):

    if (import_135282 != 'pyd_module'):
        __import__(import_135282)
        sys_modules_135283 = sys.modules[import_135282]
        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'six', sys_modules_135283.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'six', import_135282)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from six.moves import xrange' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_135284 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'six.moves')

if (type(import_135284) is not StypyTypeError):

    if (import_135284 != 'pyd_module'):
        __import__(import_135284)
        sys_modules_135285 = sys.modules[import_135284]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'six.moves', sys_modules_135285.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_135285, sys_modules_135285.module_type_store, module_type_store)
    else:
        from six.moves import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'six.moves', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'six.moves' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'six.moves', import_135284)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import warnings' statement (line 28)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from matplotlib import artist' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_135286 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib')

if (type(import_135286) is not StypyTypeError):

    if (import_135286 != 'pyd_module'):
        __import__(import_135286)
        sys_modules_135287 = sys.modules[import_135286]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib', sys_modules_135287.module_type_store, module_type_store, ['artist'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_135287, sys_modules_135287.module_type_store, module_type_store)
    else:
        from matplotlib import artist

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib', None, module_type_store, ['artist'], [artist])

else:
    # Assigning a type to the variable 'matplotlib' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib', import_135286)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from matplotlib.artist import Artist, allow_rasterization' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_135288 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.artist')

if (type(import_135288) is not StypyTypeError):

    if (import_135288 != 'pyd_module'):
        __import__(import_135288)
        sys_modules_135289 = sys.modules[import_135288]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.artist', sys_modules_135289.module_type_store, module_type_store, ['Artist', 'allow_rasterization'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_135289, sys_modules_135289.module_type_store, module_type_store)
    else:
        from matplotlib.artist import Artist, allow_rasterization

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.artist', None, module_type_store, ['Artist', 'allow_rasterization'], [Artist, allow_rasterization])

else:
    # Assigning a type to the variable 'matplotlib.artist' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.artist', import_135288)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from matplotlib.patches import Rectangle' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_135290 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.patches')

if (type(import_135290) is not StypyTypeError):

    if (import_135290 != 'pyd_module'):
        __import__(import_135290)
        sys_modules_135291 = sys.modules[import_135290]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.patches', sys_modules_135291.module_type_store, module_type_store, ['Rectangle'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_135291, sys_modules_135291.module_type_store, module_type_store)
    else:
        from matplotlib.patches import Rectangle

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.patches', None, module_type_store, ['Rectangle'], [Rectangle])

else:
    # Assigning a type to the variable 'matplotlib.patches' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib.patches', import_135290)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'from matplotlib import docstring' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_135292 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib')

if (type(import_135292) is not StypyTypeError):

    if (import_135292 != 'pyd_module'):
        __import__(import_135292)
        sys_modules_135293 = sys.modules[import_135292]
        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib', sys_modules_135293.module_type_store, module_type_store, ['docstring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 33, 0), __file__, sys_modules_135293, sys_modules_135293.module_type_store, module_type_store)
    else:
        from matplotlib import docstring

        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib', None, module_type_store, ['docstring'], [docstring])

else:
    # Assigning a type to the variable 'matplotlib' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib', import_135292)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from matplotlib.text import Text' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_135294 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.text')

if (type(import_135294) is not StypyTypeError):

    if (import_135294 != 'pyd_module'):
        __import__(import_135294)
        sys_modules_135295 = sys.modules[import_135294]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.text', sys_modules_135295.module_type_store, module_type_store, ['Text'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_135295, sys_modules_135295.module_type_store, module_type_store)
    else:
        from matplotlib.text import Text

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.text', None, module_type_store, ['Text'], [Text])

else:
    # Assigning a type to the variable 'matplotlib.text' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.text', import_135294)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from matplotlib.transforms import Bbox' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_135296 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.transforms')

if (type(import_135296) is not StypyTypeError):

    if (import_135296 != 'pyd_module'):
        __import__(import_135296)
        sys_modules_135297 = sys.modules[import_135296]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.transforms', sys_modules_135297.module_type_store, module_type_store, ['Bbox'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_135297, sys_modules_135297.module_type_store, module_type_store)
    else:
        from matplotlib.transforms import Bbox

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.transforms', None, module_type_store, ['Bbox'], [Bbox])

else:
    # Assigning a type to the variable 'matplotlib.transforms' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.transforms', import_135296)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from matplotlib.path import Path' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_135298 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.path')

if (type(import_135298) is not StypyTypeError):

    if (import_135298 != 'pyd_module'):
        __import__(import_135298)
        sys_modules_135299 = sys.modules[import_135298]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.path', sys_modules_135299.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_135299, sys_modules_135299.module_type_store, module_type_store)
    else:
        from matplotlib.path import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.path', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'matplotlib.path' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.path', import_135298)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'Cell' class
# Getting the type of 'Rectangle' (line 39)
Rectangle_135300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'Rectangle')

class Cell(Rectangle_135300, ):
    unicode_135301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'unicode', u'\n    A cell is a Rectangle with some associated text.\n\n    ')
    
    # Assigning a Num to a Name (line 44):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_135302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 27), 'unicode', u'k')
        unicode_135303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 42), 'unicode', u'w')
        # Getting the type of 'True' (line 48)
        True_135304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'True')
        unicode_135305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 22), 'unicode', u'')
        # Getting the type of 'None' (line 50)
        None_135306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 21), 'None')
        # Getting the type of 'None' (line 51)
        None_135307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'None')
        defaults = [unicode_135302, unicode_135303, True_135304, unicode_135305, None_135306, None_135307]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.__init__', ['xy', 'width', 'height', 'edgecolor', 'facecolor', 'fill', 'text', 'loc', 'fontproperties'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['xy', 'width', 'height', 'edgecolor', 'facecolor', 'fill', 'text', 'loc', 'fontproperties'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'self' (line 55)
        self_135310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'self', False)
        # Getting the type of 'xy' (line 55)
        xy_135311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'xy', False)
        # Processing the call keyword arguments (line 55)
        # Getting the type of 'width' (line 55)
        width_135312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 43), 'width', False)
        keyword_135313 = width_135312
        # Getting the type of 'height' (line 55)
        height_135314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 57), 'height', False)
        keyword_135315 = height_135314
        # Getting the type of 'edgecolor' (line 56)
        edgecolor_135316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 37), 'edgecolor', False)
        keyword_135317 = edgecolor_135316
        # Getting the type of 'facecolor' (line 56)
        facecolor_135318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 58), 'facecolor', False)
        keyword_135319 = facecolor_135318
        kwargs_135320 = {'edgecolor': keyword_135317, 'width': keyword_135313, 'facecolor': keyword_135319, 'height': keyword_135315}
        # Getting the type of 'Rectangle' (line 55)
        Rectangle_135308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'Rectangle', False)
        # Obtaining the member '__init__' of a type (line 55)
        init___135309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), Rectangle_135308, '__init__')
        # Calling __init__(args, kwargs) (line 55)
        init___call_result_135321 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), init___135309, *[self_135310, xy_135311], **kwargs_135320)
        
        
        # Call to set_clip_on(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'False' (line 57)
        False_135324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'False', False)
        # Processing the call keyword arguments (line 57)
        kwargs_135325 = {}
        # Getting the type of 'self' (line 57)
        self_135322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self', False)
        # Obtaining the member 'set_clip_on' of a type (line 57)
        set_clip_on_135323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_135322, 'set_clip_on')
        # Calling set_clip_on(args, kwargs) (line 57)
        set_clip_on_call_result_135326 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), set_clip_on_135323, *[False_135324], **kwargs_135325)
        
        
        # Type idiom detected: calculating its left and rigth part (line 60)
        # Getting the type of 'loc' (line 60)
        loc_135327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'loc')
        # Getting the type of 'None' (line 60)
        None_135328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'None')
        
        (may_be_135329, more_types_in_union_135330) = may_be_none(loc_135327, None_135328)

        if may_be_135329:

            if more_types_in_union_135330:
                # Runtime conditional SSA (line 60)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 61):
            
            # Assigning a Str to a Name (line 61):
            unicode_135331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 18), 'unicode', u'right')
            # Assigning a type to the variable 'loc' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'loc', unicode_135331)

            if more_types_in_union_135330:
                # SSA join for if statement (line 60)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 62):
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'loc' (line 62)
        loc_135332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'loc')
        # Getting the type of 'self' (line 62)
        self_135333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member '_loc' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_135333, '_loc', loc_135332)
        
        # Assigning a Call to a Attribute (line 63):
        
        # Assigning a Call to a Attribute (line 63):
        
        # Call to Text(...): (line 63)
        # Processing the call keyword arguments (line 63)
        
        # Obtaining the type of the subscript
        int_135335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 31), 'int')
        # Getting the type of 'xy' (line 63)
        xy_135336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'xy', False)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___135337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 28), xy_135336, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_135338 = invoke(stypy.reporting.localization.Localization(__file__, 63, 28), getitem___135337, int_135335)
        
        keyword_135339 = subscript_call_result_135338
        
        # Obtaining the type of the subscript
        int_135340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 40), 'int')
        # Getting the type of 'xy' (line 63)
        xy_135341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'xy', False)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___135342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 37), xy_135341, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_135343 = invoke(stypy.reporting.localization.Localization(__file__, 63, 37), getitem___135342, int_135340)
        
        keyword_135344 = subscript_call_result_135343
        # Getting the type of 'text' (line 63)
        text_135345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 49), 'text', False)
        keyword_135346 = text_135345
        # Getting the type of 'fontproperties' (line 64)
        fontproperties_135347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 41), 'fontproperties', False)
        keyword_135348 = fontproperties_135347
        kwargs_135349 = {'y': keyword_135344, 'x': keyword_135339, 'fontproperties': keyword_135348, 'text': keyword_135346}
        # Getting the type of 'Text' (line 63)
        Text_135334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 21), 'Text', False)
        # Calling Text(args, kwargs) (line 63)
        Text_call_result_135350 = invoke(stypy.reporting.localization.Localization(__file__, 63, 21), Text_135334, *[], **kwargs_135349)
        
        # Getting the type of 'self' (line 63)
        self_135351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member '_text' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_135351, '_text', Text_call_result_135350)
        
        # Call to set_clip_on(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'False' (line 65)
        False_135355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 31), 'False', False)
        # Processing the call keyword arguments (line 65)
        kwargs_135356 = {}
        # Getting the type of 'self' (line 65)
        self_135352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self', False)
        # Obtaining the member '_text' of a type (line 65)
        _text_135353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_135352, '_text')
        # Obtaining the member 'set_clip_on' of a type (line 65)
        set_clip_on_135354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), _text_135353, 'set_clip_on')
        # Calling set_clip_on(args, kwargs) (line 65)
        set_clip_on_call_result_135357 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), set_clip_on_135354, *[False_135355], **kwargs_135356)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_transform(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_transform'
        module_type_store = module_type_store.open_function_context('set_transform', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.set_transform.__dict__.__setitem__('stypy_localization', localization)
        Cell.set_transform.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.set_transform.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.set_transform.__dict__.__setitem__('stypy_function_name', 'Cell.set_transform')
        Cell.set_transform.__dict__.__setitem__('stypy_param_names_list', ['trans'])
        Cell.set_transform.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.set_transform.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.set_transform.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.set_transform.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.set_transform.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.set_transform.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.set_transform', ['trans'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_transform', localization, ['trans'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_transform(...)' code ##################

        
        # Call to set_transform(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'self' (line 68)
        self_135360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 32), 'self', False)
        # Getting the type of 'trans' (line 68)
        trans_135361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 38), 'trans', False)
        # Processing the call keyword arguments (line 68)
        kwargs_135362 = {}
        # Getting the type of 'Rectangle' (line 68)
        Rectangle_135358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'Rectangle', False)
        # Obtaining the member 'set_transform' of a type (line 68)
        set_transform_135359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), Rectangle_135358, 'set_transform')
        # Calling set_transform(args, kwargs) (line 68)
        set_transform_call_result_135363 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), set_transform_135359, *[self_135360, trans_135361], **kwargs_135362)
        
        
        # Assigning a Name to a Attribute (line 70):
        
        # Assigning a Name to a Attribute (line 70):
        # Getting the type of 'True' (line 70)
        True_135364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'True')
        # Getting the type of 'self' (line 70)
        self_135365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_135365, 'stale', True_135364)
        
        # ################# End of 'set_transform(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_transform' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_135366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135366)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_transform'
        return stypy_return_type_135366


    @norecursion
    def set_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_figure'
        module_type_store = module_type_store.open_function_context('set_figure', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.set_figure.__dict__.__setitem__('stypy_localization', localization)
        Cell.set_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.set_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.set_figure.__dict__.__setitem__('stypy_function_name', 'Cell.set_figure')
        Cell.set_figure.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        Cell.set_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.set_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.set_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.set_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.set_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.set_figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.set_figure', ['fig'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_figure(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'self' (line 73)
        self_135369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 29), 'self', False)
        # Getting the type of 'fig' (line 73)
        fig_135370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 35), 'fig', False)
        # Processing the call keyword arguments (line 73)
        kwargs_135371 = {}
        # Getting the type of 'Rectangle' (line 73)
        Rectangle_135367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'Rectangle', False)
        # Obtaining the member 'set_figure' of a type (line 73)
        set_figure_135368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), Rectangle_135367, 'set_figure')
        # Calling set_figure(args, kwargs) (line 73)
        set_figure_call_result_135372 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), set_figure_135368, *[self_135369, fig_135370], **kwargs_135371)
        
        
        # Call to set_figure(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'fig' (line 74)
        fig_135376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 30), 'fig', False)
        # Processing the call keyword arguments (line 74)
        kwargs_135377 = {}
        # Getting the type of 'self' (line 74)
        self_135373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self', False)
        # Obtaining the member '_text' of a type (line 74)
        _text_135374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_135373, '_text')
        # Obtaining the member 'set_figure' of a type (line 74)
        set_figure_135375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), _text_135374, 'set_figure')
        # Calling set_figure(args, kwargs) (line 74)
        set_figure_call_result_135378 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), set_figure_135375, *[fig_135376], **kwargs_135377)
        
        
        # ################# End of 'set_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_135379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135379)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_figure'
        return stypy_return_type_135379


    @norecursion
    def get_text(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_text'
        module_type_store = module_type_store.open_function_context('get_text', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.get_text.__dict__.__setitem__('stypy_localization', localization)
        Cell.get_text.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.get_text.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.get_text.__dict__.__setitem__('stypy_function_name', 'Cell.get_text')
        Cell.get_text.__dict__.__setitem__('stypy_param_names_list', [])
        Cell.get_text.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.get_text.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.get_text.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.get_text.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.get_text.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.get_text.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.get_text', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_text', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_text(...)' code ##################

        unicode_135380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 8), 'unicode', u'Return the cell Text intance')
        # Getting the type of 'self' (line 78)
        self_135381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'self')
        # Obtaining the member '_text' of a type (line 78)
        _text_135382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), self_135381, '_text')
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type', _text_135382)
        
        # ################# End of 'get_text(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_text' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_135383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135383)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_text'
        return stypy_return_type_135383


    @norecursion
    def set_fontsize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_fontsize'
        module_type_store = module_type_store.open_function_context('set_fontsize', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.set_fontsize.__dict__.__setitem__('stypy_localization', localization)
        Cell.set_fontsize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.set_fontsize.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.set_fontsize.__dict__.__setitem__('stypy_function_name', 'Cell.set_fontsize')
        Cell.set_fontsize.__dict__.__setitem__('stypy_param_names_list', ['size'])
        Cell.set_fontsize.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.set_fontsize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.set_fontsize.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.set_fontsize.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.set_fontsize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.set_fontsize.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.set_fontsize', ['size'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_fontsize', localization, ['size'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_fontsize(...)' code ##################

        
        # Call to set_fontsize(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'size' (line 81)
        size_135387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 32), 'size', False)
        # Processing the call keyword arguments (line 81)
        kwargs_135388 = {}
        # Getting the type of 'self' (line 81)
        self_135384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member '_text' of a type (line 81)
        _text_135385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_135384, '_text')
        # Obtaining the member 'set_fontsize' of a type (line 81)
        set_fontsize_135386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), _text_135385, 'set_fontsize')
        # Calling set_fontsize(args, kwargs) (line 81)
        set_fontsize_call_result_135389 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), set_fontsize_135386, *[size_135387], **kwargs_135388)
        
        
        # Assigning a Name to a Attribute (line 82):
        
        # Assigning a Name to a Attribute (line 82):
        # Getting the type of 'True' (line 82)
        True_135390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'True')
        # Getting the type of 'self' (line 82)
        self_135391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_135391, 'stale', True_135390)
        
        # ################# End of 'set_fontsize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_fontsize' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_135392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135392)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_fontsize'
        return stypy_return_type_135392


    @norecursion
    def get_fontsize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_fontsize'
        module_type_store = module_type_store.open_function_context('get_fontsize', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.get_fontsize.__dict__.__setitem__('stypy_localization', localization)
        Cell.get_fontsize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.get_fontsize.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.get_fontsize.__dict__.__setitem__('stypy_function_name', 'Cell.get_fontsize')
        Cell.get_fontsize.__dict__.__setitem__('stypy_param_names_list', [])
        Cell.get_fontsize.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.get_fontsize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.get_fontsize.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.get_fontsize.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.get_fontsize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.get_fontsize.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.get_fontsize', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_fontsize', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_fontsize(...)' code ##################

        unicode_135393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'unicode', u'Return the cell fontsize')
        
        # Call to get_fontsize(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_135397 = {}
        # Getting the type of 'self' (line 86)
        self_135394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'self', False)
        # Obtaining the member '_text' of a type (line 86)
        _text_135395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), self_135394, '_text')
        # Obtaining the member 'get_fontsize' of a type (line 86)
        get_fontsize_135396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), _text_135395, 'get_fontsize')
        # Calling get_fontsize(args, kwargs) (line 86)
        get_fontsize_call_result_135398 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), get_fontsize_135396, *[], **kwargs_135397)
        
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', get_fontsize_call_result_135398)
        
        # ################# End of 'get_fontsize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_fontsize' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_135399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135399)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_fontsize'
        return stypy_return_type_135399


    @norecursion
    def auto_set_font_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'auto_set_font_size'
        module_type_store = module_type_store.open_function_context('auto_set_font_size', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.auto_set_font_size.__dict__.__setitem__('stypy_localization', localization)
        Cell.auto_set_font_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.auto_set_font_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.auto_set_font_size.__dict__.__setitem__('stypy_function_name', 'Cell.auto_set_font_size')
        Cell.auto_set_font_size.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Cell.auto_set_font_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.auto_set_font_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.auto_set_font_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.auto_set_font_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.auto_set_font_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.auto_set_font_size.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.auto_set_font_size', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'auto_set_font_size', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'auto_set_font_size(...)' code ##################

        unicode_135400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'unicode', u' Shrink font size until text fits. ')
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to get_fontsize(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_135403 = {}
        # Getting the type of 'self' (line 90)
        self_135401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'self', False)
        # Obtaining the member 'get_fontsize' of a type (line 90)
        get_fontsize_135402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 19), self_135401, 'get_fontsize')
        # Calling get_fontsize(args, kwargs) (line 90)
        get_fontsize_call_result_135404 = invoke(stypy.reporting.localization.Localization(__file__, 90, 19), get_fontsize_135402, *[], **kwargs_135403)
        
        # Assigning a type to the variable 'fontsize' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'fontsize', get_fontsize_call_result_135404)
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to get_required_width(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'renderer' (line 91)
        renderer_135407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'renderer', False)
        # Processing the call keyword arguments (line 91)
        kwargs_135408 = {}
        # Getting the type of 'self' (line 91)
        self_135405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'self', False)
        # Obtaining the member 'get_required_width' of a type (line 91)
        get_required_width_135406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 19), self_135405, 'get_required_width')
        # Calling get_required_width(args, kwargs) (line 91)
        get_required_width_call_result_135409 = invoke(stypy.reporting.localization.Localization(__file__, 91, 19), get_required_width_135406, *[renderer_135407], **kwargs_135408)
        
        # Assigning a type to the variable 'required' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'required', get_required_width_call_result_135409)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'fontsize' (line 92)
        fontsize_135410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'fontsize')
        int_135411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'int')
        # Applying the binary operator '>' (line 92)
        result_gt_135412 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 14), '>', fontsize_135410, int_135411)
        
        
        # Getting the type of 'required' (line 92)
        required_135413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'required')
        
        # Call to get_width(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_135416 = {}
        # Getting the type of 'self' (line 92)
        self_135414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'self', False)
        # Obtaining the member 'get_width' of a type (line 92)
        get_width_135415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 42), self_135414, 'get_width')
        # Calling get_width(args, kwargs) (line 92)
        get_width_call_result_135417 = invoke(stypy.reporting.localization.Localization(__file__, 92, 42), get_width_135415, *[], **kwargs_135416)
        
        # Applying the binary operator '>' (line 92)
        result_gt_135418 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 31), '>', required_135413, get_width_call_result_135417)
        
        # Applying the binary operator 'and' (line 92)
        result_and_keyword_135419 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 14), 'and', result_gt_135412, result_gt_135418)
        
        # Testing the type of an if condition (line 92)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), result_and_keyword_135419)
        # SSA begins for while statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Getting the type of 'fontsize' (line 93)
        fontsize_135420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'fontsize')
        int_135421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 24), 'int')
        # Applying the binary operator '-=' (line 93)
        result_isub_135422 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '-=', fontsize_135420, int_135421)
        # Assigning a type to the variable 'fontsize' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'fontsize', result_isub_135422)
        
        
        # Call to set_fontsize(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'fontsize' (line 94)
        fontsize_135425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 30), 'fontsize', False)
        # Processing the call keyword arguments (line 94)
        kwargs_135426 = {}
        # Getting the type of 'self' (line 94)
        self_135423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'self', False)
        # Obtaining the member 'set_fontsize' of a type (line 94)
        set_fontsize_135424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), self_135423, 'set_fontsize')
        # Calling set_fontsize(args, kwargs) (line 94)
        set_fontsize_call_result_135427 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), set_fontsize_135424, *[fontsize_135425], **kwargs_135426)
        
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to get_required_width(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'renderer' (line 95)
        renderer_135430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 47), 'renderer', False)
        # Processing the call keyword arguments (line 95)
        kwargs_135431 = {}
        # Getting the type of 'self' (line 95)
        self_135428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'self', False)
        # Obtaining the member 'get_required_width' of a type (line 95)
        get_required_width_135429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 23), self_135428, 'get_required_width')
        # Calling get_required_width(args, kwargs) (line 95)
        get_required_width_call_result_135432 = invoke(stypy.reporting.localization.Localization(__file__, 95, 23), get_required_width_135429, *[renderer_135430], **kwargs_135431)
        
        # Assigning a type to the variable 'required' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'required', get_required_width_call_result_135432)
        # SSA join for while statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'fontsize' (line 97)
        fontsize_135433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'fontsize')
        # Assigning a type to the variable 'stypy_return_type' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'stypy_return_type', fontsize_135433)
        
        # ################# End of 'auto_set_font_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'auto_set_font_size' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_135434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135434)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'auto_set_font_size'
        return stypy_return_type_135434


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.draw.__dict__.__setitem__('stypy_localization', localization)
        Cell.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.draw.__dict__.__setitem__('stypy_function_name', 'Cell.draw')
        Cell.draw.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Cell.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.draw.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.draw', ['renderer'], None, None, defaults, varargs, kwargs)

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

        
        
        
        # Call to get_visible(...): (line 101)
        # Processing the call keyword arguments (line 101)
        kwargs_135437 = {}
        # Getting the type of 'self' (line 101)
        self_135435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'self', False)
        # Obtaining the member 'get_visible' of a type (line 101)
        get_visible_135436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 15), self_135435, 'get_visible')
        # Calling get_visible(args, kwargs) (line 101)
        get_visible_call_result_135438 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), get_visible_135436, *[], **kwargs_135437)
        
        # Applying the 'not' unary operator (line 101)
        result_not__135439 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), 'not', get_visible_call_result_135438)
        
        # Testing the type of an if condition (line 101)
        if_condition_135440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), result_not__135439)
        # Assigning a type to the variable 'if_condition_135440' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'if_condition_135440', if_condition_135440)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'self' (line 104)
        self_135443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 'self', False)
        # Getting the type of 'renderer' (line 104)
        renderer_135444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'renderer', False)
        # Processing the call keyword arguments (line 104)
        kwargs_135445 = {}
        # Getting the type of 'Rectangle' (line 104)
        Rectangle_135441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'Rectangle', False)
        # Obtaining the member 'draw' of a type (line 104)
        draw_135442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), Rectangle_135441, 'draw')
        # Calling draw(args, kwargs) (line 104)
        draw_call_result_135446 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), draw_135442, *[self_135443, renderer_135444], **kwargs_135445)
        
        
        # Call to _set_text_position(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'renderer' (line 107)
        renderer_135449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 32), 'renderer', False)
        # Processing the call keyword arguments (line 107)
        kwargs_135450 = {}
        # Getting the type of 'self' (line 107)
        self_135447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'self', False)
        # Obtaining the member '_set_text_position' of a type (line 107)
        _set_text_position_135448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), self_135447, '_set_text_position')
        # Calling _set_text_position(args, kwargs) (line 107)
        _set_text_position_call_result_135451 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), _set_text_position_135448, *[renderer_135449], **kwargs_135450)
        
        
        # Call to draw(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'renderer' (line 108)
        renderer_135455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'renderer', False)
        # Processing the call keyword arguments (line 108)
        kwargs_135456 = {}
        # Getting the type of 'self' (line 108)
        self_135452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self', False)
        # Obtaining the member '_text' of a type (line 108)
        _text_135453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_135452, '_text')
        # Obtaining the member 'draw' of a type (line 108)
        draw_135454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), _text_135453, 'draw')
        # Calling draw(args, kwargs) (line 108)
        draw_call_result_135457 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), draw_135454, *[renderer_135455], **kwargs_135456)
        
        
        # Assigning a Name to a Attribute (line 109):
        
        # Assigning a Name to a Attribute (line 109):
        # Getting the type of 'False' (line 109)
        False_135458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'False')
        # Getting the type of 'self' (line 109)
        self_135459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_135459, 'stale', False_135458)
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_135460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135460)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_135460


    @norecursion
    def _set_text_position(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_text_position'
        module_type_store = module_type_store.open_function_context('_set_text_position', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell._set_text_position.__dict__.__setitem__('stypy_localization', localization)
        Cell._set_text_position.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell._set_text_position.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell._set_text_position.__dict__.__setitem__('stypy_function_name', 'Cell._set_text_position')
        Cell._set_text_position.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Cell._set_text_position.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell._set_text_position.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell._set_text_position.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell._set_text_position.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell._set_text_position.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell._set_text_position.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell._set_text_position', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_text_position', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_text_position(...)' code ##################

        unicode_135461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, (-1)), 'unicode', u" Set text up so it draws in the right place.\n\n        Currently support 'left', 'center' and 'right'\n        ")
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to get_window_extent(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'renderer' (line 116)
        renderer_135464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 38), 'renderer', False)
        # Processing the call keyword arguments (line 116)
        kwargs_135465 = {}
        # Getting the type of 'self' (line 116)
        self_135462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'self', False)
        # Obtaining the member 'get_window_extent' of a type (line 116)
        get_window_extent_135463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 15), self_135462, 'get_window_extent')
        # Calling get_window_extent(args, kwargs) (line 116)
        get_window_extent_call_result_135466 = invoke(stypy.reporting.localization.Localization(__file__, 116, 15), get_window_extent_135463, *[renderer_135464], **kwargs_135465)
        
        # Assigning a type to the variable 'bbox' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'bbox', get_window_extent_call_result_135466)
        
        # Assigning a Attribute to a Tuple (line 117):
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_135467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 8), 'int')
        # Getting the type of 'bbox' (line 117)
        bbox_135468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 117)
        bounds_135469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 21), bbox_135468, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___135470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), bounds_135469, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_135471 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), getitem___135470, int_135467)
        
        # Assigning a type to the variable 'tuple_var_assignment_135243' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_135243', subscript_call_result_135471)
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_135472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 8), 'int')
        # Getting the type of 'bbox' (line 117)
        bbox_135473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 117)
        bounds_135474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 21), bbox_135473, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___135475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), bounds_135474, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_135476 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), getitem___135475, int_135472)
        
        # Assigning a type to the variable 'tuple_var_assignment_135244' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_135244', subscript_call_result_135476)
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_135477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 8), 'int')
        # Getting the type of 'bbox' (line 117)
        bbox_135478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 117)
        bounds_135479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 21), bbox_135478, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___135480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), bounds_135479, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_135481 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), getitem___135480, int_135477)
        
        # Assigning a type to the variable 'tuple_var_assignment_135245' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_135245', subscript_call_result_135481)
        
        # Assigning a Subscript to a Name (line 117):
        
        # Obtaining the type of the subscript
        int_135482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 8), 'int')
        # Getting the type of 'bbox' (line 117)
        bbox_135483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 117)
        bounds_135484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 21), bbox_135483, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___135485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), bounds_135484, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_135486 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), getitem___135485, int_135482)
        
        # Assigning a type to the variable 'tuple_var_assignment_135246' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_135246', subscript_call_result_135486)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_135243' (line 117)
        tuple_var_assignment_135243_135487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_135243')
        # Assigning a type to the variable 'l' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'l', tuple_var_assignment_135243_135487)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_135244' (line 117)
        tuple_var_assignment_135244_135488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_135244')
        # Assigning a type to the variable 'b' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'b', tuple_var_assignment_135244_135488)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_135245' (line 117)
        tuple_var_assignment_135245_135489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_135245')
        # Assigning a type to the variable 'w' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 14), 'w', tuple_var_assignment_135245_135489)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'tuple_var_assignment_135246' (line 117)
        tuple_var_assignment_135246_135490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'tuple_var_assignment_135246')
        # Assigning a type to the variable 'h' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'h', tuple_var_assignment_135246_135490)
        
        # Call to set_verticalalignment(...): (line 120)
        # Processing the call arguments (line 120)
        unicode_135494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 41), 'unicode', u'center')
        # Processing the call keyword arguments (line 120)
        kwargs_135495 = {}
        # Getting the type of 'self' (line 120)
        self_135491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self', False)
        # Obtaining the member '_text' of a type (line 120)
        _text_135492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_135491, '_text')
        # Obtaining the member 'set_verticalalignment' of a type (line 120)
        set_verticalalignment_135493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), _text_135492, 'set_verticalalignment')
        # Calling set_verticalalignment(args, kwargs) (line 120)
        set_verticalalignment_call_result_135496 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), set_verticalalignment_135493, *[unicode_135494], **kwargs_135495)
        
        
        # Assigning a BinOp to a Name (line 121):
        
        # Assigning a BinOp to a Name (line 121):
        # Getting the type of 'b' (line 121)
        b_135497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'b')
        # Getting the type of 'h' (line 121)
        h_135498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'h')
        float_135499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 21), 'float')
        # Applying the binary operator 'div' (line 121)
        result_div_135500 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 17), 'div', h_135498, float_135499)
        
        # Applying the binary operator '+' (line 121)
        result_add_135501 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 12), '+', b_135497, result_div_135500)
        
        # Assigning a type to the variable 'y' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'y', result_add_135501)
        
        
        # Getting the type of 'self' (line 124)
        self_135502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'self')
        # Obtaining the member '_loc' of a type (line 124)
        _loc_135503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 11), self_135502, '_loc')
        unicode_135504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 24), 'unicode', u'center')
        # Applying the binary operator '==' (line 124)
        result_eq_135505 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), '==', _loc_135503, unicode_135504)
        
        # Testing the type of an if condition (line 124)
        if_condition_135506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 8), result_eq_135505)
        # Assigning a type to the variable 'if_condition_135506' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'if_condition_135506', if_condition_135506)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_horizontalalignment(...): (line 125)
        # Processing the call arguments (line 125)
        unicode_135510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 47), 'unicode', u'center')
        # Processing the call keyword arguments (line 125)
        kwargs_135511 = {}
        # Getting the type of 'self' (line 125)
        self_135507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'self', False)
        # Obtaining the member '_text' of a type (line 125)
        _text_135508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), self_135507, '_text')
        # Obtaining the member 'set_horizontalalignment' of a type (line 125)
        set_horizontalalignment_135509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), _text_135508, 'set_horizontalalignment')
        # Calling set_horizontalalignment(args, kwargs) (line 125)
        set_horizontalalignment_call_result_135512 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), set_horizontalalignment_135509, *[unicode_135510], **kwargs_135511)
        
        
        # Assigning a BinOp to a Name (line 126):
        
        # Assigning a BinOp to a Name (line 126):
        # Getting the type of 'l' (line 126)
        l_135513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'l')
        # Getting the type of 'w' (line 126)
        w_135514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 21), 'w')
        float_135515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 25), 'float')
        # Applying the binary operator 'div' (line 126)
        result_div_135516 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 21), 'div', w_135514, float_135515)
        
        # Applying the binary operator '+' (line 126)
        result_add_135517 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 16), '+', l_135513, result_div_135516)
        
        # Assigning a type to the variable 'x' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'x', result_add_135517)
        # SSA branch for the else part of an if statement (line 124)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 127)
        self_135518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 13), 'self')
        # Obtaining the member '_loc' of a type (line 127)
        _loc_135519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 13), self_135518, '_loc')
        unicode_135520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 26), 'unicode', u'left')
        # Applying the binary operator '==' (line 127)
        result_eq_135521 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 13), '==', _loc_135519, unicode_135520)
        
        # Testing the type of an if condition (line 127)
        if_condition_135522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 13), result_eq_135521)
        # Assigning a type to the variable 'if_condition_135522' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 13), 'if_condition_135522', if_condition_135522)
        # SSA begins for if statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_horizontalalignment(...): (line 128)
        # Processing the call arguments (line 128)
        unicode_135526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 47), 'unicode', u'left')
        # Processing the call keyword arguments (line 128)
        kwargs_135527 = {}
        # Getting the type of 'self' (line 128)
        self_135523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'self', False)
        # Obtaining the member '_text' of a type (line 128)
        _text_135524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), self_135523, '_text')
        # Obtaining the member 'set_horizontalalignment' of a type (line 128)
        set_horizontalalignment_135525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), _text_135524, 'set_horizontalalignment')
        # Calling set_horizontalalignment(args, kwargs) (line 128)
        set_horizontalalignment_call_result_135528 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), set_horizontalalignment_135525, *[unicode_135526], **kwargs_135527)
        
        
        # Assigning a BinOp to a Name (line 129):
        
        # Assigning a BinOp to a Name (line 129):
        # Getting the type of 'l' (line 129)
        l_135529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'l')
        # Getting the type of 'w' (line 129)
        w_135530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'w')
        # Getting the type of 'self' (line 129)
        self_135531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'self')
        # Obtaining the member 'PAD' of a type (line 129)
        PAD_135532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), self_135531, 'PAD')
        # Applying the binary operator '*' (line 129)
        result_mul_135533 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 21), '*', w_135530, PAD_135532)
        
        # Applying the binary operator '+' (line 129)
        result_add_135534 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 16), '+', l_135529, result_mul_135533)
        
        # Assigning a type to the variable 'x' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'x', result_add_135534)
        # SSA branch for the else part of an if statement (line 127)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_horizontalalignment(...): (line 131)
        # Processing the call arguments (line 131)
        unicode_135538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 47), 'unicode', u'right')
        # Processing the call keyword arguments (line 131)
        kwargs_135539 = {}
        # Getting the type of 'self' (line 131)
        self_135535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'self', False)
        # Obtaining the member '_text' of a type (line 131)
        _text_135536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), self_135535, '_text')
        # Obtaining the member 'set_horizontalalignment' of a type (line 131)
        set_horizontalalignment_135537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), _text_135536, 'set_horizontalalignment')
        # Calling set_horizontalalignment(args, kwargs) (line 131)
        set_horizontalalignment_call_result_135540 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), set_horizontalalignment_135537, *[unicode_135538], **kwargs_135539)
        
        
        # Assigning a BinOp to a Name (line 132):
        
        # Assigning a BinOp to a Name (line 132):
        # Getting the type of 'l' (line 132)
        l_135541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'l')
        # Getting the type of 'w' (line 132)
        w_135542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 21), 'w')
        float_135543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 26), 'float')
        # Getting the type of 'self' (line 132)
        self_135544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 32), 'self')
        # Obtaining the member 'PAD' of a type (line 132)
        PAD_135545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 32), self_135544, 'PAD')
        # Applying the binary operator '-' (line 132)
        result_sub_135546 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 26), '-', float_135543, PAD_135545)
        
        # Applying the binary operator '*' (line 132)
        result_mul_135547 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 21), '*', w_135542, result_sub_135546)
        
        # Applying the binary operator '+' (line 132)
        result_add_135548 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 16), '+', l_135541, result_mul_135547)
        
        # Assigning a type to the variable 'x' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'x', result_add_135548)
        # SSA join for if statement (line 127)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_position(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Obtaining an instance of the builtin type 'tuple' (line 134)
        tuple_135552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 134)
        # Adding element type (line 134)
        # Getting the type of 'x' (line 134)
        x_135553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 33), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 33), tuple_135552, x_135553)
        # Adding element type (line 134)
        # Getting the type of 'y' (line 134)
        y_135554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 36), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 33), tuple_135552, y_135554)
        
        # Processing the call keyword arguments (line 134)
        kwargs_135555 = {}
        # Getting the type of 'self' (line 134)
        self_135549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'self', False)
        # Obtaining the member '_text' of a type (line 134)
        _text_135550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), self_135549, '_text')
        # Obtaining the member 'set_position' of a type (line 134)
        set_position_135551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), _text_135550, 'set_position')
        # Calling set_position(args, kwargs) (line 134)
        set_position_call_result_135556 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), set_position_135551, *[tuple_135552], **kwargs_135555)
        
        
        # ################# End of '_set_text_position(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_text_position' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_135557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135557)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_text_position'
        return stypy_return_type_135557


    @norecursion
    def get_text_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_text_bounds'
        module_type_store = module_type_store.open_function_context('get_text_bounds', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.get_text_bounds.__dict__.__setitem__('stypy_localization', localization)
        Cell.get_text_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.get_text_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.get_text_bounds.__dict__.__setitem__('stypy_function_name', 'Cell.get_text_bounds')
        Cell.get_text_bounds.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Cell.get_text_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.get_text_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.get_text_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.get_text_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.get_text_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.get_text_bounds.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.get_text_bounds', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_text_bounds', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_text_bounds(...)' code ##################

        unicode_135558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'unicode', u' Get text bounds in axes co-ordinates. ')
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to get_window_extent(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'renderer' (line 138)
        renderer_135562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 44), 'renderer', False)
        # Processing the call keyword arguments (line 138)
        kwargs_135563 = {}
        # Getting the type of 'self' (line 138)
        self_135559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'self', False)
        # Obtaining the member '_text' of a type (line 138)
        _text_135560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), self_135559, '_text')
        # Obtaining the member 'get_window_extent' of a type (line 138)
        get_window_extent_135561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), _text_135560, 'get_window_extent')
        # Calling get_window_extent(args, kwargs) (line 138)
        get_window_extent_call_result_135564 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), get_window_extent_135561, *[renderer_135562], **kwargs_135563)
        
        # Assigning a type to the variable 'bbox' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'bbox', get_window_extent_call_result_135564)
        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to inverse_transformed(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Call to get_data_transform(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_135569 = {}
        # Getting the type of 'self' (line 139)
        self_135567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 41), 'self', False)
        # Obtaining the member 'get_data_transform' of a type (line 139)
        get_data_transform_135568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 41), self_135567, 'get_data_transform')
        # Calling get_data_transform(args, kwargs) (line 139)
        get_data_transform_call_result_135570 = invoke(stypy.reporting.localization.Localization(__file__, 139, 41), get_data_transform_135568, *[], **kwargs_135569)
        
        # Processing the call keyword arguments (line 139)
        kwargs_135571 = {}
        # Getting the type of 'bbox' (line 139)
        bbox_135565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'bbox', False)
        # Obtaining the member 'inverse_transformed' of a type (line 139)
        inverse_transformed_135566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 16), bbox_135565, 'inverse_transformed')
        # Calling inverse_transformed(args, kwargs) (line 139)
        inverse_transformed_call_result_135572 = invoke(stypy.reporting.localization.Localization(__file__, 139, 16), inverse_transformed_135566, *[get_data_transform_call_result_135570], **kwargs_135571)
        
        # Assigning a type to the variable 'bboxa' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'bboxa', inverse_transformed_call_result_135572)
        # Getting the type of 'bboxa' (line 140)
        bboxa_135573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'bboxa')
        # Obtaining the member 'bounds' of a type (line 140)
        bounds_135574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 15), bboxa_135573, 'bounds')
        # Assigning a type to the variable 'stypy_return_type' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'stypy_return_type', bounds_135574)
        
        # ################# End of 'get_text_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_text_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_135575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135575)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_text_bounds'
        return stypy_return_type_135575


    @norecursion
    def get_required_width(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_required_width'
        module_type_store = module_type_store.open_function_context('get_required_width', 142, 4, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.get_required_width.__dict__.__setitem__('stypy_localization', localization)
        Cell.get_required_width.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.get_required_width.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.get_required_width.__dict__.__setitem__('stypy_function_name', 'Cell.get_required_width')
        Cell.get_required_width.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Cell.get_required_width.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.get_required_width.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Cell.get_required_width.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.get_required_width.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.get_required_width.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.get_required_width.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.get_required_width', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_required_width', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_required_width(...)' code ##################

        unicode_135576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 8), 'unicode', u' Get width required for this cell. ')
        
        # Assigning a Call to a Tuple (line 144):
        
        # Assigning a Call to a Name:
        
        # Call to get_text_bounds(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'renderer' (line 144)
        renderer_135579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 42), 'renderer', False)
        # Processing the call keyword arguments (line 144)
        kwargs_135580 = {}
        # Getting the type of 'self' (line 144)
        self_135577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'self', False)
        # Obtaining the member 'get_text_bounds' of a type (line 144)
        get_text_bounds_135578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 21), self_135577, 'get_text_bounds')
        # Calling get_text_bounds(args, kwargs) (line 144)
        get_text_bounds_call_result_135581 = invoke(stypy.reporting.localization.Localization(__file__, 144, 21), get_text_bounds_135578, *[renderer_135579], **kwargs_135580)
        
        # Assigning a type to the variable 'call_assignment_135247' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135247', get_text_bounds_call_result_135581)
        
        # Assigning a Call to a Name (line 144):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_135584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 8), 'int')
        # Processing the call keyword arguments
        kwargs_135585 = {}
        # Getting the type of 'call_assignment_135247' (line 144)
        call_assignment_135247_135582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135247', False)
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___135583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), call_assignment_135247_135582, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_135586 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___135583, *[int_135584], **kwargs_135585)
        
        # Assigning a type to the variable 'call_assignment_135248' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135248', getitem___call_result_135586)
        
        # Assigning a Name to a Name (line 144):
        # Getting the type of 'call_assignment_135248' (line 144)
        call_assignment_135248_135587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135248')
        # Assigning a type to the variable 'l' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'l', call_assignment_135248_135587)
        
        # Assigning a Call to a Name (line 144):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_135590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 8), 'int')
        # Processing the call keyword arguments
        kwargs_135591 = {}
        # Getting the type of 'call_assignment_135247' (line 144)
        call_assignment_135247_135588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135247', False)
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___135589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), call_assignment_135247_135588, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_135592 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___135589, *[int_135590], **kwargs_135591)
        
        # Assigning a type to the variable 'call_assignment_135249' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135249', getitem___call_result_135592)
        
        # Assigning a Name to a Name (line 144):
        # Getting the type of 'call_assignment_135249' (line 144)
        call_assignment_135249_135593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135249')
        # Assigning a type to the variable 'b' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'b', call_assignment_135249_135593)
        
        # Assigning a Call to a Name (line 144):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_135596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 8), 'int')
        # Processing the call keyword arguments
        kwargs_135597 = {}
        # Getting the type of 'call_assignment_135247' (line 144)
        call_assignment_135247_135594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135247', False)
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___135595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), call_assignment_135247_135594, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_135598 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___135595, *[int_135596], **kwargs_135597)
        
        # Assigning a type to the variable 'call_assignment_135250' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135250', getitem___call_result_135598)
        
        # Assigning a Name to a Name (line 144):
        # Getting the type of 'call_assignment_135250' (line 144)
        call_assignment_135250_135599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135250')
        # Assigning a type to the variable 'w' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 14), 'w', call_assignment_135250_135599)
        
        # Assigning a Call to a Name (line 144):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_135602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 8), 'int')
        # Processing the call keyword arguments
        kwargs_135603 = {}
        # Getting the type of 'call_assignment_135247' (line 144)
        call_assignment_135247_135600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135247', False)
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___135601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), call_assignment_135247_135600, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_135604 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___135601, *[int_135602], **kwargs_135603)
        
        # Assigning a type to the variable 'call_assignment_135251' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135251', getitem___call_result_135604)
        
        # Assigning a Name to a Name (line 144):
        # Getting the type of 'call_assignment_135251' (line 144)
        call_assignment_135251_135605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'call_assignment_135251')
        # Assigning a type to the variable 'h' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 17), 'h', call_assignment_135251_135605)
        # Getting the type of 'w' (line 145)
        w_135606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'w')
        float_135607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 20), 'float')
        float_135608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 27), 'float')
        # Getting the type of 'self' (line 145)
        self_135609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 33), 'self')
        # Obtaining the member 'PAD' of a type (line 145)
        PAD_135610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 33), self_135609, 'PAD')
        # Applying the binary operator '*' (line 145)
        result_mul_135611 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 27), '*', float_135608, PAD_135610)
        
        # Applying the binary operator '+' (line 145)
        result_add_135612 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 20), '+', float_135607, result_mul_135611)
        
        # Applying the binary operator '*' (line 145)
        result_mul_135613 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 15), '*', w_135606, result_add_135612)
        
        # Assigning a type to the variable 'stypy_return_type' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'stypy_return_type', result_mul_135613)
        
        # ################# End of 'get_required_width(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_required_width' in the type store
        # Getting the type of 'stypy_return_type' (line 142)
        stypy_return_type_135614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135614)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_required_width'
        return stypy_return_type_135614


    @norecursion
    def set_text_props(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_text_props'
        module_type_store = module_type_store.open_function_context('set_text_props', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Cell.set_text_props.__dict__.__setitem__('stypy_localization', localization)
        Cell.set_text_props.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Cell.set_text_props.__dict__.__setitem__('stypy_type_store', module_type_store)
        Cell.set_text_props.__dict__.__setitem__('stypy_function_name', 'Cell.set_text_props')
        Cell.set_text_props.__dict__.__setitem__('stypy_param_names_list', [])
        Cell.set_text_props.__dict__.__setitem__('stypy_varargs_param_name', None)
        Cell.set_text_props.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Cell.set_text_props.__dict__.__setitem__('stypy_call_defaults', defaults)
        Cell.set_text_props.__dict__.__setitem__('stypy_call_varargs', varargs)
        Cell.set_text_props.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Cell.set_text_props.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cell.set_text_props', [], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_text_props', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_text_props(...)' code ##################

        unicode_135615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 8), 'unicode', u'update the text properties with kwargs')
        
        # Call to update(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'kwargs' (line 149)
        kwargs_135619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'kwargs', False)
        # Processing the call keyword arguments (line 149)
        kwargs_135620 = {}
        # Getting the type of 'self' (line 149)
        self_135616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self', False)
        # Obtaining the member '_text' of a type (line 149)
        _text_135617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_135616, '_text')
        # Obtaining the member 'update' of a type (line 149)
        update_135618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), _text_135617, 'update')
        # Calling update(args, kwargs) (line 149)
        update_call_result_135621 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), update_135618, *[kwargs_135619], **kwargs_135620)
        
        
        # Assigning a Name to a Attribute (line 150):
        
        # Assigning a Name to a Attribute (line 150):
        # Getting the type of 'True' (line 150)
        True_135622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'True')
        # Getting the type of 'self' (line 150)
        self_135623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_135623, 'stale', True_135622)
        
        # ################# End of 'set_text_props(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_text_props' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_135624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135624)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_text_props'
        return stypy_return_type_135624


# Assigning a type to the variable 'Cell' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'Cell', Cell)

# Assigning a Num to a Name (line 44):
float_135625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 10), 'float')
# Getting the type of 'Cell'
Cell_135626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cell')
# Setting the type of the member 'PAD' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cell_135626, 'PAD', float_135625)
# Declaration of the 'CustomCell' class
# Getting the type of 'Cell' (line 153)
Cell_135627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'Cell')

class CustomCell(Cell_135627, ):
    unicode_135628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, (-1)), 'unicode', u'\n    A subclass of Cell where the sides may be visibly toggled.\n\n    ')
    
    # Assigning a Str to a Name (line 159):
    
    # Assigning a Dict to a Name (line 160):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 166, 4, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CustomCell.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to pop(...): (line 167)
        # Processing the call arguments (line 167)
        unicode_135631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 35), 'unicode', u'visible_edges')
        # Processing the call keyword arguments (line 167)
        kwargs_135632 = {}
        # Getting the type of 'kwargs' (line 167)
        kwargs_135629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 167)
        pop_135630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 24), kwargs_135629, 'pop')
        # Calling pop(args, kwargs) (line 167)
        pop_call_result_135633 = invoke(stypy.reporting.localization.Localization(__file__, 167, 24), pop_135630, *[unicode_135631], **kwargs_135632)
        
        # Assigning a type to the variable 'visible_edges' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'visible_edges', pop_call_result_135633)
        
        # Call to __init__(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'self' (line 168)
        self_135636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'self', False)
        # Getting the type of 'args' (line 168)
        args_135637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 29), 'args', False)
        # Processing the call keyword arguments (line 168)
        # Getting the type of 'kwargs' (line 168)
        kwargs_135638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'kwargs', False)
        kwargs_135639 = {'kwargs_135638': kwargs_135638}
        # Getting the type of 'Cell' (line 168)
        Cell_135634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'Cell', False)
        # Obtaining the member '__init__' of a type (line 168)
        init___135635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), Cell_135634, '__init__')
        # Calling __init__(args, kwargs) (line 168)
        init___call_result_135640 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), init___135635, *[self_135636, args_135637], **kwargs_135639)
        
        
        # Assigning a Name to a Attribute (line 169):
        
        # Assigning a Name to a Attribute (line 169):
        # Getting the type of 'visible_edges' (line 169)
        visible_edges_135641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'visible_edges')
        # Getting the type of 'self' (line 169)
        self_135642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self')
        # Setting the type of the member 'visible_edges' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_135642, 'visible_edges', visible_edges_135641)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def visible_edges(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visible_edges'
        module_type_store = module_type_store.open_function_context('visible_edges', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CustomCell.visible_edges.__dict__.__setitem__('stypy_localization', localization)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_type_store', module_type_store)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_function_name', 'CustomCell.visible_edges')
        CustomCell.visible_edges.__dict__.__setitem__('stypy_param_names_list', [])
        CustomCell.visible_edges.__dict__.__setitem__('stypy_varargs_param_name', None)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_call_defaults', defaults)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_call_varargs', varargs)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CustomCell.visible_edges', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visible_edges', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visible_edges(...)' code ##################

        # Getting the type of 'self' (line 173)
        self_135643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'self')
        # Obtaining the member '_visible_edges' of a type (line 173)
        _visible_edges_135644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 15), self_135643, '_visible_edges')
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', _visible_edges_135644)
        
        # ################# End of 'visible_edges(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visible_edges' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_135645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135645)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visible_edges'
        return stypy_return_type_135645


    @norecursion
    def visible_edges(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visible_edges'
        module_type_store = module_type_store.open_function_context('visible_edges', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CustomCell.visible_edges.__dict__.__setitem__('stypy_localization', localization)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_type_store', module_type_store)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_function_name', 'CustomCell.visible_edges')
        CustomCell.visible_edges.__dict__.__setitem__('stypy_param_names_list', ['value'])
        CustomCell.visible_edges.__dict__.__setitem__('stypy_varargs_param_name', None)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_call_defaults', defaults)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_call_varargs', varargs)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CustomCell.visible_edges.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CustomCell.visible_edges', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visible_edges', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visible_edges(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 177)
        # Getting the type of 'value' (line 177)
        value_135646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'value')
        # Getting the type of 'None' (line 177)
        None_135647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'None')
        
        (may_be_135648, more_types_in_union_135649) = may_be_none(value_135646, None_135647)

        if may_be_135648:

            if more_types_in_union_135649:
                # Runtime conditional SSA (line 177)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Attribute (line 178):
            
            # Assigning a Attribute to a Attribute (line 178):
            # Getting the type of 'self' (line 178)
            self_135650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 34), 'self')
            # Obtaining the member '_edges' of a type (line 178)
            _edges_135651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 34), self_135650, '_edges')
            # Getting the type of 'self' (line 178)
            self_135652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'self')
            # Setting the type of the member '_visible_edges' of a type (line 178)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), self_135652, '_visible_edges', _edges_135651)

            if more_types_in_union_135649:
                # Runtime conditional SSA for else branch (line 177)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_135648) or more_types_in_union_135649):
            
            
            # Getting the type of 'value' (line 179)
            value_135653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'value')
            # Getting the type of 'self' (line 179)
            self_135654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 22), 'self')
            # Obtaining the member '_edge_aliases' of a type (line 179)
            _edge_aliases_135655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 22), self_135654, '_edge_aliases')
            # Applying the binary operator 'in' (line 179)
            result_contains_135656 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 13), 'in', value_135653, _edge_aliases_135655)
            
            # Testing the type of an if condition (line 179)
            if_condition_135657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 13), result_contains_135656)
            # Assigning a type to the variable 'if_condition_135657' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'if_condition_135657', if_condition_135657)
            # SSA begins for if statement (line 179)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Attribute (line 180):
            
            # Assigning a Subscript to a Attribute (line 180):
            
            # Obtaining the type of the subscript
            # Getting the type of 'value' (line 180)
            value_135658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 53), 'value')
            # Getting the type of 'self' (line 180)
            self_135659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 34), 'self')
            # Obtaining the member '_edge_aliases' of a type (line 180)
            _edge_aliases_135660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 34), self_135659, '_edge_aliases')
            # Obtaining the member '__getitem__' of a type (line 180)
            getitem___135661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 34), _edge_aliases_135660, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 180)
            subscript_call_result_135662 = invoke(stypy.reporting.localization.Localization(__file__, 180, 34), getitem___135661, value_135658)
            
            # Getting the type of 'self' (line 180)
            self_135663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'self')
            # Setting the type of the member '_visible_edges' of a type (line 180)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 12), self_135663, '_visible_edges', subscript_call_result_135662)
            # SSA branch for the else part of an if statement (line 179)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'value' (line 182)
            value_135664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'value')
            # Testing the type of a for loop iterable (line 182)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 12), value_135664)
            # Getting the type of the for loop variable (line 182)
            for_loop_var_135665 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 12), value_135664)
            # Assigning a type to the variable 'edge' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'edge', for_loop_var_135665)
            # SSA begins for a for statement (line 182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'edge' (line 183)
            edge_135666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'edge')
            # Getting the type of 'self' (line 183)
            self_135667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 31), 'self')
            # Obtaining the member '_edges' of a type (line 183)
            _edges_135668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 31), self_135667, '_edges')
            # Applying the binary operator 'notin' (line 183)
            result_contains_135669 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 19), 'notin', edge_135666, _edges_135668)
            
            # Testing the type of an if condition (line 183)
            if_condition_135670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 16), result_contains_135669)
            # Assigning a type to the variable 'if_condition_135670' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'if_condition_135670', if_condition_135670)
            # SSA begins for if statement (line 183)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 184):
            
            # Assigning a Call to a Name (line 184):
            
            # Call to format(...): (line 184)
            # Processing the call arguments (line 184)
            # Getting the type of 'value' (line 186)
            value_135673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 35), 'value', False)
            
            # Call to join(...): (line 187)
            # Processing the call arguments (line 187)
            # Getting the type of 'self' (line 187)
            self_135676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 45), 'self', False)
            # Obtaining the member '_edge_aliases' of a type (line 187)
            _edge_aliases_135677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 45), self_135676, '_edge_aliases')
            # Processing the call keyword arguments (line 187)
            kwargs_135678 = {}
            unicode_135674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 35), 'unicode', u', ')
            # Obtaining the member 'join' of a type (line 187)
            join_135675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 35), unicode_135674, 'join')
            # Calling join(args, kwargs) (line 187)
            join_call_result_135679 = invoke(stypy.reporting.localization.Localization(__file__, 187, 35), join_135675, *[_edge_aliases_135677], **kwargs_135678)
            
            
            # Call to join(...): (line 188)
            # Processing the call arguments (line 188)
            # Getting the type of 'self' (line 188)
            self_135682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 45), 'self', False)
            # Obtaining the member '_edges' of a type (line 188)
            _edges_135683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 45), self_135682, '_edges')
            # Processing the call keyword arguments (line 188)
            kwargs_135684 = {}
            unicode_135680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 35), 'unicode', u', ')
            # Obtaining the member 'join' of a type (line 188)
            join_135681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 35), unicode_135680, 'join')
            # Calling join(args, kwargs) (line 188)
            join_call_result_135685 = invoke(stypy.reporting.localization.Localization(__file__, 188, 35), join_135681, *[_edges_135683], **kwargs_135684)
            
            # Processing the call keyword arguments (line 184)
            kwargs_135686 = {}
            unicode_135671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 27), 'unicode', u'Invalid edge param {0}, must only be one of {1} or string of {2}.')
            # Obtaining the member 'format' of a type (line 184)
            format_135672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 27), unicode_135671, 'format')
            # Calling format(args, kwargs) (line 184)
            format_call_result_135687 = invoke(stypy.reporting.localization.Localization(__file__, 184, 27), format_135672, *[value_135673, join_call_result_135679, join_call_result_135685], **kwargs_135686)
            
            # Assigning a type to the variable 'msg' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'msg', format_call_result_135687)
            
            # Call to ValueError(...): (line 190)
            # Processing the call arguments (line 190)
            # Getting the type of 'msg' (line 190)
            msg_135689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 37), 'msg', False)
            # Processing the call keyword arguments (line 190)
            kwargs_135690 = {}
            # Getting the type of 'ValueError' (line 190)
            ValueError_135688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 26), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 190)
            ValueError_call_result_135691 = invoke(stypy.reporting.localization.Localization(__file__, 190, 26), ValueError_135688, *[msg_135689], **kwargs_135690)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 190, 20), ValueError_call_result_135691, 'raise parameter', BaseException)
            # SSA join for if statement (line 183)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Attribute (line 191):
            
            # Assigning a Name to a Attribute (line 191):
            # Getting the type of 'value' (line 191)
            value_135692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 34), 'value')
            # Getting the type of 'self' (line 191)
            self_135693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'self')
            # Setting the type of the member '_visible_edges' of a type (line 191)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), self_135693, '_visible_edges', value_135692)
            # SSA join for if statement (line 179)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_135648 and more_types_in_union_135649):
                # SSA join for if statement (line 177)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 192):
        
        # Assigning a Name to a Attribute (line 192):
        # Getting the type of 'True' (line 192)
        True_135694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 21), 'True')
        # Getting the type of 'self' (line 192)
        self_135695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 192)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_135695, 'stale', True_135694)
        
        # ################# End of 'visible_edges(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visible_edges' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_135696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135696)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visible_edges'
        return stypy_return_type_135696


    @norecursion
    def get_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_path'
        module_type_store = module_type_store.open_function_context('get_path', 194, 4, False)
        # Assigning a type to the variable 'self' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CustomCell.get_path.__dict__.__setitem__('stypy_localization', localization)
        CustomCell.get_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CustomCell.get_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        CustomCell.get_path.__dict__.__setitem__('stypy_function_name', 'CustomCell.get_path')
        CustomCell.get_path.__dict__.__setitem__('stypy_param_names_list', [])
        CustomCell.get_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        CustomCell.get_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CustomCell.get_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        CustomCell.get_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        CustomCell.get_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CustomCell.get_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CustomCell.get_path', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_path(...)' code ##################

        unicode_135697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 8), 'unicode', u'Return a path where the edges specificed by _visible_edges are drawn')
        
        # Assigning a List to a Name (line 197):
        
        # Assigning a List to a Name (line 197):
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_135698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        # Getting the type of 'Path' (line 197)
        Path_135699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 17), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 197)
        MOVETO_135700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 17), Path_135699, 'MOVETO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 16), list_135698, MOVETO_135700)
        
        # Assigning a type to the variable 'codes' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'codes', list_135698)
        
        # Getting the type of 'self' (line 199)
        self_135701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'self')
        # Obtaining the member '_edges' of a type (line 199)
        _edges_135702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 20), self_135701, '_edges')
        # Testing the type of a for loop iterable (line 199)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 199, 8), _edges_135702)
        # Getting the type of the for loop variable (line 199)
        for_loop_var_135703 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 199, 8), _edges_135702)
        # Assigning a type to the variable 'edge' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'edge', for_loop_var_135703)
        # SSA begins for a for statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'edge' (line 200)
        edge_135704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'edge')
        # Getting the type of 'self' (line 200)
        self_135705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 23), 'self')
        # Obtaining the member '_visible_edges' of a type (line 200)
        _visible_edges_135706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 23), self_135705, '_visible_edges')
        # Applying the binary operator 'in' (line 200)
        result_contains_135707 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 15), 'in', edge_135704, _visible_edges_135706)
        
        # Testing the type of an if condition (line 200)
        if_condition_135708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 12), result_contains_135707)
        # Assigning a type to the variable 'if_condition_135708' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'if_condition_135708', if_condition_135708)
        # SSA begins for if statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'Path' (line 201)
        Path_135711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 29), 'Path', False)
        # Obtaining the member 'LINETO' of a type (line 201)
        LINETO_135712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 29), Path_135711, 'LINETO')
        # Processing the call keyword arguments (line 201)
        kwargs_135713 = {}
        # Getting the type of 'codes' (line 201)
        codes_135709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'codes', False)
        # Obtaining the member 'append' of a type (line 201)
        append_135710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), codes_135709, 'append')
        # Calling append(args, kwargs) (line 201)
        append_call_result_135714 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), append_135710, *[LINETO_135712], **kwargs_135713)
        
        # SSA branch for the else part of an if statement (line 200)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'Path' (line 203)
        Path_135717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 29), 'Path', False)
        # Obtaining the member 'MOVETO' of a type (line 203)
        MOVETO_135718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 29), Path_135717, 'MOVETO')
        # Processing the call keyword arguments (line 203)
        kwargs_135719 = {}
        # Getting the type of 'codes' (line 203)
        codes_135715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'codes', False)
        # Obtaining the member 'append' of a type (line 203)
        append_135716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 16), codes_135715, 'append')
        # Calling append(args, kwargs) (line 203)
        append_call_result_135720 = invoke(stypy.reporting.localization.Localization(__file__, 203, 16), append_135716, *[MOVETO_135718], **kwargs_135719)
        
        # SSA join for if statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'Path' (line 205)
        Path_135721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 205)
        MOVETO_135722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 11), Path_135721, 'MOVETO')
        
        # Obtaining the type of the subscript
        int_135723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 36), 'int')
        slice_135724 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 205, 30), int_135723, None, None)
        # Getting the type of 'codes' (line 205)
        codes_135725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), 'codes')
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___135726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 30), codes_135725, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_135727 = invoke(stypy.reporting.localization.Localization(__file__, 205, 30), getitem___135726, slice_135724)
        
        # Applying the binary operator 'notin' (line 205)
        result_contains_135728 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), 'notin', MOVETO_135722, subscript_call_result_135727)
        
        # Testing the type of an if condition (line 205)
        if_condition_135729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), result_contains_135728)
        # Assigning a type to the variable 'if_condition_135729' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_135729', if_condition_135729)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 206):
        
        # Assigning a Attribute to a Subscript (line 206):
        # Getting the type of 'Path' (line 206)
        Path_135730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'Path')
        # Obtaining the member 'CLOSEPOLY' of a type (line 206)
        CLOSEPOLY_135731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 24), Path_135730, 'CLOSEPOLY')
        # Getting the type of 'codes' (line 206)
        codes_135732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'codes')
        int_135733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 18), 'int')
        # Storing an element on a container (line 206)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 12), codes_135732, (int_135733, CLOSEPOLY_135731))
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to Path(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_135735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_135736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        float_135737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 13), list_135736, float_135737)
        # Adding element type (line 209)
        float_135738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 13), list_135736, float_135738)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), list_135735, list_135736)
        # Adding element type (line 209)
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_135739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        float_135740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 25), list_135739, float_135740)
        # Adding element type (line 209)
        float_135741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 25), list_135739, float_135741)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), list_135735, list_135739)
        # Adding element type (line 209)
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_135742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        float_135743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 37), list_135742, float_135743)
        # Adding element type (line 209)
        float_135744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 37), list_135742, float_135744)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), list_135735, list_135742)
        # Adding element type (line 209)
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_135745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        float_135746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 49), list_135745, float_135746)
        # Adding element type (line 209)
        float_135747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 49), list_135745, float_135747)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), list_135735, list_135745)
        # Adding element type (line 209)
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_135748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        float_135749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 61), list_135748, float_135749)
        # Adding element type (line 209)
        float_135750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 61), list_135748, float_135750)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), list_135735, list_135748)
        
        # Getting the type of 'codes' (line 210)
        codes_135751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'codes', False)
        # Processing the call keyword arguments (line 208)
        # Getting the type of 'True' (line 211)
        True_135752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'True', False)
        keyword_135753 = True_135752
        kwargs_135754 = {'readonly': keyword_135753}
        # Getting the type of 'Path' (line 208)
        Path_135734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'Path', False)
        # Calling Path(args, kwargs) (line 208)
        Path_call_result_135755 = invoke(stypy.reporting.localization.Localization(__file__, 208, 15), Path_135734, *[list_135735, codes_135751], **kwargs_135754)
        
        # Assigning a type to the variable 'stypy_return_type' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', Path_call_result_135755)
        
        # ################# End of 'get_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_path' in the type store
        # Getting the type of 'stypy_return_type' (line 194)
        stypy_return_type_135756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135756)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_path'
        return stypy_return_type_135756


# Assigning a type to the variable 'CustomCell' (line 153)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'CustomCell', CustomCell)

# Assigning a Str to a Name (line 159):
unicode_135757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 13), 'unicode', u'BRTL')
# Getting the type of 'CustomCell'
CustomCell_135758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CustomCell')
# Setting the type of the member '_edges' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CustomCell_135758, '_edges', unicode_135757)

# Assigning a Dict to a Name (line 160):

# Obtaining an instance of the builtin type 'dict' (line 160)
dict_135759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 20), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 160)
# Adding element type (key, value) (line 160)
unicode_135760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 21), 'unicode', u'open')
unicode_135761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 37), 'unicode', u'')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), dict_135759, (unicode_135760, unicode_135761))
# Adding element type (key, value) (line 160)
unicode_135762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 21), 'unicode', u'closed')
# Getting the type of 'CustomCell'
CustomCell_135763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CustomCell')
# Obtaining the member '_edges' of a type
_edges_135764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CustomCell_135763, '_edges')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), dict_135759, (unicode_135762, _edges_135764))
# Adding element type (key, value) (line 160)
unicode_135765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 21), 'unicode', u'horizontal')
unicode_135766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 37), 'unicode', u'BT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), dict_135759, (unicode_135765, unicode_135766))
# Adding element type (key, value) (line 160)
unicode_135767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 21), 'unicode', u'vertical')
unicode_135768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 37), 'unicode', u'RL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 20), dict_135759, (unicode_135767, unicode_135768))

# Getting the type of 'CustomCell'
CustomCell_135769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CustomCell')
# Setting the type of the member '_edge_aliases' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CustomCell_135769, '_edge_aliases', dict_135759)
# Declaration of the 'Table' class
# Getting the type of 'Artist' (line 215)
Artist_135770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'Artist')

class Table(Artist_135770, ):
    unicode_135771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, (-1)), 'unicode', u'\n    Create a table of cells.\n\n    Table can have (optional) row and column headers.\n\n    Each entry in the table can be either text or patches.\n\n    Column widths and row heights for the table can be specified.\n\n    Return value is a sequence of text, line and patch instances that make\n    up the table\n    ')
    
    # Assigning a Dict to a Name (line 228):
    
    # Assigning a Num to a Name (line 248):
    
    # Assigning a Num to a Name (line 249):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 251)
        None_135772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 31), 'None')
        # Getting the type of 'None' (line 251)
        None_135773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 42), 'None')
        defaults = [None_135772, None_135773]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 251, 4, False)
        # Assigning a type to the variable 'self' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.__init__', ['ax', 'loc', 'bbox'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['ax', 'loc', 'bbox'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'self' (line 253)
        self_135776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 24), 'self', False)
        # Processing the call keyword arguments (line 253)
        kwargs_135777 = {}
        # Getting the type of 'Artist' (line 253)
        Artist_135774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'Artist', False)
        # Obtaining the member '__init__' of a type (line 253)
        init___135775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), Artist_135774, '__init__')
        # Calling __init__(args, kwargs) (line 253)
        init___call_result_135778 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), init___135775, *[self_135776], **kwargs_135777)
        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'loc' (line 255)
        loc_135780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 22), 'loc', False)
        # Getting the type of 'six' (line 255)
        six_135781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'six', False)
        # Obtaining the member 'string_types' of a type (line 255)
        string_types_135782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 27), six_135781, 'string_types')
        # Processing the call keyword arguments (line 255)
        kwargs_135783 = {}
        # Getting the type of 'isinstance' (line 255)
        isinstance_135779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 255)
        isinstance_call_result_135784 = invoke(stypy.reporting.localization.Localization(__file__, 255, 11), isinstance_135779, *[loc_135780, string_types_135782], **kwargs_135783)
        
        
        # Getting the type of 'loc' (line 255)
        loc_135785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 49), 'loc')
        # Getting the type of 'self' (line 255)
        self_135786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 60), 'self')
        # Obtaining the member 'codes' of a type (line 255)
        codes_135787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 60), self_135786, 'codes')
        # Applying the binary operator 'notin' (line 255)
        result_contains_135788 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 49), 'notin', loc_135785, codes_135787)
        
        # Applying the binary operator 'and' (line 255)
        result_and_keyword_135789 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 11), 'and', isinstance_call_result_135784, result_contains_135788)
        
        # Testing the type of an if condition (line 255)
        if_condition_135790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), result_and_keyword_135789)
        # Assigning a type to the variable 'if_condition_135790' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_135790', if_condition_135790)
        # SSA begins for if statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 256)
        # Processing the call arguments (line 256)
        unicode_135793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 26), 'unicode', u'Unrecognized location %s. Falling back on bottom; valid locations are\n%s\t')
        
        # Obtaining an instance of the builtin type 'tuple' (line 258)
        tuple_135794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 258)
        # Adding element type (line 258)
        # Getting the type of 'loc' (line 258)
        loc_135795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 27), 'loc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 27), tuple_135794, loc_135795)
        # Adding element type (line 258)
        
        # Call to join(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'self' (line 258)
        self_135798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 44), 'self', False)
        # Obtaining the member 'codes' of a type (line 258)
        codes_135799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 44), self_135798, 'codes')
        # Processing the call keyword arguments (line 258)
        kwargs_135800 = {}
        unicode_135796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 32), 'unicode', u'\n\t')
        # Obtaining the member 'join' of a type (line 258)
        join_135797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 32), unicode_135796, 'join')
        # Calling join(args, kwargs) (line 258)
        join_call_result_135801 = invoke(stypy.reporting.localization.Localization(__file__, 258, 32), join_135797, *[codes_135799], **kwargs_135800)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 27), tuple_135794, join_call_result_135801)
        
        # Applying the binary operator '%' (line 256)
        result_mod_135802 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 26), '%', unicode_135793, tuple_135794)
        
        # Processing the call keyword arguments (line 256)
        kwargs_135803 = {}
        # Getting the type of 'warnings' (line 256)
        warnings_135791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 256)
        warn_135792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), warnings_135791, 'warn')
        # Calling warn(args, kwargs) (line 256)
        warn_call_result_135804 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), warn_135792, *[result_mod_135802], **kwargs_135803)
        
        
        # Assigning a Str to a Name (line 259):
        
        # Assigning a Str to a Name (line 259):
        unicode_135805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 18), 'unicode', u'bottom')
        # Assigning a type to the variable 'loc' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'loc', unicode_135805)
        # SSA join for if statement (line 255)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isinstance(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'loc' (line 260)
        loc_135807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 22), 'loc', False)
        # Getting the type of 'six' (line 260)
        six_135808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 27), 'six', False)
        # Obtaining the member 'string_types' of a type (line 260)
        string_types_135809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 27), six_135808, 'string_types')
        # Processing the call keyword arguments (line 260)
        kwargs_135810 = {}
        # Getting the type of 'isinstance' (line 260)
        isinstance_135806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 260)
        isinstance_call_result_135811 = invoke(stypy.reporting.localization.Localization(__file__, 260, 11), isinstance_135806, *[loc_135807, string_types_135809], **kwargs_135810)
        
        # Testing the type of an if condition (line 260)
        if_condition_135812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 8), isinstance_call_result_135811)
        # Assigning a type to the variable 'if_condition_135812' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'if_condition_135812', if_condition_135812)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 261):
        
        # Assigning a Call to a Name (line 261):
        
        # Call to get(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'loc' (line 261)
        loc_135816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 33), 'loc', False)
        int_135817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 38), 'int')
        # Processing the call keyword arguments (line 261)
        kwargs_135818 = {}
        # Getting the type of 'self' (line 261)
        self_135813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 18), 'self', False)
        # Obtaining the member 'codes' of a type (line 261)
        codes_135814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 18), self_135813, 'codes')
        # Obtaining the member 'get' of a type (line 261)
        get_135815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 18), codes_135814, 'get')
        # Calling get(args, kwargs) (line 261)
        get_call_result_135819 = invoke(stypy.reporting.localization.Localization(__file__, 261, 18), get_135815, *[loc_135816, int_135817], **kwargs_135818)
        
        # Assigning a type to the variable 'loc' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'loc', get_call_result_135819)
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_figure(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'ax' (line 262)
        ax_135822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'ax', False)
        # Obtaining the member 'figure' of a type (line 262)
        figure_135823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 24), ax_135822, 'figure')
        # Processing the call keyword arguments (line 262)
        kwargs_135824 = {}
        # Getting the type of 'self' (line 262)
        self_135820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', False)
        # Obtaining the member 'set_figure' of a type (line 262)
        set_figure_135821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_135820, 'set_figure')
        # Calling set_figure(args, kwargs) (line 262)
        set_figure_call_result_135825 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), set_figure_135821, *[figure_135823], **kwargs_135824)
        
        
        # Assigning a Name to a Attribute (line 263):
        
        # Assigning a Name to a Attribute (line 263):
        # Getting the type of 'ax' (line 263)
        ax_135826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 21), 'ax')
        # Getting the type of 'self' (line 263)
        self_135827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'self')
        # Setting the type of the member '_axes' of a type (line 263)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), self_135827, '_axes', ax_135826)
        
        # Assigning a Name to a Attribute (line 264):
        
        # Assigning a Name to a Attribute (line 264):
        # Getting the type of 'loc' (line 264)
        loc_135828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'loc')
        # Getting the type of 'self' (line 264)
        self_135829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'self')
        # Setting the type of the member '_loc' of a type (line 264)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_135829, '_loc', loc_135828)
        
        # Assigning a Name to a Attribute (line 265):
        
        # Assigning a Name to a Attribute (line 265):
        # Getting the type of 'bbox' (line 265)
        bbox_135830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 21), 'bbox')
        # Getting the type of 'self' (line 265)
        self_135831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'self')
        # Setting the type of the member '_bbox' of a type (line 265)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), self_135831, '_bbox', bbox_135830)
        
        # Call to set_transform(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'ax' (line 268)
        ax_135834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 27), 'ax', False)
        # Obtaining the member 'transAxes' of a type (line 268)
        transAxes_135835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 27), ax_135834, 'transAxes')
        # Processing the call keyword arguments (line 268)
        kwargs_135836 = {}
        # Getting the type of 'self' (line 268)
        self_135832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self', False)
        # Obtaining the member 'set_transform' of a type (line 268)
        set_transform_135833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), self_135832, 'set_transform')
        # Calling set_transform(args, kwargs) (line 268)
        set_transform_call_result_135837 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), set_transform_135833, *[transAxes_135835], **kwargs_135836)
        
        
        # Assigning a List to a Attribute (line 270):
        
        # Assigning a List to a Attribute (line 270):
        
        # Obtaining an instance of the builtin type 'list' (line 270)
        list_135838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 270)
        
        # Getting the type of 'self' (line 270)
        self_135839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self')
        # Setting the type of the member '_texts' of a type (line 270)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_135839, '_texts', list_135838)
        
        # Assigning a Dict to a Attribute (line 271):
        
        # Assigning a Dict to a Attribute (line 271):
        
        # Obtaining an instance of the builtin type 'dict' (line 271)
        dict_135840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 271)
        
        # Getting the type of 'self' (line 271)
        self_135841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'self')
        # Setting the type of the member '_cells' of a type (line 271)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), self_135841, '_cells', dict_135840)
        
        # Assigning a Name to a Attribute (line 272):
        
        # Assigning a Name to a Attribute (line 272):
        # Getting the type of 'None' (line 272)
        None_135842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'None')
        # Getting the type of 'self' (line 272)
        self_135843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self')
        # Setting the type of the member '_edges' of a type (line 272)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_135843, '_edges', None_135842)
        
        # Assigning a List to a Attribute (line 273):
        
        # Assigning a List to a Attribute (line 273):
        
        # Obtaining an instance of the builtin type 'list' (line 273)
        list_135844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 273)
        
        # Getting the type of 'self' (line 273)
        self_135845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'self')
        # Setting the type of the member '_autoRows' of a type (line 273)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), self_135845, '_autoRows', list_135844)
        
        # Assigning a List to a Attribute (line 274):
        
        # Assigning a List to a Attribute (line 274):
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_135846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        
        # Getting the type of 'self' (line 274)
        self_135847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'self')
        # Setting the type of the member '_autoColumns' of a type (line 274)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), self_135847, '_autoColumns', list_135846)
        
        # Assigning a Name to a Attribute (line 275):
        
        # Assigning a Name to a Attribute (line 275):
        # Getting the type of 'True' (line 275)
        True_135848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 29), 'True')
        # Getting the type of 'self' (line 275)
        self_135849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self')
        # Setting the type of the member '_autoFontsize' of a type (line 275)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_135849, '_autoFontsize', True_135848)
        
        # Call to update(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'kwargs' (line 276)
        kwargs_135852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'kwargs', False)
        # Processing the call keyword arguments (line 276)
        kwargs_135853 = {}
        # Getting the type of 'self' (line 276)
        self_135850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'self', False)
        # Obtaining the member 'update' of a type (line 276)
        update_135851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), self_135850, 'update')
        # Calling update(args, kwargs) (line 276)
        update_call_result_135854 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), update_135851, *[kwargs_135852], **kwargs_135853)
        
        
        # Call to set_clip_on(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'False' (line 278)
        False_135857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 25), 'False', False)
        # Processing the call keyword arguments (line 278)
        kwargs_135858 = {}
        # Getting the type of 'self' (line 278)
        self_135855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'self', False)
        # Obtaining the member 'set_clip_on' of a type (line 278)
        set_clip_on_135856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), self_135855, 'set_clip_on')
        # Calling set_clip_on(args, kwargs) (line 278)
        set_clip_on_call_result_135859 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), set_clip_on_135856, *[False_135857], **kwargs_135858)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def add_cell(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_cell'
        module_type_store = module_type_store.open_function_context('add_cell', 280, 4, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.add_cell.__dict__.__setitem__('stypy_localization', localization)
        Table.add_cell.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.add_cell.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.add_cell.__dict__.__setitem__('stypy_function_name', 'Table.add_cell')
        Table.add_cell.__dict__.__setitem__('stypy_param_names_list', ['row', 'col'])
        Table.add_cell.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Table.add_cell.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Table.add_cell.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.add_cell.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.add_cell.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.add_cell.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.add_cell', ['row', 'col'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_cell', localization, ['row', 'col'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_cell(...)' code ##################

        unicode_135860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 8), 'unicode', u' Add a cell to the table. ')
        
        # Assigning a Tuple to a Name (line 282):
        
        # Assigning a Tuple to a Name (line 282):
        
        # Obtaining an instance of the builtin type 'tuple' (line 282)
        tuple_135861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 282)
        # Adding element type (line 282)
        int_135862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 14), tuple_135861, int_135862)
        # Adding element type (line 282)
        int_135863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 14), tuple_135861, int_135863)
        
        # Assigning a type to the variable 'xy' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'xy', tuple_135861)
        
        # Assigning a Call to a Name (line 284):
        
        # Assigning a Call to a Name (line 284):
        
        # Call to CustomCell(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'xy' (line 284)
        xy_135865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 26), 'xy', False)
        # Getting the type of 'args' (line 284)
        args_135866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 57), 'args', False)
        # Processing the call keyword arguments (line 284)
        # Getting the type of 'self' (line 284)
        self_135867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 44), 'self', False)
        # Obtaining the member 'edges' of a type (line 284)
        edges_135868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 44), self_135867, 'edges')
        keyword_135869 = edges_135868
        # Getting the type of 'kwargs' (line 284)
        kwargs_135870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 65), 'kwargs', False)
        kwargs_135871 = {'visible_edges': keyword_135869, 'kwargs_135870': kwargs_135870}
        # Getting the type of 'CustomCell' (line 284)
        CustomCell_135864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 15), 'CustomCell', False)
        # Calling CustomCell(args, kwargs) (line 284)
        CustomCell_call_result_135872 = invoke(stypy.reporting.localization.Localization(__file__, 284, 15), CustomCell_135864, *[xy_135865, args_135866], **kwargs_135871)
        
        # Assigning a type to the variable 'cell' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'cell', CustomCell_call_result_135872)
        
        # Call to set_figure(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'self' (line 285)
        self_135875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'self', False)
        # Obtaining the member 'figure' of a type (line 285)
        figure_135876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 24), self_135875, 'figure')
        # Processing the call keyword arguments (line 285)
        kwargs_135877 = {}
        # Getting the type of 'cell' (line 285)
        cell_135873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'cell', False)
        # Obtaining the member 'set_figure' of a type (line 285)
        set_figure_135874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), cell_135873, 'set_figure')
        # Calling set_figure(args, kwargs) (line 285)
        set_figure_call_result_135878 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), set_figure_135874, *[figure_135876], **kwargs_135877)
        
        
        # Call to set_transform(...): (line 286)
        # Processing the call arguments (line 286)
        
        # Call to get_transform(...): (line 286)
        # Processing the call keyword arguments (line 286)
        kwargs_135883 = {}
        # Getting the type of 'self' (line 286)
        self_135881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 27), 'self', False)
        # Obtaining the member 'get_transform' of a type (line 286)
        get_transform_135882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 27), self_135881, 'get_transform')
        # Calling get_transform(args, kwargs) (line 286)
        get_transform_call_result_135884 = invoke(stypy.reporting.localization.Localization(__file__, 286, 27), get_transform_135882, *[], **kwargs_135883)
        
        # Processing the call keyword arguments (line 286)
        kwargs_135885 = {}
        # Getting the type of 'cell' (line 286)
        cell_135879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'cell', False)
        # Obtaining the member 'set_transform' of a type (line 286)
        set_transform_135880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), cell_135879, 'set_transform')
        # Calling set_transform(args, kwargs) (line 286)
        set_transform_call_result_135886 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), set_transform_135880, *[get_transform_call_result_135884], **kwargs_135885)
        
        
        # Call to set_clip_on(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'False' (line 288)
        False_135889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 25), 'False', False)
        # Processing the call keyword arguments (line 288)
        kwargs_135890 = {}
        # Getting the type of 'cell' (line 288)
        cell_135887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'cell', False)
        # Obtaining the member 'set_clip_on' of a type (line 288)
        set_clip_on_135888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), cell_135887, 'set_clip_on')
        # Calling set_clip_on(args, kwargs) (line 288)
        set_clip_on_call_result_135891 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), set_clip_on_135888, *[False_135889], **kwargs_135890)
        
        
        # Assigning a Name to a Subscript (line 289):
        
        # Assigning a Name to a Subscript (line 289):
        # Getting the type of 'cell' (line 289)
        cell_135892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 32), 'cell')
        # Getting the type of 'self' (line 289)
        self_135893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'self')
        # Obtaining the member '_cells' of a type (line 289)
        _cells_135894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), self_135893, '_cells')
        
        # Obtaining an instance of the builtin type 'tuple' (line 289)
        tuple_135895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 289)
        # Adding element type (line 289)
        # Getting the type of 'row' (line 289)
        row_135896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 20), tuple_135895, row_135896)
        # Adding element type (line 289)
        # Getting the type of 'col' (line 289)
        col_135897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 25), 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 20), tuple_135895, col_135897)
        
        # Storing an element on a container (line 289)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 8), _cells_135894, (tuple_135895, cell_135892))
        
        # Assigning a Name to a Attribute (line 290):
        
        # Assigning a Name to a Attribute (line 290):
        # Getting the type of 'True' (line 290)
        True_135898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 21), 'True')
        # Getting the type of 'self' (line 290)
        self_135899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 290)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), self_135899, 'stale', True_135898)
        
        # ################# End of 'add_cell(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_cell' in the type store
        # Getting the type of 'stypy_return_type' (line 280)
        stypy_return_type_135900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135900)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_cell'
        return stypy_return_type_135900


    @norecursion
    def edges(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'edges'
        module_type_store = module_type_store.open_function_context('edges', 292, 4, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.edges.__dict__.__setitem__('stypy_localization', localization)
        Table.edges.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.edges.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.edges.__dict__.__setitem__('stypy_function_name', 'Table.edges')
        Table.edges.__dict__.__setitem__('stypy_param_names_list', [])
        Table.edges.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table.edges.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table.edges.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.edges.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.edges.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.edges.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.edges', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'edges', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'edges(...)' code ##################

        # Getting the type of 'self' (line 294)
        self_135901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'self')
        # Obtaining the member '_edges' of a type (line 294)
        _edges_135902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), self_135901, '_edges')
        # Assigning a type to the variable 'stypy_return_type' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'stypy_return_type', _edges_135902)
        
        # ################# End of 'edges(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'edges' in the type store
        # Getting the type of 'stypy_return_type' (line 292)
        stypy_return_type_135903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135903)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'edges'
        return stypy_return_type_135903


    @norecursion
    def edges(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'edges'
        module_type_store = module_type_store.open_function_context('edges', 296, 4, False)
        # Assigning a type to the variable 'self' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.edges.__dict__.__setitem__('stypy_localization', localization)
        Table.edges.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.edges.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.edges.__dict__.__setitem__('stypy_function_name', 'Table.edges')
        Table.edges.__dict__.__setitem__('stypy_param_names_list', ['value'])
        Table.edges.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table.edges.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table.edges.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.edges.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.edges.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.edges.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.edges', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'edges', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'edges(...)' code ##################

        
        # Assigning a Name to a Attribute (line 298):
        
        # Assigning a Name to a Attribute (line 298):
        # Getting the type of 'value' (line 298)
        value_135904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 22), 'value')
        # Getting the type of 'self' (line 298)
        self_135905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'self')
        # Setting the type of the member '_edges' of a type (line 298)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), self_135905, '_edges', value_135904)
        
        # Assigning a Name to a Attribute (line 299):
        
        # Assigning a Name to a Attribute (line 299):
        # Getting the type of 'True' (line 299)
        True_135906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 21), 'True')
        # Getting the type of 'self' (line 299)
        self_135907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 299)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), self_135907, 'stale', True_135906)
        
        # ################# End of 'edges(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'edges' in the type store
        # Getting the type of 'stypy_return_type' (line 296)
        stypy_return_type_135908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135908)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'edges'
        return stypy_return_type_135908


    @norecursion
    def _approx_text_height(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_approx_text_height'
        module_type_store = module_type_store.open_function_context('_approx_text_height', 301, 4, False)
        # Assigning a type to the variable 'self' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table._approx_text_height.__dict__.__setitem__('stypy_localization', localization)
        Table._approx_text_height.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table._approx_text_height.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table._approx_text_height.__dict__.__setitem__('stypy_function_name', 'Table._approx_text_height')
        Table._approx_text_height.__dict__.__setitem__('stypy_param_names_list', [])
        Table._approx_text_height.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table._approx_text_height.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table._approx_text_height.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table._approx_text_height.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table._approx_text_height.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table._approx_text_height.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table._approx_text_height', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_approx_text_height', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_approx_text_height(...)' code ##################

        # Getting the type of 'self' (line 302)
        self_135909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'self')
        # Obtaining the member 'FONTSIZE' of a type (line 302)
        FONTSIZE_135910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), self_135909, 'FONTSIZE')
        float_135911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 32), 'float')
        # Applying the binary operator 'div' (line 302)
        result_div_135912 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 16), 'div', FONTSIZE_135910, float_135911)
        
        # Getting the type of 'self' (line 302)
        self_135913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 39), 'self')
        # Obtaining the member 'figure' of a type (line 302)
        figure_135914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 39), self_135913, 'figure')
        # Obtaining the member 'dpi' of a type (line 302)
        dpi_135915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 39), figure_135914, 'dpi')
        # Applying the binary operator '*' (line 302)
        result_mul_135916 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 37), '*', result_div_135912, dpi_135915)
        
        # Getting the type of 'self' (line 303)
        self_135917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'self')
        # Obtaining the member '_axes' of a type (line 303)
        _axes_135918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 16), self_135917, '_axes')
        # Obtaining the member 'bbox' of a type (line 303)
        bbox_135919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 16), _axes_135918, 'bbox')
        # Obtaining the member 'height' of a type (line 303)
        height_135920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 16), bbox_135919, 'height')
        # Applying the binary operator 'div' (line 302)
        result_div_135921 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 55), 'div', result_mul_135916, height_135920)
        
        float_135922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 41), 'float')
        # Applying the binary operator '*' (line 303)
        result_mul_135923 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 39), '*', result_div_135921, float_135922)
        
        # Assigning a type to the variable 'stypy_return_type' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'stypy_return_type', result_mul_135923)
        
        # ################# End of '_approx_text_height(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_approx_text_height' in the type store
        # Getting the type of 'stypy_return_type' (line 301)
        stypy_return_type_135924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135924)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_approx_text_height'
        return stypy_return_type_135924


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 305, 4, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.draw.__dict__.__setitem__('stypy_localization', localization)
        Table.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.draw.__dict__.__setitem__('stypy_function_name', 'Table.draw')
        Table.draw.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Table.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.draw.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.draw', ['renderer'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 309)
        # Getting the type of 'renderer' (line 309)
        renderer_135925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 11), 'renderer')
        # Getting the type of 'None' (line 309)
        None_135926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 23), 'None')
        
        (may_be_135927, more_types_in_union_135928) = may_be_none(renderer_135925, None_135926)

        if may_be_135927:

            if more_types_in_union_135928:
                # Runtime conditional SSA (line 309)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 310):
            
            # Assigning a Attribute to a Name (line 310):
            # Getting the type of 'self' (line 310)
            self_135929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 23), 'self')
            # Obtaining the member 'figure' of a type (line 310)
            figure_135930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 23), self_135929, 'figure')
            # Obtaining the member '_cachedRenderer' of a type (line 310)
            _cachedRenderer_135931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 23), figure_135930, '_cachedRenderer')
            # Assigning a type to the variable 'renderer' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'renderer', _cachedRenderer_135931)

            if more_types_in_union_135928:
                # SSA join for if statement (line 309)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 311)
        # Getting the type of 'renderer' (line 311)
        renderer_135932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 11), 'renderer')
        # Getting the type of 'None' (line 311)
        None_135933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 23), 'None')
        
        (may_be_135934, more_types_in_union_135935) = may_be_none(renderer_135932, None_135933)

        if may_be_135934:

            if more_types_in_union_135935:
                # Runtime conditional SSA (line 311)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to RuntimeError(...): (line 312)
            # Processing the call arguments (line 312)
            unicode_135937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 31), 'unicode', u'No renderer defined')
            # Processing the call keyword arguments (line 312)
            kwargs_135938 = {}
            # Getting the type of 'RuntimeError' (line 312)
            RuntimeError_135936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 312)
            RuntimeError_call_result_135939 = invoke(stypy.reporting.localization.Localization(__file__, 312, 18), RuntimeError_135936, *[unicode_135937], **kwargs_135938)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 312, 12), RuntimeError_call_result_135939, 'raise parameter', BaseException)

            if more_types_in_union_135935:
                # SSA join for if statement (line 311)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to get_visible(...): (line 314)
        # Processing the call keyword arguments (line 314)
        kwargs_135942 = {}
        # Getting the type of 'self' (line 314)
        self_135940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), 'self', False)
        # Obtaining the member 'get_visible' of a type (line 314)
        get_visible_135941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 15), self_135940, 'get_visible')
        # Calling get_visible(args, kwargs) (line 314)
        get_visible_call_result_135943 = invoke(stypy.reporting.localization.Localization(__file__, 314, 15), get_visible_135941, *[], **kwargs_135942)
        
        # Applying the 'not' unary operator (line 314)
        result_not__135944 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 11), 'not', get_visible_call_result_135943)
        
        # Testing the type of an if condition (line 314)
        if_condition_135945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 8), result_not__135944)
        # Assigning a type to the variable 'if_condition_135945' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'if_condition_135945', if_condition_135945)
        # SSA begins for if statement (line 314)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 314)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to open_group(...): (line 316)
        # Processing the call arguments (line 316)
        unicode_135948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 28), 'unicode', u'table')
        # Processing the call keyword arguments (line 316)
        kwargs_135949 = {}
        # Getting the type of 'renderer' (line 316)
        renderer_135946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'renderer', False)
        # Obtaining the member 'open_group' of a type (line 316)
        open_group_135947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), renderer_135946, 'open_group')
        # Calling open_group(args, kwargs) (line 316)
        open_group_call_result_135950 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), open_group_135947, *[unicode_135948], **kwargs_135949)
        
        
        # Call to _update_positions(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'renderer' (line 317)
        renderer_135953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 31), 'renderer', False)
        # Processing the call keyword arguments (line 317)
        kwargs_135954 = {}
        # Getting the type of 'self' (line 317)
        self_135951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'self', False)
        # Obtaining the member '_update_positions' of a type (line 317)
        _update_positions_135952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), self_135951, '_update_positions')
        # Calling _update_positions(args, kwargs) (line 317)
        _update_positions_call_result_135955 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), _update_positions_135952, *[renderer_135953], **kwargs_135954)
        
        
        
        # Call to sorted(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'self' (line 319)
        self_135957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 26), 'self', False)
        # Obtaining the member '_cells' of a type (line 319)
        _cells_135958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 26), self_135957, '_cells')
        # Processing the call keyword arguments (line 319)
        kwargs_135959 = {}
        # Getting the type of 'sorted' (line 319)
        sorted_135956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 19), 'sorted', False)
        # Calling sorted(args, kwargs) (line 319)
        sorted_call_result_135960 = invoke(stypy.reporting.localization.Localization(__file__, 319, 19), sorted_135956, *[_cells_135958], **kwargs_135959)
        
        # Testing the type of a for loop iterable (line 319)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 319, 8), sorted_call_result_135960)
        # Getting the type of the for loop variable (line 319)
        for_loop_var_135961 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 319, 8), sorted_call_result_135960)
        # Assigning a type to the variable 'key' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'key', for_loop_var_135961)
        # SSA begins for a for statement (line 319)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to draw(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'renderer' (line 320)
        renderer_135968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 34), 'renderer', False)
        # Processing the call keyword arguments (line 320)
        kwargs_135969 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 320)
        key_135962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'key', False)
        # Getting the type of 'self' (line 320)
        self_135963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'self', False)
        # Obtaining the member '_cells' of a type (line 320)
        _cells_135964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), self_135963, '_cells')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___135965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), _cells_135964, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_135966 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), getitem___135965, key_135962)
        
        # Obtaining the member 'draw' of a type (line 320)
        draw_135967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), subscript_call_result_135966, 'draw')
        # Calling draw(args, kwargs) (line 320)
        draw_call_result_135970 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), draw_135967, *[renderer_135968], **kwargs_135969)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to close_group(...): (line 322)
        # Processing the call arguments (line 322)
        unicode_135973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 29), 'unicode', u'table')
        # Processing the call keyword arguments (line 322)
        kwargs_135974 = {}
        # Getting the type of 'renderer' (line 322)
        renderer_135971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'renderer', False)
        # Obtaining the member 'close_group' of a type (line 322)
        close_group_135972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), renderer_135971, 'close_group')
        # Calling close_group(args, kwargs) (line 322)
        close_group_call_result_135975 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), close_group_135972, *[unicode_135973], **kwargs_135974)
        
        
        # Assigning a Name to a Attribute (line 323):
        
        # Assigning a Name to a Attribute (line 323):
        # Getting the type of 'False' (line 323)
        False_135976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 21), 'False')
        # Getting the type of 'self' (line 323)
        self_135977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 323)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), self_135977, 'stale', False_135976)
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 305)
        stypy_return_type_135978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_135978)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_135978


    @norecursion
    def _get_grid_bbox(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_grid_bbox'
        module_type_store = module_type_store.open_function_context('_get_grid_bbox', 325, 4, False)
        # Assigning a type to the variable 'self' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table._get_grid_bbox.__dict__.__setitem__('stypy_localization', localization)
        Table._get_grid_bbox.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table._get_grid_bbox.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table._get_grid_bbox.__dict__.__setitem__('stypy_function_name', 'Table._get_grid_bbox')
        Table._get_grid_bbox.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Table._get_grid_bbox.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table._get_grid_bbox.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table._get_grid_bbox.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table._get_grid_bbox.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table._get_grid_bbox.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table._get_grid_bbox.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table._get_grid_bbox', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_grid_bbox', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_grid_bbox(...)' code ##################

        unicode_135979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, (-1)), 'unicode', u'Get a bbox, in axes co-ordinates for the cells.\n\n        Only include those in the range (0,0) to (maxRow, maxCol)')
        
        # Assigning a ListComp to a Name (line 329):
        
        # Assigning a ListComp to a Name (line 329):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to iteritems(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'self' (line 330)
        self_135994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 55), 'self', False)
        # Obtaining the member '_cells' of a type (line 330)
        _cells_135995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 55), self_135994, '_cells')
        # Processing the call keyword arguments (line 330)
        kwargs_135996 = {}
        # Getting the type of 'six' (line 330)
        six_135992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 41), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 330)
        iteritems_135993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 41), six_135992, 'iteritems')
        # Calling iteritems(args, kwargs) (line 330)
        iteritems_call_result_135997 = invoke(stypy.reporting.localization.Localization(__file__, 330, 41), iteritems_135993, *[_cells_135995], **kwargs_135996)
        
        comprehension_135998 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 17), iteritems_call_result_135997)
        # Assigning a type to the variable 'tuple_135999' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 17), 'tuple_135999', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 17), comprehension_135998))
        
        # Obtaining an instance of the builtin type 'tuple' (line 330)
        tuple_135999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 330)
        # Adding element type (line 330)row
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 22), tuple_135999, )
        # Adding element type (line 330)col
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 22), tuple_135999, )
        
        # Assigning a type to the variable 'tuple_135999' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 17), 'tuple_135999', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 17), comprehension_135998))
        # Assigning a type to the variable 'cell' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 17), 'cell', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 17), comprehension_135998))
        
        # Evaluating a boolean operation
        
        # Getting the type of 'row' (line 331)
        row_135985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'row')
        int_135986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 27), 'int')
        # Applying the binary operator '>=' (line 331)
        result_ge_135987 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 20), '>=', row_135985, int_135986)
        
        
        # Getting the type of 'col' (line 331)
        col_135988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 33), 'col')
        int_135989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 40), 'int')
        # Applying the binary operator '>=' (line 331)
        result_ge_135990 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 33), '>=', col_135988, int_135989)
        
        # Applying the binary operator 'and' (line 331)
        result_and_keyword_135991 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 20), 'and', result_ge_135987, result_ge_135990)
        
        
        # Call to get_window_extent(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'renderer' (line 329)
        renderer_135982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 40), 'renderer', False)
        # Processing the call keyword arguments (line 329)
        kwargs_135983 = {}
        # Getting the type of 'cell' (line 329)
        cell_135980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 17), 'cell', False)
        # Obtaining the member 'get_window_extent' of a type (line 329)
        get_window_extent_135981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 17), cell_135980, 'get_window_extent')
        # Calling get_window_extent(args, kwargs) (line 329)
        get_window_extent_call_result_135984 = invoke(stypy.reporting.localization.Localization(__file__, 329, 17), get_window_extent_135981, *[renderer_135982], **kwargs_135983)
        
        list_136000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 17), list_136000, get_window_extent_call_result_135984)
        # Assigning a type to the variable 'boxes' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'boxes', list_136000)
        
        # Assigning a Call to a Name (line 332):
        
        # Assigning a Call to a Name (line 332):
        
        # Call to union(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'boxes' (line 332)
        boxes_136003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 26), 'boxes', False)
        # Processing the call keyword arguments (line 332)
        kwargs_136004 = {}
        # Getting the type of 'Bbox' (line 332)
        Bbox_136001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 15), 'Bbox', False)
        # Obtaining the member 'union' of a type (line 332)
        union_136002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 15), Bbox_136001, 'union')
        # Calling union(args, kwargs) (line 332)
        union_call_result_136005 = invoke(stypy.reporting.localization.Localization(__file__, 332, 15), union_136002, *[boxes_136003], **kwargs_136004)
        
        # Assigning a type to the variable 'bbox' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'bbox', union_call_result_136005)
        
        # Call to inverse_transformed(...): (line 333)
        # Processing the call arguments (line 333)
        
        # Call to get_transform(...): (line 333)
        # Processing the call keyword arguments (line 333)
        kwargs_136010 = {}
        # Getting the type of 'self' (line 333)
        self_136008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 40), 'self', False)
        # Obtaining the member 'get_transform' of a type (line 333)
        get_transform_136009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 40), self_136008, 'get_transform')
        # Calling get_transform(args, kwargs) (line 333)
        get_transform_call_result_136011 = invoke(stypy.reporting.localization.Localization(__file__, 333, 40), get_transform_136009, *[], **kwargs_136010)
        
        # Processing the call keyword arguments (line 333)
        kwargs_136012 = {}
        # Getting the type of 'bbox' (line 333)
        bbox_136006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 15), 'bbox', False)
        # Obtaining the member 'inverse_transformed' of a type (line 333)
        inverse_transformed_136007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 15), bbox_136006, 'inverse_transformed')
        # Calling inverse_transformed(args, kwargs) (line 333)
        inverse_transformed_call_result_136013 = invoke(stypy.reporting.localization.Localization(__file__, 333, 15), inverse_transformed_136007, *[get_transform_call_result_136011], **kwargs_136012)
        
        # Assigning a type to the variable 'stypy_return_type' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'stypy_return_type', inverse_transformed_call_result_136013)
        
        # ################# End of '_get_grid_bbox(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_grid_bbox' in the type store
        # Getting the type of 'stypy_return_type' (line 325)
        stypy_return_type_136014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136014)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_grid_bbox'
        return stypy_return_type_136014


    @norecursion
    def contains(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'contains'
        module_type_store = module_type_store.open_function_context('contains', 335, 4, False)
        # Assigning a type to the variable 'self' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.contains.__dict__.__setitem__('stypy_localization', localization)
        Table.contains.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.contains.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.contains.__dict__.__setitem__('stypy_function_name', 'Table.contains')
        Table.contains.__dict__.__setitem__('stypy_param_names_list', ['mouseevent'])
        Table.contains.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table.contains.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table.contains.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.contains.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.contains.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.contains.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.contains', ['mouseevent'], None, None, defaults, varargs, kwargs)

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

        unicode_136015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, (-1)), 'unicode', u'Test whether the mouse event occurred in the table.\n\n        Returns T/F, {}\n        ')
        
        
        # Call to callable(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'self' (line 340)
        self_136017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 20), 'self', False)
        # Obtaining the member '_contains' of a type (line 340)
        _contains_136018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 20), self_136017, '_contains')
        # Processing the call keyword arguments (line 340)
        kwargs_136019 = {}
        # Getting the type of 'callable' (line 340)
        callable_136016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 11), 'callable', False)
        # Calling callable(args, kwargs) (line 340)
        callable_call_result_136020 = invoke(stypy.reporting.localization.Localization(__file__, 340, 11), callable_136016, *[_contains_136018], **kwargs_136019)
        
        # Testing the type of an if condition (line 340)
        if_condition_136021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 8), callable_call_result_136020)
        # Assigning a type to the variable 'if_condition_136021' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'if_condition_136021', if_condition_136021)
        # SSA begins for if statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _contains(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'self' (line 341)
        self_136024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 34), 'self', False)
        # Getting the type of 'mouseevent' (line 341)
        mouseevent_136025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 40), 'mouseevent', False)
        # Processing the call keyword arguments (line 341)
        kwargs_136026 = {}
        # Getting the type of 'self' (line 341)
        self_136022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 19), 'self', False)
        # Obtaining the member '_contains' of a type (line 341)
        _contains_136023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 19), self_136022, '_contains')
        # Calling _contains(args, kwargs) (line 341)
        _contains_call_result_136027 = invoke(stypy.reporting.localization.Localization(__file__, 341, 19), _contains_136023, *[self_136024, mouseevent_136025], **kwargs_136026)
        
        # Assigning a type to the variable 'stypy_return_type' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'stypy_return_type', _contains_call_result_136027)
        # SSA join for if statement (line 340)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 345):
        
        # Assigning a Attribute to a Name (line 345):
        # Getting the type of 'self' (line 345)
        self_136028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'self')
        # Obtaining the member 'figure' of a type (line 345)
        figure_136029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 19), self_136028, 'figure')
        # Obtaining the member '_cachedRenderer' of a type (line 345)
        _cachedRenderer_136030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 19), figure_136029, '_cachedRenderer')
        # Assigning a type to the variable 'renderer' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'renderer', _cachedRenderer_136030)
        
        # Type idiom detected: calculating its left and rigth part (line 346)
        # Getting the type of 'renderer' (line 346)
        renderer_136031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'renderer')
        # Getting the type of 'None' (line 346)
        None_136032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 27), 'None')
        
        (may_be_136033, more_types_in_union_136034) = may_not_be_none(renderer_136031, None_136032)

        if may_be_136033:

            if more_types_in_union_136034:
                # Runtime conditional SSA (line 346)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a ListComp to a Name (line 347):
            
            # Assigning a ListComp to a Name (line 347):
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to iteritems(...): (line 348)
            # Processing the call arguments (line 348)
            # Getting the type of 'self' (line 348)
            self_136049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 59), 'self', False)
            # Obtaining the member '_cells' of a type (line 348)
            _cells_136050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 59), self_136049, '_cells')
            # Processing the call keyword arguments (line 348)
            kwargs_136051 = {}
            # Getting the type of 'six' (line 348)
            six_136047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 45), 'six', False)
            # Obtaining the member 'iteritems' of a type (line 348)
            iteritems_136048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 45), six_136047, 'iteritems')
            # Calling iteritems(args, kwargs) (line 348)
            iteritems_call_result_136052 = invoke(stypy.reporting.localization.Localization(__file__, 348, 45), iteritems_136048, *[_cells_136050], **kwargs_136051)
            
            comprehension_136053 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 21), iteritems_call_result_136052)
            # Assigning a type to the variable 'tuple_136054' (line 347)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 21), 'tuple_136054', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 21), comprehension_136053))
            
            # Obtaining an instance of the builtin type 'tuple' (line 348)
            tuple_136054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 26), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 348)
            # Adding element type (line 348)row
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 26), tuple_136054, )
            # Adding element type (line 348)col
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 26), tuple_136054, )
            
            # Assigning a type to the variable 'tuple_136054' (line 347)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 21), 'tuple_136054', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 21), comprehension_136053))
            # Assigning a type to the variable 'cell' (line 347)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 21), 'cell', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 21), comprehension_136053))
            
            # Evaluating a boolean operation
            
            # Getting the type of 'row' (line 349)
            row_136040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 24), 'row')
            int_136041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 31), 'int')
            # Applying the binary operator '>=' (line 349)
            result_ge_136042 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 24), '>=', row_136040, int_136041)
            
            
            # Getting the type of 'col' (line 349)
            col_136043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 37), 'col')
            int_136044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 44), 'int')
            # Applying the binary operator '>=' (line 349)
            result_ge_136045 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 37), '>=', col_136043, int_136044)
            
            # Applying the binary operator 'and' (line 349)
            result_and_keyword_136046 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 24), 'and', result_ge_136042, result_ge_136045)
            
            
            # Call to get_window_extent(...): (line 347)
            # Processing the call arguments (line 347)
            # Getting the type of 'renderer' (line 347)
            renderer_136037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 44), 'renderer', False)
            # Processing the call keyword arguments (line 347)
            kwargs_136038 = {}
            # Getting the type of 'cell' (line 347)
            cell_136035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 21), 'cell', False)
            # Obtaining the member 'get_window_extent' of a type (line 347)
            get_window_extent_136036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 21), cell_136035, 'get_window_extent')
            # Calling get_window_extent(args, kwargs) (line 347)
            get_window_extent_call_result_136039 = invoke(stypy.reporting.localization.Localization(__file__, 347, 21), get_window_extent_136036, *[renderer_136037], **kwargs_136038)
            
            list_136055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 21), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 21), list_136055, get_window_extent_call_result_136039)
            # Assigning a type to the variable 'boxes' (line 347)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'boxes', list_136055)
            
            # Assigning a Call to a Name (line 350):
            
            # Assigning a Call to a Name (line 350):
            
            # Call to union(...): (line 350)
            # Processing the call arguments (line 350)
            # Getting the type of 'boxes' (line 350)
            boxes_136058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 30), 'boxes', False)
            # Processing the call keyword arguments (line 350)
            kwargs_136059 = {}
            # Getting the type of 'Bbox' (line 350)
            Bbox_136056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 'Bbox', False)
            # Obtaining the member 'union' of a type (line 350)
            union_136057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 19), Bbox_136056, 'union')
            # Calling union(args, kwargs) (line 350)
            union_call_result_136060 = invoke(stypy.reporting.localization.Localization(__file__, 350, 19), union_136057, *[boxes_136058], **kwargs_136059)
            
            # Assigning a type to the variable 'bbox' (line 350)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'bbox', union_call_result_136060)
            
            # Obtaining an instance of the builtin type 'tuple' (line 351)
            tuple_136061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 351)
            # Adding element type (line 351)
            
            # Call to contains(...): (line 351)
            # Processing the call arguments (line 351)
            # Getting the type of 'mouseevent' (line 351)
            mouseevent_136064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 33), 'mouseevent', False)
            # Obtaining the member 'x' of a type (line 351)
            x_136065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 33), mouseevent_136064, 'x')
            # Getting the type of 'mouseevent' (line 351)
            mouseevent_136066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 47), 'mouseevent', False)
            # Obtaining the member 'y' of a type (line 351)
            y_136067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 47), mouseevent_136066, 'y')
            # Processing the call keyword arguments (line 351)
            kwargs_136068 = {}
            # Getting the type of 'bbox' (line 351)
            bbox_136062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 19), 'bbox', False)
            # Obtaining the member 'contains' of a type (line 351)
            contains_136063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 19), bbox_136062, 'contains')
            # Calling contains(args, kwargs) (line 351)
            contains_call_result_136069 = invoke(stypy.reporting.localization.Localization(__file__, 351, 19), contains_136063, *[x_136065, y_136067], **kwargs_136068)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 19), tuple_136061, contains_call_result_136069)
            # Adding element type (line 351)
            
            # Obtaining an instance of the builtin type 'dict' (line 351)
            dict_136070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 62), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 351)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 19), tuple_136061, dict_136070)
            
            # Assigning a type to the variable 'stypy_return_type' (line 351)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'stypy_return_type', tuple_136061)

            if more_types_in_union_136034:
                # Runtime conditional SSA for else branch (line 346)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_136033) or more_types_in_union_136034):
            
            # Obtaining an instance of the builtin type 'tuple' (line 353)
            tuple_136071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 353)
            # Adding element type (line 353)
            # Getting the type of 'False' (line 353)
            False_136072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 19), 'False')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 19), tuple_136071, False_136072)
            # Adding element type (line 353)
            
            # Obtaining an instance of the builtin type 'dict' (line 353)
            dict_136073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 26), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 353)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 19), tuple_136071, dict_136073)
            
            # Assigning a type to the variable 'stypy_return_type' (line 353)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'stypy_return_type', tuple_136071)

            if (may_be_136033 and more_types_in_union_136034):
                # SSA join for if statement (line 346)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'contains(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'contains' in the type store
        # Getting the type of 'stypy_return_type' (line 335)
        stypy_return_type_136074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136074)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'contains'
        return stypy_return_type_136074


    @norecursion
    def get_children(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_children'
        module_type_store = module_type_store.open_function_context('get_children', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.get_children.__dict__.__setitem__('stypy_localization', localization)
        Table.get_children.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.get_children.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.get_children.__dict__.__setitem__('stypy_function_name', 'Table.get_children')
        Table.get_children.__dict__.__setitem__('stypy_param_names_list', [])
        Table.get_children.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table.get_children.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table.get_children.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.get_children.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.get_children.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.get_children.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.get_children', [], None, None, defaults, varargs, kwargs)

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

        unicode_136075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 8), 'unicode', u'Return the Artists contained by the table')
        
        # Call to list(...): (line 357)
        # Processing the call arguments (line 357)
        
        # Call to itervalues(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'self' (line 357)
        self_136079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 35), 'self', False)
        # Obtaining the member '_cells' of a type (line 357)
        _cells_136080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 35), self_136079, '_cells')
        # Processing the call keyword arguments (line 357)
        kwargs_136081 = {}
        # Getting the type of 'six' (line 357)
        six_136077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'six', False)
        # Obtaining the member 'itervalues' of a type (line 357)
        itervalues_136078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 20), six_136077, 'itervalues')
        # Calling itervalues(args, kwargs) (line 357)
        itervalues_call_result_136082 = invoke(stypy.reporting.localization.Localization(__file__, 357, 20), itervalues_136078, *[_cells_136080], **kwargs_136081)
        
        # Processing the call keyword arguments (line 357)
        kwargs_136083 = {}
        # Getting the type of 'list' (line 357)
        list_136076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 15), 'list', False)
        # Calling list(args, kwargs) (line 357)
        list_call_result_136084 = invoke(stypy.reporting.localization.Localization(__file__, 357, 15), list_136076, *[itervalues_call_result_136082], **kwargs_136083)
        
        # Assigning a type to the variable 'stypy_return_type' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'stypy_return_type', list_call_result_136084)
        
        # ################# End of 'get_children(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_children' in the type store
        # Getting the type of 'stypy_return_type' (line 355)
        stypy_return_type_136085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136085)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_children'
        return stypy_return_type_136085

    
    # Assigning a Name to a Name (line 358):

    @norecursion
    def get_window_extent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_window_extent'
        module_type_store = module_type_store.open_function_context('get_window_extent', 360, 4, False)
        # Assigning a type to the variable 'self' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.get_window_extent.__dict__.__setitem__('stypy_localization', localization)
        Table.get_window_extent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.get_window_extent.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.get_window_extent.__dict__.__setitem__('stypy_function_name', 'Table.get_window_extent')
        Table.get_window_extent.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Table.get_window_extent.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table.get_window_extent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table.get_window_extent.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.get_window_extent.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.get_window_extent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.get_window_extent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.get_window_extent', ['renderer'], None, None, defaults, varargs, kwargs)

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

        unicode_136086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 8), 'unicode', u'Return the bounding box of the table in window coords')
        
        # Assigning a ListComp to a Name (line 362):
        
        # Assigning a ListComp to a Name (line 362):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to itervalues(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'self' (line 363)
        self_136094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 44), 'self', False)
        # Obtaining the member '_cells' of a type (line 363)
        _cells_136095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 44), self_136094, '_cells')
        # Processing the call keyword arguments (line 363)
        kwargs_136096 = {}
        # Getting the type of 'six' (line 363)
        six_136092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 29), 'six', False)
        # Obtaining the member 'itervalues' of a type (line 363)
        itervalues_136093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 29), six_136092, 'itervalues')
        # Calling itervalues(args, kwargs) (line 363)
        itervalues_call_result_136097 = invoke(stypy.reporting.localization.Localization(__file__, 363, 29), itervalues_136093, *[_cells_136095], **kwargs_136096)
        
        comprehension_136098 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 17), itervalues_call_result_136097)
        # Assigning a type to the variable 'cell' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 17), 'cell', comprehension_136098)
        
        # Call to get_window_extent(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'renderer' (line 362)
        renderer_136089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 40), 'renderer', False)
        # Processing the call keyword arguments (line 362)
        kwargs_136090 = {}
        # Getting the type of 'cell' (line 362)
        cell_136087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 17), 'cell', False)
        # Obtaining the member 'get_window_extent' of a type (line 362)
        get_window_extent_136088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 17), cell_136087, 'get_window_extent')
        # Calling get_window_extent(args, kwargs) (line 362)
        get_window_extent_call_result_136091 = invoke(stypy.reporting.localization.Localization(__file__, 362, 17), get_window_extent_136088, *[renderer_136089], **kwargs_136090)
        
        list_136099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 17), list_136099, get_window_extent_call_result_136091)
        # Assigning a type to the variable 'boxes' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'boxes', list_136099)
        
        # Call to union(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'boxes' (line 364)
        boxes_136102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 26), 'boxes', False)
        # Processing the call keyword arguments (line 364)
        kwargs_136103 = {}
        # Getting the type of 'Bbox' (line 364)
        Bbox_136100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 15), 'Bbox', False)
        # Obtaining the member 'union' of a type (line 364)
        union_136101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 15), Bbox_136100, 'union')
        # Calling union(args, kwargs) (line 364)
        union_call_result_136104 = invoke(stypy.reporting.localization.Localization(__file__, 364, 15), union_136101, *[boxes_136102], **kwargs_136103)
        
        # Assigning a type to the variable 'stypy_return_type' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'stypy_return_type', union_call_result_136104)
        
        # ################# End of 'get_window_extent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_window_extent' in the type store
        # Getting the type of 'stypy_return_type' (line 360)
        stypy_return_type_136105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136105)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_window_extent'
        return stypy_return_type_136105


    @norecursion
    def _do_cell_alignment(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_do_cell_alignment'
        module_type_store = module_type_store.open_function_context('_do_cell_alignment', 366, 4, False)
        # Assigning a type to the variable 'self' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table._do_cell_alignment.__dict__.__setitem__('stypy_localization', localization)
        Table._do_cell_alignment.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table._do_cell_alignment.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table._do_cell_alignment.__dict__.__setitem__('stypy_function_name', 'Table._do_cell_alignment')
        Table._do_cell_alignment.__dict__.__setitem__('stypy_param_names_list', [])
        Table._do_cell_alignment.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table._do_cell_alignment.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table._do_cell_alignment.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table._do_cell_alignment.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table._do_cell_alignment.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table._do_cell_alignment.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table._do_cell_alignment', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_do_cell_alignment', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_do_cell_alignment(...)' code ##################

        unicode_136106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, (-1)), 'unicode', u' Calculate row heights and column widths.\n\n        Position cells accordingly.\n        ')
        
        # Assigning a Dict to a Name (line 372):
        
        # Assigning a Dict to a Name (line 372):
        
        # Obtaining an instance of the builtin type 'dict' (line 372)
        dict_136107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 17), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 372)
        
        # Assigning a type to the variable 'widths' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'widths', dict_136107)
        
        # Assigning a Dict to a Name (line 373):
        
        # Assigning a Dict to a Name (line 373):
        
        # Obtaining an instance of the builtin type 'dict' (line 373)
        dict_136108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 373)
        
        # Assigning a type to the variable 'heights' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'heights', dict_136108)
        
        
        # Call to iteritems(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'self' (line 374)
        self_136111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 46), 'self', False)
        # Obtaining the member '_cells' of a type (line 374)
        _cells_136112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 46), self_136111, '_cells')
        # Processing the call keyword arguments (line 374)
        kwargs_136113 = {}
        # Getting the type of 'six' (line 374)
        six_136109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 32), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 374)
        iteritems_136110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 32), six_136109, 'iteritems')
        # Calling iteritems(args, kwargs) (line 374)
        iteritems_call_result_136114 = invoke(stypy.reporting.localization.Localization(__file__, 374, 32), iteritems_136110, *[_cells_136112], **kwargs_136113)
        
        # Testing the type of a for loop iterable (line 374)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 374, 8), iteritems_call_result_136114)
        # Getting the type of the for loop variable (line 374)
        for_loop_var_136115 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 374, 8), iteritems_call_result_136114)
        # Assigning a type to the variable 'row' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 8), for_loop_var_136115))
        # Assigning a type to the variable 'col' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 8), for_loop_var_136115))
        # Assigning a type to the variable 'cell' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'cell', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 8), for_loop_var_136115))
        # SSA begins for a for statement (line 374)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 375):
        
        # Assigning a Call to a Name (line 375):
        
        # Call to setdefault(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 'row' (line 375)
        row_136118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 40), 'row', False)
        float_136119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 45), 'float')
        # Processing the call keyword arguments (line 375)
        kwargs_136120 = {}
        # Getting the type of 'heights' (line 375)
        heights_136116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 21), 'heights', False)
        # Obtaining the member 'setdefault' of a type (line 375)
        setdefault_136117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 21), heights_136116, 'setdefault')
        # Calling setdefault(args, kwargs) (line 375)
        setdefault_call_result_136121 = invoke(stypy.reporting.localization.Localization(__file__, 375, 21), setdefault_136117, *[row_136118, float_136119], **kwargs_136120)
        
        # Assigning a type to the variable 'height' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'height', setdefault_call_result_136121)
        
        # Assigning a Call to a Subscript (line 376):
        
        # Assigning a Call to a Subscript (line 376):
        
        # Call to max(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'height' (line 376)
        height_136123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 31), 'height', False)
        
        # Call to get_height(...): (line 376)
        # Processing the call keyword arguments (line 376)
        kwargs_136126 = {}
        # Getting the type of 'cell' (line 376)
        cell_136124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 39), 'cell', False)
        # Obtaining the member 'get_height' of a type (line 376)
        get_height_136125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 39), cell_136124, 'get_height')
        # Calling get_height(args, kwargs) (line 376)
        get_height_call_result_136127 = invoke(stypy.reporting.localization.Localization(__file__, 376, 39), get_height_136125, *[], **kwargs_136126)
        
        # Processing the call keyword arguments (line 376)
        kwargs_136128 = {}
        # Getting the type of 'max' (line 376)
        max_136122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 27), 'max', False)
        # Calling max(args, kwargs) (line 376)
        max_call_result_136129 = invoke(stypy.reporting.localization.Localization(__file__, 376, 27), max_136122, *[height_136123, get_height_call_result_136127], **kwargs_136128)
        
        # Getting the type of 'heights' (line 376)
        heights_136130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'heights')
        # Getting the type of 'row' (line 376)
        row_136131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 20), 'row')
        # Storing an element on a container (line 376)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 12), heights_136130, (row_136131, max_call_result_136129))
        
        # Assigning a Call to a Name (line 377):
        
        # Assigning a Call to a Name (line 377):
        
        # Call to setdefault(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'col' (line 377)
        col_136134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 38), 'col', False)
        float_136135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 43), 'float')
        # Processing the call keyword arguments (line 377)
        kwargs_136136 = {}
        # Getting the type of 'widths' (line 377)
        widths_136132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 20), 'widths', False)
        # Obtaining the member 'setdefault' of a type (line 377)
        setdefault_136133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 20), widths_136132, 'setdefault')
        # Calling setdefault(args, kwargs) (line 377)
        setdefault_call_result_136137 = invoke(stypy.reporting.localization.Localization(__file__, 377, 20), setdefault_136133, *[col_136134, float_136135], **kwargs_136136)
        
        # Assigning a type to the variable 'width' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'width', setdefault_call_result_136137)
        
        # Assigning a Call to a Subscript (line 378):
        
        # Assigning a Call to a Subscript (line 378):
        
        # Call to max(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'width' (line 378)
        width_136139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 30), 'width', False)
        
        # Call to get_width(...): (line 378)
        # Processing the call keyword arguments (line 378)
        kwargs_136142 = {}
        # Getting the type of 'cell' (line 378)
        cell_136140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 37), 'cell', False)
        # Obtaining the member 'get_width' of a type (line 378)
        get_width_136141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 37), cell_136140, 'get_width')
        # Calling get_width(args, kwargs) (line 378)
        get_width_call_result_136143 = invoke(stypy.reporting.localization.Localization(__file__, 378, 37), get_width_136141, *[], **kwargs_136142)
        
        # Processing the call keyword arguments (line 378)
        kwargs_136144 = {}
        # Getting the type of 'max' (line 378)
        max_136138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 26), 'max', False)
        # Calling max(args, kwargs) (line 378)
        max_call_result_136145 = invoke(stypy.reporting.localization.Localization(__file__, 378, 26), max_136138, *[width_136139, get_width_call_result_136143], **kwargs_136144)
        
        # Getting the type of 'widths' (line 378)
        widths_136146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'widths')
        # Getting the type of 'col' (line 378)
        col_136147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 19), 'col')
        # Storing an element on a container (line 378)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 12), widths_136146, (col_136147, max_call_result_136145))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 381):
        
        # Assigning a Num to a Name (line 381):
        int_136148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 15), 'int')
        # Assigning a type to the variable 'xpos' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'xpos', int_136148)
        
        # Assigning a Dict to a Name (line 382):
        
        # Assigning a Dict to a Name (line 382):
        
        # Obtaining an instance of the builtin type 'dict' (line 382)
        dict_136149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 382)
        
        # Assigning a type to the variable 'lefts' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'lefts', dict_136149)
        
        
        # Call to sorted(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'widths' (line 383)
        widths_136151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 26), 'widths', False)
        # Processing the call keyword arguments (line 383)
        kwargs_136152 = {}
        # Getting the type of 'sorted' (line 383)
        sorted_136150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'sorted', False)
        # Calling sorted(args, kwargs) (line 383)
        sorted_call_result_136153 = invoke(stypy.reporting.localization.Localization(__file__, 383, 19), sorted_136150, *[widths_136151], **kwargs_136152)
        
        # Testing the type of a for loop iterable (line 383)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 383, 8), sorted_call_result_136153)
        # Getting the type of the for loop variable (line 383)
        for_loop_var_136154 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 383, 8), sorted_call_result_136153)
        # Assigning a type to the variable 'col' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'col', for_loop_var_136154)
        # SSA begins for a for statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 384):
        
        # Assigning a Name to a Subscript (line 384):
        # Getting the type of 'xpos' (line 384)
        xpos_136155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 25), 'xpos')
        # Getting the type of 'lefts' (line 384)
        lefts_136156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'lefts')
        # Getting the type of 'col' (line 384)
        col_136157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 18), 'col')
        # Storing an element on a container (line 384)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 12), lefts_136156, (col_136157, xpos_136155))
        
        # Getting the type of 'xpos' (line 385)
        xpos_136158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'xpos')
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 385)
        col_136159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 27), 'col')
        # Getting the type of 'widths' (line 385)
        widths_136160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'widths')
        # Obtaining the member '__getitem__' of a type (line 385)
        getitem___136161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 20), widths_136160, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 385)
        subscript_call_result_136162 = invoke(stypy.reporting.localization.Localization(__file__, 385, 20), getitem___136161, col_136159)
        
        # Applying the binary operator '+=' (line 385)
        result_iadd_136163 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 12), '+=', xpos_136158, subscript_call_result_136162)
        # Assigning a type to the variable 'xpos' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'xpos', result_iadd_136163)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Name (line 387):
        
        # Assigning a Num to a Name (line 387):
        int_136164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 15), 'int')
        # Assigning a type to the variable 'ypos' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'ypos', int_136164)
        
        # Assigning a Dict to a Name (line 388):
        
        # Assigning a Dict to a Name (line 388):
        
        # Obtaining an instance of the builtin type 'dict' (line 388)
        dict_136165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 388)
        
        # Assigning a type to the variable 'bottoms' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'bottoms', dict_136165)
        
        
        # Call to sorted(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'heights' (line 389)
        heights_136167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 26), 'heights', False)
        # Processing the call keyword arguments (line 389)
        # Getting the type of 'True' (line 389)
        True_136168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 43), 'True', False)
        keyword_136169 = True_136168
        kwargs_136170 = {'reverse': keyword_136169}
        # Getting the type of 'sorted' (line 389)
        sorted_136166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 'sorted', False)
        # Calling sorted(args, kwargs) (line 389)
        sorted_call_result_136171 = invoke(stypy.reporting.localization.Localization(__file__, 389, 19), sorted_136166, *[heights_136167], **kwargs_136170)
        
        # Testing the type of a for loop iterable (line 389)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 389, 8), sorted_call_result_136171)
        # Getting the type of the for loop variable (line 389)
        for_loop_var_136172 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 389, 8), sorted_call_result_136171)
        # Assigning a type to the variable 'row' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'row', for_loop_var_136172)
        # SSA begins for a for statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 390):
        
        # Assigning a Name to a Subscript (line 390):
        # Getting the type of 'ypos' (line 390)
        ypos_136173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 27), 'ypos')
        # Getting the type of 'bottoms' (line 390)
        bottoms_136174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'bottoms')
        # Getting the type of 'row' (line 390)
        row_136175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 20), 'row')
        # Storing an element on a container (line 390)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 12), bottoms_136174, (row_136175, ypos_136173))
        
        # Getting the type of 'ypos' (line 391)
        ypos_136176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'ypos')
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 391)
        row_136177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 28), 'row')
        # Getting the type of 'heights' (line 391)
        heights_136178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'heights')
        # Obtaining the member '__getitem__' of a type (line 391)
        getitem___136179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 20), heights_136178, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 391)
        subscript_call_result_136180 = invoke(stypy.reporting.localization.Localization(__file__, 391, 20), getitem___136179, row_136177)
        
        # Applying the binary operator '+=' (line 391)
        result_iadd_136181 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 12), '+=', ypos_136176, subscript_call_result_136180)
        # Assigning a type to the variable 'ypos' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'ypos', result_iadd_136181)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to iteritems(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'self' (line 394)
        self_136184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 46), 'self', False)
        # Obtaining the member '_cells' of a type (line 394)
        _cells_136185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 46), self_136184, '_cells')
        # Processing the call keyword arguments (line 394)
        kwargs_136186 = {}
        # Getting the type of 'six' (line 394)
        six_136182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 32), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 394)
        iteritems_136183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 32), six_136182, 'iteritems')
        # Calling iteritems(args, kwargs) (line 394)
        iteritems_call_result_136187 = invoke(stypy.reporting.localization.Localization(__file__, 394, 32), iteritems_136183, *[_cells_136185], **kwargs_136186)
        
        # Testing the type of a for loop iterable (line 394)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 394, 8), iteritems_call_result_136187)
        # Getting the type of the for loop variable (line 394)
        for_loop_var_136188 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 394, 8), iteritems_call_result_136187)
        # Assigning a type to the variable 'row' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 8), for_loop_var_136188))
        # Assigning a type to the variable 'col' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 8), for_loop_var_136188))
        # Assigning a type to the variable 'cell' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'cell', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 8), for_loop_var_136188))
        # SSA begins for a for statement (line 394)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_x(...): (line 395)
        # Processing the call arguments (line 395)
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 395)
        col_136191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 29), 'col', False)
        # Getting the type of 'lefts' (line 395)
        lefts_136192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 23), 'lefts', False)
        # Obtaining the member '__getitem__' of a type (line 395)
        getitem___136193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 23), lefts_136192, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 395)
        subscript_call_result_136194 = invoke(stypy.reporting.localization.Localization(__file__, 395, 23), getitem___136193, col_136191)
        
        # Processing the call keyword arguments (line 395)
        kwargs_136195 = {}
        # Getting the type of 'cell' (line 395)
        cell_136189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'cell', False)
        # Obtaining the member 'set_x' of a type (line 395)
        set_x_136190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), cell_136189, 'set_x')
        # Calling set_x(args, kwargs) (line 395)
        set_x_call_result_136196 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), set_x_136190, *[subscript_call_result_136194], **kwargs_136195)
        
        
        # Call to set_y(...): (line 396)
        # Processing the call arguments (line 396)
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 396)
        row_136199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 31), 'row', False)
        # Getting the type of 'bottoms' (line 396)
        bottoms_136200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 23), 'bottoms', False)
        # Obtaining the member '__getitem__' of a type (line 396)
        getitem___136201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 23), bottoms_136200, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 396)
        subscript_call_result_136202 = invoke(stypy.reporting.localization.Localization(__file__, 396, 23), getitem___136201, row_136199)
        
        # Processing the call keyword arguments (line 396)
        kwargs_136203 = {}
        # Getting the type of 'cell' (line 396)
        cell_136197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'cell', False)
        # Obtaining the member 'set_y' of a type (line 396)
        set_y_136198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 12), cell_136197, 'set_y')
        # Calling set_y(args, kwargs) (line 396)
        set_y_call_result_136204 = invoke(stypy.reporting.localization.Localization(__file__, 396, 12), set_y_136198, *[subscript_call_result_136202], **kwargs_136203)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_do_cell_alignment(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_do_cell_alignment' in the type store
        # Getting the type of 'stypy_return_type' (line 366)
        stypy_return_type_136205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136205)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_do_cell_alignment'
        return stypy_return_type_136205


    @norecursion
    def auto_set_column_width(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'auto_set_column_width'
        module_type_store = module_type_store.open_function_context('auto_set_column_width', 398, 4, False)
        # Assigning a type to the variable 'self' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.auto_set_column_width.__dict__.__setitem__('stypy_localization', localization)
        Table.auto_set_column_width.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.auto_set_column_width.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.auto_set_column_width.__dict__.__setitem__('stypy_function_name', 'Table.auto_set_column_width')
        Table.auto_set_column_width.__dict__.__setitem__('stypy_param_names_list', ['col'])
        Table.auto_set_column_width.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table.auto_set_column_width.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table.auto_set_column_width.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.auto_set_column_width.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.auto_set_column_width.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.auto_set_column_width.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.auto_set_column_width', ['col'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'auto_set_column_width', localization, ['col'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'auto_set_column_width(...)' code ##################

        unicode_136206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, (-1)), 'unicode', u' Given column indexs in either List, Tuple or int. Will be able to\n        automatically set the columns into optimal sizes.\n\n        Here is the example of the input, which triger automatic adjustment on\n        columns to optimal size by given index numbers.\n        -1: the row labling\n        0: the 1st column\n        1: the 2nd column\n\n        Args:\n            col(List): list of indexs\n            >>>table.auto_set_column_width([-1,0,1])\n\n            col(Tuple): tuple of indexs\n            >>>table.auto_set_column_width((-1,0,1))\n\n            col(int): index integer\n            >>>table.auto_set_column_width(-1)\n            >>>table.auto_set_column_width(0)\n            >>>table.auto_set_column_width(1)\n        ')
        
        
        # SSA begins for try-except statement (line 421)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to iter(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'col' (line 422)
        col_136208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 17), 'col', False)
        # Processing the call keyword arguments (line 422)
        kwargs_136209 = {}
        # Getting the type of 'iter' (line 422)
        iter_136207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'iter', False)
        # Calling iter(args, kwargs) (line 422)
        iter_call_result_136210 = invoke(stypy.reporting.localization.Localization(__file__, 422, 12), iter_136207, *[col_136208], **kwargs_136209)
        
        # SSA branch for the except part of a try statement (line 421)
        # SSA branch for the except 'Tuple' branch of a try statement (line 421)
        module_type_store.open_ssa_branch('except')
        
        # Call to append(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'col' (line 424)
        col_136214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 37), 'col', False)
        # Processing the call keyword arguments (line 424)
        kwargs_136215 = {}
        # Getting the type of 'self' (line 424)
        self_136211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'self', False)
        # Obtaining the member '_autoColumns' of a type (line 424)
        _autoColumns_136212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), self_136211, '_autoColumns')
        # Obtaining the member 'append' of a type (line 424)
        append_136213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), _autoColumns_136212, 'append')
        # Calling append(args, kwargs) (line 424)
        append_call_result_136216 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), append_136213, *[col_136214], **kwargs_136215)
        
        # SSA branch for the else branch of a try statement (line 421)
        module_type_store.open_ssa_branch('except else')
        
        # Getting the type of 'col' (line 426)
        col_136217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 24), 'col')
        # Testing the type of a for loop iterable (line 426)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 426, 12), col_136217)
        # Getting the type of the for loop variable (line 426)
        for_loop_var_136218 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 426, 12), col_136217)
        # Assigning a type to the variable 'cell' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'cell', for_loop_var_136218)
        # SSA begins for a for statement (line 426)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'cell' (line 427)
        cell_136222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 41), 'cell', False)
        # Processing the call keyword arguments (line 427)
        kwargs_136223 = {}
        # Getting the type of 'self' (line 427)
        self_136219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'self', False)
        # Obtaining the member '_autoColumns' of a type (line 427)
        _autoColumns_136220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 16), self_136219, '_autoColumns')
        # Obtaining the member 'append' of a type (line 427)
        append_136221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 16), _autoColumns_136220, 'append')
        # Calling append(args, kwargs) (line 427)
        append_call_result_136224 = invoke(stypy.reporting.localization.Localization(__file__, 427, 16), append_136221, *[cell_136222], **kwargs_136223)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 421)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 429):
        
        # Assigning a Name to a Attribute (line 429):
        # Getting the type of 'True' (line 429)
        True_136225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 21), 'True')
        # Getting the type of 'self' (line 429)
        self_136226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 429)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 8), self_136226, 'stale', True_136225)
        
        # ################# End of 'auto_set_column_width(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'auto_set_column_width' in the type store
        # Getting the type of 'stypy_return_type' (line 398)
        stypy_return_type_136227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'auto_set_column_width'
        return stypy_return_type_136227


    @norecursion
    def _auto_set_column_width(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_auto_set_column_width'
        module_type_store = module_type_store.open_function_context('_auto_set_column_width', 431, 4, False)
        # Assigning a type to the variable 'self' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table._auto_set_column_width.__dict__.__setitem__('stypy_localization', localization)
        Table._auto_set_column_width.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table._auto_set_column_width.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table._auto_set_column_width.__dict__.__setitem__('stypy_function_name', 'Table._auto_set_column_width')
        Table._auto_set_column_width.__dict__.__setitem__('stypy_param_names_list', ['col', 'renderer'])
        Table._auto_set_column_width.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table._auto_set_column_width.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table._auto_set_column_width.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table._auto_set_column_width.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table._auto_set_column_width.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table._auto_set_column_width.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table._auto_set_column_width', ['col', 'renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_auto_set_column_width', localization, ['col', 'renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_auto_set_column_width(...)' code ##################

        unicode_136228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, (-1)), 'unicode', u' Automagically set width for column.\n        ')
        
        # Assigning a ListComp to a Name (line 434):
        
        # Assigning a ListComp to a Name (line 434):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 434)
        self_136236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 32), 'self')
        # Obtaining the member '_cells' of a type (line 434)
        _cells_136237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 32), self_136236, '_cells')
        comprehension_136238 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 17), _cells_136237)
        # Assigning a type to the variable 'key' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 17), 'key', comprehension_136238)
        
        
        # Obtaining the type of the subscript
        int_136230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 51), 'int')
        # Getting the type of 'key' (line 434)
        key_136231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 47), 'key')
        # Obtaining the member '__getitem__' of a type (line 434)
        getitem___136232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 47), key_136231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 434)
        subscript_call_result_136233 = invoke(stypy.reporting.localization.Localization(__file__, 434, 47), getitem___136232, int_136230)
        
        # Getting the type of 'col' (line 434)
        col_136234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 57), 'col')
        # Applying the binary operator '==' (line 434)
        result_eq_136235 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 47), '==', subscript_call_result_136233, col_136234)
        
        # Getting the type of 'key' (line 434)
        key_136229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 17), 'key')
        list_136239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 17), list_136239, key_136229)
        # Assigning a type to the variable 'cells' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'cells', list_136239)
        
        # Assigning a Num to a Name (line 437):
        
        # Assigning a Num to a Name (line 437):
        int_136240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 16), 'int')
        # Assigning a type to the variable 'width' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'width', int_136240)
        
        # Getting the type of 'cells' (line 438)
        cells_136241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 20), 'cells')
        # Testing the type of a for loop iterable (line 438)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 438, 8), cells_136241)
        # Getting the type of the for loop variable (line 438)
        for_loop_var_136242 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 438, 8), cells_136241)
        # Assigning a type to the variable 'cell' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'cell', for_loop_var_136242)
        # SSA begins for a for statement (line 438)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 439):
        
        # Assigning a Subscript to a Name (line 439):
        
        # Obtaining the type of the subscript
        # Getting the type of 'cell' (line 439)
        cell_136243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 28), 'cell')
        # Getting the type of 'self' (line 439)
        self_136244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'self')
        # Obtaining the member '_cells' of a type (line 439)
        _cells_136245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 16), self_136244, '_cells')
        # Obtaining the member '__getitem__' of a type (line 439)
        getitem___136246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 16), _cells_136245, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 439)
        subscript_call_result_136247 = invoke(stypy.reporting.localization.Localization(__file__, 439, 16), getitem___136246, cell_136243)
        
        # Assigning a type to the variable 'c' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'c', subscript_call_result_136247)
        
        # Assigning a Call to a Name (line 440):
        
        # Assigning a Call to a Name (line 440):
        
        # Call to max(...): (line 440)
        # Processing the call arguments (line 440)
        
        # Call to get_required_width(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'renderer' (line 440)
        renderer_136251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 45), 'renderer', False)
        # Processing the call keyword arguments (line 440)
        kwargs_136252 = {}
        # Getting the type of 'c' (line 440)
        c_136249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 24), 'c', False)
        # Obtaining the member 'get_required_width' of a type (line 440)
        get_required_width_136250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 24), c_136249, 'get_required_width')
        # Calling get_required_width(args, kwargs) (line 440)
        get_required_width_call_result_136253 = invoke(stypy.reporting.localization.Localization(__file__, 440, 24), get_required_width_136250, *[renderer_136251], **kwargs_136252)
        
        # Getting the type of 'width' (line 440)
        width_136254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 56), 'width', False)
        # Processing the call keyword arguments (line 440)
        kwargs_136255 = {}
        # Getting the type of 'max' (line 440)
        max_136248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 20), 'max', False)
        # Calling max(args, kwargs) (line 440)
        max_call_result_136256 = invoke(stypy.reporting.localization.Localization(__file__, 440, 20), max_136248, *[get_required_width_call_result_136253, width_136254], **kwargs_136255)
        
        # Assigning a type to the variable 'width' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'width', max_call_result_136256)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'cells' (line 443)
        cells_136257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 20), 'cells')
        # Testing the type of a for loop iterable (line 443)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 443, 8), cells_136257)
        # Getting the type of the for loop variable (line 443)
        for_loop_var_136258 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 443, 8), cells_136257)
        # Assigning a type to the variable 'cell' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'cell', for_loop_var_136258)
        # SSA begins for a for statement (line 443)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_width(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'width' (line 444)
        width_136265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 40), 'width', False)
        # Processing the call keyword arguments (line 444)
        kwargs_136266 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'cell' (line 444)
        cell_136259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 24), 'cell', False)
        # Getting the type of 'self' (line 444)
        self_136260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'self', False)
        # Obtaining the member '_cells' of a type (line 444)
        _cells_136261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), self_136260, '_cells')
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___136262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), _cells_136261, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 444)
        subscript_call_result_136263 = invoke(stypy.reporting.localization.Localization(__file__, 444, 12), getitem___136262, cell_136259)
        
        # Obtaining the member 'set_width' of a type (line 444)
        set_width_136264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), subscript_call_result_136263, 'set_width')
        # Calling set_width(args, kwargs) (line 444)
        set_width_call_result_136267 = invoke(stypy.reporting.localization.Localization(__file__, 444, 12), set_width_136264, *[width_136265], **kwargs_136266)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_auto_set_column_width(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_auto_set_column_width' in the type store
        # Getting the type of 'stypy_return_type' (line 431)
        stypy_return_type_136268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136268)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_auto_set_column_width'
        return stypy_return_type_136268


    @norecursion
    def auto_set_font_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 446)
        True_136269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 39), 'True')
        defaults = [True_136269]
        # Create a new context for function 'auto_set_font_size'
        module_type_store = module_type_store.open_function_context('auto_set_font_size', 446, 4, False)
        # Assigning a type to the variable 'self' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.auto_set_font_size.__dict__.__setitem__('stypy_localization', localization)
        Table.auto_set_font_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.auto_set_font_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.auto_set_font_size.__dict__.__setitem__('stypy_function_name', 'Table.auto_set_font_size')
        Table.auto_set_font_size.__dict__.__setitem__('stypy_param_names_list', ['value'])
        Table.auto_set_font_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table.auto_set_font_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table.auto_set_font_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.auto_set_font_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.auto_set_font_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.auto_set_font_size.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.auto_set_font_size', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'auto_set_font_size', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'auto_set_font_size(...)' code ##################

        unicode_136270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 8), 'unicode', u' Automatically set font size. ')
        
        # Assigning a Name to a Attribute (line 448):
        
        # Assigning a Name to a Attribute (line 448):
        # Getting the type of 'value' (line 448)
        value_136271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 29), 'value')
        # Getting the type of 'self' (line 448)
        self_136272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'self')
        # Setting the type of the member '_autoFontsize' of a type (line 448)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), self_136272, '_autoFontsize', value_136271)
        
        # Assigning a Name to a Attribute (line 449):
        
        # Assigning a Name to a Attribute (line 449):
        # Getting the type of 'True' (line 449)
        True_136273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 21), 'True')
        # Getting the type of 'self' (line 449)
        self_136274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 449)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), self_136274, 'stale', True_136273)
        
        # ################# End of 'auto_set_font_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'auto_set_font_size' in the type store
        # Getting the type of 'stypy_return_type' (line 446)
        stypy_return_type_136275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136275)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'auto_set_font_size'
        return stypy_return_type_136275


    @norecursion
    def _auto_set_font_size(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_auto_set_font_size'
        module_type_store = module_type_store.open_function_context('_auto_set_font_size', 451, 4, False)
        # Assigning a type to the variable 'self' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table._auto_set_font_size.__dict__.__setitem__('stypy_localization', localization)
        Table._auto_set_font_size.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table._auto_set_font_size.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table._auto_set_font_size.__dict__.__setitem__('stypy_function_name', 'Table._auto_set_font_size')
        Table._auto_set_font_size.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Table._auto_set_font_size.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table._auto_set_font_size.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table._auto_set_font_size.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table._auto_set_font_size.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table._auto_set_font_size.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table._auto_set_font_size.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table._auto_set_font_size', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_auto_set_font_size', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_auto_set_font_size(...)' code ##################

        
        
        
        # Call to len(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'self' (line 453)
        self_136277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 15), 'self', False)
        # Obtaining the member '_cells' of a type (line 453)
        _cells_136278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 15), self_136277, '_cells')
        # Processing the call keyword arguments (line 453)
        kwargs_136279 = {}
        # Getting the type of 'len' (line 453)
        len_136276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 11), 'len', False)
        # Calling len(args, kwargs) (line 453)
        len_call_result_136280 = invoke(stypy.reporting.localization.Localization(__file__, 453, 11), len_136276, *[_cells_136278], **kwargs_136279)
        
        int_136281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 31), 'int')
        # Applying the binary operator '==' (line 453)
        result_eq_136282 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 11), '==', len_call_result_136280, int_136281)
        
        # Testing the type of an if condition (line 453)
        if_condition_136283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 8), result_eq_136282)
        # Assigning a type to the variable 'if_condition_136283' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'if_condition_136283', if_condition_136283)
        # SSA begins for if statement (line 453)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 453)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 455):
        
        # Assigning a Call to a Name (line 455):
        
        # Call to get_fontsize(...): (line 455)
        # Processing the call keyword arguments (line 455)
        kwargs_136297 = {}
        
        # Obtaining the type of the subscript
        int_136284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 53), 'int')
        
        # Call to list(...): (line 455)
        # Processing the call arguments (line 455)
        
        # Call to itervalues(...): (line 455)
        # Processing the call arguments (line 455)
        # Getting the type of 'self' (line 455)
        self_136288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 39), 'self', False)
        # Obtaining the member '_cells' of a type (line 455)
        _cells_136289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 39), self_136288, '_cells')
        # Processing the call keyword arguments (line 455)
        kwargs_136290 = {}
        # Getting the type of 'six' (line 455)
        six_136286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 24), 'six', False)
        # Obtaining the member 'itervalues' of a type (line 455)
        itervalues_136287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 24), six_136286, 'itervalues')
        # Calling itervalues(args, kwargs) (line 455)
        itervalues_call_result_136291 = invoke(stypy.reporting.localization.Localization(__file__, 455, 24), itervalues_136287, *[_cells_136289], **kwargs_136290)
        
        # Processing the call keyword arguments (line 455)
        kwargs_136292 = {}
        # Getting the type of 'list' (line 455)
        list_136285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 19), 'list', False)
        # Calling list(args, kwargs) (line 455)
        list_call_result_136293 = invoke(stypy.reporting.localization.Localization(__file__, 455, 19), list_136285, *[itervalues_call_result_136291], **kwargs_136292)
        
        # Obtaining the member '__getitem__' of a type (line 455)
        getitem___136294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 19), list_call_result_136293, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 455)
        subscript_call_result_136295 = invoke(stypy.reporting.localization.Localization(__file__, 455, 19), getitem___136294, int_136284)
        
        # Obtaining the member 'get_fontsize' of a type (line 455)
        get_fontsize_136296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 19), subscript_call_result_136295, 'get_fontsize')
        # Calling get_fontsize(args, kwargs) (line 455)
        get_fontsize_call_result_136298 = invoke(stypy.reporting.localization.Localization(__file__, 455, 19), get_fontsize_136296, *[], **kwargs_136297)
        
        # Assigning a type to the variable 'fontsize' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'fontsize', get_fontsize_call_result_136298)
        
        # Assigning a List to a Name (line 456):
        
        # Assigning a List to a Name (line 456):
        
        # Obtaining an instance of the builtin type 'list' (line 456)
        list_136299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 456)
        
        # Assigning a type to the variable 'cells' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'cells', list_136299)
        
        
        # Call to iteritems(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'self' (line 457)
        self_136302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 39), 'self', False)
        # Obtaining the member '_cells' of a type (line 457)
        _cells_136303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 39), self_136302, '_cells')
        # Processing the call keyword arguments (line 457)
        kwargs_136304 = {}
        # Getting the type of 'six' (line 457)
        six_136300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 25), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 457)
        iteritems_136301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 25), six_136300, 'iteritems')
        # Calling iteritems(args, kwargs) (line 457)
        iteritems_call_result_136305 = invoke(stypy.reporting.localization.Localization(__file__, 457, 25), iteritems_136301, *[_cells_136303], **kwargs_136304)
        
        # Testing the type of a for loop iterable (line 457)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 457, 8), iteritems_call_result_136305)
        # Getting the type of the for loop variable (line 457)
        for_loop_var_136306 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 457, 8), iteritems_call_result_136305)
        # Assigning a type to the variable 'key' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 8), for_loop_var_136306))
        # Assigning a type to the variable 'cell' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'cell', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 8), for_loop_var_136306))
        # SSA begins for a for statement (line 457)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        int_136307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 19), 'int')
        # Getting the type of 'key' (line 459)
        key_136308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'key')
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___136309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 15), key_136308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 459)
        subscript_call_result_136310 = invoke(stypy.reporting.localization.Localization(__file__, 459, 15), getitem___136309, int_136307)
        
        # Getting the type of 'self' (line 459)
        self_136311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 25), 'self')
        # Obtaining the member '_autoColumns' of a type (line 459)
        _autoColumns_136312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 25), self_136311, '_autoColumns')
        # Applying the binary operator 'in' (line 459)
        result_contains_136313 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 15), 'in', subscript_call_result_136310, _autoColumns_136312)
        
        # Testing the type of an if condition (line 459)
        if_condition_136314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 12), result_contains_136313)
        # Assigning a type to the variable 'if_condition_136314' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'if_condition_136314', if_condition_136314)
        # SSA begins for if statement (line 459)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 459)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 461):
        
        # Assigning a Call to a Name (line 461):
        
        # Call to auto_set_font_size(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'renderer' (line 461)
        renderer_136317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 43), 'renderer', False)
        # Processing the call keyword arguments (line 461)
        kwargs_136318 = {}
        # Getting the type of 'cell' (line 461)
        cell_136315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 'cell', False)
        # Obtaining the member 'auto_set_font_size' of a type (line 461)
        auto_set_font_size_136316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 19), cell_136315, 'auto_set_font_size')
        # Calling auto_set_font_size(args, kwargs) (line 461)
        auto_set_font_size_call_result_136319 = invoke(stypy.reporting.localization.Localization(__file__, 461, 19), auto_set_font_size_136316, *[renderer_136317], **kwargs_136318)
        
        # Assigning a type to the variable 'size' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'size', auto_set_font_size_call_result_136319)
        
        # Assigning a Call to a Name (line 462):
        
        # Assigning a Call to a Name (line 462):
        
        # Call to min(...): (line 462)
        # Processing the call arguments (line 462)
        # Getting the type of 'fontsize' (line 462)
        fontsize_136321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), 'fontsize', False)
        # Getting the type of 'size' (line 462)
        size_136322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 37), 'size', False)
        # Processing the call keyword arguments (line 462)
        kwargs_136323 = {}
        # Getting the type of 'min' (line 462)
        min_136320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 23), 'min', False)
        # Calling min(args, kwargs) (line 462)
        min_call_result_136324 = invoke(stypy.reporting.localization.Localization(__file__, 462, 23), min_136320, *[fontsize_136321, size_136322], **kwargs_136323)
        
        # Assigning a type to the variable 'fontsize' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'fontsize', min_call_result_136324)
        
        # Call to append(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'cell' (line 463)
        cell_136327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 25), 'cell', False)
        # Processing the call keyword arguments (line 463)
        kwargs_136328 = {}
        # Getting the type of 'cells' (line 463)
        cells_136325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'cells', False)
        # Obtaining the member 'append' of a type (line 463)
        append_136326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 12), cells_136325, 'append')
        # Calling append(args, kwargs) (line 463)
        append_call_result_136329 = invoke(stypy.reporting.localization.Localization(__file__, 463, 12), append_136326, *[cell_136327], **kwargs_136328)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to itervalues(...): (line 466)
        # Processing the call arguments (line 466)
        # Getting the type of 'self' (line 466)
        self_136332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 35), 'self', False)
        # Obtaining the member '_cells' of a type (line 466)
        _cells_136333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 35), self_136332, '_cells')
        # Processing the call keyword arguments (line 466)
        kwargs_136334 = {}
        # Getting the type of 'six' (line 466)
        six_136330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 20), 'six', False)
        # Obtaining the member 'itervalues' of a type (line 466)
        itervalues_136331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 20), six_136330, 'itervalues')
        # Calling itervalues(args, kwargs) (line 466)
        itervalues_call_result_136335 = invoke(stypy.reporting.localization.Localization(__file__, 466, 20), itervalues_136331, *[_cells_136333], **kwargs_136334)
        
        # Testing the type of a for loop iterable (line 466)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 466, 8), itervalues_call_result_136335)
        # Getting the type of the for loop variable (line 466)
        for_loop_var_136336 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 466, 8), itervalues_call_result_136335)
        # Assigning a type to the variable 'cell' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'cell', for_loop_var_136336)
        # SSA begins for a for statement (line 466)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_fontsize(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'fontsize' (line 467)
        fontsize_136339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 30), 'fontsize', False)
        # Processing the call keyword arguments (line 467)
        kwargs_136340 = {}
        # Getting the type of 'cell' (line 467)
        cell_136337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'cell', False)
        # Obtaining the member 'set_fontsize' of a type (line 467)
        set_fontsize_136338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 12), cell_136337, 'set_fontsize')
        # Calling set_fontsize(args, kwargs) (line 467)
        set_fontsize_call_result_136341 = invoke(stypy.reporting.localization.Localization(__file__, 467, 12), set_fontsize_136338, *[fontsize_136339], **kwargs_136340)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_auto_set_font_size(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_auto_set_font_size' in the type store
        # Getting the type of 'stypy_return_type' (line 451)
        stypy_return_type_136342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136342)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_auto_set_font_size'
        return stypy_return_type_136342


    @norecursion
    def scale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'scale'
        module_type_store = module_type_store.open_function_context('scale', 469, 4, False)
        # Assigning a type to the variable 'self' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.scale.__dict__.__setitem__('stypy_localization', localization)
        Table.scale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.scale.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.scale.__dict__.__setitem__('stypy_function_name', 'Table.scale')
        Table.scale.__dict__.__setitem__('stypy_param_names_list', ['xscale', 'yscale'])
        Table.scale.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table.scale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table.scale.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.scale.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.scale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.scale.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.scale', ['xscale', 'yscale'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'scale', localization, ['xscale', 'yscale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'scale(...)' code ##################

        unicode_136343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 8), 'unicode', u' Scale column widths by xscale and row heights by yscale. ')
        
        
        # Call to itervalues(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'self' (line 471)
        self_136346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 32), 'self', False)
        # Obtaining the member '_cells' of a type (line 471)
        _cells_136347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 32), self_136346, '_cells')
        # Processing the call keyword arguments (line 471)
        kwargs_136348 = {}
        # Getting the type of 'six' (line 471)
        six_136344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 17), 'six', False)
        # Obtaining the member 'itervalues' of a type (line 471)
        itervalues_136345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 17), six_136344, 'itervalues')
        # Calling itervalues(args, kwargs) (line 471)
        itervalues_call_result_136349 = invoke(stypy.reporting.localization.Localization(__file__, 471, 17), itervalues_136345, *[_cells_136347], **kwargs_136348)
        
        # Testing the type of a for loop iterable (line 471)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 471, 8), itervalues_call_result_136349)
        # Getting the type of the for loop variable (line 471)
        for_loop_var_136350 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 471, 8), itervalues_call_result_136349)
        # Assigning a type to the variable 'c' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'c', for_loop_var_136350)
        # SSA begins for a for statement (line 471)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_width(...): (line 472)
        # Processing the call arguments (line 472)
        
        # Call to get_width(...): (line 472)
        # Processing the call keyword arguments (line 472)
        kwargs_136355 = {}
        # Getting the type of 'c' (line 472)
        c_136353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 24), 'c', False)
        # Obtaining the member 'get_width' of a type (line 472)
        get_width_136354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 24), c_136353, 'get_width')
        # Calling get_width(args, kwargs) (line 472)
        get_width_call_result_136356 = invoke(stypy.reporting.localization.Localization(__file__, 472, 24), get_width_136354, *[], **kwargs_136355)
        
        # Getting the type of 'xscale' (line 472)
        xscale_136357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 40), 'xscale', False)
        # Applying the binary operator '*' (line 472)
        result_mul_136358 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 24), '*', get_width_call_result_136356, xscale_136357)
        
        # Processing the call keyword arguments (line 472)
        kwargs_136359 = {}
        # Getting the type of 'c' (line 472)
        c_136351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'c', False)
        # Obtaining the member 'set_width' of a type (line 472)
        set_width_136352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 12), c_136351, 'set_width')
        # Calling set_width(args, kwargs) (line 472)
        set_width_call_result_136360 = invoke(stypy.reporting.localization.Localization(__file__, 472, 12), set_width_136352, *[result_mul_136358], **kwargs_136359)
        
        
        # Call to set_height(...): (line 473)
        # Processing the call arguments (line 473)
        
        # Call to get_height(...): (line 473)
        # Processing the call keyword arguments (line 473)
        kwargs_136365 = {}
        # Getting the type of 'c' (line 473)
        c_136363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 25), 'c', False)
        # Obtaining the member 'get_height' of a type (line 473)
        get_height_136364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 25), c_136363, 'get_height')
        # Calling get_height(args, kwargs) (line 473)
        get_height_call_result_136366 = invoke(stypy.reporting.localization.Localization(__file__, 473, 25), get_height_136364, *[], **kwargs_136365)
        
        # Getting the type of 'yscale' (line 473)
        yscale_136367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 42), 'yscale', False)
        # Applying the binary operator '*' (line 473)
        result_mul_136368 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 25), '*', get_height_call_result_136366, yscale_136367)
        
        # Processing the call keyword arguments (line 473)
        kwargs_136369 = {}
        # Getting the type of 'c' (line 473)
        c_136361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'c', False)
        # Obtaining the member 'set_height' of a type (line 473)
        set_height_136362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 12), c_136361, 'set_height')
        # Calling set_height(args, kwargs) (line 473)
        set_height_call_result_136370 = invoke(stypy.reporting.localization.Localization(__file__, 473, 12), set_height_136362, *[result_mul_136368], **kwargs_136369)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'scale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'scale' in the type store
        # Getting the type of 'stypy_return_type' (line 469)
        stypy_return_type_136371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136371)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'scale'
        return stypy_return_type_136371


    @norecursion
    def set_fontsize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_fontsize'
        module_type_store = module_type_store.open_function_context('set_fontsize', 475, 4, False)
        # Assigning a type to the variable 'self' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.set_fontsize.__dict__.__setitem__('stypy_localization', localization)
        Table.set_fontsize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.set_fontsize.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.set_fontsize.__dict__.__setitem__('stypy_function_name', 'Table.set_fontsize')
        Table.set_fontsize.__dict__.__setitem__('stypy_param_names_list', ['size'])
        Table.set_fontsize.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table.set_fontsize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table.set_fontsize.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.set_fontsize.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.set_fontsize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.set_fontsize.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.set_fontsize', ['size'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_fontsize', localization, ['size'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_fontsize(...)' code ##################

        unicode_136372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, (-1)), 'unicode', u'\n        Set the fontsize of the cell text\n\n        ACCEPTS: a float in points\n        ')
        
        
        # Call to itervalues(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'self' (line 482)
        self_136375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 35), 'self', False)
        # Obtaining the member '_cells' of a type (line 482)
        _cells_136376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 35), self_136375, '_cells')
        # Processing the call keyword arguments (line 482)
        kwargs_136377 = {}
        # Getting the type of 'six' (line 482)
        six_136373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 20), 'six', False)
        # Obtaining the member 'itervalues' of a type (line 482)
        itervalues_136374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 20), six_136373, 'itervalues')
        # Calling itervalues(args, kwargs) (line 482)
        itervalues_call_result_136378 = invoke(stypy.reporting.localization.Localization(__file__, 482, 20), itervalues_136374, *[_cells_136376], **kwargs_136377)
        
        # Testing the type of a for loop iterable (line 482)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 482, 8), itervalues_call_result_136378)
        # Getting the type of the for loop variable (line 482)
        for_loop_var_136379 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 482, 8), itervalues_call_result_136378)
        # Assigning a type to the variable 'cell' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'cell', for_loop_var_136379)
        # SSA begins for a for statement (line 482)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_fontsize(...): (line 483)
        # Processing the call arguments (line 483)
        # Getting the type of 'size' (line 483)
        size_136382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 30), 'size', False)
        # Processing the call keyword arguments (line 483)
        kwargs_136383 = {}
        # Getting the type of 'cell' (line 483)
        cell_136380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'cell', False)
        # Obtaining the member 'set_fontsize' of a type (line 483)
        set_fontsize_136381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 12), cell_136380, 'set_fontsize')
        # Calling set_fontsize(args, kwargs) (line 483)
        set_fontsize_call_result_136384 = invoke(stypy.reporting.localization.Localization(__file__, 483, 12), set_fontsize_136381, *[size_136382], **kwargs_136383)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 484):
        
        # Assigning a Name to a Attribute (line 484):
        # Getting the type of 'True' (line 484)
        True_136385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 21), 'True')
        # Getting the type of 'self' (line 484)
        self_136386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'self')
        # Setting the type of the member 'stale' of a type (line 484)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 8), self_136386, 'stale', True_136385)
        
        # ################# End of 'set_fontsize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_fontsize' in the type store
        # Getting the type of 'stypy_return_type' (line 475)
        stypy_return_type_136387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_fontsize'
        return stypy_return_type_136387


    @norecursion
    def _offset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_offset'
        module_type_store = module_type_store.open_function_context('_offset', 486, 4, False)
        # Assigning a type to the variable 'self' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table._offset.__dict__.__setitem__('stypy_localization', localization)
        Table._offset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table._offset.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table._offset.__dict__.__setitem__('stypy_function_name', 'Table._offset')
        Table._offset.__dict__.__setitem__('stypy_param_names_list', ['ox', 'oy'])
        Table._offset.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table._offset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table._offset.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table._offset.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table._offset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table._offset.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table._offset', ['ox', 'oy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_offset', localization, ['ox', 'oy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_offset(...)' code ##################

        unicode_136388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 8), 'unicode', u'Move all the artists by ox,oy (axes coords)')
        
        
        # Call to itervalues(...): (line 489)
        # Processing the call arguments (line 489)
        # Getting the type of 'self' (line 489)
        self_136391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 32), 'self', False)
        # Obtaining the member '_cells' of a type (line 489)
        _cells_136392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 32), self_136391, '_cells')
        # Processing the call keyword arguments (line 489)
        kwargs_136393 = {}
        # Getting the type of 'six' (line 489)
        six_136389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 17), 'six', False)
        # Obtaining the member 'itervalues' of a type (line 489)
        itervalues_136390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 17), six_136389, 'itervalues')
        # Calling itervalues(args, kwargs) (line 489)
        itervalues_call_result_136394 = invoke(stypy.reporting.localization.Localization(__file__, 489, 17), itervalues_136390, *[_cells_136392], **kwargs_136393)
        
        # Testing the type of a for loop iterable (line 489)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 489, 8), itervalues_call_result_136394)
        # Getting the type of the for loop variable (line 489)
        for_loop_var_136395 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 489, 8), itervalues_call_result_136394)
        # Assigning a type to the variable 'c' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'c', for_loop_var_136395)
        # SSA begins for a for statement (line 489)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Tuple to a Tuple (line 490):
        
        # Assigning a Call to a Name (line 490):
        
        # Call to get_x(...): (line 490)
        # Processing the call keyword arguments (line 490)
        kwargs_136398 = {}
        # Getting the type of 'c' (line 490)
        c_136396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 19), 'c', False)
        # Obtaining the member 'get_x' of a type (line 490)
        get_x_136397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 19), c_136396, 'get_x')
        # Calling get_x(args, kwargs) (line 490)
        get_x_call_result_136399 = invoke(stypy.reporting.localization.Localization(__file__, 490, 19), get_x_136397, *[], **kwargs_136398)
        
        # Assigning a type to the variable 'tuple_assignment_135252' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'tuple_assignment_135252', get_x_call_result_136399)
        
        # Assigning a Call to a Name (line 490):
        
        # Call to get_y(...): (line 490)
        # Processing the call keyword arguments (line 490)
        kwargs_136402 = {}
        # Getting the type of 'c' (line 490)
        c_136400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 30), 'c', False)
        # Obtaining the member 'get_y' of a type (line 490)
        get_y_136401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 30), c_136400, 'get_y')
        # Calling get_y(args, kwargs) (line 490)
        get_y_call_result_136403 = invoke(stypy.reporting.localization.Localization(__file__, 490, 30), get_y_136401, *[], **kwargs_136402)
        
        # Assigning a type to the variable 'tuple_assignment_135253' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'tuple_assignment_135253', get_y_call_result_136403)
        
        # Assigning a Name to a Name (line 490):
        # Getting the type of 'tuple_assignment_135252' (line 490)
        tuple_assignment_135252_136404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'tuple_assignment_135252')
        # Assigning a type to the variable 'x' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'x', tuple_assignment_135252_136404)
        
        # Assigning a Name to a Name (line 490):
        # Getting the type of 'tuple_assignment_135253' (line 490)
        tuple_assignment_135253_136405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'tuple_assignment_135253')
        # Assigning a type to the variable 'y' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'y', tuple_assignment_135253_136405)
        
        # Call to set_x(...): (line 491)
        # Processing the call arguments (line 491)
        # Getting the type of 'x' (line 491)
        x_136408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 20), 'x', False)
        # Getting the type of 'ox' (line 491)
        ox_136409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 24), 'ox', False)
        # Applying the binary operator '+' (line 491)
        result_add_136410 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 20), '+', x_136408, ox_136409)
        
        # Processing the call keyword arguments (line 491)
        kwargs_136411 = {}
        # Getting the type of 'c' (line 491)
        c_136406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'c', False)
        # Obtaining the member 'set_x' of a type (line 491)
        set_x_136407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 12), c_136406, 'set_x')
        # Calling set_x(args, kwargs) (line 491)
        set_x_call_result_136412 = invoke(stypy.reporting.localization.Localization(__file__, 491, 12), set_x_136407, *[result_add_136410], **kwargs_136411)
        
        
        # Call to set_y(...): (line 492)
        # Processing the call arguments (line 492)
        # Getting the type of 'y' (line 492)
        y_136415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 20), 'y', False)
        # Getting the type of 'oy' (line 492)
        oy_136416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 24), 'oy', False)
        # Applying the binary operator '+' (line 492)
        result_add_136417 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 20), '+', y_136415, oy_136416)
        
        # Processing the call keyword arguments (line 492)
        kwargs_136418 = {}
        # Getting the type of 'c' (line 492)
        c_136413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'c', False)
        # Obtaining the member 'set_y' of a type (line 492)
        set_y_136414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 12), c_136413, 'set_y')
        # Calling set_y(args, kwargs) (line 492)
        set_y_call_result_136419 = invoke(stypy.reporting.localization.Localization(__file__, 492, 12), set_y_136414, *[result_add_136417], **kwargs_136418)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_offset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_offset' in the type store
        # Getting the type of 'stypy_return_type' (line 486)
        stypy_return_type_136420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136420)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_offset'
        return stypy_return_type_136420


    @norecursion
    def _update_positions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update_positions'
        module_type_store = module_type_store.open_function_context('_update_positions', 494, 4, False)
        # Assigning a type to the variable 'self' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table._update_positions.__dict__.__setitem__('stypy_localization', localization)
        Table._update_positions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table._update_positions.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table._update_positions.__dict__.__setitem__('stypy_function_name', 'Table._update_positions')
        Table._update_positions.__dict__.__setitem__('stypy_param_names_list', ['renderer'])
        Table._update_positions.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table._update_positions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table._update_positions.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table._update_positions.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table._update_positions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table._update_positions.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table._update_positions', ['renderer'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update_positions', localization, ['renderer'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update_positions(...)' code ##################

        
        # Getting the type of 'self' (line 499)
        self_136421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 19), 'self')
        # Obtaining the member '_autoColumns' of a type (line 499)
        _autoColumns_136422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 19), self_136421, '_autoColumns')
        # Testing the type of a for loop iterable (line 499)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 499, 8), _autoColumns_136422)
        # Getting the type of the for loop variable (line 499)
        for_loop_var_136423 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 499, 8), _autoColumns_136422)
        # Assigning a type to the variable 'col' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'col', for_loop_var_136423)
        # SSA begins for a for statement (line 499)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _auto_set_column_width(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'col' (line 500)
        col_136426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 40), 'col', False)
        # Getting the type of 'renderer' (line 500)
        renderer_136427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 45), 'renderer', False)
        # Processing the call keyword arguments (line 500)
        kwargs_136428 = {}
        # Getting the type of 'self' (line 500)
        self_136424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'self', False)
        # Obtaining the member '_auto_set_column_width' of a type (line 500)
        _auto_set_column_width_136425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 12), self_136424, '_auto_set_column_width')
        # Calling _auto_set_column_width(args, kwargs) (line 500)
        _auto_set_column_width_call_result_136429 = invoke(stypy.reporting.localization.Localization(__file__, 500, 12), _auto_set_column_width_136425, *[col_136426, renderer_136427], **kwargs_136428)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 502)
        self_136430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 11), 'self')
        # Obtaining the member '_autoFontsize' of a type (line 502)
        _autoFontsize_136431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 11), self_136430, '_autoFontsize')
        # Testing the type of an if condition (line 502)
        if_condition_136432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 8), _autoFontsize_136431)
        # Assigning a type to the variable 'if_condition_136432' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'if_condition_136432', if_condition_136432)
        # SSA begins for if statement (line 502)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _auto_set_font_size(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'renderer' (line 503)
        renderer_136435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 37), 'renderer', False)
        # Processing the call keyword arguments (line 503)
        kwargs_136436 = {}
        # Getting the type of 'self' (line 503)
        self_136433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'self', False)
        # Obtaining the member '_auto_set_font_size' of a type (line 503)
        _auto_set_font_size_136434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 12), self_136433, '_auto_set_font_size')
        # Calling _auto_set_font_size(args, kwargs) (line 503)
        _auto_set_font_size_call_result_136437 = invoke(stypy.reporting.localization.Localization(__file__, 503, 12), _auto_set_font_size_136434, *[renderer_136435], **kwargs_136436)
        
        # SSA join for if statement (line 502)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _do_cell_alignment(...): (line 506)
        # Processing the call keyword arguments (line 506)
        kwargs_136440 = {}
        # Getting the type of 'self' (line 506)
        self_136438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'self', False)
        # Obtaining the member '_do_cell_alignment' of a type (line 506)
        _do_cell_alignment_136439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 8), self_136438, '_do_cell_alignment')
        # Calling _do_cell_alignment(args, kwargs) (line 506)
        _do_cell_alignment_call_result_136441 = invoke(stypy.reporting.localization.Localization(__file__, 506, 8), _do_cell_alignment_136439, *[], **kwargs_136440)
        
        
        # Assigning a Call to a Name (line 508):
        
        # Assigning a Call to a Name (line 508):
        
        # Call to _get_grid_bbox(...): (line 508)
        # Processing the call arguments (line 508)
        # Getting the type of 'renderer' (line 508)
        renderer_136444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 35), 'renderer', False)
        # Processing the call keyword arguments (line 508)
        kwargs_136445 = {}
        # Getting the type of 'self' (line 508)
        self_136442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 15), 'self', False)
        # Obtaining the member '_get_grid_bbox' of a type (line 508)
        _get_grid_bbox_136443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 15), self_136442, '_get_grid_bbox')
        # Calling _get_grid_bbox(args, kwargs) (line 508)
        _get_grid_bbox_call_result_136446 = invoke(stypy.reporting.localization.Localization(__file__, 508, 15), _get_grid_bbox_136443, *[renderer_136444], **kwargs_136445)
        
        # Assigning a type to the variable 'bbox' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'bbox', _get_grid_bbox_call_result_136446)
        
        # Assigning a Attribute to a Tuple (line 509):
        
        # Assigning a Subscript to a Name (line 509):
        
        # Obtaining the type of the subscript
        int_136447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 8), 'int')
        # Getting the type of 'bbox' (line 509)
        bbox_136448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 509)
        bounds_136449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 21), bbox_136448, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 509)
        getitem___136450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), bounds_136449, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 509)
        subscript_call_result_136451 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), getitem___136450, int_136447)
        
        # Assigning a type to the variable 'tuple_var_assignment_135254' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'tuple_var_assignment_135254', subscript_call_result_136451)
        
        # Assigning a Subscript to a Name (line 509):
        
        # Obtaining the type of the subscript
        int_136452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 8), 'int')
        # Getting the type of 'bbox' (line 509)
        bbox_136453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 509)
        bounds_136454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 21), bbox_136453, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 509)
        getitem___136455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), bounds_136454, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 509)
        subscript_call_result_136456 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), getitem___136455, int_136452)
        
        # Assigning a type to the variable 'tuple_var_assignment_135255' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'tuple_var_assignment_135255', subscript_call_result_136456)
        
        # Assigning a Subscript to a Name (line 509):
        
        # Obtaining the type of the subscript
        int_136457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 8), 'int')
        # Getting the type of 'bbox' (line 509)
        bbox_136458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 509)
        bounds_136459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 21), bbox_136458, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 509)
        getitem___136460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), bounds_136459, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 509)
        subscript_call_result_136461 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), getitem___136460, int_136457)
        
        # Assigning a type to the variable 'tuple_var_assignment_135256' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'tuple_var_assignment_135256', subscript_call_result_136461)
        
        # Assigning a Subscript to a Name (line 509):
        
        # Obtaining the type of the subscript
        int_136462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 8), 'int')
        # Getting the type of 'bbox' (line 509)
        bbox_136463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 21), 'bbox')
        # Obtaining the member 'bounds' of a type (line 509)
        bounds_136464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 21), bbox_136463, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 509)
        getitem___136465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 8), bounds_136464, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 509)
        subscript_call_result_136466 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), getitem___136465, int_136462)
        
        # Assigning a type to the variable 'tuple_var_assignment_135257' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'tuple_var_assignment_135257', subscript_call_result_136466)
        
        # Assigning a Name to a Name (line 509):
        # Getting the type of 'tuple_var_assignment_135254' (line 509)
        tuple_var_assignment_135254_136467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'tuple_var_assignment_135254')
        # Assigning a type to the variable 'l' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'l', tuple_var_assignment_135254_136467)
        
        # Assigning a Name to a Name (line 509):
        # Getting the type of 'tuple_var_assignment_135255' (line 509)
        tuple_var_assignment_135255_136468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'tuple_var_assignment_135255')
        # Assigning a type to the variable 'b' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 11), 'b', tuple_var_assignment_135255_136468)
        
        # Assigning a Name to a Name (line 509):
        # Getting the type of 'tuple_var_assignment_135256' (line 509)
        tuple_var_assignment_135256_136469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'tuple_var_assignment_135256')
        # Assigning a type to the variable 'w' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 14), 'w', tuple_var_assignment_135256_136469)
        
        # Assigning a Name to a Name (line 509):
        # Getting the type of 'tuple_var_assignment_135257' (line 509)
        tuple_var_assignment_135257_136470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'tuple_var_assignment_135257')
        # Assigning a type to the variable 'h' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 17), 'h', tuple_var_assignment_135257_136470)
        
        
        # Getting the type of 'self' (line 511)
        self_136471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 11), 'self')
        # Obtaining the member '_bbox' of a type (line 511)
        _bbox_136472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 11), self_136471, '_bbox')
        # Getting the type of 'None' (line 511)
        None_136473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 29), 'None')
        # Applying the binary operator 'isnot' (line 511)
        result_is_not_136474 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 11), 'isnot', _bbox_136472, None_136473)
        
        # Testing the type of an if condition (line 511)
        if_condition_136475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 511, 8), result_is_not_136474)
        # Assigning a type to the variable 'if_condition_136475' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'if_condition_136475', if_condition_136475)
        # SSA begins for if statement (line 511)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Tuple (line 513):
        
        # Assigning a Subscript to a Name (line 513):
        
        # Obtaining the type of the subscript
        int_136476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 12), 'int')
        # Getting the type of 'self' (line 513)
        self_136477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 29), 'self')
        # Obtaining the member '_bbox' of a type (line 513)
        _bbox_136478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 29), self_136477, '_bbox')
        # Obtaining the member '__getitem__' of a type (line 513)
        getitem___136479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 12), _bbox_136478, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 513)
        subscript_call_result_136480 = invoke(stypy.reporting.localization.Localization(__file__, 513, 12), getitem___136479, int_136476)
        
        # Assigning a type to the variable 'tuple_var_assignment_135258' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'tuple_var_assignment_135258', subscript_call_result_136480)
        
        # Assigning a Subscript to a Name (line 513):
        
        # Obtaining the type of the subscript
        int_136481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 12), 'int')
        # Getting the type of 'self' (line 513)
        self_136482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 29), 'self')
        # Obtaining the member '_bbox' of a type (line 513)
        _bbox_136483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 29), self_136482, '_bbox')
        # Obtaining the member '__getitem__' of a type (line 513)
        getitem___136484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 12), _bbox_136483, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 513)
        subscript_call_result_136485 = invoke(stypy.reporting.localization.Localization(__file__, 513, 12), getitem___136484, int_136481)
        
        # Assigning a type to the variable 'tuple_var_assignment_135259' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'tuple_var_assignment_135259', subscript_call_result_136485)
        
        # Assigning a Subscript to a Name (line 513):
        
        # Obtaining the type of the subscript
        int_136486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 12), 'int')
        # Getting the type of 'self' (line 513)
        self_136487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 29), 'self')
        # Obtaining the member '_bbox' of a type (line 513)
        _bbox_136488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 29), self_136487, '_bbox')
        # Obtaining the member '__getitem__' of a type (line 513)
        getitem___136489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 12), _bbox_136488, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 513)
        subscript_call_result_136490 = invoke(stypy.reporting.localization.Localization(__file__, 513, 12), getitem___136489, int_136486)
        
        # Assigning a type to the variable 'tuple_var_assignment_135260' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'tuple_var_assignment_135260', subscript_call_result_136490)
        
        # Assigning a Subscript to a Name (line 513):
        
        # Obtaining the type of the subscript
        int_136491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 12), 'int')
        # Getting the type of 'self' (line 513)
        self_136492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 29), 'self')
        # Obtaining the member '_bbox' of a type (line 513)
        _bbox_136493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 29), self_136492, '_bbox')
        # Obtaining the member '__getitem__' of a type (line 513)
        getitem___136494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 12), _bbox_136493, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 513)
        subscript_call_result_136495 = invoke(stypy.reporting.localization.Localization(__file__, 513, 12), getitem___136494, int_136491)
        
        # Assigning a type to the variable 'tuple_var_assignment_135261' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'tuple_var_assignment_135261', subscript_call_result_136495)
        
        # Assigning a Name to a Name (line 513):
        # Getting the type of 'tuple_var_assignment_135258' (line 513)
        tuple_var_assignment_135258_136496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'tuple_var_assignment_135258')
        # Assigning a type to the variable 'rl' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'rl', tuple_var_assignment_135258_136496)
        
        # Assigning a Name to a Name (line 513):
        # Getting the type of 'tuple_var_assignment_135259' (line 513)
        tuple_var_assignment_135259_136497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'tuple_var_assignment_135259')
        # Assigning a type to the variable 'rb' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 16), 'rb', tuple_var_assignment_135259_136497)
        
        # Assigning a Name to a Name (line 513):
        # Getting the type of 'tuple_var_assignment_135260' (line 513)
        tuple_var_assignment_135260_136498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'tuple_var_assignment_135260')
        # Assigning a type to the variable 'rw' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 20), 'rw', tuple_var_assignment_135260_136498)
        
        # Assigning a Name to a Name (line 513):
        # Getting the type of 'tuple_var_assignment_135261' (line 513)
        tuple_var_assignment_135261_136499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'tuple_var_assignment_135261')
        # Assigning a type to the variable 'rh' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 24), 'rh', tuple_var_assignment_135261_136499)
        
        # Call to scale(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'rw' (line 514)
        rw_136502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 23), 'rw', False)
        # Getting the type of 'w' (line 514)
        w_136503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 28), 'w', False)
        # Applying the binary operator 'div' (line 514)
        result_div_136504 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 23), 'div', rw_136502, w_136503)
        
        # Getting the type of 'rh' (line 514)
        rh_136505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 31), 'rh', False)
        # Getting the type of 'h' (line 514)
        h_136506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 36), 'h', False)
        # Applying the binary operator 'div' (line 514)
        result_div_136507 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 31), 'div', rh_136505, h_136506)
        
        # Processing the call keyword arguments (line 514)
        kwargs_136508 = {}
        # Getting the type of 'self' (line 514)
        self_136500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'self', False)
        # Obtaining the member 'scale' of a type (line 514)
        scale_136501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 12), self_136500, 'scale')
        # Calling scale(args, kwargs) (line 514)
        scale_call_result_136509 = invoke(stypy.reporting.localization.Localization(__file__, 514, 12), scale_136501, *[result_div_136504, result_div_136507], **kwargs_136508)
        
        
        # Assigning a BinOp to a Name (line 515):
        
        # Assigning a BinOp to a Name (line 515):
        # Getting the type of 'rl' (line 515)
        rl_136510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 17), 'rl')
        # Getting the type of 'l' (line 515)
        l_136511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 22), 'l')
        # Applying the binary operator '-' (line 515)
        result_sub_136512 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 17), '-', rl_136510, l_136511)
        
        # Assigning a type to the variable 'ox' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'ox', result_sub_136512)
        
        # Assigning a BinOp to a Name (line 516):
        
        # Assigning a BinOp to a Name (line 516):
        # Getting the type of 'rb' (line 516)
        rb_136513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 17), 'rb')
        # Getting the type of 'b' (line 516)
        b_136514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 22), 'b')
        # Applying the binary operator '-' (line 516)
        result_sub_136515 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 17), '-', rb_136513, b_136514)
        
        # Assigning a type to the variable 'oy' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'oy', result_sub_136515)
        
        # Call to _do_cell_alignment(...): (line 517)
        # Processing the call keyword arguments (line 517)
        kwargs_136518 = {}
        # Getting the type of 'self' (line 517)
        self_136516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'self', False)
        # Obtaining the member '_do_cell_alignment' of a type (line 517)
        _do_cell_alignment_136517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 12), self_136516, '_do_cell_alignment')
        # Calling _do_cell_alignment(args, kwargs) (line 517)
        _do_cell_alignment_call_result_136519 = invoke(stypy.reporting.localization.Localization(__file__, 517, 12), _do_cell_alignment_136517, *[], **kwargs_136518)
        
        # SSA branch for the else part of an if statement (line 511)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 520):
        
        # Assigning a Call to a Name:
        
        # Call to xrange(...): (line 521)
        # Processing the call arguments (line 521)
        
        # Call to len(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'self' (line 521)
        self_136522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 54), 'self', False)
        # Obtaining the member 'codes' of a type (line 521)
        codes_136523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 54), self_136522, 'codes')
        # Processing the call keyword arguments (line 521)
        kwargs_136524 = {}
        # Getting the type of 'len' (line 521)
        len_136521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 50), 'len', False)
        # Calling len(args, kwargs) (line 521)
        len_call_result_136525 = invoke(stypy.reporting.localization.Localization(__file__, 521, 50), len_136521, *[codes_136523], **kwargs_136524)
        
        # Processing the call keyword arguments (line 521)
        kwargs_136526 = {}
        # Getting the type of 'xrange' (line 521)
        xrange_136520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 43), 'xrange', False)
        # Calling xrange(args, kwargs) (line 521)
        xrange_call_result_136527 = invoke(stypy.reporting.localization.Localization(__file__, 521, 43), xrange_136520, *[len_call_result_136525], **kwargs_136526)
        
        # Assigning a type to the variable 'call_assignment_135262' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', xrange_call_result_136527)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136531 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136528, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136532 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136529, *[int_136530], **kwargs_136531)
        
        # Assigning a type to the variable 'call_assignment_135263' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135263', getitem___call_result_136532)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135263' (line 520)
        call_assignment_135263_136533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135263')
        # Assigning a type to the variable 'BEST' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 13), 'BEST', call_assignment_135263_136533)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136537 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136534, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136538 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136535, *[int_136536], **kwargs_136537)
        
        # Assigning a type to the variable 'call_assignment_135264' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135264', getitem___call_result_136538)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135264' (line 520)
        call_assignment_135264_136539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135264')
        # Assigning a type to the variable 'UR' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 19), 'UR', call_assignment_135264_136539)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136543 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136540, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136544 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136541, *[int_136542], **kwargs_136543)
        
        # Assigning a type to the variable 'call_assignment_135265' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135265', getitem___call_result_136544)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135265' (line 520)
        call_assignment_135265_136545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135265')
        # Assigning a type to the variable 'UL' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 23), 'UL', call_assignment_135265_136545)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136549 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136546, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136550 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136547, *[int_136548], **kwargs_136549)
        
        # Assigning a type to the variable 'call_assignment_135266' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135266', getitem___call_result_136550)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135266' (line 520)
        call_assignment_135266_136551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135266')
        # Assigning a type to the variable 'LL' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 27), 'LL', call_assignment_135266_136551)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136555 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136552, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136556 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136553, *[int_136554], **kwargs_136555)
        
        # Assigning a type to the variable 'call_assignment_135267' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135267', getitem___call_result_136556)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135267' (line 520)
        call_assignment_135267_136557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135267')
        # Assigning a type to the variable 'LR' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 31), 'LR', call_assignment_135267_136557)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136561 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136558, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136562 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136559, *[int_136560], **kwargs_136561)
        
        # Assigning a type to the variable 'call_assignment_135268' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135268', getitem___call_result_136562)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135268' (line 520)
        call_assignment_135268_136563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135268')
        # Assigning a type to the variable 'CL' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 35), 'CL', call_assignment_135268_136563)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136567 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136564, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136568 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136565, *[int_136566], **kwargs_136567)
        
        # Assigning a type to the variable 'call_assignment_135269' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135269', getitem___call_result_136568)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135269' (line 520)
        call_assignment_135269_136569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135269')
        # Assigning a type to the variable 'CR' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 39), 'CR', call_assignment_135269_136569)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136573 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136570, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136574 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136571, *[int_136572], **kwargs_136573)
        
        # Assigning a type to the variable 'call_assignment_135270' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135270', getitem___call_result_136574)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135270' (line 520)
        call_assignment_135270_136575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135270')
        # Assigning a type to the variable 'LC' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 43), 'LC', call_assignment_135270_136575)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136579 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136576, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136580 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136577, *[int_136578], **kwargs_136579)
        
        # Assigning a type to the variable 'call_assignment_135271' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135271', getitem___call_result_136580)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135271' (line 520)
        call_assignment_135271_136581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135271')
        # Assigning a type to the variable 'UC' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 47), 'UC', call_assignment_135271_136581)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136585 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136582, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136586 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136583, *[int_136584], **kwargs_136585)
        
        # Assigning a type to the variable 'call_assignment_135272' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135272', getitem___call_result_136586)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135272' (line 520)
        call_assignment_135272_136587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135272')
        # Assigning a type to the variable 'C' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 51), 'C', call_assignment_135272_136587)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136591 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136588, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136592 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136589, *[int_136590], **kwargs_136591)
        
        # Assigning a type to the variable 'call_assignment_135273' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135273', getitem___call_result_136592)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135273' (line 520)
        call_assignment_135273_136593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135273')
        # Assigning a type to the variable 'TR' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 13), 'TR', call_assignment_135273_136593)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136597 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136594, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136598 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136595, *[int_136596], **kwargs_136597)
        
        # Assigning a type to the variable 'call_assignment_135274' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135274', getitem___call_result_136598)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135274' (line 520)
        call_assignment_135274_136599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135274')
        # Assigning a type to the variable 'TL' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 17), 'TL', call_assignment_135274_136599)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136603 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136600, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136604 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136601, *[int_136602], **kwargs_136603)
        
        # Assigning a type to the variable 'call_assignment_135275' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135275', getitem___call_result_136604)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135275' (line 520)
        call_assignment_135275_136605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135275')
        # Assigning a type to the variable 'BL' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 21), 'BL', call_assignment_135275_136605)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136609 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136606, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136610 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136607, *[int_136608], **kwargs_136609)
        
        # Assigning a type to the variable 'call_assignment_135276' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135276', getitem___call_result_136610)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135276' (line 520)
        call_assignment_135276_136611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135276')
        # Assigning a type to the variable 'BR' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 25), 'BR', call_assignment_135276_136611)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136615 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136612, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136616 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136613, *[int_136614], **kwargs_136615)
        
        # Assigning a type to the variable 'call_assignment_135277' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135277', getitem___call_result_136616)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135277' (line 520)
        call_assignment_135277_136617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135277')
        # Assigning a type to the variable 'R' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 29), 'R', call_assignment_135277_136617)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136621 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136618, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136622 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136619, *[int_136620], **kwargs_136621)
        
        # Assigning a type to the variable 'call_assignment_135278' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135278', getitem___call_result_136622)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135278' (line 520)
        call_assignment_135278_136623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135278')
        # Assigning a type to the variable 'L' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 32), 'L', call_assignment_135278_136623)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136627 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136624, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136628 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136625, *[int_136626], **kwargs_136627)
        
        # Assigning a type to the variable 'call_assignment_135279' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135279', getitem___call_result_136628)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135279' (line 520)
        call_assignment_135279_136629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135279')
        # Assigning a type to the variable 'T' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 35), 'T', call_assignment_135279_136629)
        
        # Assigning a Call to a Name (line 520):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_136632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 12), 'int')
        # Processing the call keyword arguments
        kwargs_136633 = {}
        # Getting the type of 'call_assignment_135262' (line 520)
        call_assignment_135262_136630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135262', False)
        # Obtaining the member '__getitem__' of a type (line 520)
        getitem___136631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 12), call_assignment_135262_136630, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_136634 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___136631, *[int_136632], **kwargs_136633)
        
        # Assigning a type to the variable 'call_assignment_135280' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135280', getitem___call_result_136634)
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'call_assignment_135280' (line 520)
        call_assignment_135280_136635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'call_assignment_135280')
        # Assigning a type to the variable 'B' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 38), 'B', call_assignment_135280_136635)
        
        # Assigning a BinOp to a Name (line 523):
        
        # Assigning a BinOp to a Name (line 523):
        float_136636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 18), 'float')
        # Getting the type of 'w' (line 523)
        w_136637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 24), 'w')
        int_136638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 28), 'int')
        # Applying the binary operator 'div' (line 523)
        result_div_136639 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 24), 'div', w_136637, int_136638)
        
        # Applying the binary operator '-' (line 523)
        result_sub_136640 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 18), '-', float_136636, result_div_136639)
        
        # Getting the type of 'l' (line 523)
        l_136641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 33), 'l')
        # Applying the binary operator '-' (line 523)
        result_sub_136642 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 17), '-', result_sub_136640, l_136641)
        
        # Assigning a type to the variable 'ox' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'ox', result_sub_136642)
        
        # Assigning a BinOp to a Name (line 524):
        
        # Assigning a BinOp to a Name (line 524):
        float_136643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 18), 'float')
        # Getting the type of 'h' (line 524)
        h_136644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 24), 'h')
        int_136645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 28), 'int')
        # Applying the binary operator 'div' (line 524)
        result_div_136646 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 24), 'div', h_136644, int_136645)
        
        # Applying the binary operator '-' (line 524)
        result_sub_136647 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 18), '-', float_136643, result_div_136646)
        
        # Getting the type of 'b' (line 524)
        b_136648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 33), 'b')
        # Applying the binary operator '-' (line 524)
        result_sub_136649 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 17), '-', result_sub_136647, b_136648)
        
        # Assigning a type to the variable 'oy' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'oy', result_sub_136649)
        
        
        # Getting the type of 'self' (line 525)
        self_136650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 15), 'self')
        # Obtaining the member '_loc' of a type (line 525)
        _loc_136651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 15), self_136650, '_loc')
        
        # Obtaining an instance of the builtin type 'tuple' (line 525)
        tuple_136652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 525)
        # Adding element type (line 525)
        # Getting the type of 'UL' (line 525)
        UL_136653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 29), 'UL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 29), tuple_136652, UL_136653)
        # Adding element type (line 525)
        # Getting the type of 'LL' (line 525)
        LL_136654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 33), 'LL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 29), tuple_136652, LL_136654)
        # Adding element type (line 525)
        # Getting the type of 'CL' (line 525)
        CL_136655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 37), 'CL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 29), tuple_136652, CL_136655)
        
        # Applying the binary operator 'in' (line 525)
        result_contains_136656 = python_operator(stypy.reporting.localization.Localization(__file__, 525, 15), 'in', _loc_136651, tuple_136652)
        
        # Testing the type of an if condition (line 525)
        if_condition_136657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 525, 12), result_contains_136656)
        # Assigning a type to the variable 'if_condition_136657' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'if_condition_136657', if_condition_136657)
        # SSA begins for if statement (line 525)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 526):
        
        # Assigning a BinOp to a Name (line 526):
        # Getting the type of 'self' (line 526)
        self_136658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 21), 'self')
        # Obtaining the member 'AXESPAD' of a type (line 526)
        AXESPAD_136659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 21), self_136658, 'AXESPAD')
        # Getting the type of 'l' (line 526)
        l_136660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 36), 'l')
        # Applying the binary operator '-' (line 526)
        result_sub_136661 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 21), '-', AXESPAD_136659, l_136660)
        
        # Assigning a type to the variable 'ox' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 16), 'ox', result_sub_136661)
        # SSA join for if statement (line 525)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 527)
        self_136662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 15), 'self')
        # Obtaining the member '_loc' of a type (line 527)
        _loc_136663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 15), self_136662, '_loc')
        
        # Obtaining an instance of the builtin type 'tuple' (line 527)
        tuple_136664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 527)
        # Adding element type (line 527)
        # Getting the type of 'BEST' (line 527)
        BEST_136665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 29), 'BEST')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 29), tuple_136664, BEST_136665)
        # Adding element type (line 527)
        # Getting the type of 'UR' (line 527)
        UR_136666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 35), 'UR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 29), tuple_136664, UR_136666)
        # Adding element type (line 527)
        # Getting the type of 'LR' (line 527)
        LR_136667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 39), 'LR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 29), tuple_136664, LR_136667)
        # Adding element type (line 527)
        # Getting the type of 'R' (line 527)
        R_136668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 43), 'R')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 29), tuple_136664, R_136668)
        # Adding element type (line 527)
        # Getting the type of 'CR' (line 527)
        CR_136669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 46), 'CR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 29), tuple_136664, CR_136669)
        
        # Applying the binary operator 'in' (line 527)
        result_contains_136670 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 15), 'in', _loc_136663, tuple_136664)
        
        # Testing the type of an if condition (line 527)
        if_condition_136671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 527, 12), result_contains_136670)
        # Assigning a type to the variable 'if_condition_136671' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'if_condition_136671', if_condition_136671)
        # SSA begins for if statement (line 527)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 528):
        
        # Assigning a BinOp to a Name (line 528):
        int_136672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 21), 'int')
        # Getting the type of 'l' (line 528)
        l_136673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 26), 'l')
        # Getting the type of 'w' (line 528)
        w_136674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 30), 'w')
        # Applying the binary operator '+' (line 528)
        result_add_136675 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 26), '+', l_136673, w_136674)
        
        # Getting the type of 'self' (line 528)
        self_136676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 34), 'self')
        # Obtaining the member 'AXESPAD' of a type (line 528)
        AXESPAD_136677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 34), self_136676, 'AXESPAD')
        # Applying the binary operator '+' (line 528)
        result_add_136678 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 32), '+', result_add_136675, AXESPAD_136677)
        
        # Applying the binary operator '-' (line 528)
        result_sub_136679 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 21), '-', int_136672, result_add_136678)
        
        # Assigning a type to the variable 'ox' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 16), 'ox', result_sub_136679)
        # SSA join for if statement (line 527)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 529)
        self_136680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 15), 'self')
        # Obtaining the member '_loc' of a type (line 529)
        _loc_136681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 15), self_136680, '_loc')
        
        # Obtaining an instance of the builtin type 'tuple' (line 529)
        tuple_136682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 529)
        # Adding element type (line 529)
        # Getting the type of 'BEST' (line 529)
        BEST_136683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 29), 'BEST')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 29), tuple_136682, BEST_136683)
        # Adding element type (line 529)
        # Getting the type of 'UR' (line 529)
        UR_136684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 35), 'UR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 29), tuple_136682, UR_136684)
        # Adding element type (line 529)
        # Getting the type of 'UL' (line 529)
        UL_136685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 39), 'UL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 29), tuple_136682, UL_136685)
        # Adding element type (line 529)
        # Getting the type of 'UC' (line 529)
        UC_136686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 43), 'UC')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 29), tuple_136682, UC_136686)
        
        # Applying the binary operator 'in' (line 529)
        result_contains_136687 = python_operator(stypy.reporting.localization.Localization(__file__, 529, 15), 'in', _loc_136681, tuple_136682)
        
        # Testing the type of an if condition (line 529)
        if_condition_136688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 529, 12), result_contains_136687)
        # Assigning a type to the variable 'if_condition_136688' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'if_condition_136688', if_condition_136688)
        # SSA begins for if statement (line 529)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 530):
        
        # Assigning a BinOp to a Name (line 530):
        int_136689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 21), 'int')
        # Getting the type of 'b' (line 530)
        b_136690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 26), 'b')
        # Getting the type of 'h' (line 530)
        h_136691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 30), 'h')
        # Applying the binary operator '+' (line 530)
        result_add_136692 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 26), '+', b_136690, h_136691)
        
        # Getting the type of 'self' (line 530)
        self_136693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 34), 'self')
        # Obtaining the member 'AXESPAD' of a type (line 530)
        AXESPAD_136694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 34), self_136693, 'AXESPAD')
        # Applying the binary operator '+' (line 530)
        result_add_136695 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 32), '+', result_add_136692, AXESPAD_136694)
        
        # Applying the binary operator '-' (line 530)
        result_sub_136696 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 21), '-', int_136689, result_add_136695)
        
        # Assigning a type to the variable 'oy' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 16), 'oy', result_sub_136696)
        # SSA join for if statement (line 529)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 531)
        self_136697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 15), 'self')
        # Obtaining the member '_loc' of a type (line 531)
        _loc_136698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 15), self_136697, '_loc')
        
        # Obtaining an instance of the builtin type 'tuple' (line 531)
        tuple_136699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 531)
        # Adding element type (line 531)
        # Getting the type of 'LL' (line 531)
        LL_136700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 29), 'LL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 29), tuple_136699, LL_136700)
        # Adding element type (line 531)
        # Getting the type of 'LR' (line 531)
        LR_136701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 33), 'LR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 29), tuple_136699, LR_136701)
        # Adding element type (line 531)
        # Getting the type of 'LC' (line 531)
        LC_136702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 37), 'LC')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 29), tuple_136699, LC_136702)
        
        # Applying the binary operator 'in' (line 531)
        result_contains_136703 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 15), 'in', _loc_136698, tuple_136699)
        
        # Testing the type of an if condition (line 531)
        if_condition_136704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 531, 12), result_contains_136703)
        # Assigning a type to the variable 'if_condition_136704' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'if_condition_136704', if_condition_136704)
        # SSA begins for if statement (line 531)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 532):
        
        # Assigning a BinOp to a Name (line 532):
        # Getting the type of 'self' (line 532)
        self_136705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 21), 'self')
        # Obtaining the member 'AXESPAD' of a type (line 532)
        AXESPAD_136706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 21), self_136705, 'AXESPAD')
        # Getting the type of 'b' (line 532)
        b_136707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 36), 'b')
        # Applying the binary operator '-' (line 532)
        result_sub_136708 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 21), '-', AXESPAD_136706, b_136707)
        
        # Assigning a type to the variable 'oy' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 16), 'oy', result_sub_136708)
        # SSA join for if statement (line 531)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 533)
        self_136709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 15), 'self')
        # Obtaining the member '_loc' of a type (line 533)
        _loc_136710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 15), self_136709, '_loc')
        
        # Obtaining an instance of the builtin type 'tuple' (line 533)
        tuple_136711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 533)
        # Adding element type (line 533)
        # Getting the type of 'LC' (line 533)
        LC_136712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 29), 'LC')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 29), tuple_136711, LC_136712)
        # Adding element type (line 533)
        # Getting the type of 'UC' (line 533)
        UC_136713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 33), 'UC')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 29), tuple_136711, UC_136713)
        # Adding element type (line 533)
        # Getting the type of 'C' (line 533)
        C_136714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 37), 'C')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 29), tuple_136711, C_136714)
        
        # Applying the binary operator 'in' (line 533)
        result_contains_136715 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 15), 'in', _loc_136710, tuple_136711)
        
        # Testing the type of an if condition (line 533)
        if_condition_136716 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 533, 12), result_contains_136715)
        # Assigning a type to the variable 'if_condition_136716' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'if_condition_136716', if_condition_136716)
        # SSA begins for if statement (line 533)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 534):
        
        # Assigning a BinOp to a Name (line 534):
        float_136717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 22), 'float')
        # Getting the type of 'w' (line 534)
        w_136718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 28), 'w')
        int_136719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 32), 'int')
        # Applying the binary operator 'div' (line 534)
        result_div_136720 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 28), 'div', w_136718, int_136719)
        
        # Applying the binary operator '-' (line 534)
        result_sub_136721 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 22), '-', float_136717, result_div_136720)
        
        # Getting the type of 'l' (line 534)
        l_136722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 37), 'l')
        # Applying the binary operator '-' (line 534)
        result_sub_136723 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 21), '-', result_sub_136721, l_136722)
        
        # Assigning a type to the variable 'ox' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'ox', result_sub_136723)
        # SSA join for if statement (line 533)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 535)
        self_136724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 15), 'self')
        # Obtaining the member '_loc' of a type (line 535)
        _loc_136725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 15), self_136724, '_loc')
        
        # Obtaining an instance of the builtin type 'tuple' (line 535)
        tuple_136726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 535)
        # Adding element type (line 535)
        # Getting the type of 'CL' (line 535)
        CL_136727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 29), 'CL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 29), tuple_136726, CL_136727)
        # Adding element type (line 535)
        # Getting the type of 'CR' (line 535)
        CR_136728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 33), 'CR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 29), tuple_136726, CR_136728)
        # Adding element type (line 535)
        # Getting the type of 'C' (line 535)
        C_136729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 37), 'C')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 29), tuple_136726, C_136729)
        
        # Applying the binary operator 'in' (line 535)
        result_contains_136730 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 15), 'in', _loc_136725, tuple_136726)
        
        # Testing the type of an if condition (line 535)
        if_condition_136731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 12), result_contains_136730)
        # Assigning a type to the variable 'if_condition_136731' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'if_condition_136731', if_condition_136731)
        # SSA begins for if statement (line 535)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 536):
        
        # Assigning a BinOp to a Name (line 536):
        float_136732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 22), 'float')
        # Getting the type of 'h' (line 536)
        h_136733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 28), 'h')
        int_136734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 32), 'int')
        # Applying the binary operator 'div' (line 536)
        result_div_136735 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 28), 'div', h_136733, int_136734)
        
        # Applying the binary operator '-' (line 536)
        result_sub_136736 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 22), '-', float_136732, result_div_136735)
        
        # Getting the type of 'b' (line 536)
        b_136737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 37), 'b')
        # Applying the binary operator '-' (line 536)
        result_sub_136738 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 21), '-', result_sub_136736, b_136737)
        
        # Assigning a type to the variable 'oy' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'oy', result_sub_136738)
        # SSA join for if statement (line 535)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 538)
        self_136739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 15), 'self')
        # Obtaining the member '_loc' of a type (line 538)
        _loc_136740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 15), self_136739, '_loc')
        
        # Obtaining an instance of the builtin type 'tuple' (line 538)
        tuple_136741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 538)
        # Adding element type (line 538)
        # Getting the type of 'TL' (line 538)
        TL_136742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 29), 'TL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 29), tuple_136741, TL_136742)
        # Adding element type (line 538)
        # Getting the type of 'BL' (line 538)
        BL_136743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 33), 'BL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 29), tuple_136741, BL_136743)
        # Adding element type (line 538)
        # Getting the type of 'L' (line 538)
        L_136744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 37), 'L')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 29), tuple_136741, L_136744)
        
        # Applying the binary operator 'in' (line 538)
        result_contains_136745 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 15), 'in', _loc_136740, tuple_136741)
        
        # Testing the type of an if condition (line 538)
        if_condition_136746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 538, 12), result_contains_136745)
        # Assigning a type to the variable 'if_condition_136746' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'if_condition_136746', if_condition_136746)
        # SSA begins for if statement (line 538)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a UnaryOp to a Name (line 539):
        
        # Assigning a UnaryOp to a Name (line 539):
        
        # Getting the type of 'l' (line 539)
        l_136747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 24), 'l')
        # Getting the type of 'w' (line 539)
        w_136748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 28), 'w')
        # Applying the binary operator '+' (line 539)
        result_add_136749 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 24), '+', l_136747, w_136748)
        
        # Applying the 'usub' unary operator (line 539)
        result___neg___136750 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 21), 'usub', result_add_136749)
        
        # Assigning a type to the variable 'ox' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 16), 'ox', result___neg___136750)
        # SSA join for if statement (line 538)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 540)
        self_136751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 15), 'self')
        # Obtaining the member '_loc' of a type (line 540)
        _loc_136752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 15), self_136751, '_loc')
        
        # Obtaining an instance of the builtin type 'tuple' (line 540)
        tuple_136753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 540)
        # Adding element type (line 540)
        # Getting the type of 'TR' (line 540)
        TR_136754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 29), 'TR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 29), tuple_136753, TR_136754)
        # Adding element type (line 540)
        # Getting the type of 'BR' (line 540)
        BR_136755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 33), 'BR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 29), tuple_136753, BR_136755)
        # Adding element type (line 540)
        # Getting the type of 'R' (line 540)
        R_136756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 37), 'R')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 29), tuple_136753, R_136756)
        
        # Applying the binary operator 'in' (line 540)
        result_contains_136757 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 15), 'in', _loc_136752, tuple_136753)
        
        # Testing the type of an if condition (line 540)
        if_condition_136758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 540, 12), result_contains_136757)
        # Assigning a type to the variable 'if_condition_136758' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'if_condition_136758', if_condition_136758)
        # SSA begins for if statement (line 540)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 541):
        
        # Assigning a BinOp to a Name (line 541):
        float_136759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 21), 'float')
        # Getting the type of 'l' (line 541)
        l_136760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 27), 'l')
        # Applying the binary operator '-' (line 541)
        result_sub_136761 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 21), '-', float_136759, l_136760)
        
        # Assigning a type to the variable 'ox' (line 541)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 16), 'ox', result_sub_136761)
        # SSA join for if statement (line 540)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 542)
        self_136762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 15), 'self')
        # Obtaining the member '_loc' of a type (line 542)
        _loc_136763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 15), self_136762, '_loc')
        
        # Obtaining an instance of the builtin type 'tuple' (line 542)
        tuple_136764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 542)
        # Adding element type (line 542)
        # Getting the type of 'TR' (line 542)
        TR_136765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 29), 'TR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 29), tuple_136764, TR_136765)
        # Adding element type (line 542)
        # Getting the type of 'TL' (line 542)
        TL_136766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 33), 'TL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 29), tuple_136764, TL_136766)
        # Adding element type (line 542)
        # Getting the type of 'T' (line 542)
        T_136767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 37), 'T')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 29), tuple_136764, T_136767)
        
        # Applying the binary operator 'in' (line 542)
        result_contains_136768 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 15), 'in', _loc_136763, tuple_136764)
        
        # Testing the type of an if condition (line 542)
        if_condition_136769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 542, 12), result_contains_136768)
        # Assigning a type to the variable 'if_condition_136769' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'if_condition_136769', if_condition_136769)
        # SSA begins for if statement (line 542)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 543):
        
        # Assigning a BinOp to a Name (line 543):
        float_136770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 21), 'float')
        # Getting the type of 'b' (line 543)
        b_136771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 27), 'b')
        # Applying the binary operator '-' (line 543)
        result_sub_136772 = python_operator(stypy.reporting.localization.Localization(__file__, 543, 21), '-', float_136770, b_136771)
        
        # Assigning a type to the variable 'oy' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 16), 'oy', result_sub_136772)
        # SSA join for if statement (line 542)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 544)
        self_136773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 15), 'self')
        # Obtaining the member '_loc' of a type (line 544)
        _loc_136774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 15), self_136773, '_loc')
        
        # Obtaining an instance of the builtin type 'tuple' (line 544)
        tuple_136775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 544)
        # Adding element type (line 544)
        # Getting the type of 'BL' (line 544)
        BL_136776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 29), 'BL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 29), tuple_136775, BL_136776)
        # Adding element type (line 544)
        # Getting the type of 'BR' (line 544)
        BR_136777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 33), 'BR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 29), tuple_136775, BR_136777)
        # Adding element type (line 544)
        # Getting the type of 'B' (line 544)
        B_136778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 37), 'B')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 29), tuple_136775, B_136778)
        
        # Applying the binary operator 'in' (line 544)
        result_contains_136779 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 15), 'in', _loc_136774, tuple_136775)
        
        # Testing the type of an if condition (line 544)
        if_condition_136780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 544, 12), result_contains_136779)
        # Assigning a type to the variable 'if_condition_136780' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'if_condition_136780', if_condition_136780)
        # SSA begins for if statement (line 544)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a UnaryOp to a Name (line 545):
        
        # Assigning a UnaryOp to a Name (line 545):
        
        # Getting the type of 'b' (line 545)
        b_136781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 24), 'b')
        # Getting the type of 'h' (line 545)
        h_136782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 28), 'h')
        # Applying the binary operator '+' (line 545)
        result_add_136783 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 24), '+', b_136781, h_136782)
        
        # Applying the 'usub' unary operator (line 545)
        result___neg___136784 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 21), 'usub', result_add_136783)
        
        # Assigning a type to the variable 'oy' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'oy', result___neg___136784)
        # SSA join for if statement (line 544)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 511)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _offset(...): (line 547)
        # Processing the call arguments (line 547)
        # Getting the type of 'ox' (line 547)
        ox_136787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 21), 'ox', False)
        # Getting the type of 'oy' (line 547)
        oy_136788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 25), 'oy', False)
        # Processing the call keyword arguments (line 547)
        kwargs_136789 = {}
        # Getting the type of 'self' (line 547)
        self_136785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'self', False)
        # Obtaining the member '_offset' of a type (line 547)
        _offset_136786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 8), self_136785, '_offset')
        # Calling _offset(args, kwargs) (line 547)
        _offset_call_result_136790 = invoke(stypy.reporting.localization.Localization(__file__, 547, 8), _offset_136786, *[ox_136787, oy_136788], **kwargs_136789)
        
        
        # ################# End of '_update_positions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update_positions' in the type store
        # Getting the type of 'stypy_return_type' (line 494)
        stypy_return_type_136791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136791)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update_positions'
        return stypy_return_type_136791


    @norecursion
    def get_celld(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_celld'
        module_type_store = module_type_store.open_function_context('get_celld', 549, 4, False)
        # Assigning a type to the variable 'self' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Table.get_celld.__dict__.__setitem__('stypy_localization', localization)
        Table.get_celld.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Table.get_celld.__dict__.__setitem__('stypy_type_store', module_type_store)
        Table.get_celld.__dict__.__setitem__('stypy_function_name', 'Table.get_celld')
        Table.get_celld.__dict__.__setitem__('stypy_param_names_list', [])
        Table.get_celld.__dict__.__setitem__('stypy_varargs_param_name', None)
        Table.get_celld.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Table.get_celld.__dict__.__setitem__('stypy_call_defaults', defaults)
        Table.get_celld.__dict__.__setitem__('stypy_call_varargs', varargs)
        Table.get_celld.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Table.get_celld.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Table.get_celld', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_celld', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_celld(...)' code ##################

        unicode_136792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 8), 'unicode', u'return a dict of cells in the table')
        # Getting the type of 'self' (line 551)
        self_136793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 15), 'self')
        # Obtaining the member '_cells' of a type (line 551)
        _cells_136794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 15), self_136793, '_cells')
        # Assigning a type to the variable 'stypy_return_type' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'stypy_return_type', _cells_136794)
        
        # ################# End of 'get_celld(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_celld' in the type store
        # Getting the type of 'stypy_return_type' (line 549)
        stypy_return_type_136795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_136795)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_celld'
        return stypy_return_type_136795


# Assigning a type to the variable 'Table' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'Table', Table)

# Assigning a Dict to a Name (line 228):

# Obtaining an instance of the builtin type 'dict' (line 228)
dict_136796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 228)
# Adding element type (key, value) (line 228)
unicode_136797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 13), 'unicode', u'best')
int_136798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 21), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136797, int_136798))
# Adding element type (key, value) (line 228)
unicode_136799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 13), 'unicode', u'upper right')
int_136800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136799, int_136800))
# Adding element type (key, value) (line 228)
unicode_136801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 13), 'unicode', u'upper left')
int_136802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136801, int_136802))
# Adding element type (key, value) (line 228)
unicode_136803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 13), 'unicode', u'lower left')
int_136804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136803, int_136804))
# Adding element type (key, value) (line 228)
unicode_136805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 13), 'unicode', u'lower right')
int_136806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136805, int_136806))
# Adding element type (key, value) (line 228)
unicode_136807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 13), 'unicode', u'center left')
int_136808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136807, int_136808))
# Adding element type (key, value) (line 228)
unicode_136809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 13), 'unicode', u'center right')
int_136810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136809, int_136810))
# Adding element type (key, value) (line 228)
unicode_136811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 13), 'unicode', u'lower center')
int_136812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136811, int_136812))
# Adding element type (key, value) (line 228)
unicode_136813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 13), 'unicode', u'upper center')
int_136814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136813, int_136814))
# Adding element type (key, value) (line 228)
unicode_136815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 13), 'unicode', u'center')
int_136816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136815, int_136816))
# Adding element type (key, value) (line 228)
unicode_136817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 13), 'unicode', u'top right')
int_136818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136817, int_136818))
# Adding element type (key, value) (line 228)
unicode_136819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 13), 'unicode', u'top left')
int_136820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136819, int_136820))
# Adding element type (key, value) (line 228)
unicode_136821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 13), 'unicode', u'bottom left')
int_136822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136821, int_136822))
# Adding element type (key, value) (line 228)
unicode_136823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 13), 'unicode', u'bottom right')
int_136824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136823, int_136824))
# Adding element type (key, value) (line 228)
unicode_136825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 13), 'unicode', u'right')
int_136826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136825, int_136826))
# Adding element type (key, value) (line 228)
unicode_136827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 13), 'unicode', u'left')
int_136828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136827, int_136828))
# Adding element type (key, value) (line 228)
unicode_136829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 13), 'unicode', u'top')
int_136830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136829, int_136830))
# Adding element type (key, value) (line 228)
unicode_136831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 13), 'unicode', u'bottom')
int_136832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 12), dict_136796, (unicode_136831, int_136832))

# Getting the type of 'Table'
Table_136833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Table')
# Setting the type of the member 'codes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Table_136833, 'codes', dict_136796)

# Assigning a Num to a Name (line 248):
int_136834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 15), 'int')
# Getting the type of 'Table'
Table_136835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Table')
# Setting the type of the member 'FONTSIZE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Table_136835, 'FONTSIZE', int_136834)

# Assigning a Num to a Name (line 249):
float_136836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 14), 'float')
# Getting the type of 'Table'
Table_136837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Table')
# Setting the type of the member 'AXESPAD' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Table_136837, 'AXESPAD', float_136836)

# Assigning a Name to a Name (line 358):
# Getting the type of 'Table'
Table_136838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Table')
# Obtaining the member 'get_children' of a type
get_children_136839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Table_136838, 'get_children')
# Getting the type of 'Table'
Table_136840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Table')
# Setting the type of the member 'get_child_artists' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Table_136840, 'get_child_artists', get_children_136839)

@norecursion
def table(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 555)
    None_136841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 19), 'None')
    # Getting the type of 'None' (line 555)
    None_136842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 37), 'None')
    unicode_136843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 18), 'unicode', u'right')
    # Getting the type of 'None' (line 556)
    None_136844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 37), 'None')
    # Getting the type of 'None' (line 557)
    None_136845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 20), 'None')
    # Getting the type of 'None' (line 557)
    None_136846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 37), 'None')
    unicode_136847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 50), 'unicode', u'left')
    # Getting the type of 'None' (line 558)
    None_136848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 20), 'None')
    # Getting the type of 'None' (line 558)
    None_136849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 37), 'None')
    unicode_136850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 50), 'unicode', u'center')
    unicode_136851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 14), 'unicode', u'bottom')
    # Getting the type of 'None' (line 559)
    None_136852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 29), 'None')
    unicode_136853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 41), 'unicode', u'closed')
    defaults = [None_136841, None_136842, unicode_136843, None_136844, None_136845, None_136846, unicode_136847, None_136848, None_136849, unicode_136850, unicode_136851, None_136852, unicode_136853]
    # Create a new context for function 'table'
    module_type_store = module_type_store.open_function_context('table', 554, 0, False)
    
    # Passed parameters checking function
    table.stypy_localization = localization
    table.stypy_type_of_self = None
    table.stypy_type_store = module_type_store
    table.stypy_function_name = 'table'
    table.stypy_param_names_list = ['ax', 'cellText', 'cellColours', 'cellLoc', 'colWidths', 'rowLabels', 'rowColours', 'rowLoc', 'colLabels', 'colColours', 'colLoc', 'loc', 'bbox', 'edges']
    table.stypy_varargs_param_name = None
    table.stypy_kwargs_param_name = 'kwargs'
    table.stypy_call_defaults = defaults
    table.stypy_call_varargs = varargs
    table.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'table', ['ax', 'cellText', 'cellColours', 'cellLoc', 'colWidths', 'rowLabels', 'rowColours', 'rowLoc', 'colLabels', 'colColours', 'colLoc', 'loc', 'bbox', 'edges'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'table', localization, ['ax', 'cellText', 'cellColours', 'cellLoc', 'colWidths', 'rowLabels', 'rowColours', 'rowLoc', 'colLabels', 'colColours', 'colLoc', 'loc', 'bbox', 'edges'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'table(...)' code ##################

    unicode_136854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, (-1)), 'unicode', u"\n    TABLE(cellText=None, cellColours=None,\n          cellLoc='right', colWidths=None,\n          rowLabels=None, rowColours=None, rowLoc='left',\n          colLabels=None, colColours=None, colLoc='center',\n          loc='bottom', bbox=None, edges='closed')\n\n    Factory function to generate a Table instance.\n\n    Thanks to John Gill for providing the class and table.\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'cellColours' (line 573)
    cellColours_136855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 7), 'cellColours')
    # Getting the type of 'None' (line 573)
    None_136856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 22), 'None')
    # Applying the binary operator 'is' (line 573)
    result_is__136857 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 7), 'is', cellColours_136855, None_136856)
    
    
    # Getting the type of 'cellText' (line 573)
    cellText_136858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 31), 'cellText')
    # Getting the type of 'None' (line 573)
    None_136859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 43), 'None')
    # Applying the binary operator 'is' (line 573)
    result_is__136860 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 31), 'is', cellText_136858, None_136859)
    
    # Applying the binary operator 'and' (line 573)
    result_and_keyword_136861 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 7), 'and', result_is__136857, result_is__136860)
    
    # Testing the type of an if condition (line 573)
    if_condition_136862 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 573, 4), result_and_keyword_136861)
    # Assigning a type to the variable 'if_condition_136862' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'if_condition_136862', if_condition_136862)
    # SSA begins for if statement (line 573)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 574)
    # Processing the call arguments (line 574)
    unicode_136864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 25), 'unicode', u'At least one argument from "cellColours" or "cellText" must be provided to create a table.')
    # Processing the call keyword arguments (line 574)
    kwargs_136865 = {}
    # Getting the type of 'ValueError' (line 574)
    ValueError_136863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 574)
    ValueError_call_result_136866 = invoke(stypy.reporting.localization.Localization(__file__, 574, 14), ValueError_136863, *[unicode_136864], **kwargs_136865)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 574, 8), ValueError_call_result_136866, 'raise parameter', BaseException)
    # SSA join for if statement (line 573)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 578)
    # Getting the type of 'cellText' (line 578)
    cellText_136867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 7), 'cellText')
    # Getting the type of 'None' (line 578)
    None_136868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 19), 'None')
    
    (may_be_136869, more_types_in_union_136870) = may_be_none(cellText_136867, None_136868)

    if may_be_136869:

        if more_types_in_union_136870:
            # Runtime conditional SSA (line 578)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 580):
        
        # Assigning a Call to a Name (line 580):
        
        # Call to len(...): (line 580)
        # Processing the call arguments (line 580)
        # Getting the type of 'cellColours' (line 580)
        cellColours_136872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 19), 'cellColours', False)
        # Processing the call keyword arguments (line 580)
        kwargs_136873 = {}
        # Getting the type of 'len' (line 580)
        len_136871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 15), 'len', False)
        # Calling len(args, kwargs) (line 580)
        len_call_result_136874 = invoke(stypy.reporting.localization.Localization(__file__, 580, 15), len_136871, *[cellColours_136872], **kwargs_136873)
        
        # Assigning a type to the variable 'rows' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'rows', len_call_result_136874)
        
        # Assigning a Call to a Name (line 581):
        
        # Assigning a Call to a Name (line 581):
        
        # Call to len(...): (line 581)
        # Processing the call arguments (line 581)
        
        # Obtaining the type of the subscript
        int_136876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 31), 'int')
        # Getting the type of 'cellColours' (line 581)
        cellColours_136877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 19), 'cellColours', False)
        # Obtaining the member '__getitem__' of a type (line 581)
        getitem___136878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 19), cellColours_136877, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 581)
        subscript_call_result_136879 = invoke(stypy.reporting.localization.Localization(__file__, 581, 19), getitem___136878, int_136876)
        
        # Processing the call keyword arguments (line 581)
        kwargs_136880 = {}
        # Getting the type of 'len' (line 581)
        len_136875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 15), 'len', False)
        # Calling len(args, kwargs) (line 581)
        len_call_result_136881 = invoke(stypy.reporting.localization.Localization(__file__, 581, 15), len_136875, *[subscript_call_result_136879], **kwargs_136880)
        
        # Assigning a type to the variable 'cols' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'cols', len_call_result_136881)
        
        # Assigning a BinOp to a Name (line 582):
        
        # Assigning a BinOp to a Name (line 582):
        
        # Obtaining an instance of the builtin type 'list' (line 582)
        list_136882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 582)
        # Adding element type (line 582)
        
        # Obtaining an instance of the builtin type 'list' (line 582)
        list_136883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 582)
        # Adding element type (line 582)
        unicode_136884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 21), 'unicode', u'')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 20), list_136883, unicode_136884)
        
        # Getting the type of 'cols' (line 582)
        cols_136885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 27), 'cols')
        # Applying the binary operator '*' (line 582)
        result_mul_136886 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 20), '*', list_136883, cols_136885)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 19), list_136882, result_mul_136886)
        
        # Getting the type of 'rows' (line 582)
        rows_136887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 35), 'rows')
        # Applying the binary operator '*' (line 582)
        result_mul_136888 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 19), '*', list_136882, rows_136887)
        
        # Assigning a type to the variable 'cellText' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'cellText', result_mul_136888)

        if more_types_in_union_136870:
            # SSA join for if statement (line 578)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 584):
    
    # Assigning a Call to a Name (line 584):
    
    # Call to len(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'cellText' (line 584)
    cellText_136890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 15), 'cellText', False)
    # Processing the call keyword arguments (line 584)
    kwargs_136891 = {}
    # Getting the type of 'len' (line 584)
    len_136889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 11), 'len', False)
    # Calling len(args, kwargs) (line 584)
    len_call_result_136892 = invoke(stypy.reporting.localization.Localization(__file__, 584, 11), len_136889, *[cellText_136890], **kwargs_136891)
    
    # Assigning a type to the variable 'rows' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'rows', len_call_result_136892)
    
    # Assigning a Call to a Name (line 585):
    
    # Assigning a Call to a Name (line 585):
    
    # Call to len(...): (line 585)
    # Processing the call arguments (line 585)
    
    # Obtaining the type of the subscript
    int_136894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 24), 'int')
    # Getting the type of 'cellText' (line 585)
    cellText_136895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 15), 'cellText', False)
    # Obtaining the member '__getitem__' of a type (line 585)
    getitem___136896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 15), cellText_136895, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 585)
    subscript_call_result_136897 = invoke(stypy.reporting.localization.Localization(__file__, 585, 15), getitem___136896, int_136894)
    
    # Processing the call keyword arguments (line 585)
    kwargs_136898 = {}
    # Getting the type of 'len' (line 585)
    len_136893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 11), 'len', False)
    # Calling len(args, kwargs) (line 585)
    len_call_result_136899 = invoke(stypy.reporting.localization.Localization(__file__, 585, 11), len_136893, *[subscript_call_result_136897], **kwargs_136898)
    
    # Assigning a type to the variable 'cols' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'cols', len_call_result_136899)
    
    # Getting the type of 'cellText' (line 586)
    cellText_136900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 15), 'cellText')
    # Testing the type of a for loop iterable (line 586)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 586, 4), cellText_136900)
    # Getting the type of the for loop variable (line 586)
    for_loop_var_136901 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 586, 4), cellText_136900)
    # Assigning a type to the variable 'row' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'row', for_loop_var_136901)
    # SSA begins for a for statement (line 586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to len(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'row' (line 587)
    row_136903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 15), 'row', False)
    # Processing the call keyword arguments (line 587)
    kwargs_136904 = {}
    # Getting the type of 'len' (line 587)
    len_136902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 11), 'len', False)
    # Calling len(args, kwargs) (line 587)
    len_call_result_136905 = invoke(stypy.reporting.localization.Localization(__file__, 587, 11), len_136902, *[row_136903], **kwargs_136904)
    
    # Getting the type of 'cols' (line 587)
    cols_136906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 23), 'cols')
    # Applying the binary operator '!=' (line 587)
    result_ne_136907 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 11), '!=', len_call_result_136905, cols_136906)
    
    # Testing the type of an if condition (line 587)
    if_condition_136908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 587, 8), result_ne_136907)
    # Assigning a type to the variable 'if_condition_136908' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'if_condition_136908', if_condition_136908)
    # SSA begins for if statement (line 587)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 588):
    
    # Assigning a Str to a Name (line 588):
    unicode_136909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 18), 'unicode', u"Each row in 'cellText' must have {0} columns")
    # Assigning a type to the variable 'msg' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'msg', unicode_136909)
    
    # Call to ValueError(...): (line 589)
    # Processing the call arguments (line 589)
    
    # Call to format(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'cols' (line 589)
    cols_136913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 40), 'cols', False)
    # Processing the call keyword arguments (line 589)
    kwargs_136914 = {}
    # Getting the type of 'msg' (line 589)
    msg_136911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 29), 'msg', False)
    # Obtaining the member 'format' of a type (line 589)
    format_136912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 29), msg_136911, 'format')
    # Calling format(args, kwargs) (line 589)
    format_call_result_136915 = invoke(stypy.reporting.localization.Localization(__file__, 589, 29), format_136912, *[cols_136913], **kwargs_136914)
    
    # Processing the call keyword arguments (line 589)
    kwargs_136916 = {}
    # Getting the type of 'ValueError' (line 589)
    ValueError_136910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 589)
    ValueError_call_result_136917 = invoke(stypy.reporting.localization.Localization(__file__, 589, 18), ValueError_136910, *[format_call_result_136915], **kwargs_136916)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 589, 12), ValueError_call_result_136917, 'raise parameter', BaseException)
    # SSA join for if statement (line 587)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 591)
    # Getting the type of 'cellColours' (line 591)
    cellColours_136918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'cellColours')
    # Getting the type of 'None' (line 591)
    None_136919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 26), 'None')
    
    (may_be_136920, more_types_in_union_136921) = may_not_be_none(cellColours_136918, None_136919)

    if may_be_136920:

        if more_types_in_union_136921:
            # Runtime conditional SSA (line 591)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 592)
        # Processing the call arguments (line 592)
        # Getting the type of 'cellColours' (line 592)
        cellColours_136923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 15), 'cellColours', False)
        # Processing the call keyword arguments (line 592)
        kwargs_136924 = {}
        # Getting the type of 'len' (line 592)
        len_136922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 11), 'len', False)
        # Calling len(args, kwargs) (line 592)
        len_call_result_136925 = invoke(stypy.reporting.localization.Localization(__file__, 592, 11), len_136922, *[cellColours_136923], **kwargs_136924)
        
        # Getting the type of 'rows' (line 592)
        rows_136926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 31), 'rows')
        # Applying the binary operator '!=' (line 592)
        result_ne_136927 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 11), '!=', len_call_result_136925, rows_136926)
        
        # Testing the type of an if condition (line 592)
        if_condition_136928 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 592, 8), result_ne_136927)
        # Assigning a type to the variable 'if_condition_136928' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'if_condition_136928', if_condition_136928)
        # SSA begins for if statement (line 592)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 593)
        # Processing the call arguments (line 593)
        
        # Call to format(...): (line 593)
        # Processing the call arguments (line 593)
        # Getting the type of 'rows' (line 593)
        rows_136932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 71), 'rows', False)
        # Processing the call keyword arguments (line 593)
        kwargs_136933 = {}
        unicode_136930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 29), 'unicode', u"'cellColours' must have {0} rows")
        # Obtaining the member 'format' of a type (line 593)
        format_136931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 29), unicode_136930, 'format')
        # Calling format(args, kwargs) (line 593)
        format_call_result_136934 = invoke(stypy.reporting.localization.Localization(__file__, 593, 29), format_136931, *[rows_136932], **kwargs_136933)
        
        # Processing the call keyword arguments (line 593)
        kwargs_136935 = {}
        # Getting the type of 'ValueError' (line 593)
        ValueError_136929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 593)
        ValueError_call_result_136936 = invoke(stypy.reporting.localization.Localization(__file__, 593, 18), ValueError_136929, *[format_call_result_136934], **kwargs_136935)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 593, 12), ValueError_call_result_136936, 'raise parameter', BaseException)
        # SSA join for if statement (line 592)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'cellColours' (line 594)
        cellColours_136937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 19), 'cellColours')
        # Testing the type of a for loop iterable (line 594)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 594, 8), cellColours_136937)
        # Getting the type of the for loop variable (line 594)
        for_loop_var_136938 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 594, 8), cellColours_136937)
        # Assigning a type to the variable 'row' (line 594)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'row', for_loop_var_136938)
        # SSA begins for a for statement (line 594)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to len(...): (line 595)
        # Processing the call arguments (line 595)
        # Getting the type of 'row' (line 595)
        row_136940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 19), 'row', False)
        # Processing the call keyword arguments (line 595)
        kwargs_136941 = {}
        # Getting the type of 'len' (line 595)
        len_136939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 15), 'len', False)
        # Calling len(args, kwargs) (line 595)
        len_call_result_136942 = invoke(stypy.reporting.localization.Localization(__file__, 595, 15), len_136939, *[row_136940], **kwargs_136941)
        
        # Getting the type of 'cols' (line 595)
        cols_136943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 27), 'cols')
        # Applying the binary operator '!=' (line 595)
        result_ne_136944 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 15), '!=', len_call_result_136942, cols_136943)
        
        # Testing the type of an if condition (line 595)
        if_condition_136945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 595, 12), result_ne_136944)
        # Assigning a type to the variable 'if_condition_136945' (line 595)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'if_condition_136945', if_condition_136945)
        # SSA begins for if statement (line 595)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 596):
        
        # Assigning a Str to a Name (line 596):
        unicode_136946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 22), 'unicode', u"Each row in 'cellColours' must have {0} columns")
        # Assigning a type to the variable 'msg' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'msg', unicode_136946)
        
        # Call to ValueError(...): (line 597)
        # Processing the call arguments (line 597)
        
        # Call to format(...): (line 597)
        # Processing the call arguments (line 597)
        # Getting the type of 'cols' (line 597)
        cols_136950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 44), 'cols', False)
        # Processing the call keyword arguments (line 597)
        kwargs_136951 = {}
        # Getting the type of 'msg' (line 597)
        msg_136948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 33), 'msg', False)
        # Obtaining the member 'format' of a type (line 597)
        format_136949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 33), msg_136948, 'format')
        # Calling format(args, kwargs) (line 597)
        format_call_result_136952 = invoke(stypy.reporting.localization.Localization(__file__, 597, 33), format_136949, *[cols_136950], **kwargs_136951)
        
        # Processing the call keyword arguments (line 597)
        kwargs_136953 = {}
        # Getting the type of 'ValueError' (line 597)
        ValueError_136947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 597)
        ValueError_call_result_136954 = invoke(stypy.reporting.localization.Localization(__file__, 597, 22), ValueError_136947, *[format_call_result_136952], **kwargs_136953)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 597, 16), ValueError_call_result_136954, 'raise parameter', BaseException)
        # SSA join for if statement (line 595)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_136921:
            # Runtime conditional SSA for else branch (line 591)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_136920) or more_types_in_union_136921):
        
        # Assigning a BinOp to a Name (line 599):
        
        # Assigning a BinOp to a Name (line 599):
        
        # Obtaining an instance of the builtin type 'list' (line 599)
        list_136955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 599)
        # Adding element type (line 599)
        unicode_136956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 23), 'unicode', u'w')
        # Getting the type of 'cols' (line 599)
        cols_136957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 29), 'cols')
        # Applying the binary operator '*' (line 599)
        result_mul_136958 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 23), '*', unicode_136956, cols_136957)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 22), list_136955, result_mul_136958)
        
        # Getting the type of 'rows' (line 599)
        rows_136959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 37), 'rows')
        # Applying the binary operator '*' (line 599)
        result_mul_136960 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 22), '*', list_136955, rows_136959)
        
        # Assigning a type to the variable 'cellColours' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'cellColours', result_mul_136960)

        if (may_be_136920 and more_types_in_union_136921):
            # SSA join for if statement (line 591)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 602)
    # Getting the type of 'colWidths' (line 602)
    colWidths_136961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 7), 'colWidths')
    # Getting the type of 'None' (line 602)
    None_136962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 20), 'None')
    
    (may_be_136963, more_types_in_union_136964) = may_be_none(colWidths_136961, None_136962)

    if may_be_136963:

        if more_types_in_union_136964:
            # Runtime conditional SSA (line 602)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 603):
        
        # Assigning a BinOp to a Name (line 603):
        
        # Obtaining an instance of the builtin type 'list' (line 603)
        list_136965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 603)
        # Adding element type (line 603)
        float_136966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 21), 'float')
        # Getting the type of 'cols' (line 603)
        cols_136967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 27), 'cols')
        # Applying the binary operator 'div' (line 603)
        result_div_136968 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 21), 'div', float_136966, cols_136967)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 603, 20), list_136965, result_div_136968)
        
        # Getting the type of 'cols' (line 603)
        cols_136969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 35), 'cols')
        # Applying the binary operator '*' (line 603)
        result_mul_136970 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 20), '*', list_136965, cols_136969)
        
        # Assigning a type to the variable 'colWidths' (line 603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'colWidths', result_mul_136970)

        if more_types_in_union_136964:
            # SSA join for if statement (line 602)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Num to a Name (line 607):
    
    # Assigning a Num to a Name (line 607):
    int_136971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 20), 'int')
    # Assigning a type to the variable 'rowLabelWidth' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'rowLabelWidth', int_136971)
    
    # Type idiom detected: calculating its left and rigth part (line 608)
    # Getting the type of 'rowLabels' (line 608)
    rowLabels_136972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 7), 'rowLabels')
    # Getting the type of 'None' (line 608)
    None_136973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 20), 'None')
    
    (may_be_136974, more_types_in_union_136975) = may_be_none(rowLabels_136972, None_136973)

    if may_be_136974:

        if more_types_in_union_136975:
            # Runtime conditional SSA (line 608)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 609)
        # Getting the type of 'rowColours' (line 609)
        rowColours_136976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'rowColours')
        # Getting the type of 'None' (line 609)
        None_136977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 29), 'None')
        
        (may_be_136978, more_types_in_union_136979) = may_not_be_none(rowColours_136976, None_136977)

        if may_be_136978:

            if more_types_in_union_136979:
                # Runtime conditional SSA (line 609)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 610):
            
            # Assigning a BinOp to a Name (line 610):
            
            # Obtaining an instance of the builtin type 'list' (line 610)
            list_136980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 610)
            # Adding element type (line 610)
            unicode_136981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 25), 'unicode', u'')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 610, 24), list_136980, unicode_136981)
            
            # Getting the type of 'rows' (line 610)
            rows_136982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 31), 'rows')
            # Applying the binary operator '*' (line 610)
            result_mul_136983 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 24), '*', list_136980, rows_136982)
            
            # Assigning a type to the variable 'rowLabels' (line 610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'rowLabels', result_mul_136983)
            
            # Assigning a Subscript to a Name (line 611):
            
            # Assigning a Subscript to a Name (line 611):
            
            # Obtaining the type of the subscript
            int_136984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 38), 'int')
            # Getting the type of 'colWidths' (line 611)
            colWidths_136985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 28), 'colWidths')
            # Obtaining the member '__getitem__' of a type (line 611)
            getitem___136986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 28), colWidths_136985, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 611)
            subscript_call_result_136987 = invoke(stypy.reporting.localization.Localization(__file__, 611, 28), getitem___136986, int_136984)
            
            # Assigning a type to the variable 'rowLabelWidth' (line 611)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'rowLabelWidth', subscript_call_result_136987)

            if more_types_in_union_136979:
                # SSA join for if statement (line 609)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_136975:
            # Runtime conditional SSA for else branch (line 608)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_136974) or more_types_in_union_136975):
        
        # Type idiom detected: calculating its left and rigth part (line 612)
        # Getting the type of 'rowColours' (line 612)
        rowColours_136988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 9), 'rowColours')
        # Getting the type of 'None' (line 612)
        None_136989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 23), 'None')
        
        (may_be_136990, more_types_in_union_136991) = may_be_none(rowColours_136988, None_136989)

        if may_be_136990:

            if more_types_in_union_136991:
                # Runtime conditional SSA (line 612)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 613):
            
            # Assigning a BinOp to a Name (line 613):
            unicode_136992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 21), 'unicode', u'w')
            # Getting the type of 'rows' (line 613)
            rows_136993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 27), 'rows')
            # Applying the binary operator '*' (line 613)
            result_mul_136994 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 21), '*', unicode_136992, rows_136993)
            
            # Assigning a type to the variable 'rowColours' (line 613)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'rowColours', result_mul_136994)

            if more_types_in_union_136991:
                # SSA join for if statement (line 612)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_136974 and more_types_in_union_136975):
            # SSA join for if statement (line 608)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 615)
    # Getting the type of 'rowLabels' (line 615)
    rowLabels_136995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'rowLabels')
    # Getting the type of 'None' (line 615)
    None_136996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 24), 'None')
    
    (may_be_136997, more_types_in_union_136998) = may_not_be_none(rowLabels_136995, None_136996)

    if may_be_136997:

        if more_types_in_union_136998:
            # Runtime conditional SSA (line 615)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 616)
        # Processing the call arguments (line 616)
        # Getting the type of 'rowLabels' (line 616)
        rowLabels_137000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 15), 'rowLabels', False)
        # Processing the call keyword arguments (line 616)
        kwargs_137001 = {}
        # Getting the type of 'len' (line 616)
        len_136999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 11), 'len', False)
        # Calling len(args, kwargs) (line 616)
        len_call_result_137002 = invoke(stypy.reporting.localization.Localization(__file__, 616, 11), len_136999, *[rowLabels_137000], **kwargs_137001)
        
        # Getting the type of 'rows' (line 616)
        rows_137003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 29), 'rows')
        # Applying the binary operator '!=' (line 616)
        result_ne_137004 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 11), '!=', len_call_result_137002, rows_137003)
        
        # Testing the type of an if condition (line 616)
        if_condition_137005 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 8), result_ne_137004)
        # Assigning a type to the variable 'if_condition_137005' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'if_condition_137005', if_condition_137005)
        # SSA begins for if statement (line 616)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 617)
        # Processing the call arguments (line 617)
        
        # Call to format(...): (line 617)
        # Processing the call arguments (line 617)
        # Getting the type of 'rows' (line 617)
        rows_137009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 72), 'rows', False)
        # Processing the call keyword arguments (line 617)
        kwargs_137010 = {}
        unicode_137007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 29), 'unicode', u"'rowLabels' must be of length {0}")
        # Obtaining the member 'format' of a type (line 617)
        format_137008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 29), unicode_137007, 'format')
        # Calling format(args, kwargs) (line 617)
        format_call_result_137011 = invoke(stypy.reporting.localization.Localization(__file__, 617, 29), format_137008, *[rows_137009], **kwargs_137010)
        
        # Processing the call keyword arguments (line 617)
        kwargs_137012 = {}
        # Getting the type of 'ValueError' (line 617)
        ValueError_137006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 617)
        ValueError_call_result_137013 = invoke(stypy.reporting.localization.Localization(__file__, 617, 18), ValueError_137006, *[format_call_result_137011], **kwargs_137012)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 617, 12), ValueError_call_result_137013, 'raise parameter', BaseException)
        # SSA join for if statement (line 616)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_136998:
            # SSA join for if statement (line 615)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Num to a Name (line 621):
    
    # Assigning a Num to a Name (line 621):
    int_137014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 13), 'int')
    # Assigning a type to the variable 'offset' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'offset', int_137014)
    
    # Type idiom detected: calculating its left and rigth part (line 622)
    # Getting the type of 'colLabels' (line 622)
    colLabels_137015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 7), 'colLabels')
    # Getting the type of 'None' (line 622)
    None_137016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 20), 'None')
    
    (may_be_137017, more_types_in_union_137018) = may_be_none(colLabels_137015, None_137016)

    if may_be_137017:

        if more_types_in_union_137018:
            # Runtime conditional SSA (line 622)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 623)
        # Getting the type of 'colColours' (line 623)
        colColours_137019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'colColours')
        # Getting the type of 'None' (line 623)
        None_137020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 29), 'None')
        
        (may_be_137021, more_types_in_union_137022) = may_not_be_none(colColours_137019, None_137020)

        if may_be_137021:

            if more_types_in_union_137022:
                # Runtime conditional SSA (line 623)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 624):
            
            # Assigning a BinOp to a Name (line 624):
            
            # Obtaining an instance of the builtin type 'list' (line 624)
            list_137023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 624)
            # Adding element type (line 624)
            unicode_137024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 25), 'unicode', u'')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 24), list_137023, unicode_137024)
            
            # Getting the type of 'cols' (line 624)
            cols_137025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 31), 'cols')
            # Applying the binary operator '*' (line 624)
            result_mul_137026 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 24), '*', list_137023, cols_137025)
            
            # Assigning a type to the variable 'colLabels' (line 624)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 12), 'colLabels', result_mul_137026)

            if more_types_in_union_137022:
                # Runtime conditional SSA for else branch (line 623)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_137021) or more_types_in_union_137022):
            
            # Assigning a Num to a Name (line 626):
            
            # Assigning a Num to a Name (line 626):
            int_137027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 21), 'int')
            # Assigning a type to the variable 'offset' (line 626)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 12), 'offset', int_137027)

            if (may_be_137021 and more_types_in_union_137022):
                # SSA join for if statement (line 623)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_137018:
            # Runtime conditional SSA for else branch (line 622)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_137017) or more_types_in_union_137018):
        
        # Type idiom detected: calculating its left and rigth part (line 627)
        # Getting the type of 'colColours' (line 627)
        colColours_137028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 9), 'colColours')
        # Getting the type of 'None' (line 627)
        None_137029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 23), 'None')
        
        (may_be_137030, more_types_in_union_137031) = may_be_none(colColours_137028, None_137029)

        if may_be_137030:

            if more_types_in_union_137031:
                # Runtime conditional SSA (line 627)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 628):
            
            # Assigning a BinOp to a Name (line 628):
            unicode_137032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 21), 'unicode', u'w')
            # Getting the type of 'cols' (line 628)
            cols_137033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 27), 'cols')
            # Applying the binary operator '*' (line 628)
            result_mul_137034 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 21), '*', unicode_137032, cols_137033)
            
            # Assigning a type to the variable 'colColours' (line 628)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'colColours', result_mul_137034)

            if more_types_in_union_137031:
                # SSA join for if statement (line 627)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_137017 and more_types_in_union_137018):
            # SSA join for if statement (line 622)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 631)
    # Getting the type of 'cellColours' (line 631)
    cellColours_137035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 7), 'cellColours')
    # Getting the type of 'None' (line 631)
    None_137036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 22), 'None')
    
    (may_be_137037, more_types_in_union_137038) = may_be_none(cellColours_137035, None_137036)

    if may_be_137037:

        if more_types_in_union_137038:
            # Runtime conditional SSA (line 631)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 632):
        
        # Assigning a BinOp to a Name (line 632):
        
        # Obtaining an instance of the builtin type 'list' (line 632)
        list_137039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 632)
        # Adding element type (line 632)
        unicode_137040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 23), 'unicode', u'w')
        # Getting the type of 'cols' (line 632)
        cols_137041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 29), 'cols')
        # Applying the binary operator '*' (line 632)
        result_mul_137042 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 23), '*', unicode_137040, cols_137041)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 22), list_137039, result_mul_137042)
        
        # Getting the type of 'rows' (line 632)
        rows_137043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 37), 'rows')
        # Applying the binary operator '*' (line 632)
        result_mul_137044 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 22), '*', list_137039, rows_137043)
        
        # Assigning a type to the variable 'cellColours' (line 632)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'cellColours', result_mul_137044)

        if more_types_in_union_137038:
            # SSA join for if statement (line 631)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 635):
    
    # Assigning a Call to a Name (line 635):
    
    # Call to Table(...): (line 635)
    # Processing the call arguments (line 635)
    # Getting the type of 'ax' (line 635)
    ax_137046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 18), 'ax', False)
    # Getting the type of 'loc' (line 635)
    loc_137047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 22), 'loc', False)
    # Getting the type of 'bbox' (line 635)
    bbox_137048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 27), 'bbox', False)
    # Processing the call keyword arguments (line 635)
    # Getting the type of 'kwargs' (line 635)
    kwargs_137049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 35), 'kwargs', False)
    kwargs_137050 = {'kwargs_137049': kwargs_137049}
    # Getting the type of 'Table' (line 635)
    Table_137045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 12), 'Table', False)
    # Calling Table(args, kwargs) (line 635)
    Table_call_result_137051 = invoke(stypy.reporting.localization.Localization(__file__, 635, 12), Table_137045, *[ax_137046, loc_137047, bbox_137048], **kwargs_137050)
    
    # Assigning a type to the variable 'table' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'table', Table_call_result_137051)
    
    # Assigning a Name to a Attribute (line 636):
    
    # Assigning a Name to a Attribute (line 636):
    # Getting the type of 'edges' (line 636)
    edges_137052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 18), 'edges')
    # Getting the type of 'table' (line 636)
    table_137053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'table')
    # Setting the type of the member 'edges' of a type (line 636)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 4), table_137053, 'edges', edges_137052)
    
    # Assigning a Call to a Name (line 637):
    
    # Assigning a Call to a Name (line 637):
    
    # Call to _approx_text_height(...): (line 637)
    # Processing the call keyword arguments (line 637)
    kwargs_137056 = {}
    # Getting the type of 'table' (line 637)
    table_137054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 13), 'table', False)
    # Obtaining the member '_approx_text_height' of a type (line 637)
    _approx_text_height_137055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 13), table_137054, '_approx_text_height')
    # Calling _approx_text_height(args, kwargs) (line 637)
    _approx_text_height_call_result_137057 = invoke(stypy.reporting.localization.Localization(__file__, 637, 13), _approx_text_height_137055, *[], **kwargs_137056)
    
    # Assigning a type to the variable 'height' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'height', _approx_text_height_call_result_137057)
    
    
    # Call to xrange(...): (line 640)
    # Processing the call arguments (line 640)
    # Getting the type of 'rows' (line 640)
    rows_137059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 22), 'rows', False)
    # Processing the call keyword arguments (line 640)
    kwargs_137060 = {}
    # Getting the type of 'xrange' (line 640)
    xrange_137058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 15), 'xrange', False)
    # Calling xrange(args, kwargs) (line 640)
    xrange_call_result_137061 = invoke(stypy.reporting.localization.Localization(__file__, 640, 15), xrange_137058, *[rows_137059], **kwargs_137060)
    
    # Testing the type of a for loop iterable (line 640)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 640, 4), xrange_call_result_137061)
    # Getting the type of the for loop variable (line 640)
    for_loop_var_137062 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 640, 4), xrange_call_result_137061)
    # Assigning a type to the variable 'row' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), 'row', for_loop_var_137062)
    # SSA begins for a for statement (line 640)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to xrange(...): (line 641)
    # Processing the call arguments (line 641)
    # Getting the type of 'cols' (line 641)
    cols_137064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 26), 'cols', False)
    # Processing the call keyword arguments (line 641)
    kwargs_137065 = {}
    # Getting the type of 'xrange' (line 641)
    xrange_137063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 19), 'xrange', False)
    # Calling xrange(args, kwargs) (line 641)
    xrange_call_result_137066 = invoke(stypy.reporting.localization.Localization(__file__, 641, 19), xrange_137063, *[cols_137064], **kwargs_137065)
    
    # Testing the type of a for loop iterable (line 641)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 641, 8), xrange_call_result_137066)
    # Getting the type of the for loop variable (line 641)
    for_loop_var_137067 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 641, 8), xrange_call_result_137066)
    # Assigning a type to the variable 'col' (line 641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'col', for_loop_var_137067)
    # SSA begins for a for statement (line 641)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to add_cell(...): (line 642)
    # Processing the call arguments (line 642)
    # Getting the type of 'row' (line 642)
    row_137070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 27), 'row', False)
    # Getting the type of 'offset' (line 642)
    offset_137071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 33), 'offset', False)
    # Applying the binary operator '+' (line 642)
    result_add_137072 = python_operator(stypy.reporting.localization.Localization(__file__, 642, 27), '+', row_137070, offset_137071)
    
    # Getting the type of 'col' (line 642)
    col_137073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 41), 'col', False)
    # Processing the call keyword arguments (line 642)
    
    # Obtaining the type of the subscript
    # Getting the type of 'col' (line 643)
    col_137074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 43), 'col', False)
    # Getting the type of 'colWidths' (line 643)
    colWidths_137075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 33), 'colWidths', False)
    # Obtaining the member '__getitem__' of a type (line 643)
    getitem___137076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 33), colWidths_137075, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 643)
    subscript_call_result_137077 = invoke(stypy.reporting.localization.Localization(__file__, 643, 33), getitem___137076, col_137074)
    
    keyword_137078 = subscript_call_result_137077
    # Getting the type of 'height' (line 643)
    height_137079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 56), 'height', False)
    keyword_137080 = height_137079
    
    # Obtaining the type of the subscript
    # Getting the type of 'col' (line 644)
    col_137081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 46), 'col', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 644)
    row_137082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 41), 'row', False)
    # Getting the type of 'cellText' (line 644)
    cellText_137083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 32), 'cellText', False)
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___137084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 32), cellText_137083, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_137085 = invoke(stypy.reporting.localization.Localization(__file__, 644, 32), getitem___137084, row_137082)
    
    # Obtaining the member '__getitem__' of a type (line 644)
    getitem___137086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 32), subscript_call_result_137085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 644)
    subscript_call_result_137087 = invoke(stypy.reporting.localization.Localization(__file__, 644, 32), getitem___137086, col_137081)
    
    keyword_137088 = subscript_call_result_137087
    
    # Obtaining the type of the subscript
    # Getting the type of 'col' (line 645)
    col_137089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 54), 'col', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'row' (line 645)
    row_137090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 49), 'row', False)
    # Getting the type of 'cellColours' (line 645)
    cellColours_137091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 37), 'cellColours', False)
    # Obtaining the member '__getitem__' of a type (line 645)
    getitem___137092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 37), cellColours_137091, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 645)
    subscript_call_result_137093 = invoke(stypy.reporting.localization.Localization(__file__, 645, 37), getitem___137092, row_137090)
    
    # Obtaining the member '__getitem__' of a type (line 645)
    getitem___137094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 37), subscript_call_result_137093, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 645)
    subscript_call_result_137095 = invoke(stypy.reporting.localization.Localization(__file__, 645, 37), getitem___137094, col_137089)
    
    keyword_137096 = subscript_call_result_137095
    # Getting the type of 'cellLoc' (line 646)
    cellLoc_137097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 31), 'cellLoc', False)
    keyword_137098 = cellLoc_137097
    kwargs_137099 = {'loc': keyword_137098, 'width': keyword_137078, 'text': keyword_137088, 'facecolor': keyword_137096, 'height': keyword_137080}
    # Getting the type of 'table' (line 642)
    table_137068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 12), 'table', False)
    # Obtaining the member 'add_cell' of a type (line 642)
    add_cell_137069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 12), table_137068, 'add_cell')
    # Calling add_cell(args, kwargs) (line 642)
    add_cell_call_result_137100 = invoke(stypy.reporting.localization.Localization(__file__, 642, 12), add_cell_137069, *[result_add_137072, col_137073], **kwargs_137099)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 648)
    # Getting the type of 'colLabels' (line 648)
    colLabels_137101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'colLabels')
    # Getting the type of 'None' (line 648)
    None_137102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 24), 'None')
    
    (may_be_137103, more_types_in_union_137104) = may_not_be_none(colLabels_137101, None_137102)

    if may_be_137103:

        if more_types_in_union_137104:
            # Runtime conditional SSA (line 648)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Call to xrange(...): (line 649)
        # Processing the call arguments (line 649)
        # Getting the type of 'cols' (line 649)
        cols_137106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 26), 'cols', False)
        # Processing the call keyword arguments (line 649)
        kwargs_137107 = {}
        # Getting the type of 'xrange' (line 649)
        xrange_137105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 19), 'xrange', False)
        # Calling xrange(args, kwargs) (line 649)
        xrange_call_result_137108 = invoke(stypy.reporting.localization.Localization(__file__, 649, 19), xrange_137105, *[cols_137106], **kwargs_137107)
        
        # Testing the type of a for loop iterable (line 649)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 649, 8), xrange_call_result_137108)
        # Getting the type of the for loop variable (line 649)
        for_loop_var_137109 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 649, 8), xrange_call_result_137108)
        # Assigning a type to the variable 'col' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'col', for_loop_var_137109)
        # SSA begins for a for statement (line 649)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to add_cell(...): (line 650)
        # Processing the call arguments (line 650)
        int_137112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 27), 'int')
        # Getting the type of 'col' (line 650)
        col_137113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 30), 'col', False)
        # Processing the call keyword arguments (line 650)
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 651)
        col_137114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 43), 'col', False)
        # Getting the type of 'colWidths' (line 651)
        colWidths_137115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 33), 'colWidths', False)
        # Obtaining the member '__getitem__' of a type (line 651)
        getitem___137116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 33), colWidths_137115, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 651)
        subscript_call_result_137117 = invoke(stypy.reporting.localization.Localization(__file__, 651, 33), getitem___137116, col_137114)
        
        keyword_137118 = subscript_call_result_137117
        # Getting the type of 'height' (line 651)
        height_137119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 56), 'height', False)
        keyword_137120 = height_137119
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 652)
        col_137121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 42), 'col', False)
        # Getting the type of 'colLabels' (line 652)
        colLabels_137122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 32), 'colLabels', False)
        # Obtaining the member '__getitem__' of a type (line 652)
        getitem___137123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 32), colLabels_137122, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 652)
        subscript_call_result_137124 = invoke(stypy.reporting.localization.Localization(__file__, 652, 32), getitem___137123, col_137121)
        
        keyword_137125 = subscript_call_result_137124
        
        # Obtaining the type of the subscript
        # Getting the type of 'col' (line 652)
        col_137126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 69), 'col', False)
        # Getting the type of 'colColours' (line 652)
        colColours_137127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 58), 'colColours', False)
        # Obtaining the member '__getitem__' of a type (line 652)
        getitem___137128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 58), colColours_137127, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 652)
        subscript_call_result_137129 = invoke(stypy.reporting.localization.Localization(__file__, 652, 58), getitem___137128, col_137126)
        
        keyword_137130 = subscript_call_result_137129
        # Getting the type of 'colLoc' (line 653)
        colLoc_137131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 31), 'colLoc', False)
        keyword_137132 = colLoc_137131
        kwargs_137133 = {'loc': keyword_137132, 'width': keyword_137118, 'text': keyword_137125, 'facecolor': keyword_137130, 'height': keyword_137120}
        # Getting the type of 'table' (line 650)
        table_137110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'table', False)
        # Obtaining the member 'add_cell' of a type (line 650)
        add_cell_137111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 12), table_137110, 'add_cell')
        # Calling add_cell(args, kwargs) (line 650)
        add_cell_call_result_137134 = invoke(stypy.reporting.localization.Localization(__file__, 650, 12), add_cell_137111, *[int_137112, col_137113], **kwargs_137133)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_137104:
            # SSA join for if statement (line 648)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 656)
    # Getting the type of 'rowLabels' (line 656)
    rowLabels_137135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'rowLabels')
    # Getting the type of 'None' (line 656)
    None_137136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 24), 'None')
    
    (may_be_137137, more_types_in_union_137138) = may_not_be_none(rowLabels_137135, None_137136)

    if may_be_137137:

        if more_types_in_union_137138:
            # Runtime conditional SSA (line 656)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Call to xrange(...): (line 657)
        # Processing the call arguments (line 657)
        # Getting the type of 'rows' (line 657)
        rows_137140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 26), 'rows', False)
        # Processing the call keyword arguments (line 657)
        kwargs_137141 = {}
        # Getting the type of 'xrange' (line 657)
        xrange_137139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 19), 'xrange', False)
        # Calling xrange(args, kwargs) (line 657)
        xrange_call_result_137142 = invoke(stypy.reporting.localization.Localization(__file__, 657, 19), xrange_137139, *[rows_137140], **kwargs_137141)
        
        # Testing the type of a for loop iterable (line 657)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 657, 8), xrange_call_result_137142)
        # Getting the type of the for loop variable (line 657)
        for_loop_var_137143 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 657, 8), xrange_call_result_137142)
        # Assigning a type to the variable 'row' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'row', for_loop_var_137143)
        # SSA begins for a for statement (line 657)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to add_cell(...): (line 658)
        # Processing the call arguments (line 658)
        # Getting the type of 'row' (line 658)
        row_137146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 27), 'row', False)
        # Getting the type of 'offset' (line 658)
        offset_137147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 33), 'offset', False)
        # Applying the binary operator '+' (line 658)
        result_add_137148 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 27), '+', row_137146, offset_137147)
        
        int_137149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 41), 'int')
        # Processing the call keyword arguments (line 658)
        
        # Evaluating a boolean operation
        # Getting the type of 'rowLabelWidth' (line 659)
        rowLabelWidth_137150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 33), 'rowLabelWidth', False)
        float_137151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 50), 'float')
        # Applying the binary operator 'or' (line 659)
        result_or_keyword_137152 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 33), 'or', rowLabelWidth_137150, float_137151)
        
        keyword_137153 = result_or_keyword_137152
        # Getting the type of 'height' (line 659)
        height_137154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 64), 'height', False)
        keyword_137155 = height_137154
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 660)
        row_137156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 42), 'row', False)
        # Getting the type of 'rowLabels' (line 660)
        rowLabels_137157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 32), 'rowLabels', False)
        # Obtaining the member '__getitem__' of a type (line 660)
        getitem___137158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 32), rowLabels_137157, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 660)
        subscript_call_result_137159 = invoke(stypy.reporting.localization.Localization(__file__, 660, 32), getitem___137158, row_137156)
        
        keyword_137160 = subscript_call_result_137159
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 660)
        row_137161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 69), 'row', False)
        # Getting the type of 'rowColours' (line 660)
        rowColours_137162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 58), 'rowColours', False)
        # Obtaining the member '__getitem__' of a type (line 660)
        getitem___137163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 58), rowColours_137162, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 660)
        subscript_call_result_137164 = invoke(stypy.reporting.localization.Localization(__file__, 660, 58), getitem___137163, row_137161)
        
        keyword_137165 = subscript_call_result_137164
        # Getting the type of 'rowLoc' (line 661)
        rowLoc_137166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 31), 'rowLoc', False)
        keyword_137167 = rowLoc_137166
        kwargs_137168 = {'loc': keyword_137167, 'width': keyword_137153, 'text': keyword_137160, 'facecolor': keyword_137165, 'height': keyword_137155}
        # Getting the type of 'table' (line 658)
        table_137144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'table', False)
        # Obtaining the member 'add_cell' of a type (line 658)
        add_cell_137145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 12), table_137144, 'add_cell')
        # Calling add_cell(args, kwargs) (line 658)
        add_cell_call_result_137169 = invoke(stypy.reporting.localization.Localization(__file__, 658, 12), add_cell_137145, *[result_add_137148, int_137149], **kwargs_137168)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'rowLabelWidth' (line 662)
        rowLabelWidth_137170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 11), 'rowLabelWidth')
        int_137171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 28), 'int')
        # Applying the binary operator '==' (line 662)
        result_eq_137172 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 11), '==', rowLabelWidth_137170, int_137171)
        
        # Testing the type of an if condition (line 662)
        if_condition_137173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 662, 8), result_eq_137172)
        # Assigning a type to the variable 'if_condition_137173' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'if_condition_137173', if_condition_137173)
        # SSA begins for if statement (line 662)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to auto_set_column_width(...): (line 663)
        # Processing the call arguments (line 663)
        int_137176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 40), 'int')
        # Processing the call keyword arguments (line 663)
        kwargs_137177 = {}
        # Getting the type of 'table' (line 663)
        table_137174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 12), 'table', False)
        # Obtaining the member 'auto_set_column_width' of a type (line 663)
        auto_set_column_width_137175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 12), table_137174, 'auto_set_column_width')
        # Calling auto_set_column_width(args, kwargs) (line 663)
        auto_set_column_width_call_result_137178 = invoke(stypy.reporting.localization.Localization(__file__, 663, 12), auto_set_column_width_137175, *[int_137176], **kwargs_137177)
        
        # SSA join for if statement (line 662)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_137138:
            # SSA join for if statement (line 656)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to add_table(...): (line 665)
    # Processing the call arguments (line 665)
    # Getting the type of 'table' (line 665)
    table_137181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 17), 'table', False)
    # Processing the call keyword arguments (line 665)
    kwargs_137182 = {}
    # Getting the type of 'ax' (line 665)
    ax_137179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'ax', False)
    # Obtaining the member 'add_table' of a type (line 665)
    add_table_137180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 4), ax_137179, 'add_table')
    # Calling add_table(args, kwargs) (line 665)
    add_table_call_result_137183 = invoke(stypy.reporting.localization.Localization(__file__, 665, 4), add_table_137180, *[table_137181], **kwargs_137182)
    
    # Getting the type of 'table' (line 666)
    table_137184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 11), 'table')
    # Assigning a type to the variable 'stypy_return_type' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 4), 'stypy_return_type', table_137184)
    
    # ################# End of 'table(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'table' in the type store
    # Getting the type of 'stypy_return_type' (line 554)
    stypy_return_type_137185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_137185)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'table'
    return stypy_return_type_137185

# Assigning a type to the variable 'table' (line 554)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 0), 'table', table)

# Call to update(...): (line 669)
# Processing the call keyword arguments (line 669)

# Call to kwdoc(...): (line 669)
# Processing the call arguments (line 669)
# Getting the type of 'Table' (line 669)
Table_137191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 44), 'Table', False)
# Processing the call keyword arguments (line 669)
kwargs_137192 = {}
# Getting the type of 'artist' (line 669)
artist_137189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 31), 'artist', False)
# Obtaining the member 'kwdoc' of a type (line 669)
kwdoc_137190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 31), artist_137189, 'kwdoc')
# Calling kwdoc(args, kwargs) (line 669)
kwdoc_call_result_137193 = invoke(stypy.reporting.localization.Localization(__file__, 669, 31), kwdoc_137190, *[Table_137191], **kwargs_137192)

keyword_137194 = kwdoc_call_result_137193
kwargs_137195 = {'Table': keyword_137194}
# Getting the type of 'docstring' (line 669)
docstring_137186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 0), 'docstring', False)
# Obtaining the member 'interpd' of a type (line 669)
interpd_137187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 0), docstring_137186, 'interpd')
# Obtaining the member 'update' of a type (line 669)
update_137188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 0), interpd_137187, 'update')
# Calling update(args, kwargs) (line 669)
update_call_result_137196 = invoke(stypy.reporting.localization.Localization(__file__, 669, 0), update_137188, *[], **kwargs_137195)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
