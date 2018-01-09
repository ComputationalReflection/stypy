
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Abstract base classes define the primitives for Tools.
3: These tools are used by `matplotlib.backend_managers.ToolManager`
4: 
5: :class:`ToolBase`
6:     Simple stateless tool
7: 
8: :class:`ToolToggleBase`
9:     Tool that has two states, only one Toggle tool can be
10:     active at any given time for the same
11:     `matplotlib.backend_managers.ToolManager`
12: '''
13: 
14: 
15: from matplotlib import rcParams
16: from matplotlib._pylab_helpers import Gcf
17: import matplotlib.cbook as cbook
18: from weakref import WeakKeyDictionary
19: import six
20: import time
21: import warnings
22: import numpy as np
23: 
24: 
25: class Cursors(object):
26:     '''Simple namespace for cursor reference'''
27:     HAND, POINTER, SELECT_REGION, MOVE, WAIT = list(range(5))
28: cursors = Cursors()
29: 
30: # Views positions tool
31: _views_positions = 'viewpos'
32: 
33: 
34: class ToolBase(object):
35:     '''
36:     Base tool class
37: 
38:     A base tool, only implements `trigger` method or not method at all.
39:     The tool is instantiated by `matplotlib.backend_managers.ToolManager`
40: 
41:     Attributes
42:     ----------
43:     toolmanager: `matplotlib.backend_managers.ToolManager`
44:         ToolManager that controls this Tool
45:     figure: `FigureCanvas`
46:         Figure instance that is affected by this Tool
47:     name: String
48:         Used as **Id** of the tool, has to be unique among tools of the same
49:         ToolManager
50:     '''
51: 
52:     default_keymap = None
53:     '''
54:     Keymap to associate with this tool
55: 
56:     **String**: List of comma separated keys that will be used to call this
57:     tool when the keypress event of *self.figure.canvas* is emited
58:     '''
59: 
60:     description = None
61:     '''
62:     Description of the Tool
63: 
64:     **String**: If the Tool is included in the Toolbar this text is used
65:     as a Tooltip
66:     '''
67: 
68:     image = None
69:     '''
70:     Filename of the image
71: 
72:     **String**: Filename of the image to use in the toolbar. If None, the
73:     `name` is used as a label in the toolbar button
74:     '''
75: 
76:     def __init__(self, toolmanager, name):
77:         warnings.warn('Treat the new Tool classes introduced in v1.5 as ' +
78:                       'experimental for now, the API will likely change in ' +
79:                       'version 2.1, and some tools might change name')
80:         self._name = name
81:         self._toolmanager = toolmanager
82:         self._figure = None
83: 
84:     @property
85:     def figure(self):
86:         return self._figure
87: 
88:     @figure.setter
89:     def figure(self, figure):
90:         self.set_figure(figure)
91: 
92:     @property
93:     def canvas(self):
94:         if not self._figure:
95:             return None
96:         return self._figure.canvas
97: 
98:     @property
99:     def toolmanager(self):
100:         return self._toolmanager
101: 
102:     def set_figure(self, figure):
103:         '''
104:         Assign a figure to the tool
105: 
106:         Parameters
107:         ----------
108:         figure: `Figure`
109:         '''
110:         self._figure = figure
111: 
112:     def trigger(self, sender, event, data=None):
113:         '''
114:         Called when this tool gets used
115: 
116:         This method is called by
117:         `matplotlib.backend_managers.ToolManager.trigger_tool`
118: 
119:         Parameters
120:         ----------
121:         event: `Event`
122:             The Canvas event that caused this tool to be called
123:         sender: object
124:             Object that requested the tool to be triggered
125:         data: object
126:             Extra data
127:         '''
128: 
129:         pass
130: 
131:     @property
132:     def name(self):
133:         '''Tool Id'''
134:         return self._name
135: 
136:     def destroy(self):
137:         '''
138:         Destroy the tool
139: 
140:         This method is called when the tool is removed by
141:         `matplotlib.backend_managers.ToolManager.remove_tool`
142:         '''
143:         pass
144: 
145: 
146: class ToolToggleBase(ToolBase):
147:     '''
148:     Toggleable tool
149: 
150:     Every time it is triggered, it switches between enable and disable
151: 
152:     Parameters
153:     ----------
154:     ``*args``
155:         Variable length argument to be used by the Tool
156:     ``**kwargs``
157:         `toggled` if present and True, sets the initial state ot the Tool
158:         Arbitrary keyword arguments to be consumed by the Tool
159:     '''
160: 
161:     radio_group = None
162:     '''Attribute to group 'radio' like tools (mutually exclusive)
163: 
164:     **String** that identifies the group or **None** if not belonging to a
165:     group
166:     '''
167: 
168:     cursor = None
169:     '''Cursor to use when the tool is active'''
170: 
171:     default_toggled = False
172:     '''Default of toggled state'''
173: 
174:     def __init__(self, *args, **kwargs):
175:         self._toggled = kwargs.pop('toggled', self.default_toggled)
176:         ToolBase.__init__(self, *args, **kwargs)
177: 
178:     def trigger(self, sender, event, data=None):
179:         '''Calls `enable` or `disable` based on `toggled` value'''
180:         if self._toggled:
181:             self.disable(event)
182:         else:
183:             self.enable(event)
184:         self._toggled = not self._toggled
185: 
186:     def enable(self, event=None):
187:         '''
188:         Enable the toggle tool
189: 
190:         `trigger` calls this method when `toggled` is False
191:         '''
192: 
193:         pass
194: 
195:     def disable(self, event=None):
196:         '''
197:         Disable the toggle tool
198: 
199:         `trigger` call this method when `toggled` is True.
200: 
201:         This can happen in different circumstances
202: 
203:         * Click on the toolbar tool button
204:         * Call to `matplotlib.backend_managers.ToolManager.trigger_tool`
205:         * Another `ToolToggleBase` derived tool is triggered
206:           (from the same `ToolManager`)
207:         '''
208: 
209:         pass
210: 
211:     @property
212:     def toggled(self):
213:         '''State of the toggled tool'''
214: 
215:         return self._toggled
216: 
217:     def set_figure(self, figure):
218:         toggled = self.toggled
219:         if toggled:
220:             if self.figure:
221:                 self.trigger(self, None)
222:             else:
223:                 # if no figure the internal state is not changed
224:                 # we change it here so next call to trigger will change it back
225:                 self._toggled = False
226:         ToolBase.set_figure(self, figure)
227:         if toggled:
228:             if figure:
229:                 self.trigger(self, None)
230:             else:
231:                 # if there is no figure, triggen wont change the internal state
232:                 # we change it back
233:                 self._toggled = True
234: 
235: 
236: class SetCursorBase(ToolBase):
237:     '''
238:     Change to the current cursor while inaxes
239: 
240:     This tool, keeps track of all `ToolToggleBase` derived tools, and calls
241:     set_cursor when a tool gets triggered
242:     '''
243:     def __init__(self, *args, **kwargs):
244:         ToolBase.__init__(self, *args, **kwargs)
245:         self._idDrag = None
246:         self._cursor = None
247:         self._default_cursor = cursors.POINTER
248:         self._last_cursor = self._default_cursor
249:         self.toolmanager.toolmanager_connect('tool_added_event',
250:                                              self._add_tool_cbk)
251: 
252:         # process current tools
253:         for tool in self.toolmanager.tools.values():
254:             self._add_tool(tool)
255: 
256:     def set_figure(self, figure):
257:         if self._idDrag:
258:             self.canvas.mpl_disconnect(self._idDrag)
259:         ToolBase.set_figure(self, figure)
260:         if figure:
261:             self._idDrag = self.canvas.mpl_connect(
262:                 'motion_notify_event', self._set_cursor_cbk)
263: 
264:     def _tool_trigger_cbk(self, event):
265:         if event.tool.toggled:
266:             self._cursor = event.tool.cursor
267:         else:
268:             self._cursor = None
269: 
270:         self._set_cursor_cbk(event.canvasevent)
271: 
272:     def _add_tool(self, tool):
273:         '''set the cursor when the tool is triggered'''
274:         if getattr(tool, 'cursor', None) is not None:
275:             self.toolmanager.toolmanager_connect('tool_trigger_%s' % tool.name,
276:                                                  self._tool_trigger_cbk)
277: 
278:     def _add_tool_cbk(self, event):
279:         '''Process every newly added tool'''
280:         if event.tool is self:
281:             return
282: 
283:         self._add_tool(event.tool)
284: 
285:     def _set_cursor_cbk(self, event):
286:         if not event:
287:             return
288: 
289:         if not getattr(event, 'inaxes', False) or not self._cursor:
290:             if self._last_cursor != self._default_cursor:
291:                 self.set_cursor(self._default_cursor)
292:                 self._last_cursor = self._default_cursor
293:         elif self._cursor:
294:             cursor = self._cursor
295:             if cursor and self._last_cursor != cursor:
296:                 self.set_cursor(cursor)
297:                 self._last_cursor = cursor
298: 
299:     def set_cursor(self, cursor):
300:         '''
301:         Set the cursor
302: 
303:         This method has to be implemented per backend
304:         '''
305:         raise NotImplementedError
306: 
307: 
308: class ToolCursorPosition(ToolBase):
309:     '''
310:     Send message with the current pointer position
311: 
312:     This tool runs in the background reporting the position of the cursor
313:     '''
314:     def __init__(self, *args, **kwargs):
315:         self._idDrag = None
316:         ToolBase.__init__(self, *args, **kwargs)
317: 
318:     def set_figure(self, figure):
319:         if self._idDrag:
320:             self.canvas.mpl_disconnect(self._idDrag)
321:         ToolBase.set_figure(self, figure)
322:         if figure:
323:             self._idDrag = self.canvas.mpl_connect(
324:                 'motion_notify_event', self.send_message)
325: 
326:     def send_message(self, event):
327:         '''Call `matplotlib.backend_managers.ToolManager.message_event`'''
328:         if self.toolmanager.messagelock.locked():
329:             return
330: 
331:         message = ' '
332: 
333:         if event.inaxes and event.inaxes.get_navigate():
334:             try:
335:                 s = event.inaxes.format_coord(event.xdata, event.ydata)
336:             except (ValueError, OverflowError):
337:                 pass
338:             else:
339:                 artists = [a for a in event.inaxes.mouseover_set
340:                            if a.contains(event) and a.get_visible()]
341: 
342:                 if artists:
343:                     a = max(artists, key=lambda x: x.zorder)
344:                     if a is not event.inaxes.patch:
345:                         data = a.get_cursor_data(event)
346:                         if data is not None:
347:                             s += ' [%s]' % a.format_cursor_data(data)
348: 
349:                 message = s
350:         self.toolmanager.message_event(message, self)
351: 
352: 
353: class RubberbandBase(ToolBase):
354:     '''Draw and remove rubberband'''
355:     def trigger(self, sender, event, data):
356:         '''Call `draw_rubberband` or `remove_rubberband` based on data'''
357:         if not self.figure.canvas.widgetlock.available(sender):
358:             return
359:         if data is not None:
360:             self.draw_rubberband(*data)
361:         else:
362:             self.remove_rubberband()
363: 
364:     def draw_rubberband(self, *data):
365:         '''
366:         Draw rubberband
367: 
368:         This method must get implemented per backend
369:         '''
370:         raise NotImplementedError
371: 
372:     def remove_rubberband(self):
373:         '''
374:         Remove rubberband
375: 
376:         This method should get implemented per backend
377:         '''
378:         pass
379: 
380: 
381: class ToolQuit(ToolBase):
382:     '''Tool to call the figure manager destroy method'''
383: 
384:     description = 'Quit the figure'
385:     default_keymap = rcParams['keymap.quit']
386: 
387:     def trigger(self, sender, event, data=None):
388:         Gcf.destroy_fig(self.figure)
389: 
390: 
391: class ToolQuitAll(ToolBase):
392:     '''Tool to call the figure manager destroy method'''
393: 
394:     description = 'Quit all figures'
395:     default_keymap = rcParams['keymap.quit_all']
396: 
397:     def trigger(self, sender, event, data=None):
398:         Gcf.destroy_all()
399: 
400: 
401: class ToolEnableAllNavigation(ToolBase):
402:     '''Tool to enable all axes for toolmanager interaction'''
403: 
404:     description = 'Enables all axes toolmanager'
405:     default_keymap = rcParams['keymap.all_axes']
406: 
407:     def trigger(self, sender, event, data=None):
408:         if event.inaxes is None:
409:             return
410: 
411:         for a in self.figure.get_axes():
412:             if (event.x is not None and event.y is not None
413:                     and a.in_axes(event)):
414:                 a.set_navigate(True)
415: 
416: 
417: class ToolEnableNavigation(ToolBase):
418:     '''Tool to enable a specific axes for toolmanager interaction'''
419: 
420:     description = 'Enables one axes toolmanager'
421:     default_keymap = (1, 2, 3, 4, 5, 6, 7, 8, 9)
422: 
423:     def trigger(self, sender, event, data=None):
424:         if event.inaxes is None:
425:             return
426: 
427:         n = int(event.key) - 1
428:         for i, a in enumerate(self.figure.get_axes()):
429:             if (event.x is not None and event.y is not None
430:                     and a.in_axes(event)):
431:                 a.set_navigate(i == n)
432: 
433: 
434: class _ToolGridBase(ToolBase):
435:     '''Common functionality between ToolGrid and ToolMinorGrid.'''
436: 
437:     _cycle = [(False, False), (True, False), (True, True), (False, True)]
438: 
439:     def trigger(self, sender, event, data=None):
440:         ax = event.inaxes
441:         if ax is None:
442:             return
443:         try:
444:             x_state, x_which, y_state, y_which = self._get_next_grid_states(ax)
445:         except ValueError:
446:             pass
447:         else:
448:             ax.grid(x_state, which=x_which, axis="x")
449:             ax.grid(y_state, which=y_which, axis="y")
450:             ax.figure.canvas.draw_idle()
451: 
452:     @staticmethod
453:     def _get_uniform_grid_state(ticks):
454:         '''
455:         Check whether all grid lines are in the same visibility state.
456: 
457:         Returns True/False if all grid lines are on or off, None if they are
458:         not all in the same state.
459:         '''
460:         if all(tick.gridOn for tick in ticks):
461:             return True
462:         elif not any(tick.gridOn for tick in ticks):
463:             return False
464:         else:
465:             return None
466: 
467: 
468: class ToolGrid(_ToolGridBase):
469:     '''Tool to toggle the major grids of the figure'''
470: 
471:     description = 'Toogle major grids'
472:     default_keymap = rcParams['keymap.grid']
473: 
474:     def _get_next_grid_states(self, ax):
475:         if None in map(self._get_uniform_grid_state,
476:                        [ax.xaxis.minorTicks, ax.yaxis.minorTicks]):
477:             # Bail out if minor grids are not in a uniform state.
478:             raise ValueError
479:         x_state, y_state = map(self._get_uniform_grid_state,
480:                                [ax.xaxis.majorTicks, ax.yaxis.majorTicks])
481:         cycle = self._cycle
482:         # Bail out (via ValueError) if major grids are not in a uniform state.
483:         x_state, y_state = (
484:             cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)])
485:         return (x_state, "major" if x_state else "both",
486:                 y_state, "major" if y_state else "both")
487: 
488: 
489: class ToolMinorGrid(_ToolGridBase):
490:     '''Tool to toggle the major and minor grids of the figure'''
491: 
492:     description = 'Toogle major and minor grids'
493:     default_keymap = rcParams['keymap.grid_minor']
494: 
495:     def _get_next_grid_states(self, ax):
496:         if None in map(self._get_uniform_grid_state,
497:                        [ax.xaxis.majorTicks, ax.yaxis.majorTicks]):
498:             # Bail out if major grids are not in a uniform state.
499:             raise ValueError
500:         x_state, y_state = map(self._get_uniform_grid_state,
501:                                [ax.xaxis.minorTicks, ax.yaxis.minorTicks])
502:         cycle = self._cycle
503:         # Bail out (via ValueError) if minor grids are not in a uniform state.
504:         x_state, y_state = (
505:             cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)])
506:         return x_state, "both", y_state, "both"
507: 
508: 
509: class ToolFullScreen(ToolToggleBase):
510:     '''Tool to toggle full screen'''
511: 
512:     description = 'Toogle Fullscreen mode'
513:     default_keymap = rcParams['keymap.fullscreen']
514: 
515:     def enable(self, event):
516:         self.figure.canvas.manager.full_screen_toggle()
517: 
518:     def disable(self, event):
519:         self.figure.canvas.manager.full_screen_toggle()
520: 
521: 
522: class AxisScaleBase(ToolToggleBase):
523:     '''Base Tool to toggle between linear and logarithmic'''
524: 
525:     def trigger(self, sender, event, data=None):
526:         if event.inaxes is None:
527:             return
528:         ToolToggleBase.trigger(self, sender, event, data)
529: 
530:     def enable(self, event):
531:         self.set_scale(event.inaxes, 'log')
532:         self.figure.canvas.draw_idle()
533: 
534:     def disable(self, event):
535:         self.set_scale(event.inaxes, 'linear')
536:         self.figure.canvas.draw_idle()
537: 
538: 
539: class ToolYScale(AxisScaleBase):
540:     '''Tool to toggle between linear and logarithmic scales on the Y axis'''
541: 
542:     description = 'Toogle Scale Y axis'
543:     default_keymap = rcParams['keymap.yscale']
544: 
545:     def set_scale(self, ax, scale):
546:         ax.set_yscale(scale)
547: 
548: 
549: class ToolXScale(AxisScaleBase):
550:     '''Tool to toggle between linear and logarithmic scales on the X axis'''
551: 
552:     description = 'Toogle Scale X axis'
553:     default_keymap = rcParams['keymap.xscale']
554: 
555:     def set_scale(self, ax, scale):
556:         ax.set_xscale(scale)
557: 
558: 
559: class ToolViewsPositions(ToolBase):
560:     '''
561:     Auxiliary Tool to handle changes in views and positions
562: 
563:     Runs in the background and should get used by all the tools that
564:     need to access the figure's history of views and positions, e.g.
565: 
566:     * `ToolZoom`
567:     * `ToolPan`
568:     * `ToolHome`
569:     * `ToolBack`
570:     * `ToolForward`
571:     '''
572: 
573:     def __init__(self, *args, **kwargs):
574:         self.views = WeakKeyDictionary()
575:         self.positions = WeakKeyDictionary()
576:         self.home_views = WeakKeyDictionary()
577:         ToolBase.__init__(self, *args, **kwargs)
578: 
579:     def add_figure(self, figure):
580:         '''Add the current figure to the stack of views and positions'''
581: 
582:         if figure not in self.views:
583:             self.views[figure] = cbook.Stack()
584:             self.positions[figure] = cbook.Stack()
585:             self.home_views[figure] = WeakKeyDictionary()
586:             # Define Home
587:             self.push_current(figure)
588:             # Make sure we add a home view for new axes as they're added
589:             figure.add_axobserver(lambda fig: self.update_home_views(fig))
590: 
591:     def clear(self, figure):
592:         '''Reset the axes stack'''
593:         if figure in self.views:
594:             self.views[figure].clear()
595:             self.positions[figure].clear()
596:             self.home_views[figure].clear()
597:             self.update_home_views()
598: 
599:     def update_view(self):
600:         '''
601:         Update the view limits and position for each axes from the current
602:         stack position. If any axes are present in the figure that aren't in
603:         the current stack position, use the home view limits for those axes and
604:         don't update *any* positions.
605:         '''
606: 
607:         views = self.views[self.figure]()
608:         if views is None:
609:             return
610:         pos = self.positions[self.figure]()
611:         if pos is None:
612:             return
613:         home_views = self.home_views[self.figure]
614:         all_axes = self.figure.get_axes()
615:         for a in all_axes:
616:             if a in views:
617:                 cur_view = views[a]
618:             else:
619:                 cur_view = home_views[a]
620:             a._set_view(cur_view)
621: 
622:         if set(all_axes).issubset(pos):
623:             for a in all_axes:
624:                 # Restore both the original and modified positions
625:                 a.set_position(pos[a][0], 'original')
626:                 a.set_position(pos[a][1], 'active')
627: 
628:         self.figure.canvas.draw_idle()
629: 
630:     def push_current(self, figure=None):
631:         '''
632:         Push the current view limits and position onto their respective stacks
633:         '''
634:         if not figure:
635:             figure = self.figure
636:         views = WeakKeyDictionary()
637:         pos = WeakKeyDictionary()
638:         for a in figure.get_axes():
639:             views[a] = a._get_view()
640:             pos[a] = self._axes_pos(a)
641:         self.views[figure].push(views)
642:         self.positions[figure].push(pos)
643: 
644:     def _axes_pos(self, ax):
645:         '''
646:         Return the original and modified positions for the specified axes
647: 
648:         Parameters
649:         ----------
650:         ax : (matplotlib.axes.AxesSubplot)
651:         The axes to get the positions for
652: 
653:         Returns
654:         -------
655:         limits : (tuple)
656:         A tuple of the original and modified positions
657:         '''
658: 
659:         return (ax.get_position(True).frozen(),
660:                 ax.get_position().frozen())
661: 
662:     def update_home_views(self, figure=None):
663:         '''
664:         Make sure that self.home_views has an entry for all axes present in the
665:         figure
666:         '''
667: 
668:         if not figure:
669:             figure = self.figure
670:         for a in figure.get_axes():
671:             if a not in self.home_views[figure]:
672:                 self.home_views[figure][a] = a._get_view()
673: 
674:     def refresh_locators(self):
675:         '''Redraw the canvases, update the locators'''
676:         for a in self.figure.get_axes():
677:             xaxis = getattr(a, 'xaxis', None)
678:             yaxis = getattr(a, 'yaxis', None)
679:             zaxis = getattr(a, 'zaxis', None)
680:             locators = []
681:             if xaxis is not None:
682:                 locators.append(xaxis.get_major_locator())
683:                 locators.append(xaxis.get_minor_locator())
684:             if yaxis is not None:
685:                 locators.append(yaxis.get_major_locator())
686:                 locators.append(yaxis.get_minor_locator())
687:             if zaxis is not None:
688:                 locators.append(zaxis.get_major_locator())
689:                 locators.append(zaxis.get_minor_locator())
690: 
691:             for loc in locators:
692:                 loc.refresh()
693:         self.figure.canvas.draw_idle()
694: 
695:     def home(self):
696:         '''Recall the first view and position from the stack'''
697:         self.views[self.figure].home()
698:         self.positions[self.figure].home()
699: 
700:     def back(self):
701:         '''Back one step in the stack of views and positions'''
702:         self.views[self.figure].back()
703:         self.positions[self.figure].back()
704: 
705:     def forward(self):
706:         '''Forward one step in the stack of views and positions'''
707:         self.views[self.figure].forward()
708:         self.positions[self.figure].forward()
709: 
710: 
711: class ViewsPositionsBase(ToolBase):
712:     '''Base class for `ToolHome`, `ToolBack` and `ToolForward`'''
713: 
714:     _on_trigger = None
715: 
716:     def trigger(self, sender, event, data=None):
717:         self.toolmanager.get_tool(_views_positions).add_figure(self.figure)
718:         getattr(self.toolmanager.get_tool(_views_positions),
719:                 self._on_trigger)()
720:         self.toolmanager.get_tool(_views_positions).update_view()
721: 
722: 
723: class ToolHome(ViewsPositionsBase):
724:     '''Restore the original view lim'''
725: 
726:     description = 'Reset original view'
727:     image = 'home.png'
728:     default_keymap = rcParams['keymap.home']
729:     _on_trigger = 'home'
730: 
731: 
732: class ToolBack(ViewsPositionsBase):
733:     '''Move back up the view lim stack'''
734: 
735:     description = 'Back to previous view'
736:     image = 'back.png'
737:     default_keymap = rcParams['keymap.back']
738:     _on_trigger = 'back'
739: 
740: 
741: class ToolForward(ViewsPositionsBase):
742:     '''Move forward in the view lim stack'''
743: 
744:     description = 'Forward to next view'
745:     image = 'forward.png'
746:     default_keymap = rcParams['keymap.forward']
747:     _on_trigger = 'forward'
748: 
749: 
750: class ConfigureSubplotsBase(ToolBase):
751:     '''Base tool for the configuration of subplots'''
752: 
753:     description = 'Configure subplots'
754:     image = 'subplots.png'
755: 
756: 
757: class SaveFigureBase(ToolBase):
758:     '''Base tool for figure saving'''
759: 
760:     description = 'Save the figure'
761:     image = 'filesave.png'
762:     default_keymap = rcParams['keymap.save']
763: 
764: 
765: class ZoomPanBase(ToolToggleBase):
766:     '''Base class for `ToolZoom` and `ToolPan`'''
767:     def __init__(self, *args):
768:         ToolToggleBase.__init__(self, *args)
769:         self._button_pressed = None
770:         self._xypress = None
771:         self._idPress = None
772:         self._idRelease = None
773:         self._idScroll = None
774:         self.base_scale = 2.
775:         self.scrollthresh = .5  # .5 second scroll threshold
776:         self.lastscroll = time.time()-self.scrollthresh
777: 
778:     def enable(self, event):
779:         '''Connect press/release events and lock the canvas'''
780:         self.figure.canvas.widgetlock(self)
781:         self._idPress = self.figure.canvas.mpl_connect(
782:             'button_press_event', self._press)
783:         self._idRelease = self.figure.canvas.mpl_connect(
784:             'button_release_event', self._release)
785:         self._idScroll = self.figure.canvas.mpl_connect(
786:             'scroll_event', self.scroll_zoom)
787: 
788:     def disable(self, event):
789:         '''Release the canvas and disconnect press/release events'''
790:         self._cancel_action()
791:         self.figure.canvas.widgetlock.release(self)
792:         self.figure.canvas.mpl_disconnect(self._idPress)
793:         self.figure.canvas.mpl_disconnect(self._idRelease)
794:         self.figure.canvas.mpl_disconnect(self._idScroll)
795: 
796:     def trigger(self, sender, event, data=None):
797:         self.toolmanager.get_tool(_views_positions).add_figure(self.figure)
798:         ToolToggleBase.trigger(self, sender, event, data)
799: 
800:     def scroll_zoom(self, event):
801:         # https://gist.github.com/tacaswell/3144287
802:         if event.inaxes is None:
803:             return
804: 
805:         if event.button == 'up':
806:             # deal with zoom in
807:             scl = self.base_scale
808:         elif event.button == 'down':
809:             # deal with zoom out
810:             scl = 1/self.base_scale
811:         else:
812:             # deal with something that should never happen
813:             scl = 1
814: 
815:         ax = event.inaxes
816:         ax._set_view_from_bbox([event.x, event.y, scl])
817: 
818:         # If last scroll was done within the timing threshold, delete the
819:         # previous view
820:         if (time.time()-self.lastscroll) < self.scrollthresh:
821:             self.toolmanager.get_tool(_views_positions).back()
822: 
823:         self.figure.canvas.draw_idle()  # force re-draw
824: 
825:         self.lastscroll = time.time()
826:         self.toolmanager.get_tool(_views_positions).push_current()
827: 
828: 
829: class ToolZoom(ZoomPanBase):
830:     '''Zoom to rectangle'''
831: 
832:     description = 'Zoom to rectangle'
833:     image = 'zoom_to_rect.png'
834:     default_keymap = rcParams['keymap.zoom']
835:     cursor = cursors.SELECT_REGION
836:     radio_group = 'default'
837: 
838:     def __init__(self, *args):
839:         ZoomPanBase.__init__(self, *args)
840:         self._ids_zoom = []
841: 
842:     def _cancel_action(self):
843:         for zoom_id in self._ids_zoom:
844:             self.figure.canvas.mpl_disconnect(zoom_id)
845:         self.toolmanager.trigger_tool('rubberband', self)
846:         self.toolmanager.get_tool(_views_positions).refresh_locators()
847:         self._xypress = None
848:         self._button_pressed = None
849:         self._ids_zoom = []
850:         return
851: 
852:     def _press(self, event):
853:         '''the _press mouse button in zoom to rect mode callback'''
854: 
855:         # If we're already in the middle of a zoom, pressing another
856:         # button works to "cancel"
857:         if self._ids_zoom != []:
858:             self._cancel_action()
859: 
860:         if event.button == 1:
861:             self._button_pressed = 1
862:         elif event.button == 3:
863:             self._button_pressed = 3
864:         else:
865:             self._cancel_action()
866:             return
867: 
868:         x, y = event.x, event.y
869: 
870:         self._xypress = []
871:         for i, a in enumerate(self.figure.get_axes()):
872:             if (x is not None and y is not None and a.in_axes(event) and
873:                     a.get_navigate() and a.can_zoom()):
874:                 self._xypress.append((x, y, a, i, a._get_view()))
875: 
876:         id1 = self.figure.canvas.mpl_connect(
877:             'motion_notify_event', self._mouse_move)
878:         id2 = self.figure.canvas.mpl_connect(
879:             'key_press_event', self._switch_on_zoom_mode)
880:         id3 = self.figure.canvas.mpl_connect(
881:             'key_release_event', self._switch_off_zoom_mode)
882: 
883:         self._ids_zoom = id1, id2, id3
884:         self._zoom_mode = event.key
885: 
886:     def _switch_on_zoom_mode(self, event):
887:         self._zoom_mode = event.key
888:         self._mouse_move(event)
889: 
890:     def _switch_off_zoom_mode(self, event):
891:         self._zoom_mode = None
892:         self._mouse_move(event)
893: 
894:     def _mouse_move(self, event):
895:         '''the drag callback in zoom mode'''
896: 
897:         if self._xypress:
898:             x, y = event.x, event.y
899:             lastx, lasty, a, ind, view = self._xypress[0]
900:             (x1, y1), (x2, y2) = np.clip(
901:                 [[lastx, lasty], [x, y]], a.bbox.min, a.bbox.max)
902:             if self._zoom_mode == "x":
903:                 y1, y2 = a.bbox.intervaly
904:             elif self._zoom_mode == "y":
905:                 x1, x2 = a.bbox.intervalx
906:             self.toolmanager.trigger_tool(
907:                 'rubberband', self, data=(x1, y1, x2, y2))
908: 
909:     def _release(self, event):
910:         '''the release mouse button callback in zoom to rect mode'''
911: 
912:         for zoom_id in self._ids_zoom:
913:             self.figure.canvas.mpl_disconnect(zoom_id)
914:         self._ids_zoom = []
915: 
916:         if not self._xypress:
917:             self._cancel_action()
918:             return
919: 
920:         last_a = []
921: 
922:         for cur_xypress in self._xypress:
923:             x, y = event.x, event.y
924:             lastx, lasty, a, _ind, view = cur_xypress
925:             # ignore singular clicks - 5 pixels is a threshold
926:             if abs(x - lastx) < 5 or abs(y - lasty) < 5:
927:                 self._cancel_action()
928:                 return
929: 
930:             # detect twinx,y axes and avoid double zooming
931:             twinx, twiny = False, False
932:             if last_a:
933:                 for la in last_a:
934:                     if a.get_shared_x_axes().joined(a, la):
935:                         twinx = True
936:                     if a.get_shared_y_axes().joined(a, la):
937:                         twiny = True
938:             last_a.append(a)
939: 
940:             if self._button_pressed == 1:
941:                 direction = 'in'
942:             elif self._button_pressed == 3:
943:                 direction = 'out'
944:             else:
945:                 continue
946: 
947:             a._set_view_from_bbox((lastx, lasty, x, y), direction,
948:                                   self._zoom_mode, twinx, twiny)
949: 
950:         self._zoom_mode = None
951:         self.toolmanager.get_tool(_views_positions).push_current()
952:         self._cancel_action()
953: 
954: 
955: class ToolPan(ZoomPanBase):
956:     '''Pan axes with left mouse, zoom with right'''
957: 
958:     default_keymap = rcParams['keymap.pan']
959:     description = 'Pan axes with left mouse, zoom with right'
960:     image = 'move.png'
961:     cursor = cursors.MOVE
962:     radio_group = 'default'
963: 
964:     def __init__(self, *args):
965:         ZoomPanBase.__init__(self, *args)
966:         self._idDrag = None
967: 
968:     def _cancel_action(self):
969:         self._button_pressed = None
970:         self._xypress = []
971:         self.figure.canvas.mpl_disconnect(self._idDrag)
972:         self.toolmanager.messagelock.release(self)
973:         self.toolmanager.get_tool(_views_positions).refresh_locators()
974: 
975:     def _press(self, event):
976:         if event.button == 1:
977:             self._button_pressed = 1
978:         elif event.button == 3:
979:             self._button_pressed = 3
980:         else:
981:             self._cancel_action()
982:             return
983: 
984:         x, y = event.x, event.y
985: 
986:         self._xypress = []
987:         for i, a in enumerate(self.figure.get_axes()):
988:             if (x is not None and y is not None and a.in_axes(event) and
989:                     a.get_navigate() and a.can_pan()):
990:                 a.start_pan(x, y, event.button)
991:                 self._xypress.append((a, i))
992:                 self.toolmanager.messagelock(self)
993:                 self._idDrag = self.figure.canvas.mpl_connect(
994:                     'motion_notify_event', self._mouse_move)
995: 
996:     def _release(self, event):
997:         if self._button_pressed is None:
998:             self._cancel_action()
999:             return
1000: 
1001:         self.figure.canvas.mpl_disconnect(self._idDrag)
1002:         self.toolmanager.messagelock.release(self)
1003: 
1004:         for a, _ind in self._xypress:
1005:             a.end_pan()
1006:         if not self._xypress:
1007:             self._cancel_action()
1008:             return
1009: 
1010:         self.toolmanager.get_tool(_views_positions).push_current()
1011:         self._cancel_action()
1012: 
1013:     def _mouse_move(self, event):
1014:         for a, _ind in self._xypress:
1015:             # safer to use the recorded button at the _press than current
1016:             # button: # multiple button can get pressed during motion...
1017:             a.drag_pan(self._button_pressed, event.key, event.x, event.y)
1018:         self.toolmanager.canvas.draw_idle()
1019: 
1020: 
1021: default_tools = {'home': ToolHome, 'back': ToolBack, 'forward': ToolForward,
1022:                  'zoom': ToolZoom, 'pan': ToolPan,
1023:                  'subplots': 'ToolConfigureSubplots',
1024:                  'save': 'ToolSaveFigure',
1025:                  'grid': ToolGrid,
1026:                  'grid_minor': ToolMinorGrid,
1027:                  'fullscreen': ToolFullScreen,
1028:                  'quit': ToolQuit,
1029:                  'quit_all': ToolQuitAll,
1030:                  'allnav': ToolEnableAllNavigation,
1031:                  'nav': ToolEnableNavigation,
1032:                  'xscale': ToolXScale,
1033:                  'yscale': ToolYScale,
1034:                  'position': ToolCursorPosition,
1035:                  _views_positions: ToolViewsPositions,
1036:                  'cursor': 'ToolSetCursor',
1037:                  'rubberband': 'ToolRubberband',
1038:                  }
1039: '''Default tools'''
1040: 
1041: default_toolbar_tools = [['navigation', ['home', 'back', 'forward']],
1042:                          ['zoompan', ['pan', 'zoom', 'subplots']],
1043:                          ['io', ['save']]]
1044: '''Default tools in the toolbar'''
1045: 
1046: 
1047: def add_tools_to_manager(toolmanager, tools=default_tools):
1048:     '''
1049:     Add multiple tools to `ToolManager`
1050: 
1051:     Parameters
1052:     ----------
1053:     toolmanager: ToolManager
1054:         `backend_managers.ToolManager` object that will get the tools added
1055:     tools : {str: class_like}, optional
1056:         The tools to add in a {name: tool} dict, see `add_tool` for more
1057:         info.
1058:     '''
1059: 
1060:     for name, tool in six.iteritems(tools):
1061:         toolmanager.add_tool(name, tool)
1062: 
1063: 
1064: def add_tools_to_container(container, tools=default_toolbar_tools):
1065:     '''
1066:     Add multiple tools to the container.
1067: 
1068:     Parameters
1069:     ----------
1070:     container: Container
1071:         `backend_bases.ToolContainerBase` object that will get the tools added
1072:     tools : list, optional
1073:         List in the form
1074:         [[group1, [tool1, tool2 ...]], [group2, [...]]]
1075:         Where the tools given by tool1, and tool2 will display in group1.
1076:         See `add_tool` for details.
1077:     '''
1078: 
1079:     for group, grouptools in tools:
1080:         for position, tool in enumerate(grouptools):
1081:             container.add_tool(tool, group, position)
1082: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, (-1)), 'str', '\nAbstract base classes define the primitives for Tools.\nThese tools are used by `matplotlib.backend_managers.ToolManager`\n\n:class:`ToolBase`\n    Simple stateless tool\n\n:class:`ToolToggleBase`\n    Tool that has two states, only one Toggle tool can be\n    active at any given time for the same\n    `matplotlib.backend_managers.ToolManager`\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib import rcParams' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_20199 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib')

if (type(import_20199) is not StypyTypeError):

    if (import_20199 != 'pyd_module'):
        __import__(import_20199)
        sys_modules_20200 = sys.modules[import_20199]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', sys_modules_20200.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_20200, sys_modules_20200.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib', import_20199)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from matplotlib._pylab_helpers import Gcf' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_20201 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib._pylab_helpers')

if (type(import_20201) is not StypyTypeError):

    if (import_20201 != 'pyd_module'):
        __import__(import_20201)
        sys_modules_20202 = sys.modules[import_20201]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib._pylab_helpers', sys_modules_20202.module_type_store, module_type_store, ['Gcf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_20202, sys_modules_20202.module_type_store, module_type_store)
    else:
        from matplotlib._pylab_helpers import Gcf

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib._pylab_helpers', None, module_type_store, ['Gcf'], [Gcf])

else:
    # Assigning a type to the variable 'matplotlib._pylab_helpers' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'matplotlib._pylab_helpers', import_20201)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import matplotlib.cbook' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_20203 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.cbook')

if (type(import_20203) is not StypyTypeError):

    if (import_20203 != 'pyd_module'):
        __import__(import_20203)
        sys_modules_20204 = sys.modules[import_20203]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'cbook', sys_modules_20204.module_type_store, module_type_store)
    else:
        import matplotlib.cbook as cbook

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'cbook', matplotlib.cbook, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'matplotlib.cbook', import_20203)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from weakref import WeakKeyDictionary' statement (line 18)
try:
    from weakref import WeakKeyDictionary

except:
    WeakKeyDictionary = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'weakref', None, module_type_store, ['WeakKeyDictionary'], [WeakKeyDictionary])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import six' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_20205 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'six')

if (type(import_20205) is not StypyTypeError):

    if (import_20205 != 'pyd_module'):
        __import__(import_20205)
        sys_modules_20206 = sys.modules[import_20205]
        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'six', sys_modules_20206.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'six', import_20205)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import time' statement (line 20)
import time

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import warnings' statement (line 21)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import numpy' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_20207 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy')

if (type(import_20207) is not StypyTypeError):

    if (import_20207 != 'pyd_module'):
        __import__(import_20207)
        sys_modules_20208 = sys.modules[import_20207]
        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'np', sys_modules_20208.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy', import_20207)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'Cursors' class

class Cursors(object, ):
    str_20209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'str', 'Simple namespace for cursor reference')
    
    # Assigning a Call to a Tuple (line 27):
    
    # Assigning a Call to a Name:
    
    # Assigning a Call to a Name (line 27):
    
    # Assigning a Name to a Name (line 27):
    
    # Assigning a Call to a Name (line 27):
    
    # Assigning a Name to a Name (line 27):
    
    # Assigning a Call to a Name (line 27):
    
    # Assigning a Name to a Name (line 27):
    
    # Assigning a Call to a Name (line 27):
    
    # Assigning a Name to a Name (line 27):
    
    # Assigning a Call to a Name (line 27):
    
    # Assigning a Name to a Name (line 27):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 0, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Cursors.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Cursors' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'Cursors', Cursors)

# Assigning a Call to a Name:

# Call to list(...): (line 27)
# Processing the call arguments (line 27)

# Call to range(...): (line 27)
# Processing the call arguments (line 27)
int_20212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 58), 'int')
# Processing the call keyword arguments (line 27)
kwargs_20213 = {}
# Getting the type of 'range' (line 27)
range_20211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 52), 'range', False)
# Calling range(args, kwargs) (line 27)
range_call_result_20214 = invoke(stypy.reporting.localization.Localization(__file__, 27, 52), range_20211, *[int_20212], **kwargs_20213)

# Processing the call keyword arguments (line 27)
kwargs_20215 = {}
# Getting the type of 'list' (line 27)
list_20210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 47), 'list', False)
# Calling list(args, kwargs) (line 27)
list_call_result_20216 = invoke(stypy.reporting.localization.Localization(__file__, 27, 47), list_20210, *[range_call_result_20214], **kwargs_20215)

# Getting the type of 'Cursors'
Cursors_20217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Setting the type of the member 'call_assignment_20146' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20217, 'call_assignment_20146', list_call_result_20216)

# Assigning a Call to a Name (line 27):

# Call to __getitem__(...):
# Processing the call arguments
int_20221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'int')
# Processing the call keyword arguments
kwargs_20222 = {}
# Getting the type of 'Cursors'
Cursors_20218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors', False)
# Obtaining the member 'call_assignment_20146' of a type
call_assignment_20146_20219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20218, 'call_assignment_20146')
# Obtaining the member '__getitem__' of a type
getitem___20220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), call_assignment_20146_20219, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_20223 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20220, *[int_20221], **kwargs_20222)

# Getting the type of 'Cursors'
Cursors_20224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Setting the type of the member 'call_assignment_20147' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20224, 'call_assignment_20147', getitem___call_result_20223)

# Assigning a Name to a Name (line 27):
# Getting the type of 'Cursors'
Cursors_20225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Obtaining the member 'call_assignment_20147' of a type
call_assignment_20147_20226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20225, 'call_assignment_20147')
# Getting the type of 'Cursors'
Cursors_20227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Setting the type of the member 'HAND' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20227, 'HAND', call_assignment_20147_20226)

# Assigning a Call to a Name (line 27):

# Call to __getitem__(...):
# Processing the call arguments
int_20231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'int')
# Processing the call keyword arguments
kwargs_20232 = {}
# Getting the type of 'Cursors'
Cursors_20228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors', False)
# Obtaining the member 'call_assignment_20146' of a type
call_assignment_20146_20229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20228, 'call_assignment_20146')
# Obtaining the member '__getitem__' of a type
getitem___20230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), call_assignment_20146_20229, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_20233 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20230, *[int_20231], **kwargs_20232)

# Getting the type of 'Cursors'
Cursors_20234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Setting the type of the member 'call_assignment_20148' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20234, 'call_assignment_20148', getitem___call_result_20233)

# Assigning a Name to a Name (line 27):
# Getting the type of 'Cursors'
Cursors_20235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Obtaining the member 'call_assignment_20148' of a type
call_assignment_20148_20236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20235, 'call_assignment_20148')
# Getting the type of 'Cursors'
Cursors_20237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Setting the type of the member 'POINTER' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20237, 'POINTER', call_assignment_20148_20236)

# Assigning a Call to a Name (line 27):

# Call to __getitem__(...):
# Processing the call arguments
int_20241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'int')
# Processing the call keyword arguments
kwargs_20242 = {}
# Getting the type of 'Cursors'
Cursors_20238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors', False)
# Obtaining the member 'call_assignment_20146' of a type
call_assignment_20146_20239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20238, 'call_assignment_20146')
# Obtaining the member '__getitem__' of a type
getitem___20240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), call_assignment_20146_20239, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_20243 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20240, *[int_20241], **kwargs_20242)

# Getting the type of 'Cursors'
Cursors_20244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Setting the type of the member 'call_assignment_20149' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20244, 'call_assignment_20149', getitem___call_result_20243)

# Assigning a Name to a Name (line 27):
# Getting the type of 'Cursors'
Cursors_20245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Obtaining the member 'call_assignment_20149' of a type
call_assignment_20149_20246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20245, 'call_assignment_20149')
# Getting the type of 'Cursors'
Cursors_20247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Setting the type of the member 'SELECT_REGION' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20247, 'SELECT_REGION', call_assignment_20149_20246)

# Assigning a Call to a Name (line 27):

# Call to __getitem__(...):
# Processing the call arguments
int_20251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'int')
# Processing the call keyword arguments
kwargs_20252 = {}
# Getting the type of 'Cursors'
Cursors_20248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors', False)
# Obtaining the member 'call_assignment_20146' of a type
call_assignment_20146_20249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20248, 'call_assignment_20146')
# Obtaining the member '__getitem__' of a type
getitem___20250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), call_assignment_20146_20249, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_20253 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20250, *[int_20251], **kwargs_20252)

# Getting the type of 'Cursors'
Cursors_20254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Setting the type of the member 'call_assignment_20150' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20254, 'call_assignment_20150', getitem___call_result_20253)

# Assigning a Name to a Name (line 27):
# Getting the type of 'Cursors'
Cursors_20255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Obtaining the member 'call_assignment_20150' of a type
call_assignment_20150_20256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20255, 'call_assignment_20150')
# Getting the type of 'Cursors'
Cursors_20257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Setting the type of the member 'MOVE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20257, 'MOVE', call_assignment_20150_20256)

# Assigning a Call to a Name (line 27):

# Call to __getitem__(...):
# Processing the call arguments
int_20261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'int')
# Processing the call keyword arguments
kwargs_20262 = {}
# Getting the type of 'Cursors'
Cursors_20258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors', False)
# Obtaining the member 'call_assignment_20146' of a type
call_assignment_20146_20259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20258, 'call_assignment_20146')
# Obtaining the member '__getitem__' of a type
getitem___20260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), call_assignment_20146_20259, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_20263 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20260, *[int_20261], **kwargs_20262)

# Getting the type of 'Cursors'
Cursors_20264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Setting the type of the member 'call_assignment_20151' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20264, 'call_assignment_20151', getitem___call_result_20263)

# Assigning a Name to a Name (line 27):
# Getting the type of 'Cursors'
Cursors_20265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Obtaining the member 'call_assignment_20151' of a type
call_assignment_20151_20266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20265, 'call_assignment_20151')
# Getting the type of 'Cursors'
Cursors_20267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Cursors')
# Setting the type of the member 'WAIT' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Cursors_20267, 'WAIT', call_assignment_20151_20266)

# Assigning a Call to a Name (line 28):

# Assigning a Call to a Name (line 28):

# Assigning a Call to a Name (line 28):

# Call to Cursors(...): (line 28)
# Processing the call keyword arguments (line 28)
kwargs_20269 = {}
# Getting the type of 'Cursors' (line 28)
Cursors_20268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'Cursors', False)
# Calling Cursors(args, kwargs) (line 28)
Cursors_call_result_20270 = invoke(stypy.reporting.localization.Localization(__file__, 28, 10), Cursors_20268, *[], **kwargs_20269)

# Assigning a type to the variable 'cursors' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'cursors', Cursors_call_result_20270)

# Assigning a Str to a Name (line 31):

# Assigning a Str to a Name (line 31):

# Assigning a Str to a Name (line 31):
str_20271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'str', 'viewpos')
# Assigning a type to the variable '_views_positions' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), '_views_positions', str_20271)
# Declaration of the 'ToolBase' class

class ToolBase(object, ):
    str_20272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', '\n    Base tool class\n\n    A base tool, only implements `trigger` method or not method at all.\n    The tool is instantiated by `matplotlib.backend_managers.ToolManager`\n\n    Attributes\n    ----------\n    toolmanager: `matplotlib.backend_managers.ToolManager`\n        ToolManager that controls this Tool\n    figure: `FigureCanvas`\n        Figure instance that is affected by this Tool\n    name: String\n        Used as **Id** of the tool, has to be unique among tools of the same\n        ToolManager\n    ')
    
    # Assigning a Name to a Name (line 52):
    
    # Assigning a Name to a Name (line 52):
    str_20273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'str', '\n    Keymap to associate with this tool\n\n    **String**: List of comma separated keys that will be used to call this\n    tool when the keypress event of *self.figure.canvas* is emited\n    ')
    
    # Assigning a Name to a Name (line 60):
    
    # Assigning a Name to a Name (line 60):
    str_20274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'str', '\n    Description of the Tool\n\n    **String**: If the Tool is included in the Toolbar this text is used\n    as a Tooltip\n    ')
    
    # Assigning a Name to a Name (line 68):
    
    # Assigning a Name to a Name (line 68):
    str_20275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, (-1)), 'str', '\n    Filename of the image\n\n    **String**: Filename of the image to use in the toolbar. If None, the\n    `name` is used as a label in the toolbar button\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolBase.__init__', ['toolmanager', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['toolmanager', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to warn(...): (line 77)
        # Processing the call arguments (line 77)
        str_20278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 22), 'str', 'Treat the new Tool classes introduced in v1.5 as ')
        str_20279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 22), 'str', 'experimental for now, the API will likely change in ')
        # Applying the binary operator '+' (line 77)
        result_add_20280 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 22), '+', str_20278, str_20279)
        
        str_20281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'str', 'version 2.1, and some tools might change name')
        # Applying the binary operator '+' (line 78)
        result_add_20282 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 77), '+', result_add_20280, str_20281)
        
        # Processing the call keyword arguments (line 77)
        kwargs_20283 = {}
        # Getting the type of 'warnings' (line 77)
        warnings_20276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 77)
        warn_20277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), warnings_20276, 'warn')
        # Calling warn(args, kwargs) (line 77)
        warn_call_result_20284 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), warn_20277, *[result_add_20282], **kwargs_20283)
        
        
        # Assigning a Name to a Attribute (line 80):
        
        # Assigning a Name to a Attribute (line 80):
        
        # Assigning a Name to a Attribute (line 80):
        # Getting the type of 'name' (line 80)
        name_20285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 21), 'name')
        # Getting the type of 'self' (line 80)
        self_20286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self')
        # Setting the type of the member '_name' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_20286, '_name', name_20285)
        
        # Assigning a Name to a Attribute (line 81):
        
        # Assigning a Name to a Attribute (line 81):
        
        # Assigning a Name to a Attribute (line 81):
        # Getting the type of 'toolmanager' (line 81)
        toolmanager_20287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'toolmanager')
        # Getting the type of 'self' (line 81)
        self_20288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self')
        # Setting the type of the member '_toolmanager' of a type (line 81)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_20288, '_toolmanager', toolmanager_20287)
        
        # Assigning a Name to a Attribute (line 82):
        
        # Assigning a Name to a Attribute (line 82):
        
        # Assigning a Name to a Attribute (line 82):
        # Getting the type of 'None' (line 82)
        None_20289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'None')
        # Getting the type of 'self' (line 82)
        self_20290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member '_figure' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_20290, '_figure', None_20289)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'figure'
        module_type_store = module_type_store.open_function_context('figure', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolBase.figure.__dict__.__setitem__('stypy_localization', localization)
        ToolBase.figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolBase.figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolBase.figure.__dict__.__setitem__('stypy_function_name', 'ToolBase.figure')
        ToolBase.figure.__dict__.__setitem__('stypy_param_names_list', [])
        ToolBase.figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolBase.figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolBase.figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolBase.figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolBase.figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolBase.figure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolBase.figure', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'figure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'figure(...)' code ##################

        # Getting the type of 'self' (line 86)
        self_20291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'self')
        # Obtaining the member '_figure' of a type (line 86)
        _figure_20292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), self_20291, '_figure')
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', _figure_20292)
        
        # ################# End of 'figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'figure' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_20293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20293)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'figure'
        return stypy_return_type_20293


    @norecursion
    def figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'figure'
        module_type_store = module_type_store.open_function_context('figure', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolBase.figure.__dict__.__setitem__('stypy_localization', localization)
        ToolBase.figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolBase.figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolBase.figure.__dict__.__setitem__('stypy_function_name', 'ToolBase.figure')
        ToolBase.figure.__dict__.__setitem__('stypy_param_names_list', ['figure'])
        ToolBase.figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolBase.figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolBase.figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolBase.figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolBase.figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolBase.figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolBase.figure', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'figure', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'figure(...)' code ##################

        
        # Call to set_figure(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'figure' (line 90)
        figure_20296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'figure', False)
        # Processing the call keyword arguments (line 90)
        kwargs_20297 = {}
        # Getting the type of 'self' (line 90)
        self_20294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self', False)
        # Obtaining the member 'set_figure' of a type (line 90)
        set_figure_20295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_20294, 'set_figure')
        # Calling set_figure(args, kwargs) (line 90)
        set_figure_call_result_20298 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), set_figure_20295, *[figure_20296], **kwargs_20297)
        
        
        # ################# End of 'figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'figure' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_20299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20299)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'figure'
        return stypy_return_type_20299


    @norecursion
    def canvas(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'canvas'
        module_type_store = module_type_store.open_function_context('canvas', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolBase.canvas.__dict__.__setitem__('stypy_localization', localization)
        ToolBase.canvas.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolBase.canvas.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolBase.canvas.__dict__.__setitem__('stypy_function_name', 'ToolBase.canvas')
        ToolBase.canvas.__dict__.__setitem__('stypy_param_names_list', [])
        ToolBase.canvas.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolBase.canvas.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolBase.canvas.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolBase.canvas.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolBase.canvas.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolBase.canvas.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolBase.canvas', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'canvas', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'canvas(...)' code ##################

        
        
        # Getting the type of 'self' (line 94)
        self_20300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'self')
        # Obtaining the member '_figure' of a type (line 94)
        _figure_20301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 15), self_20300, '_figure')
        # Applying the 'not' unary operator (line 94)
        result_not__20302 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 11), 'not', _figure_20301)
        
        # Testing the type of an if condition (line 94)
        if_condition_20303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 8), result_not__20302)
        # Assigning a type to the variable 'if_condition_20303' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'if_condition_20303', if_condition_20303)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 95)
        None_20304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'stypy_return_type', None_20304)
        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 96)
        self_20305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'self')
        # Obtaining the member '_figure' of a type (line 96)
        _figure_20306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 15), self_20305, '_figure')
        # Obtaining the member 'canvas' of a type (line 96)
        canvas_20307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 15), _figure_20306, 'canvas')
        # Assigning a type to the variable 'stypy_return_type' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'stypy_return_type', canvas_20307)
        
        # ################# End of 'canvas(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'canvas' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_20308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20308)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'canvas'
        return stypy_return_type_20308


    @norecursion
    def toolmanager(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'toolmanager'
        module_type_store = module_type_store.open_function_context('toolmanager', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolBase.toolmanager.__dict__.__setitem__('stypy_localization', localization)
        ToolBase.toolmanager.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolBase.toolmanager.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolBase.toolmanager.__dict__.__setitem__('stypy_function_name', 'ToolBase.toolmanager')
        ToolBase.toolmanager.__dict__.__setitem__('stypy_param_names_list', [])
        ToolBase.toolmanager.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolBase.toolmanager.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolBase.toolmanager.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolBase.toolmanager.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolBase.toolmanager.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolBase.toolmanager.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolBase.toolmanager', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'toolmanager', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'toolmanager(...)' code ##################

        # Getting the type of 'self' (line 100)
        self_20309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'self')
        # Obtaining the member '_toolmanager' of a type (line 100)
        _toolmanager_20310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), self_20309, '_toolmanager')
        # Assigning a type to the variable 'stypy_return_type' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'stypy_return_type', _toolmanager_20310)
        
        # ################# End of 'toolmanager(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'toolmanager' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_20311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20311)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'toolmanager'
        return stypy_return_type_20311


    @norecursion
    def set_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_figure'
        module_type_store = module_type_store.open_function_context('set_figure', 102, 4, False)
        # Assigning a type to the variable 'self' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolBase.set_figure.__dict__.__setitem__('stypy_localization', localization)
        ToolBase.set_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolBase.set_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolBase.set_figure.__dict__.__setitem__('stypy_function_name', 'ToolBase.set_figure')
        ToolBase.set_figure.__dict__.__setitem__('stypy_param_names_list', ['figure'])
        ToolBase.set_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolBase.set_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolBase.set_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolBase.set_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolBase.set_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolBase.set_figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolBase.set_figure', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_figure', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_figure(...)' code ##################

        str_20312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', '\n        Assign a figure to the tool\n\n        Parameters\n        ----------\n        figure: `Figure`\n        ')
        
        # Assigning a Name to a Attribute (line 110):
        
        # Assigning a Name to a Attribute (line 110):
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'figure' (line 110)
        figure_20313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'figure')
        # Getting the type of 'self' (line 110)
        self_20314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member '_figure' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_20314, '_figure', figure_20313)
        
        # ################# End of 'set_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 102)
        stypy_return_type_20315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20315)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_figure'
        return stypy_return_type_20315


    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 112)
        None_20316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 42), 'None')
        defaults = [None_20316]
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolBase.trigger.__dict__.__setitem__('stypy_localization', localization)
        ToolBase.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolBase.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolBase.trigger.__dict__.__setitem__('stypy_function_name', 'ToolBase.trigger')
        ToolBase.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        ToolBase.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolBase.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolBase.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolBase.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolBase.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolBase.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolBase.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, ['sender', 'event', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        str_20317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, (-1)), 'str', '\n        Called when this tool gets used\n\n        This method is called by\n        `matplotlib.backend_managers.ToolManager.trigger_tool`\n\n        Parameters\n        ----------\n        event: `Event`\n            The Canvas event that caused this tool to be called\n        sender: object\n            Object that requested the tool to be triggered\n        data: object\n            Extra data\n        ')
        pass
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_20318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20318)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_20318


    @norecursion
    def name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'name'
        module_type_store = module_type_store.open_function_context('name', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolBase.name.__dict__.__setitem__('stypy_localization', localization)
        ToolBase.name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolBase.name.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolBase.name.__dict__.__setitem__('stypy_function_name', 'ToolBase.name')
        ToolBase.name.__dict__.__setitem__('stypy_param_names_list', [])
        ToolBase.name.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolBase.name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolBase.name.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolBase.name.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolBase.name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolBase.name.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolBase.name', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'name', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'name(...)' code ##################

        str_20319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'str', 'Tool Id')
        # Getting the type of 'self' (line 134)
        self_20320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'self')
        # Obtaining the member '_name' of a type (line 134)
        _name_20321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 15), self_20320, '_name')
        # Assigning a type to the variable 'stypy_return_type' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type', _name_20321)
        
        # ################# End of 'name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'name' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_20322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20322)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'name'
        return stypy_return_type_20322


    @norecursion
    def destroy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'destroy'
        module_type_store = module_type_store.open_function_context('destroy', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolBase.destroy.__dict__.__setitem__('stypy_localization', localization)
        ToolBase.destroy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolBase.destroy.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolBase.destroy.__dict__.__setitem__('stypy_function_name', 'ToolBase.destroy')
        ToolBase.destroy.__dict__.__setitem__('stypy_param_names_list', [])
        ToolBase.destroy.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolBase.destroy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolBase.destroy.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolBase.destroy.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolBase.destroy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolBase.destroy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolBase.destroy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'destroy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'destroy(...)' code ##################

        str_20323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, (-1)), 'str', '\n        Destroy the tool\n\n        This method is called when the tool is removed by\n        `matplotlib.backend_managers.ToolManager.remove_tool`\n        ')
        pass
        
        # ################# End of 'destroy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'destroy' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_20324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20324)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'destroy'
        return stypy_return_type_20324


# Assigning a type to the variable 'ToolBase' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'ToolBase', ToolBase)

# Assigning a Name to a Name (line 52):
# Getting the type of 'None' (line 52)
None_20325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'None')
# Getting the type of 'ToolBase'
ToolBase_20326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolBase')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolBase_20326, 'default_keymap', None_20325)

# Assigning a Name to a Name (line 60):
# Getting the type of 'None' (line 60)
None_20327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'None')
# Getting the type of 'ToolBase'
ToolBase_20328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolBase')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolBase_20328, 'description', None_20327)

# Assigning a Name to a Name (line 68):
# Getting the type of 'None' (line 68)
None_20329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'None')
# Getting the type of 'ToolBase'
ToolBase_20330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolBase')
# Setting the type of the member 'image' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolBase_20330, 'image', None_20329)
# Declaration of the 'ToolToggleBase' class
# Getting the type of 'ToolBase' (line 146)
ToolBase_20331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 21), 'ToolBase')

class ToolToggleBase(ToolBase_20331, ):
    str_20332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, (-1)), 'str', '\n    Toggleable tool\n\n    Every time it is triggered, it switches between enable and disable\n\n    Parameters\n    ----------\n    ``*args``\n        Variable length argument to be used by the Tool\n    ``**kwargs``\n        `toggled` if present and True, sets the initial state ot the Tool\n        Arbitrary keyword arguments to be consumed by the Tool\n    ')
    
    # Assigning a Name to a Name (line 161):
    
    # Assigning a Name to a Name (line 161):
    str_20333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, (-1)), 'str', "Attribute to group 'radio' like tools (mutually exclusive)\n\n    **String** that identifies the group or **None** if not belonging to a\n    group\n    ")
    
    # Assigning a Name to a Name (line 168):
    
    # Assigning a Name to a Name (line 168):
    str_20334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 4), 'str', 'Cursor to use when the tool is active')
    
    # Assigning a Name to a Name (line 171):
    
    # Assigning a Name to a Name (line 171):
    str_20335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 4), 'str', 'Default of toggled state')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolToggleBase.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 175):
        
        # Assigning a Call to a Attribute (line 175):
        
        # Assigning a Call to a Attribute (line 175):
        
        # Call to pop(...): (line 175)
        # Processing the call arguments (line 175)
        str_20338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 35), 'str', 'toggled')
        # Getting the type of 'self' (line 175)
        self_20339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 46), 'self', False)
        # Obtaining the member 'default_toggled' of a type (line 175)
        default_toggled_20340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 46), self_20339, 'default_toggled')
        # Processing the call keyword arguments (line 175)
        kwargs_20341 = {}
        # Getting the type of 'kwargs' (line 175)
        kwargs_20336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 175)
        pop_20337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 24), kwargs_20336, 'pop')
        # Calling pop(args, kwargs) (line 175)
        pop_call_result_20342 = invoke(stypy.reporting.localization.Localization(__file__, 175, 24), pop_20337, *[str_20338, default_toggled_20340], **kwargs_20341)
        
        # Getting the type of 'self' (line 175)
        self_20343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self')
        # Setting the type of the member '_toggled' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_20343, '_toggled', pop_call_result_20342)
        
        # Call to __init__(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'self' (line 176)
        self_20346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'self', False)
        # Getting the type of 'args' (line 176)
        args_20347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), 'args', False)
        # Processing the call keyword arguments (line 176)
        # Getting the type of 'kwargs' (line 176)
        kwargs_20348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 41), 'kwargs', False)
        kwargs_20349 = {'kwargs_20348': kwargs_20348}
        # Getting the type of 'ToolBase' (line 176)
        ToolBase_20344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'ToolBase', False)
        # Obtaining the member '__init__' of a type (line 176)
        init___20345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), ToolBase_20344, '__init__')
        # Calling __init__(args, kwargs) (line 176)
        init___call_result_20350 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), init___20345, *[self_20346, args_20347], **kwargs_20349)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 178)
        None_20351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 42), 'None')
        defaults = [None_20351]
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolToggleBase.trigger.__dict__.__setitem__('stypy_localization', localization)
        ToolToggleBase.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolToggleBase.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolToggleBase.trigger.__dict__.__setitem__('stypy_function_name', 'ToolToggleBase.trigger')
        ToolToggleBase.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        ToolToggleBase.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolToggleBase.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolToggleBase.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolToggleBase.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolToggleBase.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolToggleBase.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolToggleBase.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, ['sender', 'event', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        str_20352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 8), 'str', 'Calls `enable` or `disable` based on `toggled` value')
        
        # Getting the type of 'self' (line 180)
        self_20353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'self')
        # Obtaining the member '_toggled' of a type (line 180)
        _toggled_20354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 11), self_20353, '_toggled')
        # Testing the type of an if condition (line 180)
        if_condition_20355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 8), _toggled_20354)
        # Assigning a type to the variable 'if_condition_20355' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'if_condition_20355', if_condition_20355)
        # SSA begins for if statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to disable(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'event' (line 181)
        event_20358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 'event', False)
        # Processing the call keyword arguments (line 181)
        kwargs_20359 = {}
        # Getting the type of 'self' (line 181)
        self_20356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'self', False)
        # Obtaining the member 'disable' of a type (line 181)
        disable_20357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 12), self_20356, 'disable')
        # Calling disable(args, kwargs) (line 181)
        disable_call_result_20360 = invoke(stypy.reporting.localization.Localization(__file__, 181, 12), disable_20357, *[event_20358], **kwargs_20359)
        
        # SSA branch for the else part of an if statement (line 180)
        module_type_store.open_ssa_branch('else')
        
        # Call to enable(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'event' (line 183)
        event_20363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 24), 'event', False)
        # Processing the call keyword arguments (line 183)
        kwargs_20364 = {}
        # Getting the type of 'self' (line 183)
        self_20361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'self', False)
        # Obtaining the member 'enable' of a type (line 183)
        enable_20362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), self_20361, 'enable')
        # Calling enable(args, kwargs) (line 183)
        enable_call_result_20365 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), enable_20362, *[event_20363], **kwargs_20364)
        
        # SSA join for if statement (line 180)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a UnaryOp to a Attribute (line 184):
        
        # Assigning a UnaryOp to a Attribute (line 184):
        
        # Assigning a UnaryOp to a Attribute (line 184):
        
        # Getting the type of 'self' (line 184)
        self_20366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'self')
        # Obtaining the member '_toggled' of a type (line 184)
        _toggled_20367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 28), self_20366, '_toggled')
        # Applying the 'not' unary operator (line 184)
        result_not__20368 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 24), 'not', _toggled_20367)
        
        # Getting the type of 'self' (line 184)
        self_20369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'self')
        # Setting the type of the member '_toggled' of a type (line 184)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), self_20369, '_toggled', result_not__20368)
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_20370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20370)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_20370


    @norecursion
    def enable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 186)
        None_20371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'None')
        defaults = [None_20371]
        # Create a new context for function 'enable'
        module_type_store = module_type_store.open_function_context('enable', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolToggleBase.enable.__dict__.__setitem__('stypy_localization', localization)
        ToolToggleBase.enable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolToggleBase.enable.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolToggleBase.enable.__dict__.__setitem__('stypy_function_name', 'ToolToggleBase.enable')
        ToolToggleBase.enable.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolToggleBase.enable.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolToggleBase.enable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolToggleBase.enable.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolToggleBase.enable.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolToggleBase.enable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolToggleBase.enable.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolToggleBase.enable', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'enable', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'enable(...)' code ##################

        str_20372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, (-1)), 'str', '\n        Enable the toggle tool\n\n        `trigger` calls this method when `toggled` is False\n        ')
        pass
        
        # ################# End of 'enable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'enable' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_20373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20373)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'enable'
        return stypy_return_type_20373


    @norecursion
    def disable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 195)
        None_20374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 'None')
        defaults = [None_20374]
        # Create a new context for function 'disable'
        module_type_store = module_type_store.open_function_context('disable', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolToggleBase.disable.__dict__.__setitem__('stypy_localization', localization)
        ToolToggleBase.disable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolToggleBase.disable.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolToggleBase.disable.__dict__.__setitem__('stypy_function_name', 'ToolToggleBase.disable')
        ToolToggleBase.disable.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolToggleBase.disable.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolToggleBase.disable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolToggleBase.disable.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolToggleBase.disable.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolToggleBase.disable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolToggleBase.disable.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolToggleBase.disable', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'disable', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'disable(...)' code ##################

        str_20375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, (-1)), 'str', '\n        Disable the toggle tool\n\n        `trigger` call this method when `toggled` is True.\n\n        This can happen in different circumstances\n\n        * Click on the toolbar tool button\n        * Call to `matplotlib.backend_managers.ToolManager.trigger_tool`\n        * Another `ToolToggleBase` derived tool is triggered\n          (from the same `ToolManager`)\n        ')
        pass
        
        # ################# End of 'disable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'disable' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_20376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20376)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'disable'
        return stypy_return_type_20376


    @norecursion
    def toggled(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'toggled'
        module_type_store = module_type_store.open_function_context('toggled', 211, 4, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolToggleBase.toggled.__dict__.__setitem__('stypy_localization', localization)
        ToolToggleBase.toggled.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolToggleBase.toggled.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolToggleBase.toggled.__dict__.__setitem__('stypy_function_name', 'ToolToggleBase.toggled')
        ToolToggleBase.toggled.__dict__.__setitem__('stypy_param_names_list', [])
        ToolToggleBase.toggled.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolToggleBase.toggled.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolToggleBase.toggled.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolToggleBase.toggled.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolToggleBase.toggled.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolToggleBase.toggled.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolToggleBase.toggled', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'toggled', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'toggled(...)' code ##################

        str_20377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 8), 'str', 'State of the toggled tool')
        # Getting the type of 'self' (line 215)
        self_20378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'self')
        # Obtaining the member '_toggled' of a type (line 215)
        _toggled_20379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 15), self_20378, '_toggled')
        # Assigning a type to the variable 'stypy_return_type' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'stypy_return_type', _toggled_20379)
        
        # ################# End of 'toggled(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'toggled' in the type store
        # Getting the type of 'stypy_return_type' (line 211)
        stypy_return_type_20380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20380)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'toggled'
        return stypy_return_type_20380


    @norecursion
    def set_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_figure'
        module_type_store = module_type_store.open_function_context('set_figure', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolToggleBase.set_figure.__dict__.__setitem__('stypy_localization', localization)
        ToolToggleBase.set_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolToggleBase.set_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolToggleBase.set_figure.__dict__.__setitem__('stypy_function_name', 'ToolToggleBase.set_figure')
        ToolToggleBase.set_figure.__dict__.__setitem__('stypy_param_names_list', ['figure'])
        ToolToggleBase.set_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolToggleBase.set_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolToggleBase.set_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolToggleBase.set_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolToggleBase.set_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolToggleBase.set_figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolToggleBase.set_figure', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_figure', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_figure(...)' code ##################

        
        # Assigning a Attribute to a Name (line 218):
        
        # Assigning a Attribute to a Name (line 218):
        
        # Assigning a Attribute to a Name (line 218):
        # Getting the type of 'self' (line 218)
        self_20381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'self')
        # Obtaining the member 'toggled' of a type (line 218)
        toggled_20382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 18), self_20381, 'toggled')
        # Assigning a type to the variable 'toggled' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'toggled', toggled_20382)
        
        # Getting the type of 'toggled' (line 219)
        toggled_20383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'toggled')
        # Testing the type of an if condition (line 219)
        if_condition_20384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 8), toggled_20383)
        # Assigning a type to the variable 'if_condition_20384' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'if_condition_20384', if_condition_20384)
        # SSA begins for if statement (line 219)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 220)
        self_20385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), 'self')
        # Obtaining the member 'figure' of a type (line 220)
        figure_20386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 15), self_20385, 'figure')
        # Testing the type of an if condition (line 220)
        if_condition_20387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 12), figure_20386)
        # Assigning a type to the variable 'if_condition_20387' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'if_condition_20387', if_condition_20387)
        # SSA begins for if statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to trigger(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_20390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'self', False)
        # Getting the type of 'None' (line 221)
        None_20391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 35), 'None', False)
        # Processing the call keyword arguments (line 221)
        kwargs_20392 = {}
        # Getting the type of 'self' (line 221)
        self_20388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'self', False)
        # Obtaining the member 'trigger' of a type (line 221)
        trigger_20389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), self_20388, 'trigger')
        # Calling trigger(args, kwargs) (line 221)
        trigger_call_result_20393 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), trigger_20389, *[self_20390, None_20391], **kwargs_20392)
        
        # SSA branch for the else part of an if statement (line 220)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 225):
        
        # Assigning a Name to a Attribute (line 225):
        
        # Assigning a Name to a Attribute (line 225):
        # Getting the type of 'False' (line 225)
        False_20394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 32), 'False')
        # Getting the type of 'self' (line 225)
        self_20395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'self')
        # Setting the type of the member '_toggled' of a type (line 225)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), self_20395, '_toggled', False_20394)
        # SSA join for if statement (line 220)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 219)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_figure(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'self' (line 226)
        self_20398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 28), 'self', False)
        # Getting the type of 'figure' (line 226)
        figure_20399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 34), 'figure', False)
        # Processing the call keyword arguments (line 226)
        kwargs_20400 = {}
        # Getting the type of 'ToolBase' (line 226)
        ToolBase_20396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'ToolBase', False)
        # Obtaining the member 'set_figure' of a type (line 226)
        set_figure_20397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), ToolBase_20396, 'set_figure')
        # Calling set_figure(args, kwargs) (line 226)
        set_figure_call_result_20401 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), set_figure_20397, *[self_20398, figure_20399], **kwargs_20400)
        
        
        # Getting the type of 'toggled' (line 227)
        toggled_20402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'toggled')
        # Testing the type of an if condition (line 227)
        if_condition_20403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), toggled_20402)
        # Assigning a type to the variable 'if_condition_20403' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_20403', if_condition_20403)
        # SSA begins for if statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'figure' (line 228)
        figure_20404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'figure')
        # Testing the type of an if condition (line 228)
        if_condition_20405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 12), figure_20404)
        # Assigning a type to the variable 'if_condition_20405' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'if_condition_20405', if_condition_20405)
        # SSA begins for if statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to trigger(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'self' (line 229)
        self_20408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 29), 'self', False)
        # Getting the type of 'None' (line 229)
        None_20409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), 'None', False)
        # Processing the call keyword arguments (line 229)
        kwargs_20410 = {}
        # Getting the type of 'self' (line 229)
        self_20406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'self', False)
        # Obtaining the member 'trigger' of a type (line 229)
        trigger_20407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 16), self_20406, 'trigger')
        # Calling trigger(args, kwargs) (line 229)
        trigger_call_result_20411 = invoke(stypy.reporting.localization.Localization(__file__, 229, 16), trigger_20407, *[self_20408, None_20409], **kwargs_20410)
        
        # SSA branch for the else part of an if statement (line 228)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 233):
        
        # Assigning a Name to a Attribute (line 233):
        
        # Assigning a Name to a Attribute (line 233):
        # Getting the type of 'True' (line 233)
        True_20412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 32), 'True')
        # Getting the type of 'self' (line 233)
        self_20413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'self')
        # Setting the type of the member '_toggled' of a type (line 233)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 16), self_20413, '_toggled', True_20412)
        # SSA join for if statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 227)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_20414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20414)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_figure'
        return stypy_return_type_20414


# Assigning a type to the variable 'ToolToggleBase' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'ToolToggleBase', ToolToggleBase)

# Assigning a Name to a Name (line 161):
# Getting the type of 'None' (line 161)
None_20415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 18), 'None')
# Getting the type of 'ToolToggleBase'
ToolToggleBase_20416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolToggleBase')
# Setting the type of the member 'radio_group' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolToggleBase_20416, 'radio_group', None_20415)

# Assigning a Name to a Name (line 168):
# Getting the type of 'None' (line 168)
None_20417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'None')
# Getting the type of 'ToolToggleBase'
ToolToggleBase_20418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolToggleBase')
# Setting the type of the member 'cursor' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolToggleBase_20418, 'cursor', None_20417)

# Assigning a Name to a Name (line 171):
# Getting the type of 'False' (line 171)
False_20419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 22), 'False')
# Getting the type of 'ToolToggleBase'
ToolToggleBase_20420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolToggleBase')
# Setting the type of the member 'default_toggled' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolToggleBase_20420, 'default_toggled', False_20419)
# Declaration of the 'SetCursorBase' class
# Getting the type of 'ToolBase' (line 236)
ToolBase_20421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'ToolBase')

class SetCursorBase(ToolBase_20421, ):
    str_20422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, (-1)), 'str', '\n    Change to the current cursor while inaxes\n\n    This tool, keeps track of all `ToolToggleBase` derived tools, and calls\n    set_cursor when a tool gets triggered\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 243, 4, False)
        # Assigning a type to the variable 'self' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SetCursorBase.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'self' (line 244)
        self_20425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 26), 'self', False)
        # Getting the type of 'args' (line 244)
        args_20426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 33), 'args', False)
        # Processing the call keyword arguments (line 244)
        # Getting the type of 'kwargs' (line 244)
        kwargs_20427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 41), 'kwargs', False)
        kwargs_20428 = {'kwargs_20427': kwargs_20427}
        # Getting the type of 'ToolBase' (line 244)
        ToolBase_20423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'ToolBase', False)
        # Obtaining the member '__init__' of a type (line 244)
        init___20424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), ToolBase_20423, '__init__')
        # Calling __init__(args, kwargs) (line 244)
        init___call_result_20429 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), init___20424, *[self_20425, args_20426], **kwargs_20428)
        
        
        # Assigning a Name to a Attribute (line 245):
        
        # Assigning a Name to a Attribute (line 245):
        
        # Assigning a Name to a Attribute (line 245):
        # Getting the type of 'None' (line 245)
        None_20430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 23), 'None')
        # Getting the type of 'self' (line 245)
        self_20431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self')
        # Setting the type of the member '_idDrag' of a type (line 245)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_20431, '_idDrag', None_20430)
        
        # Assigning a Name to a Attribute (line 246):
        
        # Assigning a Name to a Attribute (line 246):
        
        # Assigning a Name to a Attribute (line 246):
        # Getting the type of 'None' (line 246)
        None_20432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'None')
        # Getting the type of 'self' (line 246)
        self_20433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'self')
        # Setting the type of the member '_cursor' of a type (line 246)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), self_20433, '_cursor', None_20432)
        
        # Assigning a Attribute to a Attribute (line 247):
        
        # Assigning a Attribute to a Attribute (line 247):
        
        # Assigning a Attribute to a Attribute (line 247):
        # Getting the type of 'cursors' (line 247)
        cursors_20434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 31), 'cursors')
        # Obtaining the member 'POINTER' of a type (line 247)
        POINTER_20435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 31), cursors_20434, 'POINTER')
        # Getting the type of 'self' (line 247)
        self_20436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'self')
        # Setting the type of the member '_default_cursor' of a type (line 247)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), self_20436, '_default_cursor', POINTER_20435)
        
        # Assigning a Attribute to a Attribute (line 248):
        
        # Assigning a Attribute to a Attribute (line 248):
        
        # Assigning a Attribute to a Attribute (line 248):
        # Getting the type of 'self' (line 248)
        self_20437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 28), 'self')
        # Obtaining the member '_default_cursor' of a type (line 248)
        _default_cursor_20438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 28), self_20437, '_default_cursor')
        # Getting the type of 'self' (line 248)
        self_20439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self')
        # Setting the type of the member '_last_cursor' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_20439, '_last_cursor', _default_cursor_20438)
        
        # Call to toolmanager_connect(...): (line 249)
        # Processing the call arguments (line 249)
        str_20443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 45), 'str', 'tool_added_event')
        # Getting the type of 'self' (line 250)
        self_20444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 45), 'self', False)
        # Obtaining the member '_add_tool_cbk' of a type (line 250)
        _add_tool_cbk_20445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 45), self_20444, '_add_tool_cbk')
        # Processing the call keyword arguments (line 249)
        kwargs_20446 = {}
        # Getting the type of 'self' (line 249)
        self_20440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 249)
        toolmanager_20441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), self_20440, 'toolmanager')
        # Obtaining the member 'toolmanager_connect' of a type (line 249)
        toolmanager_connect_20442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), toolmanager_20441, 'toolmanager_connect')
        # Calling toolmanager_connect(args, kwargs) (line 249)
        toolmanager_connect_call_result_20447 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), toolmanager_connect_20442, *[str_20443, _add_tool_cbk_20445], **kwargs_20446)
        
        
        
        # Call to values(...): (line 253)
        # Processing the call keyword arguments (line 253)
        kwargs_20452 = {}
        # Getting the type of 'self' (line 253)
        self_20448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 253)
        toolmanager_20449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), self_20448, 'toolmanager')
        # Obtaining the member 'tools' of a type (line 253)
        tools_20450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), toolmanager_20449, 'tools')
        # Obtaining the member 'values' of a type (line 253)
        values_20451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), tools_20450, 'values')
        # Calling values(args, kwargs) (line 253)
        values_call_result_20453 = invoke(stypy.reporting.localization.Localization(__file__, 253, 20), values_20451, *[], **kwargs_20452)
        
        # Testing the type of a for loop iterable (line 253)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 253, 8), values_call_result_20453)
        # Getting the type of the for loop variable (line 253)
        for_loop_var_20454 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 253, 8), values_call_result_20453)
        # Assigning a type to the variable 'tool' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'tool', for_loop_var_20454)
        # SSA begins for a for statement (line 253)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _add_tool(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'tool' (line 254)
        tool_20457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 27), 'tool', False)
        # Processing the call keyword arguments (line 254)
        kwargs_20458 = {}
        # Getting the type of 'self' (line 254)
        self_20455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'self', False)
        # Obtaining the member '_add_tool' of a type (line 254)
        _add_tool_20456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), self_20455, '_add_tool')
        # Calling _add_tool(args, kwargs) (line 254)
        _add_tool_call_result_20459 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), _add_tool_20456, *[tool_20457], **kwargs_20458)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_figure'
        module_type_store = module_type_store.open_function_context('set_figure', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SetCursorBase.set_figure.__dict__.__setitem__('stypy_localization', localization)
        SetCursorBase.set_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SetCursorBase.set_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        SetCursorBase.set_figure.__dict__.__setitem__('stypy_function_name', 'SetCursorBase.set_figure')
        SetCursorBase.set_figure.__dict__.__setitem__('stypy_param_names_list', ['figure'])
        SetCursorBase.set_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        SetCursorBase.set_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SetCursorBase.set_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        SetCursorBase.set_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        SetCursorBase.set_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SetCursorBase.set_figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SetCursorBase.set_figure', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_figure', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_figure(...)' code ##################

        
        # Getting the type of 'self' (line 257)
        self_20460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 'self')
        # Obtaining the member '_idDrag' of a type (line 257)
        _idDrag_20461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 11), self_20460, '_idDrag')
        # Testing the type of an if condition (line 257)
        if_condition_20462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 8), _idDrag_20461)
        # Assigning a type to the variable 'if_condition_20462' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'if_condition_20462', if_condition_20462)
        # SSA begins for if statement (line 257)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mpl_disconnect(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'self' (line 258)
        self_20466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 39), 'self', False)
        # Obtaining the member '_idDrag' of a type (line 258)
        _idDrag_20467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 39), self_20466, '_idDrag')
        # Processing the call keyword arguments (line 258)
        kwargs_20468 = {}
        # Getting the type of 'self' (line 258)
        self_20463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'self', False)
        # Obtaining the member 'canvas' of a type (line 258)
        canvas_20464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), self_20463, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 258)
        mpl_disconnect_20465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), canvas_20464, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 258)
        mpl_disconnect_call_result_20469 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), mpl_disconnect_20465, *[_idDrag_20467], **kwargs_20468)
        
        # SSA join for if statement (line 257)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_figure(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'self' (line 259)
        self_20472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'self', False)
        # Getting the type of 'figure' (line 259)
        figure_20473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 34), 'figure', False)
        # Processing the call keyword arguments (line 259)
        kwargs_20474 = {}
        # Getting the type of 'ToolBase' (line 259)
        ToolBase_20470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'ToolBase', False)
        # Obtaining the member 'set_figure' of a type (line 259)
        set_figure_20471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), ToolBase_20470, 'set_figure')
        # Calling set_figure(args, kwargs) (line 259)
        set_figure_call_result_20475 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), set_figure_20471, *[self_20472, figure_20473], **kwargs_20474)
        
        
        # Getting the type of 'figure' (line 260)
        figure_20476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'figure')
        # Testing the type of an if condition (line 260)
        if_condition_20477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 8), figure_20476)
        # Assigning a type to the variable 'if_condition_20477' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'if_condition_20477', if_condition_20477)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 261):
        
        # Assigning a Call to a Attribute (line 261):
        
        # Assigning a Call to a Attribute (line 261):
        
        # Call to mpl_connect(...): (line 261)
        # Processing the call arguments (line 261)
        str_20481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 16), 'str', 'motion_notify_event')
        # Getting the type of 'self' (line 262)
        self_20482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 39), 'self', False)
        # Obtaining the member '_set_cursor_cbk' of a type (line 262)
        _set_cursor_cbk_20483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 39), self_20482, '_set_cursor_cbk')
        # Processing the call keyword arguments (line 261)
        kwargs_20484 = {}
        # Getting the type of 'self' (line 261)
        self_20478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 27), 'self', False)
        # Obtaining the member 'canvas' of a type (line 261)
        canvas_20479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 27), self_20478, 'canvas')
        # Obtaining the member 'mpl_connect' of a type (line 261)
        mpl_connect_20480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 27), canvas_20479, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 261)
        mpl_connect_call_result_20485 = invoke(stypy.reporting.localization.Localization(__file__, 261, 27), mpl_connect_20480, *[str_20481, _set_cursor_cbk_20483], **kwargs_20484)
        
        # Getting the type of 'self' (line 261)
        self_20486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'self')
        # Setting the type of the member '_idDrag' of a type (line 261)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 12), self_20486, '_idDrag', mpl_connect_call_result_20485)
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_20487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20487)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_figure'
        return stypy_return_type_20487


    @norecursion
    def _tool_trigger_cbk(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_tool_trigger_cbk'
        module_type_store = module_type_store.open_function_context('_tool_trigger_cbk', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SetCursorBase._tool_trigger_cbk.__dict__.__setitem__('stypy_localization', localization)
        SetCursorBase._tool_trigger_cbk.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SetCursorBase._tool_trigger_cbk.__dict__.__setitem__('stypy_type_store', module_type_store)
        SetCursorBase._tool_trigger_cbk.__dict__.__setitem__('stypy_function_name', 'SetCursorBase._tool_trigger_cbk')
        SetCursorBase._tool_trigger_cbk.__dict__.__setitem__('stypy_param_names_list', ['event'])
        SetCursorBase._tool_trigger_cbk.__dict__.__setitem__('stypy_varargs_param_name', None)
        SetCursorBase._tool_trigger_cbk.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SetCursorBase._tool_trigger_cbk.__dict__.__setitem__('stypy_call_defaults', defaults)
        SetCursorBase._tool_trigger_cbk.__dict__.__setitem__('stypy_call_varargs', varargs)
        SetCursorBase._tool_trigger_cbk.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SetCursorBase._tool_trigger_cbk.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SetCursorBase._tool_trigger_cbk', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_tool_trigger_cbk', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_tool_trigger_cbk(...)' code ##################

        
        # Getting the type of 'event' (line 265)
        event_20488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'event')
        # Obtaining the member 'tool' of a type (line 265)
        tool_20489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), event_20488, 'tool')
        # Obtaining the member 'toggled' of a type (line 265)
        toggled_20490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), tool_20489, 'toggled')
        # Testing the type of an if condition (line 265)
        if_condition_20491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), toggled_20490)
        # Assigning a type to the variable 'if_condition_20491' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_20491', if_condition_20491)
        # SSA begins for if statement (line 265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 266):
        
        # Assigning a Attribute to a Attribute (line 266):
        
        # Assigning a Attribute to a Attribute (line 266):
        # Getting the type of 'event' (line 266)
        event_20492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 27), 'event')
        # Obtaining the member 'tool' of a type (line 266)
        tool_20493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 27), event_20492, 'tool')
        # Obtaining the member 'cursor' of a type (line 266)
        cursor_20494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 27), tool_20493, 'cursor')
        # Getting the type of 'self' (line 266)
        self_20495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'self')
        # Setting the type of the member '_cursor' of a type (line 266)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), self_20495, '_cursor', cursor_20494)
        # SSA branch for the else part of an if statement (line 265)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 268):
        
        # Assigning a Name to a Attribute (line 268):
        
        # Assigning a Name to a Attribute (line 268):
        # Getting the type of 'None' (line 268)
        None_20496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 27), 'None')
        # Getting the type of 'self' (line 268)
        self_20497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'self')
        # Setting the type of the member '_cursor' of a type (line 268)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 12), self_20497, '_cursor', None_20496)
        # SSA join for if statement (line 265)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _set_cursor_cbk(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'event' (line 270)
        event_20500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 29), 'event', False)
        # Obtaining the member 'canvasevent' of a type (line 270)
        canvasevent_20501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 29), event_20500, 'canvasevent')
        # Processing the call keyword arguments (line 270)
        kwargs_20502 = {}
        # Getting the type of 'self' (line 270)
        self_20498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self', False)
        # Obtaining the member '_set_cursor_cbk' of a type (line 270)
        _set_cursor_cbk_20499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_20498, '_set_cursor_cbk')
        # Calling _set_cursor_cbk(args, kwargs) (line 270)
        _set_cursor_cbk_call_result_20503 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), _set_cursor_cbk_20499, *[canvasevent_20501], **kwargs_20502)
        
        
        # ################# End of '_tool_trigger_cbk(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_tool_trigger_cbk' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_20504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20504)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_tool_trigger_cbk'
        return stypy_return_type_20504


    @norecursion
    def _add_tool(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_tool'
        module_type_store = module_type_store.open_function_context('_add_tool', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SetCursorBase._add_tool.__dict__.__setitem__('stypy_localization', localization)
        SetCursorBase._add_tool.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SetCursorBase._add_tool.__dict__.__setitem__('stypy_type_store', module_type_store)
        SetCursorBase._add_tool.__dict__.__setitem__('stypy_function_name', 'SetCursorBase._add_tool')
        SetCursorBase._add_tool.__dict__.__setitem__('stypy_param_names_list', ['tool'])
        SetCursorBase._add_tool.__dict__.__setitem__('stypy_varargs_param_name', None)
        SetCursorBase._add_tool.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SetCursorBase._add_tool.__dict__.__setitem__('stypy_call_defaults', defaults)
        SetCursorBase._add_tool.__dict__.__setitem__('stypy_call_varargs', varargs)
        SetCursorBase._add_tool.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SetCursorBase._add_tool.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SetCursorBase._add_tool', ['tool'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_tool', localization, ['tool'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_tool(...)' code ##################

        str_20505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 8), 'str', 'set the cursor when the tool is triggered')
        
        
        
        # Call to getattr(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'tool' (line 274)
        tool_20507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 'tool', False)
        str_20508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 25), 'str', 'cursor')
        # Getting the type of 'None' (line 274)
        None_20509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'None', False)
        # Processing the call keyword arguments (line 274)
        kwargs_20510 = {}
        # Getting the type of 'getattr' (line 274)
        getattr_20506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 11), 'getattr', False)
        # Calling getattr(args, kwargs) (line 274)
        getattr_call_result_20511 = invoke(stypy.reporting.localization.Localization(__file__, 274, 11), getattr_20506, *[tool_20507, str_20508, None_20509], **kwargs_20510)
        
        # Getting the type of 'None' (line 274)
        None_20512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 48), 'None')
        # Applying the binary operator 'isnot' (line 274)
        result_is_not_20513 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 11), 'isnot', getattr_call_result_20511, None_20512)
        
        # Testing the type of an if condition (line 274)
        if_condition_20514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 8), result_is_not_20513)
        # Assigning a type to the variable 'if_condition_20514' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'if_condition_20514', if_condition_20514)
        # SSA begins for if statement (line 274)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to toolmanager_connect(...): (line 275)
        # Processing the call arguments (line 275)
        str_20518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 49), 'str', 'tool_trigger_%s')
        # Getting the type of 'tool' (line 275)
        tool_20519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 69), 'tool', False)
        # Obtaining the member 'name' of a type (line 275)
        name_20520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 69), tool_20519, 'name')
        # Applying the binary operator '%' (line 275)
        result_mod_20521 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 49), '%', str_20518, name_20520)
        
        # Getting the type of 'self' (line 276)
        self_20522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 49), 'self', False)
        # Obtaining the member '_tool_trigger_cbk' of a type (line 276)
        _tool_trigger_cbk_20523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 49), self_20522, '_tool_trigger_cbk')
        # Processing the call keyword arguments (line 275)
        kwargs_20524 = {}
        # Getting the type of 'self' (line 275)
        self_20515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 275)
        toolmanager_20516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), self_20515, 'toolmanager')
        # Obtaining the member 'toolmanager_connect' of a type (line 275)
        toolmanager_connect_20517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), toolmanager_20516, 'toolmanager_connect')
        # Calling toolmanager_connect(args, kwargs) (line 275)
        toolmanager_connect_call_result_20525 = invoke(stypy.reporting.localization.Localization(__file__, 275, 12), toolmanager_connect_20517, *[result_mod_20521, _tool_trigger_cbk_20523], **kwargs_20524)
        
        # SSA join for if statement (line 274)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_add_tool(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_tool' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_20526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20526)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_tool'
        return stypy_return_type_20526


    @norecursion
    def _add_tool_cbk(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_tool_cbk'
        module_type_store = module_type_store.open_function_context('_add_tool_cbk', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SetCursorBase._add_tool_cbk.__dict__.__setitem__('stypy_localization', localization)
        SetCursorBase._add_tool_cbk.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SetCursorBase._add_tool_cbk.__dict__.__setitem__('stypy_type_store', module_type_store)
        SetCursorBase._add_tool_cbk.__dict__.__setitem__('stypy_function_name', 'SetCursorBase._add_tool_cbk')
        SetCursorBase._add_tool_cbk.__dict__.__setitem__('stypy_param_names_list', ['event'])
        SetCursorBase._add_tool_cbk.__dict__.__setitem__('stypy_varargs_param_name', None)
        SetCursorBase._add_tool_cbk.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SetCursorBase._add_tool_cbk.__dict__.__setitem__('stypy_call_defaults', defaults)
        SetCursorBase._add_tool_cbk.__dict__.__setitem__('stypy_call_varargs', varargs)
        SetCursorBase._add_tool_cbk.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SetCursorBase._add_tool_cbk.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SetCursorBase._add_tool_cbk', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_tool_cbk', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_tool_cbk(...)' code ##################

        str_20527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 8), 'str', 'Process every newly added tool')
        
        
        # Getting the type of 'event' (line 280)
        event_20528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'event')
        # Obtaining the member 'tool' of a type (line 280)
        tool_20529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), event_20528, 'tool')
        # Getting the type of 'self' (line 280)
        self_20530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'self')
        # Applying the binary operator 'is' (line 280)
        result_is__20531 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 11), 'is', tool_20529, self_20530)
        
        # Testing the type of an if condition (line 280)
        if_condition_20532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 8), result_is__20531)
        # Assigning a type to the variable 'if_condition_20532' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'if_condition_20532', if_condition_20532)
        # SSA begins for if statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _add_tool(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'event' (line 283)
        event_20535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 23), 'event', False)
        # Obtaining the member 'tool' of a type (line 283)
        tool_20536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 23), event_20535, 'tool')
        # Processing the call keyword arguments (line 283)
        kwargs_20537 = {}
        # Getting the type of 'self' (line 283)
        self_20533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self', False)
        # Obtaining the member '_add_tool' of a type (line 283)
        _add_tool_20534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_20533, '_add_tool')
        # Calling _add_tool(args, kwargs) (line 283)
        _add_tool_call_result_20538 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), _add_tool_20534, *[tool_20536], **kwargs_20537)
        
        
        # ################# End of '_add_tool_cbk(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_tool_cbk' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_20539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20539)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_tool_cbk'
        return stypy_return_type_20539


    @norecursion
    def _set_cursor_cbk(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_cursor_cbk'
        module_type_store = module_type_store.open_function_context('_set_cursor_cbk', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SetCursorBase._set_cursor_cbk.__dict__.__setitem__('stypy_localization', localization)
        SetCursorBase._set_cursor_cbk.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SetCursorBase._set_cursor_cbk.__dict__.__setitem__('stypy_type_store', module_type_store)
        SetCursorBase._set_cursor_cbk.__dict__.__setitem__('stypy_function_name', 'SetCursorBase._set_cursor_cbk')
        SetCursorBase._set_cursor_cbk.__dict__.__setitem__('stypy_param_names_list', ['event'])
        SetCursorBase._set_cursor_cbk.__dict__.__setitem__('stypy_varargs_param_name', None)
        SetCursorBase._set_cursor_cbk.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SetCursorBase._set_cursor_cbk.__dict__.__setitem__('stypy_call_defaults', defaults)
        SetCursorBase._set_cursor_cbk.__dict__.__setitem__('stypy_call_varargs', varargs)
        SetCursorBase._set_cursor_cbk.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SetCursorBase._set_cursor_cbk.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SetCursorBase._set_cursor_cbk', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_cursor_cbk', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_cursor_cbk(...)' code ##################

        
        
        # Getting the type of 'event' (line 286)
        event_20540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 15), 'event')
        # Applying the 'not' unary operator (line 286)
        result_not__20541 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 11), 'not', event_20540)
        
        # Testing the type of an if condition (line 286)
        if_condition_20542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 8), result_not__20541)
        # Assigning a type to the variable 'if_condition_20542' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'if_condition_20542', if_condition_20542)
        # SSA begins for if statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 286)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to getattr(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'event' (line 289)
        event_20544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 23), 'event', False)
        str_20545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 30), 'str', 'inaxes')
        # Getting the type of 'False' (line 289)
        False_20546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 40), 'False', False)
        # Processing the call keyword arguments (line 289)
        kwargs_20547 = {}
        # Getting the type of 'getattr' (line 289)
        getattr_20543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 289)
        getattr_call_result_20548 = invoke(stypy.reporting.localization.Localization(__file__, 289, 15), getattr_20543, *[event_20544, str_20545, False_20546], **kwargs_20547)
        
        # Applying the 'not' unary operator (line 289)
        result_not__20549 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 11), 'not', getattr_call_result_20548)
        
        
        # Getting the type of 'self' (line 289)
        self_20550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 54), 'self')
        # Obtaining the member '_cursor' of a type (line 289)
        _cursor_20551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 54), self_20550, '_cursor')
        # Applying the 'not' unary operator (line 289)
        result_not__20552 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 50), 'not', _cursor_20551)
        
        # Applying the binary operator 'or' (line 289)
        result_or_keyword_20553 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 11), 'or', result_not__20549, result_not__20552)
        
        # Testing the type of an if condition (line 289)
        if_condition_20554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 8), result_or_keyword_20553)
        # Assigning a type to the variable 'if_condition_20554' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'if_condition_20554', if_condition_20554)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 290)
        self_20555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'self')
        # Obtaining the member '_last_cursor' of a type (line 290)
        _last_cursor_20556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 15), self_20555, '_last_cursor')
        # Getting the type of 'self' (line 290)
        self_20557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 36), 'self')
        # Obtaining the member '_default_cursor' of a type (line 290)
        _default_cursor_20558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 36), self_20557, '_default_cursor')
        # Applying the binary operator '!=' (line 290)
        result_ne_20559 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 15), '!=', _last_cursor_20556, _default_cursor_20558)
        
        # Testing the type of an if condition (line 290)
        if_condition_20560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 12), result_ne_20559)
        # Assigning a type to the variable 'if_condition_20560' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'if_condition_20560', if_condition_20560)
        # SSA begins for if statement (line 290)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_cursor(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'self' (line 291)
        self_20563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 32), 'self', False)
        # Obtaining the member '_default_cursor' of a type (line 291)
        _default_cursor_20564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 32), self_20563, '_default_cursor')
        # Processing the call keyword arguments (line 291)
        kwargs_20565 = {}
        # Getting the type of 'self' (line 291)
        self_20561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'self', False)
        # Obtaining the member 'set_cursor' of a type (line 291)
        set_cursor_20562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 16), self_20561, 'set_cursor')
        # Calling set_cursor(args, kwargs) (line 291)
        set_cursor_call_result_20566 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), set_cursor_20562, *[_default_cursor_20564], **kwargs_20565)
        
        
        # Assigning a Attribute to a Attribute (line 292):
        
        # Assigning a Attribute to a Attribute (line 292):
        
        # Assigning a Attribute to a Attribute (line 292):
        # Getting the type of 'self' (line 292)
        self_20567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 36), 'self')
        # Obtaining the member '_default_cursor' of a type (line 292)
        _default_cursor_20568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 36), self_20567, '_default_cursor')
        # Getting the type of 'self' (line 292)
        self_20569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'self')
        # Setting the type of the member '_last_cursor' of a type (line 292)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 16), self_20569, '_last_cursor', _default_cursor_20568)
        # SSA join for if statement (line 290)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 289)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 293)
        self_20570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'self')
        # Obtaining the member '_cursor' of a type (line 293)
        _cursor_20571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 13), self_20570, '_cursor')
        # Testing the type of an if condition (line 293)
        if_condition_20572 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 13), _cursor_20571)
        # Assigning a type to the variable 'if_condition_20572' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'if_condition_20572', if_condition_20572)
        # SSA begins for if statement (line 293)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 294):
        
        # Assigning a Attribute to a Name (line 294):
        
        # Assigning a Attribute to a Name (line 294):
        # Getting the type of 'self' (line 294)
        self_20573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'self')
        # Obtaining the member '_cursor' of a type (line 294)
        _cursor_20574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 21), self_20573, '_cursor')
        # Assigning a type to the variable 'cursor' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'cursor', _cursor_20574)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'cursor' (line 295)
        cursor_20575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'cursor')
        
        # Getting the type of 'self' (line 295)
        self_20576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 26), 'self')
        # Obtaining the member '_last_cursor' of a type (line 295)
        _last_cursor_20577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 26), self_20576, '_last_cursor')
        # Getting the type of 'cursor' (line 295)
        cursor_20578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 47), 'cursor')
        # Applying the binary operator '!=' (line 295)
        result_ne_20579 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 26), '!=', _last_cursor_20577, cursor_20578)
        
        # Applying the binary operator 'and' (line 295)
        result_and_keyword_20580 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 15), 'and', cursor_20575, result_ne_20579)
        
        # Testing the type of an if condition (line 295)
        if_condition_20581 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 12), result_and_keyword_20580)
        # Assigning a type to the variable 'if_condition_20581' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'if_condition_20581', if_condition_20581)
        # SSA begins for if statement (line 295)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_cursor(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'cursor' (line 296)
        cursor_20584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 32), 'cursor', False)
        # Processing the call keyword arguments (line 296)
        kwargs_20585 = {}
        # Getting the type of 'self' (line 296)
        self_20582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'self', False)
        # Obtaining the member 'set_cursor' of a type (line 296)
        set_cursor_20583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 16), self_20582, 'set_cursor')
        # Calling set_cursor(args, kwargs) (line 296)
        set_cursor_call_result_20586 = invoke(stypy.reporting.localization.Localization(__file__, 296, 16), set_cursor_20583, *[cursor_20584], **kwargs_20585)
        
        
        # Assigning a Name to a Attribute (line 297):
        
        # Assigning a Name to a Attribute (line 297):
        
        # Assigning a Name to a Attribute (line 297):
        # Getting the type of 'cursor' (line 297)
        cursor_20587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 36), 'cursor')
        # Getting the type of 'self' (line 297)
        self_20588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'self')
        # Setting the type of the member '_last_cursor' of a type (line 297)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 16), self_20588, '_last_cursor', cursor_20587)
        # SSA join for if statement (line 295)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 293)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_set_cursor_cbk(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_cursor_cbk' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_20589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20589)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_cursor_cbk'
        return stypy_return_type_20589


    @norecursion
    def set_cursor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_cursor'
        module_type_store = module_type_store.open_function_context('set_cursor', 299, 4, False)
        # Assigning a type to the variable 'self' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SetCursorBase.set_cursor.__dict__.__setitem__('stypy_localization', localization)
        SetCursorBase.set_cursor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SetCursorBase.set_cursor.__dict__.__setitem__('stypy_type_store', module_type_store)
        SetCursorBase.set_cursor.__dict__.__setitem__('stypy_function_name', 'SetCursorBase.set_cursor')
        SetCursorBase.set_cursor.__dict__.__setitem__('stypy_param_names_list', ['cursor'])
        SetCursorBase.set_cursor.__dict__.__setitem__('stypy_varargs_param_name', None)
        SetCursorBase.set_cursor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SetCursorBase.set_cursor.__dict__.__setitem__('stypy_call_defaults', defaults)
        SetCursorBase.set_cursor.__dict__.__setitem__('stypy_call_varargs', varargs)
        SetCursorBase.set_cursor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SetCursorBase.set_cursor.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SetCursorBase.set_cursor', ['cursor'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_cursor', localization, ['cursor'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_cursor(...)' code ##################

        str_20590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, (-1)), 'str', '\n        Set the cursor\n\n        This method has to be implemented per backend\n        ')
        # Getting the type of 'NotImplementedError' (line 305)
        NotImplementedError_20591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 305, 8), NotImplementedError_20591, 'raise parameter', BaseException)
        
        # ################# End of 'set_cursor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_cursor' in the type store
        # Getting the type of 'stypy_return_type' (line 299)
        stypy_return_type_20592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20592)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_cursor'
        return stypy_return_type_20592


# Assigning a type to the variable 'SetCursorBase' (line 236)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'SetCursorBase', SetCursorBase)
# Declaration of the 'ToolCursorPosition' class
# Getting the type of 'ToolBase' (line 308)
ToolBase_20593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'ToolBase')

class ToolCursorPosition(ToolBase_20593, ):
    str_20594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, (-1)), 'str', '\n    Send message with the current pointer position\n\n    This tool runs in the background reporting the position of the cursor\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 314, 4, False)
        # Assigning a type to the variable 'self' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolCursorPosition.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 315):
        
        # Assigning a Name to a Attribute (line 315):
        
        # Assigning a Name to a Attribute (line 315):
        # Getting the type of 'None' (line 315)
        None_20595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'None')
        # Getting the type of 'self' (line 315)
        self_20596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self')
        # Setting the type of the member '_idDrag' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_20596, '_idDrag', None_20595)
        
        # Call to __init__(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'self' (line 316)
        self_20599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 26), 'self', False)
        # Getting the type of 'args' (line 316)
        args_20600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 33), 'args', False)
        # Processing the call keyword arguments (line 316)
        # Getting the type of 'kwargs' (line 316)
        kwargs_20601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 41), 'kwargs', False)
        kwargs_20602 = {'kwargs_20601': kwargs_20601}
        # Getting the type of 'ToolBase' (line 316)
        ToolBase_20597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'ToolBase', False)
        # Obtaining the member '__init__' of a type (line 316)
        init___20598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), ToolBase_20597, '__init__')
        # Calling __init__(args, kwargs) (line 316)
        init___call_result_20603 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), init___20598, *[self_20599, args_20600], **kwargs_20602)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_figure'
        module_type_store = module_type_store.open_function_context('set_figure', 318, 4, False)
        # Assigning a type to the variable 'self' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolCursorPosition.set_figure.__dict__.__setitem__('stypy_localization', localization)
        ToolCursorPosition.set_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolCursorPosition.set_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolCursorPosition.set_figure.__dict__.__setitem__('stypy_function_name', 'ToolCursorPosition.set_figure')
        ToolCursorPosition.set_figure.__dict__.__setitem__('stypy_param_names_list', ['figure'])
        ToolCursorPosition.set_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolCursorPosition.set_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolCursorPosition.set_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolCursorPosition.set_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolCursorPosition.set_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolCursorPosition.set_figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolCursorPosition.set_figure', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_figure', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_figure(...)' code ##################

        
        # Getting the type of 'self' (line 319)
        self_20604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 11), 'self')
        # Obtaining the member '_idDrag' of a type (line 319)
        _idDrag_20605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 11), self_20604, '_idDrag')
        # Testing the type of an if condition (line 319)
        if_condition_20606 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 8), _idDrag_20605)
        # Assigning a type to the variable 'if_condition_20606' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'if_condition_20606', if_condition_20606)
        # SSA begins for if statement (line 319)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mpl_disconnect(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'self' (line 320)
        self_20610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 39), 'self', False)
        # Obtaining the member '_idDrag' of a type (line 320)
        _idDrag_20611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 39), self_20610, '_idDrag')
        # Processing the call keyword arguments (line 320)
        kwargs_20612 = {}
        # Getting the type of 'self' (line 320)
        self_20607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'self', False)
        # Obtaining the member 'canvas' of a type (line 320)
        canvas_20608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), self_20607, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 320)
        mpl_disconnect_20609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), canvas_20608, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 320)
        mpl_disconnect_call_result_20613 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), mpl_disconnect_20609, *[_idDrag_20611], **kwargs_20612)
        
        # SSA join for if statement (line 319)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_figure(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'self' (line 321)
        self_20616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 28), 'self', False)
        # Getting the type of 'figure' (line 321)
        figure_20617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 34), 'figure', False)
        # Processing the call keyword arguments (line 321)
        kwargs_20618 = {}
        # Getting the type of 'ToolBase' (line 321)
        ToolBase_20614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'ToolBase', False)
        # Obtaining the member 'set_figure' of a type (line 321)
        set_figure_20615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), ToolBase_20614, 'set_figure')
        # Calling set_figure(args, kwargs) (line 321)
        set_figure_call_result_20619 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), set_figure_20615, *[self_20616, figure_20617], **kwargs_20618)
        
        
        # Getting the type of 'figure' (line 322)
        figure_20620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 11), 'figure')
        # Testing the type of an if condition (line 322)
        if_condition_20621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 8), figure_20620)
        # Assigning a type to the variable 'if_condition_20621' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'if_condition_20621', if_condition_20621)
        # SSA begins for if statement (line 322)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 323):
        
        # Assigning a Call to a Attribute (line 323):
        
        # Assigning a Call to a Attribute (line 323):
        
        # Call to mpl_connect(...): (line 323)
        # Processing the call arguments (line 323)
        str_20625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 16), 'str', 'motion_notify_event')
        # Getting the type of 'self' (line 324)
        self_20626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 39), 'self', False)
        # Obtaining the member 'send_message' of a type (line 324)
        send_message_20627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 39), self_20626, 'send_message')
        # Processing the call keyword arguments (line 323)
        kwargs_20628 = {}
        # Getting the type of 'self' (line 323)
        self_20622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 27), 'self', False)
        # Obtaining the member 'canvas' of a type (line 323)
        canvas_20623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 27), self_20622, 'canvas')
        # Obtaining the member 'mpl_connect' of a type (line 323)
        mpl_connect_20624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 27), canvas_20623, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 323)
        mpl_connect_call_result_20629 = invoke(stypy.reporting.localization.Localization(__file__, 323, 27), mpl_connect_20624, *[str_20625, send_message_20627], **kwargs_20628)
        
        # Getting the type of 'self' (line 323)
        self_20630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'self')
        # Setting the type of the member '_idDrag' of a type (line 323)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), self_20630, '_idDrag', mpl_connect_call_result_20629)
        # SSA join for if statement (line 322)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 318)
        stypy_return_type_20631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20631)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_figure'
        return stypy_return_type_20631


    @norecursion
    def send_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'send_message'
        module_type_store = module_type_store.open_function_context('send_message', 326, 4, False)
        # Assigning a type to the variable 'self' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolCursorPosition.send_message.__dict__.__setitem__('stypy_localization', localization)
        ToolCursorPosition.send_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolCursorPosition.send_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolCursorPosition.send_message.__dict__.__setitem__('stypy_function_name', 'ToolCursorPosition.send_message')
        ToolCursorPosition.send_message.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolCursorPosition.send_message.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolCursorPosition.send_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolCursorPosition.send_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolCursorPosition.send_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolCursorPosition.send_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolCursorPosition.send_message.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolCursorPosition.send_message', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'send_message', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'send_message(...)' code ##################

        str_20632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 8), 'str', 'Call `matplotlib.backend_managers.ToolManager.message_event`')
        
        
        # Call to locked(...): (line 328)
        # Processing the call keyword arguments (line 328)
        kwargs_20637 = {}
        # Getting the type of 'self' (line 328)
        self_20633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 11), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 328)
        toolmanager_20634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 11), self_20633, 'toolmanager')
        # Obtaining the member 'messagelock' of a type (line 328)
        messagelock_20635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 11), toolmanager_20634, 'messagelock')
        # Obtaining the member 'locked' of a type (line 328)
        locked_20636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 11), messagelock_20635, 'locked')
        # Calling locked(args, kwargs) (line 328)
        locked_call_result_20638 = invoke(stypy.reporting.localization.Localization(__file__, 328, 11), locked_20636, *[], **kwargs_20637)
        
        # Testing the type of an if condition (line 328)
        if_condition_20639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 8), locked_call_result_20638)
        # Assigning a type to the variable 'if_condition_20639' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'if_condition_20639', if_condition_20639)
        # SSA begins for if statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 328)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 331):
        
        # Assigning a Str to a Name (line 331):
        
        # Assigning a Str to a Name (line 331):
        str_20640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 18), 'str', ' ')
        # Assigning a type to the variable 'message' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'message', str_20640)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'event' (line 333)
        event_20641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 11), 'event')
        # Obtaining the member 'inaxes' of a type (line 333)
        inaxes_20642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 11), event_20641, 'inaxes')
        
        # Call to get_navigate(...): (line 333)
        # Processing the call keyword arguments (line 333)
        kwargs_20646 = {}
        # Getting the type of 'event' (line 333)
        event_20643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 28), 'event', False)
        # Obtaining the member 'inaxes' of a type (line 333)
        inaxes_20644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 28), event_20643, 'inaxes')
        # Obtaining the member 'get_navigate' of a type (line 333)
        get_navigate_20645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 28), inaxes_20644, 'get_navigate')
        # Calling get_navigate(args, kwargs) (line 333)
        get_navigate_call_result_20647 = invoke(stypy.reporting.localization.Localization(__file__, 333, 28), get_navigate_20645, *[], **kwargs_20646)
        
        # Applying the binary operator 'and' (line 333)
        result_and_keyword_20648 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 11), 'and', inaxes_20642, get_navigate_call_result_20647)
        
        # Testing the type of an if condition (line 333)
        if_condition_20649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 8), result_and_keyword_20648)
        # Assigning a type to the variable 'if_condition_20649' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'if_condition_20649', if_condition_20649)
        # SSA begins for if statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to format_coord(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'event' (line 335)
        event_20653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 46), 'event', False)
        # Obtaining the member 'xdata' of a type (line 335)
        xdata_20654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 46), event_20653, 'xdata')
        # Getting the type of 'event' (line 335)
        event_20655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 59), 'event', False)
        # Obtaining the member 'ydata' of a type (line 335)
        ydata_20656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 59), event_20655, 'ydata')
        # Processing the call keyword arguments (line 335)
        kwargs_20657 = {}
        # Getting the type of 'event' (line 335)
        event_20650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 20), 'event', False)
        # Obtaining the member 'inaxes' of a type (line 335)
        inaxes_20651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 20), event_20650, 'inaxes')
        # Obtaining the member 'format_coord' of a type (line 335)
        format_coord_20652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 20), inaxes_20651, 'format_coord')
        # Calling format_coord(args, kwargs) (line 335)
        format_coord_call_result_20658 = invoke(stypy.reporting.localization.Localization(__file__, 335, 20), format_coord_20652, *[xdata_20654, ydata_20656], **kwargs_20657)
        
        # Assigning a type to the variable 's' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 's', format_coord_call_result_20658)
        # SSA branch for the except part of a try statement (line 334)
        # SSA branch for the except 'Tuple' branch of a try statement (line 334)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 334)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a ListComp to a Name (line 339):
        
        # Assigning a ListComp to a Name (line 339):
        
        # Assigning a ListComp to a Name (line 339):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'event' (line 339)
        event_20670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 38), 'event')
        # Obtaining the member 'inaxes' of a type (line 339)
        inaxes_20671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 38), event_20670, 'inaxes')
        # Obtaining the member 'mouseover_set' of a type (line 339)
        mouseover_set_20672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 38), inaxes_20671, 'mouseover_set')
        comprehension_20673 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 27), mouseover_set_20672)
        # Assigning a type to the variable 'a' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 27), 'a', comprehension_20673)
        
        # Evaluating a boolean operation
        
        # Call to contains(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'event' (line 340)
        event_20662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 41), 'event', False)
        # Processing the call keyword arguments (line 340)
        kwargs_20663 = {}
        # Getting the type of 'a' (line 340)
        a_20660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 30), 'a', False)
        # Obtaining the member 'contains' of a type (line 340)
        contains_20661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 30), a_20660, 'contains')
        # Calling contains(args, kwargs) (line 340)
        contains_call_result_20664 = invoke(stypy.reporting.localization.Localization(__file__, 340, 30), contains_20661, *[event_20662], **kwargs_20663)
        
        
        # Call to get_visible(...): (line 340)
        # Processing the call keyword arguments (line 340)
        kwargs_20667 = {}
        # Getting the type of 'a' (line 340)
        a_20665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 52), 'a', False)
        # Obtaining the member 'get_visible' of a type (line 340)
        get_visible_20666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 52), a_20665, 'get_visible')
        # Calling get_visible(args, kwargs) (line 340)
        get_visible_call_result_20668 = invoke(stypy.reporting.localization.Localization(__file__, 340, 52), get_visible_20666, *[], **kwargs_20667)
        
        # Applying the binary operator 'and' (line 340)
        result_and_keyword_20669 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 30), 'and', contains_call_result_20664, get_visible_call_result_20668)
        
        # Getting the type of 'a' (line 339)
        a_20659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 27), 'a')
        list_20674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 27), list_20674, a_20659)
        # Assigning a type to the variable 'artists' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 16), 'artists', list_20674)
        
        # Getting the type of 'artists' (line 342)
        artists_20675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 19), 'artists')
        # Testing the type of an if condition (line 342)
        if_condition_20676 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 16), artists_20675)
        # Assigning a type to the variable 'if_condition_20676' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 16), 'if_condition_20676', if_condition_20676)
        # SSA begins for if statement (line 342)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to max(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'artists' (line 343)
        artists_20678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 28), 'artists', False)
        # Processing the call keyword arguments (line 343)

        @norecursion
        def _stypy_temp_lambda_7(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_7'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_7', 343, 41, True)
            # Passed parameters checking function
            _stypy_temp_lambda_7.stypy_localization = localization
            _stypy_temp_lambda_7.stypy_type_of_self = None
            _stypy_temp_lambda_7.stypy_type_store = module_type_store
            _stypy_temp_lambda_7.stypy_function_name = '_stypy_temp_lambda_7'
            _stypy_temp_lambda_7.stypy_param_names_list = ['x']
            _stypy_temp_lambda_7.stypy_varargs_param_name = None
            _stypy_temp_lambda_7.stypy_kwargs_param_name = None
            _stypy_temp_lambda_7.stypy_call_defaults = defaults
            _stypy_temp_lambda_7.stypy_call_varargs = varargs
            _stypy_temp_lambda_7.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_7', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_7', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 343)
            x_20679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 51), 'x', False)
            # Obtaining the member 'zorder' of a type (line 343)
            zorder_20680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 51), x_20679, 'zorder')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 343)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 41), 'stypy_return_type', zorder_20680)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_7' in the type store
            # Getting the type of 'stypy_return_type' (line 343)
            stypy_return_type_20681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 41), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_20681)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_7'
            return stypy_return_type_20681

        # Assigning a type to the variable '_stypy_temp_lambda_7' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 41), '_stypy_temp_lambda_7', _stypy_temp_lambda_7)
        # Getting the type of '_stypy_temp_lambda_7' (line 343)
        _stypy_temp_lambda_7_20682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 41), '_stypy_temp_lambda_7')
        keyword_20683 = _stypy_temp_lambda_7_20682
        kwargs_20684 = {'key': keyword_20683}
        # Getting the type of 'max' (line 343)
        max_20677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'max', False)
        # Calling max(args, kwargs) (line 343)
        max_call_result_20685 = invoke(stypy.reporting.localization.Localization(__file__, 343, 24), max_20677, *[artists_20678], **kwargs_20684)
        
        # Assigning a type to the variable 'a' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 20), 'a', max_call_result_20685)
        
        
        # Getting the type of 'a' (line 344)
        a_20686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 23), 'a')
        # Getting the type of 'event' (line 344)
        event_20687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 32), 'event')
        # Obtaining the member 'inaxes' of a type (line 344)
        inaxes_20688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 32), event_20687, 'inaxes')
        # Obtaining the member 'patch' of a type (line 344)
        patch_20689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 32), inaxes_20688, 'patch')
        # Applying the binary operator 'isnot' (line 344)
        result_is_not_20690 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 23), 'isnot', a_20686, patch_20689)
        
        # Testing the type of an if condition (line 344)
        if_condition_20691 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 344, 20), result_is_not_20690)
        # Assigning a type to the variable 'if_condition_20691' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 20), 'if_condition_20691', if_condition_20691)
        # SSA begins for if statement (line 344)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 345):
        
        # Assigning a Call to a Name (line 345):
        
        # Assigning a Call to a Name (line 345):
        
        # Call to get_cursor_data(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'event' (line 345)
        event_20694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 49), 'event', False)
        # Processing the call keyword arguments (line 345)
        kwargs_20695 = {}
        # Getting the type of 'a' (line 345)
        a_20692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 31), 'a', False)
        # Obtaining the member 'get_cursor_data' of a type (line 345)
        get_cursor_data_20693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 31), a_20692, 'get_cursor_data')
        # Calling get_cursor_data(args, kwargs) (line 345)
        get_cursor_data_call_result_20696 = invoke(stypy.reporting.localization.Localization(__file__, 345, 31), get_cursor_data_20693, *[event_20694], **kwargs_20695)
        
        # Assigning a type to the variable 'data' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), 'data', get_cursor_data_call_result_20696)
        
        # Type idiom detected: calculating its left and rigth part (line 346)
        # Getting the type of 'data' (line 346)
        data_20697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 24), 'data')
        # Getting the type of 'None' (line 346)
        None_20698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 39), 'None')
        
        (may_be_20699, more_types_in_union_20700) = may_not_be_none(data_20697, None_20698)

        if may_be_20699:

            if more_types_in_union_20700:
                # Runtime conditional SSA (line 346)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 's' (line 347)
            s_20701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 28), 's')
            str_20702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 33), 'str', ' [%s]')
            
            # Call to format_cursor_data(...): (line 347)
            # Processing the call arguments (line 347)
            # Getting the type of 'data' (line 347)
            data_20705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 64), 'data', False)
            # Processing the call keyword arguments (line 347)
            kwargs_20706 = {}
            # Getting the type of 'a' (line 347)
            a_20703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 43), 'a', False)
            # Obtaining the member 'format_cursor_data' of a type (line 347)
            format_cursor_data_20704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 43), a_20703, 'format_cursor_data')
            # Calling format_cursor_data(args, kwargs) (line 347)
            format_cursor_data_call_result_20707 = invoke(stypy.reporting.localization.Localization(__file__, 347, 43), format_cursor_data_20704, *[data_20705], **kwargs_20706)
            
            # Applying the binary operator '%' (line 347)
            result_mod_20708 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 33), '%', str_20702, format_cursor_data_call_result_20707)
            
            # Applying the binary operator '+=' (line 347)
            result_iadd_20709 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 28), '+=', s_20701, result_mod_20708)
            # Assigning a type to the variable 's' (line 347)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 28), 's', result_iadd_20709)
            

            if more_types_in_union_20700:
                # SSA join for if statement (line 346)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 344)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 342)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 349):
        
        # Assigning a Name to a Name (line 349):
        
        # Assigning a Name to a Name (line 349):
        # Getting the type of 's' (line 349)
        s_20710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 26), 's')
        # Assigning a type to the variable 'message' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'message', s_20710)
        # SSA join for try-except statement (line 334)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 333)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to message_event(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'message' (line 350)
        message_20714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 39), 'message', False)
        # Getting the type of 'self' (line 350)
        self_20715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 48), 'self', False)
        # Processing the call keyword arguments (line 350)
        kwargs_20716 = {}
        # Getting the type of 'self' (line 350)
        self_20711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 350)
        toolmanager_20712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), self_20711, 'toolmanager')
        # Obtaining the member 'message_event' of a type (line 350)
        message_event_20713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), toolmanager_20712, 'message_event')
        # Calling message_event(args, kwargs) (line 350)
        message_event_call_result_20717 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), message_event_20713, *[message_20714, self_20715], **kwargs_20716)
        
        
        # ################# End of 'send_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'send_message' in the type store
        # Getting the type of 'stypy_return_type' (line 326)
        stypy_return_type_20718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20718)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'send_message'
        return stypy_return_type_20718


# Assigning a type to the variable 'ToolCursorPosition' (line 308)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 0), 'ToolCursorPosition', ToolCursorPosition)
# Declaration of the 'RubberbandBase' class
# Getting the type of 'ToolBase' (line 353)
ToolBase_20719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 21), 'ToolBase')

class RubberbandBase(ToolBase_20719, ):
    str_20720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 4), 'str', 'Draw and remove rubberband')

    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RubberbandBase.trigger.__dict__.__setitem__('stypy_localization', localization)
        RubberbandBase.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RubberbandBase.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        RubberbandBase.trigger.__dict__.__setitem__('stypy_function_name', 'RubberbandBase.trigger')
        RubberbandBase.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        RubberbandBase.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        RubberbandBase.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RubberbandBase.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        RubberbandBase.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        RubberbandBase.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RubberbandBase.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RubberbandBase.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, ['sender', 'event', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        str_20721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 8), 'str', 'Call `draw_rubberband` or `remove_rubberband` based on data')
        
        
        
        # Call to available(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'sender' (line 357)
        sender_20727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 55), 'sender', False)
        # Processing the call keyword arguments (line 357)
        kwargs_20728 = {}
        # Getting the type of 'self' (line 357)
        self_20722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 15), 'self', False)
        # Obtaining the member 'figure' of a type (line 357)
        figure_20723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 15), self_20722, 'figure')
        # Obtaining the member 'canvas' of a type (line 357)
        canvas_20724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 15), figure_20723, 'canvas')
        # Obtaining the member 'widgetlock' of a type (line 357)
        widgetlock_20725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 15), canvas_20724, 'widgetlock')
        # Obtaining the member 'available' of a type (line 357)
        available_20726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 15), widgetlock_20725, 'available')
        # Calling available(args, kwargs) (line 357)
        available_call_result_20729 = invoke(stypy.reporting.localization.Localization(__file__, 357, 15), available_20726, *[sender_20727], **kwargs_20728)
        
        # Applying the 'not' unary operator (line 357)
        result_not__20730 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 11), 'not', available_call_result_20729)
        
        # Testing the type of an if condition (line 357)
        if_condition_20731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 8), result_not__20730)
        # Assigning a type to the variable 'if_condition_20731' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'if_condition_20731', if_condition_20731)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 359)
        # Getting the type of 'data' (line 359)
        data_20732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'data')
        # Getting the type of 'None' (line 359)
        None_20733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 23), 'None')
        
        (may_be_20734, more_types_in_union_20735) = may_not_be_none(data_20732, None_20733)

        if may_be_20734:

            if more_types_in_union_20735:
                # Runtime conditional SSA (line 359)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to draw_rubberband(...): (line 360)
            # Getting the type of 'data' (line 360)
            data_20738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 34), 'data', False)
            # Processing the call keyword arguments (line 360)
            kwargs_20739 = {}
            # Getting the type of 'self' (line 360)
            self_20736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'self', False)
            # Obtaining the member 'draw_rubberband' of a type (line 360)
            draw_rubberband_20737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), self_20736, 'draw_rubberband')
            # Calling draw_rubberband(args, kwargs) (line 360)
            draw_rubberband_call_result_20740 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), draw_rubberband_20737, *[data_20738], **kwargs_20739)
            

            if more_types_in_union_20735:
                # Runtime conditional SSA for else branch (line 359)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_20734) or more_types_in_union_20735):
            
            # Call to remove_rubberband(...): (line 362)
            # Processing the call keyword arguments (line 362)
            kwargs_20743 = {}
            # Getting the type of 'self' (line 362)
            self_20741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'self', False)
            # Obtaining the member 'remove_rubberband' of a type (line 362)
            remove_rubberband_20742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), self_20741, 'remove_rubberband')
            # Calling remove_rubberband(args, kwargs) (line 362)
            remove_rubberband_call_result_20744 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), remove_rubberband_20742, *[], **kwargs_20743)
            

            if (may_be_20734 and more_types_in_union_20735):
                # SSA join for if statement (line 359)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 355)
        stypy_return_type_20745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20745)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_20745


    @norecursion
    def draw_rubberband(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_rubberband'
        module_type_store = module_type_store.open_function_context('draw_rubberband', 364, 4, False)
        # Assigning a type to the variable 'self' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RubberbandBase.draw_rubberband.__dict__.__setitem__('stypy_localization', localization)
        RubberbandBase.draw_rubberband.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RubberbandBase.draw_rubberband.__dict__.__setitem__('stypy_type_store', module_type_store)
        RubberbandBase.draw_rubberband.__dict__.__setitem__('stypy_function_name', 'RubberbandBase.draw_rubberband')
        RubberbandBase.draw_rubberband.__dict__.__setitem__('stypy_param_names_list', [])
        RubberbandBase.draw_rubberband.__dict__.__setitem__('stypy_varargs_param_name', 'data')
        RubberbandBase.draw_rubberband.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RubberbandBase.draw_rubberband.__dict__.__setitem__('stypy_call_defaults', defaults)
        RubberbandBase.draw_rubberband.__dict__.__setitem__('stypy_call_varargs', varargs)
        RubberbandBase.draw_rubberband.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RubberbandBase.draw_rubberband.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RubberbandBase.draw_rubberband', [], 'data', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_rubberband', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_rubberband(...)' code ##################

        str_20746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, (-1)), 'str', '\n        Draw rubberband\n\n        This method must get implemented per backend\n        ')
        # Getting the type of 'NotImplementedError' (line 370)
        NotImplementedError_20747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 370, 8), NotImplementedError_20747, 'raise parameter', BaseException)
        
        # ################# End of 'draw_rubberband(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_rubberband' in the type store
        # Getting the type of 'stypy_return_type' (line 364)
        stypy_return_type_20748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20748)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_rubberband'
        return stypy_return_type_20748


    @norecursion
    def remove_rubberband(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove_rubberband'
        module_type_store = module_type_store.open_function_context('remove_rubberband', 372, 4, False)
        # Assigning a type to the variable 'self' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RubberbandBase.remove_rubberband.__dict__.__setitem__('stypy_localization', localization)
        RubberbandBase.remove_rubberband.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RubberbandBase.remove_rubberband.__dict__.__setitem__('stypy_type_store', module_type_store)
        RubberbandBase.remove_rubberband.__dict__.__setitem__('stypy_function_name', 'RubberbandBase.remove_rubberband')
        RubberbandBase.remove_rubberband.__dict__.__setitem__('stypy_param_names_list', [])
        RubberbandBase.remove_rubberband.__dict__.__setitem__('stypy_varargs_param_name', None)
        RubberbandBase.remove_rubberband.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RubberbandBase.remove_rubberband.__dict__.__setitem__('stypy_call_defaults', defaults)
        RubberbandBase.remove_rubberband.__dict__.__setitem__('stypy_call_varargs', varargs)
        RubberbandBase.remove_rubberband.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RubberbandBase.remove_rubberband.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RubberbandBase.remove_rubberband', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove_rubberband', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove_rubberband(...)' code ##################

        str_20749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, (-1)), 'str', '\n        Remove rubberband\n\n        This method should get implemented per backend\n        ')
        pass
        
        # ################# End of 'remove_rubberband(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove_rubberband' in the type store
        # Getting the type of 'stypy_return_type' (line 372)
        stypy_return_type_20750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20750)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove_rubberband'
        return stypy_return_type_20750


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 353, 0, False)
        # Assigning a type to the variable 'self' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RubberbandBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'RubberbandBase' (line 353)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), 'RubberbandBase', RubberbandBase)
# Declaration of the 'ToolQuit' class
# Getting the type of 'ToolBase' (line 381)
ToolBase_20751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 15), 'ToolBase')

class ToolQuit(ToolBase_20751, ):
    str_20752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 4), 'str', 'Tool to call the figure manager destroy method')
    
    # Assigning a Str to a Name (line 384):
    
    # Assigning a Str to a Name (line 384):
    
    # Assigning a Subscript to a Name (line 385):
    
    # Assigning a Subscript to a Name (line 385):

    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 387)
        None_20753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 42), 'None')
        defaults = [None_20753]
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 387, 4, False)
        # Assigning a type to the variable 'self' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolQuit.trigger.__dict__.__setitem__('stypy_localization', localization)
        ToolQuit.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolQuit.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolQuit.trigger.__dict__.__setitem__('stypy_function_name', 'ToolQuit.trigger')
        ToolQuit.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        ToolQuit.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolQuit.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolQuit.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolQuit.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolQuit.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolQuit.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolQuit.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, ['sender', 'event', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        
        # Call to destroy_fig(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'self' (line 388)
        self_20756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 24), 'self', False)
        # Obtaining the member 'figure' of a type (line 388)
        figure_20757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 24), self_20756, 'figure')
        # Processing the call keyword arguments (line 388)
        kwargs_20758 = {}
        # Getting the type of 'Gcf' (line 388)
        Gcf_20754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'Gcf', False)
        # Obtaining the member 'destroy_fig' of a type (line 388)
        destroy_fig_20755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), Gcf_20754, 'destroy_fig')
        # Calling destroy_fig(args, kwargs) (line 388)
        destroy_fig_call_result_20759 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), destroy_fig_20755, *[figure_20757], **kwargs_20758)
        
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 387)
        stypy_return_type_20760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20760)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_20760


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 381, 0, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolQuit.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolQuit' (line 381)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 0), 'ToolQuit', ToolQuit)

# Assigning a Str to a Name (line 384):
str_20761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 18), 'str', 'Quit the figure')
# Getting the type of 'ToolQuit'
ToolQuit_20762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolQuit')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolQuit_20762, 'description', str_20761)

# Assigning a Subscript to a Name (line 385):

# Obtaining the type of the subscript
str_20763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 30), 'str', 'keymap.quit')
# Getting the type of 'rcParams' (line 385)
rcParams_20764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 385)
getitem___20765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 21), rcParams_20764, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 385)
subscript_call_result_20766 = invoke(stypy.reporting.localization.Localization(__file__, 385, 21), getitem___20765, str_20763)

# Getting the type of 'ToolQuit'
ToolQuit_20767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolQuit')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolQuit_20767, 'default_keymap', subscript_call_result_20766)
# Declaration of the 'ToolQuitAll' class
# Getting the type of 'ToolBase' (line 391)
ToolBase_20768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 18), 'ToolBase')

class ToolQuitAll(ToolBase_20768, ):
    str_20769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 4), 'str', 'Tool to call the figure manager destroy method')
    
    # Assigning a Str to a Name (line 394):
    
    # Assigning a Str to a Name (line 394):
    
    # Assigning a Subscript to a Name (line 395):
    
    # Assigning a Subscript to a Name (line 395):

    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 397)
        None_20770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 42), 'None')
        defaults = [None_20770]
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 397, 4, False)
        # Assigning a type to the variable 'self' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolQuitAll.trigger.__dict__.__setitem__('stypy_localization', localization)
        ToolQuitAll.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolQuitAll.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolQuitAll.trigger.__dict__.__setitem__('stypy_function_name', 'ToolQuitAll.trigger')
        ToolQuitAll.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        ToolQuitAll.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolQuitAll.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolQuitAll.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolQuitAll.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolQuitAll.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolQuitAll.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolQuitAll.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, ['sender', 'event', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        
        # Call to destroy_all(...): (line 398)
        # Processing the call keyword arguments (line 398)
        kwargs_20773 = {}
        # Getting the type of 'Gcf' (line 398)
        Gcf_20771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'Gcf', False)
        # Obtaining the member 'destroy_all' of a type (line 398)
        destroy_all_20772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 8), Gcf_20771, 'destroy_all')
        # Calling destroy_all(args, kwargs) (line 398)
        destroy_all_call_result_20774 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), destroy_all_20772, *[], **kwargs_20773)
        
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 397)
        stypy_return_type_20775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20775)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_20775


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 391, 0, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolQuitAll.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolQuitAll' (line 391)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), 'ToolQuitAll', ToolQuitAll)

# Assigning a Str to a Name (line 394):
str_20776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 18), 'str', 'Quit all figures')
# Getting the type of 'ToolQuitAll'
ToolQuitAll_20777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolQuitAll')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolQuitAll_20777, 'description', str_20776)

# Assigning a Subscript to a Name (line 395):

# Obtaining the type of the subscript
str_20778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 30), 'str', 'keymap.quit_all')
# Getting the type of 'rcParams' (line 395)
rcParams_20779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 395)
getitem___20780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 21), rcParams_20779, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 395)
subscript_call_result_20781 = invoke(stypy.reporting.localization.Localization(__file__, 395, 21), getitem___20780, str_20778)

# Getting the type of 'ToolQuitAll'
ToolQuitAll_20782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolQuitAll')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolQuitAll_20782, 'default_keymap', subscript_call_result_20781)
# Declaration of the 'ToolEnableAllNavigation' class
# Getting the type of 'ToolBase' (line 401)
ToolBase_20783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 30), 'ToolBase')

class ToolEnableAllNavigation(ToolBase_20783, ):
    str_20784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 4), 'str', 'Tool to enable all axes for toolmanager interaction')
    
    # Assigning a Str to a Name (line 404):
    
    # Assigning a Str to a Name (line 404):
    
    # Assigning a Subscript to a Name (line 405):
    
    # Assigning a Subscript to a Name (line 405):

    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 407)
        None_20785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 42), 'None')
        defaults = [None_20785]
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 407, 4, False)
        # Assigning a type to the variable 'self' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolEnableAllNavigation.trigger.__dict__.__setitem__('stypy_localization', localization)
        ToolEnableAllNavigation.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolEnableAllNavigation.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolEnableAllNavigation.trigger.__dict__.__setitem__('stypy_function_name', 'ToolEnableAllNavigation.trigger')
        ToolEnableAllNavigation.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        ToolEnableAllNavigation.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolEnableAllNavigation.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolEnableAllNavigation.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolEnableAllNavigation.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolEnableAllNavigation.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolEnableAllNavigation.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolEnableAllNavigation.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, ['sender', 'event', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 408)
        # Getting the type of 'event' (line 408)
        event_20786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 11), 'event')
        # Obtaining the member 'inaxes' of a type (line 408)
        inaxes_20787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 11), event_20786, 'inaxes')
        # Getting the type of 'None' (line 408)
        None_20788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 27), 'None')
        
        (may_be_20789, more_types_in_union_20790) = may_be_none(inaxes_20787, None_20788)

        if may_be_20789:

            if more_types_in_union_20790:
                # Runtime conditional SSA (line 408)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_20790:
                # SSA join for if statement (line 408)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to get_axes(...): (line 411)
        # Processing the call keyword arguments (line 411)
        kwargs_20794 = {}
        # Getting the type of 'self' (line 411)
        self_20791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 17), 'self', False)
        # Obtaining the member 'figure' of a type (line 411)
        figure_20792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 17), self_20791, 'figure')
        # Obtaining the member 'get_axes' of a type (line 411)
        get_axes_20793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 17), figure_20792, 'get_axes')
        # Calling get_axes(args, kwargs) (line 411)
        get_axes_call_result_20795 = invoke(stypy.reporting.localization.Localization(__file__, 411, 17), get_axes_20793, *[], **kwargs_20794)
        
        # Testing the type of a for loop iterable (line 411)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 411, 8), get_axes_call_result_20795)
        # Getting the type of the for loop variable (line 411)
        for_loop_var_20796 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 411, 8), get_axes_call_result_20795)
        # Assigning a type to the variable 'a' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'a', for_loop_var_20796)
        # SSA begins for a for statement (line 411)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'event' (line 412)
        event_20797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 16), 'event')
        # Obtaining the member 'x' of a type (line 412)
        x_20798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 16), event_20797, 'x')
        # Getting the type of 'None' (line 412)
        None_20799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 31), 'None')
        # Applying the binary operator 'isnot' (line 412)
        result_is_not_20800 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 16), 'isnot', x_20798, None_20799)
        
        
        # Getting the type of 'event' (line 412)
        event_20801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 40), 'event')
        # Obtaining the member 'y' of a type (line 412)
        y_20802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 40), event_20801, 'y')
        # Getting the type of 'None' (line 412)
        None_20803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 55), 'None')
        # Applying the binary operator 'isnot' (line 412)
        result_is_not_20804 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 40), 'isnot', y_20802, None_20803)
        
        # Applying the binary operator 'and' (line 412)
        result_and_keyword_20805 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 16), 'and', result_is_not_20800, result_is_not_20804)
        
        # Call to in_axes(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'event' (line 413)
        event_20808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 34), 'event', False)
        # Processing the call keyword arguments (line 413)
        kwargs_20809 = {}
        # Getting the type of 'a' (line 413)
        a_20806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 24), 'a', False)
        # Obtaining the member 'in_axes' of a type (line 413)
        in_axes_20807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 24), a_20806, 'in_axes')
        # Calling in_axes(args, kwargs) (line 413)
        in_axes_call_result_20810 = invoke(stypy.reporting.localization.Localization(__file__, 413, 24), in_axes_20807, *[event_20808], **kwargs_20809)
        
        # Applying the binary operator 'and' (line 412)
        result_and_keyword_20811 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 16), 'and', result_and_keyword_20805, in_axes_call_result_20810)
        
        # Testing the type of an if condition (line 412)
        if_condition_20812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 412, 12), result_and_keyword_20811)
        # Assigning a type to the variable 'if_condition_20812' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'if_condition_20812', if_condition_20812)
        # SSA begins for if statement (line 412)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_navigate(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'True' (line 414)
        True_20815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 31), 'True', False)
        # Processing the call keyword arguments (line 414)
        kwargs_20816 = {}
        # Getting the type of 'a' (line 414)
        a_20813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 16), 'a', False)
        # Obtaining the member 'set_navigate' of a type (line 414)
        set_navigate_20814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 16), a_20813, 'set_navigate')
        # Calling set_navigate(args, kwargs) (line 414)
        set_navigate_call_result_20817 = invoke(stypy.reporting.localization.Localization(__file__, 414, 16), set_navigate_20814, *[True_20815], **kwargs_20816)
        
        # SSA join for if statement (line 412)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 407)
        stypy_return_type_20818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20818)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_20818


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 401, 0, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolEnableAllNavigation.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolEnableAllNavigation' (line 401)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 0), 'ToolEnableAllNavigation', ToolEnableAllNavigation)

# Assigning a Str to a Name (line 404):
str_20819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 18), 'str', 'Enables all axes toolmanager')
# Getting the type of 'ToolEnableAllNavigation'
ToolEnableAllNavigation_20820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolEnableAllNavigation')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolEnableAllNavigation_20820, 'description', str_20819)

# Assigning a Subscript to a Name (line 405):

# Obtaining the type of the subscript
str_20821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 30), 'str', 'keymap.all_axes')
# Getting the type of 'rcParams' (line 405)
rcParams_20822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 405)
getitem___20823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 21), rcParams_20822, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 405)
subscript_call_result_20824 = invoke(stypy.reporting.localization.Localization(__file__, 405, 21), getitem___20823, str_20821)

# Getting the type of 'ToolEnableAllNavigation'
ToolEnableAllNavigation_20825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolEnableAllNavigation')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolEnableAllNavigation_20825, 'default_keymap', subscript_call_result_20824)
# Declaration of the 'ToolEnableNavigation' class
# Getting the type of 'ToolBase' (line 417)
ToolBase_20826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 27), 'ToolBase')

class ToolEnableNavigation(ToolBase_20826, ):
    str_20827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 4), 'str', 'Tool to enable a specific axes for toolmanager interaction')
    
    # Assigning a Str to a Name (line 420):
    
    # Assigning a Str to a Name (line 420):
    
    # Assigning a Tuple to a Name (line 421):
    
    # Assigning a Tuple to a Name (line 421):

    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 423)
        None_20828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 42), 'None')
        defaults = [None_20828]
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 423, 4, False)
        # Assigning a type to the variable 'self' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolEnableNavigation.trigger.__dict__.__setitem__('stypy_localization', localization)
        ToolEnableNavigation.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolEnableNavigation.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolEnableNavigation.trigger.__dict__.__setitem__('stypy_function_name', 'ToolEnableNavigation.trigger')
        ToolEnableNavigation.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        ToolEnableNavigation.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolEnableNavigation.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolEnableNavigation.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolEnableNavigation.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolEnableNavigation.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolEnableNavigation.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolEnableNavigation.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, ['sender', 'event', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 424)
        # Getting the type of 'event' (line 424)
        event_20829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 11), 'event')
        # Obtaining the member 'inaxes' of a type (line 424)
        inaxes_20830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 11), event_20829, 'inaxes')
        # Getting the type of 'None' (line 424)
        None_20831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 27), 'None')
        
        (may_be_20832, more_types_in_union_20833) = may_be_none(inaxes_20830, None_20831)

        if may_be_20832:

            if more_types_in_union_20833:
                # Runtime conditional SSA (line 424)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 425)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_20833:
                # SSA join for if statement (line 424)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 427):
        
        # Assigning a BinOp to a Name (line 427):
        
        # Assigning a BinOp to a Name (line 427):
        
        # Call to int(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'event' (line 427)
        event_20835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'event', False)
        # Obtaining the member 'key' of a type (line 427)
        key_20836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 16), event_20835, 'key')
        # Processing the call keyword arguments (line 427)
        kwargs_20837 = {}
        # Getting the type of 'int' (line 427)
        int_20834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'int', False)
        # Calling int(args, kwargs) (line 427)
        int_call_result_20838 = invoke(stypy.reporting.localization.Localization(__file__, 427, 12), int_20834, *[key_20836], **kwargs_20837)
        
        int_20839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 29), 'int')
        # Applying the binary operator '-' (line 427)
        result_sub_20840 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 12), '-', int_call_result_20838, int_20839)
        
        # Assigning a type to the variable 'n' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'n', result_sub_20840)
        
        
        # Call to enumerate(...): (line 428)
        # Processing the call arguments (line 428)
        
        # Call to get_axes(...): (line 428)
        # Processing the call keyword arguments (line 428)
        kwargs_20845 = {}
        # Getting the type of 'self' (line 428)
        self_20842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 30), 'self', False)
        # Obtaining the member 'figure' of a type (line 428)
        figure_20843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 30), self_20842, 'figure')
        # Obtaining the member 'get_axes' of a type (line 428)
        get_axes_20844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 30), figure_20843, 'get_axes')
        # Calling get_axes(args, kwargs) (line 428)
        get_axes_call_result_20846 = invoke(stypy.reporting.localization.Localization(__file__, 428, 30), get_axes_20844, *[], **kwargs_20845)
        
        # Processing the call keyword arguments (line 428)
        kwargs_20847 = {}
        # Getting the type of 'enumerate' (line 428)
        enumerate_20841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 20), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 428)
        enumerate_call_result_20848 = invoke(stypy.reporting.localization.Localization(__file__, 428, 20), enumerate_20841, *[get_axes_call_result_20846], **kwargs_20847)
        
        # Testing the type of a for loop iterable (line 428)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 428, 8), enumerate_call_result_20848)
        # Getting the type of the for loop variable (line 428)
        for_loop_var_20849 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 428, 8), enumerate_call_result_20848)
        # Assigning a type to the variable 'i' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 8), for_loop_var_20849))
        # Assigning a type to the variable 'a' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 8), for_loop_var_20849))
        # SSA begins for a for statement (line 428)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'event' (line 429)
        event_20850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 16), 'event')
        # Obtaining the member 'x' of a type (line 429)
        x_20851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 16), event_20850, 'x')
        # Getting the type of 'None' (line 429)
        None_20852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 31), 'None')
        # Applying the binary operator 'isnot' (line 429)
        result_is_not_20853 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 16), 'isnot', x_20851, None_20852)
        
        
        # Getting the type of 'event' (line 429)
        event_20854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 40), 'event')
        # Obtaining the member 'y' of a type (line 429)
        y_20855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 40), event_20854, 'y')
        # Getting the type of 'None' (line 429)
        None_20856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 55), 'None')
        # Applying the binary operator 'isnot' (line 429)
        result_is_not_20857 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 40), 'isnot', y_20855, None_20856)
        
        # Applying the binary operator 'and' (line 429)
        result_and_keyword_20858 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 16), 'and', result_is_not_20853, result_is_not_20857)
        
        # Call to in_axes(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 'event' (line 430)
        event_20861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 34), 'event', False)
        # Processing the call keyword arguments (line 430)
        kwargs_20862 = {}
        # Getting the type of 'a' (line 430)
        a_20859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 24), 'a', False)
        # Obtaining the member 'in_axes' of a type (line 430)
        in_axes_20860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 24), a_20859, 'in_axes')
        # Calling in_axes(args, kwargs) (line 430)
        in_axes_call_result_20863 = invoke(stypy.reporting.localization.Localization(__file__, 430, 24), in_axes_20860, *[event_20861], **kwargs_20862)
        
        # Applying the binary operator 'and' (line 429)
        result_and_keyword_20864 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 16), 'and', result_and_keyword_20858, in_axes_call_result_20863)
        
        # Testing the type of an if condition (line 429)
        if_condition_20865 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 429, 12), result_and_keyword_20864)
        # Assigning a type to the variable 'if_condition_20865' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'if_condition_20865', if_condition_20865)
        # SSA begins for if statement (line 429)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_navigate(...): (line 431)
        # Processing the call arguments (line 431)
        
        # Getting the type of 'i' (line 431)
        i_20868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 31), 'i', False)
        # Getting the type of 'n' (line 431)
        n_20869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 36), 'n', False)
        # Applying the binary operator '==' (line 431)
        result_eq_20870 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 31), '==', i_20868, n_20869)
        
        # Processing the call keyword arguments (line 431)
        kwargs_20871 = {}
        # Getting the type of 'a' (line 431)
        a_20866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 16), 'a', False)
        # Obtaining the member 'set_navigate' of a type (line 431)
        set_navigate_20867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 16), a_20866, 'set_navigate')
        # Calling set_navigate(args, kwargs) (line 431)
        set_navigate_call_result_20872 = invoke(stypy.reporting.localization.Localization(__file__, 431, 16), set_navigate_20867, *[result_eq_20870], **kwargs_20871)
        
        # SSA join for if statement (line 429)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 423)
        stypy_return_type_20873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20873)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_20873


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 417, 0, False)
        # Assigning a type to the variable 'self' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolEnableNavigation.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolEnableNavigation' (line 417)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 0), 'ToolEnableNavigation', ToolEnableNavigation)

# Assigning a Str to a Name (line 420):
str_20874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 18), 'str', 'Enables one axes toolmanager')
# Getting the type of 'ToolEnableNavigation'
ToolEnableNavigation_20875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolEnableNavigation')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolEnableNavigation_20875, 'description', str_20874)

# Assigning a Tuple to a Name (line 421):

# Obtaining an instance of the builtin type 'tuple' (line 421)
tuple_20876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 421)
# Adding element type (line 421)
int_20877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 22), tuple_20876, int_20877)
# Adding element type (line 421)
int_20878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 22), tuple_20876, int_20878)
# Adding element type (line 421)
int_20879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 22), tuple_20876, int_20879)
# Adding element type (line 421)
int_20880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 22), tuple_20876, int_20880)
# Adding element type (line 421)
int_20881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 22), tuple_20876, int_20881)
# Adding element type (line 421)
int_20882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 22), tuple_20876, int_20882)
# Adding element type (line 421)
int_20883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 22), tuple_20876, int_20883)
# Adding element type (line 421)
int_20884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 22), tuple_20876, int_20884)
# Adding element type (line 421)
int_20885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 22), tuple_20876, int_20885)

# Getting the type of 'ToolEnableNavigation'
ToolEnableNavigation_20886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolEnableNavigation')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolEnableNavigation_20886, 'default_keymap', tuple_20876)
# Declaration of the '_ToolGridBase' class
# Getting the type of 'ToolBase' (line 434)
ToolBase_20887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 20), 'ToolBase')

class _ToolGridBase(ToolBase_20887, ):
    str_20888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 4), 'str', 'Common functionality between ToolGrid and ToolMinorGrid.')
    
    # Assigning a List to a Name (line 437):
    
    # Assigning a List to a Name (line 437):

    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 439)
        None_20889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 42), 'None')
        defaults = [None_20889]
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 439, 4, False)
        # Assigning a type to the variable 'self' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ToolGridBase.trigger.__dict__.__setitem__('stypy_localization', localization)
        _ToolGridBase.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ToolGridBase.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ToolGridBase.trigger.__dict__.__setitem__('stypy_function_name', '_ToolGridBase.trigger')
        _ToolGridBase.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        _ToolGridBase.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ToolGridBase.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ToolGridBase.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ToolGridBase.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ToolGridBase.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ToolGridBase.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ToolGridBase.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, ['sender', 'event', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        
        # Assigning a Attribute to a Name (line 440):
        
        # Assigning a Attribute to a Name (line 440):
        
        # Assigning a Attribute to a Name (line 440):
        # Getting the type of 'event' (line 440)
        event_20890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 13), 'event')
        # Obtaining the member 'inaxes' of a type (line 440)
        inaxes_20891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 13), event_20890, 'inaxes')
        # Assigning a type to the variable 'ax' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'ax', inaxes_20891)
        
        # Type idiom detected: calculating its left and rigth part (line 441)
        # Getting the type of 'ax' (line 441)
        ax_20892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 11), 'ax')
        # Getting the type of 'None' (line 441)
        None_20893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 17), 'None')
        
        (may_be_20894, more_types_in_union_20895) = may_be_none(ax_20892, None_20893)

        if may_be_20894:

            if more_types_in_union_20895:
                # Runtime conditional SSA (line 441)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 442)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_20895:
                # SSA join for if statement (line 441)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # SSA begins for try-except statement (line 443)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Tuple (line 444):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to _get_next_grid_states(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'ax' (line 444)
        ax_20898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 76), 'ax', False)
        # Processing the call keyword arguments (line 444)
        kwargs_20899 = {}
        # Getting the type of 'self' (line 444)
        self_20896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 49), 'self', False)
        # Obtaining the member '_get_next_grid_states' of a type (line 444)
        _get_next_grid_states_20897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 49), self_20896, '_get_next_grid_states')
        # Calling _get_next_grid_states(args, kwargs) (line 444)
        _get_next_grid_states_call_result_20900 = invoke(stypy.reporting.localization.Localization(__file__, 444, 49), _get_next_grid_states_20897, *[ax_20898], **kwargs_20899)
        
        # Assigning a type to the variable 'call_assignment_20152' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20152', _get_next_grid_states_call_result_20900)
        
        # Assigning a Call to a Name (line 444):
        
        # Assigning a Call to a Name (line 444):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_20903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 12), 'int')
        # Processing the call keyword arguments
        kwargs_20904 = {}
        # Getting the type of 'call_assignment_20152' (line 444)
        call_assignment_20152_20901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20152', False)
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___20902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), call_assignment_20152_20901, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_20905 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20902, *[int_20903], **kwargs_20904)
        
        # Assigning a type to the variable 'call_assignment_20153' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20153', getitem___call_result_20905)
        
        # Assigning a Name to a Name (line 444):
        
        # Assigning a Name to a Name (line 444):
        # Getting the type of 'call_assignment_20153' (line 444)
        call_assignment_20153_20906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20153')
        # Assigning a type to the variable 'x_state' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'x_state', call_assignment_20153_20906)
        
        # Assigning a Call to a Name (line 444):
        
        # Assigning a Call to a Name (line 444):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_20909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 12), 'int')
        # Processing the call keyword arguments
        kwargs_20910 = {}
        # Getting the type of 'call_assignment_20152' (line 444)
        call_assignment_20152_20907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20152', False)
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___20908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), call_assignment_20152_20907, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_20911 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20908, *[int_20909], **kwargs_20910)
        
        # Assigning a type to the variable 'call_assignment_20154' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20154', getitem___call_result_20911)
        
        # Assigning a Name to a Name (line 444):
        
        # Assigning a Name to a Name (line 444):
        # Getting the type of 'call_assignment_20154' (line 444)
        call_assignment_20154_20912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20154')
        # Assigning a type to the variable 'x_which' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 21), 'x_which', call_assignment_20154_20912)
        
        # Assigning a Call to a Name (line 444):
        
        # Assigning a Call to a Name (line 444):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_20915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 12), 'int')
        # Processing the call keyword arguments
        kwargs_20916 = {}
        # Getting the type of 'call_assignment_20152' (line 444)
        call_assignment_20152_20913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20152', False)
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___20914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), call_assignment_20152_20913, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_20917 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20914, *[int_20915], **kwargs_20916)
        
        # Assigning a type to the variable 'call_assignment_20155' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20155', getitem___call_result_20917)
        
        # Assigning a Name to a Name (line 444):
        
        # Assigning a Name to a Name (line 444):
        # Getting the type of 'call_assignment_20155' (line 444)
        call_assignment_20155_20918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20155')
        # Assigning a type to the variable 'y_state' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 30), 'y_state', call_assignment_20155_20918)
        
        # Assigning a Call to a Name (line 444):
        
        # Assigning a Call to a Name (line 444):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_20921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 12), 'int')
        # Processing the call keyword arguments
        kwargs_20922 = {}
        # Getting the type of 'call_assignment_20152' (line 444)
        call_assignment_20152_20919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20152', False)
        # Obtaining the member '__getitem__' of a type (line 444)
        getitem___20920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), call_assignment_20152_20919, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_20923 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___20920, *[int_20921], **kwargs_20922)
        
        # Assigning a type to the variable 'call_assignment_20156' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20156', getitem___call_result_20923)
        
        # Assigning a Name to a Name (line 444):
        
        # Assigning a Name to a Name (line 444):
        # Getting the type of 'call_assignment_20156' (line 444)
        call_assignment_20156_20924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'call_assignment_20156')
        # Assigning a type to the variable 'y_which' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 39), 'y_which', call_assignment_20156_20924)
        # SSA branch for the except part of a try statement (line 443)
        # SSA branch for the except 'ValueError' branch of a try statement (line 443)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 443)
        module_type_store.open_ssa_branch('except else')
        
        # Call to grid(...): (line 448)
        # Processing the call arguments (line 448)
        # Getting the type of 'x_state' (line 448)
        x_state_20927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 20), 'x_state', False)
        # Processing the call keyword arguments (line 448)
        # Getting the type of 'x_which' (line 448)
        x_which_20928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 35), 'x_which', False)
        keyword_20929 = x_which_20928
        str_20930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 49), 'str', 'x')
        keyword_20931 = str_20930
        kwargs_20932 = {'which': keyword_20929, 'axis': keyword_20931}
        # Getting the type of 'ax' (line 448)
        ax_20925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'ax', False)
        # Obtaining the member 'grid' of a type (line 448)
        grid_20926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 12), ax_20925, 'grid')
        # Calling grid(args, kwargs) (line 448)
        grid_call_result_20933 = invoke(stypy.reporting.localization.Localization(__file__, 448, 12), grid_20926, *[x_state_20927], **kwargs_20932)
        
        
        # Call to grid(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'y_state' (line 449)
        y_state_20936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 20), 'y_state', False)
        # Processing the call keyword arguments (line 449)
        # Getting the type of 'y_which' (line 449)
        y_which_20937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 35), 'y_which', False)
        keyword_20938 = y_which_20937
        str_20939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 49), 'str', 'y')
        keyword_20940 = str_20939
        kwargs_20941 = {'which': keyword_20938, 'axis': keyword_20940}
        # Getting the type of 'ax' (line 449)
        ax_20934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'ax', False)
        # Obtaining the member 'grid' of a type (line 449)
        grid_20935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 12), ax_20934, 'grid')
        # Calling grid(args, kwargs) (line 449)
        grid_call_result_20942 = invoke(stypy.reporting.localization.Localization(__file__, 449, 12), grid_20935, *[y_state_20936], **kwargs_20941)
        
        
        # Call to draw_idle(...): (line 450)
        # Processing the call keyword arguments (line 450)
        kwargs_20947 = {}
        # Getting the type of 'ax' (line 450)
        ax_20943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'ax', False)
        # Obtaining the member 'figure' of a type (line 450)
        figure_20944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 12), ax_20943, 'figure')
        # Obtaining the member 'canvas' of a type (line 450)
        canvas_20945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 12), figure_20944, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 450)
        draw_idle_20946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 12), canvas_20945, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 450)
        draw_idle_call_result_20948 = invoke(stypy.reporting.localization.Localization(__file__, 450, 12), draw_idle_20946, *[], **kwargs_20947)
        
        # SSA join for try-except statement (line 443)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 439)
        stypy_return_type_20949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20949)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_20949


    @staticmethod
    @norecursion
    def _get_uniform_grid_state(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_uniform_grid_state'
        module_type_store = module_type_store.open_function_context('_get_uniform_grid_state', 452, 4, False)
        
        # Passed parameters checking function
        _ToolGridBase._get_uniform_grid_state.__dict__.__setitem__('stypy_localization', localization)
        _ToolGridBase._get_uniform_grid_state.__dict__.__setitem__('stypy_type_of_self', None)
        _ToolGridBase._get_uniform_grid_state.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ToolGridBase._get_uniform_grid_state.__dict__.__setitem__('stypy_function_name', '_get_uniform_grid_state')
        _ToolGridBase._get_uniform_grid_state.__dict__.__setitem__('stypy_param_names_list', ['ticks'])
        _ToolGridBase._get_uniform_grid_state.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ToolGridBase._get_uniform_grid_state.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ToolGridBase._get_uniform_grid_state.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ToolGridBase._get_uniform_grid_state.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ToolGridBase._get_uniform_grid_state.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ToolGridBase._get_uniform_grid_state.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '_get_uniform_grid_state', ['ticks'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_uniform_grid_state', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_uniform_grid_state(...)' code ##################

        str_20950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, (-1)), 'str', '\n        Check whether all grid lines are in the same visibility state.\n\n        Returns True/False if all grid lines are on or off, None if they are\n        not all in the same state.\n        ')
        
        
        # Call to all(...): (line 460)
        # Processing the call arguments (line 460)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 460, 15, True)
        # Calculating comprehension expression
        # Getting the type of 'ticks' (line 460)
        ticks_20954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 39), 'ticks', False)
        comprehension_20955 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 15), ticks_20954)
        # Assigning a type to the variable 'tick' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 15), 'tick', comprehension_20955)
        # Getting the type of 'tick' (line 460)
        tick_20952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 15), 'tick', False)
        # Obtaining the member 'gridOn' of a type (line 460)
        gridOn_20953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 15), tick_20952, 'gridOn')
        list_20956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 15), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 15), list_20956, gridOn_20953)
        # Processing the call keyword arguments (line 460)
        kwargs_20957 = {}
        # Getting the type of 'all' (line 460)
        all_20951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 11), 'all', False)
        # Calling all(args, kwargs) (line 460)
        all_call_result_20958 = invoke(stypy.reporting.localization.Localization(__file__, 460, 11), all_20951, *[list_20956], **kwargs_20957)
        
        # Testing the type of an if condition (line 460)
        if_condition_20959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 8), all_call_result_20958)
        # Assigning a type to the variable 'if_condition_20959' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'if_condition_20959', if_condition_20959)
        # SSA begins for if statement (line 460)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 461)
        True_20960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'stypy_return_type', True_20960)
        # SSA branch for the else part of an if statement (line 460)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to any(...): (line 462)
        # Processing the call arguments (line 462)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 462, 21, True)
        # Calculating comprehension expression
        # Getting the type of 'ticks' (line 462)
        ticks_20964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 45), 'ticks', False)
        comprehension_20965 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 21), ticks_20964)
        # Assigning a type to the variable 'tick' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 21), 'tick', comprehension_20965)
        # Getting the type of 'tick' (line 462)
        tick_20962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 21), 'tick', False)
        # Obtaining the member 'gridOn' of a type (line 462)
        gridOn_20963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 21), tick_20962, 'gridOn')
        list_20966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 21), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 21), list_20966, gridOn_20963)
        # Processing the call keyword arguments (line 462)
        kwargs_20967 = {}
        # Getting the type of 'any' (line 462)
        any_20961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 17), 'any', False)
        # Calling any(args, kwargs) (line 462)
        any_call_result_20968 = invoke(stypy.reporting.localization.Localization(__file__, 462, 17), any_20961, *[list_20966], **kwargs_20967)
        
        # Applying the 'not' unary operator (line 462)
        result_not__20969 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 13), 'not', any_call_result_20968)
        
        # Testing the type of an if condition (line 462)
        if_condition_20970 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 13), result_not__20969)
        # Assigning a type to the variable 'if_condition_20970' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 13), 'if_condition_20970', if_condition_20970)
        # SSA begins for if statement (line 462)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 463)
        False_20971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'stypy_return_type', False_20971)
        # SSA branch for the else part of an if statement (line 462)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'None' (line 465)
        None_20972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'stypy_return_type', None_20972)
        # SSA join for if statement (line 462)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 460)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_get_uniform_grid_state(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_uniform_grid_state' in the type store
        # Getting the type of 'stypy_return_type' (line 452)
        stypy_return_type_20973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20973)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_uniform_grid_state'
        return stypy_return_type_20973


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 434, 0, False)
        # Assigning a type to the variable 'self' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ToolGridBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_ToolGridBase' (line 434)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 0), '_ToolGridBase', _ToolGridBase)

# Assigning a List to a Name (line 437):

# Obtaining an instance of the builtin type 'list' (line 437)
list_20974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 437)
# Adding element type (line 437)

# Obtaining an instance of the builtin type 'tuple' (line 437)
tuple_20975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 437)
# Adding element type (line 437)
# Getting the type of 'False' (line 437)
False_20976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 15), tuple_20975, False_20976)
# Adding element type (line 437)
# Getting the type of 'False' (line 437)
False_20977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 22), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 15), tuple_20975, False_20977)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 13), list_20974, tuple_20975)
# Adding element type (line 437)

# Obtaining an instance of the builtin type 'tuple' (line 437)
tuple_20978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 437)
# Adding element type (line 437)
# Getting the type of 'True' (line 437)
True_20979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 31), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 31), tuple_20978, True_20979)
# Adding element type (line 437)
# Getting the type of 'False' (line 437)
False_20980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 37), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 31), tuple_20978, False_20980)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 13), list_20974, tuple_20978)
# Adding element type (line 437)

# Obtaining an instance of the builtin type 'tuple' (line 437)
tuple_20981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 437)
# Adding element type (line 437)
# Getting the type of 'True' (line 437)
True_20982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 46), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 46), tuple_20981, True_20982)
# Adding element type (line 437)
# Getting the type of 'True' (line 437)
True_20983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 52), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 46), tuple_20981, True_20983)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 13), list_20974, tuple_20981)
# Adding element type (line 437)

# Obtaining an instance of the builtin type 'tuple' (line 437)
tuple_20984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 60), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 437)
# Adding element type (line 437)
# Getting the type of 'False' (line 437)
False_20985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 60), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 60), tuple_20984, False_20985)
# Adding element type (line 437)
# Getting the type of 'True' (line 437)
True_20986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 67), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 60), tuple_20984, True_20986)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 13), list_20974, tuple_20984)

# Getting the type of '_ToolGridBase'
_ToolGridBase_20987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_ToolGridBase')
# Setting the type of the member '_cycle' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _ToolGridBase_20987, '_cycle', list_20974)
# Declaration of the 'ToolGrid' class
# Getting the type of '_ToolGridBase' (line 468)
_ToolGridBase_20988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 15), '_ToolGridBase')

class ToolGrid(_ToolGridBase_20988, ):
    str_20989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 4), 'str', 'Tool to toggle the major grids of the figure')
    
    # Assigning a Str to a Name (line 471):
    
    # Assigning a Str to a Name (line 471):
    
    # Assigning a Subscript to a Name (line 472):
    
    # Assigning a Subscript to a Name (line 472):

    @norecursion
    def _get_next_grid_states(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_next_grid_states'
        module_type_store = module_type_store.open_function_context('_get_next_grid_states', 474, 4, False)
        # Assigning a type to the variable 'self' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolGrid._get_next_grid_states.__dict__.__setitem__('stypy_localization', localization)
        ToolGrid._get_next_grid_states.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolGrid._get_next_grid_states.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolGrid._get_next_grid_states.__dict__.__setitem__('stypy_function_name', 'ToolGrid._get_next_grid_states')
        ToolGrid._get_next_grid_states.__dict__.__setitem__('stypy_param_names_list', ['ax'])
        ToolGrid._get_next_grid_states.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolGrid._get_next_grid_states.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolGrid._get_next_grid_states.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolGrid._get_next_grid_states.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolGrid._get_next_grid_states.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolGrid._get_next_grid_states.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolGrid._get_next_grid_states', ['ax'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_next_grid_states', localization, ['ax'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_next_grid_states(...)' code ##################

        
        
        # Getting the type of 'None' (line 475)
        None_20990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 11), 'None')
        
        # Call to map(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'self' (line 475)
        self_20992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 23), 'self', False)
        # Obtaining the member '_get_uniform_grid_state' of a type (line 475)
        _get_uniform_grid_state_20993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 23), self_20992, '_get_uniform_grid_state')
        
        # Obtaining an instance of the builtin type 'list' (line 476)
        list_20994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 476)
        # Adding element type (line 476)
        # Getting the type of 'ax' (line 476)
        ax_20995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 24), 'ax', False)
        # Obtaining the member 'xaxis' of a type (line 476)
        xaxis_20996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 24), ax_20995, 'xaxis')
        # Obtaining the member 'minorTicks' of a type (line 476)
        minorTicks_20997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 24), xaxis_20996, 'minorTicks')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 23), list_20994, minorTicks_20997)
        # Adding element type (line 476)
        # Getting the type of 'ax' (line 476)
        ax_20998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 45), 'ax', False)
        # Obtaining the member 'yaxis' of a type (line 476)
        yaxis_20999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 45), ax_20998, 'yaxis')
        # Obtaining the member 'minorTicks' of a type (line 476)
        minorTicks_21000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 45), yaxis_20999, 'minorTicks')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 23), list_20994, minorTicks_21000)
        
        # Processing the call keyword arguments (line 475)
        kwargs_21001 = {}
        # Getting the type of 'map' (line 475)
        map_20991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'map', False)
        # Calling map(args, kwargs) (line 475)
        map_call_result_21002 = invoke(stypy.reporting.localization.Localization(__file__, 475, 19), map_20991, *[_get_uniform_grid_state_20993, list_20994], **kwargs_21001)
        
        # Applying the binary operator 'in' (line 475)
        result_contains_21003 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 11), 'in', None_20990, map_call_result_21002)
        
        # Testing the type of an if condition (line 475)
        if_condition_21004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 475, 8), result_contains_21003)
        # Assigning a type to the variable 'if_condition_21004' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'if_condition_21004', if_condition_21004)
        # SSA begins for if statement (line 475)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'ValueError' (line 478)
        ValueError_21005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 18), 'ValueError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 478, 12), ValueError_21005, 'raise parameter', BaseException)
        # SSA join for if statement (line 475)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 479):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to map(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'self' (line 479)
        self_21007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 31), 'self', False)
        # Obtaining the member '_get_uniform_grid_state' of a type (line 479)
        _get_uniform_grid_state_21008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 31), self_21007, '_get_uniform_grid_state')
        
        # Obtaining an instance of the builtin type 'list' (line 480)
        list_21009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 480)
        # Adding element type (line 480)
        # Getting the type of 'ax' (line 480)
        ax_21010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 32), 'ax', False)
        # Obtaining the member 'xaxis' of a type (line 480)
        xaxis_21011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 32), ax_21010, 'xaxis')
        # Obtaining the member 'majorTicks' of a type (line 480)
        majorTicks_21012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 32), xaxis_21011, 'majorTicks')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 31), list_21009, majorTicks_21012)
        # Adding element type (line 480)
        # Getting the type of 'ax' (line 480)
        ax_21013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 53), 'ax', False)
        # Obtaining the member 'yaxis' of a type (line 480)
        yaxis_21014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 53), ax_21013, 'yaxis')
        # Obtaining the member 'majorTicks' of a type (line 480)
        majorTicks_21015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 53), yaxis_21014, 'majorTicks')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 31), list_21009, majorTicks_21015)
        
        # Processing the call keyword arguments (line 479)
        kwargs_21016 = {}
        # Getting the type of 'map' (line 479)
        map_21006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 27), 'map', False)
        # Calling map(args, kwargs) (line 479)
        map_call_result_21017 = invoke(stypy.reporting.localization.Localization(__file__, 479, 27), map_21006, *[_get_uniform_grid_state_21008, list_21009], **kwargs_21016)
        
        # Assigning a type to the variable 'call_assignment_20157' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'call_assignment_20157', map_call_result_21017)
        
        # Assigning a Call to a Name (line 479):
        
        # Assigning a Call to a Name (line 479):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_21020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 8), 'int')
        # Processing the call keyword arguments
        kwargs_21021 = {}
        # Getting the type of 'call_assignment_20157' (line 479)
        call_assignment_20157_21018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'call_assignment_20157', False)
        # Obtaining the member '__getitem__' of a type (line 479)
        getitem___21019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 8), call_assignment_20157_21018, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_21022 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___21019, *[int_21020], **kwargs_21021)
        
        # Assigning a type to the variable 'call_assignment_20158' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'call_assignment_20158', getitem___call_result_21022)
        
        # Assigning a Name to a Name (line 479):
        
        # Assigning a Name to a Name (line 479):
        # Getting the type of 'call_assignment_20158' (line 479)
        call_assignment_20158_21023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'call_assignment_20158')
        # Assigning a type to the variable 'x_state' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'x_state', call_assignment_20158_21023)
        
        # Assigning a Call to a Name (line 479):
        
        # Assigning a Call to a Name (line 479):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_21026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 8), 'int')
        # Processing the call keyword arguments
        kwargs_21027 = {}
        # Getting the type of 'call_assignment_20157' (line 479)
        call_assignment_20157_21024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'call_assignment_20157', False)
        # Obtaining the member '__getitem__' of a type (line 479)
        getitem___21025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 8), call_assignment_20157_21024, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_21028 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___21025, *[int_21026], **kwargs_21027)
        
        # Assigning a type to the variable 'call_assignment_20159' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'call_assignment_20159', getitem___call_result_21028)
        
        # Assigning a Name to a Name (line 479):
        
        # Assigning a Name to a Name (line 479):
        # Getting the type of 'call_assignment_20159' (line 479)
        call_assignment_20159_21029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'call_assignment_20159')
        # Assigning a type to the variable 'y_state' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 17), 'y_state', call_assignment_20159_21029)
        
        # Assigning a Attribute to a Name (line 481):
        
        # Assigning a Attribute to a Name (line 481):
        
        # Assigning a Attribute to a Name (line 481):
        # Getting the type of 'self' (line 481)
        self_21030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 16), 'self')
        # Obtaining the member '_cycle' of a type (line 481)
        _cycle_21031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 16), self_21030, '_cycle')
        # Assigning a type to the variable 'cycle' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'cycle', _cycle_21031)
        
        # Assigning a Subscript to a Tuple (line 483):
        
        # Assigning a Subscript to a Name (line 483):
        
        # Assigning a Subscript to a Name (line 483):
        
        # Obtaining the type of the subscript
        int_21032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 8), 'int')
        
        # Obtaining the type of the subscript
        
        # Call to index(...): (line 484)
        # Processing the call arguments (line 484)
        
        # Obtaining an instance of the builtin type 'tuple' (line 484)
        tuple_21035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 484)
        # Adding element type (line 484)
        # Getting the type of 'x_state' (line 484)
        x_state_21036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 32), 'x_state', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 32), tuple_21035, x_state_21036)
        # Adding element type (line 484)
        # Getting the type of 'y_state' (line 484)
        y_state_21037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 41), 'y_state', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 32), tuple_21035, y_state_21037)
        
        # Processing the call keyword arguments (line 484)
        kwargs_21038 = {}
        # Getting the type of 'cycle' (line 484)
        cycle_21033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 19), 'cycle', False)
        # Obtaining the member 'index' of a type (line 484)
        index_21034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 19), cycle_21033, 'index')
        # Calling index(args, kwargs) (line 484)
        index_call_result_21039 = invoke(stypy.reporting.localization.Localization(__file__, 484, 19), index_21034, *[tuple_21035], **kwargs_21038)
        
        int_21040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 53), 'int')
        # Applying the binary operator '+' (line 484)
        result_add_21041 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 19), '+', index_call_result_21039, int_21040)
        
        
        # Call to len(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'cycle' (line 484)
        cycle_21043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 62), 'cycle', False)
        # Processing the call keyword arguments (line 484)
        kwargs_21044 = {}
        # Getting the type of 'len' (line 484)
        len_21042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 58), 'len', False)
        # Calling len(args, kwargs) (line 484)
        len_call_result_21045 = invoke(stypy.reporting.localization.Localization(__file__, 484, 58), len_21042, *[cycle_21043], **kwargs_21044)
        
        # Applying the binary operator '%' (line 484)
        result_mod_21046 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 18), '%', result_add_21041, len_call_result_21045)
        
        # Getting the type of 'cycle' (line 484)
        cycle_21047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'cycle')
        # Obtaining the member '__getitem__' of a type (line 484)
        getitem___21048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 12), cycle_21047, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 484)
        subscript_call_result_21049 = invoke(stypy.reporting.localization.Localization(__file__, 484, 12), getitem___21048, result_mod_21046)
        
        # Obtaining the member '__getitem__' of a type (line 483)
        getitem___21050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), subscript_call_result_21049, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 483)
        subscript_call_result_21051 = invoke(stypy.reporting.localization.Localization(__file__, 483, 8), getitem___21050, int_21032)
        
        # Assigning a type to the variable 'tuple_var_assignment_20160' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'tuple_var_assignment_20160', subscript_call_result_21051)
        
        # Assigning a Subscript to a Name (line 483):
        
        # Assigning a Subscript to a Name (line 483):
        
        # Obtaining the type of the subscript
        int_21052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 8), 'int')
        
        # Obtaining the type of the subscript
        
        # Call to index(...): (line 484)
        # Processing the call arguments (line 484)
        
        # Obtaining an instance of the builtin type 'tuple' (line 484)
        tuple_21055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 484)
        # Adding element type (line 484)
        # Getting the type of 'x_state' (line 484)
        x_state_21056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 32), 'x_state', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 32), tuple_21055, x_state_21056)
        # Adding element type (line 484)
        # Getting the type of 'y_state' (line 484)
        y_state_21057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 41), 'y_state', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 32), tuple_21055, y_state_21057)
        
        # Processing the call keyword arguments (line 484)
        kwargs_21058 = {}
        # Getting the type of 'cycle' (line 484)
        cycle_21053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 19), 'cycle', False)
        # Obtaining the member 'index' of a type (line 484)
        index_21054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 19), cycle_21053, 'index')
        # Calling index(args, kwargs) (line 484)
        index_call_result_21059 = invoke(stypy.reporting.localization.Localization(__file__, 484, 19), index_21054, *[tuple_21055], **kwargs_21058)
        
        int_21060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 53), 'int')
        # Applying the binary operator '+' (line 484)
        result_add_21061 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 19), '+', index_call_result_21059, int_21060)
        
        
        # Call to len(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'cycle' (line 484)
        cycle_21063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 62), 'cycle', False)
        # Processing the call keyword arguments (line 484)
        kwargs_21064 = {}
        # Getting the type of 'len' (line 484)
        len_21062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 58), 'len', False)
        # Calling len(args, kwargs) (line 484)
        len_call_result_21065 = invoke(stypy.reporting.localization.Localization(__file__, 484, 58), len_21062, *[cycle_21063], **kwargs_21064)
        
        # Applying the binary operator '%' (line 484)
        result_mod_21066 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 18), '%', result_add_21061, len_call_result_21065)
        
        # Getting the type of 'cycle' (line 484)
        cycle_21067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'cycle')
        # Obtaining the member '__getitem__' of a type (line 484)
        getitem___21068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 12), cycle_21067, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 484)
        subscript_call_result_21069 = invoke(stypy.reporting.localization.Localization(__file__, 484, 12), getitem___21068, result_mod_21066)
        
        # Obtaining the member '__getitem__' of a type (line 483)
        getitem___21070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 8), subscript_call_result_21069, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 483)
        subscript_call_result_21071 = invoke(stypy.reporting.localization.Localization(__file__, 483, 8), getitem___21070, int_21052)
        
        # Assigning a type to the variable 'tuple_var_assignment_20161' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'tuple_var_assignment_20161', subscript_call_result_21071)
        
        # Assigning a Name to a Name (line 483):
        
        # Assigning a Name to a Name (line 483):
        # Getting the type of 'tuple_var_assignment_20160' (line 483)
        tuple_var_assignment_20160_21072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'tuple_var_assignment_20160')
        # Assigning a type to the variable 'x_state' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'x_state', tuple_var_assignment_20160_21072)
        
        # Assigning a Name to a Name (line 483):
        
        # Assigning a Name to a Name (line 483):
        # Getting the type of 'tuple_var_assignment_20161' (line 483)
        tuple_var_assignment_20161_21073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'tuple_var_assignment_20161')
        # Assigning a type to the variable 'y_state' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 17), 'y_state', tuple_var_assignment_20161_21073)
        
        # Obtaining an instance of the builtin type 'tuple' (line 485)
        tuple_21074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 485)
        # Adding element type (line 485)
        # Getting the type of 'x_state' (line 485)
        x_state_21075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 16), 'x_state')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 16), tuple_21074, x_state_21075)
        # Adding element type (line 485)
        
        # Getting the type of 'x_state' (line 485)
        x_state_21076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 36), 'x_state')
        # Testing the type of an if expression (line 485)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 485, 25), x_state_21076)
        # SSA begins for if expression (line 485)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        str_21077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 25), 'str', 'major')
        # SSA branch for the else part of an if expression (line 485)
        module_type_store.open_ssa_branch('if expression else')
        str_21078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 49), 'str', 'both')
        # SSA join for if expression (line 485)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_21079 = union_type.UnionType.add(str_21077, str_21078)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 16), tuple_21074, if_exp_21079)
        # Adding element type (line 485)
        # Getting the type of 'y_state' (line 486)
        y_state_21080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 16), 'y_state')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 16), tuple_21074, y_state_21080)
        # Adding element type (line 485)
        
        # Getting the type of 'y_state' (line 486)
        y_state_21081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 36), 'y_state')
        # Testing the type of an if expression (line 486)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 486, 25), y_state_21081)
        # SSA begins for if expression (line 486)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        str_21082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 25), 'str', 'major')
        # SSA branch for the else part of an if expression (line 486)
        module_type_store.open_ssa_branch('if expression else')
        str_21083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 49), 'str', 'both')
        # SSA join for if expression (line 486)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_21084 = union_type.UnionType.add(str_21082, str_21083)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 16), tuple_21074, if_exp_21084)
        
        # Assigning a type to the variable 'stypy_return_type' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'stypy_return_type', tuple_21074)
        
        # ################# End of '_get_next_grid_states(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_next_grid_states' in the type store
        # Getting the type of 'stypy_return_type' (line 474)
        stypy_return_type_21085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21085)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_next_grid_states'
        return stypy_return_type_21085


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 468, 0, False)
        # Assigning a type to the variable 'self' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolGrid.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolGrid' (line 468)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 0), 'ToolGrid', ToolGrid)

# Assigning a Str to a Name (line 471):
str_21086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 18), 'str', 'Toogle major grids')
# Getting the type of 'ToolGrid'
ToolGrid_21087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolGrid')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolGrid_21087, 'description', str_21086)

# Assigning a Subscript to a Name (line 472):

# Obtaining the type of the subscript
str_21088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 30), 'str', 'keymap.grid')
# Getting the type of 'rcParams' (line 472)
rcParams_21089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 472)
getitem___21090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 21), rcParams_21089, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 472)
subscript_call_result_21091 = invoke(stypy.reporting.localization.Localization(__file__, 472, 21), getitem___21090, str_21088)

# Getting the type of 'ToolGrid'
ToolGrid_21092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolGrid')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolGrid_21092, 'default_keymap', subscript_call_result_21091)
# Declaration of the 'ToolMinorGrid' class
# Getting the type of '_ToolGridBase' (line 489)
_ToolGridBase_21093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 20), '_ToolGridBase')

class ToolMinorGrid(_ToolGridBase_21093, ):
    str_21094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 4), 'str', 'Tool to toggle the major and minor grids of the figure')
    
    # Assigning a Str to a Name (line 492):
    
    # Assigning a Str to a Name (line 492):
    
    # Assigning a Subscript to a Name (line 493):
    
    # Assigning a Subscript to a Name (line 493):

    @norecursion
    def _get_next_grid_states(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_next_grid_states'
        module_type_store = module_type_store.open_function_context('_get_next_grid_states', 495, 4, False)
        # Assigning a type to the variable 'self' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolMinorGrid._get_next_grid_states.__dict__.__setitem__('stypy_localization', localization)
        ToolMinorGrid._get_next_grid_states.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolMinorGrid._get_next_grid_states.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolMinorGrid._get_next_grid_states.__dict__.__setitem__('stypy_function_name', 'ToolMinorGrid._get_next_grid_states')
        ToolMinorGrid._get_next_grid_states.__dict__.__setitem__('stypy_param_names_list', ['ax'])
        ToolMinorGrid._get_next_grid_states.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolMinorGrid._get_next_grid_states.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolMinorGrid._get_next_grid_states.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolMinorGrid._get_next_grid_states.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolMinorGrid._get_next_grid_states.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolMinorGrid._get_next_grid_states.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolMinorGrid._get_next_grid_states', ['ax'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_next_grid_states', localization, ['ax'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_next_grid_states(...)' code ##################

        
        
        # Getting the type of 'None' (line 496)
        None_21095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 11), 'None')
        
        # Call to map(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'self' (line 496)
        self_21097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 23), 'self', False)
        # Obtaining the member '_get_uniform_grid_state' of a type (line 496)
        _get_uniform_grid_state_21098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 23), self_21097, '_get_uniform_grid_state')
        
        # Obtaining an instance of the builtin type 'list' (line 497)
        list_21099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 497)
        # Adding element type (line 497)
        # Getting the type of 'ax' (line 497)
        ax_21100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 24), 'ax', False)
        # Obtaining the member 'xaxis' of a type (line 497)
        xaxis_21101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 24), ax_21100, 'xaxis')
        # Obtaining the member 'majorTicks' of a type (line 497)
        majorTicks_21102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 24), xaxis_21101, 'majorTicks')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 23), list_21099, majorTicks_21102)
        # Adding element type (line 497)
        # Getting the type of 'ax' (line 497)
        ax_21103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 45), 'ax', False)
        # Obtaining the member 'yaxis' of a type (line 497)
        yaxis_21104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 45), ax_21103, 'yaxis')
        # Obtaining the member 'majorTicks' of a type (line 497)
        majorTicks_21105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 45), yaxis_21104, 'majorTicks')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 23), list_21099, majorTicks_21105)
        
        # Processing the call keyword arguments (line 496)
        kwargs_21106 = {}
        # Getting the type of 'map' (line 496)
        map_21096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 19), 'map', False)
        # Calling map(args, kwargs) (line 496)
        map_call_result_21107 = invoke(stypy.reporting.localization.Localization(__file__, 496, 19), map_21096, *[_get_uniform_grid_state_21098, list_21099], **kwargs_21106)
        
        # Applying the binary operator 'in' (line 496)
        result_contains_21108 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 11), 'in', None_21095, map_call_result_21107)
        
        # Testing the type of an if condition (line 496)
        if_condition_21109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 496, 8), result_contains_21108)
        # Assigning a type to the variable 'if_condition_21109' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'if_condition_21109', if_condition_21109)
        # SSA begins for if statement (line 496)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'ValueError' (line 499)
        ValueError_21110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 18), 'ValueError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 499, 12), ValueError_21110, 'raise parameter', BaseException)
        # SSA join for if statement (line 496)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 500):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to map(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'self' (line 500)
        self_21112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 31), 'self', False)
        # Obtaining the member '_get_uniform_grid_state' of a type (line 500)
        _get_uniform_grid_state_21113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 31), self_21112, '_get_uniform_grid_state')
        
        # Obtaining an instance of the builtin type 'list' (line 501)
        list_21114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 501)
        # Adding element type (line 501)
        # Getting the type of 'ax' (line 501)
        ax_21115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 32), 'ax', False)
        # Obtaining the member 'xaxis' of a type (line 501)
        xaxis_21116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 32), ax_21115, 'xaxis')
        # Obtaining the member 'minorTicks' of a type (line 501)
        minorTicks_21117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 32), xaxis_21116, 'minorTicks')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 31), list_21114, minorTicks_21117)
        # Adding element type (line 501)
        # Getting the type of 'ax' (line 501)
        ax_21118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 53), 'ax', False)
        # Obtaining the member 'yaxis' of a type (line 501)
        yaxis_21119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 53), ax_21118, 'yaxis')
        # Obtaining the member 'minorTicks' of a type (line 501)
        minorTicks_21120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 53), yaxis_21119, 'minorTicks')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 31), list_21114, minorTicks_21120)
        
        # Processing the call keyword arguments (line 500)
        kwargs_21121 = {}
        # Getting the type of 'map' (line 500)
        map_21111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 27), 'map', False)
        # Calling map(args, kwargs) (line 500)
        map_call_result_21122 = invoke(stypy.reporting.localization.Localization(__file__, 500, 27), map_21111, *[_get_uniform_grid_state_21113, list_21114], **kwargs_21121)
        
        # Assigning a type to the variable 'call_assignment_20162' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'call_assignment_20162', map_call_result_21122)
        
        # Assigning a Call to a Name (line 500):
        
        # Assigning a Call to a Name (line 500):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_21125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 8), 'int')
        # Processing the call keyword arguments
        kwargs_21126 = {}
        # Getting the type of 'call_assignment_20162' (line 500)
        call_assignment_20162_21123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'call_assignment_20162', False)
        # Obtaining the member '__getitem__' of a type (line 500)
        getitem___21124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), call_assignment_20162_21123, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_21127 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___21124, *[int_21125], **kwargs_21126)
        
        # Assigning a type to the variable 'call_assignment_20163' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'call_assignment_20163', getitem___call_result_21127)
        
        # Assigning a Name to a Name (line 500):
        
        # Assigning a Name to a Name (line 500):
        # Getting the type of 'call_assignment_20163' (line 500)
        call_assignment_20163_21128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'call_assignment_20163')
        # Assigning a type to the variable 'x_state' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'x_state', call_assignment_20163_21128)
        
        # Assigning a Call to a Name (line 500):
        
        # Assigning a Call to a Name (line 500):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_21131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 8), 'int')
        # Processing the call keyword arguments
        kwargs_21132 = {}
        # Getting the type of 'call_assignment_20162' (line 500)
        call_assignment_20162_21129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'call_assignment_20162', False)
        # Obtaining the member '__getitem__' of a type (line 500)
        getitem___21130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), call_assignment_20162_21129, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_21133 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___21130, *[int_21131], **kwargs_21132)
        
        # Assigning a type to the variable 'call_assignment_20164' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'call_assignment_20164', getitem___call_result_21133)
        
        # Assigning a Name to a Name (line 500):
        
        # Assigning a Name to a Name (line 500):
        # Getting the type of 'call_assignment_20164' (line 500)
        call_assignment_20164_21134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'call_assignment_20164')
        # Assigning a type to the variable 'y_state' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 17), 'y_state', call_assignment_20164_21134)
        
        # Assigning a Attribute to a Name (line 502):
        
        # Assigning a Attribute to a Name (line 502):
        
        # Assigning a Attribute to a Name (line 502):
        # Getting the type of 'self' (line 502)
        self_21135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 16), 'self')
        # Obtaining the member '_cycle' of a type (line 502)
        _cycle_21136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 16), self_21135, '_cycle')
        # Assigning a type to the variable 'cycle' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'cycle', _cycle_21136)
        
        # Assigning a Subscript to a Tuple (line 504):
        
        # Assigning a Subscript to a Name (line 504):
        
        # Assigning a Subscript to a Name (line 504):
        
        # Obtaining the type of the subscript
        int_21137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 8), 'int')
        
        # Obtaining the type of the subscript
        
        # Call to index(...): (line 505)
        # Processing the call arguments (line 505)
        
        # Obtaining an instance of the builtin type 'tuple' (line 505)
        tuple_21140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 505)
        # Adding element type (line 505)
        # Getting the type of 'x_state' (line 505)
        x_state_21141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 32), 'x_state', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 32), tuple_21140, x_state_21141)
        # Adding element type (line 505)
        # Getting the type of 'y_state' (line 505)
        y_state_21142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 41), 'y_state', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 32), tuple_21140, y_state_21142)
        
        # Processing the call keyword arguments (line 505)
        kwargs_21143 = {}
        # Getting the type of 'cycle' (line 505)
        cycle_21138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 19), 'cycle', False)
        # Obtaining the member 'index' of a type (line 505)
        index_21139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 19), cycle_21138, 'index')
        # Calling index(args, kwargs) (line 505)
        index_call_result_21144 = invoke(stypy.reporting.localization.Localization(__file__, 505, 19), index_21139, *[tuple_21140], **kwargs_21143)
        
        int_21145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 53), 'int')
        # Applying the binary operator '+' (line 505)
        result_add_21146 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 19), '+', index_call_result_21144, int_21145)
        
        
        # Call to len(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'cycle' (line 505)
        cycle_21148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 62), 'cycle', False)
        # Processing the call keyword arguments (line 505)
        kwargs_21149 = {}
        # Getting the type of 'len' (line 505)
        len_21147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 58), 'len', False)
        # Calling len(args, kwargs) (line 505)
        len_call_result_21150 = invoke(stypy.reporting.localization.Localization(__file__, 505, 58), len_21147, *[cycle_21148], **kwargs_21149)
        
        # Applying the binary operator '%' (line 505)
        result_mod_21151 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 18), '%', result_add_21146, len_call_result_21150)
        
        # Getting the type of 'cycle' (line 505)
        cycle_21152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'cycle')
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___21153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 12), cycle_21152, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_21154 = invoke(stypy.reporting.localization.Localization(__file__, 505, 12), getitem___21153, result_mod_21151)
        
        # Obtaining the member '__getitem__' of a type (line 504)
        getitem___21155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 8), subscript_call_result_21154, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 504)
        subscript_call_result_21156 = invoke(stypy.reporting.localization.Localization(__file__, 504, 8), getitem___21155, int_21137)
        
        # Assigning a type to the variable 'tuple_var_assignment_20165' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'tuple_var_assignment_20165', subscript_call_result_21156)
        
        # Assigning a Subscript to a Name (line 504):
        
        # Assigning a Subscript to a Name (line 504):
        
        # Obtaining the type of the subscript
        int_21157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 8), 'int')
        
        # Obtaining the type of the subscript
        
        # Call to index(...): (line 505)
        # Processing the call arguments (line 505)
        
        # Obtaining an instance of the builtin type 'tuple' (line 505)
        tuple_21160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 505)
        # Adding element type (line 505)
        # Getting the type of 'x_state' (line 505)
        x_state_21161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 32), 'x_state', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 32), tuple_21160, x_state_21161)
        # Adding element type (line 505)
        # Getting the type of 'y_state' (line 505)
        y_state_21162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 41), 'y_state', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 32), tuple_21160, y_state_21162)
        
        # Processing the call keyword arguments (line 505)
        kwargs_21163 = {}
        # Getting the type of 'cycle' (line 505)
        cycle_21158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 19), 'cycle', False)
        # Obtaining the member 'index' of a type (line 505)
        index_21159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 19), cycle_21158, 'index')
        # Calling index(args, kwargs) (line 505)
        index_call_result_21164 = invoke(stypy.reporting.localization.Localization(__file__, 505, 19), index_21159, *[tuple_21160], **kwargs_21163)
        
        int_21165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 53), 'int')
        # Applying the binary operator '+' (line 505)
        result_add_21166 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 19), '+', index_call_result_21164, int_21165)
        
        
        # Call to len(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'cycle' (line 505)
        cycle_21168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 62), 'cycle', False)
        # Processing the call keyword arguments (line 505)
        kwargs_21169 = {}
        # Getting the type of 'len' (line 505)
        len_21167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 58), 'len', False)
        # Calling len(args, kwargs) (line 505)
        len_call_result_21170 = invoke(stypy.reporting.localization.Localization(__file__, 505, 58), len_21167, *[cycle_21168], **kwargs_21169)
        
        # Applying the binary operator '%' (line 505)
        result_mod_21171 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 18), '%', result_add_21166, len_call_result_21170)
        
        # Getting the type of 'cycle' (line 505)
        cycle_21172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'cycle')
        # Obtaining the member '__getitem__' of a type (line 505)
        getitem___21173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 12), cycle_21172, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 505)
        subscript_call_result_21174 = invoke(stypy.reporting.localization.Localization(__file__, 505, 12), getitem___21173, result_mod_21171)
        
        # Obtaining the member '__getitem__' of a type (line 504)
        getitem___21175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 8), subscript_call_result_21174, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 504)
        subscript_call_result_21176 = invoke(stypy.reporting.localization.Localization(__file__, 504, 8), getitem___21175, int_21157)
        
        # Assigning a type to the variable 'tuple_var_assignment_20166' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'tuple_var_assignment_20166', subscript_call_result_21176)
        
        # Assigning a Name to a Name (line 504):
        
        # Assigning a Name to a Name (line 504):
        # Getting the type of 'tuple_var_assignment_20165' (line 504)
        tuple_var_assignment_20165_21177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'tuple_var_assignment_20165')
        # Assigning a type to the variable 'x_state' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'x_state', tuple_var_assignment_20165_21177)
        
        # Assigning a Name to a Name (line 504):
        
        # Assigning a Name to a Name (line 504):
        # Getting the type of 'tuple_var_assignment_20166' (line 504)
        tuple_var_assignment_20166_21178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'tuple_var_assignment_20166')
        # Assigning a type to the variable 'y_state' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 17), 'y_state', tuple_var_assignment_20166_21178)
        
        # Obtaining an instance of the builtin type 'tuple' (line 506)
        tuple_21179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 506)
        # Adding element type (line 506)
        # Getting the type of 'x_state' (line 506)
        x_state_21180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 15), 'x_state')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 15), tuple_21179, x_state_21180)
        # Adding element type (line 506)
        str_21181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 24), 'str', 'both')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 15), tuple_21179, str_21181)
        # Adding element type (line 506)
        # Getting the type of 'y_state' (line 506)
        y_state_21182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 32), 'y_state')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 15), tuple_21179, y_state_21182)
        # Adding element type (line 506)
        str_21183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 41), 'str', 'both')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 15), tuple_21179, str_21183)
        
        # Assigning a type to the variable 'stypy_return_type' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'stypy_return_type', tuple_21179)
        
        # ################# End of '_get_next_grid_states(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_next_grid_states' in the type store
        # Getting the type of 'stypy_return_type' (line 495)
        stypy_return_type_21184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21184)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_next_grid_states'
        return stypy_return_type_21184


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 489, 0, False)
        # Assigning a type to the variable 'self' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolMinorGrid.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolMinorGrid' (line 489)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 0), 'ToolMinorGrid', ToolMinorGrid)

# Assigning a Str to a Name (line 492):
str_21185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 18), 'str', 'Toogle major and minor grids')
# Getting the type of 'ToolMinorGrid'
ToolMinorGrid_21186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolMinorGrid')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolMinorGrid_21186, 'description', str_21185)

# Assigning a Subscript to a Name (line 493):

# Obtaining the type of the subscript
str_21187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 30), 'str', 'keymap.grid_minor')
# Getting the type of 'rcParams' (line 493)
rcParams_21188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 493)
getitem___21189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 21), rcParams_21188, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 493)
subscript_call_result_21190 = invoke(stypy.reporting.localization.Localization(__file__, 493, 21), getitem___21189, str_21187)

# Getting the type of 'ToolMinorGrid'
ToolMinorGrid_21191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolMinorGrid')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolMinorGrid_21191, 'default_keymap', subscript_call_result_21190)
# Declaration of the 'ToolFullScreen' class
# Getting the type of 'ToolToggleBase' (line 509)
ToolToggleBase_21192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 21), 'ToolToggleBase')

class ToolFullScreen(ToolToggleBase_21192, ):
    str_21193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 4), 'str', 'Tool to toggle full screen')
    
    # Assigning a Str to a Name (line 512):
    
    # Assigning a Str to a Name (line 512):
    
    # Assigning a Subscript to a Name (line 513):
    
    # Assigning a Subscript to a Name (line 513):

    @norecursion
    def enable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'enable'
        module_type_store = module_type_store.open_function_context('enable', 515, 4, False)
        # Assigning a type to the variable 'self' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolFullScreen.enable.__dict__.__setitem__('stypy_localization', localization)
        ToolFullScreen.enable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolFullScreen.enable.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolFullScreen.enable.__dict__.__setitem__('stypy_function_name', 'ToolFullScreen.enable')
        ToolFullScreen.enable.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolFullScreen.enable.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolFullScreen.enable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolFullScreen.enable.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolFullScreen.enable.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolFullScreen.enable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolFullScreen.enable.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolFullScreen.enable', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'enable', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'enable(...)' code ##################

        
        # Call to full_screen_toggle(...): (line 516)
        # Processing the call keyword arguments (line 516)
        kwargs_21199 = {}
        # Getting the type of 'self' (line 516)
        self_21194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 516)
        figure_21195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), self_21194, 'figure')
        # Obtaining the member 'canvas' of a type (line 516)
        canvas_21196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), figure_21195, 'canvas')
        # Obtaining the member 'manager' of a type (line 516)
        manager_21197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), canvas_21196, 'manager')
        # Obtaining the member 'full_screen_toggle' of a type (line 516)
        full_screen_toggle_21198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), manager_21197, 'full_screen_toggle')
        # Calling full_screen_toggle(args, kwargs) (line 516)
        full_screen_toggle_call_result_21200 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), full_screen_toggle_21198, *[], **kwargs_21199)
        
        
        # ################# End of 'enable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'enable' in the type store
        # Getting the type of 'stypy_return_type' (line 515)
        stypy_return_type_21201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21201)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'enable'
        return stypy_return_type_21201


    @norecursion
    def disable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'disable'
        module_type_store = module_type_store.open_function_context('disable', 518, 4, False)
        # Assigning a type to the variable 'self' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolFullScreen.disable.__dict__.__setitem__('stypy_localization', localization)
        ToolFullScreen.disable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolFullScreen.disable.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolFullScreen.disable.__dict__.__setitem__('stypy_function_name', 'ToolFullScreen.disable')
        ToolFullScreen.disable.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolFullScreen.disable.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolFullScreen.disable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolFullScreen.disable.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolFullScreen.disable.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolFullScreen.disable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolFullScreen.disable.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolFullScreen.disable', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'disable', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'disable(...)' code ##################

        
        # Call to full_screen_toggle(...): (line 519)
        # Processing the call keyword arguments (line 519)
        kwargs_21207 = {}
        # Getting the type of 'self' (line 519)
        self_21202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 519)
        figure_21203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), self_21202, 'figure')
        # Obtaining the member 'canvas' of a type (line 519)
        canvas_21204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), figure_21203, 'canvas')
        # Obtaining the member 'manager' of a type (line 519)
        manager_21205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), canvas_21204, 'manager')
        # Obtaining the member 'full_screen_toggle' of a type (line 519)
        full_screen_toggle_21206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), manager_21205, 'full_screen_toggle')
        # Calling full_screen_toggle(args, kwargs) (line 519)
        full_screen_toggle_call_result_21208 = invoke(stypy.reporting.localization.Localization(__file__, 519, 8), full_screen_toggle_21206, *[], **kwargs_21207)
        
        
        # ################# End of 'disable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'disable' in the type store
        # Getting the type of 'stypy_return_type' (line 518)
        stypy_return_type_21209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21209)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'disable'
        return stypy_return_type_21209


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 509, 0, False)
        # Assigning a type to the variable 'self' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolFullScreen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolFullScreen' (line 509)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 0), 'ToolFullScreen', ToolFullScreen)

# Assigning a Str to a Name (line 512):
str_21210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 18), 'str', 'Toogle Fullscreen mode')
# Getting the type of 'ToolFullScreen'
ToolFullScreen_21211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolFullScreen')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolFullScreen_21211, 'description', str_21210)

# Assigning a Subscript to a Name (line 513):

# Obtaining the type of the subscript
str_21212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 30), 'str', 'keymap.fullscreen')
# Getting the type of 'rcParams' (line 513)
rcParams_21213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 513)
getitem___21214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 21), rcParams_21213, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 513)
subscript_call_result_21215 = invoke(stypy.reporting.localization.Localization(__file__, 513, 21), getitem___21214, str_21212)

# Getting the type of 'ToolFullScreen'
ToolFullScreen_21216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolFullScreen')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolFullScreen_21216, 'default_keymap', subscript_call_result_21215)
# Declaration of the 'AxisScaleBase' class
# Getting the type of 'ToolToggleBase' (line 522)
ToolToggleBase_21217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 20), 'ToolToggleBase')

class AxisScaleBase(ToolToggleBase_21217, ):
    str_21218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 4), 'str', 'Base Tool to toggle between linear and logarithmic')

    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 525)
        None_21219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 42), 'None')
        defaults = [None_21219]
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 525, 4, False)
        # Assigning a type to the variable 'self' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AxisScaleBase.trigger.__dict__.__setitem__('stypy_localization', localization)
        AxisScaleBase.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AxisScaleBase.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        AxisScaleBase.trigger.__dict__.__setitem__('stypy_function_name', 'AxisScaleBase.trigger')
        AxisScaleBase.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        AxisScaleBase.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        AxisScaleBase.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AxisScaleBase.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        AxisScaleBase.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        AxisScaleBase.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AxisScaleBase.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AxisScaleBase.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, ['sender', 'event', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 526)
        # Getting the type of 'event' (line 526)
        event_21220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 11), 'event')
        # Obtaining the member 'inaxes' of a type (line 526)
        inaxes_21221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 11), event_21220, 'inaxes')
        # Getting the type of 'None' (line 526)
        None_21222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 27), 'None')
        
        (may_be_21223, more_types_in_union_21224) = may_be_none(inaxes_21221, None_21222)

        if may_be_21223:

            if more_types_in_union_21224:
                # Runtime conditional SSA (line 526)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 527)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_21224:
                # SSA join for if statement (line 526)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to trigger(...): (line 528)
        # Processing the call arguments (line 528)
        # Getting the type of 'self' (line 528)
        self_21227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 31), 'self', False)
        # Getting the type of 'sender' (line 528)
        sender_21228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 37), 'sender', False)
        # Getting the type of 'event' (line 528)
        event_21229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 45), 'event', False)
        # Getting the type of 'data' (line 528)
        data_21230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 52), 'data', False)
        # Processing the call keyword arguments (line 528)
        kwargs_21231 = {}
        # Getting the type of 'ToolToggleBase' (line 528)
        ToolToggleBase_21225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'ToolToggleBase', False)
        # Obtaining the member 'trigger' of a type (line 528)
        trigger_21226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 8), ToolToggleBase_21225, 'trigger')
        # Calling trigger(args, kwargs) (line 528)
        trigger_call_result_21232 = invoke(stypy.reporting.localization.Localization(__file__, 528, 8), trigger_21226, *[self_21227, sender_21228, event_21229, data_21230], **kwargs_21231)
        
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 525)
        stypy_return_type_21233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21233)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_21233


    @norecursion
    def enable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'enable'
        module_type_store = module_type_store.open_function_context('enable', 530, 4, False)
        # Assigning a type to the variable 'self' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AxisScaleBase.enable.__dict__.__setitem__('stypy_localization', localization)
        AxisScaleBase.enable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AxisScaleBase.enable.__dict__.__setitem__('stypy_type_store', module_type_store)
        AxisScaleBase.enable.__dict__.__setitem__('stypy_function_name', 'AxisScaleBase.enable')
        AxisScaleBase.enable.__dict__.__setitem__('stypy_param_names_list', ['event'])
        AxisScaleBase.enable.__dict__.__setitem__('stypy_varargs_param_name', None)
        AxisScaleBase.enable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AxisScaleBase.enable.__dict__.__setitem__('stypy_call_defaults', defaults)
        AxisScaleBase.enable.__dict__.__setitem__('stypy_call_varargs', varargs)
        AxisScaleBase.enable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AxisScaleBase.enable.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AxisScaleBase.enable', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'enable', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'enable(...)' code ##################

        
        # Call to set_scale(...): (line 531)
        # Processing the call arguments (line 531)
        # Getting the type of 'event' (line 531)
        event_21236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 23), 'event', False)
        # Obtaining the member 'inaxes' of a type (line 531)
        inaxes_21237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 23), event_21236, 'inaxes')
        str_21238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 37), 'str', 'log')
        # Processing the call keyword arguments (line 531)
        kwargs_21239 = {}
        # Getting the type of 'self' (line 531)
        self_21234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'self', False)
        # Obtaining the member 'set_scale' of a type (line 531)
        set_scale_21235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 8), self_21234, 'set_scale')
        # Calling set_scale(args, kwargs) (line 531)
        set_scale_call_result_21240 = invoke(stypy.reporting.localization.Localization(__file__, 531, 8), set_scale_21235, *[inaxes_21237, str_21238], **kwargs_21239)
        
        
        # Call to draw_idle(...): (line 532)
        # Processing the call keyword arguments (line 532)
        kwargs_21245 = {}
        # Getting the type of 'self' (line 532)
        self_21241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 532)
        figure_21242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 8), self_21241, 'figure')
        # Obtaining the member 'canvas' of a type (line 532)
        canvas_21243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 8), figure_21242, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 532)
        draw_idle_21244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 8), canvas_21243, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 532)
        draw_idle_call_result_21246 = invoke(stypy.reporting.localization.Localization(__file__, 532, 8), draw_idle_21244, *[], **kwargs_21245)
        
        
        # ################# End of 'enable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'enable' in the type store
        # Getting the type of 'stypy_return_type' (line 530)
        stypy_return_type_21247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21247)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'enable'
        return stypy_return_type_21247


    @norecursion
    def disable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'disable'
        module_type_store = module_type_store.open_function_context('disable', 534, 4, False)
        # Assigning a type to the variable 'self' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        AxisScaleBase.disable.__dict__.__setitem__('stypy_localization', localization)
        AxisScaleBase.disable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        AxisScaleBase.disable.__dict__.__setitem__('stypy_type_store', module_type_store)
        AxisScaleBase.disable.__dict__.__setitem__('stypy_function_name', 'AxisScaleBase.disable')
        AxisScaleBase.disable.__dict__.__setitem__('stypy_param_names_list', ['event'])
        AxisScaleBase.disable.__dict__.__setitem__('stypy_varargs_param_name', None)
        AxisScaleBase.disable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        AxisScaleBase.disable.__dict__.__setitem__('stypy_call_defaults', defaults)
        AxisScaleBase.disable.__dict__.__setitem__('stypy_call_varargs', varargs)
        AxisScaleBase.disable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        AxisScaleBase.disable.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AxisScaleBase.disable', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'disable', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'disable(...)' code ##################

        
        # Call to set_scale(...): (line 535)
        # Processing the call arguments (line 535)
        # Getting the type of 'event' (line 535)
        event_21250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 23), 'event', False)
        # Obtaining the member 'inaxes' of a type (line 535)
        inaxes_21251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 23), event_21250, 'inaxes')
        str_21252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 37), 'str', 'linear')
        # Processing the call keyword arguments (line 535)
        kwargs_21253 = {}
        # Getting the type of 'self' (line 535)
        self_21248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'self', False)
        # Obtaining the member 'set_scale' of a type (line 535)
        set_scale_21249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 8), self_21248, 'set_scale')
        # Calling set_scale(args, kwargs) (line 535)
        set_scale_call_result_21254 = invoke(stypy.reporting.localization.Localization(__file__, 535, 8), set_scale_21249, *[inaxes_21251, str_21252], **kwargs_21253)
        
        
        # Call to draw_idle(...): (line 536)
        # Processing the call keyword arguments (line 536)
        kwargs_21259 = {}
        # Getting the type of 'self' (line 536)
        self_21255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 536)
        figure_21256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), self_21255, 'figure')
        # Obtaining the member 'canvas' of a type (line 536)
        canvas_21257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), figure_21256, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 536)
        draw_idle_21258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), canvas_21257, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 536)
        draw_idle_call_result_21260 = invoke(stypy.reporting.localization.Localization(__file__, 536, 8), draw_idle_21258, *[], **kwargs_21259)
        
        
        # ################# End of 'disable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'disable' in the type store
        # Getting the type of 'stypy_return_type' (line 534)
        stypy_return_type_21261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'disable'
        return stypy_return_type_21261


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 522, 0, False)
        # Assigning a type to the variable 'self' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'AxisScaleBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'AxisScaleBase' (line 522)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 0), 'AxisScaleBase', AxisScaleBase)
# Declaration of the 'ToolYScale' class
# Getting the type of 'AxisScaleBase' (line 539)
AxisScaleBase_21262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 17), 'AxisScaleBase')

class ToolYScale(AxisScaleBase_21262, ):
    str_21263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 4), 'str', 'Tool to toggle between linear and logarithmic scales on the Y axis')
    
    # Assigning a Str to a Name (line 542):
    
    # Assigning a Str to a Name (line 542):
    
    # Assigning a Subscript to a Name (line 543):
    
    # Assigning a Subscript to a Name (line 543):

    @norecursion
    def set_scale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_scale'
        module_type_store = module_type_store.open_function_context('set_scale', 545, 4, False)
        # Assigning a type to the variable 'self' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolYScale.set_scale.__dict__.__setitem__('stypy_localization', localization)
        ToolYScale.set_scale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolYScale.set_scale.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolYScale.set_scale.__dict__.__setitem__('stypy_function_name', 'ToolYScale.set_scale')
        ToolYScale.set_scale.__dict__.__setitem__('stypy_param_names_list', ['ax', 'scale'])
        ToolYScale.set_scale.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolYScale.set_scale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolYScale.set_scale.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolYScale.set_scale.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolYScale.set_scale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolYScale.set_scale.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolYScale.set_scale', ['ax', 'scale'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_scale', localization, ['ax', 'scale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_scale(...)' code ##################

        
        # Call to set_yscale(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'scale' (line 546)
        scale_21266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 22), 'scale', False)
        # Processing the call keyword arguments (line 546)
        kwargs_21267 = {}
        # Getting the type of 'ax' (line 546)
        ax_21264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'ax', False)
        # Obtaining the member 'set_yscale' of a type (line 546)
        set_yscale_21265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 8), ax_21264, 'set_yscale')
        # Calling set_yscale(args, kwargs) (line 546)
        set_yscale_call_result_21268 = invoke(stypy.reporting.localization.Localization(__file__, 546, 8), set_yscale_21265, *[scale_21266], **kwargs_21267)
        
        
        # ################# End of 'set_scale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_scale' in the type store
        # Getting the type of 'stypy_return_type' (line 545)
        stypy_return_type_21269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21269)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_scale'
        return stypy_return_type_21269


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 539, 0, False)
        # Assigning a type to the variable 'self' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolYScale.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolYScale' (line 539)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 0), 'ToolYScale', ToolYScale)

# Assigning a Str to a Name (line 542):
str_21270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 18), 'str', 'Toogle Scale Y axis')
# Getting the type of 'ToolYScale'
ToolYScale_21271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolYScale')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolYScale_21271, 'description', str_21270)

# Assigning a Subscript to a Name (line 543):

# Obtaining the type of the subscript
str_21272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 30), 'str', 'keymap.yscale')
# Getting the type of 'rcParams' (line 543)
rcParams_21273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 543)
getitem___21274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 21), rcParams_21273, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 543)
subscript_call_result_21275 = invoke(stypy.reporting.localization.Localization(__file__, 543, 21), getitem___21274, str_21272)

# Getting the type of 'ToolYScale'
ToolYScale_21276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolYScale')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolYScale_21276, 'default_keymap', subscript_call_result_21275)
# Declaration of the 'ToolXScale' class
# Getting the type of 'AxisScaleBase' (line 549)
AxisScaleBase_21277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 17), 'AxisScaleBase')

class ToolXScale(AxisScaleBase_21277, ):
    str_21278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 4), 'str', 'Tool to toggle between linear and logarithmic scales on the X axis')
    
    # Assigning a Str to a Name (line 552):
    
    # Assigning a Str to a Name (line 552):
    
    # Assigning a Subscript to a Name (line 553):
    
    # Assigning a Subscript to a Name (line 553):

    @norecursion
    def set_scale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_scale'
        module_type_store = module_type_store.open_function_context('set_scale', 555, 4, False)
        # Assigning a type to the variable 'self' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolXScale.set_scale.__dict__.__setitem__('stypy_localization', localization)
        ToolXScale.set_scale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolXScale.set_scale.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolXScale.set_scale.__dict__.__setitem__('stypy_function_name', 'ToolXScale.set_scale')
        ToolXScale.set_scale.__dict__.__setitem__('stypy_param_names_list', ['ax', 'scale'])
        ToolXScale.set_scale.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolXScale.set_scale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolXScale.set_scale.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolXScale.set_scale.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolXScale.set_scale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolXScale.set_scale.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolXScale.set_scale', ['ax', 'scale'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_scale', localization, ['ax', 'scale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_scale(...)' code ##################

        
        # Call to set_xscale(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'scale' (line 556)
        scale_21281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 22), 'scale', False)
        # Processing the call keyword arguments (line 556)
        kwargs_21282 = {}
        # Getting the type of 'ax' (line 556)
        ax_21279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'ax', False)
        # Obtaining the member 'set_xscale' of a type (line 556)
        set_xscale_21280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 8), ax_21279, 'set_xscale')
        # Calling set_xscale(args, kwargs) (line 556)
        set_xscale_call_result_21283 = invoke(stypy.reporting.localization.Localization(__file__, 556, 8), set_xscale_21280, *[scale_21281], **kwargs_21282)
        
        
        # ################# End of 'set_scale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_scale' in the type store
        # Getting the type of 'stypy_return_type' (line 555)
        stypy_return_type_21284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_scale'
        return stypy_return_type_21284


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 549, 0, False)
        # Assigning a type to the variable 'self' (line 550)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolXScale.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolXScale' (line 549)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 0), 'ToolXScale', ToolXScale)

# Assigning a Str to a Name (line 552):
str_21285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 18), 'str', 'Toogle Scale X axis')
# Getting the type of 'ToolXScale'
ToolXScale_21286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolXScale')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolXScale_21286, 'description', str_21285)

# Assigning a Subscript to a Name (line 553):

# Obtaining the type of the subscript
str_21287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 30), 'str', 'keymap.xscale')
# Getting the type of 'rcParams' (line 553)
rcParams_21288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 553)
getitem___21289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 21), rcParams_21288, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 553)
subscript_call_result_21290 = invoke(stypy.reporting.localization.Localization(__file__, 553, 21), getitem___21289, str_21287)

# Getting the type of 'ToolXScale'
ToolXScale_21291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolXScale')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolXScale_21291, 'default_keymap', subscript_call_result_21290)
# Declaration of the 'ToolViewsPositions' class
# Getting the type of 'ToolBase' (line 559)
ToolBase_21292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 25), 'ToolBase')

class ToolViewsPositions(ToolBase_21292, ):
    str_21293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, (-1)), 'str', "\n    Auxiliary Tool to handle changes in views and positions\n\n    Runs in the background and should get used by all the tools that\n    need to access the figure's history of views and positions, e.g.\n\n    * `ToolZoom`\n    * `ToolPan`\n    * `ToolHome`\n    * `ToolBack`\n    * `ToolForward`\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 573, 4, False)
        # Assigning a type to the variable 'self' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolViewsPositions.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 574):
        
        # Assigning a Call to a Attribute (line 574):
        
        # Assigning a Call to a Attribute (line 574):
        
        # Call to WeakKeyDictionary(...): (line 574)
        # Processing the call keyword arguments (line 574)
        kwargs_21295 = {}
        # Getting the type of 'WeakKeyDictionary' (line 574)
        WeakKeyDictionary_21294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 21), 'WeakKeyDictionary', False)
        # Calling WeakKeyDictionary(args, kwargs) (line 574)
        WeakKeyDictionary_call_result_21296 = invoke(stypy.reporting.localization.Localization(__file__, 574, 21), WeakKeyDictionary_21294, *[], **kwargs_21295)
        
        # Getting the type of 'self' (line 574)
        self_21297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'self')
        # Setting the type of the member 'views' of a type (line 574)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 8), self_21297, 'views', WeakKeyDictionary_call_result_21296)
        
        # Assigning a Call to a Attribute (line 575):
        
        # Assigning a Call to a Attribute (line 575):
        
        # Assigning a Call to a Attribute (line 575):
        
        # Call to WeakKeyDictionary(...): (line 575)
        # Processing the call keyword arguments (line 575)
        kwargs_21299 = {}
        # Getting the type of 'WeakKeyDictionary' (line 575)
        WeakKeyDictionary_21298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 25), 'WeakKeyDictionary', False)
        # Calling WeakKeyDictionary(args, kwargs) (line 575)
        WeakKeyDictionary_call_result_21300 = invoke(stypy.reporting.localization.Localization(__file__, 575, 25), WeakKeyDictionary_21298, *[], **kwargs_21299)
        
        # Getting the type of 'self' (line 575)
        self_21301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'self')
        # Setting the type of the member 'positions' of a type (line 575)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 8), self_21301, 'positions', WeakKeyDictionary_call_result_21300)
        
        # Assigning a Call to a Attribute (line 576):
        
        # Assigning a Call to a Attribute (line 576):
        
        # Assigning a Call to a Attribute (line 576):
        
        # Call to WeakKeyDictionary(...): (line 576)
        # Processing the call keyword arguments (line 576)
        kwargs_21303 = {}
        # Getting the type of 'WeakKeyDictionary' (line 576)
        WeakKeyDictionary_21302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 26), 'WeakKeyDictionary', False)
        # Calling WeakKeyDictionary(args, kwargs) (line 576)
        WeakKeyDictionary_call_result_21304 = invoke(stypy.reporting.localization.Localization(__file__, 576, 26), WeakKeyDictionary_21302, *[], **kwargs_21303)
        
        # Getting the type of 'self' (line 576)
        self_21305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'self')
        # Setting the type of the member 'home_views' of a type (line 576)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 8), self_21305, 'home_views', WeakKeyDictionary_call_result_21304)
        
        # Call to __init__(...): (line 577)
        # Processing the call arguments (line 577)
        # Getting the type of 'self' (line 577)
        self_21308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 26), 'self', False)
        # Getting the type of 'args' (line 577)
        args_21309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 33), 'args', False)
        # Processing the call keyword arguments (line 577)
        # Getting the type of 'kwargs' (line 577)
        kwargs_21310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 41), 'kwargs', False)
        kwargs_21311 = {'kwargs_21310': kwargs_21310}
        # Getting the type of 'ToolBase' (line 577)
        ToolBase_21306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'ToolBase', False)
        # Obtaining the member '__init__' of a type (line 577)
        init___21307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 8), ToolBase_21306, '__init__')
        # Calling __init__(args, kwargs) (line 577)
        init___call_result_21312 = invoke(stypy.reporting.localization.Localization(__file__, 577, 8), init___21307, *[self_21308, args_21309], **kwargs_21311)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def add_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_figure'
        module_type_store = module_type_store.open_function_context('add_figure', 579, 4, False)
        # Assigning a type to the variable 'self' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolViewsPositions.add_figure.__dict__.__setitem__('stypy_localization', localization)
        ToolViewsPositions.add_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolViewsPositions.add_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolViewsPositions.add_figure.__dict__.__setitem__('stypy_function_name', 'ToolViewsPositions.add_figure')
        ToolViewsPositions.add_figure.__dict__.__setitem__('stypy_param_names_list', ['figure'])
        ToolViewsPositions.add_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolViewsPositions.add_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolViewsPositions.add_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolViewsPositions.add_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolViewsPositions.add_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolViewsPositions.add_figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolViewsPositions.add_figure', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_figure', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_figure(...)' code ##################

        str_21313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 8), 'str', 'Add the current figure to the stack of views and positions')
        
        
        # Getting the type of 'figure' (line 582)
        figure_21314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 11), 'figure')
        # Getting the type of 'self' (line 582)
        self_21315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 25), 'self')
        # Obtaining the member 'views' of a type (line 582)
        views_21316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 25), self_21315, 'views')
        # Applying the binary operator 'notin' (line 582)
        result_contains_21317 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 11), 'notin', figure_21314, views_21316)
        
        # Testing the type of an if condition (line 582)
        if_condition_21318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 582, 8), result_contains_21317)
        # Assigning a type to the variable 'if_condition_21318' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'if_condition_21318', if_condition_21318)
        # SSA begins for if statement (line 582)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 583):
        
        # Assigning a Call to a Subscript (line 583):
        
        # Assigning a Call to a Subscript (line 583):
        
        # Call to Stack(...): (line 583)
        # Processing the call keyword arguments (line 583)
        kwargs_21321 = {}
        # Getting the type of 'cbook' (line 583)
        cbook_21319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 33), 'cbook', False)
        # Obtaining the member 'Stack' of a type (line 583)
        Stack_21320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 33), cbook_21319, 'Stack')
        # Calling Stack(args, kwargs) (line 583)
        Stack_call_result_21322 = invoke(stypy.reporting.localization.Localization(__file__, 583, 33), Stack_21320, *[], **kwargs_21321)
        
        # Getting the type of 'self' (line 583)
        self_21323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'self')
        # Obtaining the member 'views' of a type (line 583)
        views_21324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 12), self_21323, 'views')
        # Getting the type of 'figure' (line 583)
        figure_21325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 23), 'figure')
        # Storing an element on a container (line 583)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 583, 12), views_21324, (figure_21325, Stack_call_result_21322))
        
        # Assigning a Call to a Subscript (line 584):
        
        # Assigning a Call to a Subscript (line 584):
        
        # Assigning a Call to a Subscript (line 584):
        
        # Call to Stack(...): (line 584)
        # Processing the call keyword arguments (line 584)
        kwargs_21328 = {}
        # Getting the type of 'cbook' (line 584)
        cbook_21326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 37), 'cbook', False)
        # Obtaining the member 'Stack' of a type (line 584)
        Stack_21327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 37), cbook_21326, 'Stack')
        # Calling Stack(args, kwargs) (line 584)
        Stack_call_result_21329 = invoke(stypy.reporting.localization.Localization(__file__, 584, 37), Stack_21327, *[], **kwargs_21328)
        
        # Getting the type of 'self' (line 584)
        self_21330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'self')
        # Obtaining the member 'positions' of a type (line 584)
        positions_21331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 12), self_21330, 'positions')
        # Getting the type of 'figure' (line 584)
        figure_21332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 27), 'figure')
        # Storing an element on a container (line 584)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 12), positions_21331, (figure_21332, Stack_call_result_21329))
        
        # Assigning a Call to a Subscript (line 585):
        
        # Assigning a Call to a Subscript (line 585):
        
        # Assigning a Call to a Subscript (line 585):
        
        # Call to WeakKeyDictionary(...): (line 585)
        # Processing the call keyword arguments (line 585)
        kwargs_21334 = {}
        # Getting the type of 'WeakKeyDictionary' (line 585)
        WeakKeyDictionary_21333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 38), 'WeakKeyDictionary', False)
        # Calling WeakKeyDictionary(args, kwargs) (line 585)
        WeakKeyDictionary_call_result_21335 = invoke(stypy.reporting.localization.Localization(__file__, 585, 38), WeakKeyDictionary_21333, *[], **kwargs_21334)
        
        # Getting the type of 'self' (line 585)
        self_21336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'self')
        # Obtaining the member 'home_views' of a type (line 585)
        home_views_21337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 12), self_21336, 'home_views')
        # Getting the type of 'figure' (line 585)
        figure_21338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 28), 'figure')
        # Storing an element on a container (line 585)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 12), home_views_21337, (figure_21338, WeakKeyDictionary_call_result_21335))
        
        # Call to push_current(...): (line 587)
        # Processing the call arguments (line 587)
        # Getting the type of 'figure' (line 587)
        figure_21341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'figure', False)
        # Processing the call keyword arguments (line 587)
        kwargs_21342 = {}
        # Getting the type of 'self' (line 587)
        self_21339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'self', False)
        # Obtaining the member 'push_current' of a type (line 587)
        push_current_21340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 12), self_21339, 'push_current')
        # Calling push_current(args, kwargs) (line 587)
        push_current_call_result_21343 = invoke(stypy.reporting.localization.Localization(__file__, 587, 12), push_current_21340, *[figure_21341], **kwargs_21342)
        
        
        # Call to add_axobserver(...): (line 589)
        # Processing the call arguments (line 589)

        @norecursion
        def _stypy_temp_lambda_8(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_8'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_8', 589, 34, True)
            # Passed parameters checking function
            _stypy_temp_lambda_8.stypy_localization = localization
            _stypy_temp_lambda_8.stypy_type_of_self = None
            _stypy_temp_lambda_8.stypy_type_store = module_type_store
            _stypy_temp_lambda_8.stypy_function_name = '_stypy_temp_lambda_8'
            _stypy_temp_lambda_8.stypy_param_names_list = ['fig']
            _stypy_temp_lambda_8.stypy_varargs_param_name = None
            _stypy_temp_lambda_8.stypy_kwargs_param_name = None
            _stypy_temp_lambda_8.stypy_call_defaults = defaults
            _stypy_temp_lambda_8.stypy_call_varargs = varargs
            _stypy_temp_lambda_8.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_8', ['fig'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_8', ['fig'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to update_home_views(...): (line 589)
            # Processing the call arguments (line 589)
            # Getting the type of 'fig' (line 589)
            fig_21348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 69), 'fig', False)
            # Processing the call keyword arguments (line 589)
            kwargs_21349 = {}
            # Getting the type of 'self' (line 589)
            self_21346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 46), 'self', False)
            # Obtaining the member 'update_home_views' of a type (line 589)
            update_home_views_21347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 46), self_21346, 'update_home_views')
            # Calling update_home_views(args, kwargs) (line 589)
            update_home_views_call_result_21350 = invoke(stypy.reporting.localization.Localization(__file__, 589, 46), update_home_views_21347, *[fig_21348], **kwargs_21349)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 589)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 34), 'stypy_return_type', update_home_views_call_result_21350)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_8' in the type store
            # Getting the type of 'stypy_return_type' (line 589)
            stypy_return_type_21351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 34), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_21351)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_8'
            return stypy_return_type_21351

        # Assigning a type to the variable '_stypy_temp_lambda_8' (line 589)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 34), '_stypy_temp_lambda_8', _stypy_temp_lambda_8)
        # Getting the type of '_stypy_temp_lambda_8' (line 589)
        _stypy_temp_lambda_8_21352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 34), '_stypy_temp_lambda_8')
        # Processing the call keyword arguments (line 589)
        kwargs_21353 = {}
        # Getting the type of 'figure' (line 589)
        figure_21344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'figure', False)
        # Obtaining the member 'add_axobserver' of a type (line 589)
        add_axobserver_21345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 12), figure_21344, 'add_axobserver')
        # Calling add_axobserver(args, kwargs) (line 589)
        add_axobserver_call_result_21354 = invoke(stypy.reporting.localization.Localization(__file__, 589, 12), add_axobserver_21345, *[_stypy_temp_lambda_8_21352], **kwargs_21353)
        
        # SSA join for if statement (line 582)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'add_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 579)
        stypy_return_type_21355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21355)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_figure'
        return stypy_return_type_21355


    @norecursion
    def clear(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clear'
        module_type_store = module_type_store.open_function_context('clear', 591, 4, False)
        # Assigning a type to the variable 'self' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolViewsPositions.clear.__dict__.__setitem__('stypy_localization', localization)
        ToolViewsPositions.clear.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolViewsPositions.clear.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolViewsPositions.clear.__dict__.__setitem__('stypy_function_name', 'ToolViewsPositions.clear')
        ToolViewsPositions.clear.__dict__.__setitem__('stypy_param_names_list', ['figure'])
        ToolViewsPositions.clear.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolViewsPositions.clear.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolViewsPositions.clear.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolViewsPositions.clear.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolViewsPositions.clear.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolViewsPositions.clear.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolViewsPositions.clear', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clear', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clear(...)' code ##################

        str_21356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 8), 'str', 'Reset the axes stack')
        
        
        # Getting the type of 'figure' (line 593)
        figure_21357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 11), 'figure')
        # Getting the type of 'self' (line 593)
        self_21358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 21), 'self')
        # Obtaining the member 'views' of a type (line 593)
        views_21359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 21), self_21358, 'views')
        # Applying the binary operator 'in' (line 593)
        result_contains_21360 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 11), 'in', figure_21357, views_21359)
        
        # Testing the type of an if condition (line 593)
        if_condition_21361 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 593, 8), result_contains_21360)
        # Assigning a type to the variable 'if_condition_21361' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'if_condition_21361', if_condition_21361)
        # SSA begins for if statement (line 593)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to clear(...): (line 594)
        # Processing the call keyword arguments (line 594)
        kwargs_21368 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'figure' (line 594)
        figure_21362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 23), 'figure', False)
        # Getting the type of 'self' (line 594)
        self_21363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'self', False)
        # Obtaining the member 'views' of a type (line 594)
        views_21364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 12), self_21363, 'views')
        # Obtaining the member '__getitem__' of a type (line 594)
        getitem___21365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 12), views_21364, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 594)
        subscript_call_result_21366 = invoke(stypy.reporting.localization.Localization(__file__, 594, 12), getitem___21365, figure_21362)
        
        # Obtaining the member 'clear' of a type (line 594)
        clear_21367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 12), subscript_call_result_21366, 'clear')
        # Calling clear(args, kwargs) (line 594)
        clear_call_result_21369 = invoke(stypy.reporting.localization.Localization(__file__, 594, 12), clear_21367, *[], **kwargs_21368)
        
        
        # Call to clear(...): (line 595)
        # Processing the call keyword arguments (line 595)
        kwargs_21376 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'figure' (line 595)
        figure_21370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 27), 'figure', False)
        # Getting the type of 'self' (line 595)
        self_21371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'self', False)
        # Obtaining the member 'positions' of a type (line 595)
        positions_21372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 12), self_21371, 'positions')
        # Obtaining the member '__getitem__' of a type (line 595)
        getitem___21373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 12), positions_21372, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 595)
        subscript_call_result_21374 = invoke(stypy.reporting.localization.Localization(__file__, 595, 12), getitem___21373, figure_21370)
        
        # Obtaining the member 'clear' of a type (line 595)
        clear_21375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 12), subscript_call_result_21374, 'clear')
        # Calling clear(args, kwargs) (line 595)
        clear_call_result_21377 = invoke(stypy.reporting.localization.Localization(__file__, 595, 12), clear_21375, *[], **kwargs_21376)
        
        
        # Call to clear(...): (line 596)
        # Processing the call keyword arguments (line 596)
        kwargs_21384 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'figure' (line 596)
        figure_21378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 28), 'figure', False)
        # Getting the type of 'self' (line 596)
        self_21379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 12), 'self', False)
        # Obtaining the member 'home_views' of a type (line 596)
        home_views_21380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 12), self_21379, 'home_views')
        # Obtaining the member '__getitem__' of a type (line 596)
        getitem___21381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 12), home_views_21380, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 596)
        subscript_call_result_21382 = invoke(stypy.reporting.localization.Localization(__file__, 596, 12), getitem___21381, figure_21378)
        
        # Obtaining the member 'clear' of a type (line 596)
        clear_21383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 12), subscript_call_result_21382, 'clear')
        # Calling clear(args, kwargs) (line 596)
        clear_call_result_21385 = invoke(stypy.reporting.localization.Localization(__file__, 596, 12), clear_21383, *[], **kwargs_21384)
        
        
        # Call to update_home_views(...): (line 597)
        # Processing the call keyword arguments (line 597)
        kwargs_21388 = {}
        # Getting the type of 'self' (line 597)
        self_21386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'self', False)
        # Obtaining the member 'update_home_views' of a type (line 597)
        update_home_views_21387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 12), self_21386, 'update_home_views')
        # Calling update_home_views(args, kwargs) (line 597)
        update_home_views_call_result_21389 = invoke(stypy.reporting.localization.Localization(__file__, 597, 12), update_home_views_21387, *[], **kwargs_21388)
        
        # SSA join for if statement (line 593)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'clear(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clear' in the type store
        # Getting the type of 'stypy_return_type' (line 591)
        stypy_return_type_21390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21390)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clear'
        return stypy_return_type_21390


    @norecursion
    def update_view(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_view'
        module_type_store = module_type_store.open_function_context('update_view', 599, 4, False)
        # Assigning a type to the variable 'self' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolViewsPositions.update_view.__dict__.__setitem__('stypy_localization', localization)
        ToolViewsPositions.update_view.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolViewsPositions.update_view.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolViewsPositions.update_view.__dict__.__setitem__('stypy_function_name', 'ToolViewsPositions.update_view')
        ToolViewsPositions.update_view.__dict__.__setitem__('stypy_param_names_list', [])
        ToolViewsPositions.update_view.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolViewsPositions.update_view.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolViewsPositions.update_view.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolViewsPositions.update_view.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolViewsPositions.update_view.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolViewsPositions.update_view.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolViewsPositions.update_view', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_view', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_view(...)' code ##################

        str_21391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, (-1)), 'str', "\n        Update the view limits and position for each axes from the current\n        stack position. If any axes are present in the figure that aren't in\n        the current stack position, use the home view limits for those axes and\n        don't update *any* positions.\n        ")
        
        # Assigning a Call to a Name (line 607):
        
        # Assigning a Call to a Name (line 607):
        
        # Assigning a Call to a Name (line 607):
        
        # Call to (...): (line 607)
        # Processing the call keyword arguments (line 607)
        kwargs_21398 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 607)
        self_21392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 27), 'self', False)
        # Obtaining the member 'figure' of a type (line 607)
        figure_21393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 27), self_21392, 'figure')
        # Getting the type of 'self' (line 607)
        self_21394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 16), 'self', False)
        # Obtaining the member 'views' of a type (line 607)
        views_21395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 16), self_21394, 'views')
        # Obtaining the member '__getitem__' of a type (line 607)
        getitem___21396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 16), views_21395, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 607)
        subscript_call_result_21397 = invoke(stypy.reporting.localization.Localization(__file__, 607, 16), getitem___21396, figure_21393)
        
        # Calling (args, kwargs) (line 607)
        _call_result_21399 = invoke(stypy.reporting.localization.Localization(__file__, 607, 16), subscript_call_result_21397, *[], **kwargs_21398)
        
        # Assigning a type to the variable 'views' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'views', _call_result_21399)
        
        # Type idiom detected: calculating its left and rigth part (line 608)
        # Getting the type of 'views' (line 608)
        views_21400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 11), 'views')
        # Getting the type of 'None' (line 608)
        None_21401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 20), 'None')
        
        (may_be_21402, more_types_in_union_21403) = may_be_none(views_21400, None_21401)

        if may_be_21402:

            if more_types_in_union_21403:
                # Runtime conditional SSA (line 608)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 609)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_21403:
                # SSA join for if statement (line 608)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 610):
        
        # Assigning a Call to a Name (line 610):
        
        # Assigning a Call to a Name (line 610):
        
        # Call to (...): (line 610)
        # Processing the call keyword arguments (line 610)
        kwargs_21410 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 610)
        self_21404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 29), 'self', False)
        # Obtaining the member 'figure' of a type (line 610)
        figure_21405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 29), self_21404, 'figure')
        # Getting the type of 'self' (line 610)
        self_21406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 14), 'self', False)
        # Obtaining the member 'positions' of a type (line 610)
        positions_21407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 14), self_21406, 'positions')
        # Obtaining the member '__getitem__' of a type (line 610)
        getitem___21408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 14), positions_21407, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 610)
        subscript_call_result_21409 = invoke(stypy.reporting.localization.Localization(__file__, 610, 14), getitem___21408, figure_21405)
        
        # Calling (args, kwargs) (line 610)
        _call_result_21411 = invoke(stypy.reporting.localization.Localization(__file__, 610, 14), subscript_call_result_21409, *[], **kwargs_21410)
        
        # Assigning a type to the variable 'pos' (line 610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'pos', _call_result_21411)
        
        # Type idiom detected: calculating its left and rigth part (line 611)
        # Getting the type of 'pos' (line 611)
        pos_21412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 11), 'pos')
        # Getting the type of 'None' (line 611)
        None_21413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 18), 'None')
        
        (may_be_21414, more_types_in_union_21415) = may_be_none(pos_21412, None_21413)

        if may_be_21414:

            if more_types_in_union_21415:
                # Runtime conditional SSA (line 611)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 612)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_21415:
                # SSA join for if statement (line 611)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Subscript to a Name (line 613):
        
        # Assigning a Subscript to a Name (line 613):
        
        # Assigning a Subscript to a Name (line 613):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 613)
        self_21416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 37), 'self')
        # Obtaining the member 'figure' of a type (line 613)
        figure_21417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 37), self_21416, 'figure')
        # Getting the type of 'self' (line 613)
        self_21418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 21), 'self')
        # Obtaining the member 'home_views' of a type (line 613)
        home_views_21419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 21), self_21418, 'home_views')
        # Obtaining the member '__getitem__' of a type (line 613)
        getitem___21420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 21), home_views_21419, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 613)
        subscript_call_result_21421 = invoke(stypy.reporting.localization.Localization(__file__, 613, 21), getitem___21420, figure_21417)
        
        # Assigning a type to the variable 'home_views' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'home_views', subscript_call_result_21421)
        
        # Assigning a Call to a Name (line 614):
        
        # Assigning a Call to a Name (line 614):
        
        # Assigning a Call to a Name (line 614):
        
        # Call to get_axes(...): (line 614)
        # Processing the call keyword arguments (line 614)
        kwargs_21425 = {}
        # Getting the type of 'self' (line 614)
        self_21422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 19), 'self', False)
        # Obtaining the member 'figure' of a type (line 614)
        figure_21423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 19), self_21422, 'figure')
        # Obtaining the member 'get_axes' of a type (line 614)
        get_axes_21424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 19), figure_21423, 'get_axes')
        # Calling get_axes(args, kwargs) (line 614)
        get_axes_call_result_21426 = invoke(stypy.reporting.localization.Localization(__file__, 614, 19), get_axes_21424, *[], **kwargs_21425)
        
        # Assigning a type to the variable 'all_axes' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'all_axes', get_axes_call_result_21426)
        
        # Getting the type of 'all_axes' (line 615)
        all_axes_21427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 17), 'all_axes')
        # Testing the type of a for loop iterable (line 615)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 615, 8), all_axes_21427)
        # Getting the type of the for loop variable (line 615)
        for_loop_var_21428 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 615, 8), all_axes_21427)
        # Assigning a type to the variable 'a' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'a', for_loop_var_21428)
        # SSA begins for a for statement (line 615)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'a' (line 616)
        a_21429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 15), 'a')
        # Getting the type of 'views' (line 616)
        views_21430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 20), 'views')
        # Applying the binary operator 'in' (line 616)
        result_contains_21431 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 15), 'in', a_21429, views_21430)
        
        # Testing the type of an if condition (line 616)
        if_condition_21432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 12), result_contains_21431)
        # Assigning a type to the variable 'if_condition_21432' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'if_condition_21432', if_condition_21432)
        # SSA begins for if statement (line 616)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 617):
        
        # Assigning a Subscript to a Name (line 617):
        
        # Assigning a Subscript to a Name (line 617):
        
        # Obtaining the type of the subscript
        # Getting the type of 'a' (line 617)
        a_21433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 33), 'a')
        # Getting the type of 'views' (line 617)
        views_21434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 27), 'views')
        # Obtaining the member '__getitem__' of a type (line 617)
        getitem___21435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 27), views_21434, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 617)
        subscript_call_result_21436 = invoke(stypy.reporting.localization.Localization(__file__, 617, 27), getitem___21435, a_21433)
        
        # Assigning a type to the variable 'cur_view' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'cur_view', subscript_call_result_21436)
        # SSA branch for the else part of an if statement (line 616)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 619):
        
        # Assigning a Subscript to a Name (line 619):
        
        # Assigning a Subscript to a Name (line 619):
        
        # Obtaining the type of the subscript
        # Getting the type of 'a' (line 619)
        a_21437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 38), 'a')
        # Getting the type of 'home_views' (line 619)
        home_views_21438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 27), 'home_views')
        # Obtaining the member '__getitem__' of a type (line 619)
        getitem___21439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 27), home_views_21438, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 619)
        subscript_call_result_21440 = invoke(stypy.reporting.localization.Localization(__file__, 619, 27), getitem___21439, a_21437)
        
        # Assigning a type to the variable 'cur_view' (line 619)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'cur_view', subscript_call_result_21440)
        # SSA join for if statement (line 616)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _set_view(...): (line 620)
        # Processing the call arguments (line 620)
        # Getting the type of 'cur_view' (line 620)
        cur_view_21443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 24), 'cur_view', False)
        # Processing the call keyword arguments (line 620)
        kwargs_21444 = {}
        # Getting the type of 'a' (line 620)
        a_21441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'a', False)
        # Obtaining the member '_set_view' of a type (line 620)
        _set_view_21442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 12), a_21441, '_set_view')
        # Calling _set_view(args, kwargs) (line 620)
        _set_view_call_result_21445 = invoke(stypy.reporting.localization.Localization(__file__, 620, 12), _set_view_21442, *[cur_view_21443], **kwargs_21444)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to issubset(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 'pos' (line 622)
        pos_21451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 34), 'pos', False)
        # Processing the call keyword arguments (line 622)
        kwargs_21452 = {}
        
        # Call to set(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 'all_axes' (line 622)
        all_axes_21447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 15), 'all_axes', False)
        # Processing the call keyword arguments (line 622)
        kwargs_21448 = {}
        # Getting the type of 'set' (line 622)
        set_21446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 11), 'set', False)
        # Calling set(args, kwargs) (line 622)
        set_call_result_21449 = invoke(stypy.reporting.localization.Localization(__file__, 622, 11), set_21446, *[all_axes_21447], **kwargs_21448)
        
        # Obtaining the member 'issubset' of a type (line 622)
        issubset_21450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 11), set_call_result_21449, 'issubset')
        # Calling issubset(args, kwargs) (line 622)
        issubset_call_result_21453 = invoke(stypy.reporting.localization.Localization(__file__, 622, 11), issubset_21450, *[pos_21451], **kwargs_21452)
        
        # Testing the type of an if condition (line 622)
        if_condition_21454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 622, 8), issubset_call_result_21453)
        # Assigning a type to the variable 'if_condition_21454' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'if_condition_21454', if_condition_21454)
        # SSA begins for if statement (line 622)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'all_axes' (line 623)
        all_axes_21455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 21), 'all_axes')
        # Testing the type of a for loop iterable (line 623)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 623, 12), all_axes_21455)
        # Getting the type of the for loop variable (line 623)
        for_loop_var_21456 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 623, 12), all_axes_21455)
        # Assigning a type to the variable 'a' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'a', for_loop_var_21456)
        # SSA begins for a for statement (line 623)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_position(...): (line 625)
        # Processing the call arguments (line 625)
        
        # Obtaining the type of the subscript
        int_21459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 38), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'a' (line 625)
        a_21460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 35), 'a', False)
        # Getting the type of 'pos' (line 625)
        pos_21461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 31), 'pos', False)
        # Obtaining the member '__getitem__' of a type (line 625)
        getitem___21462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 31), pos_21461, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 625)
        subscript_call_result_21463 = invoke(stypy.reporting.localization.Localization(__file__, 625, 31), getitem___21462, a_21460)
        
        # Obtaining the member '__getitem__' of a type (line 625)
        getitem___21464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 31), subscript_call_result_21463, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 625)
        subscript_call_result_21465 = invoke(stypy.reporting.localization.Localization(__file__, 625, 31), getitem___21464, int_21459)
        
        str_21466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 42), 'str', 'original')
        # Processing the call keyword arguments (line 625)
        kwargs_21467 = {}
        # Getting the type of 'a' (line 625)
        a_21457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 16), 'a', False)
        # Obtaining the member 'set_position' of a type (line 625)
        set_position_21458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 16), a_21457, 'set_position')
        # Calling set_position(args, kwargs) (line 625)
        set_position_call_result_21468 = invoke(stypy.reporting.localization.Localization(__file__, 625, 16), set_position_21458, *[subscript_call_result_21465, str_21466], **kwargs_21467)
        
        
        # Call to set_position(...): (line 626)
        # Processing the call arguments (line 626)
        
        # Obtaining the type of the subscript
        int_21471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 38), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'a' (line 626)
        a_21472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 35), 'a', False)
        # Getting the type of 'pos' (line 626)
        pos_21473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 31), 'pos', False)
        # Obtaining the member '__getitem__' of a type (line 626)
        getitem___21474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 31), pos_21473, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 626)
        subscript_call_result_21475 = invoke(stypy.reporting.localization.Localization(__file__, 626, 31), getitem___21474, a_21472)
        
        # Obtaining the member '__getitem__' of a type (line 626)
        getitem___21476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 31), subscript_call_result_21475, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 626)
        subscript_call_result_21477 = invoke(stypy.reporting.localization.Localization(__file__, 626, 31), getitem___21476, int_21471)
        
        str_21478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 42), 'str', 'active')
        # Processing the call keyword arguments (line 626)
        kwargs_21479 = {}
        # Getting the type of 'a' (line 626)
        a_21469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 16), 'a', False)
        # Obtaining the member 'set_position' of a type (line 626)
        set_position_21470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 16), a_21469, 'set_position')
        # Calling set_position(args, kwargs) (line 626)
        set_position_call_result_21480 = invoke(stypy.reporting.localization.Localization(__file__, 626, 16), set_position_21470, *[subscript_call_result_21477, str_21478], **kwargs_21479)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 622)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw_idle(...): (line 628)
        # Processing the call keyword arguments (line 628)
        kwargs_21485 = {}
        # Getting the type of 'self' (line 628)
        self_21481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 628)
        figure_21482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 8), self_21481, 'figure')
        # Obtaining the member 'canvas' of a type (line 628)
        canvas_21483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 8), figure_21482, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 628)
        draw_idle_21484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 8), canvas_21483, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 628)
        draw_idle_call_result_21486 = invoke(stypy.reporting.localization.Localization(__file__, 628, 8), draw_idle_21484, *[], **kwargs_21485)
        
        
        # ################# End of 'update_view(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_view' in the type store
        # Getting the type of 'stypy_return_type' (line 599)
        stypy_return_type_21487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21487)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_view'
        return stypy_return_type_21487


    @norecursion
    def push_current(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 630)
        None_21488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 34), 'None')
        defaults = [None_21488]
        # Create a new context for function 'push_current'
        module_type_store = module_type_store.open_function_context('push_current', 630, 4, False)
        # Assigning a type to the variable 'self' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolViewsPositions.push_current.__dict__.__setitem__('stypy_localization', localization)
        ToolViewsPositions.push_current.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolViewsPositions.push_current.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolViewsPositions.push_current.__dict__.__setitem__('stypy_function_name', 'ToolViewsPositions.push_current')
        ToolViewsPositions.push_current.__dict__.__setitem__('stypy_param_names_list', ['figure'])
        ToolViewsPositions.push_current.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolViewsPositions.push_current.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolViewsPositions.push_current.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolViewsPositions.push_current.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolViewsPositions.push_current.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolViewsPositions.push_current.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolViewsPositions.push_current', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'push_current', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'push_current(...)' code ##################

        str_21489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, (-1)), 'str', '\n        Push the current view limits and position onto their respective stacks\n        ')
        
        
        # Getting the type of 'figure' (line 634)
        figure_21490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 15), 'figure')
        # Applying the 'not' unary operator (line 634)
        result_not__21491 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 11), 'not', figure_21490)
        
        # Testing the type of an if condition (line 634)
        if_condition_21492 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 634, 8), result_not__21491)
        # Assigning a type to the variable 'if_condition_21492' (line 634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'if_condition_21492', if_condition_21492)
        # SSA begins for if statement (line 634)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 635):
        
        # Assigning a Attribute to a Name (line 635):
        
        # Assigning a Attribute to a Name (line 635):
        # Getting the type of 'self' (line 635)
        self_21493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 21), 'self')
        # Obtaining the member 'figure' of a type (line 635)
        figure_21494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 21), self_21493, 'figure')
        # Assigning a type to the variable 'figure' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 12), 'figure', figure_21494)
        # SSA join for if statement (line 634)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 636):
        
        # Assigning a Call to a Name (line 636):
        
        # Assigning a Call to a Name (line 636):
        
        # Call to WeakKeyDictionary(...): (line 636)
        # Processing the call keyword arguments (line 636)
        kwargs_21496 = {}
        # Getting the type of 'WeakKeyDictionary' (line 636)
        WeakKeyDictionary_21495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'WeakKeyDictionary', False)
        # Calling WeakKeyDictionary(args, kwargs) (line 636)
        WeakKeyDictionary_call_result_21497 = invoke(stypy.reporting.localization.Localization(__file__, 636, 16), WeakKeyDictionary_21495, *[], **kwargs_21496)
        
        # Assigning a type to the variable 'views' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'views', WeakKeyDictionary_call_result_21497)
        
        # Assigning a Call to a Name (line 637):
        
        # Assigning a Call to a Name (line 637):
        
        # Assigning a Call to a Name (line 637):
        
        # Call to WeakKeyDictionary(...): (line 637)
        # Processing the call keyword arguments (line 637)
        kwargs_21499 = {}
        # Getting the type of 'WeakKeyDictionary' (line 637)
        WeakKeyDictionary_21498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 14), 'WeakKeyDictionary', False)
        # Calling WeakKeyDictionary(args, kwargs) (line 637)
        WeakKeyDictionary_call_result_21500 = invoke(stypy.reporting.localization.Localization(__file__, 637, 14), WeakKeyDictionary_21498, *[], **kwargs_21499)
        
        # Assigning a type to the variable 'pos' (line 637)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'pos', WeakKeyDictionary_call_result_21500)
        
        
        # Call to get_axes(...): (line 638)
        # Processing the call keyword arguments (line 638)
        kwargs_21503 = {}
        # Getting the type of 'figure' (line 638)
        figure_21501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 17), 'figure', False)
        # Obtaining the member 'get_axes' of a type (line 638)
        get_axes_21502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 17), figure_21501, 'get_axes')
        # Calling get_axes(args, kwargs) (line 638)
        get_axes_call_result_21504 = invoke(stypy.reporting.localization.Localization(__file__, 638, 17), get_axes_21502, *[], **kwargs_21503)
        
        # Testing the type of a for loop iterable (line 638)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 638, 8), get_axes_call_result_21504)
        # Getting the type of the for loop variable (line 638)
        for_loop_var_21505 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 638, 8), get_axes_call_result_21504)
        # Assigning a type to the variable 'a' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'a', for_loop_var_21505)
        # SSA begins for a for statement (line 638)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Subscript (line 639):
        
        # Assigning a Call to a Subscript (line 639):
        
        # Assigning a Call to a Subscript (line 639):
        
        # Call to _get_view(...): (line 639)
        # Processing the call keyword arguments (line 639)
        kwargs_21508 = {}
        # Getting the type of 'a' (line 639)
        a_21506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 23), 'a', False)
        # Obtaining the member '_get_view' of a type (line 639)
        _get_view_21507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 23), a_21506, '_get_view')
        # Calling _get_view(args, kwargs) (line 639)
        _get_view_call_result_21509 = invoke(stypy.reporting.localization.Localization(__file__, 639, 23), _get_view_21507, *[], **kwargs_21508)
        
        # Getting the type of 'views' (line 639)
        views_21510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 12), 'views')
        # Getting the type of 'a' (line 639)
        a_21511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 18), 'a')
        # Storing an element on a container (line 639)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 639, 12), views_21510, (a_21511, _get_view_call_result_21509))
        
        # Assigning a Call to a Subscript (line 640):
        
        # Assigning a Call to a Subscript (line 640):
        
        # Assigning a Call to a Subscript (line 640):
        
        # Call to _axes_pos(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 'a' (line 640)
        a_21514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 36), 'a', False)
        # Processing the call keyword arguments (line 640)
        kwargs_21515 = {}
        # Getting the type of 'self' (line 640)
        self_21512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 21), 'self', False)
        # Obtaining the member '_axes_pos' of a type (line 640)
        _axes_pos_21513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 21), self_21512, '_axes_pos')
        # Calling _axes_pos(args, kwargs) (line 640)
        _axes_pos_call_result_21516 = invoke(stypy.reporting.localization.Localization(__file__, 640, 21), _axes_pos_21513, *[a_21514], **kwargs_21515)
        
        # Getting the type of 'pos' (line 640)
        pos_21517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'pos')
        # Getting the type of 'a' (line 640)
        a_21518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 16), 'a')
        # Storing an element on a container (line 640)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 12), pos_21517, (a_21518, _axes_pos_call_result_21516))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to push(...): (line 641)
        # Processing the call arguments (line 641)
        # Getting the type of 'views' (line 641)
        views_21525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 32), 'views', False)
        # Processing the call keyword arguments (line 641)
        kwargs_21526 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'figure' (line 641)
        figure_21519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 19), 'figure', False)
        # Getting the type of 'self' (line 641)
        self_21520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'self', False)
        # Obtaining the member 'views' of a type (line 641)
        views_21521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 8), self_21520, 'views')
        # Obtaining the member '__getitem__' of a type (line 641)
        getitem___21522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 8), views_21521, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 641)
        subscript_call_result_21523 = invoke(stypy.reporting.localization.Localization(__file__, 641, 8), getitem___21522, figure_21519)
        
        # Obtaining the member 'push' of a type (line 641)
        push_21524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 8), subscript_call_result_21523, 'push')
        # Calling push(args, kwargs) (line 641)
        push_call_result_21527 = invoke(stypy.reporting.localization.Localization(__file__, 641, 8), push_21524, *[views_21525], **kwargs_21526)
        
        
        # Call to push(...): (line 642)
        # Processing the call arguments (line 642)
        # Getting the type of 'pos' (line 642)
        pos_21534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 36), 'pos', False)
        # Processing the call keyword arguments (line 642)
        kwargs_21535 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'figure' (line 642)
        figure_21528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 23), 'figure', False)
        # Getting the type of 'self' (line 642)
        self_21529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'self', False)
        # Obtaining the member 'positions' of a type (line 642)
        positions_21530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 8), self_21529, 'positions')
        # Obtaining the member '__getitem__' of a type (line 642)
        getitem___21531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 8), positions_21530, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 642)
        subscript_call_result_21532 = invoke(stypy.reporting.localization.Localization(__file__, 642, 8), getitem___21531, figure_21528)
        
        # Obtaining the member 'push' of a type (line 642)
        push_21533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 8), subscript_call_result_21532, 'push')
        # Calling push(args, kwargs) (line 642)
        push_call_result_21536 = invoke(stypy.reporting.localization.Localization(__file__, 642, 8), push_21533, *[pos_21534], **kwargs_21535)
        
        
        # ################# End of 'push_current(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'push_current' in the type store
        # Getting the type of 'stypy_return_type' (line 630)
        stypy_return_type_21537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21537)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'push_current'
        return stypy_return_type_21537


    @norecursion
    def _axes_pos(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_axes_pos'
        module_type_store = module_type_store.open_function_context('_axes_pos', 644, 4, False)
        # Assigning a type to the variable 'self' (line 645)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolViewsPositions._axes_pos.__dict__.__setitem__('stypy_localization', localization)
        ToolViewsPositions._axes_pos.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolViewsPositions._axes_pos.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolViewsPositions._axes_pos.__dict__.__setitem__('stypy_function_name', 'ToolViewsPositions._axes_pos')
        ToolViewsPositions._axes_pos.__dict__.__setitem__('stypy_param_names_list', ['ax'])
        ToolViewsPositions._axes_pos.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolViewsPositions._axes_pos.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolViewsPositions._axes_pos.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolViewsPositions._axes_pos.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolViewsPositions._axes_pos.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolViewsPositions._axes_pos.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolViewsPositions._axes_pos', ['ax'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_axes_pos', localization, ['ax'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_axes_pos(...)' code ##################

        str_21538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, (-1)), 'str', '\n        Return the original and modified positions for the specified axes\n\n        Parameters\n        ----------\n        ax : (matplotlib.axes.AxesSubplot)\n        The axes to get the positions for\n\n        Returns\n        -------\n        limits : (tuple)\n        A tuple of the original and modified positions\n        ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 659)
        tuple_21539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 659)
        # Adding element type (line 659)
        
        # Call to frozen(...): (line 659)
        # Processing the call keyword arguments (line 659)
        kwargs_21546 = {}
        
        # Call to get_position(...): (line 659)
        # Processing the call arguments (line 659)
        # Getting the type of 'True' (line 659)
        True_21542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 32), 'True', False)
        # Processing the call keyword arguments (line 659)
        kwargs_21543 = {}
        # Getting the type of 'ax' (line 659)
        ax_21540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 16), 'ax', False)
        # Obtaining the member 'get_position' of a type (line 659)
        get_position_21541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 16), ax_21540, 'get_position')
        # Calling get_position(args, kwargs) (line 659)
        get_position_call_result_21544 = invoke(stypy.reporting.localization.Localization(__file__, 659, 16), get_position_21541, *[True_21542], **kwargs_21543)
        
        # Obtaining the member 'frozen' of a type (line 659)
        frozen_21545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 16), get_position_call_result_21544, 'frozen')
        # Calling frozen(args, kwargs) (line 659)
        frozen_call_result_21547 = invoke(stypy.reporting.localization.Localization(__file__, 659, 16), frozen_21545, *[], **kwargs_21546)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 16), tuple_21539, frozen_call_result_21547)
        # Adding element type (line 659)
        
        # Call to frozen(...): (line 660)
        # Processing the call keyword arguments (line 660)
        kwargs_21553 = {}
        
        # Call to get_position(...): (line 660)
        # Processing the call keyword arguments (line 660)
        kwargs_21550 = {}
        # Getting the type of 'ax' (line 660)
        ax_21548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 16), 'ax', False)
        # Obtaining the member 'get_position' of a type (line 660)
        get_position_21549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 16), ax_21548, 'get_position')
        # Calling get_position(args, kwargs) (line 660)
        get_position_call_result_21551 = invoke(stypy.reporting.localization.Localization(__file__, 660, 16), get_position_21549, *[], **kwargs_21550)
        
        # Obtaining the member 'frozen' of a type (line 660)
        frozen_21552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 16), get_position_call_result_21551, 'frozen')
        # Calling frozen(args, kwargs) (line 660)
        frozen_call_result_21554 = invoke(stypy.reporting.localization.Localization(__file__, 660, 16), frozen_21552, *[], **kwargs_21553)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 16), tuple_21539, frozen_call_result_21554)
        
        # Assigning a type to the variable 'stypy_return_type' (line 659)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 8), 'stypy_return_type', tuple_21539)
        
        # ################# End of '_axes_pos(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_axes_pos' in the type store
        # Getting the type of 'stypy_return_type' (line 644)
        stypy_return_type_21555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21555)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_axes_pos'
        return stypy_return_type_21555


    @norecursion
    def update_home_views(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 662)
        None_21556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 39), 'None')
        defaults = [None_21556]
        # Create a new context for function 'update_home_views'
        module_type_store = module_type_store.open_function_context('update_home_views', 662, 4, False)
        # Assigning a type to the variable 'self' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolViewsPositions.update_home_views.__dict__.__setitem__('stypy_localization', localization)
        ToolViewsPositions.update_home_views.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolViewsPositions.update_home_views.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolViewsPositions.update_home_views.__dict__.__setitem__('stypy_function_name', 'ToolViewsPositions.update_home_views')
        ToolViewsPositions.update_home_views.__dict__.__setitem__('stypy_param_names_list', ['figure'])
        ToolViewsPositions.update_home_views.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolViewsPositions.update_home_views.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolViewsPositions.update_home_views.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolViewsPositions.update_home_views.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolViewsPositions.update_home_views.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolViewsPositions.update_home_views.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolViewsPositions.update_home_views', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_home_views', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_home_views(...)' code ##################

        str_21557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, (-1)), 'str', '\n        Make sure that self.home_views has an entry for all axes present in the\n        figure\n        ')
        
        
        # Getting the type of 'figure' (line 668)
        figure_21558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 15), 'figure')
        # Applying the 'not' unary operator (line 668)
        result_not__21559 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 11), 'not', figure_21558)
        
        # Testing the type of an if condition (line 668)
        if_condition_21560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 668, 8), result_not__21559)
        # Assigning a type to the variable 'if_condition_21560' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'if_condition_21560', if_condition_21560)
        # SSA begins for if statement (line 668)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 669):
        
        # Assigning a Attribute to a Name (line 669):
        
        # Assigning a Attribute to a Name (line 669):
        # Getting the type of 'self' (line 669)
        self_21561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 21), 'self')
        # Obtaining the member 'figure' of a type (line 669)
        figure_21562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 21), self_21561, 'figure')
        # Assigning a type to the variable 'figure' (line 669)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'figure', figure_21562)
        # SSA join for if statement (line 668)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to get_axes(...): (line 670)
        # Processing the call keyword arguments (line 670)
        kwargs_21565 = {}
        # Getting the type of 'figure' (line 670)
        figure_21563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 17), 'figure', False)
        # Obtaining the member 'get_axes' of a type (line 670)
        get_axes_21564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 17), figure_21563, 'get_axes')
        # Calling get_axes(args, kwargs) (line 670)
        get_axes_call_result_21566 = invoke(stypy.reporting.localization.Localization(__file__, 670, 17), get_axes_21564, *[], **kwargs_21565)
        
        # Testing the type of a for loop iterable (line 670)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 670, 8), get_axes_call_result_21566)
        # Getting the type of the for loop variable (line 670)
        for_loop_var_21567 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 670, 8), get_axes_call_result_21566)
        # Assigning a type to the variable 'a' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'a', for_loop_var_21567)
        # SSA begins for a for statement (line 670)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'a' (line 671)
        a_21568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 15), 'a')
        
        # Obtaining the type of the subscript
        # Getting the type of 'figure' (line 671)
        figure_21569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 40), 'figure')
        # Getting the type of 'self' (line 671)
        self_21570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 24), 'self')
        # Obtaining the member 'home_views' of a type (line 671)
        home_views_21571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 24), self_21570, 'home_views')
        # Obtaining the member '__getitem__' of a type (line 671)
        getitem___21572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 24), home_views_21571, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 671)
        subscript_call_result_21573 = invoke(stypy.reporting.localization.Localization(__file__, 671, 24), getitem___21572, figure_21569)
        
        # Applying the binary operator 'notin' (line 671)
        result_contains_21574 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 15), 'notin', a_21568, subscript_call_result_21573)
        
        # Testing the type of an if condition (line 671)
        if_condition_21575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 671, 12), result_contains_21574)
        # Assigning a type to the variable 'if_condition_21575' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 12), 'if_condition_21575', if_condition_21575)
        # SSA begins for if statement (line 671)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 672):
        
        # Assigning a Call to a Subscript (line 672):
        
        # Assigning a Call to a Subscript (line 672):
        
        # Call to _get_view(...): (line 672)
        # Processing the call keyword arguments (line 672)
        kwargs_21578 = {}
        # Getting the type of 'a' (line 672)
        a_21576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 45), 'a', False)
        # Obtaining the member '_get_view' of a type (line 672)
        _get_view_21577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 45), a_21576, '_get_view')
        # Calling _get_view(args, kwargs) (line 672)
        _get_view_call_result_21579 = invoke(stypy.reporting.localization.Localization(__file__, 672, 45), _get_view_21577, *[], **kwargs_21578)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'figure' (line 672)
        figure_21580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 32), 'figure')
        # Getting the type of 'self' (line 672)
        self_21581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 16), 'self')
        # Obtaining the member 'home_views' of a type (line 672)
        home_views_21582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 16), self_21581, 'home_views')
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___21583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 16), home_views_21582, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_21584 = invoke(stypy.reporting.localization.Localization(__file__, 672, 16), getitem___21583, figure_21580)
        
        # Getting the type of 'a' (line 672)
        a_21585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 40), 'a')
        # Storing an element on a container (line 672)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 16), subscript_call_result_21584, (a_21585, _get_view_call_result_21579))
        # SSA join for if statement (line 671)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'update_home_views(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_home_views' in the type store
        # Getting the type of 'stypy_return_type' (line 662)
        stypy_return_type_21586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21586)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_home_views'
        return stypy_return_type_21586


    @norecursion
    def refresh_locators(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'refresh_locators'
        module_type_store = module_type_store.open_function_context('refresh_locators', 674, 4, False)
        # Assigning a type to the variable 'self' (line 675)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolViewsPositions.refresh_locators.__dict__.__setitem__('stypy_localization', localization)
        ToolViewsPositions.refresh_locators.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolViewsPositions.refresh_locators.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolViewsPositions.refresh_locators.__dict__.__setitem__('stypy_function_name', 'ToolViewsPositions.refresh_locators')
        ToolViewsPositions.refresh_locators.__dict__.__setitem__('stypy_param_names_list', [])
        ToolViewsPositions.refresh_locators.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolViewsPositions.refresh_locators.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolViewsPositions.refresh_locators.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolViewsPositions.refresh_locators.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolViewsPositions.refresh_locators.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolViewsPositions.refresh_locators.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolViewsPositions.refresh_locators', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'refresh_locators', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'refresh_locators(...)' code ##################

        str_21587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 8), 'str', 'Redraw the canvases, update the locators')
        
        
        # Call to get_axes(...): (line 676)
        # Processing the call keyword arguments (line 676)
        kwargs_21591 = {}
        # Getting the type of 'self' (line 676)
        self_21588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 17), 'self', False)
        # Obtaining the member 'figure' of a type (line 676)
        figure_21589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 17), self_21588, 'figure')
        # Obtaining the member 'get_axes' of a type (line 676)
        get_axes_21590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 17), figure_21589, 'get_axes')
        # Calling get_axes(args, kwargs) (line 676)
        get_axes_call_result_21592 = invoke(stypy.reporting.localization.Localization(__file__, 676, 17), get_axes_21590, *[], **kwargs_21591)
        
        # Testing the type of a for loop iterable (line 676)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 676, 8), get_axes_call_result_21592)
        # Getting the type of the for loop variable (line 676)
        for_loop_var_21593 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 676, 8), get_axes_call_result_21592)
        # Assigning a type to the variable 'a' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 8), 'a', for_loop_var_21593)
        # SSA begins for a for statement (line 676)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 677):
        
        # Assigning a Call to a Name (line 677):
        
        # Assigning a Call to a Name (line 677):
        
        # Call to getattr(...): (line 677)
        # Processing the call arguments (line 677)
        # Getting the type of 'a' (line 677)
        a_21595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 28), 'a', False)
        str_21596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 31), 'str', 'xaxis')
        # Getting the type of 'None' (line 677)
        None_21597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 40), 'None', False)
        # Processing the call keyword arguments (line 677)
        kwargs_21598 = {}
        # Getting the type of 'getattr' (line 677)
        getattr_21594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 677)
        getattr_call_result_21599 = invoke(stypy.reporting.localization.Localization(__file__, 677, 20), getattr_21594, *[a_21595, str_21596, None_21597], **kwargs_21598)
        
        # Assigning a type to the variable 'xaxis' (line 677)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 12), 'xaxis', getattr_call_result_21599)
        
        # Assigning a Call to a Name (line 678):
        
        # Assigning a Call to a Name (line 678):
        
        # Assigning a Call to a Name (line 678):
        
        # Call to getattr(...): (line 678)
        # Processing the call arguments (line 678)
        # Getting the type of 'a' (line 678)
        a_21601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 28), 'a', False)
        str_21602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 31), 'str', 'yaxis')
        # Getting the type of 'None' (line 678)
        None_21603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 40), 'None', False)
        # Processing the call keyword arguments (line 678)
        kwargs_21604 = {}
        # Getting the type of 'getattr' (line 678)
        getattr_21600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 678)
        getattr_call_result_21605 = invoke(stypy.reporting.localization.Localization(__file__, 678, 20), getattr_21600, *[a_21601, str_21602, None_21603], **kwargs_21604)
        
        # Assigning a type to the variable 'yaxis' (line 678)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'yaxis', getattr_call_result_21605)
        
        # Assigning a Call to a Name (line 679):
        
        # Assigning a Call to a Name (line 679):
        
        # Assigning a Call to a Name (line 679):
        
        # Call to getattr(...): (line 679)
        # Processing the call arguments (line 679)
        # Getting the type of 'a' (line 679)
        a_21607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 28), 'a', False)
        str_21608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 31), 'str', 'zaxis')
        # Getting the type of 'None' (line 679)
        None_21609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 40), 'None', False)
        # Processing the call keyword arguments (line 679)
        kwargs_21610 = {}
        # Getting the type of 'getattr' (line 679)
        getattr_21606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 679)
        getattr_call_result_21611 = invoke(stypy.reporting.localization.Localization(__file__, 679, 20), getattr_21606, *[a_21607, str_21608, None_21609], **kwargs_21610)
        
        # Assigning a type to the variable 'zaxis' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 12), 'zaxis', getattr_call_result_21611)
        
        # Assigning a List to a Name (line 680):
        
        # Assigning a List to a Name (line 680):
        
        # Assigning a List to a Name (line 680):
        
        # Obtaining an instance of the builtin type 'list' (line 680)
        list_21612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 680)
        
        # Assigning a type to the variable 'locators' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 12), 'locators', list_21612)
        
        # Type idiom detected: calculating its left and rigth part (line 681)
        # Getting the type of 'xaxis' (line 681)
        xaxis_21613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 12), 'xaxis')
        # Getting the type of 'None' (line 681)
        None_21614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 28), 'None')
        
        (may_be_21615, more_types_in_union_21616) = may_not_be_none(xaxis_21613, None_21614)

        if may_be_21615:

            if more_types_in_union_21616:
                # Runtime conditional SSA (line 681)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 682)
            # Processing the call arguments (line 682)
            
            # Call to get_major_locator(...): (line 682)
            # Processing the call keyword arguments (line 682)
            kwargs_21621 = {}
            # Getting the type of 'xaxis' (line 682)
            xaxis_21619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 32), 'xaxis', False)
            # Obtaining the member 'get_major_locator' of a type (line 682)
            get_major_locator_21620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 32), xaxis_21619, 'get_major_locator')
            # Calling get_major_locator(args, kwargs) (line 682)
            get_major_locator_call_result_21622 = invoke(stypy.reporting.localization.Localization(__file__, 682, 32), get_major_locator_21620, *[], **kwargs_21621)
            
            # Processing the call keyword arguments (line 682)
            kwargs_21623 = {}
            # Getting the type of 'locators' (line 682)
            locators_21617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 16), 'locators', False)
            # Obtaining the member 'append' of a type (line 682)
            append_21618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 16), locators_21617, 'append')
            # Calling append(args, kwargs) (line 682)
            append_call_result_21624 = invoke(stypy.reporting.localization.Localization(__file__, 682, 16), append_21618, *[get_major_locator_call_result_21622], **kwargs_21623)
            
            
            # Call to append(...): (line 683)
            # Processing the call arguments (line 683)
            
            # Call to get_minor_locator(...): (line 683)
            # Processing the call keyword arguments (line 683)
            kwargs_21629 = {}
            # Getting the type of 'xaxis' (line 683)
            xaxis_21627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 32), 'xaxis', False)
            # Obtaining the member 'get_minor_locator' of a type (line 683)
            get_minor_locator_21628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 32), xaxis_21627, 'get_minor_locator')
            # Calling get_minor_locator(args, kwargs) (line 683)
            get_minor_locator_call_result_21630 = invoke(stypy.reporting.localization.Localization(__file__, 683, 32), get_minor_locator_21628, *[], **kwargs_21629)
            
            # Processing the call keyword arguments (line 683)
            kwargs_21631 = {}
            # Getting the type of 'locators' (line 683)
            locators_21625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 16), 'locators', False)
            # Obtaining the member 'append' of a type (line 683)
            append_21626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 16), locators_21625, 'append')
            # Calling append(args, kwargs) (line 683)
            append_call_result_21632 = invoke(stypy.reporting.localization.Localization(__file__, 683, 16), append_21626, *[get_minor_locator_call_result_21630], **kwargs_21631)
            

            if more_types_in_union_21616:
                # SSA join for if statement (line 681)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 684)
        # Getting the type of 'yaxis' (line 684)
        yaxis_21633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 12), 'yaxis')
        # Getting the type of 'None' (line 684)
        None_21634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 28), 'None')
        
        (may_be_21635, more_types_in_union_21636) = may_not_be_none(yaxis_21633, None_21634)

        if may_be_21635:

            if more_types_in_union_21636:
                # Runtime conditional SSA (line 684)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 685)
            # Processing the call arguments (line 685)
            
            # Call to get_major_locator(...): (line 685)
            # Processing the call keyword arguments (line 685)
            kwargs_21641 = {}
            # Getting the type of 'yaxis' (line 685)
            yaxis_21639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 32), 'yaxis', False)
            # Obtaining the member 'get_major_locator' of a type (line 685)
            get_major_locator_21640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 32), yaxis_21639, 'get_major_locator')
            # Calling get_major_locator(args, kwargs) (line 685)
            get_major_locator_call_result_21642 = invoke(stypy.reporting.localization.Localization(__file__, 685, 32), get_major_locator_21640, *[], **kwargs_21641)
            
            # Processing the call keyword arguments (line 685)
            kwargs_21643 = {}
            # Getting the type of 'locators' (line 685)
            locators_21637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'locators', False)
            # Obtaining the member 'append' of a type (line 685)
            append_21638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 16), locators_21637, 'append')
            # Calling append(args, kwargs) (line 685)
            append_call_result_21644 = invoke(stypy.reporting.localization.Localization(__file__, 685, 16), append_21638, *[get_major_locator_call_result_21642], **kwargs_21643)
            
            
            # Call to append(...): (line 686)
            # Processing the call arguments (line 686)
            
            # Call to get_minor_locator(...): (line 686)
            # Processing the call keyword arguments (line 686)
            kwargs_21649 = {}
            # Getting the type of 'yaxis' (line 686)
            yaxis_21647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 32), 'yaxis', False)
            # Obtaining the member 'get_minor_locator' of a type (line 686)
            get_minor_locator_21648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 32), yaxis_21647, 'get_minor_locator')
            # Calling get_minor_locator(args, kwargs) (line 686)
            get_minor_locator_call_result_21650 = invoke(stypy.reporting.localization.Localization(__file__, 686, 32), get_minor_locator_21648, *[], **kwargs_21649)
            
            # Processing the call keyword arguments (line 686)
            kwargs_21651 = {}
            # Getting the type of 'locators' (line 686)
            locators_21645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 16), 'locators', False)
            # Obtaining the member 'append' of a type (line 686)
            append_21646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 16), locators_21645, 'append')
            # Calling append(args, kwargs) (line 686)
            append_call_result_21652 = invoke(stypy.reporting.localization.Localization(__file__, 686, 16), append_21646, *[get_minor_locator_call_result_21650], **kwargs_21651)
            

            if more_types_in_union_21636:
                # SSA join for if statement (line 684)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 687)
        # Getting the type of 'zaxis' (line 687)
        zaxis_21653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 12), 'zaxis')
        # Getting the type of 'None' (line 687)
        None_21654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 28), 'None')
        
        (may_be_21655, more_types_in_union_21656) = may_not_be_none(zaxis_21653, None_21654)

        if may_be_21655:

            if more_types_in_union_21656:
                # Runtime conditional SSA (line 687)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 688)
            # Processing the call arguments (line 688)
            
            # Call to get_major_locator(...): (line 688)
            # Processing the call keyword arguments (line 688)
            kwargs_21661 = {}
            # Getting the type of 'zaxis' (line 688)
            zaxis_21659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 32), 'zaxis', False)
            # Obtaining the member 'get_major_locator' of a type (line 688)
            get_major_locator_21660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 32), zaxis_21659, 'get_major_locator')
            # Calling get_major_locator(args, kwargs) (line 688)
            get_major_locator_call_result_21662 = invoke(stypy.reporting.localization.Localization(__file__, 688, 32), get_major_locator_21660, *[], **kwargs_21661)
            
            # Processing the call keyword arguments (line 688)
            kwargs_21663 = {}
            # Getting the type of 'locators' (line 688)
            locators_21657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 16), 'locators', False)
            # Obtaining the member 'append' of a type (line 688)
            append_21658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 16), locators_21657, 'append')
            # Calling append(args, kwargs) (line 688)
            append_call_result_21664 = invoke(stypy.reporting.localization.Localization(__file__, 688, 16), append_21658, *[get_major_locator_call_result_21662], **kwargs_21663)
            
            
            # Call to append(...): (line 689)
            # Processing the call arguments (line 689)
            
            # Call to get_minor_locator(...): (line 689)
            # Processing the call keyword arguments (line 689)
            kwargs_21669 = {}
            # Getting the type of 'zaxis' (line 689)
            zaxis_21667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 32), 'zaxis', False)
            # Obtaining the member 'get_minor_locator' of a type (line 689)
            get_minor_locator_21668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 32), zaxis_21667, 'get_minor_locator')
            # Calling get_minor_locator(args, kwargs) (line 689)
            get_minor_locator_call_result_21670 = invoke(stypy.reporting.localization.Localization(__file__, 689, 32), get_minor_locator_21668, *[], **kwargs_21669)
            
            # Processing the call keyword arguments (line 689)
            kwargs_21671 = {}
            # Getting the type of 'locators' (line 689)
            locators_21665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'locators', False)
            # Obtaining the member 'append' of a type (line 689)
            append_21666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 16), locators_21665, 'append')
            # Calling append(args, kwargs) (line 689)
            append_call_result_21672 = invoke(stypy.reporting.localization.Localization(__file__, 689, 16), append_21666, *[get_minor_locator_call_result_21670], **kwargs_21671)
            

            if more_types_in_union_21656:
                # SSA join for if statement (line 687)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'locators' (line 691)
        locators_21673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 23), 'locators')
        # Testing the type of a for loop iterable (line 691)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 691, 12), locators_21673)
        # Getting the type of the for loop variable (line 691)
        for_loop_var_21674 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 691, 12), locators_21673)
        # Assigning a type to the variable 'loc' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 12), 'loc', for_loop_var_21674)
        # SSA begins for a for statement (line 691)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to refresh(...): (line 692)
        # Processing the call keyword arguments (line 692)
        kwargs_21677 = {}
        # Getting the type of 'loc' (line 692)
        loc_21675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'loc', False)
        # Obtaining the member 'refresh' of a type (line 692)
        refresh_21676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 16), loc_21675, 'refresh')
        # Calling refresh(args, kwargs) (line 692)
        refresh_call_result_21678 = invoke(stypy.reporting.localization.Localization(__file__, 692, 16), refresh_21676, *[], **kwargs_21677)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw_idle(...): (line 693)
        # Processing the call keyword arguments (line 693)
        kwargs_21683 = {}
        # Getting the type of 'self' (line 693)
        self_21679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 693)
        figure_21680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 8), self_21679, 'figure')
        # Obtaining the member 'canvas' of a type (line 693)
        canvas_21681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 8), figure_21680, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 693)
        draw_idle_21682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 8), canvas_21681, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 693)
        draw_idle_call_result_21684 = invoke(stypy.reporting.localization.Localization(__file__, 693, 8), draw_idle_21682, *[], **kwargs_21683)
        
        
        # ################# End of 'refresh_locators(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'refresh_locators' in the type store
        # Getting the type of 'stypy_return_type' (line 674)
        stypy_return_type_21685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'refresh_locators'
        return stypy_return_type_21685


    @norecursion
    def home(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'home'
        module_type_store = module_type_store.open_function_context('home', 695, 4, False)
        # Assigning a type to the variable 'self' (line 696)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolViewsPositions.home.__dict__.__setitem__('stypy_localization', localization)
        ToolViewsPositions.home.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolViewsPositions.home.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolViewsPositions.home.__dict__.__setitem__('stypy_function_name', 'ToolViewsPositions.home')
        ToolViewsPositions.home.__dict__.__setitem__('stypy_param_names_list', [])
        ToolViewsPositions.home.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolViewsPositions.home.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolViewsPositions.home.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolViewsPositions.home.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolViewsPositions.home.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolViewsPositions.home.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolViewsPositions.home', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'home', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'home(...)' code ##################

        str_21686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 8), 'str', 'Recall the first view and position from the stack')
        
        # Call to home(...): (line 697)
        # Processing the call keyword arguments (line 697)
        kwargs_21694 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 697)
        self_21687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 19), 'self', False)
        # Obtaining the member 'figure' of a type (line 697)
        figure_21688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 19), self_21687, 'figure')
        # Getting the type of 'self' (line 697)
        self_21689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'self', False)
        # Obtaining the member 'views' of a type (line 697)
        views_21690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), self_21689, 'views')
        # Obtaining the member '__getitem__' of a type (line 697)
        getitem___21691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), views_21690, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 697)
        subscript_call_result_21692 = invoke(stypy.reporting.localization.Localization(__file__, 697, 8), getitem___21691, figure_21688)
        
        # Obtaining the member 'home' of a type (line 697)
        home_21693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), subscript_call_result_21692, 'home')
        # Calling home(args, kwargs) (line 697)
        home_call_result_21695 = invoke(stypy.reporting.localization.Localization(__file__, 697, 8), home_21693, *[], **kwargs_21694)
        
        
        # Call to home(...): (line 698)
        # Processing the call keyword arguments (line 698)
        kwargs_21703 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 698)
        self_21696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 23), 'self', False)
        # Obtaining the member 'figure' of a type (line 698)
        figure_21697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 23), self_21696, 'figure')
        # Getting the type of 'self' (line 698)
        self_21698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'self', False)
        # Obtaining the member 'positions' of a type (line 698)
        positions_21699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), self_21698, 'positions')
        # Obtaining the member '__getitem__' of a type (line 698)
        getitem___21700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), positions_21699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 698)
        subscript_call_result_21701 = invoke(stypy.reporting.localization.Localization(__file__, 698, 8), getitem___21700, figure_21697)
        
        # Obtaining the member 'home' of a type (line 698)
        home_21702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), subscript_call_result_21701, 'home')
        # Calling home(args, kwargs) (line 698)
        home_call_result_21704 = invoke(stypy.reporting.localization.Localization(__file__, 698, 8), home_21702, *[], **kwargs_21703)
        
        
        # ################# End of 'home(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'home' in the type store
        # Getting the type of 'stypy_return_type' (line 695)
        stypy_return_type_21705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21705)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'home'
        return stypy_return_type_21705


    @norecursion
    def back(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'back'
        module_type_store = module_type_store.open_function_context('back', 700, 4, False)
        # Assigning a type to the variable 'self' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolViewsPositions.back.__dict__.__setitem__('stypy_localization', localization)
        ToolViewsPositions.back.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolViewsPositions.back.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolViewsPositions.back.__dict__.__setitem__('stypy_function_name', 'ToolViewsPositions.back')
        ToolViewsPositions.back.__dict__.__setitem__('stypy_param_names_list', [])
        ToolViewsPositions.back.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolViewsPositions.back.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolViewsPositions.back.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolViewsPositions.back.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolViewsPositions.back.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolViewsPositions.back.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolViewsPositions.back', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'back', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'back(...)' code ##################

        str_21706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 8), 'str', 'Back one step in the stack of views and positions')
        
        # Call to back(...): (line 702)
        # Processing the call keyword arguments (line 702)
        kwargs_21714 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 702)
        self_21707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 19), 'self', False)
        # Obtaining the member 'figure' of a type (line 702)
        figure_21708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 19), self_21707, 'figure')
        # Getting the type of 'self' (line 702)
        self_21709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'self', False)
        # Obtaining the member 'views' of a type (line 702)
        views_21710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 8), self_21709, 'views')
        # Obtaining the member '__getitem__' of a type (line 702)
        getitem___21711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 8), views_21710, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 702)
        subscript_call_result_21712 = invoke(stypy.reporting.localization.Localization(__file__, 702, 8), getitem___21711, figure_21708)
        
        # Obtaining the member 'back' of a type (line 702)
        back_21713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 8), subscript_call_result_21712, 'back')
        # Calling back(args, kwargs) (line 702)
        back_call_result_21715 = invoke(stypy.reporting.localization.Localization(__file__, 702, 8), back_21713, *[], **kwargs_21714)
        
        
        # Call to back(...): (line 703)
        # Processing the call keyword arguments (line 703)
        kwargs_21723 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 703)
        self_21716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 23), 'self', False)
        # Obtaining the member 'figure' of a type (line 703)
        figure_21717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 23), self_21716, 'figure')
        # Getting the type of 'self' (line 703)
        self_21718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'self', False)
        # Obtaining the member 'positions' of a type (line 703)
        positions_21719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 8), self_21718, 'positions')
        # Obtaining the member '__getitem__' of a type (line 703)
        getitem___21720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 8), positions_21719, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 703)
        subscript_call_result_21721 = invoke(stypy.reporting.localization.Localization(__file__, 703, 8), getitem___21720, figure_21717)
        
        # Obtaining the member 'back' of a type (line 703)
        back_21722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 8), subscript_call_result_21721, 'back')
        # Calling back(args, kwargs) (line 703)
        back_call_result_21724 = invoke(stypy.reporting.localization.Localization(__file__, 703, 8), back_21722, *[], **kwargs_21723)
        
        
        # ################# End of 'back(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'back' in the type store
        # Getting the type of 'stypy_return_type' (line 700)
        stypy_return_type_21725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21725)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'back'
        return stypy_return_type_21725


    @norecursion
    def forward(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'forward'
        module_type_store = module_type_store.open_function_context('forward', 705, 4, False)
        # Assigning a type to the variable 'self' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolViewsPositions.forward.__dict__.__setitem__('stypy_localization', localization)
        ToolViewsPositions.forward.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolViewsPositions.forward.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolViewsPositions.forward.__dict__.__setitem__('stypy_function_name', 'ToolViewsPositions.forward')
        ToolViewsPositions.forward.__dict__.__setitem__('stypy_param_names_list', [])
        ToolViewsPositions.forward.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolViewsPositions.forward.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolViewsPositions.forward.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolViewsPositions.forward.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolViewsPositions.forward.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolViewsPositions.forward.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolViewsPositions.forward', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'forward', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'forward(...)' code ##################

        str_21726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 8), 'str', 'Forward one step in the stack of views and positions')
        
        # Call to forward(...): (line 707)
        # Processing the call keyword arguments (line 707)
        kwargs_21734 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 707)
        self_21727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 19), 'self', False)
        # Obtaining the member 'figure' of a type (line 707)
        figure_21728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 19), self_21727, 'figure')
        # Getting the type of 'self' (line 707)
        self_21729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'self', False)
        # Obtaining the member 'views' of a type (line 707)
        views_21730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 8), self_21729, 'views')
        # Obtaining the member '__getitem__' of a type (line 707)
        getitem___21731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 8), views_21730, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 707)
        subscript_call_result_21732 = invoke(stypy.reporting.localization.Localization(__file__, 707, 8), getitem___21731, figure_21728)
        
        # Obtaining the member 'forward' of a type (line 707)
        forward_21733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 8), subscript_call_result_21732, 'forward')
        # Calling forward(args, kwargs) (line 707)
        forward_call_result_21735 = invoke(stypy.reporting.localization.Localization(__file__, 707, 8), forward_21733, *[], **kwargs_21734)
        
        
        # Call to forward(...): (line 708)
        # Processing the call keyword arguments (line 708)
        kwargs_21743 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 708)
        self_21736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 23), 'self', False)
        # Obtaining the member 'figure' of a type (line 708)
        figure_21737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 23), self_21736, 'figure')
        # Getting the type of 'self' (line 708)
        self_21738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'self', False)
        # Obtaining the member 'positions' of a type (line 708)
        positions_21739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 8), self_21738, 'positions')
        # Obtaining the member '__getitem__' of a type (line 708)
        getitem___21740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 8), positions_21739, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 708)
        subscript_call_result_21741 = invoke(stypy.reporting.localization.Localization(__file__, 708, 8), getitem___21740, figure_21737)
        
        # Obtaining the member 'forward' of a type (line 708)
        forward_21742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 8), subscript_call_result_21741, 'forward')
        # Calling forward(args, kwargs) (line 708)
        forward_call_result_21744 = invoke(stypy.reporting.localization.Localization(__file__, 708, 8), forward_21742, *[], **kwargs_21743)
        
        
        # ################# End of 'forward(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'forward' in the type store
        # Getting the type of 'stypy_return_type' (line 705)
        stypy_return_type_21745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21745)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'forward'
        return stypy_return_type_21745


# Assigning a type to the variable 'ToolViewsPositions' (line 559)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 0), 'ToolViewsPositions', ToolViewsPositions)
# Declaration of the 'ViewsPositionsBase' class
# Getting the type of 'ToolBase' (line 711)
ToolBase_21746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 25), 'ToolBase')

class ViewsPositionsBase(ToolBase_21746, ):
    str_21747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 4), 'str', 'Base class for `ToolHome`, `ToolBack` and `ToolForward`')
    
    # Assigning a Name to a Name (line 714):
    
    # Assigning a Name to a Name (line 714):

    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 716)
        None_21748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 42), 'None')
        defaults = [None_21748]
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 716, 4, False)
        # Assigning a type to the variable 'self' (line 717)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ViewsPositionsBase.trigger.__dict__.__setitem__('stypy_localization', localization)
        ViewsPositionsBase.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ViewsPositionsBase.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        ViewsPositionsBase.trigger.__dict__.__setitem__('stypy_function_name', 'ViewsPositionsBase.trigger')
        ViewsPositionsBase.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        ViewsPositionsBase.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        ViewsPositionsBase.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ViewsPositionsBase.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        ViewsPositionsBase.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        ViewsPositionsBase.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ViewsPositionsBase.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ViewsPositionsBase.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, ['sender', 'event', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        
        # Call to add_figure(...): (line 717)
        # Processing the call arguments (line 717)
        # Getting the type of 'self' (line 717)
        self_21756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 63), 'self', False)
        # Obtaining the member 'figure' of a type (line 717)
        figure_21757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 63), self_21756, 'figure')
        # Processing the call keyword arguments (line 717)
        kwargs_21758 = {}
        
        # Call to get_tool(...): (line 717)
        # Processing the call arguments (line 717)
        # Getting the type of '_views_positions' (line 717)
        _views_positions_21752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 34), '_views_positions', False)
        # Processing the call keyword arguments (line 717)
        kwargs_21753 = {}
        # Getting the type of 'self' (line 717)
        self_21749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 717)
        toolmanager_21750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 8), self_21749, 'toolmanager')
        # Obtaining the member 'get_tool' of a type (line 717)
        get_tool_21751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 8), toolmanager_21750, 'get_tool')
        # Calling get_tool(args, kwargs) (line 717)
        get_tool_call_result_21754 = invoke(stypy.reporting.localization.Localization(__file__, 717, 8), get_tool_21751, *[_views_positions_21752], **kwargs_21753)
        
        # Obtaining the member 'add_figure' of a type (line 717)
        add_figure_21755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 8), get_tool_call_result_21754, 'add_figure')
        # Calling add_figure(args, kwargs) (line 717)
        add_figure_call_result_21759 = invoke(stypy.reporting.localization.Localization(__file__, 717, 8), add_figure_21755, *[figure_21757], **kwargs_21758)
        
        
        # Call to (...): (line 718)
        # Processing the call keyword arguments (line 718)
        kwargs_21771 = {}
        
        # Call to getattr(...): (line 718)
        # Processing the call arguments (line 718)
        
        # Call to get_tool(...): (line 718)
        # Processing the call arguments (line 718)
        # Getting the type of '_views_positions' (line 718)
        _views_positions_21764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 42), '_views_positions', False)
        # Processing the call keyword arguments (line 718)
        kwargs_21765 = {}
        # Getting the type of 'self' (line 718)
        self_21761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 16), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 718)
        toolmanager_21762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 16), self_21761, 'toolmanager')
        # Obtaining the member 'get_tool' of a type (line 718)
        get_tool_21763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 16), toolmanager_21762, 'get_tool')
        # Calling get_tool(args, kwargs) (line 718)
        get_tool_call_result_21766 = invoke(stypy.reporting.localization.Localization(__file__, 718, 16), get_tool_21763, *[_views_positions_21764], **kwargs_21765)
        
        # Getting the type of 'self' (line 719)
        self_21767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 16), 'self', False)
        # Obtaining the member '_on_trigger' of a type (line 719)
        _on_trigger_21768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 16), self_21767, '_on_trigger')
        # Processing the call keyword arguments (line 718)
        kwargs_21769 = {}
        # Getting the type of 'getattr' (line 718)
        getattr_21760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'getattr', False)
        # Calling getattr(args, kwargs) (line 718)
        getattr_call_result_21770 = invoke(stypy.reporting.localization.Localization(__file__, 718, 8), getattr_21760, *[get_tool_call_result_21766, _on_trigger_21768], **kwargs_21769)
        
        # Calling (args, kwargs) (line 718)
        _call_result_21772 = invoke(stypy.reporting.localization.Localization(__file__, 718, 8), getattr_call_result_21770, *[], **kwargs_21771)
        
        
        # Call to update_view(...): (line 720)
        # Processing the call keyword arguments (line 720)
        kwargs_21780 = {}
        
        # Call to get_tool(...): (line 720)
        # Processing the call arguments (line 720)
        # Getting the type of '_views_positions' (line 720)
        _views_positions_21776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 34), '_views_positions', False)
        # Processing the call keyword arguments (line 720)
        kwargs_21777 = {}
        # Getting the type of 'self' (line 720)
        self_21773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 720)
        toolmanager_21774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 8), self_21773, 'toolmanager')
        # Obtaining the member 'get_tool' of a type (line 720)
        get_tool_21775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 8), toolmanager_21774, 'get_tool')
        # Calling get_tool(args, kwargs) (line 720)
        get_tool_call_result_21778 = invoke(stypy.reporting.localization.Localization(__file__, 720, 8), get_tool_21775, *[_views_positions_21776], **kwargs_21777)
        
        # Obtaining the member 'update_view' of a type (line 720)
        update_view_21779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 8), get_tool_call_result_21778, 'update_view')
        # Calling update_view(args, kwargs) (line 720)
        update_view_call_result_21781 = invoke(stypy.reporting.localization.Localization(__file__, 720, 8), update_view_21779, *[], **kwargs_21780)
        
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 716)
        stypy_return_type_21782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21782)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_21782


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 711, 0, False)
        # Assigning a type to the variable 'self' (line 712)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ViewsPositionsBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ViewsPositionsBase' (line 711)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 0), 'ViewsPositionsBase', ViewsPositionsBase)

# Assigning a Name to a Name (line 714):
# Getting the type of 'None' (line 714)
None_21783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 18), 'None')
# Getting the type of 'ViewsPositionsBase'
ViewsPositionsBase_21784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ViewsPositionsBase')
# Setting the type of the member '_on_trigger' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ViewsPositionsBase_21784, '_on_trigger', None_21783)
# Declaration of the 'ToolHome' class
# Getting the type of 'ViewsPositionsBase' (line 723)
ViewsPositionsBase_21785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 15), 'ViewsPositionsBase')

class ToolHome(ViewsPositionsBase_21785, ):
    str_21786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 4), 'str', 'Restore the original view lim')
    
    # Assigning a Str to a Name (line 726):
    
    # Assigning a Str to a Name (line 726):
    
    # Assigning a Str to a Name (line 727):
    
    # Assigning a Str to a Name (line 727):
    
    # Assigning a Subscript to a Name (line 728):
    
    # Assigning a Subscript to a Name (line 728):
    
    # Assigning a Str to a Name (line 729):
    
    # Assigning a Str to a Name (line 729):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 723, 0, False)
        # Assigning a type to the variable 'self' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolHome.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolHome' (line 723)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 0), 'ToolHome', ToolHome)

# Assigning a Str to a Name (line 726):
str_21787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 18), 'str', 'Reset original view')
# Getting the type of 'ToolHome'
ToolHome_21788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolHome')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolHome_21788, 'description', str_21787)

# Assigning a Str to a Name (line 727):
str_21789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 12), 'str', 'home.png')
# Getting the type of 'ToolHome'
ToolHome_21790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolHome')
# Setting the type of the member 'image' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolHome_21790, 'image', str_21789)

# Assigning a Subscript to a Name (line 728):

# Obtaining the type of the subscript
str_21791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 30), 'str', 'keymap.home')
# Getting the type of 'rcParams' (line 728)
rcParams_21792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 728)
getitem___21793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 21), rcParams_21792, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 728)
subscript_call_result_21794 = invoke(stypy.reporting.localization.Localization(__file__, 728, 21), getitem___21793, str_21791)

# Getting the type of 'ToolHome'
ToolHome_21795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolHome')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolHome_21795, 'default_keymap', subscript_call_result_21794)

# Assigning a Str to a Name (line 729):
str_21796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 18), 'str', 'home')
# Getting the type of 'ToolHome'
ToolHome_21797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolHome')
# Setting the type of the member '_on_trigger' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolHome_21797, '_on_trigger', str_21796)
# Declaration of the 'ToolBack' class
# Getting the type of 'ViewsPositionsBase' (line 732)
ViewsPositionsBase_21798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 15), 'ViewsPositionsBase')

class ToolBack(ViewsPositionsBase_21798, ):
    str_21799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 4), 'str', 'Move back up the view lim stack')
    
    # Assigning a Str to a Name (line 735):
    
    # Assigning a Str to a Name (line 735):
    
    # Assigning a Str to a Name (line 736):
    
    # Assigning a Str to a Name (line 736):
    
    # Assigning a Subscript to a Name (line 737):
    
    # Assigning a Subscript to a Name (line 737):
    
    # Assigning a Str to a Name (line 738):
    
    # Assigning a Str to a Name (line 738):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 732, 0, False)
        # Assigning a type to the variable 'self' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolBack.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolBack' (line 732)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 0), 'ToolBack', ToolBack)

# Assigning a Str to a Name (line 735):
str_21800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 18), 'str', 'Back to previous view')
# Getting the type of 'ToolBack'
ToolBack_21801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolBack')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolBack_21801, 'description', str_21800)

# Assigning a Str to a Name (line 736):
str_21802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 12), 'str', 'back.png')
# Getting the type of 'ToolBack'
ToolBack_21803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolBack')
# Setting the type of the member 'image' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolBack_21803, 'image', str_21802)

# Assigning a Subscript to a Name (line 737):

# Obtaining the type of the subscript
str_21804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 30), 'str', 'keymap.back')
# Getting the type of 'rcParams' (line 737)
rcParams_21805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 737)
getitem___21806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 21), rcParams_21805, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 737)
subscript_call_result_21807 = invoke(stypy.reporting.localization.Localization(__file__, 737, 21), getitem___21806, str_21804)

# Getting the type of 'ToolBack'
ToolBack_21808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolBack')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolBack_21808, 'default_keymap', subscript_call_result_21807)

# Assigning a Str to a Name (line 738):
str_21809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 18), 'str', 'back')
# Getting the type of 'ToolBack'
ToolBack_21810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolBack')
# Setting the type of the member '_on_trigger' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolBack_21810, '_on_trigger', str_21809)
# Declaration of the 'ToolForward' class
# Getting the type of 'ViewsPositionsBase' (line 741)
ViewsPositionsBase_21811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 18), 'ViewsPositionsBase')

class ToolForward(ViewsPositionsBase_21811, ):
    str_21812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 4), 'str', 'Move forward in the view lim stack')
    
    # Assigning a Str to a Name (line 744):
    
    # Assigning a Str to a Name (line 744):
    
    # Assigning a Str to a Name (line 745):
    
    # Assigning a Str to a Name (line 745):
    
    # Assigning a Subscript to a Name (line 746):
    
    # Assigning a Subscript to a Name (line 746):
    
    # Assigning a Str to a Name (line 747):
    
    # Assigning a Str to a Name (line 747):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 741, 0, False)
        # Assigning a type to the variable 'self' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolForward.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ToolForward' (line 741)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 0), 'ToolForward', ToolForward)

# Assigning a Str to a Name (line 744):
str_21813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 18), 'str', 'Forward to next view')
# Getting the type of 'ToolForward'
ToolForward_21814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolForward')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolForward_21814, 'description', str_21813)

# Assigning a Str to a Name (line 745):
str_21815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 12), 'str', 'forward.png')
# Getting the type of 'ToolForward'
ToolForward_21816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolForward')
# Setting the type of the member 'image' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolForward_21816, 'image', str_21815)

# Assigning a Subscript to a Name (line 746):

# Obtaining the type of the subscript
str_21817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 30), 'str', 'keymap.forward')
# Getting the type of 'rcParams' (line 746)
rcParams_21818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 746)
getitem___21819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 21), rcParams_21818, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 746)
subscript_call_result_21820 = invoke(stypy.reporting.localization.Localization(__file__, 746, 21), getitem___21819, str_21817)

# Getting the type of 'ToolForward'
ToolForward_21821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolForward')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolForward_21821, 'default_keymap', subscript_call_result_21820)

# Assigning a Str to a Name (line 747):
str_21822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 18), 'str', 'forward')
# Getting the type of 'ToolForward'
ToolForward_21823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolForward')
# Setting the type of the member '_on_trigger' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolForward_21823, '_on_trigger', str_21822)
# Declaration of the 'ConfigureSubplotsBase' class
# Getting the type of 'ToolBase' (line 750)
ToolBase_21824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 28), 'ToolBase')

class ConfigureSubplotsBase(ToolBase_21824, ):
    str_21825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 4), 'str', 'Base tool for the configuration of subplots')
    
    # Assigning a Str to a Name (line 753):
    
    # Assigning a Str to a Name (line 753):
    
    # Assigning a Str to a Name (line 754):
    
    # Assigning a Str to a Name (line 754):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 750, 0, False)
        # Assigning a type to the variable 'self' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigureSubplotsBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ConfigureSubplotsBase' (line 750)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 0), 'ConfigureSubplotsBase', ConfigureSubplotsBase)

# Assigning a Str to a Name (line 753):
str_21826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 18), 'str', 'Configure subplots')
# Getting the type of 'ConfigureSubplotsBase'
ConfigureSubplotsBase_21827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ConfigureSubplotsBase')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ConfigureSubplotsBase_21827, 'description', str_21826)

# Assigning a Str to a Name (line 754):
str_21828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 12), 'str', 'subplots.png')
# Getting the type of 'ConfigureSubplotsBase'
ConfigureSubplotsBase_21829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ConfigureSubplotsBase')
# Setting the type of the member 'image' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ConfigureSubplotsBase_21829, 'image', str_21828)
# Declaration of the 'SaveFigureBase' class
# Getting the type of 'ToolBase' (line 757)
ToolBase_21830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 21), 'ToolBase')

class SaveFigureBase(ToolBase_21830, ):
    str_21831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 4), 'str', 'Base tool for figure saving')
    
    # Assigning a Str to a Name (line 760):
    
    # Assigning a Str to a Name (line 760):
    
    # Assigning a Str to a Name (line 761):
    
    # Assigning a Str to a Name (line 761):
    
    # Assigning a Subscript to a Name (line 762):
    
    # Assigning a Subscript to a Name (line 762):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 757, 0, False)
        # Assigning a type to the variable 'self' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SaveFigureBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SaveFigureBase' (line 757)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 0), 'SaveFigureBase', SaveFigureBase)

# Assigning a Str to a Name (line 760):
str_21832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 18), 'str', 'Save the figure')
# Getting the type of 'SaveFigureBase'
SaveFigureBase_21833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SaveFigureBase')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SaveFigureBase_21833, 'description', str_21832)

# Assigning a Str to a Name (line 761):
str_21834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 12), 'str', 'filesave.png')
# Getting the type of 'SaveFigureBase'
SaveFigureBase_21835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SaveFigureBase')
# Setting the type of the member 'image' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SaveFigureBase_21835, 'image', str_21834)

# Assigning a Subscript to a Name (line 762):

# Obtaining the type of the subscript
str_21836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 30), 'str', 'keymap.save')
# Getting the type of 'rcParams' (line 762)
rcParams_21837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 762)
getitem___21838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 21), rcParams_21837, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 762)
subscript_call_result_21839 = invoke(stypy.reporting.localization.Localization(__file__, 762, 21), getitem___21838, str_21836)

# Getting the type of 'SaveFigureBase'
SaveFigureBase_21840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SaveFigureBase')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SaveFigureBase_21840, 'default_keymap', subscript_call_result_21839)
# Declaration of the 'ZoomPanBase' class
# Getting the type of 'ToolToggleBase' (line 765)
ToolToggleBase_21841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 18), 'ToolToggleBase')

class ZoomPanBase(ToolToggleBase_21841, ):
    str_21842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 4), 'str', 'Base class for `ToolZoom` and `ToolPan`')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 767, 4, False)
        # Assigning a type to the variable 'self' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ZoomPanBase.__init__', [], 'args', None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 768)
        # Processing the call arguments (line 768)
        # Getting the type of 'self' (line 768)
        self_21845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 32), 'self', False)
        # Getting the type of 'args' (line 768)
        args_21846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 39), 'args', False)
        # Processing the call keyword arguments (line 768)
        kwargs_21847 = {}
        # Getting the type of 'ToolToggleBase' (line 768)
        ToolToggleBase_21843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'ToolToggleBase', False)
        # Obtaining the member '__init__' of a type (line 768)
        init___21844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 8), ToolToggleBase_21843, '__init__')
        # Calling __init__(args, kwargs) (line 768)
        init___call_result_21848 = invoke(stypy.reporting.localization.Localization(__file__, 768, 8), init___21844, *[self_21845, args_21846], **kwargs_21847)
        
        
        # Assigning a Name to a Attribute (line 769):
        
        # Assigning a Name to a Attribute (line 769):
        
        # Assigning a Name to a Attribute (line 769):
        # Getting the type of 'None' (line 769)
        None_21849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 31), 'None')
        # Getting the type of 'self' (line 769)
        self_21850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 8), 'self')
        # Setting the type of the member '_button_pressed' of a type (line 769)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 8), self_21850, '_button_pressed', None_21849)
        
        # Assigning a Name to a Attribute (line 770):
        
        # Assigning a Name to a Attribute (line 770):
        
        # Assigning a Name to a Attribute (line 770):
        # Getting the type of 'None' (line 770)
        None_21851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 24), 'None')
        # Getting the type of 'self' (line 770)
        self_21852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'self')
        # Setting the type of the member '_xypress' of a type (line 770)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 8), self_21852, '_xypress', None_21851)
        
        # Assigning a Name to a Attribute (line 771):
        
        # Assigning a Name to a Attribute (line 771):
        
        # Assigning a Name to a Attribute (line 771):
        # Getting the type of 'None' (line 771)
        None_21853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 24), 'None')
        # Getting the type of 'self' (line 771)
        self_21854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 8), 'self')
        # Setting the type of the member '_idPress' of a type (line 771)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 8), self_21854, '_idPress', None_21853)
        
        # Assigning a Name to a Attribute (line 772):
        
        # Assigning a Name to a Attribute (line 772):
        
        # Assigning a Name to a Attribute (line 772):
        # Getting the type of 'None' (line 772)
        None_21855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 26), 'None')
        # Getting the type of 'self' (line 772)
        self_21856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 8), 'self')
        # Setting the type of the member '_idRelease' of a type (line 772)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 8), self_21856, '_idRelease', None_21855)
        
        # Assigning a Name to a Attribute (line 773):
        
        # Assigning a Name to a Attribute (line 773):
        
        # Assigning a Name to a Attribute (line 773):
        # Getting the type of 'None' (line 773)
        None_21857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 25), 'None')
        # Getting the type of 'self' (line 773)
        self_21858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'self')
        # Setting the type of the member '_idScroll' of a type (line 773)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 8), self_21858, '_idScroll', None_21857)
        
        # Assigning a Num to a Attribute (line 774):
        
        # Assigning a Num to a Attribute (line 774):
        
        # Assigning a Num to a Attribute (line 774):
        float_21859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 26), 'float')
        # Getting the type of 'self' (line 774)
        self_21860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 8), 'self')
        # Setting the type of the member 'base_scale' of a type (line 774)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 8), self_21860, 'base_scale', float_21859)
        
        # Assigning a Num to a Attribute (line 775):
        
        # Assigning a Num to a Attribute (line 775):
        
        # Assigning a Num to a Attribute (line 775):
        float_21861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 28), 'float')
        # Getting the type of 'self' (line 775)
        self_21862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'self')
        # Setting the type of the member 'scrollthresh' of a type (line 775)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 8), self_21862, 'scrollthresh', float_21861)
        
        # Assigning a BinOp to a Attribute (line 776):
        
        # Assigning a BinOp to a Attribute (line 776):
        
        # Assigning a BinOp to a Attribute (line 776):
        
        # Call to time(...): (line 776)
        # Processing the call keyword arguments (line 776)
        kwargs_21865 = {}
        # Getting the type of 'time' (line 776)
        time_21863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 26), 'time', False)
        # Obtaining the member 'time' of a type (line 776)
        time_21864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 26), time_21863, 'time')
        # Calling time(args, kwargs) (line 776)
        time_call_result_21866 = invoke(stypy.reporting.localization.Localization(__file__, 776, 26), time_21864, *[], **kwargs_21865)
        
        # Getting the type of 'self' (line 776)
        self_21867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 38), 'self')
        # Obtaining the member 'scrollthresh' of a type (line 776)
        scrollthresh_21868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 38), self_21867, 'scrollthresh')
        # Applying the binary operator '-' (line 776)
        result_sub_21869 = python_operator(stypy.reporting.localization.Localization(__file__, 776, 26), '-', time_call_result_21866, scrollthresh_21868)
        
        # Getting the type of 'self' (line 776)
        self_21870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 8), 'self')
        # Setting the type of the member 'lastscroll' of a type (line 776)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 8), self_21870, 'lastscroll', result_sub_21869)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def enable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'enable'
        module_type_store = module_type_store.open_function_context('enable', 778, 4, False)
        # Assigning a type to the variable 'self' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ZoomPanBase.enable.__dict__.__setitem__('stypy_localization', localization)
        ZoomPanBase.enable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ZoomPanBase.enable.__dict__.__setitem__('stypy_type_store', module_type_store)
        ZoomPanBase.enable.__dict__.__setitem__('stypy_function_name', 'ZoomPanBase.enable')
        ZoomPanBase.enable.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ZoomPanBase.enable.__dict__.__setitem__('stypy_varargs_param_name', None)
        ZoomPanBase.enable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ZoomPanBase.enable.__dict__.__setitem__('stypy_call_defaults', defaults)
        ZoomPanBase.enable.__dict__.__setitem__('stypy_call_varargs', varargs)
        ZoomPanBase.enable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ZoomPanBase.enable.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ZoomPanBase.enable', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'enable', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'enable(...)' code ##################

        str_21871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 8), 'str', 'Connect press/release events and lock the canvas')
        
        # Call to widgetlock(...): (line 780)
        # Processing the call arguments (line 780)
        # Getting the type of 'self' (line 780)
        self_21876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 38), 'self', False)
        # Processing the call keyword arguments (line 780)
        kwargs_21877 = {}
        # Getting the type of 'self' (line 780)
        self_21872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 780)
        figure_21873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 8), self_21872, 'figure')
        # Obtaining the member 'canvas' of a type (line 780)
        canvas_21874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 8), figure_21873, 'canvas')
        # Obtaining the member 'widgetlock' of a type (line 780)
        widgetlock_21875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 8), canvas_21874, 'widgetlock')
        # Calling widgetlock(args, kwargs) (line 780)
        widgetlock_call_result_21878 = invoke(stypy.reporting.localization.Localization(__file__, 780, 8), widgetlock_21875, *[self_21876], **kwargs_21877)
        
        
        # Assigning a Call to a Attribute (line 781):
        
        # Assigning a Call to a Attribute (line 781):
        
        # Assigning a Call to a Attribute (line 781):
        
        # Call to mpl_connect(...): (line 781)
        # Processing the call arguments (line 781)
        str_21883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 12), 'str', 'button_press_event')
        # Getting the type of 'self' (line 782)
        self_21884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 34), 'self', False)
        # Obtaining the member '_press' of a type (line 782)
        _press_21885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 34), self_21884, '_press')
        # Processing the call keyword arguments (line 781)
        kwargs_21886 = {}
        # Getting the type of 'self' (line 781)
        self_21879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 24), 'self', False)
        # Obtaining the member 'figure' of a type (line 781)
        figure_21880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 24), self_21879, 'figure')
        # Obtaining the member 'canvas' of a type (line 781)
        canvas_21881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 24), figure_21880, 'canvas')
        # Obtaining the member 'mpl_connect' of a type (line 781)
        mpl_connect_21882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 24), canvas_21881, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 781)
        mpl_connect_call_result_21887 = invoke(stypy.reporting.localization.Localization(__file__, 781, 24), mpl_connect_21882, *[str_21883, _press_21885], **kwargs_21886)
        
        # Getting the type of 'self' (line 781)
        self_21888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 8), 'self')
        # Setting the type of the member '_idPress' of a type (line 781)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 8), self_21888, '_idPress', mpl_connect_call_result_21887)
        
        # Assigning a Call to a Attribute (line 783):
        
        # Assigning a Call to a Attribute (line 783):
        
        # Assigning a Call to a Attribute (line 783):
        
        # Call to mpl_connect(...): (line 783)
        # Processing the call arguments (line 783)
        str_21893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 12), 'str', 'button_release_event')
        # Getting the type of 'self' (line 784)
        self_21894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 36), 'self', False)
        # Obtaining the member '_release' of a type (line 784)
        _release_21895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 36), self_21894, '_release')
        # Processing the call keyword arguments (line 783)
        kwargs_21896 = {}
        # Getting the type of 'self' (line 783)
        self_21889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 26), 'self', False)
        # Obtaining the member 'figure' of a type (line 783)
        figure_21890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 26), self_21889, 'figure')
        # Obtaining the member 'canvas' of a type (line 783)
        canvas_21891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 26), figure_21890, 'canvas')
        # Obtaining the member 'mpl_connect' of a type (line 783)
        mpl_connect_21892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 26), canvas_21891, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 783)
        mpl_connect_call_result_21897 = invoke(stypy.reporting.localization.Localization(__file__, 783, 26), mpl_connect_21892, *[str_21893, _release_21895], **kwargs_21896)
        
        # Getting the type of 'self' (line 783)
        self_21898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 8), 'self')
        # Setting the type of the member '_idRelease' of a type (line 783)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 8), self_21898, '_idRelease', mpl_connect_call_result_21897)
        
        # Assigning a Call to a Attribute (line 785):
        
        # Assigning a Call to a Attribute (line 785):
        
        # Assigning a Call to a Attribute (line 785):
        
        # Call to mpl_connect(...): (line 785)
        # Processing the call arguments (line 785)
        str_21903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 12), 'str', 'scroll_event')
        # Getting the type of 'self' (line 786)
        self_21904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 28), 'self', False)
        # Obtaining the member 'scroll_zoom' of a type (line 786)
        scroll_zoom_21905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 28), self_21904, 'scroll_zoom')
        # Processing the call keyword arguments (line 785)
        kwargs_21906 = {}
        # Getting the type of 'self' (line 785)
        self_21899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 25), 'self', False)
        # Obtaining the member 'figure' of a type (line 785)
        figure_21900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 25), self_21899, 'figure')
        # Obtaining the member 'canvas' of a type (line 785)
        canvas_21901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 25), figure_21900, 'canvas')
        # Obtaining the member 'mpl_connect' of a type (line 785)
        mpl_connect_21902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 25), canvas_21901, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 785)
        mpl_connect_call_result_21907 = invoke(stypy.reporting.localization.Localization(__file__, 785, 25), mpl_connect_21902, *[str_21903, scroll_zoom_21905], **kwargs_21906)
        
        # Getting the type of 'self' (line 785)
        self_21908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'self')
        # Setting the type of the member '_idScroll' of a type (line 785)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 8), self_21908, '_idScroll', mpl_connect_call_result_21907)
        
        # ################# End of 'enable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'enable' in the type store
        # Getting the type of 'stypy_return_type' (line 778)
        stypy_return_type_21909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21909)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'enable'
        return stypy_return_type_21909


    @norecursion
    def disable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'disable'
        module_type_store = module_type_store.open_function_context('disable', 788, 4, False)
        # Assigning a type to the variable 'self' (line 789)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ZoomPanBase.disable.__dict__.__setitem__('stypy_localization', localization)
        ZoomPanBase.disable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ZoomPanBase.disable.__dict__.__setitem__('stypy_type_store', module_type_store)
        ZoomPanBase.disable.__dict__.__setitem__('stypy_function_name', 'ZoomPanBase.disable')
        ZoomPanBase.disable.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ZoomPanBase.disable.__dict__.__setitem__('stypy_varargs_param_name', None)
        ZoomPanBase.disable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ZoomPanBase.disable.__dict__.__setitem__('stypy_call_defaults', defaults)
        ZoomPanBase.disable.__dict__.__setitem__('stypy_call_varargs', varargs)
        ZoomPanBase.disable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ZoomPanBase.disable.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ZoomPanBase.disable', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'disable', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'disable(...)' code ##################

        str_21910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 8), 'str', 'Release the canvas and disconnect press/release events')
        
        # Call to _cancel_action(...): (line 790)
        # Processing the call keyword arguments (line 790)
        kwargs_21913 = {}
        # Getting the type of 'self' (line 790)
        self_21911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'self', False)
        # Obtaining the member '_cancel_action' of a type (line 790)
        _cancel_action_21912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 8), self_21911, '_cancel_action')
        # Calling _cancel_action(args, kwargs) (line 790)
        _cancel_action_call_result_21914 = invoke(stypy.reporting.localization.Localization(__file__, 790, 8), _cancel_action_21912, *[], **kwargs_21913)
        
        
        # Call to release(...): (line 791)
        # Processing the call arguments (line 791)
        # Getting the type of 'self' (line 791)
        self_21920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 46), 'self', False)
        # Processing the call keyword arguments (line 791)
        kwargs_21921 = {}
        # Getting the type of 'self' (line 791)
        self_21915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 791)
        figure_21916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 8), self_21915, 'figure')
        # Obtaining the member 'canvas' of a type (line 791)
        canvas_21917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 8), figure_21916, 'canvas')
        # Obtaining the member 'widgetlock' of a type (line 791)
        widgetlock_21918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 8), canvas_21917, 'widgetlock')
        # Obtaining the member 'release' of a type (line 791)
        release_21919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 8), widgetlock_21918, 'release')
        # Calling release(args, kwargs) (line 791)
        release_call_result_21922 = invoke(stypy.reporting.localization.Localization(__file__, 791, 8), release_21919, *[self_21920], **kwargs_21921)
        
        
        # Call to mpl_disconnect(...): (line 792)
        # Processing the call arguments (line 792)
        # Getting the type of 'self' (line 792)
        self_21927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 42), 'self', False)
        # Obtaining the member '_idPress' of a type (line 792)
        _idPress_21928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 42), self_21927, '_idPress')
        # Processing the call keyword arguments (line 792)
        kwargs_21929 = {}
        # Getting the type of 'self' (line 792)
        self_21923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 792)
        figure_21924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 8), self_21923, 'figure')
        # Obtaining the member 'canvas' of a type (line 792)
        canvas_21925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 8), figure_21924, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 792)
        mpl_disconnect_21926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 8), canvas_21925, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 792)
        mpl_disconnect_call_result_21930 = invoke(stypy.reporting.localization.Localization(__file__, 792, 8), mpl_disconnect_21926, *[_idPress_21928], **kwargs_21929)
        
        
        # Call to mpl_disconnect(...): (line 793)
        # Processing the call arguments (line 793)
        # Getting the type of 'self' (line 793)
        self_21935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 42), 'self', False)
        # Obtaining the member '_idRelease' of a type (line 793)
        _idRelease_21936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 42), self_21935, '_idRelease')
        # Processing the call keyword arguments (line 793)
        kwargs_21937 = {}
        # Getting the type of 'self' (line 793)
        self_21931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 793)
        figure_21932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 8), self_21931, 'figure')
        # Obtaining the member 'canvas' of a type (line 793)
        canvas_21933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 8), figure_21932, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 793)
        mpl_disconnect_21934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 8), canvas_21933, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 793)
        mpl_disconnect_call_result_21938 = invoke(stypy.reporting.localization.Localization(__file__, 793, 8), mpl_disconnect_21934, *[_idRelease_21936], **kwargs_21937)
        
        
        # Call to mpl_disconnect(...): (line 794)
        # Processing the call arguments (line 794)
        # Getting the type of 'self' (line 794)
        self_21943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 42), 'self', False)
        # Obtaining the member '_idScroll' of a type (line 794)
        _idScroll_21944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 42), self_21943, '_idScroll')
        # Processing the call keyword arguments (line 794)
        kwargs_21945 = {}
        # Getting the type of 'self' (line 794)
        self_21939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 794)
        figure_21940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 8), self_21939, 'figure')
        # Obtaining the member 'canvas' of a type (line 794)
        canvas_21941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 8), figure_21940, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 794)
        mpl_disconnect_21942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 8), canvas_21941, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 794)
        mpl_disconnect_call_result_21946 = invoke(stypy.reporting.localization.Localization(__file__, 794, 8), mpl_disconnect_21942, *[_idScroll_21944], **kwargs_21945)
        
        
        # ################# End of 'disable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'disable' in the type store
        # Getting the type of 'stypy_return_type' (line 788)
        stypy_return_type_21947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21947)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'disable'
        return stypy_return_type_21947


    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 796)
        None_21948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 42), 'None')
        defaults = [None_21948]
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 796, 4, False)
        # Assigning a type to the variable 'self' (line 797)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ZoomPanBase.trigger.__dict__.__setitem__('stypy_localization', localization)
        ZoomPanBase.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ZoomPanBase.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        ZoomPanBase.trigger.__dict__.__setitem__('stypy_function_name', 'ZoomPanBase.trigger')
        ZoomPanBase.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        ZoomPanBase.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        ZoomPanBase.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ZoomPanBase.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        ZoomPanBase.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        ZoomPanBase.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ZoomPanBase.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ZoomPanBase.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, ['sender', 'event', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        
        # Call to add_figure(...): (line 797)
        # Processing the call arguments (line 797)
        # Getting the type of 'self' (line 797)
        self_21956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 63), 'self', False)
        # Obtaining the member 'figure' of a type (line 797)
        figure_21957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 63), self_21956, 'figure')
        # Processing the call keyword arguments (line 797)
        kwargs_21958 = {}
        
        # Call to get_tool(...): (line 797)
        # Processing the call arguments (line 797)
        # Getting the type of '_views_positions' (line 797)
        _views_positions_21952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 34), '_views_positions', False)
        # Processing the call keyword arguments (line 797)
        kwargs_21953 = {}
        # Getting the type of 'self' (line 797)
        self_21949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 797)
        toolmanager_21950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 8), self_21949, 'toolmanager')
        # Obtaining the member 'get_tool' of a type (line 797)
        get_tool_21951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 8), toolmanager_21950, 'get_tool')
        # Calling get_tool(args, kwargs) (line 797)
        get_tool_call_result_21954 = invoke(stypy.reporting.localization.Localization(__file__, 797, 8), get_tool_21951, *[_views_positions_21952], **kwargs_21953)
        
        # Obtaining the member 'add_figure' of a type (line 797)
        add_figure_21955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 8), get_tool_call_result_21954, 'add_figure')
        # Calling add_figure(args, kwargs) (line 797)
        add_figure_call_result_21959 = invoke(stypy.reporting.localization.Localization(__file__, 797, 8), add_figure_21955, *[figure_21957], **kwargs_21958)
        
        
        # Call to trigger(...): (line 798)
        # Processing the call arguments (line 798)
        # Getting the type of 'self' (line 798)
        self_21962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 31), 'self', False)
        # Getting the type of 'sender' (line 798)
        sender_21963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 37), 'sender', False)
        # Getting the type of 'event' (line 798)
        event_21964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 45), 'event', False)
        # Getting the type of 'data' (line 798)
        data_21965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 52), 'data', False)
        # Processing the call keyword arguments (line 798)
        kwargs_21966 = {}
        # Getting the type of 'ToolToggleBase' (line 798)
        ToolToggleBase_21960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 8), 'ToolToggleBase', False)
        # Obtaining the member 'trigger' of a type (line 798)
        trigger_21961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 8), ToolToggleBase_21960, 'trigger')
        # Calling trigger(args, kwargs) (line 798)
        trigger_call_result_21967 = invoke(stypy.reporting.localization.Localization(__file__, 798, 8), trigger_21961, *[self_21962, sender_21963, event_21964, data_21965], **kwargs_21966)
        
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 796)
        stypy_return_type_21968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21968)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_21968


    @norecursion
    def scroll_zoom(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'scroll_zoom'
        module_type_store = module_type_store.open_function_context('scroll_zoom', 800, 4, False)
        # Assigning a type to the variable 'self' (line 801)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ZoomPanBase.scroll_zoom.__dict__.__setitem__('stypy_localization', localization)
        ZoomPanBase.scroll_zoom.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ZoomPanBase.scroll_zoom.__dict__.__setitem__('stypy_type_store', module_type_store)
        ZoomPanBase.scroll_zoom.__dict__.__setitem__('stypy_function_name', 'ZoomPanBase.scroll_zoom')
        ZoomPanBase.scroll_zoom.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ZoomPanBase.scroll_zoom.__dict__.__setitem__('stypy_varargs_param_name', None)
        ZoomPanBase.scroll_zoom.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ZoomPanBase.scroll_zoom.__dict__.__setitem__('stypy_call_defaults', defaults)
        ZoomPanBase.scroll_zoom.__dict__.__setitem__('stypy_call_varargs', varargs)
        ZoomPanBase.scroll_zoom.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ZoomPanBase.scroll_zoom.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ZoomPanBase.scroll_zoom', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'scroll_zoom', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'scroll_zoom(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 802)
        # Getting the type of 'event' (line 802)
        event_21969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 11), 'event')
        # Obtaining the member 'inaxes' of a type (line 802)
        inaxes_21970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 11), event_21969, 'inaxes')
        # Getting the type of 'None' (line 802)
        None_21971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 27), 'None')
        
        (may_be_21972, more_types_in_union_21973) = may_be_none(inaxes_21970, None_21971)

        if may_be_21972:

            if more_types_in_union_21973:
                # Runtime conditional SSA (line 802)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 803)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_21973:
                # SSA join for if statement (line 802)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'event' (line 805)
        event_21974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 11), 'event')
        # Obtaining the member 'button' of a type (line 805)
        button_21975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 11), event_21974, 'button')
        str_21976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 27), 'str', 'up')
        # Applying the binary operator '==' (line 805)
        result_eq_21977 = python_operator(stypy.reporting.localization.Localization(__file__, 805, 11), '==', button_21975, str_21976)
        
        # Testing the type of an if condition (line 805)
        if_condition_21978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 805, 8), result_eq_21977)
        # Assigning a type to the variable 'if_condition_21978' (line 805)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 8), 'if_condition_21978', if_condition_21978)
        # SSA begins for if statement (line 805)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 807):
        
        # Assigning a Attribute to a Name (line 807):
        
        # Assigning a Attribute to a Name (line 807):
        # Getting the type of 'self' (line 807)
        self_21979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 18), 'self')
        # Obtaining the member 'base_scale' of a type (line 807)
        base_scale_21980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 18), self_21979, 'base_scale')
        # Assigning a type to the variable 'scl' (line 807)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 12), 'scl', base_scale_21980)
        # SSA branch for the else part of an if statement (line 805)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'event' (line 808)
        event_21981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 13), 'event')
        # Obtaining the member 'button' of a type (line 808)
        button_21982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 13), event_21981, 'button')
        str_21983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 29), 'str', 'down')
        # Applying the binary operator '==' (line 808)
        result_eq_21984 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 13), '==', button_21982, str_21983)
        
        # Testing the type of an if condition (line 808)
        if_condition_21985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 808, 13), result_eq_21984)
        # Assigning a type to the variable 'if_condition_21985' (line 808)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 13), 'if_condition_21985', if_condition_21985)
        # SSA begins for if statement (line 808)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 810):
        
        # Assigning a BinOp to a Name (line 810):
        
        # Assigning a BinOp to a Name (line 810):
        int_21986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 18), 'int')
        # Getting the type of 'self' (line 810)
        self_21987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 20), 'self')
        # Obtaining the member 'base_scale' of a type (line 810)
        base_scale_21988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 20), self_21987, 'base_scale')
        # Applying the binary operator 'div' (line 810)
        result_div_21989 = python_operator(stypy.reporting.localization.Localization(__file__, 810, 18), 'div', int_21986, base_scale_21988)
        
        # Assigning a type to the variable 'scl' (line 810)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 12), 'scl', result_div_21989)
        # SSA branch for the else part of an if statement (line 808)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 813):
        
        # Assigning a Num to a Name (line 813):
        
        # Assigning a Num to a Name (line 813):
        int_21990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 18), 'int')
        # Assigning a type to the variable 'scl' (line 813)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 12), 'scl', int_21990)
        # SSA join for if statement (line 808)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 805)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 815):
        
        # Assigning a Attribute to a Name (line 815):
        
        # Assigning a Attribute to a Name (line 815):
        # Getting the type of 'event' (line 815)
        event_21991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 13), 'event')
        # Obtaining the member 'inaxes' of a type (line 815)
        inaxes_21992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 13), event_21991, 'inaxes')
        # Assigning a type to the variable 'ax' (line 815)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'ax', inaxes_21992)
        
        # Call to _set_view_from_bbox(...): (line 816)
        # Processing the call arguments (line 816)
        
        # Obtaining an instance of the builtin type 'list' (line 816)
        list_21995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 816)
        # Adding element type (line 816)
        # Getting the type of 'event' (line 816)
        event_21996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 32), 'event', False)
        # Obtaining the member 'x' of a type (line 816)
        x_21997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 32), event_21996, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 31), list_21995, x_21997)
        # Adding element type (line 816)
        # Getting the type of 'event' (line 816)
        event_21998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 41), 'event', False)
        # Obtaining the member 'y' of a type (line 816)
        y_21999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 41), event_21998, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 31), list_21995, y_21999)
        # Adding element type (line 816)
        # Getting the type of 'scl' (line 816)
        scl_22000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 50), 'scl', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 31), list_21995, scl_22000)
        
        # Processing the call keyword arguments (line 816)
        kwargs_22001 = {}
        # Getting the type of 'ax' (line 816)
        ax_21993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 8), 'ax', False)
        # Obtaining the member '_set_view_from_bbox' of a type (line 816)
        _set_view_from_bbox_21994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 8), ax_21993, '_set_view_from_bbox')
        # Calling _set_view_from_bbox(args, kwargs) (line 816)
        _set_view_from_bbox_call_result_22002 = invoke(stypy.reporting.localization.Localization(__file__, 816, 8), _set_view_from_bbox_21994, *[list_21995], **kwargs_22001)
        
        
        
        
        # Call to time(...): (line 820)
        # Processing the call keyword arguments (line 820)
        kwargs_22005 = {}
        # Getting the type of 'time' (line 820)
        time_22003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 12), 'time', False)
        # Obtaining the member 'time' of a type (line 820)
        time_22004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 12), time_22003, 'time')
        # Calling time(args, kwargs) (line 820)
        time_call_result_22006 = invoke(stypy.reporting.localization.Localization(__file__, 820, 12), time_22004, *[], **kwargs_22005)
        
        # Getting the type of 'self' (line 820)
        self_22007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 24), 'self')
        # Obtaining the member 'lastscroll' of a type (line 820)
        lastscroll_22008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 24), self_22007, 'lastscroll')
        # Applying the binary operator '-' (line 820)
        result_sub_22009 = python_operator(stypy.reporting.localization.Localization(__file__, 820, 12), '-', time_call_result_22006, lastscroll_22008)
        
        # Getting the type of 'self' (line 820)
        self_22010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 43), 'self')
        # Obtaining the member 'scrollthresh' of a type (line 820)
        scrollthresh_22011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 43), self_22010, 'scrollthresh')
        # Applying the binary operator '<' (line 820)
        result_lt_22012 = python_operator(stypy.reporting.localization.Localization(__file__, 820, 11), '<', result_sub_22009, scrollthresh_22011)
        
        # Testing the type of an if condition (line 820)
        if_condition_22013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 820, 8), result_lt_22012)
        # Assigning a type to the variable 'if_condition_22013' (line 820)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 8), 'if_condition_22013', if_condition_22013)
        # SSA begins for if statement (line 820)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to back(...): (line 821)
        # Processing the call keyword arguments (line 821)
        kwargs_22021 = {}
        
        # Call to get_tool(...): (line 821)
        # Processing the call arguments (line 821)
        # Getting the type of '_views_positions' (line 821)
        _views_positions_22017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 38), '_views_positions', False)
        # Processing the call keyword arguments (line 821)
        kwargs_22018 = {}
        # Getting the type of 'self' (line 821)
        self_22014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 12), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 821)
        toolmanager_22015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 12), self_22014, 'toolmanager')
        # Obtaining the member 'get_tool' of a type (line 821)
        get_tool_22016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 12), toolmanager_22015, 'get_tool')
        # Calling get_tool(args, kwargs) (line 821)
        get_tool_call_result_22019 = invoke(stypy.reporting.localization.Localization(__file__, 821, 12), get_tool_22016, *[_views_positions_22017], **kwargs_22018)
        
        # Obtaining the member 'back' of a type (line 821)
        back_22020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 12), get_tool_call_result_22019, 'back')
        # Calling back(args, kwargs) (line 821)
        back_call_result_22022 = invoke(stypy.reporting.localization.Localization(__file__, 821, 12), back_22020, *[], **kwargs_22021)
        
        # SSA join for if statement (line 820)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw_idle(...): (line 823)
        # Processing the call keyword arguments (line 823)
        kwargs_22027 = {}
        # Getting the type of 'self' (line 823)
        self_22023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 823)
        figure_22024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 823, 8), self_22023, 'figure')
        # Obtaining the member 'canvas' of a type (line 823)
        canvas_22025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 823, 8), figure_22024, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 823)
        draw_idle_22026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 823, 8), canvas_22025, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 823)
        draw_idle_call_result_22028 = invoke(stypy.reporting.localization.Localization(__file__, 823, 8), draw_idle_22026, *[], **kwargs_22027)
        
        
        # Assigning a Call to a Attribute (line 825):
        
        # Assigning a Call to a Attribute (line 825):
        
        # Assigning a Call to a Attribute (line 825):
        
        # Call to time(...): (line 825)
        # Processing the call keyword arguments (line 825)
        kwargs_22031 = {}
        # Getting the type of 'time' (line 825)
        time_22029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 26), 'time', False)
        # Obtaining the member 'time' of a type (line 825)
        time_22030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 26), time_22029, 'time')
        # Calling time(args, kwargs) (line 825)
        time_call_result_22032 = invoke(stypy.reporting.localization.Localization(__file__, 825, 26), time_22030, *[], **kwargs_22031)
        
        # Getting the type of 'self' (line 825)
        self_22033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 8), 'self')
        # Setting the type of the member 'lastscroll' of a type (line 825)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 8), self_22033, 'lastscroll', time_call_result_22032)
        
        # Call to push_current(...): (line 826)
        # Processing the call keyword arguments (line 826)
        kwargs_22041 = {}
        
        # Call to get_tool(...): (line 826)
        # Processing the call arguments (line 826)
        # Getting the type of '_views_positions' (line 826)
        _views_positions_22037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 34), '_views_positions', False)
        # Processing the call keyword arguments (line 826)
        kwargs_22038 = {}
        # Getting the type of 'self' (line 826)
        self_22034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 826)
        toolmanager_22035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 8), self_22034, 'toolmanager')
        # Obtaining the member 'get_tool' of a type (line 826)
        get_tool_22036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 8), toolmanager_22035, 'get_tool')
        # Calling get_tool(args, kwargs) (line 826)
        get_tool_call_result_22039 = invoke(stypy.reporting.localization.Localization(__file__, 826, 8), get_tool_22036, *[_views_positions_22037], **kwargs_22038)
        
        # Obtaining the member 'push_current' of a type (line 826)
        push_current_22040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 8), get_tool_call_result_22039, 'push_current')
        # Calling push_current(args, kwargs) (line 826)
        push_current_call_result_22042 = invoke(stypy.reporting.localization.Localization(__file__, 826, 8), push_current_22040, *[], **kwargs_22041)
        
        
        # ################# End of 'scroll_zoom(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'scroll_zoom' in the type store
        # Getting the type of 'stypy_return_type' (line 800)
        stypy_return_type_22043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22043)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'scroll_zoom'
        return stypy_return_type_22043


# Assigning a type to the variable 'ZoomPanBase' (line 765)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 0), 'ZoomPanBase', ZoomPanBase)
# Declaration of the 'ToolZoom' class
# Getting the type of 'ZoomPanBase' (line 829)
ZoomPanBase_22044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 15), 'ZoomPanBase')

class ToolZoom(ZoomPanBase_22044, ):
    str_22045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 4), 'str', 'Zoom to rectangle')
    
    # Assigning a Str to a Name (line 832):
    
    # Assigning a Str to a Name (line 832):
    
    # Assigning a Str to a Name (line 833):
    
    # Assigning a Str to a Name (line 833):
    
    # Assigning a Subscript to a Name (line 834):
    
    # Assigning a Subscript to a Name (line 834):
    
    # Assigning a Attribute to a Name (line 835):
    
    # Assigning a Attribute to a Name (line 835):
    
    # Assigning a Str to a Name (line 836):
    
    # Assigning a Str to a Name (line 836):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 838, 4, False)
        # Assigning a type to the variable 'self' (line 839)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolZoom.__init__', [], 'args', None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 839)
        # Processing the call arguments (line 839)
        # Getting the type of 'self' (line 839)
        self_22048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 29), 'self', False)
        # Getting the type of 'args' (line 839)
        args_22049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 36), 'args', False)
        # Processing the call keyword arguments (line 839)
        kwargs_22050 = {}
        # Getting the type of 'ZoomPanBase' (line 839)
        ZoomPanBase_22046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 8), 'ZoomPanBase', False)
        # Obtaining the member '__init__' of a type (line 839)
        init___22047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 8), ZoomPanBase_22046, '__init__')
        # Calling __init__(args, kwargs) (line 839)
        init___call_result_22051 = invoke(stypy.reporting.localization.Localization(__file__, 839, 8), init___22047, *[self_22048, args_22049], **kwargs_22050)
        
        
        # Assigning a List to a Attribute (line 840):
        
        # Assigning a List to a Attribute (line 840):
        
        # Assigning a List to a Attribute (line 840):
        
        # Obtaining an instance of the builtin type 'list' (line 840)
        list_22052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 840)
        
        # Getting the type of 'self' (line 840)
        self_22053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'self')
        # Setting the type of the member '_ids_zoom' of a type (line 840)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 8), self_22053, '_ids_zoom', list_22052)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _cancel_action(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cancel_action'
        module_type_store = module_type_store.open_function_context('_cancel_action', 842, 4, False)
        # Assigning a type to the variable 'self' (line 843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolZoom._cancel_action.__dict__.__setitem__('stypy_localization', localization)
        ToolZoom._cancel_action.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolZoom._cancel_action.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolZoom._cancel_action.__dict__.__setitem__('stypy_function_name', 'ToolZoom._cancel_action')
        ToolZoom._cancel_action.__dict__.__setitem__('stypy_param_names_list', [])
        ToolZoom._cancel_action.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolZoom._cancel_action.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolZoom._cancel_action.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolZoom._cancel_action.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolZoom._cancel_action.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolZoom._cancel_action.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolZoom._cancel_action', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cancel_action', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cancel_action(...)' code ##################

        
        # Getting the type of 'self' (line 843)
        self_22054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 23), 'self')
        # Obtaining the member '_ids_zoom' of a type (line 843)
        _ids_zoom_22055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 23), self_22054, '_ids_zoom')
        # Testing the type of a for loop iterable (line 843)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 843, 8), _ids_zoom_22055)
        # Getting the type of the for loop variable (line 843)
        for_loop_var_22056 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 843, 8), _ids_zoom_22055)
        # Assigning a type to the variable 'zoom_id' (line 843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 8), 'zoom_id', for_loop_var_22056)
        # SSA begins for a for statement (line 843)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to mpl_disconnect(...): (line 844)
        # Processing the call arguments (line 844)
        # Getting the type of 'zoom_id' (line 844)
        zoom_id_22061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 46), 'zoom_id', False)
        # Processing the call keyword arguments (line 844)
        kwargs_22062 = {}
        # Getting the type of 'self' (line 844)
        self_22057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 12), 'self', False)
        # Obtaining the member 'figure' of a type (line 844)
        figure_22058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 12), self_22057, 'figure')
        # Obtaining the member 'canvas' of a type (line 844)
        canvas_22059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 12), figure_22058, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 844)
        mpl_disconnect_22060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 12), canvas_22059, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 844)
        mpl_disconnect_call_result_22063 = invoke(stypy.reporting.localization.Localization(__file__, 844, 12), mpl_disconnect_22060, *[zoom_id_22061], **kwargs_22062)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to trigger_tool(...): (line 845)
        # Processing the call arguments (line 845)
        str_22067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 38), 'str', 'rubberband')
        # Getting the type of 'self' (line 845)
        self_22068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 52), 'self', False)
        # Processing the call keyword arguments (line 845)
        kwargs_22069 = {}
        # Getting the type of 'self' (line 845)
        self_22064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 845)
        toolmanager_22065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 8), self_22064, 'toolmanager')
        # Obtaining the member 'trigger_tool' of a type (line 845)
        trigger_tool_22066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 8), toolmanager_22065, 'trigger_tool')
        # Calling trigger_tool(args, kwargs) (line 845)
        trigger_tool_call_result_22070 = invoke(stypy.reporting.localization.Localization(__file__, 845, 8), trigger_tool_22066, *[str_22067, self_22068], **kwargs_22069)
        
        
        # Call to refresh_locators(...): (line 846)
        # Processing the call keyword arguments (line 846)
        kwargs_22078 = {}
        
        # Call to get_tool(...): (line 846)
        # Processing the call arguments (line 846)
        # Getting the type of '_views_positions' (line 846)
        _views_positions_22074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 34), '_views_positions', False)
        # Processing the call keyword arguments (line 846)
        kwargs_22075 = {}
        # Getting the type of 'self' (line 846)
        self_22071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 846)
        toolmanager_22072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 8), self_22071, 'toolmanager')
        # Obtaining the member 'get_tool' of a type (line 846)
        get_tool_22073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 8), toolmanager_22072, 'get_tool')
        # Calling get_tool(args, kwargs) (line 846)
        get_tool_call_result_22076 = invoke(stypy.reporting.localization.Localization(__file__, 846, 8), get_tool_22073, *[_views_positions_22074], **kwargs_22075)
        
        # Obtaining the member 'refresh_locators' of a type (line 846)
        refresh_locators_22077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 8), get_tool_call_result_22076, 'refresh_locators')
        # Calling refresh_locators(args, kwargs) (line 846)
        refresh_locators_call_result_22079 = invoke(stypy.reporting.localization.Localization(__file__, 846, 8), refresh_locators_22077, *[], **kwargs_22078)
        
        
        # Assigning a Name to a Attribute (line 847):
        
        # Assigning a Name to a Attribute (line 847):
        
        # Assigning a Name to a Attribute (line 847):
        # Getting the type of 'None' (line 847)
        None_22080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 24), 'None')
        # Getting the type of 'self' (line 847)
        self_22081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 8), 'self')
        # Setting the type of the member '_xypress' of a type (line 847)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 8), self_22081, '_xypress', None_22080)
        
        # Assigning a Name to a Attribute (line 848):
        
        # Assigning a Name to a Attribute (line 848):
        
        # Assigning a Name to a Attribute (line 848):
        # Getting the type of 'None' (line 848)
        None_22082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 31), 'None')
        # Getting the type of 'self' (line 848)
        self_22083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'self')
        # Setting the type of the member '_button_pressed' of a type (line 848)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 8), self_22083, '_button_pressed', None_22082)
        
        # Assigning a List to a Attribute (line 849):
        
        # Assigning a List to a Attribute (line 849):
        
        # Assigning a List to a Attribute (line 849):
        
        # Obtaining an instance of the builtin type 'list' (line 849)
        list_22084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 849)
        
        # Getting the type of 'self' (line 849)
        self_22085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'self')
        # Setting the type of the member '_ids_zoom' of a type (line 849)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 8), self_22085, '_ids_zoom', list_22084)
        # Assigning a type to the variable 'stypy_return_type' (line 850)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of '_cancel_action(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cancel_action' in the type store
        # Getting the type of 'stypy_return_type' (line 842)
        stypy_return_type_22086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22086)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cancel_action'
        return stypy_return_type_22086


    @norecursion
    def _press(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_press'
        module_type_store = module_type_store.open_function_context('_press', 852, 4, False)
        # Assigning a type to the variable 'self' (line 853)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolZoom._press.__dict__.__setitem__('stypy_localization', localization)
        ToolZoom._press.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolZoom._press.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolZoom._press.__dict__.__setitem__('stypy_function_name', 'ToolZoom._press')
        ToolZoom._press.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolZoom._press.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolZoom._press.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolZoom._press.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolZoom._press.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolZoom._press.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolZoom._press.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolZoom._press', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_press', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_press(...)' code ##################

        str_22087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 8), 'str', 'the _press mouse button in zoom to rect mode callback')
        
        
        # Getting the type of 'self' (line 857)
        self_22088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 11), 'self')
        # Obtaining the member '_ids_zoom' of a type (line 857)
        _ids_zoom_22089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 857, 11), self_22088, '_ids_zoom')
        
        # Obtaining an instance of the builtin type 'list' (line 857)
        list_22090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 857)
        
        # Applying the binary operator '!=' (line 857)
        result_ne_22091 = python_operator(stypy.reporting.localization.Localization(__file__, 857, 11), '!=', _ids_zoom_22089, list_22090)
        
        # Testing the type of an if condition (line 857)
        if_condition_22092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 857, 8), result_ne_22091)
        # Assigning a type to the variable 'if_condition_22092' (line 857)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 8), 'if_condition_22092', if_condition_22092)
        # SSA begins for if statement (line 857)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _cancel_action(...): (line 858)
        # Processing the call keyword arguments (line 858)
        kwargs_22095 = {}
        # Getting the type of 'self' (line 858)
        self_22093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 12), 'self', False)
        # Obtaining the member '_cancel_action' of a type (line 858)
        _cancel_action_22094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 858, 12), self_22093, '_cancel_action')
        # Calling _cancel_action(args, kwargs) (line 858)
        _cancel_action_call_result_22096 = invoke(stypy.reporting.localization.Localization(__file__, 858, 12), _cancel_action_22094, *[], **kwargs_22095)
        
        # SSA join for if statement (line 857)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'event' (line 860)
        event_22097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 11), 'event')
        # Obtaining the member 'button' of a type (line 860)
        button_22098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 11), event_22097, 'button')
        int_22099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 27), 'int')
        # Applying the binary operator '==' (line 860)
        result_eq_22100 = python_operator(stypy.reporting.localization.Localization(__file__, 860, 11), '==', button_22098, int_22099)
        
        # Testing the type of an if condition (line 860)
        if_condition_22101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 860, 8), result_eq_22100)
        # Assigning a type to the variable 'if_condition_22101' (line 860)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 8), 'if_condition_22101', if_condition_22101)
        # SSA begins for if statement (line 860)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 861):
        
        # Assigning a Num to a Attribute (line 861):
        
        # Assigning a Num to a Attribute (line 861):
        int_22102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 35), 'int')
        # Getting the type of 'self' (line 861)
        self_22103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 12), 'self')
        # Setting the type of the member '_button_pressed' of a type (line 861)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 12), self_22103, '_button_pressed', int_22102)
        # SSA branch for the else part of an if statement (line 860)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'event' (line 862)
        event_22104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 13), 'event')
        # Obtaining the member 'button' of a type (line 862)
        button_22105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 13), event_22104, 'button')
        int_22106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 29), 'int')
        # Applying the binary operator '==' (line 862)
        result_eq_22107 = python_operator(stypy.reporting.localization.Localization(__file__, 862, 13), '==', button_22105, int_22106)
        
        # Testing the type of an if condition (line 862)
        if_condition_22108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 862, 13), result_eq_22107)
        # Assigning a type to the variable 'if_condition_22108' (line 862)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 13), 'if_condition_22108', if_condition_22108)
        # SSA begins for if statement (line 862)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 863):
        
        # Assigning a Num to a Attribute (line 863):
        
        # Assigning a Num to a Attribute (line 863):
        int_22109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 35), 'int')
        # Getting the type of 'self' (line 863)
        self_22110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 12), 'self')
        # Setting the type of the member '_button_pressed' of a type (line 863)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 863, 12), self_22110, '_button_pressed', int_22109)
        # SSA branch for the else part of an if statement (line 862)
        module_type_store.open_ssa_branch('else')
        
        # Call to _cancel_action(...): (line 865)
        # Processing the call keyword arguments (line 865)
        kwargs_22113 = {}
        # Getting the type of 'self' (line 865)
        self_22111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 12), 'self', False)
        # Obtaining the member '_cancel_action' of a type (line 865)
        _cancel_action_22112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 865, 12), self_22111, '_cancel_action')
        # Calling _cancel_action(args, kwargs) (line 865)
        _cancel_action_call_result_22114 = invoke(stypy.reporting.localization.Localization(__file__, 865, 12), _cancel_action_22112, *[], **kwargs_22113)
        
        # Assigning a type to the variable 'stypy_return_type' (line 866)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 862)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 860)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Tuple (line 868):
        
        # Assigning a Attribute to a Name (line 868):
        
        # Assigning a Attribute to a Name (line 868):
        # Getting the type of 'event' (line 868)
        event_22115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 15), 'event')
        # Obtaining the member 'x' of a type (line 868)
        x_22116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 15), event_22115, 'x')
        # Assigning a type to the variable 'tuple_assignment_20167' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'tuple_assignment_20167', x_22116)
        
        # Assigning a Attribute to a Name (line 868):
        
        # Assigning a Attribute to a Name (line 868):
        # Getting the type of 'event' (line 868)
        event_22117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 24), 'event')
        # Obtaining the member 'y' of a type (line 868)
        y_22118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 24), event_22117, 'y')
        # Assigning a type to the variable 'tuple_assignment_20168' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'tuple_assignment_20168', y_22118)
        
        # Assigning a Name to a Name (line 868):
        
        # Assigning a Name to a Name (line 868):
        # Getting the type of 'tuple_assignment_20167' (line 868)
        tuple_assignment_20167_22119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'tuple_assignment_20167')
        # Assigning a type to the variable 'x' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'x', tuple_assignment_20167_22119)
        
        # Assigning a Name to a Name (line 868):
        
        # Assigning a Name to a Name (line 868):
        # Getting the type of 'tuple_assignment_20168' (line 868)
        tuple_assignment_20168_22120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'tuple_assignment_20168')
        # Assigning a type to the variable 'y' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 11), 'y', tuple_assignment_20168_22120)
        
        # Assigning a List to a Attribute (line 870):
        
        # Assigning a List to a Attribute (line 870):
        
        # Assigning a List to a Attribute (line 870):
        
        # Obtaining an instance of the builtin type 'list' (line 870)
        list_22121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 870)
        
        # Getting the type of 'self' (line 870)
        self_22122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'self')
        # Setting the type of the member '_xypress' of a type (line 870)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 8), self_22122, '_xypress', list_22121)
        
        
        # Call to enumerate(...): (line 871)
        # Processing the call arguments (line 871)
        
        # Call to get_axes(...): (line 871)
        # Processing the call keyword arguments (line 871)
        kwargs_22127 = {}
        # Getting the type of 'self' (line 871)
        self_22124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 30), 'self', False)
        # Obtaining the member 'figure' of a type (line 871)
        figure_22125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 30), self_22124, 'figure')
        # Obtaining the member 'get_axes' of a type (line 871)
        get_axes_22126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 30), figure_22125, 'get_axes')
        # Calling get_axes(args, kwargs) (line 871)
        get_axes_call_result_22128 = invoke(stypy.reporting.localization.Localization(__file__, 871, 30), get_axes_22126, *[], **kwargs_22127)
        
        # Processing the call keyword arguments (line 871)
        kwargs_22129 = {}
        # Getting the type of 'enumerate' (line 871)
        enumerate_22123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 20), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 871)
        enumerate_call_result_22130 = invoke(stypy.reporting.localization.Localization(__file__, 871, 20), enumerate_22123, *[get_axes_call_result_22128], **kwargs_22129)
        
        # Testing the type of a for loop iterable (line 871)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 871, 8), enumerate_call_result_22130)
        # Getting the type of the for loop variable (line 871)
        for_loop_var_22131 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 871, 8), enumerate_call_result_22130)
        # Assigning a type to the variable 'i' (line 871)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 871, 8), for_loop_var_22131))
        # Assigning a type to the variable 'a' (line 871)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 871, 8), for_loop_var_22131))
        # SSA begins for a for statement (line 871)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 872)
        x_22132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 16), 'x')
        # Getting the type of 'None' (line 872)
        None_22133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 25), 'None')
        # Applying the binary operator 'isnot' (line 872)
        result_is_not_22134 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 16), 'isnot', x_22132, None_22133)
        
        
        # Getting the type of 'y' (line 872)
        y_22135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 34), 'y')
        # Getting the type of 'None' (line 872)
        None_22136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 43), 'None')
        # Applying the binary operator 'isnot' (line 872)
        result_is_not_22137 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 34), 'isnot', y_22135, None_22136)
        
        # Applying the binary operator 'and' (line 872)
        result_and_keyword_22138 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 16), 'and', result_is_not_22134, result_is_not_22137)
        
        # Call to in_axes(...): (line 872)
        # Processing the call arguments (line 872)
        # Getting the type of 'event' (line 872)
        event_22141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 62), 'event', False)
        # Processing the call keyword arguments (line 872)
        kwargs_22142 = {}
        # Getting the type of 'a' (line 872)
        a_22139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 52), 'a', False)
        # Obtaining the member 'in_axes' of a type (line 872)
        in_axes_22140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 52), a_22139, 'in_axes')
        # Calling in_axes(args, kwargs) (line 872)
        in_axes_call_result_22143 = invoke(stypy.reporting.localization.Localization(__file__, 872, 52), in_axes_22140, *[event_22141], **kwargs_22142)
        
        # Applying the binary operator 'and' (line 872)
        result_and_keyword_22144 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 16), 'and', result_and_keyword_22138, in_axes_call_result_22143)
        
        # Call to get_navigate(...): (line 873)
        # Processing the call keyword arguments (line 873)
        kwargs_22147 = {}
        # Getting the type of 'a' (line 873)
        a_22145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 20), 'a', False)
        # Obtaining the member 'get_navigate' of a type (line 873)
        get_navigate_22146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 873, 20), a_22145, 'get_navigate')
        # Calling get_navigate(args, kwargs) (line 873)
        get_navigate_call_result_22148 = invoke(stypy.reporting.localization.Localization(__file__, 873, 20), get_navigate_22146, *[], **kwargs_22147)
        
        # Applying the binary operator 'and' (line 872)
        result_and_keyword_22149 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 16), 'and', result_and_keyword_22144, get_navigate_call_result_22148)
        
        # Call to can_zoom(...): (line 873)
        # Processing the call keyword arguments (line 873)
        kwargs_22152 = {}
        # Getting the type of 'a' (line 873)
        a_22150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 41), 'a', False)
        # Obtaining the member 'can_zoom' of a type (line 873)
        can_zoom_22151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 873, 41), a_22150, 'can_zoom')
        # Calling can_zoom(args, kwargs) (line 873)
        can_zoom_call_result_22153 = invoke(stypy.reporting.localization.Localization(__file__, 873, 41), can_zoom_22151, *[], **kwargs_22152)
        
        # Applying the binary operator 'and' (line 872)
        result_and_keyword_22154 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 16), 'and', result_and_keyword_22149, can_zoom_call_result_22153)
        
        # Testing the type of an if condition (line 872)
        if_condition_22155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 872, 12), result_and_keyword_22154)
        # Assigning a type to the variable 'if_condition_22155' (line 872)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'if_condition_22155', if_condition_22155)
        # SSA begins for if statement (line 872)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 874)
        # Processing the call arguments (line 874)
        
        # Obtaining an instance of the builtin type 'tuple' (line 874)
        tuple_22159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 874)
        # Adding element type (line 874)
        # Getting the type of 'x' (line 874)
        x_22160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 38), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 874, 38), tuple_22159, x_22160)
        # Adding element type (line 874)
        # Getting the type of 'y' (line 874)
        y_22161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 41), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 874, 38), tuple_22159, y_22161)
        # Adding element type (line 874)
        # Getting the type of 'a' (line 874)
        a_22162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 44), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 874, 38), tuple_22159, a_22162)
        # Adding element type (line 874)
        # Getting the type of 'i' (line 874)
        i_22163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 47), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 874, 38), tuple_22159, i_22163)
        # Adding element type (line 874)
        
        # Call to _get_view(...): (line 874)
        # Processing the call keyword arguments (line 874)
        kwargs_22166 = {}
        # Getting the type of 'a' (line 874)
        a_22164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 50), 'a', False)
        # Obtaining the member '_get_view' of a type (line 874)
        _get_view_22165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 50), a_22164, '_get_view')
        # Calling _get_view(args, kwargs) (line 874)
        _get_view_call_result_22167 = invoke(stypy.reporting.localization.Localization(__file__, 874, 50), _get_view_22165, *[], **kwargs_22166)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 874, 38), tuple_22159, _get_view_call_result_22167)
        
        # Processing the call keyword arguments (line 874)
        kwargs_22168 = {}
        # Getting the type of 'self' (line 874)
        self_22156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 16), 'self', False)
        # Obtaining the member '_xypress' of a type (line 874)
        _xypress_22157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 16), self_22156, '_xypress')
        # Obtaining the member 'append' of a type (line 874)
        append_22158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 16), _xypress_22157, 'append')
        # Calling append(args, kwargs) (line 874)
        append_call_result_22169 = invoke(stypy.reporting.localization.Localization(__file__, 874, 16), append_22158, *[tuple_22159], **kwargs_22168)
        
        # SSA join for if statement (line 872)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 876):
        
        # Assigning a Call to a Name (line 876):
        
        # Assigning a Call to a Name (line 876):
        
        # Call to mpl_connect(...): (line 876)
        # Processing the call arguments (line 876)
        str_22174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 12), 'str', 'motion_notify_event')
        # Getting the type of 'self' (line 877)
        self_22175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 35), 'self', False)
        # Obtaining the member '_mouse_move' of a type (line 877)
        _mouse_move_22176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 35), self_22175, '_mouse_move')
        # Processing the call keyword arguments (line 876)
        kwargs_22177 = {}
        # Getting the type of 'self' (line 876)
        self_22170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 14), 'self', False)
        # Obtaining the member 'figure' of a type (line 876)
        figure_22171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 14), self_22170, 'figure')
        # Obtaining the member 'canvas' of a type (line 876)
        canvas_22172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 14), figure_22171, 'canvas')
        # Obtaining the member 'mpl_connect' of a type (line 876)
        mpl_connect_22173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 14), canvas_22172, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 876)
        mpl_connect_call_result_22178 = invoke(stypy.reporting.localization.Localization(__file__, 876, 14), mpl_connect_22173, *[str_22174, _mouse_move_22176], **kwargs_22177)
        
        # Assigning a type to the variable 'id1' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'id1', mpl_connect_call_result_22178)
        
        # Assigning a Call to a Name (line 878):
        
        # Assigning a Call to a Name (line 878):
        
        # Assigning a Call to a Name (line 878):
        
        # Call to mpl_connect(...): (line 878)
        # Processing the call arguments (line 878)
        str_22183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 12), 'str', 'key_press_event')
        # Getting the type of 'self' (line 879)
        self_22184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 31), 'self', False)
        # Obtaining the member '_switch_on_zoom_mode' of a type (line 879)
        _switch_on_zoom_mode_22185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 31), self_22184, '_switch_on_zoom_mode')
        # Processing the call keyword arguments (line 878)
        kwargs_22186 = {}
        # Getting the type of 'self' (line 878)
        self_22179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 14), 'self', False)
        # Obtaining the member 'figure' of a type (line 878)
        figure_22180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 14), self_22179, 'figure')
        # Obtaining the member 'canvas' of a type (line 878)
        canvas_22181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 14), figure_22180, 'canvas')
        # Obtaining the member 'mpl_connect' of a type (line 878)
        mpl_connect_22182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 14), canvas_22181, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 878)
        mpl_connect_call_result_22187 = invoke(stypy.reporting.localization.Localization(__file__, 878, 14), mpl_connect_22182, *[str_22183, _switch_on_zoom_mode_22185], **kwargs_22186)
        
        # Assigning a type to the variable 'id2' (line 878)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 8), 'id2', mpl_connect_call_result_22187)
        
        # Assigning a Call to a Name (line 880):
        
        # Assigning a Call to a Name (line 880):
        
        # Assigning a Call to a Name (line 880):
        
        # Call to mpl_connect(...): (line 880)
        # Processing the call arguments (line 880)
        str_22192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 12), 'str', 'key_release_event')
        # Getting the type of 'self' (line 881)
        self_22193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 33), 'self', False)
        # Obtaining the member '_switch_off_zoom_mode' of a type (line 881)
        _switch_off_zoom_mode_22194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 33), self_22193, '_switch_off_zoom_mode')
        # Processing the call keyword arguments (line 880)
        kwargs_22195 = {}
        # Getting the type of 'self' (line 880)
        self_22188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 14), 'self', False)
        # Obtaining the member 'figure' of a type (line 880)
        figure_22189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 14), self_22188, 'figure')
        # Obtaining the member 'canvas' of a type (line 880)
        canvas_22190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 14), figure_22189, 'canvas')
        # Obtaining the member 'mpl_connect' of a type (line 880)
        mpl_connect_22191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 14), canvas_22190, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 880)
        mpl_connect_call_result_22196 = invoke(stypy.reporting.localization.Localization(__file__, 880, 14), mpl_connect_22191, *[str_22192, _switch_off_zoom_mode_22194], **kwargs_22195)
        
        # Assigning a type to the variable 'id3' (line 880)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 8), 'id3', mpl_connect_call_result_22196)
        
        # Assigning a Tuple to a Attribute (line 883):
        
        # Assigning a Tuple to a Attribute (line 883):
        
        # Assigning a Tuple to a Attribute (line 883):
        
        # Obtaining an instance of the builtin type 'tuple' (line 883)
        tuple_22197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 883)
        # Adding element type (line 883)
        # Getting the type of 'id1' (line 883)
        id1_22198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 25), 'id1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 883, 25), tuple_22197, id1_22198)
        # Adding element type (line 883)
        # Getting the type of 'id2' (line 883)
        id2_22199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 30), 'id2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 883, 25), tuple_22197, id2_22199)
        # Adding element type (line 883)
        # Getting the type of 'id3' (line 883)
        id3_22200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 35), 'id3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 883, 25), tuple_22197, id3_22200)
        
        # Getting the type of 'self' (line 883)
        self_22201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 8), 'self')
        # Setting the type of the member '_ids_zoom' of a type (line 883)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 8), self_22201, '_ids_zoom', tuple_22197)
        
        # Assigning a Attribute to a Attribute (line 884):
        
        # Assigning a Attribute to a Attribute (line 884):
        
        # Assigning a Attribute to a Attribute (line 884):
        # Getting the type of 'event' (line 884)
        event_22202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 26), 'event')
        # Obtaining the member 'key' of a type (line 884)
        key_22203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 26), event_22202, 'key')
        # Getting the type of 'self' (line 884)
        self_22204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 8), 'self')
        # Setting the type of the member '_zoom_mode' of a type (line 884)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 8), self_22204, '_zoom_mode', key_22203)
        
        # ################# End of '_press(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_press' in the type store
        # Getting the type of 'stypy_return_type' (line 852)
        stypy_return_type_22205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22205)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_press'
        return stypy_return_type_22205


    @norecursion
    def _switch_on_zoom_mode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_switch_on_zoom_mode'
        module_type_store = module_type_store.open_function_context('_switch_on_zoom_mode', 886, 4, False)
        # Assigning a type to the variable 'self' (line 887)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 887, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolZoom._switch_on_zoom_mode.__dict__.__setitem__('stypy_localization', localization)
        ToolZoom._switch_on_zoom_mode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolZoom._switch_on_zoom_mode.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolZoom._switch_on_zoom_mode.__dict__.__setitem__('stypy_function_name', 'ToolZoom._switch_on_zoom_mode')
        ToolZoom._switch_on_zoom_mode.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolZoom._switch_on_zoom_mode.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolZoom._switch_on_zoom_mode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolZoom._switch_on_zoom_mode.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolZoom._switch_on_zoom_mode.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolZoom._switch_on_zoom_mode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolZoom._switch_on_zoom_mode.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolZoom._switch_on_zoom_mode', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_switch_on_zoom_mode', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_switch_on_zoom_mode(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 887):
        
        # Assigning a Attribute to a Attribute (line 887):
        
        # Assigning a Attribute to a Attribute (line 887):
        # Getting the type of 'event' (line 887)
        event_22206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 26), 'event')
        # Obtaining the member 'key' of a type (line 887)
        key_22207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 26), event_22206, 'key')
        # Getting the type of 'self' (line 887)
        self_22208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 8), 'self')
        # Setting the type of the member '_zoom_mode' of a type (line 887)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 8), self_22208, '_zoom_mode', key_22207)
        
        # Call to _mouse_move(...): (line 888)
        # Processing the call arguments (line 888)
        # Getting the type of 'event' (line 888)
        event_22211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 25), 'event', False)
        # Processing the call keyword arguments (line 888)
        kwargs_22212 = {}
        # Getting the type of 'self' (line 888)
        self_22209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'self', False)
        # Obtaining the member '_mouse_move' of a type (line 888)
        _mouse_move_22210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 8), self_22209, '_mouse_move')
        # Calling _mouse_move(args, kwargs) (line 888)
        _mouse_move_call_result_22213 = invoke(stypy.reporting.localization.Localization(__file__, 888, 8), _mouse_move_22210, *[event_22211], **kwargs_22212)
        
        
        # ################# End of '_switch_on_zoom_mode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_switch_on_zoom_mode' in the type store
        # Getting the type of 'stypy_return_type' (line 886)
        stypy_return_type_22214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22214)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_switch_on_zoom_mode'
        return stypy_return_type_22214


    @norecursion
    def _switch_off_zoom_mode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_switch_off_zoom_mode'
        module_type_store = module_type_store.open_function_context('_switch_off_zoom_mode', 890, 4, False)
        # Assigning a type to the variable 'self' (line 891)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolZoom._switch_off_zoom_mode.__dict__.__setitem__('stypy_localization', localization)
        ToolZoom._switch_off_zoom_mode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolZoom._switch_off_zoom_mode.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolZoom._switch_off_zoom_mode.__dict__.__setitem__('stypy_function_name', 'ToolZoom._switch_off_zoom_mode')
        ToolZoom._switch_off_zoom_mode.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolZoom._switch_off_zoom_mode.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolZoom._switch_off_zoom_mode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolZoom._switch_off_zoom_mode.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolZoom._switch_off_zoom_mode.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolZoom._switch_off_zoom_mode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolZoom._switch_off_zoom_mode.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolZoom._switch_off_zoom_mode', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_switch_off_zoom_mode', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_switch_off_zoom_mode(...)' code ##################

        
        # Assigning a Name to a Attribute (line 891):
        
        # Assigning a Name to a Attribute (line 891):
        
        # Assigning a Name to a Attribute (line 891):
        # Getting the type of 'None' (line 891)
        None_22215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 26), 'None')
        # Getting the type of 'self' (line 891)
        self_22216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 8), 'self')
        # Setting the type of the member '_zoom_mode' of a type (line 891)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 8), self_22216, '_zoom_mode', None_22215)
        
        # Call to _mouse_move(...): (line 892)
        # Processing the call arguments (line 892)
        # Getting the type of 'event' (line 892)
        event_22219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 25), 'event', False)
        # Processing the call keyword arguments (line 892)
        kwargs_22220 = {}
        # Getting the type of 'self' (line 892)
        self_22217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 8), 'self', False)
        # Obtaining the member '_mouse_move' of a type (line 892)
        _mouse_move_22218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 8), self_22217, '_mouse_move')
        # Calling _mouse_move(args, kwargs) (line 892)
        _mouse_move_call_result_22221 = invoke(stypy.reporting.localization.Localization(__file__, 892, 8), _mouse_move_22218, *[event_22219], **kwargs_22220)
        
        
        # ################# End of '_switch_off_zoom_mode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_switch_off_zoom_mode' in the type store
        # Getting the type of 'stypy_return_type' (line 890)
        stypy_return_type_22222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22222)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_switch_off_zoom_mode'
        return stypy_return_type_22222


    @norecursion
    def _mouse_move(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mouse_move'
        module_type_store = module_type_store.open_function_context('_mouse_move', 894, 4, False)
        # Assigning a type to the variable 'self' (line 895)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolZoom._mouse_move.__dict__.__setitem__('stypy_localization', localization)
        ToolZoom._mouse_move.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolZoom._mouse_move.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolZoom._mouse_move.__dict__.__setitem__('stypy_function_name', 'ToolZoom._mouse_move')
        ToolZoom._mouse_move.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolZoom._mouse_move.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolZoom._mouse_move.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolZoom._mouse_move.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolZoom._mouse_move.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolZoom._mouse_move.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolZoom._mouse_move.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolZoom._mouse_move', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mouse_move', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mouse_move(...)' code ##################

        str_22223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 8), 'str', 'the drag callback in zoom mode')
        
        # Getting the type of 'self' (line 897)
        self_22224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 11), 'self')
        # Obtaining the member '_xypress' of a type (line 897)
        _xypress_22225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 11), self_22224, '_xypress')
        # Testing the type of an if condition (line 897)
        if_condition_22226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 897, 8), _xypress_22225)
        # Assigning a type to the variable 'if_condition_22226' (line 897)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 8), 'if_condition_22226', if_condition_22226)
        # SSA begins for if statement (line 897)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Tuple (line 898):
        
        # Assigning a Attribute to a Name (line 898):
        
        # Assigning a Attribute to a Name (line 898):
        # Getting the type of 'event' (line 898)
        event_22227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 19), 'event')
        # Obtaining the member 'x' of a type (line 898)
        x_22228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 898, 19), event_22227, 'x')
        # Assigning a type to the variable 'tuple_assignment_20169' (line 898)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 898, 12), 'tuple_assignment_20169', x_22228)
        
        # Assigning a Attribute to a Name (line 898):
        
        # Assigning a Attribute to a Name (line 898):
        # Getting the type of 'event' (line 898)
        event_22229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 28), 'event')
        # Obtaining the member 'y' of a type (line 898)
        y_22230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 898, 28), event_22229, 'y')
        # Assigning a type to the variable 'tuple_assignment_20170' (line 898)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 898, 12), 'tuple_assignment_20170', y_22230)
        
        # Assigning a Name to a Name (line 898):
        
        # Assigning a Name to a Name (line 898):
        # Getting the type of 'tuple_assignment_20169' (line 898)
        tuple_assignment_20169_22231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 12), 'tuple_assignment_20169')
        # Assigning a type to the variable 'x' (line 898)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 898, 12), 'x', tuple_assignment_20169_22231)
        
        # Assigning a Name to a Name (line 898):
        
        # Assigning a Name to a Name (line 898):
        # Getting the type of 'tuple_assignment_20170' (line 898)
        tuple_assignment_20170_22232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 12), 'tuple_assignment_20170')
        # Assigning a type to the variable 'y' (line 898)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 898, 15), 'y', tuple_assignment_20170_22232)
        
        # Assigning a Subscript to a Tuple (line 899):
        
        # Assigning a Subscript to a Name (line 899):
        
        # Assigning a Subscript to a Name (line 899):
        
        # Obtaining the type of the subscript
        int_22233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 12), 'int')
        
        # Obtaining the type of the subscript
        int_22234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 55), 'int')
        # Getting the type of 'self' (line 899)
        self_22235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 41), 'self')
        # Obtaining the member '_xypress' of a type (line 899)
        _xypress_22236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 41), self_22235, '_xypress')
        # Obtaining the member '__getitem__' of a type (line 899)
        getitem___22237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 41), _xypress_22236, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 899)
        subscript_call_result_22238 = invoke(stypy.reporting.localization.Localization(__file__, 899, 41), getitem___22237, int_22234)
        
        # Obtaining the member '__getitem__' of a type (line 899)
        getitem___22239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 12), subscript_call_result_22238, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 899)
        subscript_call_result_22240 = invoke(stypy.reporting.localization.Localization(__file__, 899, 12), getitem___22239, int_22233)
        
        # Assigning a type to the variable 'tuple_var_assignment_20171' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'tuple_var_assignment_20171', subscript_call_result_22240)
        
        # Assigning a Subscript to a Name (line 899):
        
        # Assigning a Subscript to a Name (line 899):
        
        # Obtaining the type of the subscript
        int_22241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 12), 'int')
        
        # Obtaining the type of the subscript
        int_22242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 55), 'int')
        # Getting the type of 'self' (line 899)
        self_22243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 41), 'self')
        # Obtaining the member '_xypress' of a type (line 899)
        _xypress_22244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 41), self_22243, '_xypress')
        # Obtaining the member '__getitem__' of a type (line 899)
        getitem___22245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 41), _xypress_22244, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 899)
        subscript_call_result_22246 = invoke(stypy.reporting.localization.Localization(__file__, 899, 41), getitem___22245, int_22242)
        
        # Obtaining the member '__getitem__' of a type (line 899)
        getitem___22247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 12), subscript_call_result_22246, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 899)
        subscript_call_result_22248 = invoke(stypy.reporting.localization.Localization(__file__, 899, 12), getitem___22247, int_22241)
        
        # Assigning a type to the variable 'tuple_var_assignment_20172' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'tuple_var_assignment_20172', subscript_call_result_22248)
        
        # Assigning a Subscript to a Name (line 899):
        
        # Assigning a Subscript to a Name (line 899):
        
        # Obtaining the type of the subscript
        int_22249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 12), 'int')
        
        # Obtaining the type of the subscript
        int_22250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 55), 'int')
        # Getting the type of 'self' (line 899)
        self_22251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 41), 'self')
        # Obtaining the member '_xypress' of a type (line 899)
        _xypress_22252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 41), self_22251, '_xypress')
        # Obtaining the member '__getitem__' of a type (line 899)
        getitem___22253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 41), _xypress_22252, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 899)
        subscript_call_result_22254 = invoke(stypy.reporting.localization.Localization(__file__, 899, 41), getitem___22253, int_22250)
        
        # Obtaining the member '__getitem__' of a type (line 899)
        getitem___22255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 12), subscript_call_result_22254, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 899)
        subscript_call_result_22256 = invoke(stypy.reporting.localization.Localization(__file__, 899, 12), getitem___22255, int_22249)
        
        # Assigning a type to the variable 'tuple_var_assignment_20173' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'tuple_var_assignment_20173', subscript_call_result_22256)
        
        # Assigning a Subscript to a Name (line 899):
        
        # Assigning a Subscript to a Name (line 899):
        
        # Obtaining the type of the subscript
        int_22257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 12), 'int')
        
        # Obtaining the type of the subscript
        int_22258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 55), 'int')
        # Getting the type of 'self' (line 899)
        self_22259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 41), 'self')
        # Obtaining the member '_xypress' of a type (line 899)
        _xypress_22260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 41), self_22259, '_xypress')
        # Obtaining the member '__getitem__' of a type (line 899)
        getitem___22261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 41), _xypress_22260, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 899)
        subscript_call_result_22262 = invoke(stypy.reporting.localization.Localization(__file__, 899, 41), getitem___22261, int_22258)
        
        # Obtaining the member '__getitem__' of a type (line 899)
        getitem___22263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 12), subscript_call_result_22262, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 899)
        subscript_call_result_22264 = invoke(stypy.reporting.localization.Localization(__file__, 899, 12), getitem___22263, int_22257)
        
        # Assigning a type to the variable 'tuple_var_assignment_20174' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'tuple_var_assignment_20174', subscript_call_result_22264)
        
        # Assigning a Subscript to a Name (line 899):
        
        # Assigning a Subscript to a Name (line 899):
        
        # Obtaining the type of the subscript
        int_22265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 12), 'int')
        
        # Obtaining the type of the subscript
        int_22266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 55), 'int')
        # Getting the type of 'self' (line 899)
        self_22267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 41), 'self')
        # Obtaining the member '_xypress' of a type (line 899)
        _xypress_22268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 41), self_22267, '_xypress')
        # Obtaining the member '__getitem__' of a type (line 899)
        getitem___22269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 41), _xypress_22268, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 899)
        subscript_call_result_22270 = invoke(stypy.reporting.localization.Localization(__file__, 899, 41), getitem___22269, int_22266)
        
        # Obtaining the member '__getitem__' of a type (line 899)
        getitem___22271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 12), subscript_call_result_22270, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 899)
        subscript_call_result_22272 = invoke(stypy.reporting.localization.Localization(__file__, 899, 12), getitem___22271, int_22265)
        
        # Assigning a type to the variable 'tuple_var_assignment_20175' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'tuple_var_assignment_20175', subscript_call_result_22272)
        
        # Assigning a Name to a Name (line 899):
        
        # Assigning a Name to a Name (line 899):
        # Getting the type of 'tuple_var_assignment_20171' (line 899)
        tuple_var_assignment_20171_22273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'tuple_var_assignment_20171')
        # Assigning a type to the variable 'lastx' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'lastx', tuple_var_assignment_20171_22273)
        
        # Assigning a Name to a Name (line 899):
        
        # Assigning a Name to a Name (line 899):
        # Getting the type of 'tuple_var_assignment_20172' (line 899)
        tuple_var_assignment_20172_22274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'tuple_var_assignment_20172')
        # Assigning a type to the variable 'lasty' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 19), 'lasty', tuple_var_assignment_20172_22274)
        
        # Assigning a Name to a Name (line 899):
        
        # Assigning a Name to a Name (line 899):
        # Getting the type of 'tuple_var_assignment_20173' (line 899)
        tuple_var_assignment_20173_22275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'tuple_var_assignment_20173')
        # Assigning a type to the variable 'a' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 26), 'a', tuple_var_assignment_20173_22275)
        
        # Assigning a Name to a Name (line 899):
        
        # Assigning a Name to a Name (line 899):
        # Getting the type of 'tuple_var_assignment_20174' (line 899)
        tuple_var_assignment_20174_22276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'tuple_var_assignment_20174')
        # Assigning a type to the variable 'ind' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 29), 'ind', tuple_var_assignment_20174_22276)
        
        # Assigning a Name to a Name (line 899):
        
        # Assigning a Name to a Name (line 899):
        # Getting the type of 'tuple_var_assignment_20175' (line 899)
        tuple_var_assignment_20175_22277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 12), 'tuple_var_assignment_20175')
        # Assigning a type to the variable 'view' (line 899)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 34), 'view', tuple_var_assignment_20175_22277)
        
        # Assigning a Call to a Tuple (line 900):
        
        # Assigning a Call to a Name:
        
        # Assigning a Call to a Name:
        
        # Call to clip(...): (line 900)
        # Processing the call arguments (line 900)
        
        # Obtaining an instance of the builtin type 'list' (line 901)
        list_22280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 901)
        # Adding element type (line 901)
        
        # Obtaining an instance of the builtin type 'list' (line 901)
        list_22281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 901)
        # Adding element type (line 901)
        # Getting the type of 'lastx' (line 901)
        lastx_22282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 18), 'lastx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 901, 17), list_22281, lastx_22282)
        # Adding element type (line 901)
        # Getting the type of 'lasty' (line 901)
        lasty_22283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 25), 'lasty', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 901, 17), list_22281, lasty_22283)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 901, 16), list_22280, list_22281)
        # Adding element type (line 901)
        
        # Obtaining an instance of the builtin type 'list' (line 901)
        list_22284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 901)
        # Adding element type (line 901)
        # Getting the type of 'x' (line 901)
        x_22285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 34), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 901, 33), list_22284, x_22285)
        # Adding element type (line 901)
        # Getting the type of 'y' (line 901)
        y_22286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 37), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 901, 33), list_22284, y_22286)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 901, 16), list_22280, list_22284)
        
        # Getting the type of 'a' (line 901)
        a_22287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 42), 'a', False)
        # Obtaining the member 'bbox' of a type (line 901)
        bbox_22288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 42), a_22287, 'bbox')
        # Obtaining the member 'min' of a type (line 901)
        min_22289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 42), bbox_22288, 'min')
        # Getting the type of 'a' (line 901)
        a_22290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 54), 'a', False)
        # Obtaining the member 'bbox' of a type (line 901)
        bbox_22291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 54), a_22290, 'bbox')
        # Obtaining the member 'max' of a type (line 901)
        max_22292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 54), bbox_22291, 'max')
        # Processing the call keyword arguments (line 900)
        kwargs_22293 = {}
        # Getting the type of 'np' (line 900)
        np_22278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 33), 'np', False)
        # Obtaining the member 'clip' of a type (line 900)
        clip_22279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 33), np_22278, 'clip')
        # Calling clip(args, kwargs) (line 900)
        clip_call_result_22294 = invoke(stypy.reporting.localization.Localization(__file__, 900, 33), clip_22279, *[list_22280, min_22289, max_22292], **kwargs_22293)
        
        # Assigning a type to the variable 'call_assignment_20176' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'call_assignment_20176', clip_call_result_22294)
        
        # Assigning a Call to a Name (line 900):
        
        # Assigning a Call to a Name (line 900):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_22297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 12), 'int')
        # Processing the call keyword arguments
        kwargs_22298 = {}
        # Getting the type of 'call_assignment_20176' (line 900)
        call_assignment_20176_22295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'call_assignment_20176', False)
        # Obtaining the member '__getitem__' of a type (line 900)
        getitem___22296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 12), call_assignment_20176_22295, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_22299 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___22296, *[int_22297], **kwargs_22298)
        
        # Assigning a type to the variable 'call_assignment_20177' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'call_assignment_20177', getitem___call_result_22299)
        
        # Assigning a Name to a Tuple (line 900):
        
        # Assigning a Subscript to a Name (line 900):
        
        # Obtaining the type of the subscript
        int_22300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 12), 'int')
        # Getting the type of 'call_assignment_20177' (line 900)
        call_assignment_20177_22301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'call_assignment_20177')
        # Obtaining the member '__getitem__' of a type (line 900)
        getitem___22302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 12), call_assignment_20177_22301, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 900)
        subscript_call_result_22303 = invoke(stypy.reporting.localization.Localization(__file__, 900, 12), getitem___22302, int_22300)
        
        # Assigning a type to the variable 'tuple_var_assignment_20194' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'tuple_var_assignment_20194', subscript_call_result_22303)
        
        # Assigning a Subscript to a Name (line 900):
        
        # Obtaining the type of the subscript
        int_22304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 12), 'int')
        # Getting the type of 'call_assignment_20177' (line 900)
        call_assignment_20177_22305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'call_assignment_20177')
        # Obtaining the member '__getitem__' of a type (line 900)
        getitem___22306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 12), call_assignment_20177_22305, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 900)
        subscript_call_result_22307 = invoke(stypy.reporting.localization.Localization(__file__, 900, 12), getitem___22306, int_22304)
        
        # Assigning a type to the variable 'tuple_var_assignment_20195' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'tuple_var_assignment_20195', subscript_call_result_22307)
        
        # Assigning a Name to a Name (line 900):
        # Getting the type of 'tuple_var_assignment_20194' (line 900)
        tuple_var_assignment_20194_22308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'tuple_var_assignment_20194')
        # Assigning a type to the variable 'x1' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 13), 'x1', tuple_var_assignment_20194_22308)
        
        # Assigning a Name to a Name (line 900):
        # Getting the type of 'tuple_var_assignment_20195' (line 900)
        tuple_var_assignment_20195_22309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'tuple_var_assignment_20195')
        # Assigning a type to the variable 'y1' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 17), 'y1', tuple_var_assignment_20195_22309)
        
        # Assigning a Call to a Name (line 900):
        
        # Assigning a Call to a Name (line 900):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_22312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 12), 'int')
        # Processing the call keyword arguments
        kwargs_22313 = {}
        # Getting the type of 'call_assignment_20176' (line 900)
        call_assignment_20176_22310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'call_assignment_20176', False)
        # Obtaining the member '__getitem__' of a type (line 900)
        getitem___22311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 12), call_assignment_20176_22310, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_22314 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___22311, *[int_22312], **kwargs_22313)
        
        # Assigning a type to the variable 'call_assignment_20178' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'call_assignment_20178', getitem___call_result_22314)
        
        # Assigning a Name to a Tuple (line 900):
        
        # Assigning a Subscript to a Name (line 900):
        
        # Obtaining the type of the subscript
        int_22315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 12), 'int')
        # Getting the type of 'call_assignment_20178' (line 900)
        call_assignment_20178_22316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'call_assignment_20178')
        # Obtaining the member '__getitem__' of a type (line 900)
        getitem___22317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 12), call_assignment_20178_22316, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 900)
        subscript_call_result_22318 = invoke(stypy.reporting.localization.Localization(__file__, 900, 12), getitem___22317, int_22315)
        
        # Assigning a type to the variable 'tuple_var_assignment_20196' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'tuple_var_assignment_20196', subscript_call_result_22318)
        
        # Assigning a Subscript to a Name (line 900):
        
        # Obtaining the type of the subscript
        int_22319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 12), 'int')
        # Getting the type of 'call_assignment_20178' (line 900)
        call_assignment_20178_22320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'call_assignment_20178')
        # Obtaining the member '__getitem__' of a type (line 900)
        getitem___22321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 12), call_assignment_20178_22320, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 900)
        subscript_call_result_22322 = invoke(stypy.reporting.localization.Localization(__file__, 900, 12), getitem___22321, int_22319)
        
        # Assigning a type to the variable 'tuple_var_assignment_20197' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'tuple_var_assignment_20197', subscript_call_result_22322)
        
        # Assigning a Name to a Name (line 900):
        # Getting the type of 'tuple_var_assignment_20196' (line 900)
        tuple_var_assignment_20196_22323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'tuple_var_assignment_20196')
        # Assigning a type to the variable 'x2' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 23), 'x2', tuple_var_assignment_20196_22323)
        
        # Assigning a Name to a Name (line 900):
        # Getting the type of 'tuple_var_assignment_20197' (line 900)
        tuple_var_assignment_20197_22324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'tuple_var_assignment_20197')
        # Assigning a type to the variable 'y2' (line 900)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 27), 'y2', tuple_var_assignment_20197_22324)
        
        
        # Getting the type of 'self' (line 902)
        self_22325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 15), 'self')
        # Obtaining the member '_zoom_mode' of a type (line 902)
        _zoom_mode_22326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 15), self_22325, '_zoom_mode')
        str_22327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 34), 'str', 'x')
        # Applying the binary operator '==' (line 902)
        result_eq_22328 = python_operator(stypy.reporting.localization.Localization(__file__, 902, 15), '==', _zoom_mode_22326, str_22327)
        
        # Testing the type of an if condition (line 902)
        if_condition_22329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 902, 12), result_eq_22328)
        # Assigning a type to the variable 'if_condition_22329' (line 902)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 12), 'if_condition_22329', if_condition_22329)
        # SSA begins for if statement (line 902)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Tuple (line 903):
        
        # Assigning a Subscript to a Name (line 903):
        
        # Assigning a Subscript to a Name (line 903):
        
        # Obtaining the type of the subscript
        int_22330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 16), 'int')
        # Getting the type of 'a' (line 903)
        a_22331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 25), 'a')
        # Obtaining the member 'bbox' of a type (line 903)
        bbox_22332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 25), a_22331, 'bbox')
        # Obtaining the member 'intervaly' of a type (line 903)
        intervaly_22333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 25), bbox_22332, 'intervaly')
        # Obtaining the member '__getitem__' of a type (line 903)
        getitem___22334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 16), intervaly_22333, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 903)
        subscript_call_result_22335 = invoke(stypy.reporting.localization.Localization(__file__, 903, 16), getitem___22334, int_22330)
        
        # Assigning a type to the variable 'tuple_var_assignment_20179' (line 903)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 16), 'tuple_var_assignment_20179', subscript_call_result_22335)
        
        # Assigning a Subscript to a Name (line 903):
        
        # Assigning a Subscript to a Name (line 903):
        
        # Obtaining the type of the subscript
        int_22336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 16), 'int')
        # Getting the type of 'a' (line 903)
        a_22337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 25), 'a')
        # Obtaining the member 'bbox' of a type (line 903)
        bbox_22338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 25), a_22337, 'bbox')
        # Obtaining the member 'intervaly' of a type (line 903)
        intervaly_22339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 25), bbox_22338, 'intervaly')
        # Obtaining the member '__getitem__' of a type (line 903)
        getitem___22340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 16), intervaly_22339, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 903)
        subscript_call_result_22341 = invoke(stypy.reporting.localization.Localization(__file__, 903, 16), getitem___22340, int_22336)
        
        # Assigning a type to the variable 'tuple_var_assignment_20180' (line 903)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 16), 'tuple_var_assignment_20180', subscript_call_result_22341)
        
        # Assigning a Name to a Name (line 903):
        
        # Assigning a Name to a Name (line 903):
        # Getting the type of 'tuple_var_assignment_20179' (line 903)
        tuple_var_assignment_20179_22342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 16), 'tuple_var_assignment_20179')
        # Assigning a type to the variable 'y1' (line 903)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 16), 'y1', tuple_var_assignment_20179_22342)
        
        # Assigning a Name to a Name (line 903):
        
        # Assigning a Name to a Name (line 903):
        # Getting the type of 'tuple_var_assignment_20180' (line 903)
        tuple_var_assignment_20180_22343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 16), 'tuple_var_assignment_20180')
        # Assigning a type to the variable 'y2' (line 903)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 20), 'y2', tuple_var_assignment_20180_22343)
        # SSA branch for the else part of an if statement (line 902)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 904)
        self_22344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 17), 'self')
        # Obtaining the member '_zoom_mode' of a type (line 904)
        _zoom_mode_22345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 904, 17), self_22344, '_zoom_mode')
        str_22346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 904, 36), 'str', 'y')
        # Applying the binary operator '==' (line 904)
        result_eq_22347 = python_operator(stypy.reporting.localization.Localization(__file__, 904, 17), '==', _zoom_mode_22345, str_22346)
        
        # Testing the type of an if condition (line 904)
        if_condition_22348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 904, 17), result_eq_22347)
        # Assigning a type to the variable 'if_condition_22348' (line 904)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 904, 17), 'if_condition_22348', if_condition_22348)
        # SSA begins for if statement (line 904)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Tuple (line 905):
        
        # Assigning a Subscript to a Name (line 905):
        
        # Assigning a Subscript to a Name (line 905):
        
        # Obtaining the type of the subscript
        int_22349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 16), 'int')
        # Getting the type of 'a' (line 905)
        a_22350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 25), 'a')
        # Obtaining the member 'bbox' of a type (line 905)
        bbox_22351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 25), a_22350, 'bbox')
        # Obtaining the member 'intervalx' of a type (line 905)
        intervalx_22352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 25), bbox_22351, 'intervalx')
        # Obtaining the member '__getitem__' of a type (line 905)
        getitem___22353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 16), intervalx_22352, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 905)
        subscript_call_result_22354 = invoke(stypy.reporting.localization.Localization(__file__, 905, 16), getitem___22353, int_22349)
        
        # Assigning a type to the variable 'tuple_var_assignment_20181' (line 905)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 16), 'tuple_var_assignment_20181', subscript_call_result_22354)
        
        # Assigning a Subscript to a Name (line 905):
        
        # Assigning a Subscript to a Name (line 905):
        
        # Obtaining the type of the subscript
        int_22355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 16), 'int')
        # Getting the type of 'a' (line 905)
        a_22356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 25), 'a')
        # Obtaining the member 'bbox' of a type (line 905)
        bbox_22357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 25), a_22356, 'bbox')
        # Obtaining the member 'intervalx' of a type (line 905)
        intervalx_22358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 25), bbox_22357, 'intervalx')
        # Obtaining the member '__getitem__' of a type (line 905)
        getitem___22359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 16), intervalx_22358, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 905)
        subscript_call_result_22360 = invoke(stypy.reporting.localization.Localization(__file__, 905, 16), getitem___22359, int_22355)
        
        # Assigning a type to the variable 'tuple_var_assignment_20182' (line 905)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 16), 'tuple_var_assignment_20182', subscript_call_result_22360)
        
        # Assigning a Name to a Name (line 905):
        
        # Assigning a Name to a Name (line 905):
        # Getting the type of 'tuple_var_assignment_20181' (line 905)
        tuple_var_assignment_20181_22361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 16), 'tuple_var_assignment_20181')
        # Assigning a type to the variable 'x1' (line 905)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 16), 'x1', tuple_var_assignment_20181_22361)
        
        # Assigning a Name to a Name (line 905):
        
        # Assigning a Name to a Name (line 905):
        # Getting the type of 'tuple_var_assignment_20182' (line 905)
        tuple_var_assignment_20182_22362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 16), 'tuple_var_assignment_20182')
        # Assigning a type to the variable 'x2' (line 905)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 20), 'x2', tuple_var_assignment_20182_22362)
        # SSA join for if statement (line 904)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 902)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to trigger_tool(...): (line 906)
        # Processing the call arguments (line 906)
        str_22366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 16), 'str', 'rubberband')
        # Getting the type of 'self' (line 907)
        self_22367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 30), 'self', False)
        # Processing the call keyword arguments (line 906)
        
        # Obtaining an instance of the builtin type 'tuple' (line 907)
        tuple_22368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 907)
        # Adding element type (line 907)
        # Getting the type of 'x1' (line 907)
        x1_22369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 42), 'x1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 42), tuple_22368, x1_22369)
        # Adding element type (line 907)
        # Getting the type of 'y1' (line 907)
        y1_22370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 46), 'y1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 42), tuple_22368, y1_22370)
        # Adding element type (line 907)
        # Getting the type of 'x2' (line 907)
        x2_22371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 50), 'x2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 42), tuple_22368, x2_22371)
        # Adding element type (line 907)
        # Getting the type of 'y2' (line 907)
        y2_22372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 54), 'y2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 42), tuple_22368, y2_22372)
        
        keyword_22373 = tuple_22368
        kwargs_22374 = {'data': keyword_22373}
        # Getting the type of 'self' (line 906)
        self_22363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 12), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 906)
        toolmanager_22364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 12), self_22363, 'toolmanager')
        # Obtaining the member 'trigger_tool' of a type (line 906)
        trigger_tool_22365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 12), toolmanager_22364, 'trigger_tool')
        # Calling trigger_tool(args, kwargs) (line 906)
        trigger_tool_call_result_22375 = invoke(stypy.reporting.localization.Localization(__file__, 906, 12), trigger_tool_22365, *[str_22366, self_22367], **kwargs_22374)
        
        # SSA join for if statement (line 897)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_mouse_move(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mouse_move' in the type store
        # Getting the type of 'stypy_return_type' (line 894)
        stypy_return_type_22376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22376)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mouse_move'
        return stypy_return_type_22376


    @norecursion
    def _release(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_release'
        module_type_store = module_type_store.open_function_context('_release', 909, 4, False)
        # Assigning a type to the variable 'self' (line 910)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 910, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolZoom._release.__dict__.__setitem__('stypy_localization', localization)
        ToolZoom._release.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolZoom._release.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolZoom._release.__dict__.__setitem__('stypy_function_name', 'ToolZoom._release')
        ToolZoom._release.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolZoom._release.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolZoom._release.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolZoom._release.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolZoom._release.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolZoom._release.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolZoom._release.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolZoom._release', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_release', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_release(...)' code ##################

        str_22377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 8), 'str', 'the release mouse button callback in zoom to rect mode')
        
        # Getting the type of 'self' (line 912)
        self_22378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 23), 'self')
        # Obtaining the member '_ids_zoom' of a type (line 912)
        _ids_zoom_22379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 23), self_22378, '_ids_zoom')
        # Testing the type of a for loop iterable (line 912)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 912, 8), _ids_zoom_22379)
        # Getting the type of the for loop variable (line 912)
        for_loop_var_22380 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 912, 8), _ids_zoom_22379)
        # Assigning a type to the variable 'zoom_id' (line 912)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 912, 8), 'zoom_id', for_loop_var_22380)
        # SSA begins for a for statement (line 912)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to mpl_disconnect(...): (line 913)
        # Processing the call arguments (line 913)
        # Getting the type of 'zoom_id' (line 913)
        zoom_id_22385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 46), 'zoom_id', False)
        # Processing the call keyword arguments (line 913)
        kwargs_22386 = {}
        # Getting the type of 'self' (line 913)
        self_22381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 12), 'self', False)
        # Obtaining the member 'figure' of a type (line 913)
        figure_22382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 12), self_22381, 'figure')
        # Obtaining the member 'canvas' of a type (line 913)
        canvas_22383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 12), figure_22382, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 913)
        mpl_disconnect_22384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 12), canvas_22383, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 913)
        mpl_disconnect_call_result_22387 = invoke(stypy.reporting.localization.Localization(__file__, 913, 12), mpl_disconnect_22384, *[zoom_id_22385], **kwargs_22386)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 914):
        
        # Assigning a List to a Attribute (line 914):
        
        # Assigning a List to a Attribute (line 914):
        
        # Obtaining an instance of the builtin type 'list' (line 914)
        list_22388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 914)
        
        # Getting the type of 'self' (line 914)
        self_22389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 8), 'self')
        # Setting the type of the member '_ids_zoom' of a type (line 914)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 8), self_22389, '_ids_zoom', list_22388)
        
        
        # Getting the type of 'self' (line 916)
        self_22390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 15), 'self')
        # Obtaining the member '_xypress' of a type (line 916)
        _xypress_22391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 15), self_22390, '_xypress')
        # Applying the 'not' unary operator (line 916)
        result_not__22392 = python_operator(stypy.reporting.localization.Localization(__file__, 916, 11), 'not', _xypress_22391)
        
        # Testing the type of an if condition (line 916)
        if_condition_22393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 916, 8), result_not__22392)
        # Assigning a type to the variable 'if_condition_22393' (line 916)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 8), 'if_condition_22393', if_condition_22393)
        # SSA begins for if statement (line 916)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _cancel_action(...): (line 917)
        # Processing the call keyword arguments (line 917)
        kwargs_22396 = {}
        # Getting the type of 'self' (line 917)
        self_22394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 12), 'self', False)
        # Obtaining the member '_cancel_action' of a type (line 917)
        _cancel_action_22395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 12), self_22394, '_cancel_action')
        # Calling _cancel_action(args, kwargs) (line 917)
        _cancel_action_call_result_22397 = invoke(stypy.reporting.localization.Localization(__file__, 917, 12), _cancel_action_22395, *[], **kwargs_22396)
        
        # Assigning a type to the variable 'stypy_return_type' (line 918)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 916)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 920):
        
        # Assigning a List to a Name (line 920):
        
        # Assigning a List to a Name (line 920):
        
        # Obtaining an instance of the builtin type 'list' (line 920)
        list_22398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 920)
        
        # Assigning a type to the variable 'last_a' (line 920)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 8), 'last_a', list_22398)
        
        # Getting the type of 'self' (line 922)
        self_22399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 27), 'self')
        # Obtaining the member '_xypress' of a type (line 922)
        _xypress_22400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 27), self_22399, '_xypress')
        # Testing the type of a for loop iterable (line 922)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 922, 8), _xypress_22400)
        # Getting the type of the for loop variable (line 922)
        for_loop_var_22401 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 922, 8), _xypress_22400)
        # Assigning a type to the variable 'cur_xypress' (line 922)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'cur_xypress', for_loop_var_22401)
        # SSA begins for a for statement (line 922)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Tuple to a Tuple (line 923):
        
        # Assigning a Attribute to a Name (line 923):
        
        # Assigning a Attribute to a Name (line 923):
        # Getting the type of 'event' (line 923)
        event_22402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 19), 'event')
        # Obtaining the member 'x' of a type (line 923)
        x_22403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 19), event_22402, 'x')
        # Assigning a type to the variable 'tuple_assignment_20183' (line 923)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 12), 'tuple_assignment_20183', x_22403)
        
        # Assigning a Attribute to a Name (line 923):
        
        # Assigning a Attribute to a Name (line 923):
        # Getting the type of 'event' (line 923)
        event_22404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 28), 'event')
        # Obtaining the member 'y' of a type (line 923)
        y_22405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 28), event_22404, 'y')
        # Assigning a type to the variable 'tuple_assignment_20184' (line 923)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 12), 'tuple_assignment_20184', y_22405)
        
        # Assigning a Name to a Name (line 923):
        
        # Assigning a Name to a Name (line 923):
        # Getting the type of 'tuple_assignment_20183' (line 923)
        tuple_assignment_20183_22406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 12), 'tuple_assignment_20183')
        # Assigning a type to the variable 'x' (line 923)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 12), 'x', tuple_assignment_20183_22406)
        
        # Assigning a Name to a Name (line 923):
        
        # Assigning a Name to a Name (line 923):
        # Getting the type of 'tuple_assignment_20184' (line 923)
        tuple_assignment_20184_22407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 12), 'tuple_assignment_20184')
        # Assigning a type to the variable 'y' (line 923)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 15), 'y', tuple_assignment_20184_22407)
        
        # Assigning a Name to a Tuple (line 924):
        
        # Assigning a Subscript to a Name (line 924):
        
        # Assigning a Subscript to a Name (line 924):
        
        # Obtaining the type of the subscript
        int_22408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 12), 'int')
        # Getting the type of 'cur_xypress' (line 924)
        cur_xypress_22409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 42), 'cur_xypress')
        # Obtaining the member '__getitem__' of a type (line 924)
        getitem___22410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 12), cur_xypress_22409, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 924)
        subscript_call_result_22411 = invoke(stypy.reporting.localization.Localization(__file__, 924, 12), getitem___22410, int_22408)
        
        # Assigning a type to the variable 'tuple_var_assignment_20185' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'tuple_var_assignment_20185', subscript_call_result_22411)
        
        # Assigning a Subscript to a Name (line 924):
        
        # Assigning a Subscript to a Name (line 924):
        
        # Obtaining the type of the subscript
        int_22412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 12), 'int')
        # Getting the type of 'cur_xypress' (line 924)
        cur_xypress_22413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 42), 'cur_xypress')
        # Obtaining the member '__getitem__' of a type (line 924)
        getitem___22414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 12), cur_xypress_22413, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 924)
        subscript_call_result_22415 = invoke(stypy.reporting.localization.Localization(__file__, 924, 12), getitem___22414, int_22412)
        
        # Assigning a type to the variable 'tuple_var_assignment_20186' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'tuple_var_assignment_20186', subscript_call_result_22415)
        
        # Assigning a Subscript to a Name (line 924):
        
        # Assigning a Subscript to a Name (line 924):
        
        # Obtaining the type of the subscript
        int_22416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 12), 'int')
        # Getting the type of 'cur_xypress' (line 924)
        cur_xypress_22417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 42), 'cur_xypress')
        # Obtaining the member '__getitem__' of a type (line 924)
        getitem___22418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 12), cur_xypress_22417, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 924)
        subscript_call_result_22419 = invoke(stypy.reporting.localization.Localization(__file__, 924, 12), getitem___22418, int_22416)
        
        # Assigning a type to the variable 'tuple_var_assignment_20187' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'tuple_var_assignment_20187', subscript_call_result_22419)
        
        # Assigning a Subscript to a Name (line 924):
        
        # Assigning a Subscript to a Name (line 924):
        
        # Obtaining the type of the subscript
        int_22420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 12), 'int')
        # Getting the type of 'cur_xypress' (line 924)
        cur_xypress_22421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 42), 'cur_xypress')
        # Obtaining the member '__getitem__' of a type (line 924)
        getitem___22422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 12), cur_xypress_22421, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 924)
        subscript_call_result_22423 = invoke(stypy.reporting.localization.Localization(__file__, 924, 12), getitem___22422, int_22420)
        
        # Assigning a type to the variable 'tuple_var_assignment_20188' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'tuple_var_assignment_20188', subscript_call_result_22423)
        
        # Assigning a Subscript to a Name (line 924):
        
        # Assigning a Subscript to a Name (line 924):
        
        # Obtaining the type of the subscript
        int_22424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 12), 'int')
        # Getting the type of 'cur_xypress' (line 924)
        cur_xypress_22425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 42), 'cur_xypress')
        # Obtaining the member '__getitem__' of a type (line 924)
        getitem___22426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 12), cur_xypress_22425, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 924)
        subscript_call_result_22427 = invoke(stypy.reporting.localization.Localization(__file__, 924, 12), getitem___22426, int_22424)
        
        # Assigning a type to the variable 'tuple_var_assignment_20189' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'tuple_var_assignment_20189', subscript_call_result_22427)
        
        # Assigning a Name to a Name (line 924):
        
        # Assigning a Name to a Name (line 924):
        # Getting the type of 'tuple_var_assignment_20185' (line 924)
        tuple_var_assignment_20185_22428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'tuple_var_assignment_20185')
        # Assigning a type to the variable 'lastx' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'lastx', tuple_var_assignment_20185_22428)
        
        # Assigning a Name to a Name (line 924):
        
        # Assigning a Name to a Name (line 924):
        # Getting the type of 'tuple_var_assignment_20186' (line 924)
        tuple_var_assignment_20186_22429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'tuple_var_assignment_20186')
        # Assigning a type to the variable 'lasty' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 19), 'lasty', tuple_var_assignment_20186_22429)
        
        # Assigning a Name to a Name (line 924):
        
        # Assigning a Name to a Name (line 924):
        # Getting the type of 'tuple_var_assignment_20187' (line 924)
        tuple_var_assignment_20187_22430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'tuple_var_assignment_20187')
        # Assigning a type to the variable 'a' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 26), 'a', tuple_var_assignment_20187_22430)
        
        # Assigning a Name to a Name (line 924):
        
        # Assigning a Name to a Name (line 924):
        # Getting the type of 'tuple_var_assignment_20188' (line 924)
        tuple_var_assignment_20188_22431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'tuple_var_assignment_20188')
        # Assigning a type to the variable '_ind' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 29), '_ind', tuple_var_assignment_20188_22431)
        
        # Assigning a Name to a Name (line 924):
        
        # Assigning a Name to a Name (line 924):
        # Getting the type of 'tuple_var_assignment_20189' (line 924)
        tuple_var_assignment_20189_22432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'tuple_var_assignment_20189')
        # Assigning a type to the variable 'view' (line 924)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 35), 'view', tuple_var_assignment_20189_22432)
        
        
        # Evaluating a boolean operation
        
        
        # Call to abs(...): (line 926)
        # Processing the call arguments (line 926)
        # Getting the type of 'x' (line 926)
        x_22434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 19), 'x', False)
        # Getting the type of 'lastx' (line 926)
        lastx_22435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 23), 'lastx', False)
        # Applying the binary operator '-' (line 926)
        result_sub_22436 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 19), '-', x_22434, lastx_22435)
        
        # Processing the call keyword arguments (line 926)
        kwargs_22437 = {}
        # Getting the type of 'abs' (line 926)
        abs_22433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 926)
        abs_call_result_22438 = invoke(stypy.reporting.localization.Localization(__file__, 926, 15), abs_22433, *[result_sub_22436], **kwargs_22437)
        
        int_22439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 32), 'int')
        # Applying the binary operator '<' (line 926)
        result_lt_22440 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 15), '<', abs_call_result_22438, int_22439)
        
        
        
        # Call to abs(...): (line 926)
        # Processing the call arguments (line 926)
        # Getting the type of 'y' (line 926)
        y_22442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 41), 'y', False)
        # Getting the type of 'lasty' (line 926)
        lasty_22443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 45), 'lasty', False)
        # Applying the binary operator '-' (line 926)
        result_sub_22444 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 41), '-', y_22442, lasty_22443)
        
        # Processing the call keyword arguments (line 926)
        kwargs_22445 = {}
        # Getting the type of 'abs' (line 926)
        abs_22441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 37), 'abs', False)
        # Calling abs(args, kwargs) (line 926)
        abs_call_result_22446 = invoke(stypy.reporting.localization.Localization(__file__, 926, 37), abs_22441, *[result_sub_22444], **kwargs_22445)
        
        int_22447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 54), 'int')
        # Applying the binary operator '<' (line 926)
        result_lt_22448 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 37), '<', abs_call_result_22446, int_22447)
        
        # Applying the binary operator 'or' (line 926)
        result_or_keyword_22449 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 15), 'or', result_lt_22440, result_lt_22448)
        
        # Testing the type of an if condition (line 926)
        if_condition_22450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 926, 12), result_or_keyword_22449)
        # Assigning a type to the variable 'if_condition_22450' (line 926)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 12), 'if_condition_22450', if_condition_22450)
        # SSA begins for if statement (line 926)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _cancel_action(...): (line 927)
        # Processing the call keyword arguments (line 927)
        kwargs_22453 = {}
        # Getting the type of 'self' (line 927)
        self_22451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 16), 'self', False)
        # Obtaining the member '_cancel_action' of a type (line 927)
        _cancel_action_22452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 16), self_22451, '_cancel_action')
        # Calling _cancel_action(args, kwargs) (line 927)
        _cancel_action_call_result_22454 = invoke(stypy.reporting.localization.Localization(__file__, 927, 16), _cancel_action_22452, *[], **kwargs_22453)
        
        # Assigning a type to the variable 'stypy_return_type' (line 928)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 16), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 926)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Tuple (line 931):
        
        # Assigning a Name to a Name (line 931):
        
        # Assigning a Name to a Name (line 931):
        # Getting the type of 'False' (line 931)
        False_22455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 27), 'False')
        # Assigning a type to the variable 'tuple_assignment_20190' (line 931)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 12), 'tuple_assignment_20190', False_22455)
        
        # Assigning a Name to a Name (line 931):
        
        # Assigning a Name to a Name (line 931):
        # Getting the type of 'False' (line 931)
        False_22456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 34), 'False')
        # Assigning a type to the variable 'tuple_assignment_20191' (line 931)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 12), 'tuple_assignment_20191', False_22456)
        
        # Assigning a Name to a Name (line 931):
        
        # Assigning a Name to a Name (line 931):
        # Getting the type of 'tuple_assignment_20190' (line 931)
        tuple_assignment_20190_22457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 12), 'tuple_assignment_20190')
        # Assigning a type to the variable 'twinx' (line 931)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 12), 'twinx', tuple_assignment_20190_22457)
        
        # Assigning a Name to a Name (line 931):
        
        # Assigning a Name to a Name (line 931):
        # Getting the type of 'tuple_assignment_20191' (line 931)
        tuple_assignment_20191_22458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 12), 'tuple_assignment_20191')
        # Assigning a type to the variable 'twiny' (line 931)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 19), 'twiny', tuple_assignment_20191_22458)
        
        # Getting the type of 'last_a' (line 932)
        last_a_22459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 15), 'last_a')
        # Testing the type of an if condition (line 932)
        if_condition_22460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 932, 12), last_a_22459)
        # Assigning a type to the variable 'if_condition_22460' (line 932)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 12), 'if_condition_22460', if_condition_22460)
        # SSA begins for if statement (line 932)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'last_a' (line 933)
        last_a_22461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 26), 'last_a')
        # Testing the type of a for loop iterable (line 933)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 933, 16), last_a_22461)
        # Getting the type of the for loop variable (line 933)
        for_loop_var_22462 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 933, 16), last_a_22461)
        # Assigning a type to the variable 'la' (line 933)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 16), 'la', for_loop_var_22462)
        # SSA begins for a for statement (line 933)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to joined(...): (line 934)
        # Processing the call arguments (line 934)
        # Getting the type of 'a' (line 934)
        a_22468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 52), 'a', False)
        # Getting the type of 'la' (line 934)
        la_22469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 55), 'la', False)
        # Processing the call keyword arguments (line 934)
        kwargs_22470 = {}
        
        # Call to get_shared_x_axes(...): (line 934)
        # Processing the call keyword arguments (line 934)
        kwargs_22465 = {}
        # Getting the type of 'a' (line 934)
        a_22463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 23), 'a', False)
        # Obtaining the member 'get_shared_x_axes' of a type (line 934)
        get_shared_x_axes_22464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 23), a_22463, 'get_shared_x_axes')
        # Calling get_shared_x_axes(args, kwargs) (line 934)
        get_shared_x_axes_call_result_22466 = invoke(stypy.reporting.localization.Localization(__file__, 934, 23), get_shared_x_axes_22464, *[], **kwargs_22465)
        
        # Obtaining the member 'joined' of a type (line 934)
        joined_22467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 23), get_shared_x_axes_call_result_22466, 'joined')
        # Calling joined(args, kwargs) (line 934)
        joined_call_result_22471 = invoke(stypy.reporting.localization.Localization(__file__, 934, 23), joined_22467, *[a_22468, la_22469], **kwargs_22470)
        
        # Testing the type of an if condition (line 934)
        if_condition_22472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 934, 20), joined_call_result_22471)
        # Assigning a type to the variable 'if_condition_22472' (line 934)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 934, 20), 'if_condition_22472', if_condition_22472)
        # SSA begins for if statement (line 934)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 935):
        
        # Assigning a Name to a Name (line 935):
        
        # Assigning a Name to a Name (line 935):
        # Getting the type of 'True' (line 935)
        True_22473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 32), 'True')
        # Assigning a type to the variable 'twinx' (line 935)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 24), 'twinx', True_22473)
        # SSA join for if statement (line 934)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to joined(...): (line 936)
        # Processing the call arguments (line 936)
        # Getting the type of 'a' (line 936)
        a_22479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 52), 'a', False)
        # Getting the type of 'la' (line 936)
        la_22480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 55), 'la', False)
        # Processing the call keyword arguments (line 936)
        kwargs_22481 = {}
        
        # Call to get_shared_y_axes(...): (line 936)
        # Processing the call keyword arguments (line 936)
        kwargs_22476 = {}
        # Getting the type of 'a' (line 936)
        a_22474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 23), 'a', False)
        # Obtaining the member 'get_shared_y_axes' of a type (line 936)
        get_shared_y_axes_22475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 936, 23), a_22474, 'get_shared_y_axes')
        # Calling get_shared_y_axes(args, kwargs) (line 936)
        get_shared_y_axes_call_result_22477 = invoke(stypy.reporting.localization.Localization(__file__, 936, 23), get_shared_y_axes_22475, *[], **kwargs_22476)
        
        # Obtaining the member 'joined' of a type (line 936)
        joined_22478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 936, 23), get_shared_y_axes_call_result_22477, 'joined')
        # Calling joined(args, kwargs) (line 936)
        joined_call_result_22482 = invoke(stypy.reporting.localization.Localization(__file__, 936, 23), joined_22478, *[a_22479, la_22480], **kwargs_22481)
        
        # Testing the type of an if condition (line 936)
        if_condition_22483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 936, 20), joined_call_result_22482)
        # Assigning a type to the variable 'if_condition_22483' (line 936)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 20), 'if_condition_22483', if_condition_22483)
        # SSA begins for if statement (line 936)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 937):
        
        # Assigning a Name to a Name (line 937):
        
        # Assigning a Name to a Name (line 937):
        # Getting the type of 'True' (line 937)
        True_22484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 32), 'True')
        # Assigning a type to the variable 'twiny' (line 937)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 24), 'twiny', True_22484)
        # SSA join for if statement (line 936)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 932)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 938)
        # Processing the call arguments (line 938)
        # Getting the type of 'a' (line 938)
        a_22487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 26), 'a', False)
        # Processing the call keyword arguments (line 938)
        kwargs_22488 = {}
        # Getting the type of 'last_a' (line 938)
        last_a_22485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 12), 'last_a', False)
        # Obtaining the member 'append' of a type (line 938)
        append_22486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 938, 12), last_a_22485, 'append')
        # Calling append(args, kwargs) (line 938)
        append_call_result_22489 = invoke(stypy.reporting.localization.Localization(__file__, 938, 12), append_22486, *[a_22487], **kwargs_22488)
        
        
        
        # Getting the type of 'self' (line 940)
        self_22490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 15), 'self')
        # Obtaining the member '_button_pressed' of a type (line 940)
        _button_pressed_22491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 940, 15), self_22490, '_button_pressed')
        int_22492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 39), 'int')
        # Applying the binary operator '==' (line 940)
        result_eq_22493 = python_operator(stypy.reporting.localization.Localization(__file__, 940, 15), '==', _button_pressed_22491, int_22492)
        
        # Testing the type of an if condition (line 940)
        if_condition_22494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 940, 12), result_eq_22493)
        # Assigning a type to the variable 'if_condition_22494' (line 940)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 12), 'if_condition_22494', if_condition_22494)
        # SSA begins for if statement (line 940)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 941):
        
        # Assigning a Str to a Name (line 941):
        
        # Assigning a Str to a Name (line 941):
        str_22495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 28), 'str', 'in')
        # Assigning a type to the variable 'direction' (line 941)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 16), 'direction', str_22495)
        # SSA branch for the else part of an if statement (line 940)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 942)
        self_22496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 17), 'self')
        # Obtaining the member '_button_pressed' of a type (line 942)
        _button_pressed_22497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 942, 17), self_22496, '_button_pressed')
        int_22498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 41), 'int')
        # Applying the binary operator '==' (line 942)
        result_eq_22499 = python_operator(stypy.reporting.localization.Localization(__file__, 942, 17), '==', _button_pressed_22497, int_22498)
        
        # Testing the type of an if condition (line 942)
        if_condition_22500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 942, 17), result_eq_22499)
        # Assigning a type to the variable 'if_condition_22500' (line 942)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 17), 'if_condition_22500', if_condition_22500)
        # SSA begins for if statement (line 942)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 943):
        
        # Assigning a Str to a Name (line 943):
        
        # Assigning a Str to a Name (line 943):
        str_22501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 28), 'str', 'out')
        # Assigning a type to the variable 'direction' (line 943)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 943, 16), 'direction', str_22501)
        # SSA branch for the else part of an if statement (line 942)
        module_type_store.open_ssa_branch('else')
        # SSA join for if statement (line 942)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 940)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _set_view_from_bbox(...): (line 947)
        # Processing the call arguments (line 947)
        
        # Obtaining an instance of the builtin type 'tuple' (line 947)
        tuple_22504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 947, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 947)
        # Adding element type (line 947)
        # Getting the type of 'lastx' (line 947)
        lastx_22505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 35), 'lastx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 947, 35), tuple_22504, lastx_22505)
        # Adding element type (line 947)
        # Getting the type of 'lasty' (line 947)
        lasty_22506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 42), 'lasty', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 947, 35), tuple_22504, lasty_22506)
        # Adding element type (line 947)
        # Getting the type of 'x' (line 947)
        x_22507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 49), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 947, 35), tuple_22504, x_22507)
        # Adding element type (line 947)
        # Getting the type of 'y' (line 947)
        y_22508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 52), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 947, 35), tuple_22504, y_22508)
        
        # Getting the type of 'direction' (line 947)
        direction_22509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 56), 'direction', False)
        # Getting the type of 'self' (line 948)
        self_22510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 34), 'self', False)
        # Obtaining the member '_zoom_mode' of a type (line 948)
        _zoom_mode_22511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 34), self_22510, '_zoom_mode')
        # Getting the type of 'twinx' (line 948)
        twinx_22512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 51), 'twinx', False)
        # Getting the type of 'twiny' (line 948)
        twiny_22513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 58), 'twiny', False)
        # Processing the call keyword arguments (line 947)
        kwargs_22514 = {}
        # Getting the type of 'a' (line 947)
        a_22502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 12), 'a', False)
        # Obtaining the member '_set_view_from_bbox' of a type (line 947)
        _set_view_from_bbox_22503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 947, 12), a_22502, '_set_view_from_bbox')
        # Calling _set_view_from_bbox(args, kwargs) (line 947)
        _set_view_from_bbox_call_result_22515 = invoke(stypy.reporting.localization.Localization(__file__, 947, 12), _set_view_from_bbox_22503, *[tuple_22504, direction_22509, _zoom_mode_22511, twinx_22512, twiny_22513], **kwargs_22514)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 950):
        
        # Assigning a Name to a Attribute (line 950):
        
        # Assigning a Name to a Attribute (line 950):
        # Getting the type of 'None' (line 950)
        None_22516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 26), 'None')
        # Getting the type of 'self' (line 950)
        self_22517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 8), 'self')
        # Setting the type of the member '_zoom_mode' of a type (line 950)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 8), self_22517, '_zoom_mode', None_22516)
        
        # Call to push_current(...): (line 951)
        # Processing the call keyword arguments (line 951)
        kwargs_22525 = {}
        
        # Call to get_tool(...): (line 951)
        # Processing the call arguments (line 951)
        # Getting the type of '_views_positions' (line 951)
        _views_positions_22521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 34), '_views_positions', False)
        # Processing the call keyword arguments (line 951)
        kwargs_22522 = {}
        # Getting the type of 'self' (line 951)
        self_22518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 951)
        toolmanager_22519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 8), self_22518, 'toolmanager')
        # Obtaining the member 'get_tool' of a type (line 951)
        get_tool_22520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 8), toolmanager_22519, 'get_tool')
        # Calling get_tool(args, kwargs) (line 951)
        get_tool_call_result_22523 = invoke(stypy.reporting.localization.Localization(__file__, 951, 8), get_tool_22520, *[_views_positions_22521], **kwargs_22522)
        
        # Obtaining the member 'push_current' of a type (line 951)
        push_current_22524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 8), get_tool_call_result_22523, 'push_current')
        # Calling push_current(args, kwargs) (line 951)
        push_current_call_result_22526 = invoke(stypy.reporting.localization.Localization(__file__, 951, 8), push_current_22524, *[], **kwargs_22525)
        
        
        # Call to _cancel_action(...): (line 952)
        # Processing the call keyword arguments (line 952)
        kwargs_22529 = {}
        # Getting the type of 'self' (line 952)
        self_22527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'self', False)
        # Obtaining the member '_cancel_action' of a type (line 952)
        _cancel_action_22528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 8), self_22527, '_cancel_action')
        # Calling _cancel_action(args, kwargs) (line 952)
        _cancel_action_call_result_22530 = invoke(stypy.reporting.localization.Localization(__file__, 952, 8), _cancel_action_22528, *[], **kwargs_22529)
        
        
        # ################# End of '_release(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_release' in the type store
        # Getting the type of 'stypy_return_type' (line 909)
        stypy_return_type_22531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22531)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_release'
        return stypy_return_type_22531


# Assigning a type to the variable 'ToolZoom' (line 829)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 0), 'ToolZoom', ToolZoom)

# Assigning a Str to a Name (line 832):
str_22532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 18), 'str', 'Zoom to rectangle')
# Getting the type of 'ToolZoom'
ToolZoom_22533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolZoom')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolZoom_22533, 'description', str_22532)

# Assigning a Str to a Name (line 833):
str_22534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 12), 'str', 'zoom_to_rect.png')
# Getting the type of 'ToolZoom'
ToolZoom_22535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolZoom')
# Setting the type of the member 'image' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolZoom_22535, 'image', str_22534)

# Assigning a Subscript to a Name (line 834):

# Obtaining the type of the subscript
str_22536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 30), 'str', 'keymap.zoom')
# Getting the type of 'rcParams' (line 834)
rcParams_22537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 834)
getitem___22538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 21), rcParams_22537, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 834)
subscript_call_result_22539 = invoke(stypy.reporting.localization.Localization(__file__, 834, 21), getitem___22538, str_22536)

# Getting the type of 'ToolZoom'
ToolZoom_22540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolZoom')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolZoom_22540, 'default_keymap', subscript_call_result_22539)

# Assigning a Attribute to a Name (line 835):
# Getting the type of 'cursors' (line 835)
cursors_22541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 13), 'cursors')
# Obtaining the member 'SELECT_REGION' of a type (line 835)
SELECT_REGION_22542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 13), cursors_22541, 'SELECT_REGION')
# Getting the type of 'ToolZoom'
ToolZoom_22543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolZoom')
# Setting the type of the member 'cursor' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolZoom_22543, 'cursor', SELECT_REGION_22542)

# Assigning a Str to a Name (line 836):
str_22544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 18), 'str', 'default')
# Getting the type of 'ToolZoom'
ToolZoom_22545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolZoom')
# Setting the type of the member 'radio_group' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolZoom_22545, 'radio_group', str_22544)
# Declaration of the 'ToolPan' class
# Getting the type of 'ZoomPanBase' (line 955)
ZoomPanBase_22546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 14), 'ZoomPanBase')

class ToolPan(ZoomPanBase_22546, ):
    str_22547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 956, 4), 'str', 'Pan axes with left mouse, zoom with right')
    
    # Assigning a Subscript to a Name (line 958):
    
    # Assigning a Subscript to a Name (line 958):
    
    # Assigning a Str to a Name (line 959):
    
    # Assigning a Str to a Name (line 959):
    
    # Assigning a Str to a Name (line 960):
    
    # Assigning a Str to a Name (line 960):
    
    # Assigning a Attribute to a Name (line 961):
    
    # Assigning a Attribute to a Name (line 961):
    
    # Assigning a Str to a Name (line 962):
    
    # Assigning a Str to a Name (line 962):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 964, 4, False)
        # Assigning a type to the variable 'self' (line 965)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolPan.__init__', [], 'args', None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 965)
        # Processing the call arguments (line 965)
        # Getting the type of 'self' (line 965)
        self_22550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 29), 'self', False)
        # Getting the type of 'args' (line 965)
        args_22551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 36), 'args', False)
        # Processing the call keyword arguments (line 965)
        kwargs_22552 = {}
        # Getting the type of 'ZoomPanBase' (line 965)
        ZoomPanBase_22548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 8), 'ZoomPanBase', False)
        # Obtaining the member '__init__' of a type (line 965)
        init___22549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 8), ZoomPanBase_22548, '__init__')
        # Calling __init__(args, kwargs) (line 965)
        init___call_result_22553 = invoke(stypy.reporting.localization.Localization(__file__, 965, 8), init___22549, *[self_22550, args_22551], **kwargs_22552)
        
        
        # Assigning a Name to a Attribute (line 966):
        
        # Assigning a Name to a Attribute (line 966):
        
        # Assigning a Name to a Attribute (line 966):
        # Getting the type of 'None' (line 966)
        None_22554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 23), 'None')
        # Getting the type of 'self' (line 966)
        self_22555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 8), 'self')
        # Setting the type of the member '_idDrag' of a type (line 966)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 8), self_22555, '_idDrag', None_22554)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _cancel_action(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_cancel_action'
        module_type_store = module_type_store.open_function_context('_cancel_action', 968, 4, False)
        # Assigning a type to the variable 'self' (line 969)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolPan._cancel_action.__dict__.__setitem__('stypy_localization', localization)
        ToolPan._cancel_action.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolPan._cancel_action.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolPan._cancel_action.__dict__.__setitem__('stypy_function_name', 'ToolPan._cancel_action')
        ToolPan._cancel_action.__dict__.__setitem__('stypy_param_names_list', [])
        ToolPan._cancel_action.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolPan._cancel_action.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolPan._cancel_action.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolPan._cancel_action.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolPan._cancel_action.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolPan._cancel_action.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolPan._cancel_action', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_cancel_action', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_cancel_action(...)' code ##################

        
        # Assigning a Name to a Attribute (line 969):
        
        # Assigning a Name to a Attribute (line 969):
        
        # Assigning a Name to a Attribute (line 969):
        # Getting the type of 'None' (line 969)
        None_22556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 31), 'None')
        # Getting the type of 'self' (line 969)
        self_22557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 8), 'self')
        # Setting the type of the member '_button_pressed' of a type (line 969)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 969, 8), self_22557, '_button_pressed', None_22556)
        
        # Assigning a List to a Attribute (line 970):
        
        # Assigning a List to a Attribute (line 970):
        
        # Assigning a List to a Attribute (line 970):
        
        # Obtaining an instance of the builtin type 'list' (line 970)
        list_22558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 970)
        
        # Getting the type of 'self' (line 970)
        self_22559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 8), 'self')
        # Setting the type of the member '_xypress' of a type (line 970)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 8), self_22559, '_xypress', list_22558)
        
        # Call to mpl_disconnect(...): (line 971)
        # Processing the call arguments (line 971)
        # Getting the type of 'self' (line 971)
        self_22564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 42), 'self', False)
        # Obtaining the member '_idDrag' of a type (line 971)
        _idDrag_22565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 42), self_22564, '_idDrag')
        # Processing the call keyword arguments (line 971)
        kwargs_22566 = {}
        # Getting the type of 'self' (line 971)
        self_22560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 971)
        figure_22561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 8), self_22560, 'figure')
        # Obtaining the member 'canvas' of a type (line 971)
        canvas_22562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 8), figure_22561, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 971)
        mpl_disconnect_22563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 8), canvas_22562, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 971)
        mpl_disconnect_call_result_22567 = invoke(stypy.reporting.localization.Localization(__file__, 971, 8), mpl_disconnect_22563, *[_idDrag_22565], **kwargs_22566)
        
        
        # Call to release(...): (line 972)
        # Processing the call arguments (line 972)
        # Getting the type of 'self' (line 972)
        self_22572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 45), 'self', False)
        # Processing the call keyword arguments (line 972)
        kwargs_22573 = {}
        # Getting the type of 'self' (line 972)
        self_22568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 972)
        toolmanager_22569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 8), self_22568, 'toolmanager')
        # Obtaining the member 'messagelock' of a type (line 972)
        messagelock_22570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 8), toolmanager_22569, 'messagelock')
        # Obtaining the member 'release' of a type (line 972)
        release_22571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 8), messagelock_22570, 'release')
        # Calling release(args, kwargs) (line 972)
        release_call_result_22574 = invoke(stypy.reporting.localization.Localization(__file__, 972, 8), release_22571, *[self_22572], **kwargs_22573)
        
        
        # Call to refresh_locators(...): (line 973)
        # Processing the call keyword arguments (line 973)
        kwargs_22582 = {}
        
        # Call to get_tool(...): (line 973)
        # Processing the call arguments (line 973)
        # Getting the type of '_views_positions' (line 973)
        _views_positions_22578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 34), '_views_positions', False)
        # Processing the call keyword arguments (line 973)
        kwargs_22579 = {}
        # Getting the type of 'self' (line 973)
        self_22575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 973)
        toolmanager_22576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 8), self_22575, 'toolmanager')
        # Obtaining the member 'get_tool' of a type (line 973)
        get_tool_22577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 8), toolmanager_22576, 'get_tool')
        # Calling get_tool(args, kwargs) (line 973)
        get_tool_call_result_22580 = invoke(stypy.reporting.localization.Localization(__file__, 973, 8), get_tool_22577, *[_views_positions_22578], **kwargs_22579)
        
        # Obtaining the member 'refresh_locators' of a type (line 973)
        refresh_locators_22581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 8), get_tool_call_result_22580, 'refresh_locators')
        # Calling refresh_locators(args, kwargs) (line 973)
        refresh_locators_call_result_22583 = invoke(stypy.reporting.localization.Localization(__file__, 973, 8), refresh_locators_22581, *[], **kwargs_22582)
        
        
        # ################# End of '_cancel_action(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_cancel_action' in the type store
        # Getting the type of 'stypy_return_type' (line 968)
        stypy_return_type_22584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_cancel_action'
        return stypy_return_type_22584


    @norecursion
    def _press(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_press'
        module_type_store = module_type_store.open_function_context('_press', 975, 4, False)
        # Assigning a type to the variable 'self' (line 976)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolPan._press.__dict__.__setitem__('stypy_localization', localization)
        ToolPan._press.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolPan._press.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolPan._press.__dict__.__setitem__('stypy_function_name', 'ToolPan._press')
        ToolPan._press.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolPan._press.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolPan._press.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolPan._press.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolPan._press.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolPan._press.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolPan._press.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolPan._press', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_press', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_press(...)' code ##################

        
        
        # Getting the type of 'event' (line 976)
        event_22585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 11), 'event')
        # Obtaining the member 'button' of a type (line 976)
        button_22586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 976, 11), event_22585, 'button')
        int_22587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 27), 'int')
        # Applying the binary operator '==' (line 976)
        result_eq_22588 = python_operator(stypy.reporting.localization.Localization(__file__, 976, 11), '==', button_22586, int_22587)
        
        # Testing the type of an if condition (line 976)
        if_condition_22589 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 976, 8), result_eq_22588)
        # Assigning a type to the variable 'if_condition_22589' (line 976)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 8), 'if_condition_22589', if_condition_22589)
        # SSA begins for if statement (line 976)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 977):
        
        # Assigning a Num to a Attribute (line 977):
        
        # Assigning a Num to a Attribute (line 977):
        int_22590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 35), 'int')
        # Getting the type of 'self' (line 977)
        self_22591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 12), 'self')
        # Setting the type of the member '_button_pressed' of a type (line 977)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 977, 12), self_22591, '_button_pressed', int_22590)
        # SSA branch for the else part of an if statement (line 976)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'event' (line 978)
        event_22592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 13), 'event')
        # Obtaining the member 'button' of a type (line 978)
        button_22593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 978, 13), event_22592, 'button')
        int_22594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 29), 'int')
        # Applying the binary operator '==' (line 978)
        result_eq_22595 = python_operator(stypy.reporting.localization.Localization(__file__, 978, 13), '==', button_22593, int_22594)
        
        # Testing the type of an if condition (line 978)
        if_condition_22596 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 978, 13), result_eq_22595)
        # Assigning a type to the variable 'if_condition_22596' (line 978)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 13), 'if_condition_22596', if_condition_22596)
        # SSA begins for if statement (line 978)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 979):
        
        # Assigning a Num to a Attribute (line 979):
        
        # Assigning a Num to a Attribute (line 979):
        int_22597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 35), 'int')
        # Getting the type of 'self' (line 979)
        self_22598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 12), 'self')
        # Setting the type of the member '_button_pressed' of a type (line 979)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 12), self_22598, '_button_pressed', int_22597)
        # SSA branch for the else part of an if statement (line 978)
        module_type_store.open_ssa_branch('else')
        
        # Call to _cancel_action(...): (line 981)
        # Processing the call keyword arguments (line 981)
        kwargs_22601 = {}
        # Getting the type of 'self' (line 981)
        self_22599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 12), 'self', False)
        # Obtaining the member '_cancel_action' of a type (line 981)
        _cancel_action_22600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 12), self_22599, '_cancel_action')
        # Calling _cancel_action(args, kwargs) (line 981)
        _cancel_action_call_result_22602 = invoke(stypy.reporting.localization.Localization(__file__, 981, 12), _cancel_action_22600, *[], **kwargs_22601)
        
        # Assigning a type to the variable 'stypy_return_type' (line 982)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 978)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 976)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Tuple (line 984):
        
        # Assigning a Attribute to a Name (line 984):
        
        # Assigning a Attribute to a Name (line 984):
        # Getting the type of 'event' (line 984)
        event_22603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 15), 'event')
        # Obtaining the member 'x' of a type (line 984)
        x_22604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 984, 15), event_22603, 'x')
        # Assigning a type to the variable 'tuple_assignment_20192' (line 984)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 8), 'tuple_assignment_20192', x_22604)
        
        # Assigning a Attribute to a Name (line 984):
        
        # Assigning a Attribute to a Name (line 984):
        # Getting the type of 'event' (line 984)
        event_22605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 24), 'event')
        # Obtaining the member 'y' of a type (line 984)
        y_22606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 984, 24), event_22605, 'y')
        # Assigning a type to the variable 'tuple_assignment_20193' (line 984)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 8), 'tuple_assignment_20193', y_22606)
        
        # Assigning a Name to a Name (line 984):
        
        # Assigning a Name to a Name (line 984):
        # Getting the type of 'tuple_assignment_20192' (line 984)
        tuple_assignment_20192_22607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 8), 'tuple_assignment_20192')
        # Assigning a type to the variable 'x' (line 984)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 8), 'x', tuple_assignment_20192_22607)
        
        # Assigning a Name to a Name (line 984):
        
        # Assigning a Name to a Name (line 984):
        # Getting the type of 'tuple_assignment_20193' (line 984)
        tuple_assignment_20193_22608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 8), 'tuple_assignment_20193')
        # Assigning a type to the variable 'y' (line 984)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 11), 'y', tuple_assignment_20193_22608)
        
        # Assigning a List to a Attribute (line 986):
        
        # Assigning a List to a Attribute (line 986):
        
        # Assigning a List to a Attribute (line 986):
        
        # Obtaining an instance of the builtin type 'list' (line 986)
        list_22609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 986)
        
        # Getting the type of 'self' (line 986)
        self_22610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 8), 'self')
        # Setting the type of the member '_xypress' of a type (line 986)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 986, 8), self_22610, '_xypress', list_22609)
        
        
        # Call to enumerate(...): (line 987)
        # Processing the call arguments (line 987)
        
        # Call to get_axes(...): (line 987)
        # Processing the call keyword arguments (line 987)
        kwargs_22615 = {}
        # Getting the type of 'self' (line 987)
        self_22612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 30), 'self', False)
        # Obtaining the member 'figure' of a type (line 987)
        figure_22613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 987, 30), self_22612, 'figure')
        # Obtaining the member 'get_axes' of a type (line 987)
        get_axes_22614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 987, 30), figure_22613, 'get_axes')
        # Calling get_axes(args, kwargs) (line 987)
        get_axes_call_result_22616 = invoke(stypy.reporting.localization.Localization(__file__, 987, 30), get_axes_22614, *[], **kwargs_22615)
        
        # Processing the call keyword arguments (line 987)
        kwargs_22617 = {}
        # Getting the type of 'enumerate' (line 987)
        enumerate_22611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 20), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 987)
        enumerate_call_result_22618 = invoke(stypy.reporting.localization.Localization(__file__, 987, 20), enumerate_22611, *[get_axes_call_result_22616], **kwargs_22617)
        
        # Testing the type of a for loop iterable (line 987)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 987, 8), enumerate_call_result_22618)
        # Getting the type of the for loop variable (line 987)
        for_loop_var_22619 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 987, 8), enumerate_call_result_22618)
        # Assigning a type to the variable 'i' (line 987)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 987, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 987, 8), for_loop_var_22619))
        # Assigning a type to the variable 'a' (line 987)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 987, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 987, 8), for_loop_var_22619))
        # SSA begins for a for statement (line 987)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 988)
        x_22620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 16), 'x')
        # Getting the type of 'None' (line 988)
        None_22621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 25), 'None')
        # Applying the binary operator 'isnot' (line 988)
        result_is_not_22622 = python_operator(stypy.reporting.localization.Localization(__file__, 988, 16), 'isnot', x_22620, None_22621)
        
        
        # Getting the type of 'y' (line 988)
        y_22623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 34), 'y')
        # Getting the type of 'None' (line 988)
        None_22624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 43), 'None')
        # Applying the binary operator 'isnot' (line 988)
        result_is_not_22625 = python_operator(stypy.reporting.localization.Localization(__file__, 988, 34), 'isnot', y_22623, None_22624)
        
        # Applying the binary operator 'and' (line 988)
        result_and_keyword_22626 = python_operator(stypy.reporting.localization.Localization(__file__, 988, 16), 'and', result_is_not_22622, result_is_not_22625)
        
        # Call to in_axes(...): (line 988)
        # Processing the call arguments (line 988)
        # Getting the type of 'event' (line 988)
        event_22629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 62), 'event', False)
        # Processing the call keyword arguments (line 988)
        kwargs_22630 = {}
        # Getting the type of 'a' (line 988)
        a_22627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 52), 'a', False)
        # Obtaining the member 'in_axes' of a type (line 988)
        in_axes_22628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 988, 52), a_22627, 'in_axes')
        # Calling in_axes(args, kwargs) (line 988)
        in_axes_call_result_22631 = invoke(stypy.reporting.localization.Localization(__file__, 988, 52), in_axes_22628, *[event_22629], **kwargs_22630)
        
        # Applying the binary operator 'and' (line 988)
        result_and_keyword_22632 = python_operator(stypy.reporting.localization.Localization(__file__, 988, 16), 'and', result_and_keyword_22626, in_axes_call_result_22631)
        
        # Call to get_navigate(...): (line 989)
        # Processing the call keyword arguments (line 989)
        kwargs_22635 = {}
        # Getting the type of 'a' (line 989)
        a_22633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 20), 'a', False)
        # Obtaining the member 'get_navigate' of a type (line 989)
        get_navigate_22634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 989, 20), a_22633, 'get_navigate')
        # Calling get_navigate(args, kwargs) (line 989)
        get_navigate_call_result_22636 = invoke(stypy.reporting.localization.Localization(__file__, 989, 20), get_navigate_22634, *[], **kwargs_22635)
        
        # Applying the binary operator 'and' (line 988)
        result_and_keyword_22637 = python_operator(stypy.reporting.localization.Localization(__file__, 988, 16), 'and', result_and_keyword_22632, get_navigate_call_result_22636)
        
        # Call to can_pan(...): (line 989)
        # Processing the call keyword arguments (line 989)
        kwargs_22640 = {}
        # Getting the type of 'a' (line 989)
        a_22638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 41), 'a', False)
        # Obtaining the member 'can_pan' of a type (line 989)
        can_pan_22639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 989, 41), a_22638, 'can_pan')
        # Calling can_pan(args, kwargs) (line 989)
        can_pan_call_result_22641 = invoke(stypy.reporting.localization.Localization(__file__, 989, 41), can_pan_22639, *[], **kwargs_22640)
        
        # Applying the binary operator 'and' (line 988)
        result_and_keyword_22642 = python_operator(stypy.reporting.localization.Localization(__file__, 988, 16), 'and', result_and_keyword_22637, can_pan_call_result_22641)
        
        # Testing the type of an if condition (line 988)
        if_condition_22643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 988, 12), result_and_keyword_22642)
        # Assigning a type to the variable 'if_condition_22643' (line 988)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 12), 'if_condition_22643', if_condition_22643)
        # SSA begins for if statement (line 988)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to start_pan(...): (line 990)
        # Processing the call arguments (line 990)
        # Getting the type of 'x' (line 990)
        x_22646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 28), 'x', False)
        # Getting the type of 'y' (line 990)
        y_22647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 31), 'y', False)
        # Getting the type of 'event' (line 990)
        event_22648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 34), 'event', False)
        # Obtaining the member 'button' of a type (line 990)
        button_22649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 990, 34), event_22648, 'button')
        # Processing the call keyword arguments (line 990)
        kwargs_22650 = {}
        # Getting the type of 'a' (line 990)
        a_22644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 16), 'a', False)
        # Obtaining the member 'start_pan' of a type (line 990)
        start_pan_22645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 990, 16), a_22644, 'start_pan')
        # Calling start_pan(args, kwargs) (line 990)
        start_pan_call_result_22651 = invoke(stypy.reporting.localization.Localization(__file__, 990, 16), start_pan_22645, *[x_22646, y_22647, button_22649], **kwargs_22650)
        
        
        # Call to append(...): (line 991)
        # Processing the call arguments (line 991)
        
        # Obtaining an instance of the builtin type 'tuple' (line 991)
        tuple_22655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 991)
        # Adding element type (line 991)
        # Getting the type of 'a' (line 991)
        a_22656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 38), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 991, 38), tuple_22655, a_22656)
        # Adding element type (line 991)
        # Getting the type of 'i' (line 991)
        i_22657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 41), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 991, 38), tuple_22655, i_22657)
        
        # Processing the call keyword arguments (line 991)
        kwargs_22658 = {}
        # Getting the type of 'self' (line 991)
        self_22652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 16), 'self', False)
        # Obtaining the member '_xypress' of a type (line 991)
        _xypress_22653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 16), self_22652, '_xypress')
        # Obtaining the member 'append' of a type (line 991)
        append_22654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 16), _xypress_22653, 'append')
        # Calling append(args, kwargs) (line 991)
        append_call_result_22659 = invoke(stypy.reporting.localization.Localization(__file__, 991, 16), append_22654, *[tuple_22655], **kwargs_22658)
        
        
        # Call to messagelock(...): (line 992)
        # Processing the call arguments (line 992)
        # Getting the type of 'self' (line 992)
        self_22663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 992, 45), 'self', False)
        # Processing the call keyword arguments (line 992)
        kwargs_22664 = {}
        # Getting the type of 'self' (line 992)
        self_22660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 992, 16), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 992)
        toolmanager_22661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 992, 16), self_22660, 'toolmanager')
        # Obtaining the member 'messagelock' of a type (line 992)
        messagelock_22662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 992, 16), toolmanager_22661, 'messagelock')
        # Calling messagelock(args, kwargs) (line 992)
        messagelock_call_result_22665 = invoke(stypy.reporting.localization.Localization(__file__, 992, 16), messagelock_22662, *[self_22663], **kwargs_22664)
        
        
        # Assigning a Call to a Attribute (line 993):
        
        # Assigning a Call to a Attribute (line 993):
        
        # Assigning a Call to a Attribute (line 993):
        
        # Call to mpl_connect(...): (line 993)
        # Processing the call arguments (line 993)
        str_22670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 20), 'str', 'motion_notify_event')
        # Getting the type of 'self' (line 994)
        self_22671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 43), 'self', False)
        # Obtaining the member '_mouse_move' of a type (line 994)
        _mouse_move_22672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 994, 43), self_22671, '_mouse_move')
        # Processing the call keyword arguments (line 993)
        kwargs_22673 = {}
        # Getting the type of 'self' (line 993)
        self_22666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 31), 'self', False)
        # Obtaining the member 'figure' of a type (line 993)
        figure_22667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 31), self_22666, 'figure')
        # Obtaining the member 'canvas' of a type (line 993)
        canvas_22668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 31), figure_22667, 'canvas')
        # Obtaining the member 'mpl_connect' of a type (line 993)
        mpl_connect_22669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 31), canvas_22668, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 993)
        mpl_connect_call_result_22674 = invoke(stypy.reporting.localization.Localization(__file__, 993, 31), mpl_connect_22669, *[str_22670, _mouse_move_22672], **kwargs_22673)
        
        # Getting the type of 'self' (line 993)
        self_22675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 16), 'self')
        # Setting the type of the member '_idDrag' of a type (line 993)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 16), self_22675, '_idDrag', mpl_connect_call_result_22674)
        # SSA join for if statement (line 988)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_press(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_press' in the type store
        # Getting the type of 'stypy_return_type' (line 975)
        stypy_return_type_22676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22676)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_press'
        return stypy_return_type_22676


    @norecursion
    def _release(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_release'
        module_type_store = module_type_store.open_function_context('_release', 996, 4, False)
        # Assigning a type to the variable 'self' (line 997)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolPan._release.__dict__.__setitem__('stypy_localization', localization)
        ToolPan._release.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolPan._release.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolPan._release.__dict__.__setitem__('stypy_function_name', 'ToolPan._release')
        ToolPan._release.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolPan._release.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolPan._release.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolPan._release.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolPan._release.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolPan._release.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolPan._release.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolPan._release', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_release', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_release(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 997)
        # Getting the type of 'self' (line 997)
        self_22677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 11), 'self')
        # Obtaining the member '_button_pressed' of a type (line 997)
        _button_pressed_22678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 997, 11), self_22677, '_button_pressed')
        # Getting the type of 'None' (line 997)
        None_22679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 35), 'None')
        
        (may_be_22680, more_types_in_union_22681) = may_be_none(_button_pressed_22678, None_22679)

        if may_be_22680:

            if more_types_in_union_22681:
                # Runtime conditional SSA (line 997)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _cancel_action(...): (line 998)
            # Processing the call keyword arguments (line 998)
            kwargs_22684 = {}
            # Getting the type of 'self' (line 998)
            self_22682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 12), 'self', False)
            # Obtaining the member '_cancel_action' of a type (line 998)
            _cancel_action_22683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 998, 12), self_22682, '_cancel_action')
            # Calling _cancel_action(args, kwargs) (line 998)
            _cancel_action_call_result_22685 = invoke(stypy.reporting.localization.Localization(__file__, 998, 12), _cancel_action_22683, *[], **kwargs_22684)
            
            # Assigning a type to the variable 'stypy_return_type' (line 999)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 999, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_22681:
                # SSA join for if statement (line 997)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to mpl_disconnect(...): (line 1001)
        # Processing the call arguments (line 1001)
        # Getting the type of 'self' (line 1001)
        self_22690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 42), 'self', False)
        # Obtaining the member '_idDrag' of a type (line 1001)
        _idDrag_22691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 42), self_22690, '_idDrag')
        # Processing the call keyword arguments (line 1001)
        kwargs_22692 = {}
        # Getting the type of 'self' (line 1001)
        self_22686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 1001)
        figure_22687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 8), self_22686, 'figure')
        # Obtaining the member 'canvas' of a type (line 1001)
        canvas_22688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 8), figure_22687, 'canvas')
        # Obtaining the member 'mpl_disconnect' of a type (line 1001)
        mpl_disconnect_22689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 8), canvas_22688, 'mpl_disconnect')
        # Calling mpl_disconnect(args, kwargs) (line 1001)
        mpl_disconnect_call_result_22693 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 8), mpl_disconnect_22689, *[_idDrag_22691], **kwargs_22692)
        
        
        # Call to release(...): (line 1002)
        # Processing the call arguments (line 1002)
        # Getting the type of 'self' (line 1002)
        self_22698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 45), 'self', False)
        # Processing the call keyword arguments (line 1002)
        kwargs_22699 = {}
        # Getting the type of 'self' (line 1002)
        self_22694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 1002)
        toolmanager_22695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 8), self_22694, 'toolmanager')
        # Obtaining the member 'messagelock' of a type (line 1002)
        messagelock_22696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 8), toolmanager_22695, 'messagelock')
        # Obtaining the member 'release' of a type (line 1002)
        release_22697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 8), messagelock_22696, 'release')
        # Calling release(args, kwargs) (line 1002)
        release_call_result_22700 = invoke(stypy.reporting.localization.Localization(__file__, 1002, 8), release_22697, *[self_22698], **kwargs_22699)
        
        
        # Getting the type of 'self' (line 1004)
        self_22701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 23), 'self')
        # Obtaining the member '_xypress' of a type (line 1004)
        _xypress_22702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1004, 23), self_22701, '_xypress')
        # Testing the type of a for loop iterable (line 1004)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1004, 8), _xypress_22702)
        # Getting the type of the for loop variable (line 1004)
        for_loop_var_22703 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1004, 8), _xypress_22702)
        # Assigning a type to the variable 'a' (line 1004)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1004, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1004, 8), for_loop_var_22703))
        # Assigning a type to the variable '_ind' (line 1004)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1004, 8), '_ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1004, 8), for_loop_var_22703))
        # SSA begins for a for statement (line 1004)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to end_pan(...): (line 1005)
        # Processing the call keyword arguments (line 1005)
        kwargs_22706 = {}
        # Getting the type of 'a' (line 1005)
        a_22704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 12), 'a', False)
        # Obtaining the member 'end_pan' of a type (line 1005)
        end_pan_22705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1005, 12), a_22704, 'end_pan')
        # Calling end_pan(args, kwargs) (line 1005)
        end_pan_call_result_22707 = invoke(stypy.reporting.localization.Localization(__file__, 1005, 12), end_pan_22705, *[], **kwargs_22706)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 1006)
        self_22708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 15), 'self')
        # Obtaining the member '_xypress' of a type (line 1006)
        _xypress_22709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1006, 15), self_22708, '_xypress')
        # Applying the 'not' unary operator (line 1006)
        result_not__22710 = python_operator(stypy.reporting.localization.Localization(__file__, 1006, 11), 'not', _xypress_22709)
        
        # Testing the type of an if condition (line 1006)
        if_condition_22711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1006, 8), result_not__22710)
        # Assigning a type to the variable 'if_condition_22711' (line 1006)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1006, 8), 'if_condition_22711', if_condition_22711)
        # SSA begins for if statement (line 1006)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _cancel_action(...): (line 1007)
        # Processing the call keyword arguments (line 1007)
        kwargs_22714 = {}
        # Getting the type of 'self' (line 1007)
        self_22712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 12), 'self', False)
        # Obtaining the member '_cancel_action' of a type (line 1007)
        _cancel_action_22713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 12), self_22712, '_cancel_action')
        # Calling _cancel_action(args, kwargs) (line 1007)
        _cancel_action_call_result_22715 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 12), _cancel_action_22713, *[], **kwargs_22714)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1008)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1008, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 1006)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to push_current(...): (line 1010)
        # Processing the call keyword arguments (line 1010)
        kwargs_22723 = {}
        
        # Call to get_tool(...): (line 1010)
        # Processing the call arguments (line 1010)
        # Getting the type of '_views_positions' (line 1010)
        _views_positions_22719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 34), '_views_positions', False)
        # Processing the call keyword arguments (line 1010)
        kwargs_22720 = {}
        # Getting the type of 'self' (line 1010)
        self_22716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 1010)
        toolmanager_22717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1010, 8), self_22716, 'toolmanager')
        # Obtaining the member 'get_tool' of a type (line 1010)
        get_tool_22718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1010, 8), toolmanager_22717, 'get_tool')
        # Calling get_tool(args, kwargs) (line 1010)
        get_tool_call_result_22721 = invoke(stypy.reporting.localization.Localization(__file__, 1010, 8), get_tool_22718, *[_views_positions_22719], **kwargs_22720)
        
        # Obtaining the member 'push_current' of a type (line 1010)
        push_current_22722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1010, 8), get_tool_call_result_22721, 'push_current')
        # Calling push_current(args, kwargs) (line 1010)
        push_current_call_result_22724 = invoke(stypy.reporting.localization.Localization(__file__, 1010, 8), push_current_22722, *[], **kwargs_22723)
        
        
        # Call to _cancel_action(...): (line 1011)
        # Processing the call keyword arguments (line 1011)
        kwargs_22727 = {}
        # Getting the type of 'self' (line 1011)
        self_22725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 8), 'self', False)
        # Obtaining the member '_cancel_action' of a type (line 1011)
        _cancel_action_22726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1011, 8), self_22725, '_cancel_action')
        # Calling _cancel_action(args, kwargs) (line 1011)
        _cancel_action_call_result_22728 = invoke(stypy.reporting.localization.Localization(__file__, 1011, 8), _cancel_action_22726, *[], **kwargs_22727)
        
        
        # ################# End of '_release(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_release' in the type store
        # Getting the type of 'stypy_return_type' (line 996)
        stypy_return_type_22729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22729)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_release'
        return stypy_return_type_22729


    @norecursion
    def _mouse_move(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mouse_move'
        module_type_store = module_type_store.open_function_context('_mouse_move', 1013, 4, False)
        # Assigning a type to the variable 'self' (line 1014)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1014, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolPan._mouse_move.__dict__.__setitem__('stypy_localization', localization)
        ToolPan._mouse_move.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolPan._mouse_move.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolPan._mouse_move.__dict__.__setitem__('stypy_function_name', 'ToolPan._mouse_move')
        ToolPan._mouse_move.__dict__.__setitem__('stypy_param_names_list', ['event'])
        ToolPan._mouse_move.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolPan._mouse_move.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolPan._mouse_move.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolPan._mouse_move.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolPan._mouse_move.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolPan._mouse_move.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolPan._mouse_move', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mouse_move', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mouse_move(...)' code ##################

        
        # Getting the type of 'self' (line 1014)
        self_22730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 23), 'self')
        # Obtaining the member '_xypress' of a type (line 1014)
        _xypress_22731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1014, 23), self_22730, '_xypress')
        # Testing the type of a for loop iterable (line 1014)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1014, 8), _xypress_22731)
        # Getting the type of the for loop variable (line 1014)
        for_loop_var_22732 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1014, 8), _xypress_22731)
        # Assigning a type to the variable 'a' (line 1014)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1014, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1014, 8), for_loop_var_22732))
        # Assigning a type to the variable '_ind' (line 1014)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1014, 8), '_ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1014, 8), for_loop_var_22732))
        # SSA begins for a for statement (line 1014)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to drag_pan(...): (line 1017)
        # Processing the call arguments (line 1017)
        # Getting the type of 'self' (line 1017)
        self_22735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 23), 'self', False)
        # Obtaining the member '_button_pressed' of a type (line 1017)
        _button_pressed_22736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 23), self_22735, '_button_pressed')
        # Getting the type of 'event' (line 1017)
        event_22737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 45), 'event', False)
        # Obtaining the member 'key' of a type (line 1017)
        key_22738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 45), event_22737, 'key')
        # Getting the type of 'event' (line 1017)
        event_22739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 56), 'event', False)
        # Obtaining the member 'x' of a type (line 1017)
        x_22740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 56), event_22739, 'x')
        # Getting the type of 'event' (line 1017)
        event_22741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 65), 'event', False)
        # Obtaining the member 'y' of a type (line 1017)
        y_22742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 65), event_22741, 'y')
        # Processing the call keyword arguments (line 1017)
        kwargs_22743 = {}
        # Getting the type of 'a' (line 1017)
        a_22733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 12), 'a', False)
        # Obtaining the member 'drag_pan' of a type (line 1017)
        drag_pan_22734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 12), a_22733, 'drag_pan')
        # Calling drag_pan(args, kwargs) (line 1017)
        drag_pan_call_result_22744 = invoke(stypy.reporting.localization.Localization(__file__, 1017, 12), drag_pan_22734, *[_button_pressed_22736, key_22738, x_22740, y_22742], **kwargs_22743)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw_idle(...): (line 1018)
        # Processing the call keyword arguments (line 1018)
        kwargs_22749 = {}
        # Getting the type of 'self' (line 1018)
        self_22745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 8), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 1018)
        toolmanager_22746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1018, 8), self_22745, 'toolmanager')
        # Obtaining the member 'canvas' of a type (line 1018)
        canvas_22747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1018, 8), toolmanager_22746, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 1018)
        draw_idle_22748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1018, 8), canvas_22747, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 1018)
        draw_idle_call_result_22750 = invoke(stypy.reporting.localization.Localization(__file__, 1018, 8), draw_idle_22748, *[], **kwargs_22749)
        
        
        # ################# End of '_mouse_move(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mouse_move' in the type store
        # Getting the type of 'stypy_return_type' (line 1013)
        stypy_return_type_22751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22751)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mouse_move'
        return stypy_return_type_22751


# Assigning a type to the variable 'ToolPan' (line 955)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 0), 'ToolPan', ToolPan)

# Assigning a Subscript to a Name (line 958):

# Obtaining the type of the subscript
str_22752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 30), 'str', 'keymap.pan')
# Getting the type of 'rcParams' (line 958)
rcParams_22753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 21), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 958)
getitem___22754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 958, 21), rcParams_22753, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 958)
subscript_call_result_22755 = invoke(stypy.reporting.localization.Localization(__file__, 958, 21), getitem___22754, str_22752)

# Getting the type of 'ToolPan'
ToolPan_22756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolPan')
# Setting the type of the member 'default_keymap' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolPan_22756, 'default_keymap', subscript_call_result_22755)

# Assigning a Str to a Name (line 959):
str_22757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 18), 'str', 'Pan axes with left mouse, zoom with right')
# Getting the type of 'ToolPan'
ToolPan_22758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolPan')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolPan_22758, 'description', str_22757)

# Assigning a Str to a Name (line 960):
str_22759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 12), 'str', 'move.png')
# Getting the type of 'ToolPan'
ToolPan_22760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolPan')
# Setting the type of the member 'image' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolPan_22760, 'image', str_22759)

# Assigning a Attribute to a Name (line 961):
# Getting the type of 'cursors' (line 961)
cursors_22761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 13), 'cursors')
# Obtaining the member 'MOVE' of a type (line 961)
MOVE_22762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 13), cursors_22761, 'MOVE')
# Getting the type of 'ToolPan'
ToolPan_22763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolPan')
# Setting the type of the member 'cursor' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolPan_22763, 'cursor', MOVE_22762)

# Assigning a Str to a Name (line 962):
str_22764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 18), 'str', 'default')
# Getting the type of 'ToolPan'
ToolPan_22765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ToolPan')
# Setting the type of the member 'radio_group' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ToolPan_22765, 'radio_group', str_22764)

# Assigning a Dict to a Name (line 1021):

# Assigning a Dict to a Name (line 1021):

# Assigning a Dict to a Name (line 1021):

# Obtaining an instance of the builtin type 'dict' (line 1021)
dict_22766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1021)
# Adding element type (key, value) (line 1021)
str_22767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 17), 'str', 'home')
# Getting the type of 'ToolHome' (line 1021)
ToolHome_22768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 25), 'ToolHome')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22767, ToolHome_22768))
# Adding element type (key, value) (line 1021)
str_22769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 35), 'str', 'back')
# Getting the type of 'ToolBack' (line 1021)
ToolBack_22770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 43), 'ToolBack')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22769, ToolBack_22770))
# Adding element type (key, value) (line 1021)
str_22771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 53), 'str', 'forward')
# Getting the type of 'ToolForward' (line 1021)
ToolForward_22772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 64), 'ToolForward')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22771, ToolForward_22772))
# Adding element type (key, value) (line 1021)
str_22773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1022, 17), 'str', 'zoom')
# Getting the type of 'ToolZoom' (line 1022)
ToolZoom_22774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 25), 'ToolZoom')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22773, ToolZoom_22774))
# Adding element type (key, value) (line 1021)
str_22775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1022, 35), 'str', 'pan')
# Getting the type of 'ToolPan' (line 1022)
ToolPan_22776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 42), 'ToolPan')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22775, ToolPan_22776))
# Adding element type (key, value) (line 1021)
str_22777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 17), 'str', 'subplots')
str_22778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 29), 'str', 'ToolConfigureSubplots')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22777, str_22778))
# Adding element type (key, value) (line 1021)
str_22779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1024, 17), 'str', 'save')
str_22780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1024, 25), 'str', 'ToolSaveFigure')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22779, str_22780))
# Adding element type (key, value) (line 1021)
str_22781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1025, 17), 'str', 'grid')
# Getting the type of 'ToolGrid' (line 1025)
ToolGrid_22782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 25), 'ToolGrid')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22781, ToolGrid_22782))
# Adding element type (key, value) (line 1021)
str_22783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1026, 17), 'str', 'grid_minor')
# Getting the type of 'ToolMinorGrid' (line 1026)
ToolMinorGrid_22784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 31), 'ToolMinorGrid')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22783, ToolMinorGrid_22784))
# Adding element type (key, value) (line 1021)
str_22785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 17), 'str', 'fullscreen')
# Getting the type of 'ToolFullScreen' (line 1027)
ToolFullScreen_22786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 31), 'ToolFullScreen')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22785, ToolFullScreen_22786))
# Adding element type (key, value) (line 1021)
str_22787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1028, 17), 'str', 'quit')
# Getting the type of 'ToolQuit' (line 1028)
ToolQuit_22788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 25), 'ToolQuit')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22787, ToolQuit_22788))
# Adding element type (key, value) (line 1021)
str_22789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, 17), 'str', 'quit_all')
# Getting the type of 'ToolQuitAll' (line 1029)
ToolQuitAll_22790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 29), 'ToolQuitAll')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22789, ToolQuitAll_22790))
# Adding element type (key, value) (line 1021)
str_22791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 17), 'str', 'allnav')
# Getting the type of 'ToolEnableAllNavigation' (line 1030)
ToolEnableAllNavigation_22792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 27), 'ToolEnableAllNavigation')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22791, ToolEnableAllNavigation_22792))
# Adding element type (key, value) (line 1021)
str_22793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 17), 'str', 'nav')
# Getting the type of 'ToolEnableNavigation' (line 1031)
ToolEnableNavigation_22794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 24), 'ToolEnableNavigation')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22793, ToolEnableNavigation_22794))
# Adding element type (key, value) (line 1021)
str_22795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 17), 'str', 'xscale')
# Getting the type of 'ToolXScale' (line 1032)
ToolXScale_22796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 27), 'ToolXScale')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22795, ToolXScale_22796))
# Adding element type (key, value) (line 1021)
str_22797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 17), 'str', 'yscale')
# Getting the type of 'ToolYScale' (line 1033)
ToolYScale_22798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 27), 'ToolYScale')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22797, ToolYScale_22798))
# Adding element type (key, value) (line 1021)
str_22799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 17), 'str', 'position')
# Getting the type of 'ToolCursorPosition' (line 1034)
ToolCursorPosition_22800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 29), 'ToolCursorPosition')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22799, ToolCursorPosition_22800))
# Adding element type (key, value) (line 1021)
# Getting the type of '_views_positions' (line 1035)
_views_positions_22801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 17), '_views_positions')
# Getting the type of 'ToolViewsPositions' (line 1035)
ToolViewsPositions_22802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 35), 'ToolViewsPositions')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (_views_positions_22801, ToolViewsPositions_22802))
# Adding element type (key, value) (line 1021)
str_22803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 17), 'str', 'cursor')
str_22804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 27), 'str', 'ToolSetCursor')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22803, str_22804))
# Adding element type (key, value) (line 1021)
str_22805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 17), 'str', 'rubberband')
str_22806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 31), 'str', 'ToolRubberband')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1021, 16), dict_22766, (str_22805, str_22806))

# Assigning a type to the variable 'default_tools' (line 1021)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1021, 0), 'default_tools', dict_22766)
str_22807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 0), 'str', 'Default tools')

# Assigning a List to a Name (line 1041):

# Assigning a List to a Name (line 1041):

# Assigning a List to a Name (line 1041):

# Obtaining an instance of the builtin type 'list' (line 1041)
list_22808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 24), 'list')
# Adding type elements to the builtin type 'list' instance (line 1041)
# Adding element type (line 1041)

# Obtaining an instance of the builtin type 'list' (line 1041)
list_22809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 1041)
# Adding element type (line 1041)
str_22810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 26), 'str', 'navigation')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1041, 25), list_22809, str_22810)
# Adding element type (line 1041)

# Obtaining an instance of the builtin type 'list' (line 1041)
list_22811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 40), 'list')
# Adding type elements to the builtin type 'list' instance (line 1041)
# Adding element type (line 1041)
str_22812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 41), 'str', 'home')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1041, 40), list_22811, str_22812)
# Adding element type (line 1041)
str_22813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 49), 'str', 'back')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1041, 40), list_22811, str_22813)
# Adding element type (line 1041)
str_22814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 57), 'str', 'forward')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1041, 40), list_22811, str_22814)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1041, 25), list_22809, list_22811)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1041, 24), list_22808, list_22809)
# Adding element type (line 1041)

# Obtaining an instance of the builtin type 'list' (line 1042)
list_22815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 1042)
# Adding element type (line 1042)
str_22816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 26), 'str', 'zoompan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 25), list_22815, str_22816)
# Adding element type (line 1042)

# Obtaining an instance of the builtin type 'list' (line 1042)
list_22817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 37), 'list')
# Adding type elements to the builtin type 'list' instance (line 1042)
# Adding element type (line 1042)
str_22818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 38), 'str', 'pan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 37), list_22817, str_22818)
# Adding element type (line 1042)
str_22819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 45), 'str', 'zoom')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 37), list_22817, str_22819)
# Adding element type (line 1042)
str_22820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 53), 'str', 'subplots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 37), list_22817, str_22820)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1042, 25), list_22815, list_22817)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1041, 24), list_22808, list_22815)
# Adding element type (line 1041)

# Obtaining an instance of the builtin type 'list' (line 1043)
list_22821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 1043)
# Adding element type (line 1043)
str_22822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 26), 'str', 'io')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 25), list_22821, str_22822)
# Adding element type (line 1043)

# Obtaining an instance of the builtin type 'list' (line 1043)
list_22823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 1043)
# Adding element type (line 1043)
str_22824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 33), 'str', 'save')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 32), list_22823, str_22824)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 25), list_22821, list_22823)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1041, 24), list_22808, list_22821)

# Assigning a type to the variable 'default_toolbar_tools' (line 1041)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1041, 0), 'default_toolbar_tools', list_22808)
str_22825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 0), 'str', 'Default tools in the toolbar')

@norecursion
def add_tools_to_manager(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'default_tools' (line 1047)
    default_tools_22826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 44), 'default_tools')
    defaults = [default_tools_22826]
    # Create a new context for function 'add_tools_to_manager'
    module_type_store = module_type_store.open_function_context('add_tools_to_manager', 1047, 0, False)
    
    # Passed parameters checking function
    add_tools_to_manager.stypy_localization = localization
    add_tools_to_manager.stypy_type_of_self = None
    add_tools_to_manager.stypy_type_store = module_type_store
    add_tools_to_manager.stypy_function_name = 'add_tools_to_manager'
    add_tools_to_manager.stypy_param_names_list = ['toolmanager', 'tools']
    add_tools_to_manager.stypy_varargs_param_name = None
    add_tools_to_manager.stypy_kwargs_param_name = None
    add_tools_to_manager.stypy_call_defaults = defaults
    add_tools_to_manager.stypy_call_varargs = varargs
    add_tools_to_manager.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add_tools_to_manager', ['toolmanager', 'tools'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add_tools_to_manager', localization, ['toolmanager', 'tools'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add_tools_to_manager(...)' code ##################

    str_22827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, (-1)), 'str', '\n    Add multiple tools to `ToolManager`\n\n    Parameters\n    ----------\n    toolmanager: ToolManager\n        `backend_managers.ToolManager` object that will get the tools added\n    tools : {str: class_like}, optional\n        The tools to add in a {name: tool} dict, see `add_tool` for more\n        info.\n    ')
    
    
    # Call to iteritems(...): (line 1060)
    # Processing the call arguments (line 1060)
    # Getting the type of 'tools' (line 1060)
    tools_22830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 36), 'tools', False)
    # Processing the call keyword arguments (line 1060)
    kwargs_22831 = {}
    # Getting the type of 'six' (line 1060)
    six_22828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 22), 'six', False)
    # Obtaining the member 'iteritems' of a type (line 1060)
    iteritems_22829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1060, 22), six_22828, 'iteritems')
    # Calling iteritems(args, kwargs) (line 1060)
    iteritems_call_result_22832 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 22), iteritems_22829, *[tools_22830], **kwargs_22831)
    
    # Testing the type of a for loop iterable (line 1060)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1060, 4), iteritems_call_result_22832)
    # Getting the type of the for loop variable (line 1060)
    for_loop_var_22833 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1060, 4), iteritems_call_result_22832)
    # Assigning a type to the variable 'name' (line 1060)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1060, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1060, 4), for_loop_var_22833))
    # Assigning a type to the variable 'tool' (line 1060)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1060, 4), 'tool', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1060, 4), for_loop_var_22833))
    # SSA begins for a for statement (line 1060)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to add_tool(...): (line 1061)
    # Processing the call arguments (line 1061)
    # Getting the type of 'name' (line 1061)
    name_22836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 29), 'name', False)
    # Getting the type of 'tool' (line 1061)
    tool_22837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 35), 'tool', False)
    # Processing the call keyword arguments (line 1061)
    kwargs_22838 = {}
    # Getting the type of 'toolmanager' (line 1061)
    toolmanager_22834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 8), 'toolmanager', False)
    # Obtaining the member 'add_tool' of a type (line 1061)
    add_tool_22835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1061, 8), toolmanager_22834, 'add_tool')
    # Calling add_tool(args, kwargs) (line 1061)
    add_tool_call_result_22839 = invoke(stypy.reporting.localization.Localization(__file__, 1061, 8), add_tool_22835, *[name_22836, tool_22837], **kwargs_22838)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'add_tools_to_manager(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_tools_to_manager' in the type store
    # Getting the type of 'stypy_return_type' (line 1047)
    stypy_return_type_22840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22840)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_tools_to_manager'
    return stypy_return_type_22840

# Assigning a type to the variable 'add_tools_to_manager' (line 1047)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1047, 0), 'add_tools_to_manager', add_tools_to_manager)

@norecursion
def add_tools_to_container(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'default_toolbar_tools' (line 1064)
    default_toolbar_tools_22841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 44), 'default_toolbar_tools')
    defaults = [default_toolbar_tools_22841]
    # Create a new context for function 'add_tools_to_container'
    module_type_store = module_type_store.open_function_context('add_tools_to_container', 1064, 0, False)
    
    # Passed parameters checking function
    add_tools_to_container.stypy_localization = localization
    add_tools_to_container.stypy_type_of_self = None
    add_tools_to_container.stypy_type_store = module_type_store
    add_tools_to_container.stypy_function_name = 'add_tools_to_container'
    add_tools_to_container.stypy_param_names_list = ['container', 'tools']
    add_tools_to_container.stypy_varargs_param_name = None
    add_tools_to_container.stypy_kwargs_param_name = None
    add_tools_to_container.stypy_call_defaults = defaults
    add_tools_to_container.stypy_call_varargs = varargs
    add_tools_to_container.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add_tools_to_container', ['container', 'tools'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add_tools_to_container', localization, ['container', 'tools'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add_tools_to_container(...)' code ##################

    str_22842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, (-1)), 'str', '\n    Add multiple tools to the container.\n\n    Parameters\n    ----------\n    container: Container\n        `backend_bases.ToolContainerBase` object that will get the tools added\n    tools : list, optional\n        List in the form\n        [[group1, [tool1, tool2 ...]], [group2, [...]]]\n        Where the tools given by tool1, and tool2 will display in group1.\n        See `add_tool` for details.\n    ')
    
    # Getting the type of 'tools' (line 1079)
    tools_22843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 29), 'tools')
    # Testing the type of a for loop iterable (line 1079)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1079, 4), tools_22843)
    # Getting the type of the for loop variable (line 1079)
    for_loop_var_22844 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1079, 4), tools_22843)
    # Assigning a type to the variable 'group' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 4), 'group', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1079, 4), for_loop_var_22844))
    # Assigning a type to the variable 'grouptools' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 4), 'grouptools', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1079, 4), for_loop_var_22844))
    # SSA begins for a for statement (line 1079)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to enumerate(...): (line 1080)
    # Processing the call arguments (line 1080)
    # Getting the type of 'grouptools' (line 1080)
    grouptools_22846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 40), 'grouptools', False)
    # Processing the call keyword arguments (line 1080)
    kwargs_22847 = {}
    # Getting the type of 'enumerate' (line 1080)
    enumerate_22845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 30), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1080)
    enumerate_call_result_22848 = invoke(stypy.reporting.localization.Localization(__file__, 1080, 30), enumerate_22845, *[grouptools_22846], **kwargs_22847)
    
    # Testing the type of a for loop iterable (line 1080)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1080, 8), enumerate_call_result_22848)
    # Getting the type of the for loop variable (line 1080)
    for_loop_var_22849 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1080, 8), enumerate_call_result_22848)
    # Assigning a type to the variable 'position' (line 1080)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1080, 8), 'position', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1080, 8), for_loop_var_22849))
    # Assigning a type to the variable 'tool' (line 1080)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1080, 8), 'tool', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1080, 8), for_loop_var_22849))
    # SSA begins for a for statement (line 1080)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to add_tool(...): (line 1081)
    # Processing the call arguments (line 1081)
    # Getting the type of 'tool' (line 1081)
    tool_22852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 31), 'tool', False)
    # Getting the type of 'group' (line 1081)
    group_22853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 37), 'group', False)
    # Getting the type of 'position' (line 1081)
    position_22854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 44), 'position', False)
    # Processing the call keyword arguments (line 1081)
    kwargs_22855 = {}
    # Getting the type of 'container' (line 1081)
    container_22850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 12), 'container', False)
    # Obtaining the member 'add_tool' of a type (line 1081)
    add_tool_22851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1081, 12), container_22850, 'add_tool')
    # Calling add_tool(args, kwargs) (line 1081)
    add_tool_call_result_22856 = invoke(stypy.reporting.localization.Localization(__file__, 1081, 12), add_tool_22851, *[tool_22852, group_22853, position_22854], **kwargs_22855)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'add_tools_to_container(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_tools_to_container' in the type store
    # Getting the type of 'stypy_return_type' (line 1064)
    stypy_return_type_22857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22857)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_tools_to_container'
    return stypy_return_type_22857

# Assigning a type to the variable 'add_tools_to_container' (line 1064)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1064, 0), 'add_tools_to_container', add_tools_to_container)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
