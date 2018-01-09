
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import six
5: 
6: import os, sys
7: 
8: try:
9:     import gi
10: except ImportError:
11:     raise ImportError("Gtk3 backend requires pygobject to be installed.")
12: 
13: try:
14:     gi.require_version("Gtk", "3.0")
15: except AttributeError:
16:     raise ImportError(
17:         "pygobject version too old -- it must have require_version")
18: except ValueError:
19:     raise ImportError(
20:         "Gtk3 backend requires the GObject introspection bindings for Gtk 3 "
21:         "to be installed.")
22: 
23: try:
24:     from gi.repository import Gtk, Gdk, GObject, GLib
25: except ImportError:
26:     raise ImportError("Gtk3 backend requires pygobject to be installed.")
27: 
28: import matplotlib
29: from matplotlib._pylab_helpers import Gcf
30: from matplotlib.backend_bases import (
31:     _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase,
32:     NavigationToolbar2, RendererBase, TimerBase, cursors)
33: from matplotlib.backend_bases import ToolContainerBase, StatusbarBase
34: from matplotlib.backend_managers import ToolManager
35: from matplotlib.cbook import is_writable_file_like
36: from matplotlib.figure import Figure
37: from matplotlib.widgets import SubplotTool
38: 
39: from matplotlib import (
40:     backend_tools, cbook, colors as mcolors, lines, verbose, rcParams)
41: 
42: backend_version = "%s.%s.%s" % (
43:     Gtk.get_major_version(), Gtk.get_micro_version(), Gtk.get_minor_version())
44: 
45: # the true dots per inch on the screen; should be display dependent
46: # see http://groups.google.com/groups?q=screen+dpi+x11&hl=en&lr=&ie=UTF-8&oe=UTF-8&safe=off&selm=7077.26e81ad5%40swift.cs.tcd.ie&rnum=5 for some info about screen dpi
47: PIXELS_PER_INCH = 96
48: 
49: cursord = {
50:     cursors.MOVE          : Gdk.Cursor.new(Gdk.CursorType.FLEUR),
51:     cursors.HAND          : Gdk.Cursor.new(Gdk.CursorType.HAND2),
52:     cursors.POINTER       : Gdk.Cursor.new(Gdk.CursorType.LEFT_PTR),
53:     cursors.SELECT_REGION : Gdk.Cursor.new(Gdk.CursorType.TCROSS),
54:     cursors.WAIT          : Gdk.Cursor.new(Gdk.CursorType.WATCH),
55:     }
56: 
57: 
58: class TimerGTK3(TimerBase):
59:     '''
60:     Subclass of :class:`backend_bases.TimerBase` using GTK3 for timer events.
61: 
62:     Attributes
63:     ----------
64:     interval : int
65:         The time between timer events in milliseconds. Default is 1000 ms.
66:     single_shot : bool
67:         Boolean flag indicating whether this timer should operate as single
68:         shot (run once and then stop). Defaults to False.
69:     callbacks : list
70:         Stores list of (func, args) tuples that will be called upon timer
71:         events. This list can be manipulated directly, or the functions
72:         `add_callback` and `remove_callback` can be used.
73: 
74:     '''
75:     def _timer_start(self):
76:         # Need to stop it, otherwise we potentially leak a timer id that will
77:         # never be stopped.
78:         self._timer_stop()
79:         self._timer = GLib.timeout_add(self._interval, self._on_timer)
80: 
81:     def _timer_stop(self):
82:         if self._timer is not None:
83:             GLib.source_remove(self._timer)
84:             self._timer = None
85: 
86:     def _timer_set_interval(self):
87:         # Only stop and restart it if the timer has already been started
88:         if self._timer is not None:
89:             self._timer_stop()
90:             self._timer_start()
91: 
92:     def _on_timer(self):
93:         TimerBase._on_timer(self)
94: 
95:         # Gtk timeout_add() requires that the callback returns True if it
96:         # is to be called again.
97:         if len(self.callbacks) > 0 and not self._single:
98:             return True
99:         else:
100:             self._timer = None
101:             return False
102: 
103: 
104: class FigureCanvasGTK3(Gtk.DrawingArea, FigureCanvasBase):
105:     keyvald = {65507 : 'control',
106:                65505 : 'shift',
107:                65513 : 'alt',
108:                65508 : 'control',
109:                65506 : 'shift',
110:                65514 : 'alt',
111:                65361 : 'left',
112:                65362 : 'up',
113:                65363 : 'right',
114:                65364 : 'down',
115:                65307 : 'escape',
116:                65470 : 'f1',
117:                65471 : 'f2',
118:                65472 : 'f3',
119:                65473 : 'f4',
120:                65474 : 'f5',
121:                65475 : 'f6',
122:                65476 : 'f7',
123:                65477 : 'f8',
124:                65478 : 'f9',
125:                65479 : 'f10',
126:                65480 : 'f11',
127:                65481 : 'f12',
128:                65300 : 'scroll_lock',
129:                65299 : 'break',
130:                65288 : 'backspace',
131:                65293 : 'enter',
132:                65379 : 'insert',
133:                65535 : 'delete',
134:                65360 : 'home',
135:                65367 : 'end',
136:                65365 : 'pageup',
137:                65366 : 'pagedown',
138:                65438 : '0',
139:                65436 : '1',
140:                65433 : '2',
141:                65435 : '3',
142:                65430 : '4',
143:                65437 : '5',
144:                65432 : '6',
145:                65429 : '7',
146:                65431 : '8',
147:                65434 : '9',
148:                65451 : '+',
149:                65453 : '-',
150:                65450 : '*',
151:                65455 : '/',
152:                65439 : 'dec',
153:                65421 : 'enter',
154:                }
155: 
156:     # Setting this as a static constant prevents
157:     # this resulting expression from leaking
158:     event_mask = (Gdk.EventMask.BUTTON_PRESS_MASK   |
159:                   Gdk.EventMask.BUTTON_RELEASE_MASK |
160:                   Gdk.EventMask.EXPOSURE_MASK       |
161:                   Gdk.EventMask.KEY_PRESS_MASK      |
162:                   Gdk.EventMask.KEY_RELEASE_MASK    |
163:                   Gdk.EventMask.ENTER_NOTIFY_MASK   |
164:                   Gdk.EventMask.LEAVE_NOTIFY_MASK   |
165:                   Gdk.EventMask.POINTER_MOTION_MASK |
166:                   Gdk.EventMask.POINTER_MOTION_HINT_MASK|
167:                   Gdk.EventMask.SCROLL_MASK)
168: 
169:     def __init__(self, figure):
170:         FigureCanvasBase.__init__(self, figure)
171:         GObject.GObject.__init__(self)
172: 
173:         self._idle_draw_id  = 0
174:         self._lastCursor    = None
175: 
176:         self.connect('scroll_event',         self.scroll_event)
177:         self.connect('button_press_event',   self.button_press_event)
178:         self.connect('button_release_event', self.button_release_event)
179:         self.connect('configure_event',      self.configure_event)
180:         self.connect('draw',                 self.on_draw_event)
181:         self.connect('key_press_event',      self.key_press_event)
182:         self.connect('key_release_event',    self.key_release_event)
183:         self.connect('motion_notify_event',  self.motion_notify_event)
184:         self.connect('leave_notify_event',   self.leave_notify_event)
185:         self.connect('enter_notify_event',   self.enter_notify_event)
186:         self.connect('size_allocate',        self.size_allocate)
187: 
188:         self.set_events(self.__class__.event_mask)
189: 
190:         self.set_double_buffered(True)
191:         self.set_can_focus(True)
192:         self._renderer_init()
193:         default_context = GLib.main_context_get_thread_default() or GLib.main_context_default()
194: 
195:     def destroy(self):
196:         #Gtk.DrawingArea.destroy(self)
197:         self.close_event()
198:         if self._idle_draw_id != 0:
199:             GLib.source_remove(self._idle_draw_id)
200: 
201:     def scroll_event(self, widget, event):
202:         x = event.x
203:         # flipy so y=0 is bottom of canvas
204:         y = self.get_allocation().height - event.y
205:         if event.direction==Gdk.ScrollDirection.UP:
206:             step = 1
207:         else:
208:             step = -1
209:         FigureCanvasBase.scroll_event(self, x, y, step, guiEvent=event)
210:         return False  # finish event propagation?
211: 
212:     def button_press_event(self, widget, event):
213:         x = event.x
214:         # flipy so y=0 is bottom of canvas
215:         y = self.get_allocation().height - event.y
216:         FigureCanvasBase.button_press_event(self, x, y, event.button, guiEvent=event)
217:         return False  # finish event propagation?
218: 
219:     def button_release_event(self, widget, event):
220:         x = event.x
221:         # flipy so y=0 is bottom of canvas
222:         y = self.get_allocation().height - event.y
223:         FigureCanvasBase.button_release_event(self, x, y, event.button, guiEvent=event)
224:         return False  # finish event propagation?
225: 
226:     def key_press_event(self, widget, event):
227:         key = self._get_key(event)
228:         FigureCanvasBase.key_press_event(self, key, guiEvent=event)
229:         return True  # stop event propagation
230: 
231:     def key_release_event(self, widget, event):
232:         key = self._get_key(event)
233:         FigureCanvasBase.key_release_event(self, key, guiEvent=event)
234:         return True  # stop event propagation
235: 
236:     def motion_notify_event(self, widget, event):
237:         if event.is_hint:
238:             t, x, y, state = event.window.get_pointer()
239:         else:
240:             x, y, state = event.x, event.y, event.get_state()
241: 
242:         # flipy so y=0 is bottom of canvas
243:         y = self.get_allocation().height - y
244:         FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=event)
245:         return False  # finish event propagation?
246: 
247:     def leave_notify_event(self, widget, event):
248:         FigureCanvasBase.leave_notify_event(self, event)
249: 
250:     def enter_notify_event(self, widget, event):
251:         FigureCanvasBase.enter_notify_event(self, event)
252: 
253:     def size_allocate(self, widget, allocation):
254:         dpival = self.figure.dpi
255:         winch = allocation.width / dpival
256:         hinch = allocation.height / dpival
257:         self.figure.set_size_inches(winch, hinch, forward=False)
258:         FigureCanvasBase.resize_event(self)
259:         self.draw_idle()
260: 
261:     def _get_key(self, event):
262:         if event.keyval in self.keyvald:
263:             key = self.keyvald[event.keyval]
264:         elif event.keyval < 256:
265:             key = chr(event.keyval)
266:         else:
267:             key = None
268: 
269:         modifiers = [
270:                      (Gdk.ModifierType.MOD4_MASK, 'super'),
271:                      (Gdk.ModifierType.MOD1_MASK, 'alt'),
272:                      (Gdk.ModifierType.CONTROL_MASK, 'ctrl'),
273:                     ]
274:         for key_mask, prefix in modifiers:
275:             if event.state & key_mask:
276:                 key = '{0}+{1}'.format(prefix, key)
277: 
278:         return key
279: 
280:     def configure_event(self, widget, event):
281:         if widget.get_property("window") is None:
282:             return
283:         w, h = event.width, event.height
284:         if w < 3 or h < 3:
285:             return # empty fig
286:         # resize the figure (in inches)
287:         dpi = self.figure.dpi
288:         self.figure.set_size_inches(w/dpi, h/dpi, forward=False)
289:         return False  # finish event propagation?
290: 
291:     def on_draw_event(self, widget, ctx):
292:         # to be overwritten by GTK3Agg or GTK3Cairo
293:         pass
294: 
295:     def draw(self):
296:         if self.get_visible() and self.get_mapped():
297:             self.queue_draw()
298:             # do a synchronous draw (its less efficient than an async draw,
299:             # but is required if/when animation is used)
300:             self.get_property("window").process_updates (False)
301: 
302:     def draw_idle(self):
303:         if self._idle_draw_id != 0:
304:             return
305:         def idle_draw(*args):
306:             try:
307:                 self.draw()
308:             finally:
309:                 self._idle_draw_id = 0
310:             return False
311:         self._idle_draw_id = GLib.idle_add(idle_draw)
312: 
313:     def new_timer(self, *args, **kwargs):
314:         '''
315:         Creates a new backend-specific subclass of :class:`backend_bases.Timer`.
316:         This is useful for getting periodic events through the backend's native
317:         event loop. Implemented only for backends with GUIs.
318: 
319:         Other Parameters
320:         ----------------
321:         interval : scalar
322:             Timer interval in milliseconds
323:         callbacks : list
324:             Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``
325:             will be executed by the timer every *interval*.
326:         '''
327:         return TimerGTK3(*args, **kwargs)
328: 
329:     def flush_events(self):
330:         Gdk.threads_enter()
331:         while Gtk.events_pending():
332:             Gtk.main_iteration()
333:         Gdk.flush()
334:         Gdk.threads_leave()
335: 
336: 
337: class FigureManagerGTK3(FigureManagerBase):
338:     '''
339:     Attributes
340:     ----------
341:     canvas : `FigureCanvas`
342:         The FigureCanvas instance
343:     num : int or str
344:         The Figure number
345:     toolbar : Gtk.Toolbar
346:         The Gtk.Toolbar  (gtk only)
347:     vbox : Gtk.VBox
348:         The Gtk.VBox containing the canvas and toolbar (gtk only)
349:     window : Gtk.Window
350:         The Gtk.Window   (gtk only)
351: 
352:     '''
353:     def __init__(self, canvas, num):
354:         FigureManagerBase.__init__(self, canvas, num)
355: 
356:         self.window = Gtk.Window()
357:         self.window.set_wmclass("matplotlib", "Matplotlib")
358:         self.set_window_title("Figure %d" % num)
359:         try:
360:             self.window.set_icon_from_file(window_icon)
361:         except (SystemExit, KeyboardInterrupt):
362:             # re-raise exit type Exceptions
363:             raise
364:         except:
365:             # some versions of gtk throw a glib.GError but not
366:             # all, so I am not sure how to catch it.  I am unhappy
367:             # doing a blanket catch here, but am not sure what a
368:             # better way is - JDH
369:             verbose.report('Could not load matplotlib icon: %s' % sys.exc_info()[1])
370: 
371:         self.vbox = Gtk.Box()
372:         self.vbox.set_property("orientation", Gtk.Orientation.VERTICAL)
373:         self.window.add(self.vbox)
374:         self.vbox.show()
375: 
376:         self.canvas.show()
377: 
378:         self.vbox.pack_start(self.canvas, True, True, 0)
379:         # calculate size for window
380:         w = int (self.canvas.figure.bbox.width)
381:         h = int (self.canvas.figure.bbox.height)
382: 
383:         self.toolmanager = self._get_toolmanager()
384:         self.toolbar = self._get_toolbar()
385:         self.statusbar = None
386: 
387:         def add_widget(child, expand, fill, padding):
388:             child.show()
389:             self.vbox.pack_end(child, False, False, 0)
390:             size_request = child.size_request()
391:             return size_request.height
392: 
393:         if self.toolmanager:
394:             backend_tools.add_tools_to_manager(self.toolmanager)
395:             if self.toolbar:
396:                 backend_tools.add_tools_to_container(self.toolbar)
397:                 self.statusbar = StatusbarGTK3(self.toolmanager)
398:                 h += add_widget(self.statusbar, False, False, 0)
399:                 h += add_widget(Gtk.HSeparator(), False, False, 0)
400: 
401:         if self.toolbar is not None:
402:             self.toolbar.show()
403:             h += add_widget(self.toolbar, False, False, 0)
404: 
405:         self.window.set_default_size (w, h)
406: 
407:         def destroy(*args):
408:             Gcf.destroy(num)
409:         self.window.connect("destroy", destroy)
410:         self.window.connect("delete_event", destroy)
411:         if matplotlib.is_interactive():
412:             self.window.show()
413:             self.canvas.draw_idle()
414: 
415:         def notify_axes_change(fig):
416:             'this will be called whenever the current axes is changed'
417:             if self.toolmanager is not None:
418:                 pass
419:             elif self.toolbar is not None:
420:                 self.toolbar.update()
421:         self.canvas.figure.add_axobserver(notify_axes_change)
422: 
423:         self.canvas.grab_focus()
424: 
425:     def destroy(self, *args):
426:         self.vbox.destroy()
427:         self.window.destroy()
428:         self.canvas.destroy()
429:         if self.toolbar:
430:             self.toolbar.destroy()
431: 
432:         if (Gcf.get_num_fig_managers() == 0 and
433:                 not matplotlib.is_interactive() and
434:                 Gtk.main_level() >= 1):
435:             Gtk.main_quit()
436: 
437:     def show(self):
438:         # show the figure window
439:         self.window.show()
440:         self.window.present()
441: 
442:     def full_screen_toggle (self):
443:         self._full_screen_flag = not self._full_screen_flag
444:         if self._full_screen_flag:
445:             self.window.fullscreen()
446:         else:
447:             self.window.unfullscreen()
448:     _full_screen_flag = False
449: 
450:     def _get_toolbar(self):
451:         # must be inited after the window, drawingArea and figure
452:         # attrs are set
453:         if rcParams['toolbar'] == 'toolbar2':
454:             toolbar = NavigationToolbar2GTK3(self.canvas, self.window)
455:         elif rcParams['toolbar'] == 'toolmanager':
456:             toolbar = ToolbarGTK3(self.toolmanager)
457:         else:
458:             toolbar = None
459:         return toolbar
460: 
461:     def _get_toolmanager(self):
462:         # must be initialised after toolbar has been setted
463:         if rcParams['toolbar'] == 'toolmanager':
464:             toolmanager = ToolManager(self.canvas.figure)
465:         else:
466:             toolmanager = None
467:         return toolmanager
468: 
469:     def get_window_title(self):
470:         return self.window.get_title()
471: 
472:     def set_window_title(self, title):
473:         self.window.set_title(title)
474: 
475:     def resize(self, width, height):
476:         'set the canvas size in pixels'
477:         #_, _, cw, ch = self.canvas.allocation
478:         #_, _, ww, wh = self.window.allocation
479:         #self.window.resize (width-cw+ww, height-ch+wh)
480:         self.window.resize(width, height)
481: 
482: 
483: class NavigationToolbar2GTK3(NavigationToolbar2, Gtk.Toolbar):
484:     def __init__(self, canvas, window):
485:         self.win = window
486:         GObject.GObject.__init__(self)
487:         NavigationToolbar2.__init__(self, canvas)
488:         self.ctx = None
489: 
490:     def set_message(self, s):
491:         self.message.set_label(s)
492: 
493:     def set_cursor(self, cursor):
494:         self.canvas.get_property("window").set_cursor(cursord[cursor])
495:         Gtk.main_iteration()
496: 
497:     def release(self, event):
498:         try: del self._pixmapBack
499:         except AttributeError: pass
500: 
501:     def draw_rubberband(self, event, x0, y0, x1, y1):
502:         'adapted from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/189744'
503:         self.ctx = self.canvas.get_property("window").cairo_create()
504: 
505:         # todo: instead of redrawing the entire figure, copy the part of
506:         # the figure that was covered by the previous rubberband rectangle
507:         self.canvas.draw()
508: 
509:         height = self.canvas.figure.bbox.height
510:         y1 = height - y1
511:         y0 = height - y0
512:         w = abs(x1 - x0)
513:         h = abs(y1 - y0)
514:         rect = [int(val) for val in (min(x0,x1), min(y0, y1), w, h)]
515: 
516:         self.ctx.new_path()
517:         self.ctx.set_line_width(0.5)
518:         self.ctx.rectangle(rect[0], rect[1], rect[2], rect[3])
519:         self.ctx.set_source_rgb(0, 0, 0)
520:         self.ctx.stroke()
521: 
522:     def _init_toolbar(self):
523:         self.set_style(Gtk.ToolbarStyle.ICONS)
524:         basedir = os.path.join(rcParams['datapath'],'images')
525: 
526:         for text, tooltip_text, image_file, callback in self.toolitems:
527:             if text is None:
528:                 self.insert( Gtk.SeparatorToolItem(), -1 )
529:                 continue
530:             fname = os.path.join(basedir, image_file + '.png')
531:             image = Gtk.Image()
532:             image.set_from_file(fname)
533:             tbutton = Gtk.ToolButton()
534:             tbutton.set_label(text)
535:             tbutton.set_icon_widget(image)
536:             self.insert(tbutton, -1)
537:             tbutton.connect('clicked', getattr(self, callback))
538:             tbutton.set_tooltip_text(tooltip_text)
539: 
540:         toolitem = Gtk.SeparatorToolItem()
541:         self.insert(toolitem, -1)
542:         toolitem.set_draw(False)
543:         toolitem.set_expand(True)
544: 
545:         toolitem = Gtk.ToolItem()
546:         self.insert(toolitem, -1)
547:         self.message = Gtk.Label()
548:         toolitem.add(self.message)
549: 
550:         self.show_all()
551: 
552:     def get_filechooser(self):
553:         fc = FileChooserDialog(
554:             title='Save the figure',
555:             parent=self.win,
556:             path=os.path.expanduser(rcParams['savefig.directory']),
557:             filetypes=self.canvas.get_supported_filetypes(),
558:             default_filetype=self.canvas.get_default_filetype())
559:         fc.set_current_name(self.canvas.get_default_filename())
560:         return fc
561: 
562:     def save_figure(self, *args):
563:         chooser = self.get_filechooser()
564:         fname, format = chooser.get_filename_from_user()
565:         chooser.destroy()
566:         if fname:
567:             startpath = os.path.expanduser(rcParams['savefig.directory'])
568:             # Save dir for next time, unless empty str (i.e., use cwd).
569:             if startpath != "":
570:                 rcParams['savefig.directory'] = (
571:                     os.path.dirname(six.text_type(fname)))
572:             try:
573:                 self.canvas.figure.savefig(fname, format=format)
574:             except Exception as e:
575:                 error_msg_gtk(str(e), parent=self)
576: 
577:     def configure_subplots(self, button):
578:         toolfig = Figure(figsize=(6,3))
579:         canvas = self._get_canvas(toolfig)
580:         toolfig.subplots_adjust(top=0.9)
581:         tool =  SubplotTool(self.canvas.figure, toolfig)
582: 
583:         w = int(toolfig.bbox.width)
584:         h = int(toolfig.bbox.height)
585: 
586:         window = Gtk.Window()
587:         try:
588:             window.set_icon_from_file(window_icon)
589:         except (SystemExit, KeyboardInterrupt):
590:             # re-raise exit type Exceptions
591:             raise
592:         except:
593:             # we presumably already logged a message on the
594:             # failure of the main plot, don't keep reporting
595:             pass
596:         window.set_title("Subplot Configuration Tool")
597:         window.set_default_size(w, h)
598:         vbox = Gtk.Box()
599:         vbox.set_property("orientation", Gtk.Orientation.VERTICAL)
600:         window.add(vbox)
601:         vbox.show()
602: 
603:         canvas.show()
604:         vbox.pack_start(canvas, True, True, 0)
605:         window.show()
606: 
607:     def _get_canvas(self, fig):
608:         return self.canvas.__class__(fig)
609: 
610: 
611: class FileChooserDialog(Gtk.FileChooserDialog):
612:     '''GTK+ file selector which remembers the last file/directory
613:     selected and presents the user with a menu of supported image formats
614:     '''
615:     def __init__ (self,
616:                   title   = 'Save file',
617:                   parent  = None,
618:                   action  = Gtk.FileChooserAction.SAVE,
619:                   buttons = (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
620:                              Gtk.STOCK_SAVE,   Gtk.ResponseType.OK),
621:                   path    = None,
622:                   filetypes = [],
623:                   default_filetype = None
624:                   ):
625:         super (FileChooserDialog, self).__init__ (title, parent, action,
626:                                                   buttons)
627:         self.set_default_response (Gtk.ResponseType.OK)
628: 
629:         if not path: path = os.getcwd() + os.sep
630: 
631:         # create an extra widget to list supported image formats
632:         self.set_current_folder (path)
633:         self.set_current_name ('image.' + default_filetype)
634: 
635:         hbox = Gtk.Box(spacing=10)
636:         hbox.pack_start(Gtk.Label(label="File Format:"), False, False, 0)
637: 
638:         liststore = Gtk.ListStore(GObject.TYPE_STRING)
639:         cbox = Gtk.ComboBox() #liststore)
640:         cbox.set_model(liststore)
641:         cell = Gtk.CellRendererText()
642:         cbox.pack_start(cell, True)
643:         cbox.add_attribute(cell, 'text', 0)
644:         hbox.pack_start(cbox, False, False, 0)
645: 
646:         self.filetypes = filetypes
647:         self.sorted_filetypes = sorted(six.iteritems(filetypes))
648:         default = 0
649:         for i, (ext, name) in enumerate(self.sorted_filetypes):
650:             liststore.append(["%s (*.%s)" % (name, ext)])
651:             if ext == default_filetype:
652:                 default = i
653:         cbox.set_active(default)
654:         self.ext = default_filetype
655: 
656:         def cb_cbox_changed (cbox, data=None):
657:             '''File extension changed'''
658:             head, filename = os.path.split(self.get_filename())
659:             root, ext = os.path.splitext(filename)
660:             ext = ext[1:]
661:             new_ext = self.sorted_filetypes[cbox.get_active()][0]
662:             self.ext = new_ext
663: 
664:             if ext in self.filetypes:
665:                 filename = root + '.' + new_ext
666:             elif ext == '':
667:                 filename = filename.rstrip('.') + '.' + new_ext
668: 
669:             self.set_current_name (filename)
670:         cbox.connect ("changed", cb_cbox_changed)
671: 
672:         hbox.show_all()
673:         self.set_extra_widget(hbox)
674: 
675:     def get_filename_from_user (self):
676:         while True:
677:             filename = None
678:             if self.run() != int(Gtk.ResponseType.OK):
679:                 break
680:             filename = self.get_filename()
681:             break
682: 
683:         return filename, self.ext
684: 
685: 
686: class RubberbandGTK3(backend_tools.RubberbandBase):
687:     def __init__(self, *args, **kwargs):
688:         backend_tools.RubberbandBase.__init__(self, *args, **kwargs)
689:         self.ctx = None
690: 
691:     def draw_rubberband(self, x0, y0, x1, y1):
692:         # 'adapted from http://aspn.activestate.com/ASPN/Cookbook/Python/
693:         # Recipe/189744'
694:         self.ctx = self.figure.canvas.get_property("window").cairo_create()
695: 
696:         # todo: instead of redrawing the entire figure, copy the part of
697:         # the figure that was covered by the previous rubberband rectangle
698:         self.figure.canvas.draw()
699: 
700:         height = self.figure.bbox.height
701:         y1 = height - y1
702:         y0 = height - y0
703:         w = abs(x1 - x0)
704:         h = abs(y1 - y0)
705:         rect = [int(val) for val in (min(x0, x1), min(y0, y1), w, h)]
706: 
707:         self.ctx.new_path()
708:         self.ctx.set_line_width(0.5)
709:         self.ctx.rectangle(rect[0], rect[1], rect[2], rect[3])
710:         self.ctx.set_source_rgb(0, 0, 0)
711:         self.ctx.stroke()
712: 
713: 
714: class ToolbarGTK3(ToolContainerBase, Gtk.Box):
715:     def __init__(self, toolmanager):
716:         ToolContainerBase.__init__(self, toolmanager)
717:         Gtk.Box.__init__(self)
718:         self.set_property("orientation", Gtk.Orientation.VERTICAL)
719: 
720:         self._toolarea = Gtk.Box()
721:         self._toolarea.set_property('orientation', Gtk.Orientation.HORIZONTAL)
722:         self.pack_start(self._toolarea, False, False, 0)
723:         self._toolarea.show_all()
724:         self._groups = {}
725:         self._toolitems = {}
726: 
727:     def add_toolitem(self, name, group, position, image_file, description,
728:                      toggle):
729:         if toggle:
730:             tbutton = Gtk.ToggleToolButton()
731:         else:
732:             tbutton = Gtk.ToolButton()
733:         tbutton.set_label(name)
734: 
735:         if image_file is not None:
736:             image = Gtk.Image()
737:             image.set_from_file(image_file)
738:             tbutton.set_icon_widget(image)
739: 
740:         if position is None:
741:             position = -1
742: 
743:         self._add_button(tbutton, group, position)
744:         signal = tbutton.connect('clicked', self._call_tool, name)
745:         tbutton.set_tooltip_text(description)
746:         tbutton.show_all()
747:         self._toolitems.setdefault(name, [])
748:         self._toolitems[name].append((tbutton, signal))
749: 
750:     def _add_button(self, button, group, position):
751:         if group not in self._groups:
752:             if self._groups:
753:                 self._add_separator()
754:             toolbar = Gtk.Toolbar()
755:             toolbar.set_style(Gtk.ToolbarStyle.ICONS)
756:             self._toolarea.pack_start(toolbar, False, False, 0)
757:             toolbar.show_all()
758:             self._groups[group] = toolbar
759:         self._groups[group].insert(button, position)
760: 
761:     def _call_tool(self, btn, name):
762:         self.trigger_tool(name)
763: 
764:     def toggle_toolitem(self, name, toggled):
765:         if name not in self._toolitems:
766:             return
767:         for toolitem, signal in self._toolitems[name]:
768:             toolitem.handler_block(signal)
769:             toolitem.set_active(toggled)
770:             toolitem.handler_unblock(signal)
771: 
772:     def remove_toolitem(self, name):
773:         if name not in self._toolitems:
774:             self.toolmanager.message_event('%s Not in toolbar' % name, self)
775:             return
776: 
777:         for group in self._groups:
778:             for toolitem, _signal in self._toolitems[name]:
779:                 if toolitem in self._groups[group]:
780:                     self._groups[group].remove(toolitem)
781:         del self._toolitems[name]
782: 
783:     def _add_separator(self):
784:         sep = Gtk.Separator()
785:         sep.set_property("orientation", Gtk.Orientation.VERTICAL)
786:         self._toolarea.pack_start(sep, False, True, 0)
787:         sep.show_all()
788: 
789: 
790: class StatusbarGTK3(StatusbarBase, Gtk.Statusbar):
791:     def __init__(self, *args, **kwargs):
792:         StatusbarBase.__init__(self, *args, **kwargs)
793:         Gtk.Statusbar.__init__(self)
794:         self._context = self.get_context_id('message')
795: 
796:     def set_message(self, s):
797:         self.pop(self._context)
798:         self.push(self._context, s)
799: 
800: 
801: class SaveFigureGTK3(backend_tools.SaveFigureBase):
802: 
803:     def get_filechooser(self):
804:         fc = FileChooserDialog(
805:             title='Save the figure',
806:             parent=self.figure.canvas.manager.window,
807:             path=os.path.expanduser(rcParams['savefig.directory']),
808:             filetypes=self.figure.canvas.get_supported_filetypes(),
809:             default_filetype=self.figure.canvas.get_default_filetype())
810:         fc.set_current_name(self.figure.canvas.get_default_filename())
811:         return fc
812: 
813:     def trigger(self, *args, **kwargs):
814:         chooser = self.get_filechooser()
815:         fname, format_ = chooser.get_filename_from_user()
816:         chooser.destroy()
817:         if fname:
818:             startpath = os.path.expanduser(rcParams['savefig.directory'])
819:             if startpath == '':
820:                 # explicitly missing key or empty str signals to use cwd
821:                 rcParams['savefig.directory'] = startpath
822:             else:
823:                 # save dir for next time
824:                 rcParams['savefig.directory'] = os.path.dirname(
825:                     six.text_type(fname))
826:             try:
827:                 self.figure.canvas.print_figure(fname, format=format_)
828:             except Exception as e:
829:                 error_msg_gtk(str(e), parent=self)
830: 
831: 
832: class SetCursorGTK3(backend_tools.SetCursorBase):
833:     def set_cursor(self, cursor):
834:         self.figure.canvas.get_property("window").set_cursor(cursord[cursor])
835: 
836: 
837: class ConfigureSubplotsGTK3(backend_tools.ConfigureSubplotsBase, Gtk.Window):
838:     def __init__(self, *args, **kwargs):
839:         backend_tools.ConfigureSubplotsBase.__init__(self, *args, **kwargs)
840:         self.window = None
841: 
842:     def init_window(self):
843:         if self.window:
844:             return
845:         self.window = Gtk.Window(title="Subplot Configuration Tool")
846: 
847:         try:
848:             self.window.window.set_icon_from_file(window_icon)
849:         except (SystemExit, KeyboardInterrupt):
850:             # re-raise exit type Exceptions
851:             raise
852:         except:
853:             # we presumably already logged a message on the
854:             # failure of the main plot, don't keep reporting
855:             pass
856: 
857:         self.vbox = Gtk.Box()
858:         self.vbox.set_property("orientation", Gtk.Orientation.VERTICAL)
859:         self.window.add(self.vbox)
860:         self.vbox.show()
861:         self.window.connect('destroy', self.destroy)
862: 
863:         toolfig = Figure(figsize=(6, 3))
864:         canvas = self.figure.canvas.__class__(toolfig)
865: 
866:         toolfig.subplots_adjust(top=0.9)
867:         SubplotTool(self.figure, toolfig)
868: 
869:         w = int(toolfig.bbox.width)
870:         h = int(toolfig.bbox.height)
871: 
872:         self.window.set_default_size(w, h)
873: 
874:         canvas.show()
875:         self.vbox.pack_start(canvas, True, True, 0)
876:         self.window.show()
877: 
878:     def destroy(self, *args):
879:         self.window.destroy()
880:         self.window = None
881: 
882:     def _get_canvas(self, fig):
883:         return self.canvas.__class__(fig)
884: 
885:     def trigger(self, sender, event, data=None):
886:         self.init_window()
887:         self.window.present()
888: 
889: 
890: # Define the file to use as the GTk icon
891: if sys.platform == 'win32':
892:     icon_filename = 'matplotlib.png'
893: else:
894:     icon_filename = 'matplotlib.svg'
895: window_icon = os.path.join(
896:     matplotlib.rcParams['datapath'], 'images', icon_filename)
897: 
898: 
899: def error_msg_gtk(msg, parent=None):
900:     if parent is not None: # find the toplevel Gtk.Window
901:         parent = parent.get_toplevel()
902:         if not parent.is_toplevel():
903:             parent = None
904: 
905:     if not isinstance(msg, six.string_types):
906:         msg = ','.join(map(str, msg))
907: 
908:     dialog = Gtk.MessageDialog(
909:         parent         = parent,
910:         type           = Gtk.MessageType.ERROR,
911:         buttons        = Gtk.ButtonsType.OK,
912:         message_format = msg)
913:     dialog.run()
914:     dialog.destroy()
915: 
916: 
917: backend_tools.ToolSaveFigure = SaveFigureGTK3
918: backend_tools.ToolConfigureSubplots = ConfigureSubplotsGTK3
919: backend_tools.ToolSetCursor = SetCursorGTK3
920: backend_tools.ToolRubberband = RubberbandGTK3
921: 
922: Toolbar = ToolbarGTK3
923: 
924: 
925: @_Backend.export
926: class _BackendGTK3(_Backend):
927:     FigureCanvas = FigureCanvasGTK3
928:     FigureManager = FigureManagerGTK3
929: 
930:     @staticmethod
931:     def trigger_manager_draw(manager):
932:         manager.canvas.draw_idle()
933: 
934:     @staticmethod
935:     def mainloop():
936:         if Gtk.main_level() == 0:
937:             Gtk.main()
938: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226587 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_226587) is not StypyTypeError):

    if (import_226587 != 'pyd_module'):
        __import__(import_226587)
        sys_modules_226588 = sys.modules[import_226587]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_226588.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_226587)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# Multiple import statement. import os (1/2) (line 6)
import os

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os', os, module_type_store)
# Multiple import statement. import sys (2/2) (line 6)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sys', sys, module_type_store)



# SSA begins for try-except statement (line 8)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))

# 'import gi' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226589 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'gi')

if (type(import_226589) is not StypyTypeError):

    if (import_226589 != 'pyd_module'):
        __import__(import_226589)
        sys_modules_226590 = sys.modules[import_226589]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'gi', sys_modules_226590.module_type_store, module_type_store)
    else:
        import gi

        import_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'gi', gi, module_type_store)

else:
    # Assigning a type to the variable 'gi' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'gi', import_226589)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# SSA branch for the except part of a try statement (line 8)
# SSA branch for the except 'ImportError' branch of a try statement (line 8)
module_type_store.open_ssa_branch('except')

# Call to ImportError(...): (line 11)
# Processing the call arguments (line 11)
unicode_226592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'unicode', u'Gtk3 backend requires pygobject to be installed.')
# Processing the call keyword arguments (line 11)
kwargs_226593 = {}
# Getting the type of 'ImportError' (line 11)
ImportError_226591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 11)
ImportError_call_result_226594 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), ImportError_226591, *[unicode_226592], **kwargs_226593)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 11, 4), ImportError_call_result_226594, 'raise parameter', BaseException)
# SSA join for try-except statement (line 8)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 13)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to require_version(...): (line 14)
# Processing the call arguments (line 14)
unicode_226597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'unicode', u'Gtk')
unicode_226598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 30), 'unicode', u'3.0')
# Processing the call keyword arguments (line 14)
kwargs_226599 = {}
# Getting the type of 'gi' (line 14)
gi_226595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'gi', False)
# Obtaining the member 'require_version' of a type (line 14)
require_version_226596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), gi_226595, 'require_version')
# Calling require_version(args, kwargs) (line 14)
require_version_call_result_226600 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), require_version_226596, *[unicode_226597, unicode_226598], **kwargs_226599)

# SSA branch for the except part of a try statement (line 13)
# SSA branch for the except 'AttributeError' branch of a try statement (line 13)
module_type_store.open_ssa_branch('except')

# Call to ImportError(...): (line 16)
# Processing the call arguments (line 16)
unicode_226602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'unicode', u'pygobject version too old -- it must have require_version')
# Processing the call keyword arguments (line 16)
kwargs_226603 = {}
# Getting the type of 'ImportError' (line 16)
ImportError_226601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 16)
ImportError_call_result_226604 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), ImportError_226601, *[unicode_226602], **kwargs_226603)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 16, 4), ImportError_call_result_226604, 'raise parameter', BaseException)
# SSA branch for the except 'ValueError' branch of a try statement (line 13)
module_type_store.open_ssa_branch('except')

# Call to ImportError(...): (line 19)
# Processing the call arguments (line 19)
unicode_226606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'unicode', u'Gtk3 backend requires the GObject introspection bindings for Gtk 3 to be installed.')
# Processing the call keyword arguments (line 19)
kwargs_226607 = {}
# Getting the type of 'ImportError' (line 19)
ImportError_226605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 19)
ImportError_call_result_226608 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), ImportError_226605, *[unicode_226606], **kwargs_226607)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 19, 4), ImportError_call_result_226608, 'raise parameter', BaseException)
# SSA join for try-except statement (line 13)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 23)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 4))

# 'from gi.repository import Gtk, Gdk, GObject, GLib' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226609 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 4), 'gi.repository')

if (type(import_226609) is not StypyTypeError):

    if (import_226609 != 'pyd_module'):
        __import__(import_226609)
        sys_modules_226610 = sys.modules[import_226609]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 4), 'gi.repository', sys_modules_226610.module_type_store, module_type_store, ['Gtk', 'Gdk', 'GObject', 'GLib'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 4), __file__, sys_modules_226610, sys_modules_226610.module_type_store, module_type_store)
    else:
        from gi.repository import Gtk, Gdk, GObject, GLib

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 4), 'gi.repository', None, module_type_store, ['Gtk', 'Gdk', 'GObject', 'GLib'], [Gtk, Gdk, GObject, GLib])

else:
    # Assigning a type to the variable 'gi.repository' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'gi.repository', import_226609)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# SSA branch for the except part of a try statement (line 23)
# SSA branch for the except 'ImportError' branch of a try statement (line 23)
module_type_store.open_ssa_branch('except')

# Call to ImportError(...): (line 26)
# Processing the call arguments (line 26)
unicode_226612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'unicode', u'Gtk3 backend requires pygobject to be installed.')
# Processing the call keyword arguments (line 26)
kwargs_226613 = {}
# Getting the type of 'ImportError' (line 26)
ImportError_226611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 26)
ImportError_call_result_226614 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), ImportError_226611, *[unicode_226612], **kwargs_226613)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 26, 4), ImportError_call_result_226614, 'raise parameter', BaseException)
# SSA join for try-except statement (line 23)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'import matplotlib' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226615 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib')

if (type(import_226615) is not StypyTypeError):

    if (import_226615 != 'pyd_module'):
        __import__(import_226615)
        sys_modules_226616 = sys.modules[import_226615]
        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib', sys_modules_226616.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib', import_226615)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from matplotlib._pylab_helpers import Gcf' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226617 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib._pylab_helpers')

if (type(import_226617) is not StypyTypeError):

    if (import_226617 != 'pyd_module'):
        __import__(import_226617)
        sys_modules_226618 = sys.modules[import_226617]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib._pylab_helpers', sys_modules_226618.module_type_store, module_type_store, ['Gcf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_226618, sys_modules_226618.module_type_store, module_type_store)
    else:
        from matplotlib._pylab_helpers import Gcf

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib._pylab_helpers', None, module_type_store, ['Gcf'], [Gcf])

else:
    # Assigning a type to the variable 'matplotlib._pylab_helpers' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib._pylab_helpers', import_226617)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase, NavigationToolbar2, RendererBase, TimerBase, cursors' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226619 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.backend_bases')

if (type(import_226619) is not StypyTypeError):

    if (import_226619 != 'pyd_module'):
        __import__(import_226619)
        sys_modules_226620 = sys.modules[import_226619]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.backend_bases', sys_modules_226620.module_type_store, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'GraphicsContextBase', 'NavigationToolbar2', 'RendererBase', 'TimerBase', 'cursors'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_226620, sys_modules_226620.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase, NavigationToolbar2, RendererBase, TimerBase, cursors

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.backend_bases', None, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'GraphicsContextBase', 'NavigationToolbar2', 'RendererBase', 'TimerBase', 'cursors'], [_Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase, NavigationToolbar2, RendererBase, TimerBase, cursors])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.backend_bases', import_226619)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'from matplotlib.backend_bases import ToolContainerBase, StatusbarBase' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226621 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backend_bases')

if (type(import_226621) is not StypyTypeError):

    if (import_226621 != 'pyd_module'):
        __import__(import_226621)
        sys_modules_226622 = sys.modules[import_226621]
        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backend_bases', sys_modules_226622.module_type_store, module_type_store, ['ToolContainerBase', 'StatusbarBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 33, 0), __file__, sys_modules_226622, sys_modules_226622.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import ToolContainerBase, StatusbarBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backend_bases', None, module_type_store, ['ToolContainerBase', 'StatusbarBase'], [ToolContainerBase, StatusbarBase])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backend_bases', import_226621)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from matplotlib.backend_managers import ToolManager' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226623 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.backend_managers')

if (type(import_226623) is not StypyTypeError):

    if (import_226623 != 'pyd_module'):
        __import__(import_226623)
        sys_modules_226624 = sys.modules[import_226623]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.backend_managers', sys_modules_226624.module_type_store, module_type_store, ['ToolManager'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_226624, sys_modules_226624.module_type_store, module_type_store)
    else:
        from matplotlib.backend_managers import ToolManager

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.backend_managers', None, module_type_store, ['ToolManager'], [ToolManager])

else:
    # Assigning a type to the variable 'matplotlib.backend_managers' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'matplotlib.backend_managers', import_226623)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from matplotlib.cbook import is_writable_file_like' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226625 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.cbook')

if (type(import_226625) is not StypyTypeError):

    if (import_226625 != 'pyd_module'):
        __import__(import_226625)
        sys_modules_226626 = sys.modules[import_226625]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.cbook', sys_modules_226626.module_type_store, module_type_store, ['is_writable_file_like'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_226626, sys_modules_226626.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import is_writable_file_like

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.cbook', None, module_type_store, ['is_writable_file_like'], [is_writable_file_like])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'matplotlib.cbook', import_226625)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from matplotlib.figure import Figure' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226627 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.figure')

if (type(import_226627) is not StypyTypeError):

    if (import_226627 != 'pyd_module'):
        __import__(import_226627)
        sys_modules_226628 = sys.modules[import_226627]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.figure', sys_modules_226628.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_226628, sys_modules_226628.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'matplotlib.figure', import_226627)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from matplotlib.widgets import SubplotTool' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226629 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.widgets')

if (type(import_226629) is not StypyTypeError):

    if (import_226629 != 'pyd_module'):
        __import__(import_226629)
        sys_modules_226630 = sys.modules[import_226629]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.widgets', sys_modules_226630.module_type_store, module_type_store, ['SubplotTool'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_226630, sys_modules_226630.module_type_store, module_type_store)
    else:
        from matplotlib.widgets import SubplotTool

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.widgets', None, module_type_store, ['SubplotTool'], [SubplotTool])

else:
    # Assigning a type to the variable 'matplotlib.widgets' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.widgets', import_226629)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'from matplotlib import backend_tools, cbook, mcolors, lines, verbose, rcParams' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_226631 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib')

if (type(import_226631) is not StypyTypeError):

    if (import_226631 != 'pyd_module'):
        __import__(import_226631)
        sys_modules_226632 = sys.modules[import_226631]
        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib', sys_modules_226632.module_type_store, module_type_store, ['backend_tools', 'cbook', 'colors', 'lines', 'verbose', 'rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 39, 0), __file__, sys_modules_226632, sys_modules_226632.module_type_store, module_type_store)
    else:
        from matplotlib import backend_tools, cbook, colors as mcolors, lines, verbose, rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib', None, module_type_store, ['backend_tools', 'cbook', 'colors', 'lines', 'verbose', 'rcParams'], [backend_tools, cbook, mcolors, lines, verbose, rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib', import_226631)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a BinOp to a Name (line 42):

# Assigning a BinOp to a Name (line 42):
unicode_226633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'unicode', u'%s.%s.%s')

# Obtaining an instance of the builtin type 'tuple' (line 43)
tuple_226634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 43)
# Adding element type (line 43)

# Call to get_major_version(...): (line 43)
# Processing the call keyword arguments (line 43)
kwargs_226637 = {}
# Getting the type of 'Gtk' (line 43)
Gtk_226635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'Gtk', False)
# Obtaining the member 'get_major_version' of a type (line 43)
get_major_version_226636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), Gtk_226635, 'get_major_version')
# Calling get_major_version(args, kwargs) (line 43)
get_major_version_call_result_226638 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), get_major_version_226636, *[], **kwargs_226637)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), tuple_226634, get_major_version_call_result_226638)
# Adding element type (line 43)

# Call to get_micro_version(...): (line 43)
# Processing the call keyword arguments (line 43)
kwargs_226641 = {}
# Getting the type of 'Gtk' (line 43)
Gtk_226639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 29), 'Gtk', False)
# Obtaining the member 'get_micro_version' of a type (line 43)
get_micro_version_226640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 29), Gtk_226639, 'get_micro_version')
# Calling get_micro_version(args, kwargs) (line 43)
get_micro_version_call_result_226642 = invoke(stypy.reporting.localization.Localization(__file__, 43, 29), get_micro_version_226640, *[], **kwargs_226641)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), tuple_226634, get_micro_version_call_result_226642)
# Adding element type (line 43)

# Call to get_minor_version(...): (line 43)
# Processing the call keyword arguments (line 43)
kwargs_226645 = {}
# Getting the type of 'Gtk' (line 43)
Gtk_226643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 54), 'Gtk', False)
# Obtaining the member 'get_minor_version' of a type (line 43)
get_minor_version_226644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 54), Gtk_226643, 'get_minor_version')
# Calling get_minor_version(args, kwargs) (line 43)
get_minor_version_call_result_226646 = invoke(stypy.reporting.localization.Localization(__file__, 43, 54), get_minor_version_226644, *[], **kwargs_226645)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), tuple_226634, get_minor_version_call_result_226646)

# Applying the binary operator '%' (line 42)
result_mod_226647 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 18), '%', unicode_226633, tuple_226634)

# Assigning a type to the variable 'backend_version' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'backend_version', result_mod_226647)

# Assigning a Num to a Name (line 47):

# Assigning a Num to a Name (line 47):
int_226648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'int')
# Assigning a type to the variable 'PIXELS_PER_INCH' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'PIXELS_PER_INCH', int_226648)

# Assigning a Dict to a Name (line 49):

# Assigning a Dict to a Name (line 49):

# Obtaining an instance of the builtin type 'dict' (line 49)
dict_226649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 49)
# Adding element type (key, value) (line 49)
# Getting the type of 'cursors' (line 50)
cursors_226650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'cursors')
# Obtaining the member 'MOVE' of a type (line 50)
MOVE_226651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 4), cursors_226650, 'MOVE')

# Call to new(...): (line 50)
# Processing the call arguments (line 50)
# Getting the type of 'Gdk' (line 50)
Gdk_226655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 43), 'Gdk', False)
# Obtaining the member 'CursorType' of a type (line 50)
CursorType_226656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 43), Gdk_226655, 'CursorType')
# Obtaining the member 'FLEUR' of a type (line 50)
FLEUR_226657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 43), CursorType_226656, 'FLEUR')
# Processing the call keyword arguments (line 50)
kwargs_226658 = {}
# Getting the type of 'Gdk' (line 50)
Gdk_226652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'Gdk', False)
# Obtaining the member 'Cursor' of a type (line 50)
Cursor_226653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 28), Gdk_226652, 'Cursor')
# Obtaining the member 'new' of a type (line 50)
new_226654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 28), Cursor_226653, 'new')
# Calling new(args, kwargs) (line 50)
new_call_result_226659 = invoke(stypy.reporting.localization.Localization(__file__, 50, 28), new_226654, *[FLEUR_226657], **kwargs_226658)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 10), dict_226649, (MOVE_226651, new_call_result_226659))
# Adding element type (key, value) (line 49)
# Getting the type of 'cursors' (line 51)
cursors_226660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'cursors')
# Obtaining the member 'HAND' of a type (line 51)
HAND_226661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), cursors_226660, 'HAND')

# Call to new(...): (line 51)
# Processing the call arguments (line 51)
# Getting the type of 'Gdk' (line 51)
Gdk_226665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 43), 'Gdk', False)
# Obtaining the member 'CursorType' of a type (line 51)
CursorType_226666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 43), Gdk_226665, 'CursorType')
# Obtaining the member 'HAND2' of a type (line 51)
HAND2_226667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 43), CursorType_226666, 'HAND2')
# Processing the call keyword arguments (line 51)
kwargs_226668 = {}
# Getting the type of 'Gdk' (line 51)
Gdk_226662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'Gdk', False)
# Obtaining the member 'Cursor' of a type (line 51)
Cursor_226663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 28), Gdk_226662, 'Cursor')
# Obtaining the member 'new' of a type (line 51)
new_226664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 28), Cursor_226663, 'new')
# Calling new(args, kwargs) (line 51)
new_call_result_226669 = invoke(stypy.reporting.localization.Localization(__file__, 51, 28), new_226664, *[HAND2_226667], **kwargs_226668)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 10), dict_226649, (HAND_226661, new_call_result_226669))
# Adding element type (key, value) (line 49)
# Getting the type of 'cursors' (line 52)
cursors_226670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'cursors')
# Obtaining the member 'POINTER' of a type (line 52)
POINTER_226671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), cursors_226670, 'POINTER')

# Call to new(...): (line 52)
# Processing the call arguments (line 52)
# Getting the type of 'Gdk' (line 52)
Gdk_226675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 43), 'Gdk', False)
# Obtaining the member 'CursorType' of a type (line 52)
CursorType_226676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 43), Gdk_226675, 'CursorType')
# Obtaining the member 'LEFT_PTR' of a type (line 52)
LEFT_PTR_226677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 43), CursorType_226676, 'LEFT_PTR')
# Processing the call keyword arguments (line 52)
kwargs_226678 = {}
# Getting the type of 'Gdk' (line 52)
Gdk_226672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 28), 'Gdk', False)
# Obtaining the member 'Cursor' of a type (line 52)
Cursor_226673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 28), Gdk_226672, 'Cursor')
# Obtaining the member 'new' of a type (line 52)
new_226674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 28), Cursor_226673, 'new')
# Calling new(args, kwargs) (line 52)
new_call_result_226679 = invoke(stypy.reporting.localization.Localization(__file__, 52, 28), new_226674, *[LEFT_PTR_226677], **kwargs_226678)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 10), dict_226649, (POINTER_226671, new_call_result_226679))
# Adding element type (key, value) (line 49)
# Getting the type of 'cursors' (line 53)
cursors_226680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'cursors')
# Obtaining the member 'SELECT_REGION' of a type (line 53)
SELECT_REGION_226681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 4), cursors_226680, 'SELECT_REGION')

# Call to new(...): (line 53)
# Processing the call arguments (line 53)
# Getting the type of 'Gdk' (line 53)
Gdk_226685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 43), 'Gdk', False)
# Obtaining the member 'CursorType' of a type (line 53)
CursorType_226686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 43), Gdk_226685, 'CursorType')
# Obtaining the member 'TCROSS' of a type (line 53)
TCROSS_226687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 43), CursorType_226686, 'TCROSS')
# Processing the call keyword arguments (line 53)
kwargs_226688 = {}
# Getting the type of 'Gdk' (line 53)
Gdk_226682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 'Gdk', False)
# Obtaining the member 'Cursor' of a type (line 53)
Cursor_226683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 28), Gdk_226682, 'Cursor')
# Obtaining the member 'new' of a type (line 53)
new_226684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 28), Cursor_226683, 'new')
# Calling new(args, kwargs) (line 53)
new_call_result_226689 = invoke(stypy.reporting.localization.Localization(__file__, 53, 28), new_226684, *[TCROSS_226687], **kwargs_226688)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 10), dict_226649, (SELECT_REGION_226681, new_call_result_226689))
# Adding element type (key, value) (line 49)
# Getting the type of 'cursors' (line 54)
cursors_226690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'cursors')
# Obtaining the member 'WAIT' of a type (line 54)
WAIT_226691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 4), cursors_226690, 'WAIT')

# Call to new(...): (line 54)
# Processing the call arguments (line 54)
# Getting the type of 'Gdk' (line 54)
Gdk_226695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 43), 'Gdk', False)
# Obtaining the member 'CursorType' of a type (line 54)
CursorType_226696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 43), Gdk_226695, 'CursorType')
# Obtaining the member 'WATCH' of a type (line 54)
WATCH_226697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 43), CursorType_226696, 'WATCH')
# Processing the call keyword arguments (line 54)
kwargs_226698 = {}
# Getting the type of 'Gdk' (line 54)
Gdk_226692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'Gdk', False)
# Obtaining the member 'Cursor' of a type (line 54)
Cursor_226693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 28), Gdk_226692, 'Cursor')
# Obtaining the member 'new' of a type (line 54)
new_226694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 28), Cursor_226693, 'new')
# Calling new(args, kwargs) (line 54)
new_call_result_226699 = invoke(stypy.reporting.localization.Localization(__file__, 54, 28), new_226694, *[WATCH_226697], **kwargs_226698)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 10), dict_226649, (WAIT_226691, new_call_result_226699))

# Assigning a type to the variable 'cursord' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'cursord', dict_226649)
# Declaration of the 'TimerGTK3' class
# Getting the type of 'TimerBase' (line 58)
TimerBase_226700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'TimerBase')

class TimerGTK3(TimerBase_226700, ):
    unicode_226701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, (-1)), 'unicode', u'\n    Subclass of :class:`backend_bases.TimerBase` using GTK3 for timer events.\n\n    Attributes\n    ----------\n    interval : int\n        The time between timer events in milliseconds. Default is 1000 ms.\n    single_shot : bool\n        Boolean flag indicating whether this timer should operate as single\n        shot (run once and then stop). Defaults to False.\n    callbacks : list\n        Stores list of (func, args) tuples that will be called upon timer\n        events. This list can be manipulated directly, or the functions\n        `add_callback` and `remove_callback` can be used.\n\n    ')

    @norecursion
    def _timer_start(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_timer_start'
        module_type_store = module_type_store.open_function_context('_timer_start', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimerGTK3._timer_start.__dict__.__setitem__('stypy_localization', localization)
        TimerGTK3._timer_start.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimerGTK3._timer_start.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimerGTK3._timer_start.__dict__.__setitem__('stypy_function_name', 'TimerGTK3._timer_start')
        TimerGTK3._timer_start.__dict__.__setitem__('stypy_param_names_list', [])
        TimerGTK3._timer_start.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimerGTK3._timer_start.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimerGTK3._timer_start.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimerGTK3._timer_start.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimerGTK3._timer_start.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimerGTK3._timer_start.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerGTK3._timer_start', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_timer_start', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_timer_start(...)' code ##################

        
        # Call to _timer_stop(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_226704 = {}
        # Getting the type of 'self' (line 78)
        self_226702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self', False)
        # Obtaining the member '_timer_stop' of a type (line 78)
        _timer_stop_226703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_226702, '_timer_stop')
        # Calling _timer_stop(args, kwargs) (line 78)
        _timer_stop_call_result_226705 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), _timer_stop_226703, *[], **kwargs_226704)
        
        
        # Assigning a Call to a Attribute (line 79):
        
        # Assigning a Call to a Attribute (line 79):
        
        # Call to timeout_add(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'self' (line 79)
        self_226708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 39), 'self', False)
        # Obtaining the member '_interval' of a type (line 79)
        _interval_226709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 39), self_226708, '_interval')
        # Getting the type of 'self' (line 79)
        self_226710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 55), 'self', False)
        # Obtaining the member '_on_timer' of a type (line 79)
        _on_timer_226711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 55), self_226710, '_on_timer')
        # Processing the call keyword arguments (line 79)
        kwargs_226712 = {}
        # Getting the type of 'GLib' (line 79)
        GLib_226706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'GLib', False)
        # Obtaining the member 'timeout_add' of a type (line 79)
        timeout_add_226707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 22), GLib_226706, 'timeout_add')
        # Calling timeout_add(args, kwargs) (line 79)
        timeout_add_call_result_226713 = invoke(stypy.reporting.localization.Localization(__file__, 79, 22), timeout_add_226707, *[_interval_226709, _on_timer_226711], **kwargs_226712)
        
        # Getting the type of 'self' (line 79)
        self_226714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Setting the type of the member '_timer' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_226714, '_timer', timeout_add_call_result_226713)
        
        # ################# End of '_timer_start(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_timer_start' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_226715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_226715)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_timer_start'
        return stypy_return_type_226715


    @norecursion
    def _timer_stop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_timer_stop'
        module_type_store = module_type_store.open_function_context('_timer_stop', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimerGTK3._timer_stop.__dict__.__setitem__('stypy_localization', localization)
        TimerGTK3._timer_stop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimerGTK3._timer_stop.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimerGTK3._timer_stop.__dict__.__setitem__('stypy_function_name', 'TimerGTK3._timer_stop')
        TimerGTK3._timer_stop.__dict__.__setitem__('stypy_param_names_list', [])
        TimerGTK3._timer_stop.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimerGTK3._timer_stop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimerGTK3._timer_stop.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimerGTK3._timer_stop.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimerGTK3._timer_stop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimerGTK3._timer_stop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerGTK3._timer_stop', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_timer_stop', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_timer_stop(...)' code ##################

        
        
        # Getting the type of 'self' (line 82)
        self_226716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'self')
        # Obtaining the member '_timer' of a type (line 82)
        _timer_226717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 11), self_226716, '_timer')
        # Getting the type of 'None' (line 82)
        None_226718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 30), 'None')
        # Applying the binary operator 'isnot' (line 82)
        result_is_not_226719 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 11), 'isnot', _timer_226717, None_226718)
        
        # Testing the type of an if condition (line 82)
        if_condition_226720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), result_is_not_226719)
        # Assigning a type to the variable 'if_condition_226720' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'if_condition_226720', if_condition_226720)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to source_remove(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'self' (line 83)
        self_226723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 31), 'self', False)
        # Obtaining the member '_timer' of a type (line 83)
        _timer_226724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 31), self_226723, '_timer')
        # Processing the call keyword arguments (line 83)
        kwargs_226725 = {}
        # Getting the type of 'GLib' (line 83)
        GLib_226721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'GLib', False)
        # Obtaining the member 'source_remove' of a type (line 83)
        source_remove_226722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), GLib_226721, 'source_remove')
        # Calling source_remove(args, kwargs) (line 83)
        source_remove_call_result_226726 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), source_remove_226722, *[_timer_226724], **kwargs_226725)
        
        
        # Assigning a Name to a Attribute (line 84):
        
        # Assigning a Name to a Attribute (line 84):
        # Getting the type of 'None' (line 84)
        None_226727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'None')
        # Getting the type of 'self' (line 84)
        self_226728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'self')
        # Setting the type of the member '_timer' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), self_226728, '_timer', None_226727)
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_timer_stop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_timer_stop' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_226729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_226729)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_timer_stop'
        return stypy_return_type_226729


    @norecursion
    def _timer_set_interval(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_timer_set_interval'
        module_type_store = module_type_store.open_function_context('_timer_set_interval', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimerGTK3._timer_set_interval.__dict__.__setitem__('stypy_localization', localization)
        TimerGTK3._timer_set_interval.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimerGTK3._timer_set_interval.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimerGTK3._timer_set_interval.__dict__.__setitem__('stypy_function_name', 'TimerGTK3._timer_set_interval')
        TimerGTK3._timer_set_interval.__dict__.__setitem__('stypy_param_names_list', [])
        TimerGTK3._timer_set_interval.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimerGTK3._timer_set_interval.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimerGTK3._timer_set_interval.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimerGTK3._timer_set_interval.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimerGTK3._timer_set_interval.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimerGTK3._timer_set_interval.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerGTK3._timer_set_interval', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_timer_set_interval', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_timer_set_interval(...)' code ##################

        
        
        # Getting the type of 'self' (line 88)
        self_226730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'self')
        # Obtaining the member '_timer' of a type (line 88)
        _timer_226731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 11), self_226730, '_timer')
        # Getting the type of 'None' (line 88)
        None_226732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'None')
        # Applying the binary operator 'isnot' (line 88)
        result_is_not_226733 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 11), 'isnot', _timer_226731, None_226732)
        
        # Testing the type of an if condition (line 88)
        if_condition_226734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 8), result_is_not_226733)
        # Assigning a type to the variable 'if_condition_226734' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'if_condition_226734', if_condition_226734)
        # SSA begins for if statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _timer_stop(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_226737 = {}
        # Getting the type of 'self' (line 89)
        self_226735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'self', False)
        # Obtaining the member '_timer_stop' of a type (line 89)
        _timer_stop_226736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), self_226735, '_timer_stop')
        # Calling _timer_stop(args, kwargs) (line 89)
        _timer_stop_call_result_226738 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), _timer_stop_226736, *[], **kwargs_226737)
        
        
        # Call to _timer_start(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_226741 = {}
        # Getting the type of 'self' (line 90)
        self_226739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'self', False)
        # Obtaining the member '_timer_start' of a type (line 90)
        _timer_start_226740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), self_226739, '_timer_start')
        # Calling _timer_start(args, kwargs) (line 90)
        _timer_start_call_result_226742 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), _timer_start_226740, *[], **kwargs_226741)
        
        # SSA join for if statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_timer_set_interval(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_timer_set_interval' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_226743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_226743)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_timer_set_interval'
        return stypy_return_type_226743


    @norecursion
    def _on_timer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_on_timer'
        module_type_store = module_type_store.open_function_context('_on_timer', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimerGTK3._on_timer.__dict__.__setitem__('stypy_localization', localization)
        TimerGTK3._on_timer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimerGTK3._on_timer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimerGTK3._on_timer.__dict__.__setitem__('stypy_function_name', 'TimerGTK3._on_timer')
        TimerGTK3._on_timer.__dict__.__setitem__('stypy_param_names_list', [])
        TimerGTK3._on_timer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimerGTK3._on_timer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimerGTK3._on_timer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimerGTK3._on_timer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimerGTK3._on_timer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimerGTK3._on_timer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerGTK3._on_timer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_on_timer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_on_timer(...)' code ##################

        
        # Call to _on_timer(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'self' (line 93)
        self_226746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'self', False)
        # Processing the call keyword arguments (line 93)
        kwargs_226747 = {}
        # Getting the type of 'TimerBase' (line 93)
        TimerBase_226744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'TimerBase', False)
        # Obtaining the member '_on_timer' of a type (line 93)
        _on_timer_226745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), TimerBase_226744, '_on_timer')
        # Calling _on_timer(args, kwargs) (line 93)
        _on_timer_call_result_226748 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), _on_timer_226745, *[self_226746], **kwargs_226747)
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'self' (line 97)
        self_226750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'self', False)
        # Obtaining the member 'callbacks' of a type (line 97)
        callbacks_226751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), self_226750, 'callbacks')
        # Processing the call keyword arguments (line 97)
        kwargs_226752 = {}
        # Getting the type of 'len' (line 97)
        len_226749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'len', False)
        # Calling len(args, kwargs) (line 97)
        len_call_result_226753 = invoke(stypy.reporting.localization.Localization(__file__, 97, 11), len_226749, *[callbacks_226751], **kwargs_226752)
        
        int_226754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 33), 'int')
        # Applying the binary operator '>' (line 97)
        result_gt_226755 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), '>', len_call_result_226753, int_226754)
        
        
        # Getting the type of 'self' (line 97)
        self_226756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 43), 'self')
        # Obtaining the member '_single' of a type (line 97)
        _single_226757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 43), self_226756, '_single')
        # Applying the 'not' unary operator (line 97)
        result_not__226758 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 39), 'not', _single_226757)
        
        # Applying the binary operator 'and' (line 97)
        result_and_keyword_226759 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), 'and', result_gt_226755, result_not__226758)
        
        # Testing the type of an if condition (line 97)
        if_condition_226760 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), result_and_keyword_226759)
        # Assigning a type to the variable 'if_condition_226760' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_226760', if_condition_226760)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 98)
        True_226761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'stypy_return_type', True_226761)
        # SSA branch for the else part of an if statement (line 97)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 100):
        
        # Assigning a Name to a Attribute (line 100):
        # Getting the type of 'None' (line 100)
        None_226762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 26), 'None')
        # Getting the type of 'self' (line 100)
        self_226763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self')
        # Setting the type of the member '_timer' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_226763, '_timer', None_226762)
        # Getting the type of 'False' (line 101)
        False_226764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type', False_226764)
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_on_timer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_on_timer' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_226765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_226765)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_on_timer'
        return stypy_return_type_226765


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 58, 0, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerGTK3.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TimerGTK3' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'TimerGTK3', TimerGTK3)
# Declaration of the 'FigureCanvasGTK3' class
# Getting the type of 'Gtk' (line 104)
Gtk_226766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 'Gtk')
# Obtaining the member 'DrawingArea' of a type (line 104)
DrawingArea_226767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 23), Gtk_226766, 'DrawingArea')
# Getting the type of 'FigureCanvasBase' (line 104)
FigureCanvasBase_226768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 40), 'FigureCanvasBase')

class FigureCanvasGTK3(DrawingArea_226767, FigureCanvasBase_226768, ):
    
    # Assigning a Dict to a Name (line 105):
    
    # Assigning a BinOp to a Name (line 158):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.__init__', ['figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'self' (line 170)
        self_226771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 34), 'self', False)
        # Getting the type of 'figure' (line 170)
        figure_226772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 40), 'figure', False)
        # Processing the call keyword arguments (line 170)
        kwargs_226773 = {}
        # Getting the type of 'FigureCanvasBase' (line 170)
        FigureCanvasBase_226769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'FigureCanvasBase', False)
        # Obtaining the member '__init__' of a type (line 170)
        init___226770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), FigureCanvasBase_226769, '__init__')
        # Calling __init__(args, kwargs) (line 170)
        init___call_result_226774 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), init___226770, *[self_226771, figure_226772], **kwargs_226773)
        
        
        # Call to __init__(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'self' (line 171)
        self_226778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 33), 'self', False)
        # Processing the call keyword arguments (line 171)
        kwargs_226779 = {}
        # Getting the type of 'GObject' (line 171)
        GObject_226775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'GObject', False)
        # Obtaining the member 'GObject' of a type (line 171)
        GObject_226776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), GObject_226775, 'GObject')
        # Obtaining the member '__init__' of a type (line 171)
        init___226777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), GObject_226776, '__init__')
        # Calling __init__(args, kwargs) (line 171)
        init___call_result_226780 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), init___226777, *[self_226778], **kwargs_226779)
        
        
        # Assigning a Num to a Attribute (line 173):
        
        # Assigning a Num to a Attribute (line 173):
        int_226781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 30), 'int')
        # Getting the type of 'self' (line 173)
        self_226782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self')
        # Setting the type of the member '_idle_draw_id' of a type (line 173)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_226782, '_idle_draw_id', int_226781)
        
        # Assigning a Name to a Attribute (line 174):
        
        # Assigning a Name to a Attribute (line 174):
        # Getting the type of 'None' (line 174)
        None_226783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 30), 'None')
        # Getting the type of 'self' (line 174)
        self_226784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self')
        # Setting the type of the member '_lastCursor' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_226784, '_lastCursor', None_226783)
        
        # Call to connect(...): (line 176)
        # Processing the call arguments (line 176)
        unicode_226787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 21), 'unicode', u'scroll_event')
        # Getting the type of 'self' (line 176)
        self_226788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 45), 'self', False)
        # Obtaining the member 'scroll_event' of a type (line 176)
        scroll_event_226789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 45), self_226788, 'scroll_event')
        # Processing the call keyword arguments (line 176)
        kwargs_226790 = {}
        # Getting the type of 'self' (line 176)
        self_226785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', False)
        # Obtaining the member 'connect' of a type (line 176)
        connect_226786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_226785, 'connect')
        # Calling connect(args, kwargs) (line 176)
        connect_call_result_226791 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), connect_226786, *[unicode_226787, scroll_event_226789], **kwargs_226790)
        
        
        # Call to connect(...): (line 177)
        # Processing the call arguments (line 177)
        unicode_226794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 21), 'unicode', u'button_press_event')
        # Getting the type of 'self' (line 177)
        self_226795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 45), 'self', False)
        # Obtaining the member 'button_press_event' of a type (line 177)
        button_press_event_226796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 45), self_226795, 'button_press_event')
        # Processing the call keyword arguments (line 177)
        kwargs_226797 = {}
        # Getting the type of 'self' (line 177)
        self_226792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self', False)
        # Obtaining the member 'connect' of a type (line 177)
        connect_226793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_226792, 'connect')
        # Calling connect(args, kwargs) (line 177)
        connect_call_result_226798 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), connect_226793, *[unicode_226794, button_press_event_226796], **kwargs_226797)
        
        
        # Call to connect(...): (line 178)
        # Processing the call arguments (line 178)
        unicode_226801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 21), 'unicode', u'button_release_event')
        # Getting the type of 'self' (line 178)
        self_226802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 45), 'self', False)
        # Obtaining the member 'button_release_event' of a type (line 178)
        button_release_event_226803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 45), self_226802, 'button_release_event')
        # Processing the call keyword arguments (line 178)
        kwargs_226804 = {}
        # Getting the type of 'self' (line 178)
        self_226799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self', False)
        # Obtaining the member 'connect' of a type (line 178)
        connect_226800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_226799, 'connect')
        # Calling connect(args, kwargs) (line 178)
        connect_call_result_226805 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), connect_226800, *[unicode_226801, button_release_event_226803], **kwargs_226804)
        
        
        # Call to connect(...): (line 179)
        # Processing the call arguments (line 179)
        unicode_226808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 21), 'unicode', u'configure_event')
        # Getting the type of 'self' (line 179)
        self_226809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 45), 'self', False)
        # Obtaining the member 'configure_event' of a type (line 179)
        configure_event_226810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 45), self_226809, 'configure_event')
        # Processing the call keyword arguments (line 179)
        kwargs_226811 = {}
        # Getting the type of 'self' (line 179)
        self_226806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self', False)
        # Obtaining the member 'connect' of a type (line 179)
        connect_226807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_226806, 'connect')
        # Calling connect(args, kwargs) (line 179)
        connect_call_result_226812 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), connect_226807, *[unicode_226808, configure_event_226810], **kwargs_226811)
        
        
        # Call to connect(...): (line 180)
        # Processing the call arguments (line 180)
        unicode_226815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 21), 'unicode', u'draw')
        # Getting the type of 'self' (line 180)
        self_226816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 45), 'self', False)
        # Obtaining the member 'on_draw_event' of a type (line 180)
        on_draw_event_226817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 45), self_226816, 'on_draw_event')
        # Processing the call keyword arguments (line 180)
        kwargs_226818 = {}
        # Getting the type of 'self' (line 180)
        self_226813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self', False)
        # Obtaining the member 'connect' of a type (line 180)
        connect_226814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_226813, 'connect')
        # Calling connect(args, kwargs) (line 180)
        connect_call_result_226819 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), connect_226814, *[unicode_226815, on_draw_event_226817], **kwargs_226818)
        
        
        # Call to connect(...): (line 181)
        # Processing the call arguments (line 181)
        unicode_226822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 21), 'unicode', u'key_press_event')
        # Getting the type of 'self' (line 181)
        self_226823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 45), 'self', False)
        # Obtaining the member 'key_press_event' of a type (line 181)
        key_press_event_226824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 45), self_226823, 'key_press_event')
        # Processing the call keyword arguments (line 181)
        kwargs_226825 = {}
        # Getting the type of 'self' (line 181)
        self_226820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self', False)
        # Obtaining the member 'connect' of a type (line 181)
        connect_226821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_226820, 'connect')
        # Calling connect(args, kwargs) (line 181)
        connect_call_result_226826 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), connect_226821, *[unicode_226822, key_press_event_226824], **kwargs_226825)
        
        
        # Call to connect(...): (line 182)
        # Processing the call arguments (line 182)
        unicode_226829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 21), 'unicode', u'key_release_event')
        # Getting the type of 'self' (line 182)
        self_226830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 45), 'self', False)
        # Obtaining the member 'key_release_event' of a type (line 182)
        key_release_event_226831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 45), self_226830, 'key_release_event')
        # Processing the call keyword arguments (line 182)
        kwargs_226832 = {}
        # Getting the type of 'self' (line 182)
        self_226827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'self', False)
        # Obtaining the member 'connect' of a type (line 182)
        connect_226828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), self_226827, 'connect')
        # Calling connect(args, kwargs) (line 182)
        connect_call_result_226833 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), connect_226828, *[unicode_226829, key_release_event_226831], **kwargs_226832)
        
        
        # Call to connect(...): (line 183)
        # Processing the call arguments (line 183)
        unicode_226836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 21), 'unicode', u'motion_notify_event')
        # Getting the type of 'self' (line 183)
        self_226837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 45), 'self', False)
        # Obtaining the member 'motion_notify_event' of a type (line 183)
        motion_notify_event_226838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 45), self_226837, 'motion_notify_event')
        # Processing the call keyword arguments (line 183)
        kwargs_226839 = {}
        # Getting the type of 'self' (line 183)
        self_226834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'self', False)
        # Obtaining the member 'connect' of a type (line 183)
        connect_226835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), self_226834, 'connect')
        # Calling connect(args, kwargs) (line 183)
        connect_call_result_226840 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), connect_226835, *[unicode_226836, motion_notify_event_226838], **kwargs_226839)
        
        
        # Call to connect(...): (line 184)
        # Processing the call arguments (line 184)
        unicode_226843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 21), 'unicode', u'leave_notify_event')
        # Getting the type of 'self' (line 184)
        self_226844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 45), 'self', False)
        # Obtaining the member 'leave_notify_event' of a type (line 184)
        leave_notify_event_226845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 45), self_226844, 'leave_notify_event')
        # Processing the call keyword arguments (line 184)
        kwargs_226846 = {}
        # Getting the type of 'self' (line 184)
        self_226841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'self', False)
        # Obtaining the member 'connect' of a type (line 184)
        connect_226842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), self_226841, 'connect')
        # Calling connect(args, kwargs) (line 184)
        connect_call_result_226847 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), connect_226842, *[unicode_226843, leave_notify_event_226845], **kwargs_226846)
        
        
        # Call to connect(...): (line 185)
        # Processing the call arguments (line 185)
        unicode_226850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 21), 'unicode', u'enter_notify_event')
        # Getting the type of 'self' (line 185)
        self_226851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 45), 'self', False)
        # Obtaining the member 'enter_notify_event' of a type (line 185)
        enter_notify_event_226852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 45), self_226851, 'enter_notify_event')
        # Processing the call keyword arguments (line 185)
        kwargs_226853 = {}
        # Getting the type of 'self' (line 185)
        self_226848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'self', False)
        # Obtaining the member 'connect' of a type (line 185)
        connect_226849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), self_226848, 'connect')
        # Calling connect(args, kwargs) (line 185)
        connect_call_result_226854 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), connect_226849, *[unicode_226850, enter_notify_event_226852], **kwargs_226853)
        
        
        # Call to connect(...): (line 186)
        # Processing the call arguments (line 186)
        unicode_226857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 21), 'unicode', u'size_allocate')
        # Getting the type of 'self' (line 186)
        self_226858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 45), 'self', False)
        # Obtaining the member 'size_allocate' of a type (line 186)
        size_allocate_226859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 45), self_226858, 'size_allocate')
        # Processing the call keyword arguments (line 186)
        kwargs_226860 = {}
        # Getting the type of 'self' (line 186)
        self_226855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self', False)
        # Obtaining the member 'connect' of a type (line 186)
        connect_226856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_226855, 'connect')
        # Calling connect(args, kwargs) (line 186)
        connect_call_result_226861 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), connect_226856, *[unicode_226857, size_allocate_226859], **kwargs_226860)
        
        
        # Call to set_events(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'self' (line 188)
        self_226864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'self', False)
        # Obtaining the member '__class__' of a type (line 188)
        class___226865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), self_226864, '__class__')
        # Obtaining the member 'event_mask' of a type (line 188)
        event_mask_226866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), class___226865, 'event_mask')
        # Processing the call keyword arguments (line 188)
        kwargs_226867 = {}
        # Getting the type of 'self' (line 188)
        self_226862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'self', False)
        # Obtaining the member 'set_events' of a type (line 188)
        set_events_226863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), self_226862, 'set_events')
        # Calling set_events(args, kwargs) (line 188)
        set_events_call_result_226868 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), set_events_226863, *[event_mask_226866], **kwargs_226867)
        
        
        # Call to set_double_buffered(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'True' (line 190)
        True_226871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 33), 'True', False)
        # Processing the call keyword arguments (line 190)
        kwargs_226872 = {}
        # Getting the type of 'self' (line 190)
        self_226869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'self', False)
        # Obtaining the member 'set_double_buffered' of a type (line 190)
        set_double_buffered_226870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), self_226869, 'set_double_buffered')
        # Calling set_double_buffered(args, kwargs) (line 190)
        set_double_buffered_call_result_226873 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), set_double_buffered_226870, *[True_226871], **kwargs_226872)
        
        
        # Call to set_can_focus(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'True' (line 191)
        True_226876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 27), 'True', False)
        # Processing the call keyword arguments (line 191)
        kwargs_226877 = {}
        # Getting the type of 'self' (line 191)
        self_226874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'self', False)
        # Obtaining the member 'set_can_focus' of a type (line 191)
        set_can_focus_226875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), self_226874, 'set_can_focus')
        # Calling set_can_focus(args, kwargs) (line 191)
        set_can_focus_call_result_226878 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), set_can_focus_226875, *[True_226876], **kwargs_226877)
        
        
        # Call to _renderer_init(...): (line 192)
        # Processing the call keyword arguments (line 192)
        kwargs_226881 = {}
        # Getting the type of 'self' (line 192)
        self_226879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'self', False)
        # Obtaining the member '_renderer_init' of a type (line 192)
        _renderer_init_226880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 8), self_226879, '_renderer_init')
        # Calling _renderer_init(args, kwargs) (line 192)
        _renderer_init_call_result_226882 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), _renderer_init_226880, *[], **kwargs_226881)
        
        
        # Assigning a BoolOp to a Name (line 193):
        
        # Assigning a BoolOp to a Name (line 193):
        
        # Evaluating a boolean operation
        
        # Call to main_context_get_thread_default(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_226885 = {}
        # Getting the type of 'GLib' (line 193)
        GLib_226883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'GLib', False)
        # Obtaining the member 'main_context_get_thread_default' of a type (line 193)
        main_context_get_thread_default_226884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 26), GLib_226883, 'main_context_get_thread_default')
        # Calling main_context_get_thread_default(args, kwargs) (line 193)
        main_context_get_thread_default_call_result_226886 = invoke(stypy.reporting.localization.Localization(__file__, 193, 26), main_context_get_thread_default_226884, *[], **kwargs_226885)
        
        
        # Call to main_context_default(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_226889 = {}
        # Getting the type of 'GLib' (line 193)
        GLib_226887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 68), 'GLib', False)
        # Obtaining the member 'main_context_default' of a type (line 193)
        main_context_default_226888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 68), GLib_226887, 'main_context_default')
        # Calling main_context_default(args, kwargs) (line 193)
        main_context_default_call_result_226890 = invoke(stypy.reporting.localization.Localization(__file__, 193, 68), main_context_default_226888, *[], **kwargs_226889)
        
        # Applying the binary operator 'or' (line 193)
        result_or_keyword_226891 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 26), 'or', main_context_get_thread_default_call_result_226886, main_context_default_call_result_226890)
        
        # Assigning a type to the variable 'default_context' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'default_context', result_or_keyword_226891)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def destroy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'destroy'
        module_type_store = module_type_store.open_function_context('destroy', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.destroy.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.destroy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.destroy.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.destroy.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.destroy')
        FigureCanvasGTK3.destroy.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasGTK3.destroy.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.destroy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.destroy.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.destroy.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.destroy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.destroy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.destroy', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to close_event(...): (line 197)
        # Processing the call keyword arguments (line 197)
        kwargs_226894 = {}
        # Getting the type of 'self' (line 197)
        self_226892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self', False)
        # Obtaining the member 'close_event' of a type (line 197)
        close_event_226893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_226892, 'close_event')
        # Calling close_event(args, kwargs) (line 197)
        close_event_call_result_226895 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), close_event_226893, *[], **kwargs_226894)
        
        
        
        # Getting the type of 'self' (line 198)
        self_226896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'self')
        # Obtaining the member '_idle_draw_id' of a type (line 198)
        _idle_draw_id_226897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), self_226896, '_idle_draw_id')
        int_226898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 33), 'int')
        # Applying the binary operator '!=' (line 198)
        result_ne_226899 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), '!=', _idle_draw_id_226897, int_226898)
        
        # Testing the type of an if condition (line 198)
        if_condition_226900 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_ne_226899)
        # Assigning a type to the variable 'if_condition_226900' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_226900', if_condition_226900)
        # SSA begins for if statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to source_remove(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'self' (line 199)
        self_226903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 'self', False)
        # Obtaining the member '_idle_draw_id' of a type (line 199)
        _idle_draw_id_226904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 31), self_226903, '_idle_draw_id')
        # Processing the call keyword arguments (line 199)
        kwargs_226905 = {}
        # Getting the type of 'GLib' (line 199)
        GLib_226901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'GLib', False)
        # Obtaining the member 'source_remove' of a type (line 199)
        source_remove_226902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), GLib_226901, 'source_remove')
        # Calling source_remove(args, kwargs) (line 199)
        source_remove_call_result_226906 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), source_remove_226902, *[_idle_draw_id_226904], **kwargs_226905)
        
        # SSA join for if statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'destroy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'destroy' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_226907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_226907)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'destroy'
        return stypy_return_type_226907


    @norecursion
    def scroll_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'scroll_event'
        module_type_store = module_type_store.open_function_context('scroll_event', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.scroll_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.scroll_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.scroll_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.scroll_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.scroll_event')
        FigureCanvasGTK3.scroll_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'event'])
        FigureCanvasGTK3.scroll_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.scroll_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.scroll_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.scroll_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.scroll_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.scroll_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.scroll_event', ['widget', 'event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'scroll_event', localization, ['widget', 'event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'scroll_event(...)' code ##################

        
        # Assigning a Attribute to a Name (line 202):
        
        # Assigning a Attribute to a Name (line 202):
        # Getting the type of 'event' (line 202)
        event_226908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'event')
        # Obtaining the member 'x' of a type (line 202)
        x_226909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), event_226908, 'x')
        # Assigning a type to the variable 'x' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'x', x_226909)
        
        # Assigning a BinOp to a Name (line 204):
        
        # Assigning a BinOp to a Name (line 204):
        
        # Call to get_allocation(...): (line 204)
        # Processing the call keyword arguments (line 204)
        kwargs_226912 = {}
        # Getting the type of 'self' (line 204)
        self_226910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'self', False)
        # Obtaining the member 'get_allocation' of a type (line 204)
        get_allocation_226911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), self_226910, 'get_allocation')
        # Calling get_allocation(args, kwargs) (line 204)
        get_allocation_call_result_226913 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), get_allocation_226911, *[], **kwargs_226912)
        
        # Obtaining the member 'height' of a type (line 204)
        height_226914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), get_allocation_call_result_226913, 'height')
        # Getting the type of 'event' (line 204)
        event_226915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 43), 'event')
        # Obtaining the member 'y' of a type (line 204)
        y_226916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 43), event_226915, 'y')
        # Applying the binary operator '-' (line 204)
        result_sub_226917 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 12), '-', height_226914, y_226916)
        
        # Assigning a type to the variable 'y' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'y', result_sub_226917)
        
        
        # Getting the type of 'event' (line 205)
        event_226918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'event')
        # Obtaining the member 'direction' of a type (line 205)
        direction_226919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 11), event_226918, 'direction')
        # Getting the type of 'Gdk' (line 205)
        Gdk_226920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 28), 'Gdk')
        # Obtaining the member 'ScrollDirection' of a type (line 205)
        ScrollDirection_226921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 28), Gdk_226920, 'ScrollDirection')
        # Obtaining the member 'UP' of a type (line 205)
        UP_226922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 28), ScrollDirection_226921, 'UP')
        # Applying the binary operator '==' (line 205)
        result_eq_226923 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), '==', direction_226919, UP_226922)
        
        # Testing the type of an if condition (line 205)
        if_condition_226924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), result_eq_226923)
        # Assigning a type to the variable 'if_condition_226924' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_226924', if_condition_226924)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 206):
        
        # Assigning a Num to a Name (line 206):
        int_226925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 19), 'int')
        # Assigning a type to the variable 'step' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'step', int_226925)
        # SSA branch for the else part of an if statement (line 205)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 208):
        
        # Assigning a Num to a Name (line 208):
        int_226926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 19), 'int')
        # Assigning a type to the variable 'step' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'step', int_226926)
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to scroll_event(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'self' (line 209)
        self_226929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 38), 'self', False)
        # Getting the type of 'x' (line 209)
        x_226930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 44), 'x', False)
        # Getting the type of 'y' (line 209)
        y_226931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 47), 'y', False)
        # Getting the type of 'step' (line 209)
        step_226932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 50), 'step', False)
        # Processing the call keyword arguments (line 209)
        # Getting the type of 'event' (line 209)
        event_226933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 65), 'event', False)
        keyword_226934 = event_226933
        kwargs_226935 = {'guiEvent': keyword_226934}
        # Getting the type of 'FigureCanvasBase' (line 209)
        FigureCanvasBase_226927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'scroll_event' of a type (line 209)
        scroll_event_226928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), FigureCanvasBase_226927, 'scroll_event')
        # Calling scroll_event(args, kwargs) (line 209)
        scroll_event_call_result_226936 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), scroll_event_226928, *[self_226929, x_226930, y_226931, step_226932], **kwargs_226935)
        
        # Getting the type of 'False' (line 210)
        False_226937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'stypy_return_type', False_226937)
        
        # ################# End of 'scroll_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'scroll_event' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_226938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_226938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'scroll_event'
        return stypy_return_type_226938


    @norecursion
    def button_press_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'button_press_event'
        module_type_store = module_type_store.open_function_context('button_press_event', 212, 4, False)
        # Assigning a type to the variable 'self' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.button_press_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.button_press_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.button_press_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.button_press_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.button_press_event')
        FigureCanvasGTK3.button_press_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'event'])
        FigureCanvasGTK3.button_press_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.button_press_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.button_press_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.button_press_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.button_press_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.button_press_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.button_press_event', ['widget', 'event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'button_press_event', localization, ['widget', 'event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'button_press_event(...)' code ##################

        
        # Assigning a Attribute to a Name (line 213):
        
        # Assigning a Attribute to a Name (line 213):
        # Getting the type of 'event' (line 213)
        event_226939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'event')
        # Obtaining the member 'x' of a type (line 213)
        x_226940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), event_226939, 'x')
        # Assigning a type to the variable 'x' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'x', x_226940)
        
        # Assigning a BinOp to a Name (line 215):
        
        # Assigning a BinOp to a Name (line 215):
        
        # Call to get_allocation(...): (line 215)
        # Processing the call keyword arguments (line 215)
        kwargs_226943 = {}
        # Getting the type of 'self' (line 215)
        self_226941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'self', False)
        # Obtaining the member 'get_allocation' of a type (line 215)
        get_allocation_226942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), self_226941, 'get_allocation')
        # Calling get_allocation(args, kwargs) (line 215)
        get_allocation_call_result_226944 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), get_allocation_226942, *[], **kwargs_226943)
        
        # Obtaining the member 'height' of a type (line 215)
        height_226945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), get_allocation_call_result_226944, 'height')
        # Getting the type of 'event' (line 215)
        event_226946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 43), 'event')
        # Obtaining the member 'y' of a type (line 215)
        y_226947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 43), event_226946, 'y')
        # Applying the binary operator '-' (line 215)
        result_sub_226948 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 12), '-', height_226945, y_226947)
        
        # Assigning a type to the variable 'y' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'y', result_sub_226948)
        
        # Call to button_press_event(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'self' (line 216)
        self_226951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 44), 'self', False)
        # Getting the type of 'x' (line 216)
        x_226952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 50), 'x', False)
        # Getting the type of 'y' (line 216)
        y_226953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 53), 'y', False)
        # Getting the type of 'event' (line 216)
        event_226954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 56), 'event', False)
        # Obtaining the member 'button' of a type (line 216)
        button_226955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 56), event_226954, 'button')
        # Processing the call keyword arguments (line 216)
        # Getting the type of 'event' (line 216)
        event_226956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 79), 'event', False)
        keyword_226957 = event_226956
        kwargs_226958 = {'guiEvent': keyword_226957}
        # Getting the type of 'FigureCanvasBase' (line 216)
        FigureCanvasBase_226949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'button_press_event' of a type (line 216)
        button_press_event_226950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), FigureCanvasBase_226949, 'button_press_event')
        # Calling button_press_event(args, kwargs) (line 216)
        button_press_event_call_result_226959 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), button_press_event_226950, *[self_226951, x_226952, y_226953, button_226955], **kwargs_226958)
        
        # Getting the type of 'False' (line 217)
        False_226960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'stypy_return_type', False_226960)
        
        # ################# End of 'button_press_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'button_press_event' in the type store
        # Getting the type of 'stypy_return_type' (line 212)
        stypy_return_type_226961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_226961)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'button_press_event'
        return stypy_return_type_226961


    @norecursion
    def button_release_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'button_release_event'
        module_type_store = module_type_store.open_function_context('button_release_event', 219, 4, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.button_release_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.button_release_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.button_release_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.button_release_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.button_release_event')
        FigureCanvasGTK3.button_release_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'event'])
        FigureCanvasGTK3.button_release_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.button_release_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.button_release_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.button_release_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.button_release_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.button_release_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.button_release_event', ['widget', 'event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'button_release_event', localization, ['widget', 'event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'button_release_event(...)' code ##################

        
        # Assigning a Attribute to a Name (line 220):
        
        # Assigning a Attribute to a Name (line 220):
        # Getting the type of 'event' (line 220)
        event_226962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'event')
        # Obtaining the member 'x' of a type (line 220)
        x_226963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), event_226962, 'x')
        # Assigning a type to the variable 'x' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'x', x_226963)
        
        # Assigning a BinOp to a Name (line 222):
        
        # Assigning a BinOp to a Name (line 222):
        
        # Call to get_allocation(...): (line 222)
        # Processing the call keyword arguments (line 222)
        kwargs_226966 = {}
        # Getting the type of 'self' (line 222)
        self_226964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'self', False)
        # Obtaining the member 'get_allocation' of a type (line 222)
        get_allocation_226965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), self_226964, 'get_allocation')
        # Calling get_allocation(args, kwargs) (line 222)
        get_allocation_call_result_226967 = invoke(stypy.reporting.localization.Localization(__file__, 222, 12), get_allocation_226965, *[], **kwargs_226966)
        
        # Obtaining the member 'height' of a type (line 222)
        height_226968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), get_allocation_call_result_226967, 'height')
        # Getting the type of 'event' (line 222)
        event_226969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 43), 'event')
        # Obtaining the member 'y' of a type (line 222)
        y_226970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 43), event_226969, 'y')
        # Applying the binary operator '-' (line 222)
        result_sub_226971 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 12), '-', height_226968, y_226970)
        
        # Assigning a type to the variable 'y' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'y', result_sub_226971)
        
        # Call to button_release_event(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'self' (line 223)
        self_226974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 46), 'self', False)
        # Getting the type of 'x' (line 223)
        x_226975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 52), 'x', False)
        # Getting the type of 'y' (line 223)
        y_226976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 55), 'y', False)
        # Getting the type of 'event' (line 223)
        event_226977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 58), 'event', False)
        # Obtaining the member 'button' of a type (line 223)
        button_226978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 58), event_226977, 'button')
        # Processing the call keyword arguments (line 223)
        # Getting the type of 'event' (line 223)
        event_226979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 81), 'event', False)
        keyword_226980 = event_226979
        kwargs_226981 = {'guiEvent': keyword_226980}
        # Getting the type of 'FigureCanvasBase' (line 223)
        FigureCanvasBase_226972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'button_release_event' of a type (line 223)
        button_release_event_226973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), FigureCanvasBase_226972, 'button_release_event')
        # Calling button_release_event(args, kwargs) (line 223)
        button_release_event_call_result_226982 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), button_release_event_226973, *[self_226974, x_226975, y_226976, button_226978], **kwargs_226981)
        
        # Getting the type of 'False' (line 224)
        False_226983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'stypy_return_type', False_226983)
        
        # ################# End of 'button_release_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'button_release_event' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_226984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_226984)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'button_release_event'
        return stypy_return_type_226984


    @norecursion
    def key_press_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'key_press_event'
        module_type_store = module_type_store.open_function_context('key_press_event', 226, 4, False)
        # Assigning a type to the variable 'self' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.key_press_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.key_press_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.key_press_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.key_press_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.key_press_event')
        FigureCanvasGTK3.key_press_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'event'])
        FigureCanvasGTK3.key_press_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.key_press_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.key_press_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.key_press_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.key_press_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.key_press_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.key_press_event', ['widget', 'event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'key_press_event', localization, ['widget', 'event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'key_press_event(...)' code ##################

        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to _get_key(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'event' (line 227)
        event_226987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 28), 'event', False)
        # Processing the call keyword arguments (line 227)
        kwargs_226988 = {}
        # Getting the type of 'self' (line 227)
        self_226985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 14), 'self', False)
        # Obtaining the member '_get_key' of a type (line 227)
        _get_key_226986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 14), self_226985, '_get_key')
        # Calling _get_key(args, kwargs) (line 227)
        _get_key_call_result_226989 = invoke(stypy.reporting.localization.Localization(__file__, 227, 14), _get_key_226986, *[event_226987], **kwargs_226988)
        
        # Assigning a type to the variable 'key' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'key', _get_key_call_result_226989)
        
        # Call to key_press_event(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'self' (line 228)
        self_226992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 41), 'self', False)
        # Getting the type of 'key' (line 228)
        key_226993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 47), 'key', False)
        # Processing the call keyword arguments (line 228)
        # Getting the type of 'event' (line 228)
        event_226994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 61), 'event', False)
        keyword_226995 = event_226994
        kwargs_226996 = {'guiEvent': keyword_226995}
        # Getting the type of 'FigureCanvasBase' (line 228)
        FigureCanvasBase_226990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'key_press_event' of a type (line 228)
        key_press_event_226991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), FigureCanvasBase_226990, 'key_press_event')
        # Calling key_press_event(args, kwargs) (line 228)
        key_press_event_call_result_226997 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), key_press_event_226991, *[self_226992, key_226993], **kwargs_226996)
        
        # Getting the type of 'True' (line 229)
        True_226998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', True_226998)
        
        # ################# End of 'key_press_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'key_press_event' in the type store
        # Getting the type of 'stypy_return_type' (line 226)
        stypy_return_type_226999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_226999)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'key_press_event'
        return stypy_return_type_226999


    @norecursion
    def key_release_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'key_release_event'
        module_type_store = module_type_store.open_function_context('key_release_event', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.key_release_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.key_release_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.key_release_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.key_release_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.key_release_event')
        FigureCanvasGTK3.key_release_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'event'])
        FigureCanvasGTK3.key_release_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.key_release_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.key_release_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.key_release_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.key_release_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.key_release_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.key_release_event', ['widget', 'event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'key_release_event', localization, ['widget', 'event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'key_release_event(...)' code ##################

        
        # Assigning a Call to a Name (line 232):
        
        # Assigning a Call to a Name (line 232):
        
        # Call to _get_key(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'event' (line 232)
        event_227002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'event', False)
        # Processing the call keyword arguments (line 232)
        kwargs_227003 = {}
        # Getting the type of 'self' (line 232)
        self_227000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 14), 'self', False)
        # Obtaining the member '_get_key' of a type (line 232)
        _get_key_227001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 14), self_227000, '_get_key')
        # Calling _get_key(args, kwargs) (line 232)
        _get_key_call_result_227004 = invoke(stypy.reporting.localization.Localization(__file__, 232, 14), _get_key_227001, *[event_227002], **kwargs_227003)
        
        # Assigning a type to the variable 'key' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'key', _get_key_call_result_227004)
        
        # Call to key_release_event(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'self' (line 233)
        self_227007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 43), 'self', False)
        # Getting the type of 'key' (line 233)
        key_227008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 49), 'key', False)
        # Processing the call keyword arguments (line 233)
        # Getting the type of 'event' (line 233)
        event_227009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 63), 'event', False)
        keyword_227010 = event_227009
        kwargs_227011 = {'guiEvent': keyword_227010}
        # Getting the type of 'FigureCanvasBase' (line 233)
        FigureCanvasBase_227005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'key_release_event' of a type (line 233)
        key_release_event_227006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), FigureCanvasBase_227005, 'key_release_event')
        # Calling key_release_event(args, kwargs) (line 233)
        key_release_event_call_result_227012 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), key_release_event_227006, *[self_227007, key_227008], **kwargs_227011)
        
        # Getting the type of 'True' (line 234)
        True_227013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'stypy_return_type', True_227013)
        
        # ################# End of 'key_release_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'key_release_event' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_227014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227014)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'key_release_event'
        return stypy_return_type_227014


    @norecursion
    def motion_notify_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'motion_notify_event'
        module_type_store = module_type_store.open_function_context('motion_notify_event', 236, 4, False)
        # Assigning a type to the variable 'self' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.motion_notify_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.motion_notify_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.motion_notify_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.motion_notify_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.motion_notify_event')
        FigureCanvasGTK3.motion_notify_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'event'])
        FigureCanvasGTK3.motion_notify_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.motion_notify_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.motion_notify_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.motion_notify_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.motion_notify_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.motion_notify_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.motion_notify_event', ['widget', 'event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'motion_notify_event', localization, ['widget', 'event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'motion_notify_event(...)' code ##################

        
        # Getting the type of 'event' (line 237)
        event_227015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'event')
        # Obtaining the member 'is_hint' of a type (line 237)
        is_hint_227016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 11), event_227015, 'is_hint')
        # Testing the type of an if condition (line 237)
        if_condition_227017 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 8), is_hint_227016)
        # Assigning a type to the variable 'if_condition_227017' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'if_condition_227017', if_condition_227017)
        # SSA begins for if statement (line 237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 238):
        
        # Assigning a Call to a Name:
        
        # Call to get_pointer(...): (line 238)
        # Processing the call keyword arguments (line 238)
        kwargs_227021 = {}
        # Getting the type of 'event' (line 238)
        event_227018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 29), 'event', False)
        # Obtaining the member 'window' of a type (line 238)
        window_227019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 29), event_227018, 'window')
        # Obtaining the member 'get_pointer' of a type (line 238)
        get_pointer_227020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 29), window_227019, 'get_pointer')
        # Calling get_pointer(args, kwargs) (line 238)
        get_pointer_call_result_227022 = invoke(stypy.reporting.localization.Localization(__file__, 238, 29), get_pointer_227020, *[], **kwargs_227021)
        
        # Assigning a type to the variable 'call_assignment_226565' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226565', get_pointer_call_result_227022)
        
        # Assigning a Call to a Name (line 238):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_227025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 12), 'int')
        # Processing the call keyword arguments
        kwargs_227026 = {}
        # Getting the type of 'call_assignment_226565' (line 238)
        call_assignment_226565_227023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226565', False)
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___227024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), call_assignment_226565_227023, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_227027 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___227024, *[int_227025], **kwargs_227026)
        
        # Assigning a type to the variable 'call_assignment_226566' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226566', getitem___call_result_227027)
        
        # Assigning a Name to a Name (line 238):
        # Getting the type of 'call_assignment_226566' (line 238)
        call_assignment_226566_227028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226566')
        # Assigning a type to the variable 't' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 't', call_assignment_226566_227028)
        
        # Assigning a Call to a Name (line 238):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_227031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 12), 'int')
        # Processing the call keyword arguments
        kwargs_227032 = {}
        # Getting the type of 'call_assignment_226565' (line 238)
        call_assignment_226565_227029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226565', False)
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___227030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), call_assignment_226565_227029, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_227033 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___227030, *[int_227031], **kwargs_227032)
        
        # Assigning a type to the variable 'call_assignment_226567' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226567', getitem___call_result_227033)
        
        # Assigning a Name to a Name (line 238):
        # Getting the type of 'call_assignment_226567' (line 238)
        call_assignment_226567_227034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226567')
        # Assigning a type to the variable 'x' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'x', call_assignment_226567_227034)
        
        # Assigning a Call to a Name (line 238):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_227037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 12), 'int')
        # Processing the call keyword arguments
        kwargs_227038 = {}
        # Getting the type of 'call_assignment_226565' (line 238)
        call_assignment_226565_227035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226565', False)
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___227036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), call_assignment_226565_227035, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_227039 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___227036, *[int_227037], **kwargs_227038)
        
        # Assigning a type to the variable 'call_assignment_226568' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226568', getitem___call_result_227039)
        
        # Assigning a Name to a Name (line 238):
        # Getting the type of 'call_assignment_226568' (line 238)
        call_assignment_226568_227040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226568')
        # Assigning a type to the variable 'y' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 18), 'y', call_assignment_226568_227040)
        
        # Assigning a Call to a Name (line 238):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_227043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 12), 'int')
        # Processing the call keyword arguments
        kwargs_227044 = {}
        # Getting the type of 'call_assignment_226565' (line 238)
        call_assignment_226565_227041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226565', False)
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___227042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), call_assignment_226565_227041, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_227045 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___227042, *[int_227043], **kwargs_227044)
        
        # Assigning a type to the variable 'call_assignment_226569' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226569', getitem___call_result_227045)
        
        # Assigning a Name to a Name (line 238):
        # Getting the type of 'call_assignment_226569' (line 238)
        call_assignment_226569_227046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'call_assignment_226569')
        # Assigning a type to the variable 'state' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 21), 'state', call_assignment_226569_227046)
        # SSA branch for the else part of an if statement (line 237)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Tuple to a Tuple (line 240):
        
        # Assigning a Attribute to a Name (line 240):
        # Getting the type of 'event' (line 240)
        event_227047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 26), 'event')
        # Obtaining the member 'x' of a type (line 240)
        x_227048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 26), event_227047, 'x')
        # Assigning a type to the variable 'tuple_assignment_226570' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'tuple_assignment_226570', x_227048)
        
        # Assigning a Attribute to a Name (line 240):
        # Getting the type of 'event' (line 240)
        event_227049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 35), 'event')
        # Obtaining the member 'y' of a type (line 240)
        y_227050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 35), event_227049, 'y')
        # Assigning a type to the variable 'tuple_assignment_226571' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'tuple_assignment_226571', y_227050)
        
        # Assigning a Call to a Name (line 240):
        
        # Call to get_state(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_227053 = {}
        # Getting the type of 'event' (line 240)
        event_227051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 'event', False)
        # Obtaining the member 'get_state' of a type (line 240)
        get_state_227052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 44), event_227051, 'get_state')
        # Calling get_state(args, kwargs) (line 240)
        get_state_call_result_227054 = invoke(stypy.reporting.localization.Localization(__file__, 240, 44), get_state_227052, *[], **kwargs_227053)
        
        # Assigning a type to the variable 'tuple_assignment_226572' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'tuple_assignment_226572', get_state_call_result_227054)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'tuple_assignment_226570' (line 240)
        tuple_assignment_226570_227055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'tuple_assignment_226570')
        # Assigning a type to the variable 'x' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'x', tuple_assignment_226570_227055)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'tuple_assignment_226571' (line 240)
        tuple_assignment_226571_227056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'tuple_assignment_226571')
        # Assigning a type to the variable 'y' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'y', tuple_assignment_226571_227056)
        
        # Assigning a Name to a Name (line 240):
        # Getting the type of 'tuple_assignment_226572' (line 240)
        tuple_assignment_226572_227057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'tuple_assignment_226572')
        # Assigning a type to the variable 'state' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'state', tuple_assignment_226572_227057)
        # SSA join for if statement (line 237)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 243):
        
        # Assigning a BinOp to a Name (line 243):
        
        # Call to get_allocation(...): (line 243)
        # Processing the call keyword arguments (line 243)
        kwargs_227060 = {}
        # Getting the type of 'self' (line 243)
        self_227058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'self', False)
        # Obtaining the member 'get_allocation' of a type (line 243)
        get_allocation_227059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), self_227058, 'get_allocation')
        # Calling get_allocation(args, kwargs) (line 243)
        get_allocation_call_result_227061 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), get_allocation_227059, *[], **kwargs_227060)
        
        # Obtaining the member 'height' of a type (line 243)
        height_227062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), get_allocation_call_result_227061, 'height')
        # Getting the type of 'y' (line 243)
        y_227063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 43), 'y')
        # Applying the binary operator '-' (line 243)
        result_sub_227064 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 12), '-', height_227062, y_227063)
        
        # Assigning a type to the variable 'y' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'y', result_sub_227064)
        
        # Call to motion_notify_event(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'self' (line 244)
        self_227067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 45), 'self', False)
        # Getting the type of 'x' (line 244)
        x_227068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 51), 'x', False)
        # Getting the type of 'y' (line 244)
        y_227069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 54), 'y', False)
        # Processing the call keyword arguments (line 244)
        # Getting the type of 'event' (line 244)
        event_227070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 66), 'event', False)
        keyword_227071 = event_227070
        kwargs_227072 = {'guiEvent': keyword_227071}
        # Getting the type of 'FigureCanvasBase' (line 244)
        FigureCanvasBase_227065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'motion_notify_event' of a type (line 244)
        motion_notify_event_227066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), FigureCanvasBase_227065, 'motion_notify_event')
        # Calling motion_notify_event(args, kwargs) (line 244)
        motion_notify_event_call_result_227073 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), motion_notify_event_227066, *[self_227067, x_227068, y_227069], **kwargs_227072)
        
        # Getting the type of 'False' (line 245)
        False_227074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'stypy_return_type', False_227074)
        
        # ################# End of 'motion_notify_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'motion_notify_event' in the type store
        # Getting the type of 'stypy_return_type' (line 236)
        stypy_return_type_227075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227075)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'motion_notify_event'
        return stypy_return_type_227075


    @norecursion
    def leave_notify_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'leave_notify_event'
        module_type_store = module_type_store.open_function_context('leave_notify_event', 247, 4, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.leave_notify_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.leave_notify_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.leave_notify_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.leave_notify_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.leave_notify_event')
        FigureCanvasGTK3.leave_notify_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'event'])
        FigureCanvasGTK3.leave_notify_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.leave_notify_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.leave_notify_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.leave_notify_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.leave_notify_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.leave_notify_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.leave_notify_event', ['widget', 'event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'leave_notify_event', localization, ['widget', 'event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'leave_notify_event(...)' code ##################

        
        # Call to leave_notify_event(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'self' (line 248)
        self_227078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 44), 'self', False)
        # Getting the type of 'event' (line 248)
        event_227079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 50), 'event', False)
        # Processing the call keyword arguments (line 248)
        kwargs_227080 = {}
        # Getting the type of 'FigureCanvasBase' (line 248)
        FigureCanvasBase_227076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'leave_notify_event' of a type (line 248)
        leave_notify_event_227077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), FigureCanvasBase_227076, 'leave_notify_event')
        # Calling leave_notify_event(args, kwargs) (line 248)
        leave_notify_event_call_result_227081 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), leave_notify_event_227077, *[self_227078, event_227079], **kwargs_227080)
        
        
        # ################# End of 'leave_notify_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'leave_notify_event' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_227082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227082)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'leave_notify_event'
        return stypy_return_type_227082


    @norecursion
    def enter_notify_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'enter_notify_event'
        module_type_store = module_type_store.open_function_context('enter_notify_event', 250, 4, False)
        # Assigning a type to the variable 'self' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.enter_notify_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.enter_notify_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.enter_notify_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.enter_notify_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.enter_notify_event')
        FigureCanvasGTK3.enter_notify_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'event'])
        FigureCanvasGTK3.enter_notify_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.enter_notify_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.enter_notify_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.enter_notify_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.enter_notify_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.enter_notify_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.enter_notify_event', ['widget', 'event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'enter_notify_event', localization, ['widget', 'event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'enter_notify_event(...)' code ##################

        
        # Call to enter_notify_event(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'self' (line 251)
        self_227085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 44), 'self', False)
        # Getting the type of 'event' (line 251)
        event_227086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 50), 'event', False)
        # Processing the call keyword arguments (line 251)
        kwargs_227087 = {}
        # Getting the type of 'FigureCanvasBase' (line 251)
        FigureCanvasBase_227083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'enter_notify_event' of a type (line 251)
        enter_notify_event_227084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), FigureCanvasBase_227083, 'enter_notify_event')
        # Calling enter_notify_event(args, kwargs) (line 251)
        enter_notify_event_call_result_227088 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), enter_notify_event_227084, *[self_227085, event_227086], **kwargs_227087)
        
        
        # ################# End of 'enter_notify_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'enter_notify_event' in the type store
        # Getting the type of 'stypy_return_type' (line 250)
        stypy_return_type_227089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227089)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'enter_notify_event'
        return stypy_return_type_227089


    @norecursion
    def size_allocate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'size_allocate'
        module_type_store = module_type_store.open_function_context('size_allocate', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.size_allocate.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.size_allocate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.size_allocate.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.size_allocate.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.size_allocate')
        FigureCanvasGTK3.size_allocate.__dict__.__setitem__('stypy_param_names_list', ['widget', 'allocation'])
        FigureCanvasGTK3.size_allocate.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.size_allocate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.size_allocate.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.size_allocate.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.size_allocate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.size_allocate.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.size_allocate', ['widget', 'allocation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'size_allocate', localization, ['widget', 'allocation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'size_allocate(...)' code ##################

        
        # Assigning a Attribute to a Name (line 254):
        
        # Assigning a Attribute to a Name (line 254):
        # Getting the type of 'self' (line 254)
        self_227090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'self')
        # Obtaining the member 'figure' of a type (line 254)
        figure_227091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 17), self_227090, 'figure')
        # Obtaining the member 'dpi' of a type (line 254)
        dpi_227092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 17), figure_227091, 'dpi')
        # Assigning a type to the variable 'dpival' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'dpival', dpi_227092)
        
        # Assigning a BinOp to a Name (line 255):
        
        # Assigning a BinOp to a Name (line 255):
        # Getting the type of 'allocation' (line 255)
        allocation_227093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'allocation')
        # Obtaining the member 'width' of a type (line 255)
        width_227094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 16), allocation_227093, 'width')
        # Getting the type of 'dpival' (line 255)
        dpival_227095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 35), 'dpival')
        # Applying the binary operator 'div' (line 255)
        result_div_227096 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 16), 'div', width_227094, dpival_227095)
        
        # Assigning a type to the variable 'winch' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'winch', result_div_227096)
        
        # Assigning a BinOp to a Name (line 256):
        
        # Assigning a BinOp to a Name (line 256):
        # Getting the type of 'allocation' (line 256)
        allocation_227097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'allocation')
        # Obtaining the member 'height' of a type (line 256)
        height_227098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 16), allocation_227097, 'height')
        # Getting the type of 'dpival' (line 256)
        dpival_227099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 36), 'dpival')
        # Applying the binary operator 'div' (line 256)
        result_div_227100 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 16), 'div', height_227098, dpival_227099)
        
        # Assigning a type to the variable 'hinch' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'hinch', result_div_227100)
        
        # Call to set_size_inches(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'winch' (line 257)
        winch_227104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 36), 'winch', False)
        # Getting the type of 'hinch' (line 257)
        hinch_227105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 43), 'hinch', False)
        # Processing the call keyword arguments (line 257)
        # Getting the type of 'False' (line 257)
        False_227106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 58), 'False', False)
        keyword_227107 = False_227106
        kwargs_227108 = {'forward': keyword_227107}
        # Getting the type of 'self' (line 257)
        self_227101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 257)
        figure_227102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_227101, 'figure')
        # Obtaining the member 'set_size_inches' of a type (line 257)
        set_size_inches_227103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), figure_227102, 'set_size_inches')
        # Calling set_size_inches(args, kwargs) (line 257)
        set_size_inches_call_result_227109 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), set_size_inches_227103, *[winch_227104, hinch_227105], **kwargs_227108)
        
        
        # Call to resize_event(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'self' (line 258)
        self_227112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 38), 'self', False)
        # Processing the call keyword arguments (line 258)
        kwargs_227113 = {}
        # Getting the type of 'FigureCanvasBase' (line 258)
        FigureCanvasBase_227110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'resize_event' of a type (line 258)
        resize_event_227111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), FigureCanvasBase_227110, 'resize_event')
        # Calling resize_event(args, kwargs) (line 258)
        resize_event_call_result_227114 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), resize_event_227111, *[self_227112], **kwargs_227113)
        
        
        # Call to draw_idle(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_227117 = {}
        # Getting the type of 'self' (line 259)
        self_227115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self', False)
        # Obtaining the member 'draw_idle' of a type (line 259)
        draw_idle_227116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_227115, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 259)
        draw_idle_call_result_227118 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), draw_idle_227116, *[], **kwargs_227117)
        
        
        # ################# End of 'size_allocate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'size_allocate' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_227119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227119)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'size_allocate'
        return stypy_return_type_227119


    @norecursion
    def _get_key(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_key'
        module_type_store = module_type_store.open_function_context('_get_key', 261, 4, False)
        # Assigning a type to the variable 'self' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3._get_key.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3._get_key.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3._get_key.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3._get_key.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3._get_key')
        FigureCanvasGTK3._get_key.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasGTK3._get_key.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3._get_key.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3._get_key.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3._get_key.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3._get_key.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3._get_key.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3._get_key', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_key', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_key(...)' code ##################

        
        
        # Getting the type of 'event' (line 262)
        event_227120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 11), 'event')
        # Obtaining the member 'keyval' of a type (line 262)
        keyval_227121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 11), event_227120, 'keyval')
        # Getting the type of 'self' (line 262)
        self_227122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 27), 'self')
        # Obtaining the member 'keyvald' of a type (line 262)
        keyvald_227123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 27), self_227122, 'keyvald')
        # Applying the binary operator 'in' (line 262)
        result_contains_227124 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 11), 'in', keyval_227121, keyvald_227123)
        
        # Testing the type of an if condition (line 262)
        if_condition_227125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 8), result_contains_227124)
        # Assigning a type to the variable 'if_condition_227125' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'if_condition_227125', if_condition_227125)
        # SSA begins for if statement (line 262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        # Getting the type of 'event' (line 263)
        event_227126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 31), 'event')
        # Obtaining the member 'keyval' of a type (line 263)
        keyval_227127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 31), event_227126, 'keyval')
        # Getting the type of 'self' (line 263)
        self_227128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 18), 'self')
        # Obtaining the member 'keyvald' of a type (line 263)
        keyvald_227129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 18), self_227128, 'keyvald')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___227130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 18), keyvald_227129, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_227131 = invoke(stypy.reporting.localization.Localization(__file__, 263, 18), getitem___227130, keyval_227127)
        
        # Assigning a type to the variable 'key' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'key', subscript_call_result_227131)
        # SSA branch for the else part of an if statement (line 262)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'event' (line 264)
        event_227132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'event')
        # Obtaining the member 'keyval' of a type (line 264)
        keyval_227133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 13), event_227132, 'keyval')
        int_227134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 28), 'int')
        # Applying the binary operator '<' (line 264)
        result_lt_227135 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 13), '<', keyval_227133, int_227134)
        
        # Testing the type of an if condition (line 264)
        if_condition_227136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 13), result_lt_227135)
        # Assigning a type to the variable 'if_condition_227136' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'if_condition_227136', if_condition_227136)
        # SSA begins for if statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 265):
        
        # Assigning a Call to a Name (line 265):
        
        # Call to chr(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'event' (line 265)
        event_227138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 22), 'event', False)
        # Obtaining the member 'keyval' of a type (line 265)
        keyval_227139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 22), event_227138, 'keyval')
        # Processing the call keyword arguments (line 265)
        kwargs_227140 = {}
        # Getting the type of 'chr' (line 265)
        chr_227137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 18), 'chr', False)
        # Calling chr(args, kwargs) (line 265)
        chr_call_result_227141 = invoke(stypy.reporting.localization.Localization(__file__, 265, 18), chr_227137, *[keyval_227139], **kwargs_227140)
        
        # Assigning a type to the variable 'key' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'key', chr_call_result_227141)
        # SSA branch for the else part of an if statement (line 264)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 267):
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'None' (line 267)
        None_227142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 18), 'None')
        # Assigning a type to the variable 'key' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'key', None_227142)
        # SSA join for if statement (line 264)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 262)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 269):
        
        # Assigning a List to a Name (line 269):
        
        # Obtaining an instance of the builtin type 'list' (line 269)
        list_227143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 269)
        # Adding element type (line 269)
        
        # Obtaining an instance of the builtin type 'tuple' (line 270)
        tuple_227144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 270)
        # Adding element type (line 270)
        # Getting the type of 'Gdk' (line 270)
        Gdk_227145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 22), 'Gdk')
        # Obtaining the member 'ModifierType' of a type (line 270)
        ModifierType_227146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 22), Gdk_227145, 'ModifierType')
        # Obtaining the member 'MOD4_MASK' of a type (line 270)
        MOD4_MASK_227147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 22), ModifierType_227146, 'MOD4_MASK')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 22), tuple_227144, MOD4_MASK_227147)
        # Adding element type (line 270)
        unicode_227148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 50), 'unicode', u'super')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 22), tuple_227144, unicode_227148)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 20), list_227143, tuple_227144)
        # Adding element type (line 269)
        
        # Obtaining an instance of the builtin type 'tuple' (line 271)
        tuple_227149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 271)
        # Adding element type (line 271)
        # Getting the type of 'Gdk' (line 271)
        Gdk_227150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 22), 'Gdk')
        # Obtaining the member 'ModifierType' of a type (line 271)
        ModifierType_227151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 22), Gdk_227150, 'ModifierType')
        # Obtaining the member 'MOD1_MASK' of a type (line 271)
        MOD1_MASK_227152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 22), ModifierType_227151, 'MOD1_MASK')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 22), tuple_227149, MOD1_MASK_227152)
        # Adding element type (line 271)
        unicode_227153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 50), 'unicode', u'alt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 22), tuple_227149, unicode_227153)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 20), list_227143, tuple_227149)
        # Adding element type (line 269)
        
        # Obtaining an instance of the builtin type 'tuple' (line 272)
        tuple_227154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 272)
        # Adding element type (line 272)
        # Getting the type of 'Gdk' (line 272)
        Gdk_227155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'Gdk')
        # Obtaining the member 'ModifierType' of a type (line 272)
        ModifierType_227156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 22), Gdk_227155, 'ModifierType')
        # Obtaining the member 'CONTROL_MASK' of a type (line 272)
        CONTROL_MASK_227157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 22), ModifierType_227156, 'CONTROL_MASK')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 22), tuple_227154, CONTROL_MASK_227157)
        # Adding element type (line 272)
        unicode_227158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 53), 'unicode', u'ctrl')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 22), tuple_227154, unicode_227158)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 20), list_227143, tuple_227154)
        
        # Assigning a type to the variable 'modifiers' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'modifiers', list_227143)
        
        # Getting the type of 'modifiers' (line 274)
        modifiers_227159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 32), 'modifiers')
        # Testing the type of a for loop iterable (line 274)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 274, 8), modifiers_227159)
        # Getting the type of the for loop variable (line 274)
        for_loop_var_227160 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 274, 8), modifiers_227159)
        # Assigning a type to the variable 'key_mask' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'key_mask', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 8), for_loop_var_227160))
        # Assigning a type to the variable 'prefix' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'prefix', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 8), for_loop_var_227160))
        # SSA begins for a for statement (line 274)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'event' (line 275)
        event_227161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'event')
        # Obtaining the member 'state' of a type (line 275)
        state_227162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 15), event_227161, 'state')
        # Getting the type of 'key_mask' (line 275)
        key_mask_227163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 29), 'key_mask')
        # Applying the binary operator '&' (line 275)
        result_and__227164 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 15), '&', state_227162, key_mask_227163)
        
        # Testing the type of an if condition (line 275)
        if_condition_227165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 12), result_and__227164)
        # Assigning a type to the variable 'if_condition_227165' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'if_condition_227165', if_condition_227165)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to format(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'prefix' (line 276)
        prefix_227168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 39), 'prefix', False)
        # Getting the type of 'key' (line 276)
        key_227169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 47), 'key', False)
        # Processing the call keyword arguments (line 276)
        kwargs_227170 = {}
        unicode_227166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 22), 'unicode', u'{0}+{1}')
        # Obtaining the member 'format' of a type (line 276)
        format_227167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 22), unicode_227166, 'format')
        # Calling format(args, kwargs) (line 276)
        format_call_result_227171 = invoke(stypy.reporting.localization.Localization(__file__, 276, 22), format_227167, *[prefix_227168, key_227169], **kwargs_227170)
        
        # Assigning a type to the variable 'key' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'key', format_call_result_227171)
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'key' (line 278)
        key_227172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 15), 'key')
        # Assigning a type to the variable 'stypy_return_type' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'stypy_return_type', key_227172)
        
        # ################# End of '_get_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_key' in the type store
        # Getting the type of 'stypy_return_type' (line 261)
        stypy_return_type_227173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_key'
        return stypy_return_type_227173


    @norecursion
    def configure_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'configure_event'
        module_type_store = module_type_store.open_function_context('configure_event', 280, 4, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.configure_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.configure_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.configure_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.configure_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.configure_event')
        FigureCanvasGTK3.configure_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'event'])
        FigureCanvasGTK3.configure_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.configure_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.configure_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.configure_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.configure_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.configure_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.configure_event', ['widget', 'event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'configure_event', localization, ['widget', 'event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'configure_event(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 281)
        
        # Call to get_property(...): (line 281)
        # Processing the call arguments (line 281)
        unicode_227176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 31), 'unicode', u'window')
        # Processing the call keyword arguments (line 281)
        kwargs_227177 = {}
        # Getting the type of 'widget' (line 281)
        widget_227174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 11), 'widget', False)
        # Obtaining the member 'get_property' of a type (line 281)
        get_property_227175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 11), widget_227174, 'get_property')
        # Calling get_property(args, kwargs) (line 281)
        get_property_call_result_227178 = invoke(stypy.reporting.localization.Localization(__file__, 281, 11), get_property_227175, *[unicode_227176], **kwargs_227177)
        
        # Getting the type of 'None' (line 281)
        None_227179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 44), 'None')
        
        (may_be_227180, more_types_in_union_227181) = may_be_none(get_property_call_result_227178, None_227179)

        if may_be_227180:

            if more_types_in_union_227181:
                # Runtime conditional SSA (line 281)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_227181:
                # SSA join for if statement (line 281)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Tuple to a Tuple (line 283):
        
        # Assigning a Attribute to a Name (line 283):
        # Getting the type of 'event' (line 283)
        event_227182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'event')
        # Obtaining the member 'width' of a type (line 283)
        width_227183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 15), event_227182, 'width')
        # Assigning a type to the variable 'tuple_assignment_226573' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'tuple_assignment_226573', width_227183)
        
        # Assigning a Attribute to a Name (line 283):
        # Getting the type of 'event' (line 283)
        event_227184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 28), 'event')
        # Obtaining the member 'height' of a type (line 283)
        height_227185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 28), event_227184, 'height')
        # Assigning a type to the variable 'tuple_assignment_226574' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'tuple_assignment_226574', height_227185)
        
        # Assigning a Name to a Name (line 283):
        # Getting the type of 'tuple_assignment_226573' (line 283)
        tuple_assignment_226573_227186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'tuple_assignment_226573')
        # Assigning a type to the variable 'w' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'w', tuple_assignment_226573_227186)
        
        # Assigning a Name to a Name (line 283):
        # Getting the type of 'tuple_assignment_226574' (line 283)
        tuple_assignment_226574_227187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'tuple_assignment_226574')
        # Assigning a type to the variable 'h' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'h', tuple_assignment_226574_227187)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'w' (line 284)
        w_227188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'w')
        int_227189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 15), 'int')
        # Applying the binary operator '<' (line 284)
        result_lt_227190 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), '<', w_227188, int_227189)
        
        
        # Getting the type of 'h' (line 284)
        h_227191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'h')
        int_227192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 24), 'int')
        # Applying the binary operator '<' (line 284)
        result_lt_227193 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 20), '<', h_227191, int_227192)
        
        # Applying the binary operator 'or' (line 284)
        result_or_keyword_227194 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), 'or', result_lt_227190, result_lt_227193)
        
        # Testing the type of an if condition (line 284)
        if_condition_227195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 8), result_or_keyword_227194)
        # Assigning a type to the variable 'if_condition_227195' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'if_condition_227195', if_condition_227195)
        # SSA begins for if statement (line 284)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 284)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 287):
        
        # Assigning a Attribute to a Name (line 287):
        # Getting the type of 'self' (line 287)
        self_227196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 14), 'self')
        # Obtaining the member 'figure' of a type (line 287)
        figure_227197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 14), self_227196, 'figure')
        # Obtaining the member 'dpi' of a type (line 287)
        dpi_227198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 14), figure_227197, 'dpi')
        # Assigning a type to the variable 'dpi' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'dpi', dpi_227198)
        
        # Call to set_size_inches(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'w' (line 288)
        w_227202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 36), 'w', False)
        # Getting the type of 'dpi' (line 288)
        dpi_227203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 38), 'dpi', False)
        # Applying the binary operator 'div' (line 288)
        result_div_227204 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 36), 'div', w_227202, dpi_227203)
        
        # Getting the type of 'h' (line 288)
        h_227205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 43), 'h', False)
        # Getting the type of 'dpi' (line 288)
        dpi_227206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 45), 'dpi', False)
        # Applying the binary operator 'div' (line 288)
        result_div_227207 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 43), 'div', h_227205, dpi_227206)
        
        # Processing the call keyword arguments (line 288)
        # Getting the type of 'False' (line 288)
        False_227208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 58), 'False', False)
        keyword_227209 = False_227208
        kwargs_227210 = {'forward': keyword_227209}
        # Getting the type of 'self' (line 288)
        self_227199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 288)
        figure_227200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), self_227199, 'figure')
        # Obtaining the member 'set_size_inches' of a type (line 288)
        set_size_inches_227201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), figure_227200, 'set_size_inches')
        # Calling set_size_inches(args, kwargs) (line 288)
        set_size_inches_call_result_227211 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), set_size_inches_227201, *[result_div_227204, result_div_227207], **kwargs_227210)
        
        # Getting the type of 'False' (line 289)
        False_227212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'stypy_return_type', False_227212)
        
        # ################# End of 'configure_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'configure_event' in the type store
        # Getting the type of 'stypy_return_type' (line 280)
        stypy_return_type_227213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227213)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'configure_event'
        return stypy_return_type_227213


    @norecursion
    def on_draw_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'on_draw_event'
        module_type_store = module_type_store.open_function_context('on_draw_event', 291, 4, False)
        # Assigning a type to the variable 'self' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.on_draw_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.on_draw_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.on_draw_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.on_draw_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.on_draw_event')
        FigureCanvasGTK3.on_draw_event.__dict__.__setitem__('stypy_param_names_list', ['widget', 'ctx'])
        FigureCanvasGTK3.on_draw_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.on_draw_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.on_draw_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.on_draw_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.on_draw_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.on_draw_event.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.on_draw_event', ['widget', 'ctx'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'on_draw_event', localization, ['widget', 'ctx'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'on_draw_event(...)' code ##################

        pass
        
        # ################# End of 'on_draw_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'on_draw_event' in the type store
        # Getting the type of 'stypy_return_type' (line 291)
        stypy_return_type_227214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227214)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'on_draw_event'
        return stypy_return_type_227214


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 295, 4, False)
        # Assigning a type to the variable 'self' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.draw.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.draw.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.draw')
        FigureCanvasGTK3.draw.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasGTK3.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.draw', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Call to get_visible(...): (line 296)
        # Processing the call keyword arguments (line 296)
        kwargs_227217 = {}
        # Getting the type of 'self' (line 296)
        self_227215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 11), 'self', False)
        # Obtaining the member 'get_visible' of a type (line 296)
        get_visible_227216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 11), self_227215, 'get_visible')
        # Calling get_visible(args, kwargs) (line 296)
        get_visible_call_result_227218 = invoke(stypy.reporting.localization.Localization(__file__, 296, 11), get_visible_227216, *[], **kwargs_227217)
        
        
        # Call to get_mapped(...): (line 296)
        # Processing the call keyword arguments (line 296)
        kwargs_227221 = {}
        # Getting the type of 'self' (line 296)
        self_227219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 34), 'self', False)
        # Obtaining the member 'get_mapped' of a type (line 296)
        get_mapped_227220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 34), self_227219, 'get_mapped')
        # Calling get_mapped(args, kwargs) (line 296)
        get_mapped_call_result_227222 = invoke(stypy.reporting.localization.Localization(__file__, 296, 34), get_mapped_227220, *[], **kwargs_227221)
        
        # Applying the binary operator 'and' (line 296)
        result_and_keyword_227223 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 11), 'and', get_visible_call_result_227218, get_mapped_call_result_227222)
        
        # Testing the type of an if condition (line 296)
        if_condition_227224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 8), result_and_keyword_227223)
        # Assigning a type to the variable 'if_condition_227224' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'if_condition_227224', if_condition_227224)
        # SSA begins for if statement (line 296)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to queue_draw(...): (line 297)
        # Processing the call keyword arguments (line 297)
        kwargs_227227 = {}
        # Getting the type of 'self' (line 297)
        self_227225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'self', False)
        # Obtaining the member 'queue_draw' of a type (line 297)
        queue_draw_227226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 12), self_227225, 'queue_draw')
        # Calling queue_draw(args, kwargs) (line 297)
        queue_draw_call_result_227228 = invoke(stypy.reporting.localization.Localization(__file__, 297, 12), queue_draw_227226, *[], **kwargs_227227)
        
        
        # Call to process_updates(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'False' (line 300)
        False_227235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 57), 'False', False)
        # Processing the call keyword arguments (line 300)
        kwargs_227236 = {}
        
        # Call to get_property(...): (line 300)
        # Processing the call arguments (line 300)
        unicode_227231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 30), 'unicode', u'window')
        # Processing the call keyword arguments (line 300)
        kwargs_227232 = {}
        # Getting the type of 'self' (line 300)
        self_227229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'self', False)
        # Obtaining the member 'get_property' of a type (line 300)
        get_property_227230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), self_227229, 'get_property')
        # Calling get_property(args, kwargs) (line 300)
        get_property_call_result_227233 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), get_property_227230, *[unicode_227231], **kwargs_227232)
        
        # Obtaining the member 'process_updates' of a type (line 300)
        process_updates_227234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), get_property_call_result_227233, 'process_updates')
        # Calling process_updates(args, kwargs) (line 300)
        process_updates_call_result_227237 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), process_updates_227234, *[False_227235], **kwargs_227236)
        
        # SSA join for if statement (line 296)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 295)
        stypy_return_type_227238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227238)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_227238


    @norecursion
    def draw_idle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_idle'
        module_type_store = module_type_store.open_function_context('draw_idle', 302, 4, False)
        # Assigning a type to the variable 'self' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.draw_idle.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.draw_idle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.draw_idle.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.draw_idle.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.draw_idle')
        FigureCanvasGTK3.draw_idle.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasGTK3.draw_idle.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.draw_idle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.draw_idle.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.draw_idle.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.draw_idle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.draw_idle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.draw_idle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_idle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_idle(...)' code ##################

        
        
        # Getting the type of 'self' (line 303)
        self_227239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'self')
        # Obtaining the member '_idle_draw_id' of a type (line 303)
        _idle_draw_id_227240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 11), self_227239, '_idle_draw_id')
        int_227241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 33), 'int')
        # Applying the binary operator '!=' (line 303)
        result_ne_227242 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 11), '!=', _idle_draw_id_227240, int_227241)
        
        # Testing the type of an if condition (line 303)
        if_condition_227243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 303, 8), result_ne_227242)
        # Assigning a type to the variable 'if_condition_227243' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'if_condition_227243', if_condition_227243)
        # SSA begins for if statement (line 303)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 303)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def idle_draw(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'idle_draw'
            module_type_store = module_type_store.open_function_context('idle_draw', 305, 8, False)
            
            # Passed parameters checking function
            idle_draw.stypy_localization = localization
            idle_draw.stypy_type_of_self = None
            idle_draw.stypy_type_store = module_type_store
            idle_draw.stypy_function_name = 'idle_draw'
            idle_draw.stypy_param_names_list = []
            idle_draw.stypy_varargs_param_name = 'args'
            idle_draw.stypy_kwargs_param_name = None
            idle_draw.stypy_call_defaults = defaults
            idle_draw.stypy_call_varargs = varargs
            idle_draw.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'idle_draw', [], 'args', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'idle_draw', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'idle_draw(...)' code ##################

            
            # Try-finally block (line 306)
            
            # Call to draw(...): (line 307)
            # Processing the call keyword arguments (line 307)
            kwargs_227246 = {}
            # Getting the type of 'self' (line 307)
            self_227244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'self', False)
            # Obtaining the member 'draw' of a type (line 307)
            draw_227245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 16), self_227244, 'draw')
            # Calling draw(args, kwargs) (line 307)
            draw_call_result_227247 = invoke(stypy.reporting.localization.Localization(__file__, 307, 16), draw_227245, *[], **kwargs_227246)
            
            
            # finally branch of the try-finally block (line 306)
            
            # Assigning a Num to a Attribute (line 309):
            
            # Assigning a Num to a Attribute (line 309):
            int_227248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 37), 'int')
            # Getting the type of 'self' (line 309)
            self_227249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'self')
            # Setting the type of the member '_idle_draw_id' of a type (line 309)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 16), self_227249, '_idle_draw_id', int_227248)
            
            # Getting the type of 'False' (line 310)
            False_227250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'stypy_return_type', False_227250)
            
            # ################# End of 'idle_draw(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'idle_draw' in the type store
            # Getting the type of 'stypy_return_type' (line 305)
            stypy_return_type_227251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_227251)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'idle_draw'
            return stypy_return_type_227251

        # Assigning a type to the variable 'idle_draw' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'idle_draw', idle_draw)
        
        # Assigning a Call to a Attribute (line 311):
        
        # Assigning a Call to a Attribute (line 311):
        
        # Call to idle_add(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'idle_draw' (line 311)
        idle_draw_227254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 43), 'idle_draw', False)
        # Processing the call keyword arguments (line 311)
        kwargs_227255 = {}
        # Getting the type of 'GLib' (line 311)
        GLib_227252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 29), 'GLib', False)
        # Obtaining the member 'idle_add' of a type (line 311)
        idle_add_227253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 29), GLib_227252, 'idle_add')
        # Calling idle_add(args, kwargs) (line 311)
        idle_add_call_result_227256 = invoke(stypy.reporting.localization.Localization(__file__, 311, 29), idle_add_227253, *[idle_draw_227254], **kwargs_227255)
        
        # Getting the type of 'self' (line 311)
        self_227257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'self')
        # Setting the type of the member '_idle_draw_id' of a type (line 311)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), self_227257, '_idle_draw_id', idle_add_call_result_227256)
        
        # ################# End of 'draw_idle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_idle' in the type store
        # Getting the type of 'stypy_return_type' (line 302)
        stypy_return_type_227258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227258)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_idle'
        return stypy_return_type_227258


    @norecursion
    def new_timer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_timer'
        module_type_store = module_type_store.open_function_context('new_timer', 313, 4, False)
        # Assigning a type to the variable 'self' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.new_timer.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.new_timer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.new_timer.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.new_timer.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.new_timer')
        FigureCanvasGTK3.new_timer.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasGTK3.new_timer.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasGTK3.new_timer.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasGTK3.new_timer.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.new_timer.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.new_timer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.new_timer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.new_timer', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_timer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_timer(...)' code ##################

        unicode_227259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, (-1)), 'unicode', u"\n        Creates a new backend-specific subclass of :class:`backend_bases.Timer`.\n        This is useful for getting periodic events through the backend's native\n        event loop. Implemented only for backends with GUIs.\n\n        Other Parameters\n        ----------------\n        interval : scalar\n            Timer interval in milliseconds\n        callbacks : list\n            Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``\n            will be executed by the timer every *interval*.\n        ")
        
        # Call to TimerGTK3(...): (line 327)
        # Getting the type of 'args' (line 327)
        args_227261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 26), 'args', False)
        # Processing the call keyword arguments (line 327)
        # Getting the type of 'kwargs' (line 327)
        kwargs_227262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 34), 'kwargs', False)
        kwargs_227263 = {'kwargs_227262': kwargs_227262}
        # Getting the type of 'TimerGTK3' (line 327)
        TimerGTK3_227260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'TimerGTK3', False)
        # Calling TimerGTK3(args, kwargs) (line 327)
        TimerGTK3_call_result_227264 = invoke(stypy.reporting.localization.Localization(__file__, 327, 15), TimerGTK3_227260, *[args_227261], **kwargs_227263)
        
        # Assigning a type to the variable 'stypy_return_type' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'stypy_return_type', TimerGTK3_call_result_227264)
        
        # ################# End of 'new_timer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_timer' in the type store
        # Getting the type of 'stypy_return_type' (line 313)
        stypy_return_type_227265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227265)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_timer'
        return stypy_return_type_227265


    @norecursion
    def flush_events(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'flush_events'
        module_type_store = module_type_store.open_function_context('flush_events', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasGTK3.flush_events.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasGTK3.flush_events.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasGTK3.flush_events.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasGTK3.flush_events.__dict__.__setitem__('stypy_function_name', 'FigureCanvasGTK3.flush_events')
        FigureCanvasGTK3.flush_events.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasGTK3.flush_events.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasGTK3.flush_events.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasGTK3.flush_events.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasGTK3.flush_events.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasGTK3.flush_events.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasGTK3.flush_events.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasGTK3.flush_events', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'flush_events', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'flush_events(...)' code ##################

        
        # Call to threads_enter(...): (line 330)
        # Processing the call keyword arguments (line 330)
        kwargs_227268 = {}
        # Getting the type of 'Gdk' (line 330)
        Gdk_227266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'Gdk', False)
        # Obtaining the member 'threads_enter' of a type (line 330)
        threads_enter_227267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), Gdk_227266, 'threads_enter')
        # Calling threads_enter(args, kwargs) (line 330)
        threads_enter_call_result_227269 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), threads_enter_227267, *[], **kwargs_227268)
        
        
        
        # Call to events_pending(...): (line 331)
        # Processing the call keyword arguments (line 331)
        kwargs_227272 = {}
        # Getting the type of 'Gtk' (line 331)
        Gtk_227270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 14), 'Gtk', False)
        # Obtaining the member 'events_pending' of a type (line 331)
        events_pending_227271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 14), Gtk_227270, 'events_pending')
        # Calling events_pending(args, kwargs) (line 331)
        events_pending_call_result_227273 = invoke(stypy.reporting.localization.Localization(__file__, 331, 14), events_pending_227271, *[], **kwargs_227272)
        
        # Testing the type of an if condition (line 331)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 8), events_pending_call_result_227273)
        # SSA begins for while statement (line 331)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to main_iteration(...): (line 332)
        # Processing the call keyword arguments (line 332)
        kwargs_227276 = {}
        # Getting the type of 'Gtk' (line 332)
        Gtk_227274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'Gtk', False)
        # Obtaining the member 'main_iteration' of a type (line 332)
        main_iteration_227275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 12), Gtk_227274, 'main_iteration')
        # Calling main_iteration(args, kwargs) (line 332)
        main_iteration_call_result_227277 = invoke(stypy.reporting.localization.Localization(__file__, 332, 12), main_iteration_227275, *[], **kwargs_227276)
        
        # SSA join for while statement (line 331)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to flush(...): (line 333)
        # Processing the call keyword arguments (line 333)
        kwargs_227280 = {}
        # Getting the type of 'Gdk' (line 333)
        Gdk_227278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'Gdk', False)
        # Obtaining the member 'flush' of a type (line 333)
        flush_227279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), Gdk_227278, 'flush')
        # Calling flush(args, kwargs) (line 333)
        flush_call_result_227281 = invoke(stypy.reporting.localization.Localization(__file__, 333, 8), flush_227279, *[], **kwargs_227280)
        
        
        # Call to threads_leave(...): (line 334)
        # Processing the call keyword arguments (line 334)
        kwargs_227284 = {}
        # Getting the type of 'Gdk' (line 334)
        Gdk_227282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'Gdk', False)
        # Obtaining the member 'threads_leave' of a type (line 334)
        threads_leave_227283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), Gdk_227282, 'threads_leave')
        # Calling threads_leave(args, kwargs) (line 334)
        threads_leave_call_result_227285 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), threads_leave_227283, *[], **kwargs_227284)
        
        
        # ################# End of 'flush_events(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'flush_events' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_227286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227286)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'flush_events'
        return stypy_return_type_227286


# Assigning a type to the variable 'FigureCanvasGTK3' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'FigureCanvasGTK3', FigureCanvasGTK3)

# Assigning a Dict to a Name (line 105):

# Obtaining an instance of the builtin type 'dict' (line 105)
dict_227287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 105)
# Adding element type (key, value) (line 105)
int_227288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 15), 'int')
unicode_227289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'unicode', u'control')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227288, unicode_227289))
# Adding element type (key, value) (line 105)
int_227290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 15), 'int')
unicode_227291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 23), 'unicode', u'shift')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227290, unicode_227291))
# Adding element type (key, value) (line 105)
int_227292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 15), 'int')
unicode_227293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 23), 'unicode', u'alt')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227292, unicode_227293))
# Adding element type (key, value) (line 105)
int_227294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 15), 'int')
unicode_227295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 23), 'unicode', u'control')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227294, unicode_227295))
# Adding element type (key, value) (line 105)
int_227296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 15), 'int')
unicode_227297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'unicode', u'shift')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227296, unicode_227297))
# Adding element type (key, value) (line 105)
int_227298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 15), 'int')
unicode_227299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 23), 'unicode', u'alt')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227298, unicode_227299))
# Adding element type (key, value) (line 105)
int_227300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 15), 'int')
unicode_227301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 23), 'unicode', u'left')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227300, unicode_227301))
# Adding element type (key, value) (line 105)
int_227302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 15), 'int')
unicode_227303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 23), 'unicode', u'up')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227302, unicode_227303))
# Adding element type (key, value) (line 105)
int_227304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 15), 'int')
unicode_227305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 23), 'unicode', u'right')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227304, unicode_227305))
# Adding element type (key, value) (line 105)
int_227306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 15), 'int')
unicode_227307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 23), 'unicode', u'down')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227306, unicode_227307))
# Adding element type (key, value) (line 105)
int_227308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 15), 'int')
unicode_227309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 23), 'unicode', u'escape')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227308, unicode_227309))
# Adding element type (key, value) (line 105)
int_227310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 15), 'int')
unicode_227311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'unicode', u'f1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227310, unicode_227311))
# Adding element type (key, value) (line 105)
int_227312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 15), 'int')
unicode_227313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'unicode', u'f2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227312, unicode_227313))
# Adding element type (key, value) (line 105)
int_227314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 15), 'int')
unicode_227315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 23), 'unicode', u'f3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227314, unicode_227315))
# Adding element type (key, value) (line 105)
int_227316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 15), 'int')
unicode_227317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 23), 'unicode', u'f4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227316, unicode_227317))
# Adding element type (key, value) (line 105)
int_227318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'int')
unicode_227319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 23), 'unicode', u'f5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227318, unicode_227319))
# Adding element type (key, value) (line 105)
int_227320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 15), 'int')
unicode_227321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'unicode', u'f6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227320, unicode_227321))
# Adding element type (key, value) (line 105)
int_227322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 15), 'int')
unicode_227323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 23), 'unicode', u'f7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227322, unicode_227323))
# Adding element type (key, value) (line 105)
int_227324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 15), 'int')
unicode_227325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 23), 'unicode', u'f8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227324, unicode_227325))
# Adding element type (key, value) (line 105)
int_227326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 15), 'int')
unicode_227327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 23), 'unicode', u'f9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227326, unicode_227327))
# Adding element type (key, value) (line 105)
int_227328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 15), 'int')
unicode_227329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 23), 'unicode', u'f10')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227328, unicode_227329))
# Adding element type (key, value) (line 105)
int_227330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 15), 'int')
unicode_227331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 23), 'unicode', u'f11')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227330, unicode_227331))
# Adding element type (key, value) (line 105)
int_227332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 15), 'int')
unicode_227333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 23), 'unicode', u'f12')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227332, unicode_227333))
# Adding element type (key, value) (line 105)
int_227334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 15), 'int')
unicode_227335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 23), 'unicode', u'scroll_lock')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227334, unicode_227335))
# Adding element type (key, value) (line 105)
int_227336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 15), 'int')
unicode_227337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 23), 'unicode', u'break')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227336, unicode_227337))
# Adding element type (key, value) (line 105)
int_227338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 15), 'int')
unicode_227339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 23), 'unicode', u'backspace')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227338, unicode_227339))
# Adding element type (key, value) (line 105)
int_227340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 15), 'int')
unicode_227341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 23), 'unicode', u'enter')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227340, unicode_227341))
# Adding element type (key, value) (line 105)
int_227342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 15), 'int')
unicode_227343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 23), 'unicode', u'insert')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227342, unicode_227343))
# Adding element type (key, value) (line 105)
int_227344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 15), 'int')
unicode_227345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 23), 'unicode', u'delete')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227344, unicode_227345))
# Adding element type (key, value) (line 105)
int_227346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 15), 'int')
unicode_227347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 23), 'unicode', u'home')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227346, unicode_227347))
# Adding element type (key, value) (line 105)
int_227348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 15), 'int')
unicode_227349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'unicode', u'end')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227348, unicode_227349))
# Adding element type (key, value) (line 105)
int_227350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 15), 'int')
unicode_227351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'unicode', u'pageup')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227350, unicode_227351))
# Adding element type (key, value) (line 105)
int_227352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 15), 'int')
unicode_227353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'unicode', u'pagedown')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227352, unicode_227353))
# Adding element type (key, value) (line 105)
int_227354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 15), 'int')
unicode_227355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'unicode', u'0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227354, unicode_227355))
# Adding element type (key, value) (line 105)
int_227356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 15), 'int')
unicode_227357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 23), 'unicode', u'1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227356, unicode_227357))
# Adding element type (key, value) (line 105)
int_227358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 15), 'int')
unicode_227359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 23), 'unicode', u'2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227358, unicode_227359))
# Adding element type (key, value) (line 105)
int_227360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 15), 'int')
unicode_227361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 23), 'unicode', u'3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227360, unicode_227361))
# Adding element type (key, value) (line 105)
int_227362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 15), 'int')
unicode_227363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 23), 'unicode', u'4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227362, unicode_227363))
# Adding element type (key, value) (line 105)
int_227364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 15), 'int')
unicode_227365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 23), 'unicode', u'5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227364, unicode_227365))
# Adding element type (key, value) (line 105)
int_227366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 15), 'int')
unicode_227367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'unicode', u'6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227366, unicode_227367))
# Adding element type (key, value) (line 105)
int_227368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 15), 'int')
unicode_227369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 23), 'unicode', u'7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227368, unicode_227369))
# Adding element type (key, value) (line 105)
int_227370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 15), 'int')
unicode_227371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 23), 'unicode', u'8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227370, unicode_227371))
# Adding element type (key, value) (line 105)
int_227372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 15), 'int')
unicode_227373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 23), 'unicode', u'9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227372, unicode_227373))
# Adding element type (key, value) (line 105)
int_227374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 15), 'int')
unicode_227375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 23), 'unicode', u'+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227374, unicode_227375))
# Adding element type (key, value) (line 105)
int_227376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 15), 'int')
unicode_227377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 23), 'unicode', u'-')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227376, unicode_227377))
# Adding element type (key, value) (line 105)
int_227378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 15), 'int')
unicode_227379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 23), 'unicode', u'*')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227378, unicode_227379))
# Adding element type (key, value) (line 105)
int_227380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 15), 'int')
unicode_227381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 23), 'unicode', u'/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227380, unicode_227381))
# Adding element type (key, value) (line 105)
int_227382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 15), 'int')
unicode_227383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'unicode', u'dec')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227382, unicode_227383))
# Adding element type (key, value) (line 105)
int_227384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 15), 'int')
unicode_227385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 23), 'unicode', u'enter')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 14), dict_227287, (int_227384, unicode_227385))

# Getting the type of 'FigureCanvasGTK3'
FigureCanvasGTK3_227386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasGTK3')
# Setting the type of the member 'keyvald' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasGTK3_227386, 'keyvald', dict_227287)

# Assigning a BinOp to a Name (line 158):
# Getting the type of 'Gdk' (line 158)
Gdk_227387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 'Gdk')
# Obtaining the member 'EventMask' of a type (line 158)
EventMask_227388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 18), Gdk_227387, 'EventMask')
# Obtaining the member 'BUTTON_PRESS_MASK' of a type (line 158)
BUTTON_PRESS_MASK_227389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 18), EventMask_227388, 'BUTTON_PRESS_MASK')
# Getting the type of 'Gdk' (line 159)
Gdk_227390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 18), 'Gdk')
# Obtaining the member 'EventMask' of a type (line 159)
EventMask_227391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 18), Gdk_227390, 'EventMask')
# Obtaining the member 'BUTTON_RELEASE_MASK' of a type (line 159)
BUTTON_RELEASE_MASK_227392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 18), EventMask_227391, 'BUTTON_RELEASE_MASK')
# Applying the binary operator '|' (line 158)
result_or__227393 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 18), '|', BUTTON_PRESS_MASK_227389, BUTTON_RELEASE_MASK_227392)

# Getting the type of 'Gdk' (line 160)
Gdk_227394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 18), 'Gdk')
# Obtaining the member 'EventMask' of a type (line 160)
EventMask_227395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 18), Gdk_227394, 'EventMask')
# Obtaining the member 'EXPOSURE_MASK' of a type (line 160)
EXPOSURE_MASK_227396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 18), EventMask_227395, 'EXPOSURE_MASK')
# Applying the binary operator '|' (line 159)
result_or__227397 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 52), '|', result_or__227393, EXPOSURE_MASK_227396)

# Getting the type of 'Gdk' (line 161)
Gdk_227398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 18), 'Gdk')
# Obtaining the member 'EventMask' of a type (line 161)
EventMask_227399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 18), Gdk_227398, 'EventMask')
# Obtaining the member 'KEY_PRESS_MASK' of a type (line 161)
KEY_PRESS_MASK_227400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 18), EventMask_227399, 'KEY_PRESS_MASK')
# Applying the binary operator '|' (line 160)
result_or__227401 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 52), '|', result_or__227397, KEY_PRESS_MASK_227400)

# Getting the type of 'Gdk' (line 162)
Gdk_227402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'Gdk')
# Obtaining the member 'EventMask' of a type (line 162)
EventMask_227403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 18), Gdk_227402, 'EventMask')
# Obtaining the member 'KEY_RELEASE_MASK' of a type (line 162)
KEY_RELEASE_MASK_227404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 18), EventMask_227403, 'KEY_RELEASE_MASK')
# Applying the binary operator '|' (line 161)
result_or__227405 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 52), '|', result_or__227401, KEY_RELEASE_MASK_227404)

# Getting the type of 'Gdk' (line 163)
Gdk_227406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'Gdk')
# Obtaining the member 'EventMask' of a type (line 163)
EventMask_227407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 18), Gdk_227406, 'EventMask')
# Obtaining the member 'ENTER_NOTIFY_MASK' of a type (line 163)
ENTER_NOTIFY_MASK_227408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 18), EventMask_227407, 'ENTER_NOTIFY_MASK')
# Applying the binary operator '|' (line 162)
result_or__227409 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 52), '|', result_or__227405, ENTER_NOTIFY_MASK_227408)

# Getting the type of 'Gdk' (line 164)
Gdk_227410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'Gdk')
# Obtaining the member 'EventMask' of a type (line 164)
EventMask_227411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 18), Gdk_227410, 'EventMask')
# Obtaining the member 'LEAVE_NOTIFY_MASK' of a type (line 164)
LEAVE_NOTIFY_MASK_227412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 18), EventMask_227411, 'LEAVE_NOTIFY_MASK')
# Applying the binary operator '|' (line 163)
result_or__227413 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 52), '|', result_or__227409, LEAVE_NOTIFY_MASK_227412)

# Getting the type of 'Gdk' (line 165)
Gdk_227414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), 'Gdk')
# Obtaining the member 'EventMask' of a type (line 165)
EventMask_227415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 18), Gdk_227414, 'EventMask')
# Obtaining the member 'POINTER_MOTION_MASK' of a type (line 165)
POINTER_MOTION_MASK_227416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 18), EventMask_227415, 'POINTER_MOTION_MASK')
# Applying the binary operator '|' (line 164)
result_or__227417 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 52), '|', result_or__227413, POINTER_MOTION_MASK_227416)

# Getting the type of 'Gdk' (line 166)
Gdk_227418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 18), 'Gdk')
# Obtaining the member 'EventMask' of a type (line 166)
EventMask_227419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 18), Gdk_227418, 'EventMask')
# Obtaining the member 'POINTER_MOTION_HINT_MASK' of a type (line 166)
POINTER_MOTION_HINT_MASK_227420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 18), EventMask_227419, 'POINTER_MOTION_HINT_MASK')
# Applying the binary operator '|' (line 165)
result_or__227421 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 52), '|', result_or__227417, POINTER_MOTION_HINT_MASK_227420)

# Getting the type of 'Gdk' (line 167)
Gdk_227422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'Gdk')
# Obtaining the member 'EventMask' of a type (line 167)
EventMask_227423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 18), Gdk_227422, 'EventMask')
# Obtaining the member 'SCROLL_MASK' of a type (line 167)
SCROLL_MASK_227424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 18), EventMask_227423, 'SCROLL_MASK')
# Applying the binary operator '|' (line 166)
result_or__227425 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 56), '|', result_or__227421, SCROLL_MASK_227424)

# Getting the type of 'FigureCanvasGTK3'
FigureCanvasGTK3_227426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasGTK3')
# Setting the type of the member 'event_mask' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasGTK3_227426, 'event_mask', result_or__227425)
# Declaration of the 'FigureManagerGTK3' class
# Getting the type of 'FigureManagerBase' (line 337)
FigureManagerBase_227427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'FigureManagerBase')

class FigureManagerGTK3(FigureManagerBase_227427, ):
    unicode_227428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, (-1)), 'unicode', u'\n    Attributes\n    ----------\n    canvas : `FigureCanvas`\n        The FigureCanvas instance\n    num : int or str\n        The Figure number\n    toolbar : Gtk.Toolbar\n        The Gtk.Toolbar  (gtk only)\n    vbox : Gtk.VBox\n        The Gtk.VBox containing the canvas and toolbar (gtk only)\n    window : Gtk.Window\n        The Gtk.Window   (gtk only)\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 353, 4, False)
        # Assigning a type to the variable 'self' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTK3.__init__', ['canvas', 'num'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['canvas', 'num'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'self' (line 354)
        self_227431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 35), 'self', False)
        # Getting the type of 'canvas' (line 354)
        canvas_227432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 41), 'canvas', False)
        # Getting the type of 'num' (line 354)
        num_227433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 49), 'num', False)
        # Processing the call keyword arguments (line 354)
        kwargs_227434 = {}
        # Getting the type of 'FigureManagerBase' (line 354)
        FigureManagerBase_227429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'FigureManagerBase', False)
        # Obtaining the member '__init__' of a type (line 354)
        init___227430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), FigureManagerBase_227429, '__init__')
        # Calling __init__(args, kwargs) (line 354)
        init___call_result_227435 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), init___227430, *[self_227431, canvas_227432, num_227433], **kwargs_227434)
        
        
        # Assigning a Call to a Attribute (line 356):
        
        # Assigning a Call to a Attribute (line 356):
        
        # Call to Window(...): (line 356)
        # Processing the call keyword arguments (line 356)
        kwargs_227438 = {}
        # Getting the type of 'Gtk' (line 356)
        Gtk_227436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'Gtk', False)
        # Obtaining the member 'Window' of a type (line 356)
        Window_227437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 22), Gtk_227436, 'Window')
        # Calling Window(args, kwargs) (line 356)
        Window_call_result_227439 = invoke(stypy.reporting.localization.Localization(__file__, 356, 22), Window_227437, *[], **kwargs_227438)
        
        # Getting the type of 'self' (line 356)
        self_227440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'self')
        # Setting the type of the member 'window' of a type (line 356)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), self_227440, 'window', Window_call_result_227439)
        
        # Call to set_wmclass(...): (line 357)
        # Processing the call arguments (line 357)
        unicode_227444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 32), 'unicode', u'matplotlib')
        unicode_227445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 46), 'unicode', u'Matplotlib')
        # Processing the call keyword arguments (line 357)
        kwargs_227446 = {}
        # Getting the type of 'self' (line 357)
        self_227441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 357)
        window_227442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), self_227441, 'window')
        # Obtaining the member 'set_wmclass' of a type (line 357)
        set_wmclass_227443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), window_227442, 'set_wmclass')
        # Calling set_wmclass(args, kwargs) (line 357)
        set_wmclass_call_result_227447 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), set_wmclass_227443, *[unicode_227444, unicode_227445], **kwargs_227446)
        
        
        # Call to set_window_title(...): (line 358)
        # Processing the call arguments (line 358)
        unicode_227450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 30), 'unicode', u'Figure %d')
        # Getting the type of 'num' (line 358)
        num_227451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 44), 'num', False)
        # Applying the binary operator '%' (line 358)
        result_mod_227452 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 30), '%', unicode_227450, num_227451)
        
        # Processing the call keyword arguments (line 358)
        kwargs_227453 = {}
        # Getting the type of 'self' (line 358)
        self_227448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'self', False)
        # Obtaining the member 'set_window_title' of a type (line 358)
        set_window_title_227449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), self_227448, 'set_window_title')
        # Calling set_window_title(args, kwargs) (line 358)
        set_window_title_call_result_227454 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), set_window_title_227449, *[result_mod_227452], **kwargs_227453)
        
        
        
        # SSA begins for try-except statement (line 359)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to set_icon_from_file(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'window_icon' (line 360)
        window_icon_227458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 43), 'window_icon', False)
        # Processing the call keyword arguments (line 360)
        kwargs_227459 = {}
        # Getting the type of 'self' (line 360)
        self_227455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'self', False)
        # Obtaining the member 'window' of a type (line 360)
        window_227456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), self_227455, 'window')
        # Obtaining the member 'set_icon_from_file' of a type (line 360)
        set_icon_from_file_227457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), window_227456, 'set_icon_from_file')
        # Calling set_icon_from_file(args, kwargs) (line 360)
        set_icon_from_file_call_result_227460 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), set_icon_from_file_227457, *[window_icon_227458], **kwargs_227459)
        
        # SSA branch for the except part of a try statement (line 359)
        # SSA branch for the except 'Tuple' branch of a try statement (line 359)
        module_type_store.open_ssa_branch('except')
        # SSA branch for the except '<any exception>' branch of a try statement (line 359)
        module_type_store.open_ssa_branch('except')
        
        # Call to report(...): (line 369)
        # Processing the call arguments (line 369)
        unicode_227463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 27), 'unicode', u'Could not load matplotlib icon: %s')
        
        # Obtaining the type of the subscript
        int_227464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 81), 'int')
        
        # Call to exc_info(...): (line 369)
        # Processing the call keyword arguments (line 369)
        kwargs_227467 = {}
        # Getting the type of 'sys' (line 369)
        sys_227465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 66), 'sys', False)
        # Obtaining the member 'exc_info' of a type (line 369)
        exc_info_227466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 66), sys_227465, 'exc_info')
        # Calling exc_info(args, kwargs) (line 369)
        exc_info_call_result_227468 = invoke(stypy.reporting.localization.Localization(__file__, 369, 66), exc_info_227466, *[], **kwargs_227467)
        
        # Obtaining the member '__getitem__' of a type (line 369)
        getitem___227469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 66), exc_info_call_result_227468, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 369)
        subscript_call_result_227470 = invoke(stypy.reporting.localization.Localization(__file__, 369, 66), getitem___227469, int_227464)
        
        # Applying the binary operator '%' (line 369)
        result_mod_227471 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 27), '%', unicode_227463, subscript_call_result_227470)
        
        # Processing the call keyword arguments (line 369)
        kwargs_227472 = {}
        # Getting the type of 'verbose' (line 369)
        verbose_227461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'verbose', False)
        # Obtaining the member 'report' of a type (line 369)
        report_227462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 12), verbose_227461, 'report')
        # Calling report(args, kwargs) (line 369)
        report_call_result_227473 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), report_227462, *[result_mod_227471], **kwargs_227472)
        
        # SSA join for try-except statement (line 359)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 371):
        
        # Assigning a Call to a Attribute (line 371):
        
        # Call to Box(...): (line 371)
        # Processing the call keyword arguments (line 371)
        kwargs_227476 = {}
        # Getting the type of 'Gtk' (line 371)
        Gtk_227474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 20), 'Gtk', False)
        # Obtaining the member 'Box' of a type (line 371)
        Box_227475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 20), Gtk_227474, 'Box')
        # Calling Box(args, kwargs) (line 371)
        Box_call_result_227477 = invoke(stypy.reporting.localization.Localization(__file__, 371, 20), Box_227475, *[], **kwargs_227476)
        
        # Getting the type of 'self' (line 371)
        self_227478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'self')
        # Setting the type of the member 'vbox' of a type (line 371)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), self_227478, 'vbox', Box_call_result_227477)
        
        # Call to set_property(...): (line 372)
        # Processing the call arguments (line 372)
        unicode_227482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 31), 'unicode', u'orientation')
        # Getting the type of 'Gtk' (line 372)
        Gtk_227483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 46), 'Gtk', False)
        # Obtaining the member 'Orientation' of a type (line 372)
        Orientation_227484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 46), Gtk_227483, 'Orientation')
        # Obtaining the member 'VERTICAL' of a type (line 372)
        VERTICAL_227485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 46), Orientation_227484, 'VERTICAL')
        # Processing the call keyword arguments (line 372)
        kwargs_227486 = {}
        # Getting the type of 'self' (line 372)
        self_227479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'self', False)
        # Obtaining the member 'vbox' of a type (line 372)
        vbox_227480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 8), self_227479, 'vbox')
        # Obtaining the member 'set_property' of a type (line 372)
        set_property_227481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 8), vbox_227480, 'set_property')
        # Calling set_property(args, kwargs) (line 372)
        set_property_call_result_227487 = invoke(stypy.reporting.localization.Localization(__file__, 372, 8), set_property_227481, *[unicode_227482, VERTICAL_227485], **kwargs_227486)
        
        
        # Call to add(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'self' (line 373)
        self_227491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 24), 'self', False)
        # Obtaining the member 'vbox' of a type (line 373)
        vbox_227492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 24), self_227491, 'vbox')
        # Processing the call keyword arguments (line 373)
        kwargs_227493 = {}
        # Getting the type of 'self' (line 373)
        self_227488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 373)
        window_227489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 8), self_227488, 'window')
        # Obtaining the member 'add' of a type (line 373)
        add_227490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 8), window_227489, 'add')
        # Calling add(args, kwargs) (line 373)
        add_call_result_227494 = invoke(stypy.reporting.localization.Localization(__file__, 373, 8), add_227490, *[vbox_227492], **kwargs_227493)
        
        
        # Call to show(...): (line 374)
        # Processing the call keyword arguments (line 374)
        kwargs_227498 = {}
        # Getting the type of 'self' (line 374)
        self_227495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'self', False)
        # Obtaining the member 'vbox' of a type (line 374)
        vbox_227496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), self_227495, 'vbox')
        # Obtaining the member 'show' of a type (line 374)
        show_227497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 8), vbox_227496, 'show')
        # Calling show(args, kwargs) (line 374)
        show_call_result_227499 = invoke(stypy.reporting.localization.Localization(__file__, 374, 8), show_227497, *[], **kwargs_227498)
        
        
        # Call to show(...): (line 376)
        # Processing the call keyword arguments (line 376)
        kwargs_227503 = {}
        # Getting the type of 'self' (line 376)
        self_227500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 376)
        canvas_227501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), self_227500, 'canvas')
        # Obtaining the member 'show' of a type (line 376)
        show_227502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), canvas_227501, 'show')
        # Calling show(args, kwargs) (line 376)
        show_call_result_227504 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), show_227502, *[], **kwargs_227503)
        
        
        # Call to pack_start(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'self' (line 378)
        self_227508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 29), 'self', False)
        # Obtaining the member 'canvas' of a type (line 378)
        canvas_227509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 29), self_227508, 'canvas')
        # Getting the type of 'True' (line 378)
        True_227510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 42), 'True', False)
        # Getting the type of 'True' (line 378)
        True_227511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 48), 'True', False)
        int_227512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 54), 'int')
        # Processing the call keyword arguments (line 378)
        kwargs_227513 = {}
        # Getting the type of 'self' (line 378)
        self_227505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self', False)
        # Obtaining the member 'vbox' of a type (line 378)
        vbox_227506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_227505, 'vbox')
        # Obtaining the member 'pack_start' of a type (line 378)
        pack_start_227507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), vbox_227506, 'pack_start')
        # Calling pack_start(args, kwargs) (line 378)
        pack_start_call_result_227514 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), pack_start_227507, *[canvas_227509, True_227510, True_227511, int_227512], **kwargs_227513)
        
        
        # Assigning a Call to a Name (line 380):
        
        # Assigning a Call to a Name (line 380):
        
        # Call to int(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'self' (line 380)
        self_227516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 17), 'self', False)
        # Obtaining the member 'canvas' of a type (line 380)
        canvas_227517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 17), self_227516, 'canvas')
        # Obtaining the member 'figure' of a type (line 380)
        figure_227518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 17), canvas_227517, 'figure')
        # Obtaining the member 'bbox' of a type (line 380)
        bbox_227519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 17), figure_227518, 'bbox')
        # Obtaining the member 'width' of a type (line 380)
        width_227520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 17), bbox_227519, 'width')
        # Processing the call keyword arguments (line 380)
        kwargs_227521 = {}
        # Getting the type of 'int' (line 380)
        int_227515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'int', False)
        # Calling int(args, kwargs) (line 380)
        int_call_result_227522 = invoke(stypy.reporting.localization.Localization(__file__, 380, 12), int_227515, *[width_227520], **kwargs_227521)
        
        # Assigning a type to the variable 'w' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'w', int_call_result_227522)
        
        # Assigning a Call to a Name (line 381):
        
        # Assigning a Call to a Name (line 381):
        
        # Call to int(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'self' (line 381)
        self_227524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 17), 'self', False)
        # Obtaining the member 'canvas' of a type (line 381)
        canvas_227525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 17), self_227524, 'canvas')
        # Obtaining the member 'figure' of a type (line 381)
        figure_227526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 17), canvas_227525, 'figure')
        # Obtaining the member 'bbox' of a type (line 381)
        bbox_227527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 17), figure_227526, 'bbox')
        # Obtaining the member 'height' of a type (line 381)
        height_227528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 17), bbox_227527, 'height')
        # Processing the call keyword arguments (line 381)
        kwargs_227529 = {}
        # Getting the type of 'int' (line 381)
        int_227523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'int', False)
        # Calling int(args, kwargs) (line 381)
        int_call_result_227530 = invoke(stypy.reporting.localization.Localization(__file__, 381, 12), int_227523, *[height_227528], **kwargs_227529)
        
        # Assigning a type to the variable 'h' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'h', int_call_result_227530)
        
        # Assigning a Call to a Attribute (line 383):
        
        # Assigning a Call to a Attribute (line 383):
        
        # Call to _get_toolmanager(...): (line 383)
        # Processing the call keyword arguments (line 383)
        kwargs_227533 = {}
        # Getting the type of 'self' (line 383)
        self_227531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 27), 'self', False)
        # Obtaining the member '_get_toolmanager' of a type (line 383)
        _get_toolmanager_227532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 27), self_227531, '_get_toolmanager')
        # Calling _get_toolmanager(args, kwargs) (line 383)
        _get_toolmanager_call_result_227534 = invoke(stypy.reporting.localization.Localization(__file__, 383, 27), _get_toolmanager_227532, *[], **kwargs_227533)
        
        # Getting the type of 'self' (line 383)
        self_227535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self')
        # Setting the type of the member 'toolmanager' of a type (line 383)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_227535, 'toolmanager', _get_toolmanager_call_result_227534)
        
        # Assigning a Call to a Attribute (line 384):
        
        # Assigning a Call to a Attribute (line 384):
        
        # Call to _get_toolbar(...): (line 384)
        # Processing the call keyword arguments (line 384)
        kwargs_227538 = {}
        # Getting the type of 'self' (line 384)
        self_227536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 23), 'self', False)
        # Obtaining the member '_get_toolbar' of a type (line 384)
        _get_toolbar_227537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 23), self_227536, '_get_toolbar')
        # Calling _get_toolbar(args, kwargs) (line 384)
        _get_toolbar_call_result_227539 = invoke(stypy.reporting.localization.Localization(__file__, 384, 23), _get_toolbar_227537, *[], **kwargs_227538)
        
        # Getting the type of 'self' (line 384)
        self_227540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'self')
        # Setting the type of the member 'toolbar' of a type (line 384)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), self_227540, 'toolbar', _get_toolbar_call_result_227539)
        
        # Assigning a Name to a Attribute (line 385):
        
        # Assigning a Name to a Attribute (line 385):
        # Getting the type of 'None' (line 385)
        None_227541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 25), 'None')
        # Getting the type of 'self' (line 385)
        self_227542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'self')
        # Setting the type of the member 'statusbar' of a type (line 385)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 8), self_227542, 'statusbar', None_227541)

        @norecursion
        def add_widget(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'add_widget'
            module_type_store = module_type_store.open_function_context('add_widget', 387, 8, False)
            
            # Passed parameters checking function
            add_widget.stypy_localization = localization
            add_widget.stypy_type_of_self = None
            add_widget.stypy_type_store = module_type_store
            add_widget.stypy_function_name = 'add_widget'
            add_widget.stypy_param_names_list = ['child', 'expand', 'fill', 'padding']
            add_widget.stypy_varargs_param_name = None
            add_widget.stypy_kwargs_param_name = None
            add_widget.stypy_call_defaults = defaults
            add_widget.stypy_call_varargs = varargs
            add_widget.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'add_widget', ['child', 'expand', 'fill', 'padding'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'add_widget', localization, ['child', 'expand', 'fill', 'padding'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'add_widget(...)' code ##################

            
            # Call to show(...): (line 388)
            # Processing the call keyword arguments (line 388)
            kwargs_227545 = {}
            # Getting the type of 'child' (line 388)
            child_227543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'child', False)
            # Obtaining the member 'show' of a type (line 388)
            show_227544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 12), child_227543, 'show')
            # Calling show(args, kwargs) (line 388)
            show_call_result_227546 = invoke(stypy.reporting.localization.Localization(__file__, 388, 12), show_227544, *[], **kwargs_227545)
            
            
            # Call to pack_end(...): (line 389)
            # Processing the call arguments (line 389)
            # Getting the type of 'child' (line 389)
            child_227550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 31), 'child', False)
            # Getting the type of 'False' (line 389)
            False_227551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 38), 'False', False)
            # Getting the type of 'False' (line 389)
            False_227552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 45), 'False', False)
            int_227553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 52), 'int')
            # Processing the call keyword arguments (line 389)
            kwargs_227554 = {}
            # Getting the type of 'self' (line 389)
            self_227547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'self', False)
            # Obtaining the member 'vbox' of a type (line 389)
            vbox_227548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), self_227547, 'vbox')
            # Obtaining the member 'pack_end' of a type (line 389)
            pack_end_227549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), vbox_227548, 'pack_end')
            # Calling pack_end(args, kwargs) (line 389)
            pack_end_call_result_227555 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), pack_end_227549, *[child_227550, False_227551, False_227552, int_227553], **kwargs_227554)
            
            
            # Assigning a Call to a Name (line 390):
            
            # Assigning a Call to a Name (line 390):
            
            # Call to size_request(...): (line 390)
            # Processing the call keyword arguments (line 390)
            kwargs_227558 = {}
            # Getting the type of 'child' (line 390)
            child_227556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 27), 'child', False)
            # Obtaining the member 'size_request' of a type (line 390)
            size_request_227557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 27), child_227556, 'size_request')
            # Calling size_request(args, kwargs) (line 390)
            size_request_call_result_227559 = invoke(stypy.reporting.localization.Localization(__file__, 390, 27), size_request_227557, *[], **kwargs_227558)
            
            # Assigning a type to the variable 'size_request' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'size_request', size_request_call_result_227559)
            # Getting the type of 'size_request' (line 391)
            size_request_227560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 19), 'size_request')
            # Obtaining the member 'height' of a type (line 391)
            height_227561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 19), size_request_227560, 'height')
            # Assigning a type to the variable 'stypy_return_type' (line 391)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'stypy_return_type', height_227561)
            
            # ################# End of 'add_widget(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'add_widget' in the type store
            # Getting the type of 'stypy_return_type' (line 387)
            stypy_return_type_227562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_227562)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'add_widget'
            return stypy_return_type_227562

        # Assigning a type to the variable 'add_widget' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'add_widget', add_widget)
        
        # Getting the type of 'self' (line 393)
        self_227563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 11), 'self')
        # Obtaining the member 'toolmanager' of a type (line 393)
        toolmanager_227564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 11), self_227563, 'toolmanager')
        # Testing the type of an if condition (line 393)
        if_condition_227565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 8), toolmanager_227564)
        # Assigning a type to the variable 'if_condition_227565' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'if_condition_227565', if_condition_227565)
        # SSA begins for if statement (line 393)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add_tools_to_manager(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'self' (line 394)
        self_227568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 47), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 394)
        toolmanager_227569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 47), self_227568, 'toolmanager')
        # Processing the call keyword arguments (line 394)
        kwargs_227570 = {}
        # Getting the type of 'backend_tools' (line 394)
        backend_tools_227566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'backend_tools', False)
        # Obtaining the member 'add_tools_to_manager' of a type (line 394)
        add_tools_to_manager_227567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), backend_tools_227566, 'add_tools_to_manager')
        # Calling add_tools_to_manager(args, kwargs) (line 394)
        add_tools_to_manager_call_result_227571 = invoke(stypy.reporting.localization.Localization(__file__, 394, 12), add_tools_to_manager_227567, *[toolmanager_227569], **kwargs_227570)
        
        
        # Getting the type of 'self' (line 395)
        self_227572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 15), 'self')
        # Obtaining the member 'toolbar' of a type (line 395)
        toolbar_227573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 15), self_227572, 'toolbar')
        # Testing the type of an if condition (line 395)
        if_condition_227574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 12), toolbar_227573)
        # Assigning a type to the variable 'if_condition_227574' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'if_condition_227574', if_condition_227574)
        # SSA begins for if statement (line 395)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to add_tools_to_container(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'self' (line 396)
        self_227577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 53), 'self', False)
        # Obtaining the member 'toolbar' of a type (line 396)
        toolbar_227578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 53), self_227577, 'toolbar')
        # Processing the call keyword arguments (line 396)
        kwargs_227579 = {}
        # Getting the type of 'backend_tools' (line 396)
        backend_tools_227575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 16), 'backend_tools', False)
        # Obtaining the member 'add_tools_to_container' of a type (line 396)
        add_tools_to_container_227576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 16), backend_tools_227575, 'add_tools_to_container')
        # Calling add_tools_to_container(args, kwargs) (line 396)
        add_tools_to_container_call_result_227580 = invoke(stypy.reporting.localization.Localization(__file__, 396, 16), add_tools_to_container_227576, *[toolbar_227578], **kwargs_227579)
        
        
        # Assigning a Call to a Attribute (line 397):
        
        # Assigning a Call to a Attribute (line 397):
        
        # Call to StatusbarGTK3(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'self' (line 397)
        self_227582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 47), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 397)
        toolmanager_227583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 47), self_227582, 'toolmanager')
        # Processing the call keyword arguments (line 397)
        kwargs_227584 = {}
        # Getting the type of 'StatusbarGTK3' (line 397)
        StatusbarGTK3_227581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 33), 'StatusbarGTK3', False)
        # Calling StatusbarGTK3(args, kwargs) (line 397)
        StatusbarGTK3_call_result_227585 = invoke(stypy.reporting.localization.Localization(__file__, 397, 33), StatusbarGTK3_227581, *[toolmanager_227583], **kwargs_227584)
        
        # Getting the type of 'self' (line 397)
        self_227586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 16), 'self')
        # Setting the type of the member 'statusbar' of a type (line 397)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 16), self_227586, 'statusbar', StatusbarGTK3_call_result_227585)
        
        # Getting the type of 'h' (line 398)
        h_227587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 16), 'h')
        
        # Call to add_widget(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'self' (line 398)
        self_227589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 32), 'self', False)
        # Obtaining the member 'statusbar' of a type (line 398)
        statusbar_227590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 32), self_227589, 'statusbar')
        # Getting the type of 'False' (line 398)
        False_227591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 48), 'False', False)
        # Getting the type of 'False' (line 398)
        False_227592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 55), 'False', False)
        int_227593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 62), 'int')
        # Processing the call keyword arguments (line 398)
        kwargs_227594 = {}
        # Getting the type of 'add_widget' (line 398)
        add_widget_227588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 21), 'add_widget', False)
        # Calling add_widget(args, kwargs) (line 398)
        add_widget_call_result_227595 = invoke(stypy.reporting.localization.Localization(__file__, 398, 21), add_widget_227588, *[statusbar_227590, False_227591, False_227592, int_227593], **kwargs_227594)
        
        # Applying the binary operator '+=' (line 398)
        result_iadd_227596 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 16), '+=', h_227587, add_widget_call_result_227595)
        # Assigning a type to the variable 'h' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 16), 'h', result_iadd_227596)
        
        
        # Getting the type of 'h' (line 399)
        h_227597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'h')
        
        # Call to add_widget(...): (line 399)
        # Processing the call arguments (line 399)
        
        # Call to HSeparator(...): (line 399)
        # Processing the call keyword arguments (line 399)
        kwargs_227601 = {}
        # Getting the type of 'Gtk' (line 399)
        Gtk_227599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 32), 'Gtk', False)
        # Obtaining the member 'HSeparator' of a type (line 399)
        HSeparator_227600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 32), Gtk_227599, 'HSeparator')
        # Calling HSeparator(args, kwargs) (line 399)
        HSeparator_call_result_227602 = invoke(stypy.reporting.localization.Localization(__file__, 399, 32), HSeparator_227600, *[], **kwargs_227601)
        
        # Getting the type of 'False' (line 399)
        False_227603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 50), 'False', False)
        # Getting the type of 'False' (line 399)
        False_227604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 57), 'False', False)
        int_227605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 64), 'int')
        # Processing the call keyword arguments (line 399)
        kwargs_227606 = {}
        # Getting the type of 'add_widget' (line 399)
        add_widget_227598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 21), 'add_widget', False)
        # Calling add_widget(args, kwargs) (line 399)
        add_widget_call_result_227607 = invoke(stypy.reporting.localization.Localization(__file__, 399, 21), add_widget_227598, *[HSeparator_call_result_227602, False_227603, False_227604, int_227605], **kwargs_227606)
        
        # Applying the binary operator '+=' (line 399)
        result_iadd_227608 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 16), '+=', h_227597, add_widget_call_result_227607)
        # Assigning a type to the variable 'h' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 16), 'h', result_iadd_227608)
        
        # SSA join for if statement (line 395)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 393)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 401)
        self_227609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 11), 'self')
        # Obtaining the member 'toolbar' of a type (line 401)
        toolbar_227610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 11), self_227609, 'toolbar')
        # Getting the type of 'None' (line 401)
        None_227611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 31), 'None')
        # Applying the binary operator 'isnot' (line 401)
        result_is_not_227612 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 11), 'isnot', toolbar_227610, None_227611)
        
        # Testing the type of an if condition (line 401)
        if_condition_227613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 401, 8), result_is_not_227612)
        # Assigning a type to the variable 'if_condition_227613' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'if_condition_227613', if_condition_227613)
        # SSA begins for if statement (line 401)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to show(...): (line 402)
        # Processing the call keyword arguments (line 402)
        kwargs_227617 = {}
        # Getting the type of 'self' (line 402)
        self_227614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 12), 'self', False)
        # Obtaining the member 'toolbar' of a type (line 402)
        toolbar_227615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 12), self_227614, 'toolbar')
        # Obtaining the member 'show' of a type (line 402)
        show_227616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 12), toolbar_227615, 'show')
        # Calling show(args, kwargs) (line 402)
        show_call_result_227618 = invoke(stypy.reporting.localization.Localization(__file__, 402, 12), show_227616, *[], **kwargs_227617)
        
        
        # Getting the type of 'h' (line 403)
        h_227619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'h')
        
        # Call to add_widget(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'self' (line 403)
        self_227621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 28), 'self', False)
        # Obtaining the member 'toolbar' of a type (line 403)
        toolbar_227622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 28), self_227621, 'toolbar')
        # Getting the type of 'False' (line 403)
        False_227623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 42), 'False', False)
        # Getting the type of 'False' (line 403)
        False_227624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 49), 'False', False)
        int_227625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 56), 'int')
        # Processing the call keyword arguments (line 403)
        kwargs_227626 = {}
        # Getting the type of 'add_widget' (line 403)
        add_widget_227620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 17), 'add_widget', False)
        # Calling add_widget(args, kwargs) (line 403)
        add_widget_call_result_227627 = invoke(stypy.reporting.localization.Localization(__file__, 403, 17), add_widget_227620, *[toolbar_227622, False_227623, False_227624, int_227625], **kwargs_227626)
        
        # Applying the binary operator '+=' (line 403)
        result_iadd_227628 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 12), '+=', h_227619, add_widget_call_result_227627)
        # Assigning a type to the variable 'h' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'h', result_iadd_227628)
        
        # SSA join for if statement (line 401)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_default_size(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'w' (line 405)
        w_227632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 38), 'w', False)
        # Getting the type of 'h' (line 405)
        h_227633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 41), 'h', False)
        # Processing the call keyword arguments (line 405)
        kwargs_227634 = {}
        # Getting the type of 'self' (line 405)
        self_227629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 405)
        window_227630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), self_227629, 'window')
        # Obtaining the member 'set_default_size' of a type (line 405)
        set_default_size_227631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 8), window_227630, 'set_default_size')
        # Calling set_default_size(args, kwargs) (line 405)
        set_default_size_call_result_227635 = invoke(stypy.reporting.localization.Localization(__file__, 405, 8), set_default_size_227631, *[w_227632, h_227633], **kwargs_227634)
        

        @norecursion
        def destroy(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'destroy'
            module_type_store = module_type_store.open_function_context('destroy', 407, 8, False)
            
            # Passed parameters checking function
            destroy.stypy_localization = localization
            destroy.stypy_type_of_self = None
            destroy.stypy_type_store = module_type_store
            destroy.stypy_function_name = 'destroy'
            destroy.stypy_param_names_list = []
            destroy.stypy_varargs_param_name = 'args'
            destroy.stypy_kwargs_param_name = None
            destroy.stypy_call_defaults = defaults
            destroy.stypy_call_varargs = varargs
            destroy.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'destroy', [], 'args', None, defaults, varargs, kwargs)

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

            
            # Call to destroy(...): (line 408)
            # Processing the call arguments (line 408)
            # Getting the type of 'num' (line 408)
            num_227638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 24), 'num', False)
            # Processing the call keyword arguments (line 408)
            kwargs_227639 = {}
            # Getting the type of 'Gcf' (line 408)
            Gcf_227636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'Gcf', False)
            # Obtaining the member 'destroy' of a type (line 408)
            destroy_227637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 12), Gcf_227636, 'destroy')
            # Calling destroy(args, kwargs) (line 408)
            destroy_call_result_227640 = invoke(stypy.reporting.localization.Localization(__file__, 408, 12), destroy_227637, *[num_227638], **kwargs_227639)
            
            
            # ################# End of 'destroy(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'destroy' in the type store
            # Getting the type of 'stypy_return_type' (line 407)
            stypy_return_type_227641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_227641)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'destroy'
            return stypy_return_type_227641

        # Assigning a type to the variable 'destroy' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'destroy', destroy)
        
        # Call to connect(...): (line 409)
        # Processing the call arguments (line 409)
        unicode_227645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 28), 'unicode', u'destroy')
        # Getting the type of 'destroy' (line 409)
        destroy_227646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 39), 'destroy', False)
        # Processing the call keyword arguments (line 409)
        kwargs_227647 = {}
        # Getting the type of 'self' (line 409)
        self_227642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 409)
        window_227643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), self_227642, 'window')
        # Obtaining the member 'connect' of a type (line 409)
        connect_227644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), window_227643, 'connect')
        # Calling connect(args, kwargs) (line 409)
        connect_call_result_227648 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), connect_227644, *[unicode_227645, destroy_227646], **kwargs_227647)
        
        
        # Call to connect(...): (line 410)
        # Processing the call arguments (line 410)
        unicode_227652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 28), 'unicode', u'delete_event')
        # Getting the type of 'destroy' (line 410)
        destroy_227653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 44), 'destroy', False)
        # Processing the call keyword arguments (line 410)
        kwargs_227654 = {}
        # Getting the type of 'self' (line 410)
        self_227649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 410)
        window_227650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 8), self_227649, 'window')
        # Obtaining the member 'connect' of a type (line 410)
        connect_227651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 8), window_227650, 'connect')
        # Calling connect(args, kwargs) (line 410)
        connect_call_result_227655 = invoke(stypy.reporting.localization.Localization(__file__, 410, 8), connect_227651, *[unicode_227652, destroy_227653], **kwargs_227654)
        
        
        
        # Call to is_interactive(...): (line 411)
        # Processing the call keyword arguments (line 411)
        kwargs_227658 = {}
        # Getting the type of 'matplotlib' (line 411)
        matplotlib_227656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 11), 'matplotlib', False)
        # Obtaining the member 'is_interactive' of a type (line 411)
        is_interactive_227657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 11), matplotlib_227656, 'is_interactive')
        # Calling is_interactive(args, kwargs) (line 411)
        is_interactive_call_result_227659 = invoke(stypy.reporting.localization.Localization(__file__, 411, 11), is_interactive_227657, *[], **kwargs_227658)
        
        # Testing the type of an if condition (line 411)
        if_condition_227660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 8), is_interactive_call_result_227659)
        # Assigning a type to the variable 'if_condition_227660' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'if_condition_227660', if_condition_227660)
        # SSA begins for if statement (line 411)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to show(...): (line 412)
        # Processing the call keyword arguments (line 412)
        kwargs_227664 = {}
        # Getting the type of 'self' (line 412)
        self_227661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'self', False)
        # Obtaining the member 'window' of a type (line 412)
        window_227662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 12), self_227661, 'window')
        # Obtaining the member 'show' of a type (line 412)
        show_227663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 12), window_227662, 'show')
        # Calling show(args, kwargs) (line 412)
        show_call_result_227665 = invoke(stypy.reporting.localization.Localization(__file__, 412, 12), show_227663, *[], **kwargs_227664)
        
        
        # Call to draw_idle(...): (line 413)
        # Processing the call keyword arguments (line 413)
        kwargs_227669 = {}
        # Getting the type of 'self' (line 413)
        self_227666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'self', False)
        # Obtaining the member 'canvas' of a type (line 413)
        canvas_227667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 12), self_227666, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 413)
        draw_idle_227668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 12), canvas_227667, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 413)
        draw_idle_call_result_227670 = invoke(stypy.reporting.localization.Localization(__file__, 413, 12), draw_idle_227668, *[], **kwargs_227669)
        
        # SSA join for if statement (line 411)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def notify_axes_change(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'notify_axes_change'
            module_type_store = module_type_store.open_function_context('notify_axes_change', 415, 8, False)
            
            # Passed parameters checking function
            notify_axes_change.stypy_localization = localization
            notify_axes_change.stypy_type_of_self = None
            notify_axes_change.stypy_type_store = module_type_store
            notify_axes_change.stypy_function_name = 'notify_axes_change'
            notify_axes_change.stypy_param_names_list = ['fig']
            notify_axes_change.stypy_varargs_param_name = None
            notify_axes_change.stypy_kwargs_param_name = None
            notify_axes_change.stypy_call_defaults = defaults
            notify_axes_change.stypy_call_varargs = varargs
            notify_axes_change.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'notify_axes_change', ['fig'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'notify_axes_change', localization, ['fig'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'notify_axes_change(...)' code ##################

            unicode_227671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 12), 'unicode', u'this will be called whenever the current axes is changed')
            
            
            # Getting the type of 'self' (line 417)
            self_227672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 15), 'self')
            # Obtaining the member 'toolmanager' of a type (line 417)
            toolmanager_227673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 15), self_227672, 'toolmanager')
            # Getting the type of 'None' (line 417)
            None_227674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 39), 'None')
            # Applying the binary operator 'isnot' (line 417)
            result_is_not_227675 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 15), 'isnot', toolmanager_227673, None_227674)
            
            # Testing the type of an if condition (line 417)
            if_condition_227676 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 417, 12), result_is_not_227675)
            # Assigning a type to the variable 'if_condition_227676' (line 417)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'if_condition_227676', if_condition_227676)
            # SSA begins for if statement (line 417)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA branch for the else part of an if statement (line 417)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'self' (line 419)
            self_227677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 17), 'self')
            # Obtaining the member 'toolbar' of a type (line 419)
            toolbar_227678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 17), self_227677, 'toolbar')
            # Getting the type of 'None' (line 419)
            None_227679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 37), 'None')
            # Applying the binary operator 'isnot' (line 419)
            result_is_not_227680 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 17), 'isnot', toolbar_227678, None_227679)
            
            # Testing the type of an if condition (line 419)
            if_condition_227681 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 17), result_is_not_227680)
            # Assigning a type to the variable 'if_condition_227681' (line 419)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 17), 'if_condition_227681', if_condition_227681)
            # SSA begins for if statement (line 419)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to update(...): (line 420)
            # Processing the call keyword arguments (line 420)
            kwargs_227685 = {}
            # Getting the type of 'self' (line 420)
            self_227682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 16), 'self', False)
            # Obtaining the member 'toolbar' of a type (line 420)
            toolbar_227683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 16), self_227682, 'toolbar')
            # Obtaining the member 'update' of a type (line 420)
            update_227684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 16), toolbar_227683, 'update')
            # Calling update(args, kwargs) (line 420)
            update_call_result_227686 = invoke(stypy.reporting.localization.Localization(__file__, 420, 16), update_227684, *[], **kwargs_227685)
            
            # SSA join for if statement (line 419)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 417)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'notify_axes_change(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'notify_axes_change' in the type store
            # Getting the type of 'stypy_return_type' (line 415)
            stypy_return_type_227687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_227687)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'notify_axes_change'
            return stypy_return_type_227687

        # Assigning a type to the variable 'notify_axes_change' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'notify_axes_change', notify_axes_change)
        
        # Call to add_axobserver(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'notify_axes_change' (line 421)
        notify_axes_change_227692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 42), 'notify_axes_change', False)
        # Processing the call keyword arguments (line 421)
        kwargs_227693 = {}
        # Getting the type of 'self' (line 421)
        self_227688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 421)
        canvas_227689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 8), self_227688, 'canvas')
        # Obtaining the member 'figure' of a type (line 421)
        figure_227690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 8), canvas_227689, 'figure')
        # Obtaining the member 'add_axobserver' of a type (line 421)
        add_axobserver_227691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 8), figure_227690, 'add_axobserver')
        # Calling add_axobserver(args, kwargs) (line 421)
        add_axobserver_call_result_227694 = invoke(stypy.reporting.localization.Localization(__file__, 421, 8), add_axobserver_227691, *[notify_axes_change_227692], **kwargs_227693)
        
        
        # Call to grab_focus(...): (line 423)
        # Processing the call keyword arguments (line 423)
        kwargs_227698 = {}
        # Getting the type of 'self' (line 423)
        self_227695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 423)
        canvas_227696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), self_227695, 'canvas')
        # Obtaining the member 'grab_focus' of a type (line 423)
        grab_focus_227697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), canvas_227696, 'grab_focus')
        # Calling grab_focus(args, kwargs) (line 423)
        grab_focus_call_result_227699 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), grab_focus_227697, *[], **kwargs_227698)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def destroy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'destroy'
        module_type_store = module_type_store.open_function_context('destroy', 425, 4, False)
        # Assigning a type to the variable 'self' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerGTK3.destroy.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerGTK3.destroy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerGTK3.destroy.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerGTK3.destroy.__dict__.__setitem__('stypy_function_name', 'FigureManagerGTK3.destroy')
        FigureManagerGTK3.destroy.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerGTK3.destroy.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureManagerGTK3.destroy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerGTK3.destroy.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerGTK3.destroy.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerGTK3.destroy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerGTK3.destroy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTK3.destroy', [], 'args', None, defaults, varargs, kwargs)

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

        
        # Call to destroy(...): (line 426)
        # Processing the call keyword arguments (line 426)
        kwargs_227703 = {}
        # Getting the type of 'self' (line 426)
        self_227700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'self', False)
        # Obtaining the member 'vbox' of a type (line 426)
        vbox_227701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 8), self_227700, 'vbox')
        # Obtaining the member 'destroy' of a type (line 426)
        destroy_227702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 8), vbox_227701, 'destroy')
        # Calling destroy(args, kwargs) (line 426)
        destroy_call_result_227704 = invoke(stypy.reporting.localization.Localization(__file__, 426, 8), destroy_227702, *[], **kwargs_227703)
        
        
        # Call to destroy(...): (line 427)
        # Processing the call keyword arguments (line 427)
        kwargs_227708 = {}
        # Getting the type of 'self' (line 427)
        self_227705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 427)
        window_227706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), self_227705, 'window')
        # Obtaining the member 'destroy' of a type (line 427)
        destroy_227707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), window_227706, 'destroy')
        # Calling destroy(args, kwargs) (line 427)
        destroy_call_result_227709 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), destroy_227707, *[], **kwargs_227708)
        
        
        # Call to destroy(...): (line 428)
        # Processing the call keyword arguments (line 428)
        kwargs_227713 = {}
        # Getting the type of 'self' (line 428)
        self_227710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 428)
        canvas_227711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), self_227710, 'canvas')
        # Obtaining the member 'destroy' of a type (line 428)
        destroy_227712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), canvas_227711, 'destroy')
        # Calling destroy(args, kwargs) (line 428)
        destroy_call_result_227714 = invoke(stypy.reporting.localization.Localization(__file__, 428, 8), destroy_227712, *[], **kwargs_227713)
        
        
        # Getting the type of 'self' (line 429)
        self_227715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 11), 'self')
        # Obtaining the member 'toolbar' of a type (line 429)
        toolbar_227716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 11), self_227715, 'toolbar')
        # Testing the type of an if condition (line 429)
        if_condition_227717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 429, 8), toolbar_227716)
        # Assigning a type to the variable 'if_condition_227717' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'if_condition_227717', if_condition_227717)
        # SSA begins for if statement (line 429)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to destroy(...): (line 430)
        # Processing the call keyword arguments (line 430)
        kwargs_227721 = {}
        # Getting the type of 'self' (line 430)
        self_227718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 12), 'self', False)
        # Obtaining the member 'toolbar' of a type (line 430)
        toolbar_227719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), self_227718, 'toolbar')
        # Obtaining the member 'destroy' of a type (line 430)
        destroy_227720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 12), toolbar_227719, 'destroy')
        # Calling destroy(args, kwargs) (line 430)
        destroy_call_result_227722 = invoke(stypy.reporting.localization.Localization(__file__, 430, 12), destroy_227720, *[], **kwargs_227721)
        
        # SSA join for if statement (line 429)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Call to get_num_fig_managers(...): (line 432)
        # Processing the call keyword arguments (line 432)
        kwargs_227725 = {}
        # Getting the type of 'Gcf' (line 432)
        Gcf_227723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'Gcf', False)
        # Obtaining the member 'get_num_fig_managers' of a type (line 432)
        get_num_fig_managers_227724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 12), Gcf_227723, 'get_num_fig_managers')
        # Calling get_num_fig_managers(args, kwargs) (line 432)
        get_num_fig_managers_call_result_227726 = invoke(stypy.reporting.localization.Localization(__file__, 432, 12), get_num_fig_managers_227724, *[], **kwargs_227725)
        
        int_227727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 42), 'int')
        # Applying the binary operator '==' (line 432)
        result_eq_227728 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 12), '==', get_num_fig_managers_call_result_227726, int_227727)
        
        
        
        # Call to is_interactive(...): (line 433)
        # Processing the call keyword arguments (line 433)
        kwargs_227731 = {}
        # Getting the type of 'matplotlib' (line 433)
        matplotlib_227729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'matplotlib', False)
        # Obtaining the member 'is_interactive' of a type (line 433)
        is_interactive_227730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 20), matplotlib_227729, 'is_interactive')
        # Calling is_interactive(args, kwargs) (line 433)
        is_interactive_call_result_227732 = invoke(stypy.reporting.localization.Localization(__file__, 433, 20), is_interactive_227730, *[], **kwargs_227731)
        
        # Applying the 'not' unary operator (line 433)
        result_not__227733 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 16), 'not', is_interactive_call_result_227732)
        
        # Applying the binary operator 'and' (line 432)
        result_and_keyword_227734 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 12), 'and', result_eq_227728, result_not__227733)
        
        
        # Call to main_level(...): (line 434)
        # Processing the call keyword arguments (line 434)
        kwargs_227737 = {}
        # Getting the type of 'Gtk' (line 434)
        Gtk_227735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'Gtk', False)
        # Obtaining the member 'main_level' of a type (line 434)
        main_level_227736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 16), Gtk_227735, 'main_level')
        # Calling main_level(args, kwargs) (line 434)
        main_level_call_result_227738 = invoke(stypy.reporting.localization.Localization(__file__, 434, 16), main_level_227736, *[], **kwargs_227737)
        
        int_227739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 36), 'int')
        # Applying the binary operator '>=' (line 434)
        result_ge_227740 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 16), '>=', main_level_call_result_227738, int_227739)
        
        # Applying the binary operator 'and' (line 432)
        result_and_keyword_227741 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 12), 'and', result_and_keyword_227734, result_ge_227740)
        
        # Testing the type of an if condition (line 432)
        if_condition_227742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 8), result_and_keyword_227741)
        # Assigning a type to the variable 'if_condition_227742' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'if_condition_227742', if_condition_227742)
        # SSA begins for if statement (line 432)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to main_quit(...): (line 435)
        # Processing the call keyword arguments (line 435)
        kwargs_227745 = {}
        # Getting the type of 'Gtk' (line 435)
        Gtk_227743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'Gtk', False)
        # Obtaining the member 'main_quit' of a type (line 435)
        main_quit_227744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 12), Gtk_227743, 'main_quit')
        # Calling main_quit(args, kwargs) (line 435)
        main_quit_call_result_227746 = invoke(stypy.reporting.localization.Localization(__file__, 435, 12), main_quit_227744, *[], **kwargs_227745)
        
        # SSA join for if statement (line 432)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'destroy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'destroy' in the type store
        # Getting the type of 'stypy_return_type' (line 425)
        stypy_return_type_227747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227747)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'destroy'
        return stypy_return_type_227747


    @norecursion
    def show(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'show'
        module_type_store = module_type_store.open_function_context('show', 437, 4, False)
        # Assigning a type to the variable 'self' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerGTK3.show.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerGTK3.show.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerGTK3.show.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerGTK3.show.__dict__.__setitem__('stypy_function_name', 'FigureManagerGTK3.show')
        FigureManagerGTK3.show.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerGTK3.show.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerGTK3.show.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerGTK3.show.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerGTK3.show.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerGTK3.show.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerGTK3.show.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTK3.show', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'show', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'show(...)' code ##################

        
        # Call to show(...): (line 439)
        # Processing the call keyword arguments (line 439)
        kwargs_227751 = {}
        # Getting the type of 'self' (line 439)
        self_227748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 439)
        window_227749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 8), self_227748, 'window')
        # Obtaining the member 'show' of a type (line 439)
        show_227750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 8), window_227749, 'show')
        # Calling show(args, kwargs) (line 439)
        show_call_result_227752 = invoke(stypy.reporting.localization.Localization(__file__, 439, 8), show_227750, *[], **kwargs_227751)
        
        
        # Call to present(...): (line 440)
        # Processing the call keyword arguments (line 440)
        kwargs_227756 = {}
        # Getting the type of 'self' (line 440)
        self_227753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 440)
        window_227754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 8), self_227753, 'window')
        # Obtaining the member 'present' of a type (line 440)
        present_227755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 8), window_227754, 'present')
        # Calling present(args, kwargs) (line 440)
        present_call_result_227757 = invoke(stypy.reporting.localization.Localization(__file__, 440, 8), present_227755, *[], **kwargs_227756)
        
        
        # ################# End of 'show(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'show' in the type store
        # Getting the type of 'stypy_return_type' (line 437)
        stypy_return_type_227758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227758)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'show'
        return stypy_return_type_227758


    @norecursion
    def full_screen_toggle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'full_screen_toggle'
        module_type_store = module_type_store.open_function_context('full_screen_toggle', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerGTK3.full_screen_toggle.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerGTK3.full_screen_toggle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerGTK3.full_screen_toggle.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerGTK3.full_screen_toggle.__dict__.__setitem__('stypy_function_name', 'FigureManagerGTK3.full_screen_toggle')
        FigureManagerGTK3.full_screen_toggle.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerGTK3.full_screen_toggle.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerGTK3.full_screen_toggle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerGTK3.full_screen_toggle.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerGTK3.full_screen_toggle.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerGTK3.full_screen_toggle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerGTK3.full_screen_toggle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTK3.full_screen_toggle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'full_screen_toggle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'full_screen_toggle(...)' code ##################

        
        # Assigning a UnaryOp to a Attribute (line 443):
        
        # Assigning a UnaryOp to a Attribute (line 443):
        
        # Getting the type of 'self' (line 443)
        self_227759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 37), 'self')
        # Obtaining the member '_full_screen_flag' of a type (line 443)
        _full_screen_flag_227760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 37), self_227759, '_full_screen_flag')
        # Applying the 'not' unary operator (line 443)
        result_not__227761 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 33), 'not', _full_screen_flag_227760)
        
        # Getting the type of 'self' (line 443)
        self_227762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'self')
        # Setting the type of the member '_full_screen_flag' of a type (line 443)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), self_227762, '_full_screen_flag', result_not__227761)
        
        # Getting the type of 'self' (line 444)
        self_227763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 11), 'self')
        # Obtaining the member '_full_screen_flag' of a type (line 444)
        _full_screen_flag_227764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 11), self_227763, '_full_screen_flag')
        # Testing the type of an if condition (line 444)
        if_condition_227765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 444, 8), _full_screen_flag_227764)
        # Assigning a type to the variable 'if_condition_227765' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'if_condition_227765', if_condition_227765)
        # SSA begins for if statement (line 444)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to fullscreen(...): (line 445)
        # Processing the call keyword arguments (line 445)
        kwargs_227769 = {}
        # Getting the type of 'self' (line 445)
        self_227766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'self', False)
        # Obtaining the member 'window' of a type (line 445)
        window_227767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 12), self_227766, 'window')
        # Obtaining the member 'fullscreen' of a type (line 445)
        fullscreen_227768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 12), window_227767, 'fullscreen')
        # Calling fullscreen(args, kwargs) (line 445)
        fullscreen_call_result_227770 = invoke(stypy.reporting.localization.Localization(__file__, 445, 12), fullscreen_227768, *[], **kwargs_227769)
        
        # SSA branch for the else part of an if statement (line 444)
        module_type_store.open_ssa_branch('else')
        
        # Call to unfullscreen(...): (line 447)
        # Processing the call keyword arguments (line 447)
        kwargs_227774 = {}
        # Getting the type of 'self' (line 447)
        self_227771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'self', False)
        # Obtaining the member 'window' of a type (line 447)
        window_227772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 12), self_227771, 'window')
        # Obtaining the member 'unfullscreen' of a type (line 447)
        unfullscreen_227773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 12), window_227772, 'unfullscreen')
        # Calling unfullscreen(args, kwargs) (line 447)
        unfullscreen_call_result_227775 = invoke(stypy.reporting.localization.Localization(__file__, 447, 12), unfullscreen_227773, *[], **kwargs_227774)
        
        # SSA join for if statement (line 444)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'full_screen_toggle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'full_screen_toggle' in the type store
        # Getting the type of 'stypy_return_type' (line 442)
        stypy_return_type_227776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227776)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'full_screen_toggle'
        return stypy_return_type_227776

    
    # Assigning a Name to a Name (line 448):

    @norecursion
    def _get_toolbar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_toolbar'
        module_type_store = module_type_store.open_function_context('_get_toolbar', 450, 4, False)
        # Assigning a type to the variable 'self' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerGTK3._get_toolbar.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerGTK3._get_toolbar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerGTK3._get_toolbar.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerGTK3._get_toolbar.__dict__.__setitem__('stypy_function_name', 'FigureManagerGTK3._get_toolbar')
        FigureManagerGTK3._get_toolbar.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerGTK3._get_toolbar.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerGTK3._get_toolbar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerGTK3._get_toolbar.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerGTK3._get_toolbar.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerGTK3._get_toolbar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerGTK3._get_toolbar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTK3._get_toolbar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_toolbar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_toolbar(...)' code ##################

        
        
        
        # Obtaining the type of the subscript
        unicode_227777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 20), 'unicode', u'toolbar')
        # Getting the type of 'rcParams' (line 453)
        rcParams_227778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 453)
        getitem___227779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 11), rcParams_227778, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 453)
        subscript_call_result_227780 = invoke(stypy.reporting.localization.Localization(__file__, 453, 11), getitem___227779, unicode_227777)
        
        unicode_227781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 34), 'unicode', u'toolbar2')
        # Applying the binary operator '==' (line 453)
        result_eq_227782 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 11), '==', subscript_call_result_227780, unicode_227781)
        
        # Testing the type of an if condition (line 453)
        if_condition_227783 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 8), result_eq_227782)
        # Assigning a type to the variable 'if_condition_227783' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'if_condition_227783', if_condition_227783)
        # SSA begins for if statement (line 453)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 454):
        
        # Assigning a Call to a Name (line 454):
        
        # Call to NavigationToolbar2GTK3(...): (line 454)
        # Processing the call arguments (line 454)
        # Getting the type of 'self' (line 454)
        self_227785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 45), 'self', False)
        # Obtaining the member 'canvas' of a type (line 454)
        canvas_227786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 45), self_227785, 'canvas')
        # Getting the type of 'self' (line 454)
        self_227787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 58), 'self', False)
        # Obtaining the member 'window' of a type (line 454)
        window_227788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 58), self_227787, 'window')
        # Processing the call keyword arguments (line 454)
        kwargs_227789 = {}
        # Getting the type of 'NavigationToolbar2GTK3' (line 454)
        NavigationToolbar2GTK3_227784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 22), 'NavigationToolbar2GTK3', False)
        # Calling NavigationToolbar2GTK3(args, kwargs) (line 454)
        NavigationToolbar2GTK3_call_result_227790 = invoke(stypy.reporting.localization.Localization(__file__, 454, 22), NavigationToolbar2GTK3_227784, *[canvas_227786, window_227788], **kwargs_227789)
        
        # Assigning a type to the variable 'toolbar' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'toolbar', NavigationToolbar2GTK3_call_result_227790)
        # SSA branch for the else part of an if statement (line 453)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        unicode_227791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 22), 'unicode', u'toolbar')
        # Getting the type of 'rcParams' (line 455)
        rcParams_227792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 13), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 455)
        getitem___227793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 13), rcParams_227792, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 455)
        subscript_call_result_227794 = invoke(stypy.reporting.localization.Localization(__file__, 455, 13), getitem___227793, unicode_227791)
        
        unicode_227795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 36), 'unicode', u'toolmanager')
        # Applying the binary operator '==' (line 455)
        result_eq_227796 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 13), '==', subscript_call_result_227794, unicode_227795)
        
        # Testing the type of an if condition (line 455)
        if_condition_227797 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 13), result_eq_227796)
        # Assigning a type to the variable 'if_condition_227797' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 13), 'if_condition_227797', if_condition_227797)
        # SSA begins for if statement (line 455)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 456):
        
        # Assigning a Call to a Name (line 456):
        
        # Call to ToolbarGTK3(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'self' (line 456)
        self_227799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 34), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 456)
        toolmanager_227800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 34), self_227799, 'toolmanager')
        # Processing the call keyword arguments (line 456)
        kwargs_227801 = {}
        # Getting the type of 'ToolbarGTK3' (line 456)
        ToolbarGTK3_227798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 22), 'ToolbarGTK3', False)
        # Calling ToolbarGTK3(args, kwargs) (line 456)
        ToolbarGTK3_call_result_227802 = invoke(stypy.reporting.localization.Localization(__file__, 456, 22), ToolbarGTK3_227798, *[toolmanager_227800], **kwargs_227801)
        
        # Assigning a type to the variable 'toolbar' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'toolbar', ToolbarGTK3_call_result_227802)
        # SSA branch for the else part of an if statement (line 455)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 458):
        
        # Assigning a Name to a Name (line 458):
        # Getting the type of 'None' (line 458)
        None_227803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 22), 'None')
        # Assigning a type to the variable 'toolbar' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'toolbar', None_227803)
        # SSA join for if statement (line 455)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 453)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'toolbar' (line 459)
        toolbar_227804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'toolbar')
        # Assigning a type to the variable 'stypy_return_type' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'stypy_return_type', toolbar_227804)
        
        # ################# End of '_get_toolbar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_toolbar' in the type store
        # Getting the type of 'stypy_return_type' (line 450)
        stypy_return_type_227805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227805)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_toolbar'
        return stypy_return_type_227805


    @norecursion
    def _get_toolmanager(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_toolmanager'
        module_type_store = module_type_store.open_function_context('_get_toolmanager', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerGTK3._get_toolmanager.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerGTK3._get_toolmanager.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerGTK3._get_toolmanager.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerGTK3._get_toolmanager.__dict__.__setitem__('stypy_function_name', 'FigureManagerGTK3._get_toolmanager')
        FigureManagerGTK3._get_toolmanager.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerGTK3._get_toolmanager.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerGTK3._get_toolmanager.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerGTK3._get_toolmanager.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerGTK3._get_toolmanager.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerGTK3._get_toolmanager.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerGTK3._get_toolmanager.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTK3._get_toolmanager', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_toolmanager', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_toolmanager(...)' code ##################

        
        
        
        # Obtaining the type of the subscript
        unicode_227806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 20), 'unicode', u'toolbar')
        # Getting the type of 'rcParams' (line 463)
        rcParams_227807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 463)
        getitem___227808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 11), rcParams_227807, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 463)
        subscript_call_result_227809 = invoke(stypy.reporting.localization.Localization(__file__, 463, 11), getitem___227808, unicode_227806)
        
        unicode_227810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 34), 'unicode', u'toolmanager')
        # Applying the binary operator '==' (line 463)
        result_eq_227811 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 11), '==', subscript_call_result_227809, unicode_227810)
        
        # Testing the type of an if condition (line 463)
        if_condition_227812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 463, 8), result_eq_227811)
        # Assigning a type to the variable 'if_condition_227812' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'if_condition_227812', if_condition_227812)
        # SSA begins for if statement (line 463)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 464):
        
        # Assigning a Call to a Name (line 464):
        
        # Call to ToolManager(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'self' (line 464)
        self_227814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 38), 'self', False)
        # Obtaining the member 'canvas' of a type (line 464)
        canvas_227815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 38), self_227814, 'canvas')
        # Obtaining the member 'figure' of a type (line 464)
        figure_227816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 38), canvas_227815, 'figure')
        # Processing the call keyword arguments (line 464)
        kwargs_227817 = {}
        # Getting the type of 'ToolManager' (line 464)
        ToolManager_227813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 26), 'ToolManager', False)
        # Calling ToolManager(args, kwargs) (line 464)
        ToolManager_call_result_227818 = invoke(stypy.reporting.localization.Localization(__file__, 464, 26), ToolManager_227813, *[figure_227816], **kwargs_227817)
        
        # Assigning a type to the variable 'toolmanager' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'toolmanager', ToolManager_call_result_227818)
        # SSA branch for the else part of an if statement (line 463)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 466):
        
        # Assigning a Name to a Name (line 466):
        # Getting the type of 'None' (line 466)
        None_227819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 26), 'None')
        # Assigning a type to the variable 'toolmanager' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'toolmanager', None_227819)
        # SSA join for if statement (line 463)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'toolmanager' (line 467)
        toolmanager_227820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 15), 'toolmanager')
        # Assigning a type to the variable 'stypy_return_type' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'stypy_return_type', toolmanager_227820)
        
        # ################# End of '_get_toolmanager(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_toolmanager' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_227821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227821)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_toolmanager'
        return stypy_return_type_227821


    @norecursion
    def get_window_title(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_window_title'
        module_type_store = module_type_store.open_function_context('get_window_title', 469, 4, False)
        # Assigning a type to the variable 'self' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerGTK3.get_window_title.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerGTK3.get_window_title.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerGTK3.get_window_title.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerGTK3.get_window_title.__dict__.__setitem__('stypy_function_name', 'FigureManagerGTK3.get_window_title')
        FigureManagerGTK3.get_window_title.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerGTK3.get_window_title.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerGTK3.get_window_title.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerGTK3.get_window_title.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerGTK3.get_window_title.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerGTK3.get_window_title.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerGTK3.get_window_title.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTK3.get_window_title', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_window_title', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_window_title(...)' code ##################

        
        # Call to get_title(...): (line 470)
        # Processing the call keyword arguments (line 470)
        kwargs_227825 = {}
        # Getting the type of 'self' (line 470)
        self_227822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 15), 'self', False)
        # Obtaining the member 'window' of a type (line 470)
        window_227823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 15), self_227822, 'window')
        # Obtaining the member 'get_title' of a type (line 470)
        get_title_227824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 15), window_227823, 'get_title')
        # Calling get_title(args, kwargs) (line 470)
        get_title_call_result_227826 = invoke(stypy.reporting.localization.Localization(__file__, 470, 15), get_title_227824, *[], **kwargs_227825)
        
        # Assigning a type to the variable 'stypy_return_type' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'stypy_return_type', get_title_call_result_227826)
        
        # ################# End of 'get_window_title(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_window_title' in the type store
        # Getting the type of 'stypy_return_type' (line 469)
        stypy_return_type_227827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227827)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_window_title'
        return stypy_return_type_227827


    @norecursion
    def set_window_title(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_window_title'
        module_type_store = module_type_store.open_function_context('set_window_title', 472, 4, False)
        # Assigning a type to the variable 'self' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerGTK3.set_window_title.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerGTK3.set_window_title.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerGTK3.set_window_title.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerGTK3.set_window_title.__dict__.__setitem__('stypy_function_name', 'FigureManagerGTK3.set_window_title')
        FigureManagerGTK3.set_window_title.__dict__.__setitem__('stypy_param_names_list', ['title'])
        FigureManagerGTK3.set_window_title.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerGTK3.set_window_title.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerGTK3.set_window_title.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerGTK3.set_window_title.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerGTK3.set_window_title.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerGTK3.set_window_title.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTK3.set_window_title', ['title'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_window_title', localization, ['title'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_window_title(...)' code ##################

        
        # Call to set_title(...): (line 473)
        # Processing the call arguments (line 473)
        # Getting the type of 'title' (line 473)
        title_227831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 30), 'title', False)
        # Processing the call keyword arguments (line 473)
        kwargs_227832 = {}
        # Getting the type of 'self' (line 473)
        self_227828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 473)
        window_227829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 8), self_227828, 'window')
        # Obtaining the member 'set_title' of a type (line 473)
        set_title_227830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 8), window_227829, 'set_title')
        # Calling set_title(args, kwargs) (line 473)
        set_title_call_result_227833 = invoke(stypy.reporting.localization.Localization(__file__, 473, 8), set_title_227830, *[title_227831], **kwargs_227832)
        
        
        # ################# End of 'set_window_title(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_window_title' in the type store
        # Getting the type of 'stypy_return_type' (line 472)
        stypy_return_type_227834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227834)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_window_title'
        return stypy_return_type_227834


    @norecursion
    def resize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'resize'
        module_type_store = module_type_store.open_function_context('resize', 475, 4, False)
        # Assigning a type to the variable 'self' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerGTK3.resize.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerGTK3.resize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerGTK3.resize.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerGTK3.resize.__dict__.__setitem__('stypy_function_name', 'FigureManagerGTK3.resize')
        FigureManagerGTK3.resize.__dict__.__setitem__('stypy_param_names_list', ['width', 'height'])
        FigureManagerGTK3.resize.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerGTK3.resize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerGTK3.resize.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerGTK3.resize.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerGTK3.resize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerGTK3.resize.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerGTK3.resize', ['width', 'height'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'resize', localization, ['width', 'height'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'resize(...)' code ##################

        unicode_227835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 8), 'unicode', u'set the canvas size in pixels')
        
        # Call to resize(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'width' (line 480)
        width_227839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 27), 'width', False)
        # Getting the type of 'height' (line 480)
        height_227840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 34), 'height', False)
        # Processing the call keyword arguments (line 480)
        kwargs_227841 = {}
        # Getting the type of 'self' (line 480)
        self_227836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 480)
        window_227837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), self_227836, 'window')
        # Obtaining the member 'resize' of a type (line 480)
        resize_227838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), window_227837, 'resize')
        # Calling resize(args, kwargs) (line 480)
        resize_call_result_227842 = invoke(stypy.reporting.localization.Localization(__file__, 480, 8), resize_227838, *[width_227839, height_227840], **kwargs_227841)
        
        
        # ################# End of 'resize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'resize' in the type store
        # Getting the type of 'stypy_return_type' (line 475)
        stypy_return_type_227843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227843)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'resize'
        return stypy_return_type_227843


# Assigning a type to the variable 'FigureManagerGTK3' (line 337)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 0), 'FigureManagerGTK3', FigureManagerGTK3)

# Assigning a Name to a Name (line 448):
# Getting the type of 'False' (line 448)
False_227844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 24), 'False')
# Getting the type of 'FigureManagerGTK3'
FigureManagerGTK3_227845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureManagerGTK3')
# Setting the type of the member '_full_screen_flag' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureManagerGTK3_227845, '_full_screen_flag', False_227844)
# Declaration of the 'NavigationToolbar2GTK3' class
# Getting the type of 'NavigationToolbar2' (line 483)
NavigationToolbar2_227846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 29), 'NavigationToolbar2')
# Getting the type of 'Gtk' (line 483)
Gtk_227847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 49), 'Gtk')
# Obtaining the member 'Toolbar' of a type (line 483)
Toolbar_227848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 49), Gtk_227847, 'Toolbar')

class NavigationToolbar2GTK3(NavigationToolbar2_227846, Toolbar_227848, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 484, 4, False)
        # Assigning a type to the variable 'self' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTK3.__init__', ['canvas', 'window'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['canvas', 'window'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 485):
        
        # Assigning a Name to a Attribute (line 485):
        # Getting the type of 'window' (line 485)
        window_227849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 19), 'window')
        # Getting the type of 'self' (line 485)
        self_227850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'self')
        # Setting the type of the member 'win' of a type (line 485)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 8), self_227850, 'win', window_227849)
        
        # Call to __init__(...): (line 486)
        # Processing the call arguments (line 486)
        # Getting the type of 'self' (line 486)
        self_227854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 33), 'self', False)
        # Processing the call keyword arguments (line 486)
        kwargs_227855 = {}
        # Getting the type of 'GObject' (line 486)
        GObject_227851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'GObject', False)
        # Obtaining the member 'GObject' of a type (line 486)
        GObject_227852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 8), GObject_227851, 'GObject')
        # Obtaining the member '__init__' of a type (line 486)
        init___227853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 8), GObject_227852, '__init__')
        # Calling __init__(args, kwargs) (line 486)
        init___call_result_227856 = invoke(stypy.reporting.localization.Localization(__file__, 486, 8), init___227853, *[self_227854], **kwargs_227855)
        
        
        # Call to __init__(...): (line 487)
        # Processing the call arguments (line 487)
        # Getting the type of 'self' (line 487)
        self_227859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 36), 'self', False)
        # Getting the type of 'canvas' (line 487)
        canvas_227860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 42), 'canvas', False)
        # Processing the call keyword arguments (line 487)
        kwargs_227861 = {}
        # Getting the type of 'NavigationToolbar2' (line 487)
        NavigationToolbar2_227857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'NavigationToolbar2', False)
        # Obtaining the member '__init__' of a type (line 487)
        init___227858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 8), NavigationToolbar2_227857, '__init__')
        # Calling __init__(args, kwargs) (line 487)
        init___call_result_227862 = invoke(stypy.reporting.localization.Localization(__file__, 487, 8), init___227858, *[self_227859, canvas_227860], **kwargs_227861)
        
        
        # Assigning a Name to a Attribute (line 488):
        
        # Assigning a Name to a Attribute (line 488):
        # Getting the type of 'None' (line 488)
        None_227863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 19), 'None')
        # Getting the type of 'self' (line 488)
        self_227864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'self')
        # Setting the type of the member 'ctx' of a type (line 488)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 8), self_227864, 'ctx', None_227863)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_message'
        module_type_store = module_type_store.open_function_context('set_message', 490, 4, False)
        # Assigning a type to the variable 'self' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2GTK3.set_message.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2GTK3.set_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2GTK3.set_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2GTK3.set_message.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2GTK3.set_message')
        NavigationToolbar2GTK3.set_message.__dict__.__setitem__('stypy_param_names_list', ['s'])
        NavigationToolbar2GTK3.set_message.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2GTK3.set_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2GTK3.set_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2GTK3.set_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2GTK3.set_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2GTK3.set_message.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTK3.set_message', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_message', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_message(...)' code ##################

        
        # Call to set_label(...): (line 491)
        # Processing the call arguments (line 491)
        # Getting the type of 's' (line 491)
        s_227868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 31), 's', False)
        # Processing the call keyword arguments (line 491)
        kwargs_227869 = {}
        # Getting the type of 'self' (line 491)
        self_227865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'self', False)
        # Obtaining the member 'message' of a type (line 491)
        message_227866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 8), self_227865, 'message')
        # Obtaining the member 'set_label' of a type (line 491)
        set_label_227867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 8), message_227866, 'set_label')
        # Calling set_label(args, kwargs) (line 491)
        set_label_call_result_227870 = invoke(stypy.reporting.localization.Localization(__file__, 491, 8), set_label_227867, *[s_227868], **kwargs_227869)
        
        
        # ################# End of 'set_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_message' in the type store
        # Getting the type of 'stypy_return_type' (line 490)
        stypy_return_type_227871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227871)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_message'
        return stypy_return_type_227871


    @norecursion
    def set_cursor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_cursor'
        module_type_store = module_type_store.open_function_context('set_cursor', 493, 4, False)
        # Assigning a type to the variable 'self' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2GTK3.set_cursor.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2GTK3.set_cursor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2GTK3.set_cursor.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2GTK3.set_cursor.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2GTK3.set_cursor')
        NavigationToolbar2GTK3.set_cursor.__dict__.__setitem__('stypy_param_names_list', ['cursor'])
        NavigationToolbar2GTK3.set_cursor.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2GTK3.set_cursor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2GTK3.set_cursor.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2GTK3.set_cursor.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2GTK3.set_cursor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2GTK3.set_cursor.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTK3.set_cursor', ['cursor'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_cursor(...): (line 494)
        # Processing the call arguments (line 494)
        
        # Obtaining the type of the subscript
        # Getting the type of 'cursor' (line 494)
        cursor_227879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 62), 'cursor', False)
        # Getting the type of 'cursord' (line 494)
        cursord_227880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 54), 'cursord', False)
        # Obtaining the member '__getitem__' of a type (line 494)
        getitem___227881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 54), cursord_227880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 494)
        subscript_call_result_227882 = invoke(stypy.reporting.localization.Localization(__file__, 494, 54), getitem___227881, cursor_227879)
        
        # Processing the call keyword arguments (line 494)
        kwargs_227883 = {}
        
        # Call to get_property(...): (line 494)
        # Processing the call arguments (line 494)
        unicode_227875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 33), 'unicode', u'window')
        # Processing the call keyword arguments (line 494)
        kwargs_227876 = {}
        # Getting the type of 'self' (line 494)
        self_227872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 494)
        canvas_227873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 8), self_227872, 'canvas')
        # Obtaining the member 'get_property' of a type (line 494)
        get_property_227874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 8), canvas_227873, 'get_property')
        # Calling get_property(args, kwargs) (line 494)
        get_property_call_result_227877 = invoke(stypy.reporting.localization.Localization(__file__, 494, 8), get_property_227874, *[unicode_227875], **kwargs_227876)
        
        # Obtaining the member 'set_cursor' of a type (line 494)
        set_cursor_227878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 8), get_property_call_result_227877, 'set_cursor')
        # Calling set_cursor(args, kwargs) (line 494)
        set_cursor_call_result_227884 = invoke(stypy.reporting.localization.Localization(__file__, 494, 8), set_cursor_227878, *[subscript_call_result_227882], **kwargs_227883)
        
        
        # Call to main_iteration(...): (line 495)
        # Processing the call keyword arguments (line 495)
        kwargs_227887 = {}
        # Getting the type of 'Gtk' (line 495)
        Gtk_227885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'Gtk', False)
        # Obtaining the member 'main_iteration' of a type (line 495)
        main_iteration_227886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 8), Gtk_227885, 'main_iteration')
        # Calling main_iteration(args, kwargs) (line 495)
        main_iteration_call_result_227888 = invoke(stypy.reporting.localization.Localization(__file__, 495, 8), main_iteration_227886, *[], **kwargs_227887)
        
        
        # ################# End of 'set_cursor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_cursor' in the type store
        # Getting the type of 'stypy_return_type' (line 493)
        stypy_return_type_227889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227889)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_cursor'
        return stypy_return_type_227889


    @norecursion
    def release(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'release'
        module_type_store = module_type_store.open_function_context('release', 497, 4, False)
        # Assigning a type to the variable 'self' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2GTK3.release.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2GTK3.release.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2GTK3.release.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2GTK3.release.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2GTK3.release')
        NavigationToolbar2GTK3.release.__dict__.__setitem__('stypy_param_names_list', ['event'])
        NavigationToolbar2GTK3.release.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2GTK3.release.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2GTK3.release.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2GTK3.release.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2GTK3.release.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2GTK3.release.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTK3.release', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'release', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'release(...)' code ##################

        
        
        # SSA begins for try-except statement (line 498)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Deleting a member
        # Getting the type of 'self' (line 498)
        self_227890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 13), 'self')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 498, 13), self_227890, '_pixmapBack')
        # SSA branch for the except part of a try statement (line 498)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 498)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 498)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'release(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'release' in the type store
        # Getting the type of 'stypy_return_type' (line 497)
        stypy_return_type_227891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227891)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'release'
        return stypy_return_type_227891


    @norecursion
    def draw_rubberband(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_rubberband'
        module_type_store = module_type_store.open_function_context('draw_rubberband', 501, 4, False)
        # Assigning a type to the variable 'self' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2GTK3.draw_rubberband.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2GTK3.draw_rubberband.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2GTK3.draw_rubberband.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2GTK3.draw_rubberband.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2GTK3.draw_rubberband')
        NavigationToolbar2GTK3.draw_rubberband.__dict__.__setitem__('stypy_param_names_list', ['event', 'x0', 'y0', 'x1', 'y1'])
        NavigationToolbar2GTK3.draw_rubberband.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2GTK3.draw_rubberband.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2GTK3.draw_rubberband.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2GTK3.draw_rubberband.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2GTK3.draw_rubberband.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2GTK3.draw_rubberband.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTK3.draw_rubberband', ['event', 'x0', 'y0', 'x1', 'y1'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_rubberband', localization, ['event', 'x0', 'y0', 'x1', 'y1'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_rubberband(...)' code ##################

        unicode_227892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 8), 'unicode', u'adapted from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/189744')
        
        # Assigning a Call to a Attribute (line 503):
        
        # Assigning a Call to a Attribute (line 503):
        
        # Call to cairo_create(...): (line 503)
        # Processing the call keyword arguments (line 503)
        kwargs_227900 = {}
        
        # Call to get_property(...): (line 503)
        # Processing the call arguments (line 503)
        unicode_227896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 44), 'unicode', u'window')
        # Processing the call keyword arguments (line 503)
        kwargs_227897 = {}
        # Getting the type of 'self' (line 503)
        self_227893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 19), 'self', False)
        # Obtaining the member 'canvas' of a type (line 503)
        canvas_227894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 19), self_227893, 'canvas')
        # Obtaining the member 'get_property' of a type (line 503)
        get_property_227895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 19), canvas_227894, 'get_property')
        # Calling get_property(args, kwargs) (line 503)
        get_property_call_result_227898 = invoke(stypy.reporting.localization.Localization(__file__, 503, 19), get_property_227895, *[unicode_227896], **kwargs_227897)
        
        # Obtaining the member 'cairo_create' of a type (line 503)
        cairo_create_227899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 19), get_property_call_result_227898, 'cairo_create')
        # Calling cairo_create(args, kwargs) (line 503)
        cairo_create_call_result_227901 = invoke(stypy.reporting.localization.Localization(__file__, 503, 19), cairo_create_227899, *[], **kwargs_227900)
        
        # Getting the type of 'self' (line 503)
        self_227902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'self')
        # Setting the type of the member 'ctx' of a type (line 503)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), self_227902, 'ctx', cairo_create_call_result_227901)
        
        # Call to draw(...): (line 507)
        # Processing the call keyword arguments (line 507)
        kwargs_227906 = {}
        # Getting the type of 'self' (line 507)
        self_227903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 507)
        canvas_227904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 8), self_227903, 'canvas')
        # Obtaining the member 'draw' of a type (line 507)
        draw_227905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 8), canvas_227904, 'draw')
        # Calling draw(args, kwargs) (line 507)
        draw_call_result_227907 = invoke(stypy.reporting.localization.Localization(__file__, 507, 8), draw_227905, *[], **kwargs_227906)
        
        
        # Assigning a Attribute to a Name (line 509):
        
        # Assigning a Attribute to a Name (line 509):
        # Getting the type of 'self' (line 509)
        self_227908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 17), 'self')
        # Obtaining the member 'canvas' of a type (line 509)
        canvas_227909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 17), self_227908, 'canvas')
        # Obtaining the member 'figure' of a type (line 509)
        figure_227910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 17), canvas_227909, 'figure')
        # Obtaining the member 'bbox' of a type (line 509)
        bbox_227911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 17), figure_227910, 'bbox')
        # Obtaining the member 'height' of a type (line 509)
        height_227912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 17), bbox_227911, 'height')
        # Assigning a type to the variable 'height' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'height', height_227912)
        
        # Assigning a BinOp to a Name (line 510):
        
        # Assigning a BinOp to a Name (line 510):
        # Getting the type of 'height' (line 510)
        height_227913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 13), 'height')
        # Getting the type of 'y1' (line 510)
        y1_227914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 22), 'y1')
        # Applying the binary operator '-' (line 510)
        result_sub_227915 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 13), '-', height_227913, y1_227914)
        
        # Assigning a type to the variable 'y1' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'y1', result_sub_227915)
        
        # Assigning a BinOp to a Name (line 511):
        
        # Assigning a BinOp to a Name (line 511):
        # Getting the type of 'height' (line 511)
        height_227916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 13), 'height')
        # Getting the type of 'y0' (line 511)
        y0_227917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 22), 'y0')
        # Applying the binary operator '-' (line 511)
        result_sub_227918 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 13), '-', height_227916, y0_227917)
        
        # Assigning a type to the variable 'y0' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'y0', result_sub_227918)
        
        # Assigning a Call to a Name (line 512):
        
        # Assigning a Call to a Name (line 512):
        
        # Call to abs(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 'x1' (line 512)
        x1_227920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 16), 'x1', False)
        # Getting the type of 'x0' (line 512)
        x0_227921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 21), 'x0', False)
        # Applying the binary operator '-' (line 512)
        result_sub_227922 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 16), '-', x1_227920, x0_227921)
        
        # Processing the call keyword arguments (line 512)
        kwargs_227923 = {}
        # Getting the type of 'abs' (line 512)
        abs_227919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'abs', False)
        # Calling abs(args, kwargs) (line 512)
        abs_call_result_227924 = invoke(stypy.reporting.localization.Localization(__file__, 512, 12), abs_227919, *[result_sub_227922], **kwargs_227923)
        
        # Assigning a type to the variable 'w' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'w', abs_call_result_227924)
        
        # Assigning a Call to a Name (line 513):
        
        # Assigning a Call to a Name (line 513):
        
        # Call to abs(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'y1' (line 513)
        y1_227926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 16), 'y1', False)
        # Getting the type of 'y0' (line 513)
        y0_227927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 21), 'y0', False)
        # Applying the binary operator '-' (line 513)
        result_sub_227928 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 16), '-', y1_227926, y0_227927)
        
        # Processing the call keyword arguments (line 513)
        kwargs_227929 = {}
        # Getting the type of 'abs' (line 513)
        abs_227925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'abs', False)
        # Calling abs(args, kwargs) (line 513)
        abs_call_result_227930 = invoke(stypy.reporting.localization.Localization(__file__, 513, 12), abs_227925, *[result_sub_227928], **kwargs_227929)
        
        # Assigning a type to the variable 'h' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'h', abs_call_result_227930)
        
        # Assigning a ListComp to a Name (line 514):
        
        # Assigning a ListComp to a Name (line 514):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 514)
        tuple_227935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 514)
        # Adding element type (line 514)
        
        # Call to min(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'x0' (line 514)
        x0_227937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 41), 'x0', False)
        # Getting the type of 'x1' (line 514)
        x1_227938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 44), 'x1', False)
        # Processing the call keyword arguments (line 514)
        kwargs_227939 = {}
        # Getting the type of 'min' (line 514)
        min_227936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 37), 'min', False)
        # Calling min(args, kwargs) (line 514)
        min_call_result_227940 = invoke(stypy.reporting.localization.Localization(__file__, 514, 37), min_227936, *[x0_227937, x1_227938], **kwargs_227939)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 37), tuple_227935, min_call_result_227940)
        # Adding element type (line 514)
        
        # Call to min(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'y0' (line 514)
        y0_227942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 53), 'y0', False)
        # Getting the type of 'y1' (line 514)
        y1_227943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 57), 'y1', False)
        # Processing the call keyword arguments (line 514)
        kwargs_227944 = {}
        # Getting the type of 'min' (line 514)
        min_227941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 49), 'min', False)
        # Calling min(args, kwargs) (line 514)
        min_call_result_227945 = invoke(stypy.reporting.localization.Localization(__file__, 514, 49), min_227941, *[y0_227942, y1_227943], **kwargs_227944)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 37), tuple_227935, min_call_result_227945)
        # Adding element type (line 514)
        # Getting the type of 'w' (line 514)
        w_227946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 62), 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 37), tuple_227935, w_227946)
        # Adding element type (line 514)
        # Getting the type of 'h' (line 514)
        h_227947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 65), 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 37), tuple_227935, h_227947)
        
        comprehension_227948 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 16), tuple_227935)
        # Assigning a type to the variable 'val' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 16), 'val', comprehension_227948)
        
        # Call to int(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'val' (line 514)
        val_227932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 20), 'val', False)
        # Processing the call keyword arguments (line 514)
        kwargs_227933 = {}
        # Getting the type of 'int' (line 514)
        int_227931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 16), 'int', False)
        # Calling int(args, kwargs) (line 514)
        int_call_result_227934 = invoke(stypy.reporting.localization.Localization(__file__, 514, 16), int_227931, *[val_227932], **kwargs_227933)
        
        list_227949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 16), list_227949, int_call_result_227934)
        # Assigning a type to the variable 'rect' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'rect', list_227949)
        
        # Call to new_path(...): (line 516)
        # Processing the call keyword arguments (line 516)
        kwargs_227953 = {}
        # Getting the type of 'self' (line 516)
        self_227950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 516)
        ctx_227951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), self_227950, 'ctx')
        # Obtaining the member 'new_path' of a type (line 516)
        new_path_227952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), ctx_227951, 'new_path')
        # Calling new_path(args, kwargs) (line 516)
        new_path_call_result_227954 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), new_path_227952, *[], **kwargs_227953)
        
        
        # Call to set_line_width(...): (line 517)
        # Processing the call arguments (line 517)
        float_227958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 32), 'float')
        # Processing the call keyword arguments (line 517)
        kwargs_227959 = {}
        # Getting the type of 'self' (line 517)
        self_227955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 517)
        ctx_227956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 8), self_227955, 'ctx')
        # Obtaining the member 'set_line_width' of a type (line 517)
        set_line_width_227957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 8), ctx_227956, 'set_line_width')
        # Calling set_line_width(args, kwargs) (line 517)
        set_line_width_call_result_227960 = invoke(stypy.reporting.localization.Localization(__file__, 517, 8), set_line_width_227957, *[float_227958], **kwargs_227959)
        
        
        # Call to rectangle(...): (line 518)
        # Processing the call arguments (line 518)
        
        # Obtaining the type of the subscript
        int_227964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 32), 'int')
        # Getting the type of 'rect' (line 518)
        rect_227965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 27), 'rect', False)
        # Obtaining the member '__getitem__' of a type (line 518)
        getitem___227966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 27), rect_227965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 518)
        subscript_call_result_227967 = invoke(stypy.reporting.localization.Localization(__file__, 518, 27), getitem___227966, int_227964)
        
        
        # Obtaining the type of the subscript
        int_227968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 41), 'int')
        # Getting the type of 'rect' (line 518)
        rect_227969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 36), 'rect', False)
        # Obtaining the member '__getitem__' of a type (line 518)
        getitem___227970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 36), rect_227969, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 518)
        subscript_call_result_227971 = invoke(stypy.reporting.localization.Localization(__file__, 518, 36), getitem___227970, int_227968)
        
        
        # Obtaining the type of the subscript
        int_227972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 50), 'int')
        # Getting the type of 'rect' (line 518)
        rect_227973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 45), 'rect', False)
        # Obtaining the member '__getitem__' of a type (line 518)
        getitem___227974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 45), rect_227973, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 518)
        subscript_call_result_227975 = invoke(stypy.reporting.localization.Localization(__file__, 518, 45), getitem___227974, int_227972)
        
        
        # Obtaining the type of the subscript
        int_227976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 59), 'int')
        # Getting the type of 'rect' (line 518)
        rect_227977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 54), 'rect', False)
        # Obtaining the member '__getitem__' of a type (line 518)
        getitem___227978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 54), rect_227977, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 518)
        subscript_call_result_227979 = invoke(stypy.reporting.localization.Localization(__file__, 518, 54), getitem___227978, int_227976)
        
        # Processing the call keyword arguments (line 518)
        kwargs_227980 = {}
        # Getting the type of 'self' (line 518)
        self_227961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 518)
        ctx_227962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 8), self_227961, 'ctx')
        # Obtaining the member 'rectangle' of a type (line 518)
        rectangle_227963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 8), ctx_227962, 'rectangle')
        # Calling rectangle(args, kwargs) (line 518)
        rectangle_call_result_227981 = invoke(stypy.reporting.localization.Localization(__file__, 518, 8), rectangle_227963, *[subscript_call_result_227967, subscript_call_result_227971, subscript_call_result_227975, subscript_call_result_227979], **kwargs_227980)
        
        
        # Call to set_source_rgb(...): (line 519)
        # Processing the call arguments (line 519)
        int_227985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 32), 'int')
        int_227986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 35), 'int')
        int_227987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 38), 'int')
        # Processing the call keyword arguments (line 519)
        kwargs_227988 = {}
        # Getting the type of 'self' (line 519)
        self_227982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 519)
        ctx_227983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), self_227982, 'ctx')
        # Obtaining the member 'set_source_rgb' of a type (line 519)
        set_source_rgb_227984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), ctx_227983, 'set_source_rgb')
        # Calling set_source_rgb(args, kwargs) (line 519)
        set_source_rgb_call_result_227989 = invoke(stypy.reporting.localization.Localization(__file__, 519, 8), set_source_rgb_227984, *[int_227985, int_227986, int_227987], **kwargs_227988)
        
        
        # Call to stroke(...): (line 520)
        # Processing the call keyword arguments (line 520)
        kwargs_227993 = {}
        # Getting the type of 'self' (line 520)
        self_227990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 520)
        ctx_227991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 8), self_227990, 'ctx')
        # Obtaining the member 'stroke' of a type (line 520)
        stroke_227992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 8), ctx_227991, 'stroke')
        # Calling stroke(args, kwargs) (line 520)
        stroke_call_result_227994 = invoke(stypy.reporting.localization.Localization(__file__, 520, 8), stroke_227992, *[], **kwargs_227993)
        
        
        # ################# End of 'draw_rubberband(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_rubberband' in the type store
        # Getting the type of 'stypy_return_type' (line 501)
        stypy_return_type_227995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_227995)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_rubberband'
        return stypy_return_type_227995


    @norecursion
    def _init_toolbar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init_toolbar'
        module_type_store = module_type_store.open_function_context('_init_toolbar', 522, 4, False)
        # Assigning a type to the variable 'self' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2GTK3._init_toolbar.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2GTK3._init_toolbar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2GTK3._init_toolbar.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2GTK3._init_toolbar.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2GTK3._init_toolbar')
        NavigationToolbar2GTK3._init_toolbar.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2GTK3._init_toolbar.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2GTK3._init_toolbar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2GTK3._init_toolbar.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2GTK3._init_toolbar.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2GTK3._init_toolbar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2GTK3._init_toolbar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTK3._init_toolbar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init_toolbar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init_toolbar(...)' code ##################

        
        # Call to set_style(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 'Gtk' (line 523)
        Gtk_227998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 23), 'Gtk', False)
        # Obtaining the member 'ToolbarStyle' of a type (line 523)
        ToolbarStyle_227999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 23), Gtk_227998, 'ToolbarStyle')
        # Obtaining the member 'ICONS' of a type (line 523)
        ICONS_228000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 23), ToolbarStyle_227999, 'ICONS')
        # Processing the call keyword arguments (line 523)
        kwargs_228001 = {}
        # Getting the type of 'self' (line 523)
        self_227996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'self', False)
        # Obtaining the member 'set_style' of a type (line 523)
        set_style_227997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 8), self_227996, 'set_style')
        # Calling set_style(args, kwargs) (line 523)
        set_style_call_result_228002 = invoke(stypy.reporting.localization.Localization(__file__, 523, 8), set_style_227997, *[ICONS_228000], **kwargs_228001)
        
        
        # Assigning a Call to a Name (line 524):
        
        # Assigning a Call to a Name (line 524):
        
        # Call to join(...): (line 524)
        # Processing the call arguments (line 524)
        
        # Obtaining the type of the subscript
        unicode_228006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 40), 'unicode', u'datapath')
        # Getting the type of 'rcParams' (line 524)
        rcParams_228007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 31), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 524)
        getitem___228008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 31), rcParams_228007, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 524)
        subscript_call_result_228009 = invoke(stypy.reporting.localization.Localization(__file__, 524, 31), getitem___228008, unicode_228006)
        
        unicode_228010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 52), 'unicode', u'images')
        # Processing the call keyword arguments (line 524)
        kwargs_228011 = {}
        # Getting the type of 'os' (line 524)
        os_228003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 524)
        path_228004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 18), os_228003, 'path')
        # Obtaining the member 'join' of a type (line 524)
        join_228005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 18), path_228004, 'join')
        # Calling join(args, kwargs) (line 524)
        join_call_result_228012 = invoke(stypy.reporting.localization.Localization(__file__, 524, 18), join_228005, *[subscript_call_result_228009, unicode_228010], **kwargs_228011)
        
        # Assigning a type to the variable 'basedir' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'basedir', join_call_result_228012)
        
        # Getting the type of 'self' (line 526)
        self_228013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 56), 'self')
        # Obtaining the member 'toolitems' of a type (line 526)
        toolitems_228014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 56), self_228013, 'toolitems')
        # Testing the type of a for loop iterable (line 526)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 526, 8), toolitems_228014)
        # Getting the type of the for loop variable (line 526)
        for_loop_var_228015 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 526, 8), toolitems_228014)
        # Assigning a type to the variable 'text' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'text', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 8), for_loop_var_228015))
        # Assigning a type to the variable 'tooltip_text' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'tooltip_text', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 8), for_loop_var_228015))
        # Assigning a type to the variable 'image_file' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'image_file', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 8), for_loop_var_228015))
        # Assigning a type to the variable 'callback' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'callback', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 8), for_loop_var_228015))
        # SSA begins for a for statement (line 526)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 527)
        # Getting the type of 'text' (line 527)
        text_228016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 15), 'text')
        # Getting the type of 'None' (line 527)
        None_228017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 23), 'None')
        
        (may_be_228018, more_types_in_union_228019) = may_be_none(text_228016, None_228017)

        if may_be_228018:

            if more_types_in_union_228019:
                # Runtime conditional SSA (line 527)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to insert(...): (line 528)
            # Processing the call arguments (line 528)
            
            # Call to SeparatorToolItem(...): (line 528)
            # Processing the call keyword arguments (line 528)
            kwargs_228024 = {}
            # Getting the type of 'Gtk' (line 528)
            Gtk_228022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 29), 'Gtk', False)
            # Obtaining the member 'SeparatorToolItem' of a type (line 528)
            SeparatorToolItem_228023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 29), Gtk_228022, 'SeparatorToolItem')
            # Calling SeparatorToolItem(args, kwargs) (line 528)
            SeparatorToolItem_call_result_228025 = invoke(stypy.reporting.localization.Localization(__file__, 528, 29), SeparatorToolItem_228023, *[], **kwargs_228024)
            
            int_228026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 54), 'int')
            # Processing the call keyword arguments (line 528)
            kwargs_228027 = {}
            # Getting the type of 'self' (line 528)
            self_228020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 16), 'self', False)
            # Obtaining the member 'insert' of a type (line 528)
            insert_228021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 16), self_228020, 'insert')
            # Calling insert(args, kwargs) (line 528)
            insert_call_result_228028 = invoke(stypy.reporting.localization.Localization(__file__, 528, 16), insert_228021, *[SeparatorToolItem_call_result_228025, int_228026], **kwargs_228027)
            

            if more_types_in_union_228019:
                # SSA join for if statement (line 527)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 530):
        
        # Assigning a Call to a Name (line 530):
        
        # Call to join(...): (line 530)
        # Processing the call arguments (line 530)
        # Getting the type of 'basedir' (line 530)
        basedir_228032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 33), 'basedir', False)
        # Getting the type of 'image_file' (line 530)
        image_file_228033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 42), 'image_file', False)
        unicode_228034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 55), 'unicode', u'.png')
        # Applying the binary operator '+' (line 530)
        result_add_228035 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 42), '+', image_file_228033, unicode_228034)
        
        # Processing the call keyword arguments (line 530)
        kwargs_228036 = {}
        # Getting the type of 'os' (line 530)
        os_228029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 530)
        path_228030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 20), os_228029, 'path')
        # Obtaining the member 'join' of a type (line 530)
        join_228031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 20), path_228030, 'join')
        # Calling join(args, kwargs) (line 530)
        join_call_result_228037 = invoke(stypy.reporting.localization.Localization(__file__, 530, 20), join_228031, *[basedir_228032, result_add_228035], **kwargs_228036)
        
        # Assigning a type to the variable 'fname' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'fname', join_call_result_228037)
        
        # Assigning a Call to a Name (line 531):
        
        # Assigning a Call to a Name (line 531):
        
        # Call to Image(...): (line 531)
        # Processing the call keyword arguments (line 531)
        kwargs_228040 = {}
        # Getting the type of 'Gtk' (line 531)
        Gtk_228038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 20), 'Gtk', False)
        # Obtaining the member 'Image' of a type (line 531)
        Image_228039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 20), Gtk_228038, 'Image')
        # Calling Image(args, kwargs) (line 531)
        Image_call_result_228041 = invoke(stypy.reporting.localization.Localization(__file__, 531, 20), Image_228039, *[], **kwargs_228040)
        
        # Assigning a type to the variable 'image' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'image', Image_call_result_228041)
        
        # Call to set_from_file(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 'fname' (line 532)
        fname_228044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 32), 'fname', False)
        # Processing the call keyword arguments (line 532)
        kwargs_228045 = {}
        # Getting the type of 'image' (line 532)
        image_228042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'image', False)
        # Obtaining the member 'set_from_file' of a type (line 532)
        set_from_file_228043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 12), image_228042, 'set_from_file')
        # Calling set_from_file(args, kwargs) (line 532)
        set_from_file_call_result_228046 = invoke(stypy.reporting.localization.Localization(__file__, 532, 12), set_from_file_228043, *[fname_228044], **kwargs_228045)
        
        
        # Assigning a Call to a Name (line 533):
        
        # Assigning a Call to a Name (line 533):
        
        # Call to ToolButton(...): (line 533)
        # Processing the call keyword arguments (line 533)
        kwargs_228049 = {}
        # Getting the type of 'Gtk' (line 533)
        Gtk_228047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 22), 'Gtk', False)
        # Obtaining the member 'ToolButton' of a type (line 533)
        ToolButton_228048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 22), Gtk_228047, 'ToolButton')
        # Calling ToolButton(args, kwargs) (line 533)
        ToolButton_call_result_228050 = invoke(stypy.reporting.localization.Localization(__file__, 533, 22), ToolButton_228048, *[], **kwargs_228049)
        
        # Assigning a type to the variable 'tbutton' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'tbutton', ToolButton_call_result_228050)
        
        # Call to set_label(...): (line 534)
        # Processing the call arguments (line 534)
        # Getting the type of 'text' (line 534)
        text_228053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 30), 'text', False)
        # Processing the call keyword arguments (line 534)
        kwargs_228054 = {}
        # Getting the type of 'tbutton' (line 534)
        tbutton_228051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'tbutton', False)
        # Obtaining the member 'set_label' of a type (line 534)
        set_label_228052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 12), tbutton_228051, 'set_label')
        # Calling set_label(args, kwargs) (line 534)
        set_label_call_result_228055 = invoke(stypy.reporting.localization.Localization(__file__, 534, 12), set_label_228052, *[text_228053], **kwargs_228054)
        
        
        # Call to set_icon_widget(...): (line 535)
        # Processing the call arguments (line 535)
        # Getting the type of 'image' (line 535)
        image_228058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 36), 'image', False)
        # Processing the call keyword arguments (line 535)
        kwargs_228059 = {}
        # Getting the type of 'tbutton' (line 535)
        tbutton_228056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'tbutton', False)
        # Obtaining the member 'set_icon_widget' of a type (line 535)
        set_icon_widget_228057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 12), tbutton_228056, 'set_icon_widget')
        # Calling set_icon_widget(args, kwargs) (line 535)
        set_icon_widget_call_result_228060 = invoke(stypy.reporting.localization.Localization(__file__, 535, 12), set_icon_widget_228057, *[image_228058], **kwargs_228059)
        
        
        # Call to insert(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'tbutton' (line 536)
        tbutton_228063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 24), 'tbutton', False)
        int_228064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 33), 'int')
        # Processing the call keyword arguments (line 536)
        kwargs_228065 = {}
        # Getting the type of 'self' (line 536)
        self_228061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'self', False)
        # Obtaining the member 'insert' of a type (line 536)
        insert_228062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 12), self_228061, 'insert')
        # Calling insert(args, kwargs) (line 536)
        insert_call_result_228066 = invoke(stypy.reporting.localization.Localization(__file__, 536, 12), insert_228062, *[tbutton_228063, int_228064], **kwargs_228065)
        
        
        # Call to connect(...): (line 537)
        # Processing the call arguments (line 537)
        unicode_228069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 28), 'unicode', u'clicked')
        
        # Call to getattr(...): (line 537)
        # Processing the call arguments (line 537)
        # Getting the type of 'self' (line 537)
        self_228071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 47), 'self', False)
        # Getting the type of 'callback' (line 537)
        callback_228072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 53), 'callback', False)
        # Processing the call keyword arguments (line 537)
        kwargs_228073 = {}
        # Getting the type of 'getattr' (line 537)
        getattr_228070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 39), 'getattr', False)
        # Calling getattr(args, kwargs) (line 537)
        getattr_call_result_228074 = invoke(stypy.reporting.localization.Localization(__file__, 537, 39), getattr_228070, *[self_228071, callback_228072], **kwargs_228073)
        
        # Processing the call keyword arguments (line 537)
        kwargs_228075 = {}
        # Getting the type of 'tbutton' (line 537)
        tbutton_228067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'tbutton', False)
        # Obtaining the member 'connect' of a type (line 537)
        connect_228068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), tbutton_228067, 'connect')
        # Calling connect(args, kwargs) (line 537)
        connect_call_result_228076 = invoke(stypy.reporting.localization.Localization(__file__, 537, 12), connect_228068, *[unicode_228069, getattr_call_result_228074], **kwargs_228075)
        
        
        # Call to set_tooltip_text(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'tooltip_text' (line 538)
        tooltip_text_228079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 37), 'tooltip_text', False)
        # Processing the call keyword arguments (line 538)
        kwargs_228080 = {}
        # Getting the type of 'tbutton' (line 538)
        tbutton_228077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'tbutton', False)
        # Obtaining the member 'set_tooltip_text' of a type (line 538)
        set_tooltip_text_228078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 12), tbutton_228077, 'set_tooltip_text')
        # Calling set_tooltip_text(args, kwargs) (line 538)
        set_tooltip_text_call_result_228081 = invoke(stypy.reporting.localization.Localization(__file__, 538, 12), set_tooltip_text_228078, *[tooltip_text_228079], **kwargs_228080)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 540):
        
        # Assigning a Call to a Name (line 540):
        
        # Call to SeparatorToolItem(...): (line 540)
        # Processing the call keyword arguments (line 540)
        kwargs_228084 = {}
        # Getting the type of 'Gtk' (line 540)
        Gtk_228082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 19), 'Gtk', False)
        # Obtaining the member 'SeparatorToolItem' of a type (line 540)
        SeparatorToolItem_228083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 19), Gtk_228082, 'SeparatorToolItem')
        # Calling SeparatorToolItem(args, kwargs) (line 540)
        SeparatorToolItem_call_result_228085 = invoke(stypy.reporting.localization.Localization(__file__, 540, 19), SeparatorToolItem_228083, *[], **kwargs_228084)
        
        # Assigning a type to the variable 'toolitem' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'toolitem', SeparatorToolItem_call_result_228085)
        
        # Call to insert(...): (line 541)
        # Processing the call arguments (line 541)
        # Getting the type of 'toolitem' (line 541)
        toolitem_228088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 20), 'toolitem', False)
        int_228089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 30), 'int')
        # Processing the call keyword arguments (line 541)
        kwargs_228090 = {}
        # Getting the type of 'self' (line 541)
        self_228086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'self', False)
        # Obtaining the member 'insert' of a type (line 541)
        insert_228087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 8), self_228086, 'insert')
        # Calling insert(args, kwargs) (line 541)
        insert_call_result_228091 = invoke(stypy.reporting.localization.Localization(__file__, 541, 8), insert_228087, *[toolitem_228088, int_228089], **kwargs_228090)
        
        
        # Call to set_draw(...): (line 542)
        # Processing the call arguments (line 542)
        # Getting the type of 'False' (line 542)
        False_228094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 26), 'False', False)
        # Processing the call keyword arguments (line 542)
        kwargs_228095 = {}
        # Getting the type of 'toolitem' (line 542)
        toolitem_228092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'toolitem', False)
        # Obtaining the member 'set_draw' of a type (line 542)
        set_draw_228093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 8), toolitem_228092, 'set_draw')
        # Calling set_draw(args, kwargs) (line 542)
        set_draw_call_result_228096 = invoke(stypy.reporting.localization.Localization(__file__, 542, 8), set_draw_228093, *[False_228094], **kwargs_228095)
        
        
        # Call to set_expand(...): (line 543)
        # Processing the call arguments (line 543)
        # Getting the type of 'True' (line 543)
        True_228099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 28), 'True', False)
        # Processing the call keyword arguments (line 543)
        kwargs_228100 = {}
        # Getting the type of 'toolitem' (line 543)
        toolitem_228097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'toolitem', False)
        # Obtaining the member 'set_expand' of a type (line 543)
        set_expand_228098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 8), toolitem_228097, 'set_expand')
        # Calling set_expand(args, kwargs) (line 543)
        set_expand_call_result_228101 = invoke(stypy.reporting.localization.Localization(__file__, 543, 8), set_expand_228098, *[True_228099], **kwargs_228100)
        
        
        # Assigning a Call to a Name (line 545):
        
        # Assigning a Call to a Name (line 545):
        
        # Call to ToolItem(...): (line 545)
        # Processing the call keyword arguments (line 545)
        kwargs_228104 = {}
        # Getting the type of 'Gtk' (line 545)
        Gtk_228102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 19), 'Gtk', False)
        # Obtaining the member 'ToolItem' of a type (line 545)
        ToolItem_228103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 19), Gtk_228102, 'ToolItem')
        # Calling ToolItem(args, kwargs) (line 545)
        ToolItem_call_result_228105 = invoke(stypy.reporting.localization.Localization(__file__, 545, 19), ToolItem_228103, *[], **kwargs_228104)
        
        # Assigning a type to the variable 'toolitem' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'toolitem', ToolItem_call_result_228105)
        
        # Call to insert(...): (line 546)
        # Processing the call arguments (line 546)
        # Getting the type of 'toolitem' (line 546)
        toolitem_228108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 20), 'toolitem', False)
        int_228109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 30), 'int')
        # Processing the call keyword arguments (line 546)
        kwargs_228110 = {}
        # Getting the type of 'self' (line 546)
        self_228106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'self', False)
        # Obtaining the member 'insert' of a type (line 546)
        insert_228107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 8), self_228106, 'insert')
        # Calling insert(args, kwargs) (line 546)
        insert_call_result_228111 = invoke(stypy.reporting.localization.Localization(__file__, 546, 8), insert_228107, *[toolitem_228108, int_228109], **kwargs_228110)
        
        
        # Assigning a Call to a Attribute (line 547):
        
        # Assigning a Call to a Attribute (line 547):
        
        # Call to Label(...): (line 547)
        # Processing the call keyword arguments (line 547)
        kwargs_228114 = {}
        # Getting the type of 'Gtk' (line 547)
        Gtk_228112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 23), 'Gtk', False)
        # Obtaining the member 'Label' of a type (line 547)
        Label_228113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 23), Gtk_228112, 'Label')
        # Calling Label(args, kwargs) (line 547)
        Label_call_result_228115 = invoke(stypy.reporting.localization.Localization(__file__, 547, 23), Label_228113, *[], **kwargs_228114)
        
        # Getting the type of 'self' (line 547)
        self_228116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'self')
        # Setting the type of the member 'message' of a type (line 547)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 8), self_228116, 'message', Label_call_result_228115)
        
        # Call to add(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'self' (line 548)
        self_228119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 21), 'self', False)
        # Obtaining the member 'message' of a type (line 548)
        message_228120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 21), self_228119, 'message')
        # Processing the call keyword arguments (line 548)
        kwargs_228121 = {}
        # Getting the type of 'toolitem' (line 548)
        toolitem_228117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'toolitem', False)
        # Obtaining the member 'add' of a type (line 548)
        add_228118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 8), toolitem_228117, 'add')
        # Calling add(args, kwargs) (line 548)
        add_call_result_228122 = invoke(stypy.reporting.localization.Localization(__file__, 548, 8), add_228118, *[message_228120], **kwargs_228121)
        
        
        # Call to show_all(...): (line 550)
        # Processing the call keyword arguments (line 550)
        kwargs_228125 = {}
        # Getting the type of 'self' (line 550)
        self_228123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 8), 'self', False)
        # Obtaining the member 'show_all' of a type (line 550)
        show_all_228124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 8), self_228123, 'show_all')
        # Calling show_all(args, kwargs) (line 550)
        show_all_call_result_228126 = invoke(stypy.reporting.localization.Localization(__file__, 550, 8), show_all_228124, *[], **kwargs_228125)
        
        
        # ################# End of '_init_toolbar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_toolbar' in the type store
        # Getting the type of 'stypy_return_type' (line 522)
        stypy_return_type_228127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228127)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_toolbar'
        return stypy_return_type_228127


    @norecursion
    def get_filechooser(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_filechooser'
        module_type_store = module_type_store.open_function_context('get_filechooser', 552, 4, False)
        # Assigning a type to the variable 'self' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2GTK3.get_filechooser.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2GTK3.get_filechooser.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2GTK3.get_filechooser.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2GTK3.get_filechooser.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2GTK3.get_filechooser')
        NavigationToolbar2GTK3.get_filechooser.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2GTK3.get_filechooser.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2GTK3.get_filechooser.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2GTK3.get_filechooser.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2GTK3.get_filechooser.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2GTK3.get_filechooser.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2GTK3.get_filechooser.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTK3.get_filechooser', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_filechooser', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_filechooser(...)' code ##################

        
        # Assigning a Call to a Name (line 553):
        
        # Assigning a Call to a Name (line 553):
        
        # Call to FileChooserDialog(...): (line 553)
        # Processing the call keyword arguments (line 553)
        unicode_228129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 18), 'unicode', u'Save the figure')
        keyword_228130 = unicode_228129
        # Getting the type of 'self' (line 555)
        self_228131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 19), 'self', False)
        # Obtaining the member 'win' of a type (line 555)
        win_228132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 19), self_228131, 'win')
        keyword_228133 = win_228132
        
        # Call to expanduser(...): (line 556)
        # Processing the call arguments (line 556)
        
        # Obtaining the type of the subscript
        unicode_228137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 45), 'unicode', u'savefig.directory')
        # Getting the type of 'rcParams' (line 556)
        rcParams_228138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 36), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 556)
        getitem___228139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 36), rcParams_228138, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 556)
        subscript_call_result_228140 = invoke(stypy.reporting.localization.Localization(__file__, 556, 36), getitem___228139, unicode_228137)
        
        # Processing the call keyword arguments (line 556)
        kwargs_228141 = {}
        # Getting the type of 'os' (line 556)
        os_228134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 556)
        path_228135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 17), os_228134, 'path')
        # Obtaining the member 'expanduser' of a type (line 556)
        expanduser_228136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 17), path_228135, 'expanduser')
        # Calling expanduser(args, kwargs) (line 556)
        expanduser_call_result_228142 = invoke(stypy.reporting.localization.Localization(__file__, 556, 17), expanduser_228136, *[subscript_call_result_228140], **kwargs_228141)
        
        keyword_228143 = expanduser_call_result_228142
        
        # Call to get_supported_filetypes(...): (line 557)
        # Processing the call keyword arguments (line 557)
        kwargs_228147 = {}
        # Getting the type of 'self' (line 557)
        self_228144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 22), 'self', False)
        # Obtaining the member 'canvas' of a type (line 557)
        canvas_228145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 22), self_228144, 'canvas')
        # Obtaining the member 'get_supported_filetypes' of a type (line 557)
        get_supported_filetypes_228146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 22), canvas_228145, 'get_supported_filetypes')
        # Calling get_supported_filetypes(args, kwargs) (line 557)
        get_supported_filetypes_call_result_228148 = invoke(stypy.reporting.localization.Localization(__file__, 557, 22), get_supported_filetypes_228146, *[], **kwargs_228147)
        
        keyword_228149 = get_supported_filetypes_call_result_228148
        
        # Call to get_default_filetype(...): (line 558)
        # Processing the call keyword arguments (line 558)
        kwargs_228153 = {}
        # Getting the type of 'self' (line 558)
        self_228150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 29), 'self', False)
        # Obtaining the member 'canvas' of a type (line 558)
        canvas_228151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 29), self_228150, 'canvas')
        # Obtaining the member 'get_default_filetype' of a type (line 558)
        get_default_filetype_228152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 29), canvas_228151, 'get_default_filetype')
        # Calling get_default_filetype(args, kwargs) (line 558)
        get_default_filetype_call_result_228154 = invoke(stypy.reporting.localization.Localization(__file__, 558, 29), get_default_filetype_228152, *[], **kwargs_228153)
        
        keyword_228155 = get_default_filetype_call_result_228154
        kwargs_228156 = {'default_filetype': keyword_228155, 'path': keyword_228143, 'filetypes': keyword_228149, 'parent': keyword_228133, 'title': keyword_228130}
        # Getting the type of 'FileChooserDialog' (line 553)
        FileChooserDialog_228128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 13), 'FileChooserDialog', False)
        # Calling FileChooserDialog(args, kwargs) (line 553)
        FileChooserDialog_call_result_228157 = invoke(stypy.reporting.localization.Localization(__file__, 553, 13), FileChooserDialog_228128, *[], **kwargs_228156)
        
        # Assigning a type to the variable 'fc' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'fc', FileChooserDialog_call_result_228157)
        
        # Call to set_current_name(...): (line 559)
        # Processing the call arguments (line 559)
        
        # Call to get_default_filename(...): (line 559)
        # Processing the call keyword arguments (line 559)
        kwargs_228163 = {}
        # Getting the type of 'self' (line 559)
        self_228160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 28), 'self', False)
        # Obtaining the member 'canvas' of a type (line 559)
        canvas_228161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 28), self_228160, 'canvas')
        # Obtaining the member 'get_default_filename' of a type (line 559)
        get_default_filename_228162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 28), canvas_228161, 'get_default_filename')
        # Calling get_default_filename(args, kwargs) (line 559)
        get_default_filename_call_result_228164 = invoke(stypy.reporting.localization.Localization(__file__, 559, 28), get_default_filename_228162, *[], **kwargs_228163)
        
        # Processing the call keyword arguments (line 559)
        kwargs_228165 = {}
        # Getting the type of 'fc' (line 559)
        fc_228158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'fc', False)
        # Obtaining the member 'set_current_name' of a type (line 559)
        set_current_name_228159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 8), fc_228158, 'set_current_name')
        # Calling set_current_name(args, kwargs) (line 559)
        set_current_name_call_result_228166 = invoke(stypy.reporting.localization.Localization(__file__, 559, 8), set_current_name_228159, *[get_default_filename_call_result_228164], **kwargs_228165)
        
        # Getting the type of 'fc' (line 560)
        fc_228167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 15), 'fc')
        # Assigning a type to the variable 'stypy_return_type' (line 560)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'stypy_return_type', fc_228167)
        
        # ################# End of 'get_filechooser(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_filechooser' in the type store
        # Getting the type of 'stypy_return_type' (line 552)
        stypy_return_type_228168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228168)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_filechooser'
        return stypy_return_type_228168


    @norecursion
    def save_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'save_figure'
        module_type_store = module_type_store.open_function_context('save_figure', 562, 4, False)
        # Assigning a type to the variable 'self' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2GTK3.save_figure.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2GTK3.save_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2GTK3.save_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2GTK3.save_figure.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2GTK3.save_figure')
        NavigationToolbar2GTK3.save_figure.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2GTK3.save_figure.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        NavigationToolbar2GTK3.save_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2GTK3.save_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2GTK3.save_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2GTK3.save_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2GTK3.save_figure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTK3.save_figure', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'save_figure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'save_figure(...)' code ##################

        
        # Assigning a Call to a Name (line 563):
        
        # Assigning a Call to a Name (line 563):
        
        # Call to get_filechooser(...): (line 563)
        # Processing the call keyword arguments (line 563)
        kwargs_228171 = {}
        # Getting the type of 'self' (line 563)
        self_228169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 18), 'self', False)
        # Obtaining the member 'get_filechooser' of a type (line 563)
        get_filechooser_228170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 18), self_228169, 'get_filechooser')
        # Calling get_filechooser(args, kwargs) (line 563)
        get_filechooser_call_result_228172 = invoke(stypy.reporting.localization.Localization(__file__, 563, 18), get_filechooser_228170, *[], **kwargs_228171)
        
        # Assigning a type to the variable 'chooser' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'chooser', get_filechooser_call_result_228172)
        
        # Assigning a Call to a Tuple (line 564):
        
        # Assigning a Call to a Name:
        
        # Call to get_filename_from_user(...): (line 564)
        # Processing the call keyword arguments (line 564)
        kwargs_228175 = {}
        # Getting the type of 'chooser' (line 564)
        chooser_228173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 24), 'chooser', False)
        # Obtaining the member 'get_filename_from_user' of a type (line 564)
        get_filename_from_user_228174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 24), chooser_228173, 'get_filename_from_user')
        # Calling get_filename_from_user(args, kwargs) (line 564)
        get_filename_from_user_call_result_228176 = invoke(stypy.reporting.localization.Localization(__file__, 564, 24), get_filename_from_user_228174, *[], **kwargs_228175)
        
        # Assigning a type to the variable 'call_assignment_226575' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_226575', get_filename_from_user_call_result_228176)
        
        # Assigning a Call to a Name (line 564):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_228179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
        # Processing the call keyword arguments
        kwargs_228180 = {}
        # Getting the type of 'call_assignment_226575' (line 564)
        call_assignment_226575_228177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_226575', False)
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___228178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), call_assignment_226575_228177, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_228181 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___228178, *[int_228179], **kwargs_228180)
        
        # Assigning a type to the variable 'call_assignment_226576' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_226576', getitem___call_result_228181)
        
        # Assigning a Name to a Name (line 564):
        # Getting the type of 'call_assignment_226576' (line 564)
        call_assignment_226576_228182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_226576')
        # Assigning a type to the variable 'fname' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'fname', call_assignment_226576_228182)
        
        # Assigning a Call to a Name (line 564):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_228185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
        # Processing the call keyword arguments
        kwargs_228186 = {}
        # Getting the type of 'call_assignment_226575' (line 564)
        call_assignment_226575_228183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_226575', False)
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___228184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), call_assignment_226575_228183, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_228187 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___228184, *[int_228185], **kwargs_228186)
        
        # Assigning a type to the variable 'call_assignment_226577' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_226577', getitem___call_result_228187)
        
        # Assigning a Name to a Name (line 564):
        # Getting the type of 'call_assignment_226577' (line 564)
        call_assignment_226577_228188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_226577')
        # Assigning a type to the variable 'format' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 15), 'format', call_assignment_226577_228188)
        
        # Call to destroy(...): (line 565)
        # Processing the call keyword arguments (line 565)
        kwargs_228191 = {}
        # Getting the type of 'chooser' (line 565)
        chooser_228189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'chooser', False)
        # Obtaining the member 'destroy' of a type (line 565)
        destroy_228190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 8), chooser_228189, 'destroy')
        # Calling destroy(args, kwargs) (line 565)
        destroy_call_result_228192 = invoke(stypy.reporting.localization.Localization(__file__, 565, 8), destroy_228190, *[], **kwargs_228191)
        
        
        # Getting the type of 'fname' (line 566)
        fname_228193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 11), 'fname')
        # Testing the type of an if condition (line 566)
        if_condition_228194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 566, 8), fname_228193)
        # Assigning a type to the variable 'if_condition_228194' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'if_condition_228194', if_condition_228194)
        # SSA begins for if statement (line 566)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 567):
        
        # Assigning a Call to a Name (line 567):
        
        # Call to expanduser(...): (line 567)
        # Processing the call arguments (line 567)
        
        # Obtaining the type of the subscript
        unicode_228198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 52), 'unicode', u'savefig.directory')
        # Getting the type of 'rcParams' (line 567)
        rcParams_228199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 43), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 567)
        getitem___228200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 43), rcParams_228199, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 567)
        subscript_call_result_228201 = invoke(stypy.reporting.localization.Localization(__file__, 567, 43), getitem___228200, unicode_228198)
        
        # Processing the call keyword arguments (line 567)
        kwargs_228202 = {}
        # Getting the type of 'os' (line 567)
        os_228195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 567)
        path_228196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 24), os_228195, 'path')
        # Obtaining the member 'expanduser' of a type (line 567)
        expanduser_228197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 24), path_228196, 'expanduser')
        # Calling expanduser(args, kwargs) (line 567)
        expanduser_call_result_228203 = invoke(stypy.reporting.localization.Localization(__file__, 567, 24), expanduser_228197, *[subscript_call_result_228201], **kwargs_228202)
        
        # Assigning a type to the variable 'startpath' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'startpath', expanduser_call_result_228203)
        
        
        # Getting the type of 'startpath' (line 569)
        startpath_228204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 15), 'startpath')
        unicode_228205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 28), 'unicode', u'')
        # Applying the binary operator '!=' (line 569)
        result_ne_228206 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 15), '!=', startpath_228204, unicode_228205)
        
        # Testing the type of an if condition (line 569)
        if_condition_228207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 569, 12), result_ne_228206)
        # Assigning a type to the variable 'if_condition_228207' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'if_condition_228207', if_condition_228207)
        # SSA begins for if statement (line 569)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 570):
        
        # Assigning a Call to a Subscript (line 570):
        
        # Call to dirname(...): (line 571)
        # Processing the call arguments (line 571)
        
        # Call to text_type(...): (line 571)
        # Processing the call arguments (line 571)
        # Getting the type of 'fname' (line 571)
        fname_228213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 50), 'fname', False)
        # Processing the call keyword arguments (line 571)
        kwargs_228214 = {}
        # Getting the type of 'six' (line 571)
        six_228211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 36), 'six', False)
        # Obtaining the member 'text_type' of a type (line 571)
        text_type_228212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 36), six_228211, 'text_type')
        # Calling text_type(args, kwargs) (line 571)
        text_type_call_result_228215 = invoke(stypy.reporting.localization.Localization(__file__, 571, 36), text_type_228212, *[fname_228213], **kwargs_228214)
        
        # Processing the call keyword arguments (line 571)
        kwargs_228216 = {}
        # Getting the type of 'os' (line 571)
        os_228208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 571)
        path_228209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 20), os_228208, 'path')
        # Obtaining the member 'dirname' of a type (line 571)
        dirname_228210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 20), path_228209, 'dirname')
        # Calling dirname(args, kwargs) (line 571)
        dirname_call_result_228217 = invoke(stypy.reporting.localization.Localization(__file__, 571, 20), dirname_228210, *[text_type_call_result_228215], **kwargs_228216)
        
        # Getting the type of 'rcParams' (line 570)
        rcParams_228218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 16), 'rcParams')
        unicode_228219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 25), 'unicode', u'savefig.directory')
        # Storing an element on a container (line 570)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 16), rcParams_228218, (unicode_228219, dirname_call_result_228217))
        # SSA join for if statement (line 569)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 572)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to savefig(...): (line 573)
        # Processing the call arguments (line 573)
        # Getting the type of 'fname' (line 573)
        fname_228224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 43), 'fname', False)
        # Processing the call keyword arguments (line 573)
        # Getting the type of 'format' (line 573)
        format_228225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 57), 'format', False)
        keyword_228226 = format_228225
        kwargs_228227 = {'format': keyword_228226}
        # Getting the type of 'self' (line 573)
        self_228220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 16), 'self', False)
        # Obtaining the member 'canvas' of a type (line 573)
        canvas_228221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 16), self_228220, 'canvas')
        # Obtaining the member 'figure' of a type (line 573)
        figure_228222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 16), canvas_228221, 'figure')
        # Obtaining the member 'savefig' of a type (line 573)
        savefig_228223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 16), figure_228222, 'savefig')
        # Calling savefig(args, kwargs) (line 573)
        savefig_call_result_228228 = invoke(stypy.reporting.localization.Localization(__file__, 573, 16), savefig_228223, *[fname_228224], **kwargs_228227)
        
        # SSA branch for the except part of a try statement (line 572)
        # SSA branch for the except 'Exception' branch of a try statement (line 572)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 574)
        Exception_228229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 19), 'Exception')
        # Assigning a type to the variable 'e' (line 574)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'e', Exception_228229)
        
        # Call to error_msg_gtk(...): (line 575)
        # Processing the call arguments (line 575)
        
        # Call to str(...): (line 575)
        # Processing the call arguments (line 575)
        # Getting the type of 'e' (line 575)
        e_228232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 34), 'e', False)
        # Processing the call keyword arguments (line 575)
        kwargs_228233 = {}
        # Getting the type of 'str' (line 575)
        str_228231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 30), 'str', False)
        # Calling str(args, kwargs) (line 575)
        str_call_result_228234 = invoke(stypy.reporting.localization.Localization(__file__, 575, 30), str_228231, *[e_228232], **kwargs_228233)
        
        # Processing the call keyword arguments (line 575)
        # Getting the type of 'self' (line 575)
        self_228235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 45), 'self', False)
        keyword_228236 = self_228235
        kwargs_228237 = {'parent': keyword_228236}
        # Getting the type of 'error_msg_gtk' (line 575)
        error_msg_gtk_228230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'error_msg_gtk', False)
        # Calling error_msg_gtk(args, kwargs) (line 575)
        error_msg_gtk_call_result_228238 = invoke(stypy.reporting.localization.Localization(__file__, 575, 16), error_msg_gtk_228230, *[str_call_result_228234], **kwargs_228237)
        
        # SSA join for try-except statement (line 572)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 566)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'save_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'save_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 562)
        stypy_return_type_228239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228239)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'save_figure'
        return stypy_return_type_228239


    @norecursion
    def configure_subplots(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'configure_subplots'
        module_type_store = module_type_store.open_function_context('configure_subplots', 577, 4, False)
        # Assigning a type to the variable 'self' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2GTK3.configure_subplots.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2GTK3.configure_subplots.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2GTK3.configure_subplots.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2GTK3.configure_subplots.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2GTK3.configure_subplots')
        NavigationToolbar2GTK3.configure_subplots.__dict__.__setitem__('stypy_param_names_list', ['button'])
        NavigationToolbar2GTK3.configure_subplots.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2GTK3.configure_subplots.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2GTK3.configure_subplots.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2GTK3.configure_subplots.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2GTK3.configure_subplots.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2GTK3.configure_subplots.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTK3.configure_subplots', ['button'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'configure_subplots', localization, ['button'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'configure_subplots(...)' code ##################

        
        # Assigning a Call to a Name (line 578):
        
        # Assigning a Call to a Name (line 578):
        
        # Call to Figure(...): (line 578)
        # Processing the call keyword arguments (line 578)
        
        # Obtaining an instance of the builtin type 'tuple' (line 578)
        tuple_228241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 578)
        # Adding element type (line 578)
        int_228242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 34), tuple_228241, int_228242)
        # Adding element type (line 578)
        int_228243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 34), tuple_228241, int_228243)
        
        keyword_228244 = tuple_228241
        kwargs_228245 = {'figsize': keyword_228244}
        # Getting the type of 'Figure' (line 578)
        Figure_228240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 18), 'Figure', False)
        # Calling Figure(args, kwargs) (line 578)
        Figure_call_result_228246 = invoke(stypy.reporting.localization.Localization(__file__, 578, 18), Figure_228240, *[], **kwargs_228245)
        
        # Assigning a type to the variable 'toolfig' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'toolfig', Figure_call_result_228246)
        
        # Assigning a Call to a Name (line 579):
        
        # Assigning a Call to a Name (line 579):
        
        # Call to _get_canvas(...): (line 579)
        # Processing the call arguments (line 579)
        # Getting the type of 'toolfig' (line 579)
        toolfig_228249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 34), 'toolfig', False)
        # Processing the call keyword arguments (line 579)
        kwargs_228250 = {}
        # Getting the type of 'self' (line 579)
        self_228247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 17), 'self', False)
        # Obtaining the member '_get_canvas' of a type (line 579)
        _get_canvas_228248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 17), self_228247, '_get_canvas')
        # Calling _get_canvas(args, kwargs) (line 579)
        _get_canvas_call_result_228251 = invoke(stypy.reporting.localization.Localization(__file__, 579, 17), _get_canvas_228248, *[toolfig_228249], **kwargs_228250)
        
        # Assigning a type to the variable 'canvas' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'canvas', _get_canvas_call_result_228251)
        
        # Call to subplots_adjust(...): (line 580)
        # Processing the call keyword arguments (line 580)
        float_228254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 36), 'float')
        keyword_228255 = float_228254
        kwargs_228256 = {'top': keyword_228255}
        # Getting the type of 'toolfig' (line 580)
        toolfig_228252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'toolfig', False)
        # Obtaining the member 'subplots_adjust' of a type (line 580)
        subplots_adjust_228253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 8), toolfig_228252, 'subplots_adjust')
        # Calling subplots_adjust(args, kwargs) (line 580)
        subplots_adjust_call_result_228257 = invoke(stypy.reporting.localization.Localization(__file__, 580, 8), subplots_adjust_228253, *[], **kwargs_228256)
        
        
        # Assigning a Call to a Name (line 581):
        
        # Assigning a Call to a Name (line 581):
        
        # Call to SubplotTool(...): (line 581)
        # Processing the call arguments (line 581)
        # Getting the type of 'self' (line 581)
        self_228259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 28), 'self', False)
        # Obtaining the member 'canvas' of a type (line 581)
        canvas_228260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 28), self_228259, 'canvas')
        # Obtaining the member 'figure' of a type (line 581)
        figure_228261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 28), canvas_228260, 'figure')
        # Getting the type of 'toolfig' (line 581)
        toolfig_228262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 48), 'toolfig', False)
        # Processing the call keyword arguments (line 581)
        kwargs_228263 = {}
        # Getting the type of 'SubplotTool' (line 581)
        SubplotTool_228258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'SubplotTool', False)
        # Calling SubplotTool(args, kwargs) (line 581)
        SubplotTool_call_result_228264 = invoke(stypy.reporting.localization.Localization(__file__, 581, 16), SubplotTool_228258, *[figure_228261, toolfig_228262], **kwargs_228263)
        
        # Assigning a type to the variable 'tool' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tool', SubplotTool_call_result_228264)
        
        # Assigning a Call to a Name (line 583):
        
        # Assigning a Call to a Name (line 583):
        
        # Call to int(...): (line 583)
        # Processing the call arguments (line 583)
        # Getting the type of 'toolfig' (line 583)
        toolfig_228266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 16), 'toolfig', False)
        # Obtaining the member 'bbox' of a type (line 583)
        bbox_228267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 16), toolfig_228266, 'bbox')
        # Obtaining the member 'width' of a type (line 583)
        width_228268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 16), bbox_228267, 'width')
        # Processing the call keyword arguments (line 583)
        kwargs_228269 = {}
        # Getting the type of 'int' (line 583)
        int_228265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'int', False)
        # Calling int(args, kwargs) (line 583)
        int_call_result_228270 = invoke(stypy.reporting.localization.Localization(__file__, 583, 12), int_228265, *[width_228268], **kwargs_228269)
        
        # Assigning a type to the variable 'w' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'w', int_call_result_228270)
        
        # Assigning a Call to a Name (line 584):
        
        # Assigning a Call to a Name (line 584):
        
        # Call to int(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 'toolfig' (line 584)
        toolfig_228272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'toolfig', False)
        # Obtaining the member 'bbox' of a type (line 584)
        bbox_228273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 16), toolfig_228272, 'bbox')
        # Obtaining the member 'height' of a type (line 584)
        height_228274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 16), bbox_228273, 'height')
        # Processing the call keyword arguments (line 584)
        kwargs_228275 = {}
        # Getting the type of 'int' (line 584)
        int_228271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 12), 'int', False)
        # Calling int(args, kwargs) (line 584)
        int_call_result_228276 = invoke(stypy.reporting.localization.Localization(__file__, 584, 12), int_228271, *[height_228274], **kwargs_228275)
        
        # Assigning a type to the variable 'h' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'h', int_call_result_228276)
        
        # Assigning a Call to a Name (line 586):
        
        # Assigning a Call to a Name (line 586):
        
        # Call to Window(...): (line 586)
        # Processing the call keyword arguments (line 586)
        kwargs_228279 = {}
        # Getting the type of 'Gtk' (line 586)
        Gtk_228277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 17), 'Gtk', False)
        # Obtaining the member 'Window' of a type (line 586)
        Window_228278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 17), Gtk_228277, 'Window')
        # Calling Window(args, kwargs) (line 586)
        Window_call_result_228280 = invoke(stypy.reporting.localization.Localization(__file__, 586, 17), Window_228278, *[], **kwargs_228279)
        
        # Assigning a type to the variable 'window' (line 586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'window', Window_call_result_228280)
        
        
        # SSA begins for try-except statement (line 587)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to set_icon_from_file(...): (line 588)
        # Processing the call arguments (line 588)
        # Getting the type of 'window_icon' (line 588)
        window_icon_228283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 38), 'window_icon', False)
        # Processing the call keyword arguments (line 588)
        kwargs_228284 = {}
        # Getting the type of 'window' (line 588)
        window_228281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'window', False)
        # Obtaining the member 'set_icon_from_file' of a type (line 588)
        set_icon_from_file_228282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 12), window_228281, 'set_icon_from_file')
        # Calling set_icon_from_file(args, kwargs) (line 588)
        set_icon_from_file_call_result_228285 = invoke(stypy.reporting.localization.Localization(__file__, 588, 12), set_icon_from_file_228282, *[window_icon_228283], **kwargs_228284)
        
        # SSA branch for the except part of a try statement (line 587)
        # SSA branch for the except 'Tuple' branch of a try statement (line 587)
        module_type_store.open_ssa_branch('except')
        # SSA branch for the except '<any exception>' branch of a try statement (line 587)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 587)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_title(...): (line 596)
        # Processing the call arguments (line 596)
        unicode_228288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 25), 'unicode', u'Subplot Configuration Tool')
        # Processing the call keyword arguments (line 596)
        kwargs_228289 = {}
        # Getting the type of 'window' (line 596)
        window_228286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'window', False)
        # Obtaining the member 'set_title' of a type (line 596)
        set_title_228287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 8), window_228286, 'set_title')
        # Calling set_title(args, kwargs) (line 596)
        set_title_call_result_228290 = invoke(stypy.reporting.localization.Localization(__file__, 596, 8), set_title_228287, *[unicode_228288], **kwargs_228289)
        
        
        # Call to set_default_size(...): (line 597)
        # Processing the call arguments (line 597)
        # Getting the type of 'w' (line 597)
        w_228293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 32), 'w', False)
        # Getting the type of 'h' (line 597)
        h_228294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 35), 'h', False)
        # Processing the call keyword arguments (line 597)
        kwargs_228295 = {}
        # Getting the type of 'window' (line 597)
        window_228291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'window', False)
        # Obtaining the member 'set_default_size' of a type (line 597)
        set_default_size_228292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 8), window_228291, 'set_default_size')
        # Calling set_default_size(args, kwargs) (line 597)
        set_default_size_call_result_228296 = invoke(stypy.reporting.localization.Localization(__file__, 597, 8), set_default_size_228292, *[w_228293, h_228294], **kwargs_228295)
        
        
        # Assigning a Call to a Name (line 598):
        
        # Assigning a Call to a Name (line 598):
        
        # Call to Box(...): (line 598)
        # Processing the call keyword arguments (line 598)
        kwargs_228299 = {}
        # Getting the type of 'Gtk' (line 598)
        Gtk_228297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 15), 'Gtk', False)
        # Obtaining the member 'Box' of a type (line 598)
        Box_228298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 15), Gtk_228297, 'Box')
        # Calling Box(args, kwargs) (line 598)
        Box_call_result_228300 = invoke(stypy.reporting.localization.Localization(__file__, 598, 15), Box_228298, *[], **kwargs_228299)
        
        # Assigning a type to the variable 'vbox' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'vbox', Box_call_result_228300)
        
        # Call to set_property(...): (line 599)
        # Processing the call arguments (line 599)
        unicode_228303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 26), 'unicode', u'orientation')
        # Getting the type of 'Gtk' (line 599)
        Gtk_228304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 41), 'Gtk', False)
        # Obtaining the member 'Orientation' of a type (line 599)
        Orientation_228305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 41), Gtk_228304, 'Orientation')
        # Obtaining the member 'VERTICAL' of a type (line 599)
        VERTICAL_228306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 41), Orientation_228305, 'VERTICAL')
        # Processing the call keyword arguments (line 599)
        kwargs_228307 = {}
        # Getting the type of 'vbox' (line 599)
        vbox_228301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'vbox', False)
        # Obtaining the member 'set_property' of a type (line 599)
        set_property_228302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 8), vbox_228301, 'set_property')
        # Calling set_property(args, kwargs) (line 599)
        set_property_call_result_228308 = invoke(stypy.reporting.localization.Localization(__file__, 599, 8), set_property_228302, *[unicode_228303, VERTICAL_228306], **kwargs_228307)
        
        
        # Call to add(...): (line 600)
        # Processing the call arguments (line 600)
        # Getting the type of 'vbox' (line 600)
        vbox_228311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 19), 'vbox', False)
        # Processing the call keyword arguments (line 600)
        kwargs_228312 = {}
        # Getting the type of 'window' (line 600)
        window_228309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'window', False)
        # Obtaining the member 'add' of a type (line 600)
        add_228310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 8), window_228309, 'add')
        # Calling add(args, kwargs) (line 600)
        add_call_result_228313 = invoke(stypy.reporting.localization.Localization(__file__, 600, 8), add_228310, *[vbox_228311], **kwargs_228312)
        
        
        # Call to show(...): (line 601)
        # Processing the call keyword arguments (line 601)
        kwargs_228316 = {}
        # Getting the type of 'vbox' (line 601)
        vbox_228314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'vbox', False)
        # Obtaining the member 'show' of a type (line 601)
        show_228315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 8), vbox_228314, 'show')
        # Calling show(args, kwargs) (line 601)
        show_call_result_228317 = invoke(stypy.reporting.localization.Localization(__file__, 601, 8), show_228315, *[], **kwargs_228316)
        
        
        # Call to show(...): (line 603)
        # Processing the call keyword arguments (line 603)
        kwargs_228320 = {}
        # Getting the type of 'canvas' (line 603)
        canvas_228318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'canvas', False)
        # Obtaining the member 'show' of a type (line 603)
        show_228319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 8), canvas_228318, 'show')
        # Calling show(args, kwargs) (line 603)
        show_call_result_228321 = invoke(stypy.reporting.localization.Localization(__file__, 603, 8), show_228319, *[], **kwargs_228320)
        
        
        # Call to pack_start(...): (line 604)
        # Processing the call arguments (line 604)
        # Getting the type of 'canvas' (line 604)
        canvas_228324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 24), 'canvas', False)
        # Getting the type of 'True' (line 604)
        True_228325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 32), 'True', False)
        # Getting the type of 'True' (line 604)
        True_228326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 38), 'True', False)
        int_228327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 44), 'int')
        # Processing the call keyword arguments (line 604)
        kwargs_228328 = {}
        # Getting the type of 'vbox' (line 604)
        vbox_228322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'vbox', False)
        # Obtaining the member 'pack_start' of a type (line 604)
        pack_start_228323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 8), vbox_228322, 'pack_start')
        # Calling pack_start(args, kwargs) (line 604)
        pack_start_call_result_228329 = invoke(stypy.reporting.localization.Localization(__file__, 604, 8), pack_start_228323, *[canvas_228324, True_228325, True_228326, int_228327], **kwargs_228328)
        
        
        # Call to show(...): (line 605)
        # Processing the call keyword arguments (line 605)
        kwargs_228332 = {}
        # Getting the type of 'window' (line 605)
        window_228330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'window', False)
        # Obtaining the member 'show' of a type (line 605)
        show_228331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 8), window_228330, 'show')
        # Calling show(args, kwargs) (line 605)
        show_call_result_228333 = invoke(stypy.reporting.localization.Localization(__file__, 605, 8), show_228331, *[], **kwargs_228332)
        
        
        # ################# End of 'configure_subplots(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'configure_subplots' in the type store
        # Getting the type of 'stypy_return_type' (line 577)
        stypy_return_type_228334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228334)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'configure_subplots'
        return stypy_return_type_228334


    @norecursion
    def _get_canvas(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_canvas'
        module_type_store = module_type_store.open_function_context('_get_canvas', 607, 4, False)
        # Assigning a type to the variable 'self' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2GTK3._get_canvas.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2GTK3._get_canvas.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2GTK3._get_canvas.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2GTK3._get_canvas.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2GTK3._get_canvas')
        NavigationToolbar2GTK3._get_canvas.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        NavigationToolbar2GTK3._get_canvas.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2GTK3._get_canvas.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2GTK3._get_canvas.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2GTK3._get_canvas.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2GTK3._get_canvas.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2GTK3._get_canvas.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2GTK3._get_canvas', ['fig'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_canvas', localization, ['fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_canvas(...)' code ##################

        
        # Call to __class__(...): (line 608)
        # Processing the call arguments (line 608)
        # Getting the type of 'fig' (line 608)
        fig_228338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 37), 'fig', False)
        # Processing the call keyword arguments (line 608)
        kwargs_228339 = {}
        # Getting the type of 'self' (line 608)
        self_228335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 15), 'self', False)
        # Obtaining the member 'canvas' of a type (line 608)
        canvas_228336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 15), self_228335, 'canvas')
        # Obtaining the member '__class__' of a type (line 608)
        class___228337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 15), canvas_228336, '__class__')
        # Calling __class__(args, kwargs) (line 608)
        class___call_result_228340 = invoke(stypy.reporting.localization.Localization(__file__, 608, 15), class___228337, *[fig_228338], **kwargs_228339)
        
        # Assigning a type to the variable 'stypy_return_type' (line 608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'stypy_return_type', class___call_result_228340)
        
        # ################# End of '_get_canvas(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_canvas' in the type store
        # Getting the type of 'stypy_return_type' (line 607)
        stypy_return_type_228341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228341)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_canvas'
        return stypy_return_type_228341


# Assigning a type to the variable 'NavigationToolbar2GTK3' (line 483)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 0), 'NavigationToolbar2GTK3', NavigationToolbar2GTK3)
# Declaration of the 'FileChooserDialog' class
# Getting the type of 'Gtk' (line 611)
Gtk_228342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 24), 'Gtk')
# Obtaining the member 'FileChooserDialog' of a type (line 611)
FileChooserDialog_228343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 24), Gtk_228342, 'FileChooserDialog')

class FileChooserDialog(FileChooserDialog_228343, ):
    unicode_228344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, (-1)), 'unicode', u'GTK+ file selector which remembers the last file/directory\n    selected and presents the user with a menu of supported image formats\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_228345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 28), 'unicode', u'Save file')
        # Getting the type of 'None' (line 617)
        None_228346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 28), 'None')
        # Getting the type of 'Gtk' (line 618)
        Gtk_228347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 28), 'Gtk')
        # Obtaining the member 'FileChooserAction' of a type (line 618)
        FileChooserAction_228348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 28), Gtk_228347, 'FileChooserAction')
        # Obtaining the member 'SAVE' of a type (line 618)
        SAVE_228349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 28), FileChooserAction_228348, 'SAVE')
        
        # Obtaining an instance of the builtin type 'tuple' (line 619)
        tuple_228350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 619)
        # Adding element type (line 619)
        # Getting the type of 'Gtk' (line 619)
        Gtk_228351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 29), 'Gtk')
        # Obtaining the member 'STOCK_CANCEL' of a type (line 619)
        STOCK_CANCEL_228352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 29), Gtk_228351, 'STOCK_CANCEL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 29), tuple_228350, STOCK_CANCEL_228352)
        # Adding element type (line 619)
        # Getting the type of 'Gtk' (line 619)
        Gtk_228353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 47), 'Gtk')
        # Obtaining the member 'ResponseType' of a type (line 619)
        ResponseType_228354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 47), Gtk_228353, 'ResponseType')
        # Obtaining the member 'CANCEL' of a type (line 619)
        CANCEL_228355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 47), ResponseType_228354, 'CANCEL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 29), tuple_228350, CANCEL_228355)
        # Adding element type (line 619)
        # Getting the type of 'Gtk' (line 620)
        Gtk_228356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 29), 'Gtk')
        # Obtaining the member 'STOCK_SAVE' of a type (line 620)
        STOCK_SAVE_228357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 29), Gtk_228356, 'STOCK_SAVE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 29), tuple_228350, STOCK_SAVE_228357)
        # Adding element type (line 619)
        # Getting the type of 'Gtk' (line 620)
        Gtk_228358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 47), 'Gtk')
        # Obtaining the member 'ResponseType' of a type (line 620)
        ResponseType_228359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 47), Gtk_228358, 'ResponseType')
        # Obtaining the member 'OK' of a type (line 620)
        OK_228360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 47), ResponseType_228359, 'OK')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 29), tuple_228350, OK_228360)
        
        # Getting the type of 'None' (line 621)
        None_228361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 28), 'None')
        
        # Obtaining an instance of the builtin type 'list' (line 622)
        list_228362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 622)
        
        # Getting the type of 'None' (line 623)
        None_228363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 37), 'None')
        defaults = [unicode_228345, None_228346, SAVE_228349, tuple_228350, None_228361, list_228362, None_228363]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 615, 4, False)
        # Assigning a type to the variable 'self' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileChooserDialog.__init__', ['title', 'parent', 'action', 'buttons', 'path', 'filetypes', 'default_filetype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['title', 'parent', 'action', 'buttons', 'path', 'filetypes', 'default_filetype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'title' (line 625)
        title_228370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 50), 'title', False)
        # Getting the type of 'parent' (line 625)
        parent_228371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 57), 'parent', False)
        # Getting the type of 'action' (line 625)
        action_228372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 65), 'action', False)
        # Getting the type of 'buttons' (line 626)
        buttons_228373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 50), 'buttons', False)
        # Processing the call keyword arguments (line 625)
        kwargs_228374 = {}
        
        # Call to super(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'FileChooserDialog' (line 625)
        FileChooserDialog_228365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 'FileChooserDialog', False)
        # Getting the type of 'self' (line 625)
        self_228366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 34), 'self', False)
        # Processing the call keyword arguments (line 625)
        kwargs_228367 = {}
        # Getting the type of 'super' (line 625)
        super_228364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'super', False)
        # Calling super(args, kwargs) (line 625)
        super_call_result_228368 = invoke(stypy.reporting.localization.Localization(__file__, 625, 8), super_228364, *[FileChooserDialog_228365, self_228366], **kwargs_228367)
        
        # Obtaining the member '__init__' of a type (line 625)
        init___228369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 8), super_call_result_228368, '__init__')
        # Calling __init__(args, kwargs) (line 625)
        init___call_result_228375 = invoke(stypy.reporting.localization.Localization(__file__, 625, 8), init___228369, *[title_228370, parent_228371, action_228372, buttons_228373], **kwargs_228374)
        
        
        # Call to set_default_response(...): (line 627)
        # Processing the call arguments (line 627)
        # Getting the type of 'Gtk' (line 627)
        Gtk_228378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 35), 'Gtk', False)
        # Obtaining the member 'ResponseType' of a type (line 627)
        ResponseType_228379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 35), Gtk_228378, 'ResponseType')
        # Obtaining the member 'OK' of a type (line 627)
        OK_228380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 35), ResponseType_228379, 'OK')
        # Processing the call keyword arguments (line 627)
        kwargs_228381 = {}
        # Getting the type of 'self' (line 627)
        self_228376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'self', False)
        # Obtaining the member 'set_default_response' of a type (line 627)
        set_default_response_228377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 8), self_228376, 'set_default_response')
        # Calling set_default_response(args, kwargs) (line 627)
        set_default_response_call_result_228382 = invoke(stypy.reporting.localization.Localization(__file__, 627, 8), set_default_response_228377, *[OK_228380], **kwargs_228381)
        
        
        
        # Getting the type of 'path' (line 629)
        path_228383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 15), 'path')
        # Applying the 'not' unary operator (line 629)
        result_not__228384 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 11), 'not', path_228383)
        
        # Testing the type of an if condition (line 629)
        if_condition_228385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 629, 8), result_not__228384)
        # Assigning a type to the variable 'if_condition_228385' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'if_condition_228385', if_condition_228385)
        # SSA begins for if statement (line 629)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 629):
        
        # Assigning a BinOp to a Name (line 629):
        
        # Call to getcwd(...): (line 629)
        # Processing the call keyword arguments (line 629)
        kwargs_228388 = {}
        # Getting the type of 'os' (line 629)
        os_228386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 28), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 629)
        getcwd_228387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 28), os_228386, 'getcwd')
        # Calling getcwd(args, kwargs) (line 629)
        getcwd_call_result_228389 = invoke(stypy.reporting.localization.Localization(__file__, 629, 28), getcwd_228387, *[], **kwargs_228388)
        
        # Getting the type of 'os' (line 629)
        os_228390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 42), 'os')
        # Obtaining the member 'sep' of a type (line 629)
        sep_228391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 42), os_228390, 'sep')
        # Applying the binary operator '+' (line 629)
        result_add_228392 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 28), '+', getcwd_call_result_228389, sep_228391)
        
        # Assigning a type to the variable 'path' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 21), 'path', result_add_228392)
        # SSA join for if statement (line 629)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_current_folder(...): (line 632)
        # Processing the call arguments (line 632)
        # Getting the type of 'path' (line 632)
        path_228395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 33), 'path', False)
        # Processing the call keyword arguments (line 632)
        kwargs_228396 = {}
        # Getting the type of 'self' (line 632)
        self_228393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'self', False)
        # Obtaining the member 'set_current_folder' of a type (line 632)
        set_current_folder_228394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 8), self_228393, 'set_current_folder')
        # Calling set_current_folder(args, kwargs) (line 632)
        set_current_folder_call_result_228397 = invoke(stypy.reporting.localization.Localization(__file__, 632, 8), set_current_folder_228394, *[path_228395], **kwargs_228396)
        
        
        # Call to set_current_name(...): (line 633)
        # Processing the call arguments (line 633)
        unicode_228400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 31), 'unicode', u'image.')
        # Getting the type of 'default_filetype' (line 633)
        default_filetype_228401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 42), 'default_filetype', False)
        # Applying the binary operator '+' (line 633)
        result_add_228402 = python_operator(stypy.reporting.localization.Localization(__file__, 633, 31), '+', unicode_228400, default_filetype_228401)
        
        # Processing the call keyword arguments (line 633)
        kwargs_228403 = {}
        # Getting the type of 'self' (line 633)
        self_228398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'self', False)
        # Obtaining the member 'set_current_name' of a type (line 633)
        set_current_name_228399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 8), self_228398, 'set_current_name')
        # Calling set_current_name(args, kwargs) (line 633)
        set_current_name_call_result_228404 = invoke(stypy.reporting.localization.Localization(__file__, 633, 8), set_current_name_228399, *[result_add_228402], **kwargs_228403)
        
        
        # Assigning a Call to a Name (line 635):
        
        # Assigning a Call to a Name (line 635):
        
        # Call to Box(...): (line 635)
        # Processing the call keyword arguments (line 635)
        int_228407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 31), 'int')
        keyword_228408 = int_228407
        kwargs_228409 = {'spacing': keyword_228408}
        # Getting the type of 'Gtk' (line 635)
        Gtk_228405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 15), 'Gtk', False)
        # Obtaining the member 'Box' of a type (line 635)
        Box_228406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 15), Gtk_228405, 'Box')
        # Calling Box(args, kwargs) (line 635)
        Box_call_result_228410 = invoke(stypy.reporting.localization.Localization(__file__, 635, 15), Box_228406, *[], **kwargs_228409)
        
        # Assigning a type to the variable 'hbox' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'hbox', Box_call_result_228410)
        
        # Call to pack_start(...): (line 636)
        # Processing the call arguments (line 636)
        
        # Call to Label(...): (line 636)
        # Processing the call keyword arguments (line 636)
        unicode_228415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 40), 'unicode', u'File Format:')
        keyword_228416 = unicode_228415
        kwargs_228417 = {'label': keyword_228416}
        # Getting the type of 'Gtk' (line 636)
        Gtk_228413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 24), 'Gtk', False)
        # Obtaining the member 'Label' of a type (line 636)
        Label_228414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 24), Gtk_228413, 'Label')
        # Calling Label(args, kwargs) (line 636)
        Label_call_result_228418 = invoke(stypy.reporting.localization.Localization(__file__, 636, 24), Label_228414, *[], **kwargs_228417)
        
        # Getting the type of 'False' (line 636)
        False_228419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 57), 'False', False)
        # Getting the type of 'False' (line 636)
        False_228420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 64), 'False', False)
        int_228421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 71), 'int')
        # Processing the call keyword arguments (line 636)
        kwargs_228422 = {}
        # Getting the type of 'hbox' (line 636)
        hbox_228411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'hbox', False)
        # Obtaining the member 'pack_start' of a type (line 636)
        pack_start_228412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 8), hbox_228411, 'pack_start')
        # Calling pack_start(args, kwargs) (line 636)
        pack_start_call_result_228423 = invoke(stypy.reporting.localization.Localization(__file__, 636, 8), pack_start_228412, *[Label_call_result_228418, False_228419, False_228420, int_228421], **kwargs_228422)
        
        
        # Assigning a Call to a Name (line 638):
        
        # Assigning a Call to a Name (line 638):
        
        # Call to ListStore(...): (line 638)
        # Processing the call arguments (line 638)
        # Getting the type of 'GObject' (line 638)
        GObject_228426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 34), 'GObject', False)
        # Obtaining the member 'TYPE_STRING' of a type (line 638)
        TYPE_STRING_228427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 34), GObject_228426, 'TYPE_STRING')
        # Processing the call keyword arguments (line 638)
        kwargs_228428 = {}
        # Getting the type of 'Gtk' (line 638)
        Gtk_228424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 20), 'Gtk', False)
        # Obtaining the member 'ListStore' of a type (line 638)
        ListStore_228425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 20), Gtk_228424, 'ListStore')
        # Calling ListStore(args, kwargs) (line 638)
        ListStore_call_result_228429 = invoke(stypy.reporting.localization.Localization(__file__, 638, 20), ListStore_228425, *[TYPE_STRING_228427], **kwargs_228428)
        
        # Assigning a type to the variable 'liststore' (line 638)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'liststore', ListStore_call_result_228429)
        
        # Assigning a Call to a Name (line 639):
        
        # Assigning a Call to a Name (line 639):
        
        # Call to ComboBox(...): (line 639)
        # Processing the call keyword arguments (line 639)
        kwargs_228432 = {}
        # Getting the type of 'Gtk' (line 639)
        Gtk_228430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 15), 'Gtk', False)
        # Obtaining the member 'ComboBox' of a type (line 639)
        ComboBox_228431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 15), Gtk_228430, 'ComboBox')
        # Calling ComboBox(args, kwargs) (line 639)
        ComboBox_call_result_228433 = invoke(stypy.reporting.localization.Localization(__file__, 639, 15), ComboBox_228431, *[], **kwargs_228432)
        
        # Assigning a type to the variable 'cbox' (line 639)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 8), 'cbox', ComboBox_call_result_228433)
        
        # Call to set_model(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 'liststore' (line 640)
        liststore_228436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 23), 'liststore', False)
        # Processing the call keyword arguments (line 640)
        kwargs_228437 = {}
        # Getting the type of 'cbox' (line 640)
        cbox_228434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'cbox', False)
        # Obtaining the member 'set_model' of a type (line 640)
        set_model_228435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 8), cbox_228434, 'set_model')
        # Calling set_model(args, kwargs) (line 640)
        set_model_call_result_228438 = invoke(stypy.reporting.localization.Localization(__file__, 640, 8), set_model_228435, *[liststore_228436], **kwargs_228437)
        
        
        # Assigning a Call to a Name (line 641):
        
        # Assigning a Call to a Name (line 641):
        
        # Call to CellRendererText(...): (line 641)
        # Processing the call keyword arguments (line 641)
        kwargs_228441 = {}
        # Getting the type of 'Gtk' (line 641)
        Gtk_228439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 15), 'Gtk', False)
        # Obtaining the member 'CellRendererText' of a type (line 641)
        CellRendererText_228440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 15), Gtk_228439, 'CellRendererText')
        # Calling CellRendererText(args, kwargs) (line 641)
        CellRendererText_call_result_228442 = invoke(stypy.reporting.localization.Localization(__file__, 641, 15), CellRendererText_228440, *[], **kwargs_228441)
        
        # Assigning a type to the variable 'cell' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'cell', CellRendererText_call_result_228442)
        
        # Call to pack_start(...): (line 642)
        # Processing the call arguments (line 642)
        # Getting the type of 'cell' (line 642)
        cell_228445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 24), 'cell', False)
        # Getting the type of 'True' (line 642)
        True_228446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 30), 'True', False)
        # Processing the call keyword arguments (line 642)
        kwargs_228447 = {}
        # Getting the type of 'cbox' (line 642)
        cbox_228443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'cbox', False)
        # Obtaining the member 'pack_start' of a type (line 642)
        pack_start_228444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 8), cbox_228443, 'pack_start')
        # Calling pack_start(args, kwargs) (line 642)
        pack_start_call_result_228448 = invoke(stypy.reporting.localization.Localization(__file__, 642, 8), pack_start_228444, *[cell_228445, True_228446], **kwargs_228447)
        
        
        # Call to add_attribute(...): (line 643)
        # Processing the call arguments (line 643)
        # Getting the type of 'cell' (line 643)
        cell_228451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 27), 'cell', False)
        unicode_228452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 33), 'unicode', u'text')
        int_228453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 41), 'int')
        # Processing the call keyword arguments (line 643)
        kwargs_228454 = {}
        # Getting the type of 'cbox' (line 643)
        cbox_228449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'cbox', False)
        # Obtaining the member 'add_attribute' of a type (line 643)
        add_attribute_228450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 8), cbox_228449, 'add_attribute')
        # Calling add_attribute(args, kwargs) (line 643)
        add_attribute_call_result_228455 = invoke(stypy.reporting.localization.Localization(__file__, 643, 8), add_attribute_228450, *[cell_228451, unicode_228452, int_228453], **kwargs_228454)
        
        
        # Call to pack_start(...): (line 644)
        # Processing the call arguments (line 644)
        # Getting the type of 'cbox' (line 644)
        cbox_228458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 24), 'cbox', False)
        # Getting the type of 'False' (line 644)
        False_228459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 30), 'False', False)
        # Getting the type of 'False' (line 644)
        False_228460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 37), 'False', False)
        int_228461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 44), 'int')
        # Processing the call keyword arguments (line 644)
        kwargs_228462 = {}
        # Getting the type of 'hbox' (line 644)
        hbox_228456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'hbox', False)
        # Obtaining the member 'pack_start' of a type (line 644)
        pack_start_228457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 8), hbox_228456, 'pack_start')
        # Calling pack_start(args, kwargs) (line 644)
        pack_start_call_result_228463 = invoke(stypy.reporting.localization.Localization(__file__, 644, 8), pack_start_228457, *[cbox_228458, False_228459, False_228460, int_228461], **kwargs_228462)
        
        
        # Assigning a Name to a Attribute (line 646):
        
        # Assigning a Name to a Attribute (line 646):
        # Getting the type of 'filetypes' (line 646)
        filetypes_228464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 25), 'filetypes')
        # Getting the type of 'self' (line 646)
        self_228465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'self')
        # Setting the type of the member 'filetypes' of a type (line 646)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 8), self_228465, 'filetypes', filetypes_228464)
        
        # Assigning a Call to a Attribute (line 647):
        
        # Assigning a Call to a Attribute (line 647):
        
        # Call to sorted(...): (line 647)
        # Processing the call arguments (line 647)
        
        # Call to iteritems(...): (line 647)
        # Processing the call arguments (line 647)
        # Getting the type of 'filetypes' (line 647)
        filetypes_228469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 53), 'filetypes', False)
        # Processing the call keyword arguments (line 647)
        kwargs_228470 = {}
        # Getting the type of 'six' (line 647)
        six_228467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 39), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 647)
        iteritems_228468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 39), six_228467, 'iteritems')
        # Calling iteritems(args, kwargs) (line 647)
        iteritems_call_result_228471 = invoke(stypy.reporting.localization.Localization(__file__, 647, 39), iteritems_228468, *[filetypes_228469], **kwargs_228470)
        
        # Processing the call keyword arguments (line 647)
        kwargs_228472 = {}
        # Getting the type of 'sorted' (line 647)
        sorted_228466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 32), 'sorted', False)
        # Calling sorted(args, kwargs) (line 647)
        sorted_call_result_228473 = invoke(stypy.reporting.localization.Localization(__file__, 647, 32), sorted_228466, *[iteritems_call_result_228471], **kwargs_228472)
        
        # Getting the type of 'self' (line 647)
        self_228474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'self')
        # Setting the type of the member 'sorted_filetypes' of a type (line 647)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 8), self_228474, 'sorted_filetypes', sorted_call_result_228473)
        
        # Assigning a Num to a Name (line 648):
        
        # Assigning a Num to a Name (line 648):
        int_228475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 18), 'int')
        # Assigning a type to the variable 'default' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'default', int_228475)
        
        
        # Call to enumerate(...): (line 649)
        # Processing the call arguments (line 649)
        # Getting the type of 'self' (line 649)
        self_228477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 40), 'self', False)
        # Obtaining the member 'sorted_filetypes' of a type (line 649)
        sorted_filetypes_228478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 40), self_228477, 'sorted_filetypes')
        # Processing the call keyword arguments (line 649)
        kwargs_228479 = {}
        # Getting the type of 'enumerate' (line 649)
        enumerate_228476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 30), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 649)
        enumerate_call_result_228480 = invoke(stypy.reporting.localization.Localization(__file__, 649, 30), enumerate_228476, *[sorted_filetypes_228478], **kwargs_228479)
        
        # Testing the type of a for loop iterable (line 649)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 649, 8), enumerate_call_result_228480)
        # Getting the type of the for loop variable (line 649)
        for_loop_var_228481 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 649, 8), enumerate_call_result_228480)
        # Assigning a type to the variable 'i' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 8), for_loop_var_228481))
        # Assigning a type to the variable 'ext' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'ext', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 8), for_loop_var_228481))
        # Assigning a type to the variable 'name' (line 649)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 8), for_loop_var_228481))
        # SSA begins for a for statement (line 649)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 650)
        # Processing the call arguments (line 650)
        
        # Obtaining an instance of the builtin type 'list' (line 650)
        list_228484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 650)
        # Adding element type (line 650)
        unicode_228485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 30), 'unicode', u'%s (*.%s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 650)
        tuple_228486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 650)
        # Adding element type (line 650)
        # Getting the type of 'name' (line 650)
        name_228487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 45), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 45), tuple_228486, name_228487)
        # Adding element type (line 650)
        # Getting the type of 'ext' (line 650)
        ext_228488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 51), 'ext', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 45), tuple_228486, ext_228488)
        
        # Applying the binary operator '%' (line 650)
        result_mod_228489 = python_operator(stypy.reporting.localization.Localization(__file__, 650, 30), '%', unicode_228485, tuple_228486)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 650, 29), list_228484, result_mod_228489)
        
        # Processing the call keyword arguments (line 650)
        kwargs_228490 = {}
        # Getting the type of 'liststore' (line 650)
        liststore_228482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 12), 'liststore', False)
        # Obtaining the member 'append' of a type (line 650)
        append_228483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 12), liststore_228482, 'append')
        # Calling append(args, kwargs) (line 650)
        append_call_result_228491 = invoke(stypy.reporting.localization.Localization(__file__, 650, 12), append_228483, *[list_228484], **kwargs_228490)
        
        
        
        # Getting the type of 'ext' (line 651)
        ext_228492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 15), 'ext')
        # Getting the type of 'default_filetype' (line 651)
        default_filetype_228493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 22), 'default_filetype')
        # Applying the binary operator '==' (line 651)
        result_eq_228494 = python_operator(stypy.reporting.localization.Localization(__file__, 651, 15), '==', ext_228492, default_filetype_228493)
        
        # Testing the type of an if condition (line 651)
        if_condition_228495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 651, 12), result_eq_228494)
        # Assigning a type to the variable 'if_condition_228495' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 12), 'if_condition_228495', if_condition_228495)
        # SSA begins for if statement (line 651)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 652):
        
        # Assigning a Name to a Name (line 652):
        # Getting the type of 'i' (line 652)
        i_228496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 26), 'i')
        # Assigning a type to the variable 'default' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 16), 'default', i_228496)
        # SSA join for if statement (line 651)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_active(...): (line 653)
        # Processing the call arguments (line 653)
        # Getting the type of 'default' (line 653)
        default_228499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 24), 'default', False)
        # Processing the call keyword arguments (line 653)
        kwargs_228500 = {}
        # Getting the type of 'cbox' (line 653)
        cbox_228497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'cbox', False)
        # Obtaining the member 'set_active' of a type (line 653)
        set_active_228498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 8), cbox_228497, 'set_active')
        # Calling set_active(args, kwargs) (line 653)
        set_active_call_result_228501 = invoke(stypy.reporting.localization.Localization(__file__, 653, 8), set_active_228498, *[default_228499], **kwargs_228500)
        
        
        # Assigning a Name to a Attribute (line 654):
        
        # Assigning a Name to a Attribute (line 654):
        # Getting the type of 'default_filetype' (line 654)
        default_filetype_228502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 19), 'default_filetype')
        # Getting the type of 'self' (line 654)
        self_228503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'self')
        # Setting the type of the member 'ext' of a type (line 654)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 8), self_228503, 'ext', default_filetype_228502)

        @norecursion
        def cb_cbox_changed(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'None' (line 656)
            None_228504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 40), 'None')
            defaults = [None_228504]
            # Create a new context for function 'cb_cbox_changed'
            module_type_store = module_type_store.open_function_context('cb_cbox_changed', 656, 8, False)
            
            # Passed parameters checking function
            cb_cbox_changed.stypy_localization = localization
            cb_cbox_changed.stypy_type_of_self = None
            cb_cbox_changed.stypy_type_store = module_type_store
            cb_cbox_changed.stypy_function_name = 'cb_cbox_changed'
            cb_cbox_changed.stypy_param_names_list = ['cbox', 'data']
            cb_cbox_changed.stypy_varargs_param_name = None
            cb_cbox_changed.stypy_kwargs_param_name = None
            cb_cbox_changed.stypy_call_defaults = defaults
            cb_cbox_changed.stypy_call_varargs = varargs
            cb_cbox_changed.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cb_cbox_changed', ['cbox', 'data'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cb_cbox_changed', localization, ['cbox', 'data'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cb_cbox_changed(...)' code ##################

            unicode_228505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 12), 'unicode', u'File extension changed')
            
            # Assigning a Call to a Tuple (line 658):
            
            # Assigning a Call to a Name:
            
            # Call to split(...): (line 658)
            # Processing the call arguments (line 658)
            
            # Call to get_filename(...): (line 658)
            # Processing the call keyword arguments (line 658)
            kwargs_228511 = {}
            # Getting the type of 'self' (line 658)
            self_228509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 43), 'self', False)
            # Obtaining the member 'get_filename' of a type (line 658)
            get_filename_228510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 43), self_228509, 'get_filename')
            # Calling get_filename(args, kwargs) (line 658)
            get_filename_call_result_228512 = invoke(stypy.reporting.localization.Localization(__file__, 658, 43), get_filename_228510, *[], **kwargs_228511)
            
            # Processing the call keyword arguments (line 658)
            kwargs_228513 = {}
            # Getting the type of 'os' (line 658)
            os_228506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 29), 'os', False)
            # Obtaining the member 'path' of a type (line 658)
            path_228507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 29), os_228506, 'path')
            # Obtaining the member 'split' of a type (line 658)
            split_228508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 29), path_228507, 'split')
            # Calling split(args, kwargs) (line 658)
            split_call_result_228514 = invoke(stypy.reporting.localization.Localization(__file__, 658, 29), split_228508, *[get_filename_call_result_228512], **kwargs_228513)
            
            # Assigning a type to the variable 'call_assignment_226578' (line 658)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'call_assignment_226578', split_call_result_228514)
            
            # Assigning a Call to a Name (line 658):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_228517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 12), 'int')
            # Processing the call keyword arguments
            kwargs_228518 = {}
            # Getting the type of 'call_assignment_226578' (line 658)
            call_assignment_226578_228515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'call_assignment_226578', False)
            # Obtaining the member '__getitem__' of a type (line 658)
            getitem___228516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 12), call_assignment_226578_228515, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_228519 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___228516, *[int_228517], **kwargs_228518)
            
            # Assigning a type to the variable 'call_assignment_226579' (line 658)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'call_assignment_226579', getitem___call_result_228519)
            
            # Assigning a Name to a Name (line 658):
            # Getting the type of 'call_assignment_226579' (line 658)
            call_assignment_226579_228520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'call_assignment_226579')
            # Assigning a type to the variable 'head' (line 658)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'head', call_assignment_226579_228520)
            
            # Assigning a Call to a Name (line 658):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_228523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 12), 'int')
            # Processing the call keyword arguments
            kwargs_228524 = {}
            # Getting the type of 'call_assignment_226578' (line 658)
            call_assignment_226578_228521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'call_assignment_226578', False)
            # Obtaining the member '__getitem__' of a type (line 658)
            getitem___228522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 12), call_assignment_226578_228521, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_228525 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___228522, *[int_228523], **kwargs_228524)
            
            # Assigning a type to the variable 'call_assignment_226580' (line 658)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'call_assignment_226580', getitem___call_result_228525)
            
            # Assigning a Name to a Name (line 658):
            # Getting the type of 'call_assignment_226580' (line 658)
            call_assignment_226580_228526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'call_assignment_226580')
            # Assigning a type to the variable 'filename' (line 658)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 18), 'filename', call_assignment_226580_228526)
            
            # Assigning a Call to a Tuple (line 659):
            
            # Assigning a Call to a Name:
            
            # Call to splitext(...): (line 659)
            # Processing the call arguments (line 659)
            # Getting the type of 'filename' (line 659)
            filename_228530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 41), 'filename', False)
            # Processing the call keyword arguments (line 659)
            kwargs_228531 = {}
            # Getting the type of 'os' (line 659)
            os_228527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 24), 'os', False)
            # Obtaining the member 'path' of a type (line 659)
            path_228528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 24), os_228527, 'path')
            # Obtaining the member 'splitext' of a type (line 659)
            splitext_228529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 24), path_228528, 'splitext')
            # Calling splitext(args, kwargs) (line 659)
            splitext_call_result_228532 = invoke(stypy.reporting.localization.Localization(__file__, 659, 24), splitext_228529, *[filename_228530], **kwargs_228531)
            
            # Assigning a type to the variable 'call_assignment_226581' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'call_assignment_226581', splitext_call_result_228532)
            
            # Assigning a Call to a Name (line 659):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_228535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 12), 'int')
            # Processing the call keyword arguments
            kwargs_228536 = {}
            # Getting the type of 'call_assignment_226581' (line 659)
            call_assignment_226581_228533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'call_assignment_226581', False)
            # Obtaining the member '__getitem__' of a type (line 659)
            getitem___228534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 12), call_assignment_226581_228533, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_228537 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___228534, *[int_228535], **kwargs_228536)
            
            # Assigning a type to the variable 'call_assignment_226582' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'call_assignment_226582', getitem___call_result_228537)
            
            # Assigning a Name to a Name (line 659):
            # Getting the type of 'call_assignment_226582' (line 659)
            call_assignment_226582_228538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'call_assignment_226582')
            # Assigning a type to the variable 'root' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'root', call_assignment_226582_228538)
            
            # Assigning a Call to a Name (line 659):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_228541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 12), 'int')
            # Processing the call keyword arguments
            kwargs_228542 = {}
            # Getting the type of 'call_assignment_226581' (line 659)
            call_assignment_226581_228539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'call_assignment_226581', False)
            # Obtaining the member '__getitem__' of a type (line 659)
            getitem___228540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 12), call_assignment_226581_228539, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_228543 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___228540, *[int_228541], **kwargs_228542)
            
            # Assigning a type to the variable 'call_assignment_226583' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'call_assignment_226583', getitem___call_result_228543)
            
            # Assigning a Name to a Name (line 659):
            # Getting the type of 'call_assignment_226583' (line 659)
            call_assignment_226583_228544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'call_assignment_226583')
            # Assigning a type to the variable 'ext' (line 659)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 18), 'ext', call_assignment_226583_228544)
            
            # Assigning a Subscript to a Name (line 660):
            
            # Assigning a Subscript to a Name (line 660):
            
            # Obtaining the type of the subscript
            int_228545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 22), 'int')
            slice_228546 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 660, 18), int_228545, None, None)
            # Getting the type of 'ext' (line 660)
            ext_228547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 18), 'ext')
            # Obtaining the member '__getitem__' of a type (line 660)
            getitem___228548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 18), ext_228547, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 660)
            subscript_call_result_228549 = invoke(stypy.reporting.localization.Localization(__file__, 660, 18), getitem___228548, slice_228546)
            
            # Assigning a type to the variable 'ext' (line 660)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 12), 'ext', subscript_call_result_228549)
            
            # Assigning a Subscript to a Name (line 661):
            
            # Assigning a Subscript to a Name (line 661):
            
            # Obtaining the type of the subscript
            int_228550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 63), 'int')
            
            # Obtaining the type of the subscript
            
            # Call to get_active(...): (line 661)
            # Processing the call keyword arguments (line 661)
            kwargs_228553 = {}
            # Getting the type of 'cbox' (line 661)
            cbox_228551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 44), 'cbox', False)
            # Obtaining the member 'get_active' of a type (line 661)
            get_active_228552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 44), cbox_228551, 'get_active')
            # Calling get_active(args, kwargs) (line 661)
            get_active_call_result_228554 = invoke(stypy.reporting.localization.Localization(__file__, 661, 44), get_active_228552, *[], **kwargs_228553)
            
            # Getting the type of 'self' (line 661)
            self_228555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 22), 'self')
            # Obtaining the member 'sorted_filetypes' of a type (line 661)
            sorted_filetypes_228556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 22), self_228555, 'sorted_filetypes')
            # Obtaining the member '__getitem__' of a type (line 661)
            getitem___228557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 22), sorted_filetypes_228556, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 661)
            subscript_call_result_228558 = invoke(stypy.reporting.localization.Localization(__file__, 661, 22), getitem___228557, get_active_call_result_228554)
            
            # Obtaining the member '__getitem__' of a type (line 661)
            getitem___228559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 22), subscript_call_result_228558, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 661)
            subscript_call_result_228560 = invoke(stypy.reporting.localization.Localization(__file__, 661, 22), getitem___228559, int_228550)
            
            # Assigning a type to the variable 'new_ext' (line 661)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'new_ext', subscript_call_result_228560)
            
            # Assigning a Name to a Attribute (line 662):
            
            # Assigning a Name to a Attribute (line 662):
            # Getting the type of 'new_ext' (line 662)
            new_ext_228561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 23), 'new_ext')
            # Getting the type of 'self' (line 662)
            self_228562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 12), 'self')
            # Setting the type of the member 'ext' of a type (line 662)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 12), self_228562, 'ext', new_ext_228561)
            
            
            # Getting the type of 'ext' (line 664)
            ext_228563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 15), 'ext')
            # Getting the type of 'self' (line 664)
            self_228564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 22), 'self')
            # Obtaining the member 'filetypes' of a type (line 664)
            filetypes_228565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 22), self_228564, 'filetypes')
            # Applying the binary operator 'in' (line 664)
            result_contains_228566 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 15), 'in', ext_228563, filetypes_228565)
            
            # Testing the type of an if condition (line 664)
            if_condition_228567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 664, 12), result_contains_228566)
            # Assigning a type to the variable 'if_condition_228567' (line 664)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'if_condition_228567', if_condition_228567)
            # SSA begins for if statement (line 664)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 665):
            
            # Assigning a BinOp to a Name (line 665):
            # Getting the type of 'root' (line 665)
            root_228568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 27), 'root')
            unicode_228569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 34), 'unicode', u'.')
            # Applying the binary operator '+' (line 665)
            result_add_228570 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 27), '+', root_228568, unicode_228569)
            
            # Getting the type of 'new_ext' (line 665)
            new_ext_228571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 40), 'new_ext')
            # Applying the binary operator '+' (line 665)
            result_add_228572 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 38), '+', result_add_228570, new_ext_228571)
            
            # Assigning a type to the variable 'filename' (line 665)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 16), 'filename', result_add_228572)
            # SSA branch for the else part of an if statement (line 664)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'ext' (line 666)
            ext_228573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 17), 'ext')
            unicode_228574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 24), 'unicode', u'')
            # Applying the binary operator '==' (line 666)
            result_eq_228575 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 17), '==', ext_228573, unicode_228574)
            
            # Testing the type of an if condition (line 666)
            if_condition_228576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 17), result_eq_228575)
            # Assigning a type to the variable 'if_condition_228576' (line 666)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 17), 'if_condition_228576', if_condition_228576)
            # SSA begins for if statement (line 666)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 667):
            
            # Assigning a BinOp to a Name (line 667):
            
            # Call to rstrip(...): (line 667)
            # Processing the call arguments (line 667)
            unicode_228579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 43), 'unicode', u'.')
            # Processing the call keyword arguments (line 667)
            kwargs_228580 = {}
            # Getting the type of 'filename' (line 667)
            filename_228577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 27), 'filename', False)
            # Obtaining the member 'rstrip' of a type (line 667)
            rstrip_228578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 27), filename_228577, 'rstrip')
            # Calling rstrip(args, kwargs) (line 667)
            rstrip_call_result_228581 = invoke(stypy.reporting.localization.Localization(__file__, 667, 27), rstrip_228578, *[unicode_228579], **kwargs_228580)
            
            unicode_228582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 50), 'unicode', u'.')
            # Applying the binary operator '+' (line 667)
            result_add_228583 = python_operator(stypy.reporting.localization.Localization(__file__, 667, 27), '+', rstrip_call_result_228581, unicode_228582)
            
            # Getting the type of 'new_ext' (line 667)
            new_ext_228584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 56), 'new_ext')
            # Applying the binary operator '+' (line 667)
            result_add_228585 = python_operator(stypy.reporting.localization.Localization(__file__, 667, 54), '+', result_add_228583, new_ext_228584)
            
            # Assigning a type to the variable 'filename' (line 667)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'filename', result_add_228585)
            # SSA join for if statement (line 666)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 664)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to set_current_name(...): (line 669)
            # Processing the call arguments (line 669)
            # Getting the type of 'filename' (line 669)
            filename_228588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 35), 'filename', False)
            # Processing the call keyword arguments (line 669)
            kwargs_228589 = {}
            # Getting the type of 'self' (line 669)
            self_228586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'self', False)
            # Obtaining the member 'set_current_name' of a type (line 669)
            set_current_name_228587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 12), self_228586, 'set_current_name')
            # Calling set_current_name(args, kwargs) (line 669)
            set_current_name_call_result_228590 = invoke(stypy.reporting.localization.Localization(__file__, 669, 12), set_current_name_228587, *[filename_228588], **kwargs_228589)
            
            
            # ################# End of 'cb_cbox_changed(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cb_cbox_changed' in the type store
            # Getting the type of 'stypy_return_type' (line 656)
            stypy_return_type_228591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_228591)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cb_cbox_changed'
            return stypy_return_type_228591

        # Assigning a type to the variable 'cb_cbox_changed' (line 656)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'cb_cbox_changed', cb_cbox_changed)
        
        # Call to connect(...): (line 670)
        # Processing the call arguments (line 670)
        unicode_228594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 22), 'unicode', u'changed')
        # Getting the type of 'cb_cbox_changed' (line 670)
        cb_cbox_changed_228595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 33), 'cb_cbox_changed', False)
        # Processing the call keyword arguments (line 670)
        kwargs_228596 = {}
        # Getting the type of 'cbox' (line 670)
        cbox_228592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'cbox', False)
        # Obtaining the member 'connect' of a type (line 670)
        connect_228593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 8), cbox_228592, 'connect')
        # Calling connect(args, kwargs) (line 670)
        connect_call_result_228597 = invoke(stypy.reporting.localization.Localization(__file__, 670, 8), connect_228593, *[unicode_228594, cb_cbox_changed_228595], **kwargs_228596)
        
        
        # Call to show_all(...): (line 672)
        # Processing the call keyword arguments (line 672)
        kwargs_228600 = {}
        # Getting the type of 'hbox' (line 672)
        hbox_228598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'hbox', False)
        # Obtaining the member 'show_all' of a type (line 672)
        show_all_228599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 8), hbox_228598, 'show_all')
        # Calling show_all(args, kwargs) (line 672)
        show_all_call_result_228601 = invoke(stypy.reporting.localization.Localization(__file__, 672, 8), show_all_228599, *[], **kwargs_228600)
        
        
        # Call to set_extra_widget(...): (line 673)
        # Processing the call arguments (line 673)
        # Getting the type of 'hbox' (line 673)
        hbox_228604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 30), 'hbox', False)
        # Processing the call keyword arguments (line 673)
        kwargs_228605 = {}
        # Getting the type of 'self' (line 673)
        self_228602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'self', False)
        # Obtaining the member 'set_extra_widget' of a type (line 673)
        set_extra_widget_228603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 8), self_228602, 'set_extra_widget')
        # Calling set_extra_widget(args, kwargs) (line 673)
        set_extra_widget_call_result_228606 = invoke(stypy.reporting.localization.Localization(__file__, 673, 8), set_extra_widget_228603, *[hbox_228604], **kwargs_228605)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_filename_from_user(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_filename_from_user'
        module_type_store = module_type_store.open_function_context('get_filename_from_user', 675, 4, False)
        # Assigning a type to the variable 'self' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileChooserDialog.get_filename_from_user.__dict__.__setitem__('stypy_localization', localization)
        FileChooserDialog.get_filename_from_user.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileChooserDialog.get_filename_from_user.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileChooserDialog.get_filename_from_user.__dict__.__setitem__('stypy_function_name', 'FileChooserDialog.get_filename_from_user')
        FileChooserDialog.get_filename_from_user.__dict__.__setitem__('stypy_param_names_list', [])
        FileChooserDialog.get_filename_from_user.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileChooserDialog.get_filename_from_user.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileChooserDialog.get_filename_from_user.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileChooserDialog.get_filename_from_user.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileChooserDialog.get_filename_from_user.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileChooserDialog.get_filename_from_user.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileChooserDialog.get_filename_from_user', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_filename_from_user', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_filename_from_user(...)' code ##################

        
        # Getting the type of 'True' (line 676)
        True_228607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 14), 'True')
        # Testing the type of an if condition (line 676)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 676, 8), True_228607)
        # SSA begins for while statement (line 676)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Name to a Name (line 677):
        
        # Assigning a Name to a Name (line 677):
        # Getting the type of 'None' (line 677)
        None_228608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 23), 'None')
        # Assigning a type to the variable 'filename' (line 677)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 12), 'filename', None_228608)
        
        
        
        # Call to run(...): (line 678)
        # Processing the call keyword arguments (line 678)
        kwargs_228611 = {}
        # Getting the type of 'self' (line 678)
        self_228609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 15), 'self', False)
        # Obtaining the member 'run' of a type (line 678)
        run_228610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 15), self_228609, 'run')
        # Calling run(args, kwargs) (line 678)
        run_call_result_228612 = invoke(stypy.reporting.localization.Localization(__file__, 678, 15), run_228610, *[], **kwargs_228611)
        
        
        # Call to int(...): (line 678)
        # Processing the call arguments (line 678)
        # Getting the type of 'Gtk' (line 678)
        Gtk_228614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 33), 'Gtk', False)
        # Obtaining the member 'ResponseType' of a type (line 678)
        ResponseType_228615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 33), Gtk_228614, 'ResponseType')
        # Obtaining the member 'OK' of a type (line 678)
        OK_228616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 33), ResponseType_228615, 'OK')
        # Processing the call keyword arguments (line 678)
        kwargs_228617 = {}
        # Getting the type of 'int' (line 678)
        int_228613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 29), 'int', False)
        # Calling int(args, kwargs) (line 678)
        int_call_result_228618 = invoke(stypy.reporting.localization.Localization(__file__, 678, 29), int_228613, *[OK_228616], **kwargs_228617)
        
        # Applying the binary operator '!=' (line 678)
        result_ne_228619 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 15), '!=', run_call_result_228612, int_call_result_228618)
        
        # Testing the type of an if condition (line 678)
        if_condition_228620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 678, 12), result_ne_228619)
        # Assigning a type to the variable 'if_condition_228620' (line 678)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 12), 'if_condition_228620', if_condition_228620)
        # SSA begins for if statement (line 678)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 678)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 680):
        
        # Assigning a Call to a Name (line 680):
        
        # Call to get_filename(...): (line 680)
        # Processing the call keyword arguments (line 680)
        kwargs_228623 = {}
        # Getting the type of 'self' (line 680)
        self_228621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 23), 'self', False)
        # Obtaining the member 'get_filename' of a type (line 680)
        get_filename_228622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 23), self_228621, 'get_filename')
        # Calling get_filename(args, kwargs) (line 680)
        get_filename_call_result_228624 = invoke(stypy.reporting.localization.Localization(__file__, 680, 23), get_filename_228622, *[], **kwargs_228623)
        
        # Assigning a type to the variable 'filename' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 12), 'filename', get_filename_call_result_228624)
        # SSA join for while statement (line 676)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 683)
        tuple_228625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 683)
        # Adding element type (line 683)
        # Getting the type of 'filename' (line 683)
        filename_228626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 15), 'filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 683, 15), tuple_228625, filename_228626)
        # Adding element type (line 683)
        # Getting the type of 'self' (line 683)
        self_228627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 25), 'self')
        # Obtaining the member 'ext' of a type (line 683)
        ext_228628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 25), self_228627, 'ext')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 683, 15), tuple_228625, ext_228628)
        
        # Assigning a type to the variable 'stypy_return_type' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'stypy_return_type', tuple_228625)
        
        # ################# End of 'get_filename_from_user(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_filename_from_user' in the type store
        # Getting the type of 'stypy_return_type' (line 675)
        stypy_return_type_228629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228629)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_filename_from_user'
        return stypy_return_type_228629


# Assigning a type to the variable 'FileChooserDialog' (line 611)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 0), 'FileChooserDialog', FileChooserDialog)
# Declaration of the 'RubberbandGTK3' class
# Getting the type of 'backend_tools' (line 686)
backend_tools_228630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 21), 'backend_tools')
# Obtaining the member 'RubberbandBase' of a type (line 686)
RubberbandBase_228631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 21), backend_tools_228630, 'RubberbandBase')

class RubberbandGTK3(RubberbandBase_228631, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 687, 4, False)
        # Assigning a type to the variable 'self' (line 688)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RubberbandGTK3.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 688)
        # Processing the call arguments (line 688)
        # Getting the type of 'self' (line 688)
        self_228635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 46), 'self', False)
        # Getting the type of 'args' (line 688)
        args_228636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 53), 'args', False)
        # Processing the call keyword arguments (line 688)
        # Getting the type of 'kwargs' (line 688)
        kwargs_228637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 61), 'kwargs', False)
        kwargs_228638 = {'kwargs_228637': kwargs_228637}
        # Getting the type of 'backend_tools' (line 688)
        backend_tools_228632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 8), 'backend_tools', False)
        # Obtaining the member 'RubberbandBase' of a type (line 688)
        RubberbandBase_228633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 8), backend_tools_228632, 'RubberbandBase')
        # Obtaining the member '__init__' of a type (line 688)
        init___228634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 8), RubberbandBase_228633, '__init__')
        # Calling __init__(args, kwargs) (line 688)
        init___call_result_228639 = invoke(stypy.reporting.localization.Localization(__file__, 688, 8), init___228634, *[self_228635, args_228636], **kwargs_228638)
        
        
        # Assigning a Name to a Attribute (line 689):
        
        # Assigning a Name to a Attribute (line 689):
        # Getting the type of 'None' (line 689)
        None_228640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 19), 'None')
        # Getting the type of 'self' (line 689)
        self_228641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'self')
        # Setting the type of the member 'ctx' of a type (line 689)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), self_228641, 'ctx', None_228640)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def draw_rubberband(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_rubberband'
        module_type_store = module_type_store.open_function_context('draw_rubberband', 691, 4, False)
        # Assigning a type to the variable 'self' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RubberbandGTK3.draw_rubberband.__dict__.__setitem__('stypy_localization', localization)
        RubberbandGTK3.draw_rubberband.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RubberbandGTK3.draw_rubberband.__dict__.__setitem__('stypy_type_store', module_type_store)
        RubberbandGTK3.draw_rubberband.__dict__.__setitem__('stypy_function_name', 'RubberbandGTK3.draw_rubberband')
        RubberbandGTK3.draw_rubberband.__dict__.__setitem__('stypy_param_names_list', ['x0', 'y0', 'x1', 'y1'])
        RubberbandGTK3.draw_rubberband.__dict__.__setitem__('stypy_varargs_param_name', None)
        RubberbandGTK3.draw_rubberband.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RubberbandGTK3.draw_rubberband.__dict__.__setitem__('stypy_call_defaults', defaults)
        RubberbandGTK3.draw_rubberband.__dict__.__setitem__('stypy_call_varargs', varargs)
        RubberbandGTK3.draw_rubberband.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RubberbandGTK3.draw_rubberband.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RubberbandGTK3.draw_rubberband', ['x0', 'y0', 'x1', 'y1'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'draw_rubberband', localization, ['x0', 'y0', 'x1', 'y1'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'draw_rubberband(...)' code ##################

        
        # Assigning a Call to a Attribute (line 694):
        
        # Assigning a Call to a Attribute (line 694):
        
        # Call to cairo_create(...): (line 694)
        # Processing the call keyword arguments (line 694)
        kwargs_228650 = {}
        
        # Call to get_property(...): (line 694)
        # Processing the call arguments (line 694)
        unicode_228646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 51), 'unicode', u'window')
        # Processing the call keyword arguments (line 694)
        kwargs_228647 = {}
        # Getting the type of 'self' (line 694)
        self_228642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 19), 'self', False)
        # Obtaining the member 'figure' of a type (line 694)
        figure_228643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 19), self_228642, 'figure')
        # Obtaining the member 'canvas' of a type (line 694)
        canvas_228644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 19), figure_228643, 'canvas')
        # Obtaining the member 'get_property' of a type (line 694)
        get_property_228645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 19), canvas_228644, 'get_property')
        # Calling get_property(args, kwargs) (line 694)
        get_property_call_result_228648 = invoke(stypy.reporting.localization.Localization(__file__, 694, 19), get_property_228645, *[unicode_228646], **kwargs_228647)
        
        # Obtaining the member 'cairo_create' of a type (line 694)
        cairo_create_228649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 19), get_property_call_result_228648, 'cairo_create')
        # Calling cairo_create(args, kwargs) (line 694)
        cairo_create_call_result_228651 = invoke(stypy.reporting.localization.Localization(__file__, 694, 19), cairo_create_228649, *[], **kwargs_228650)
        
        # Getting the type of 'self' (line 694)
        self_228652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'self')
        # Setting the type of the member 'ctx' of a type (line 694)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 8), self_228652, 'ctx', cairo_create_call_result_228651)
        
        # Call to draw(...): (line 698)
        # Processing the call keyword arguments (line 698)
        kwargs_228657 = {}
        # Getting the type of 'self' (line 698)
        self_228653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 698)
        figure_228654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), self_228653, 'figure')
        # Obtaining the member 'canvas' of a type (line 698)
        canvas_228655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), figure_228654, 'canvas')
        # Obtaining the member 'draw' of a type (line 698)
        draw_228656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), canvas_228655, 'draw')
        # Calling draw(args, kwargs) (line 698)
        draw_call_result_228658 = invoke(stypy.reporting.localization.Localization(__file__, 698, 8), draw_228656, *[], **kwargs_228657)
        
        
        # Assigning a Attribute to a Name (line 700):
        
        # Assigning a Attribute to a Name (line 700):
        # Getting the type of 'self' (line 700)
        self_228659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 17), 'self')
        # Obtaining the member 'figure' of a type (line 700)
        figure_228660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 17), self_228659, 'figure')
        # Obtaining the member 'bbox' of a type (line 700)
        bbox_228661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 17), figure_228660, 'bbox')
        # Obtaining the member 'height' of a type (line 700)
        height_228662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 17), bbox_228661, 'height')
        # Assigning a type to the variable 'height' (line 700)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 8), 'height', height_228662)
        
        # Assigning a BinOp to a Name (line 701):
        
        # Assigning a BinOp to a Name (line 701):
        # Getting the type of 'height' (line 701)
        height_228663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 13), 'height')
        # Getting the type of 'y1' (line 701)
        y1_228664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 22), 'y1')
        # Applying the binary operator '-' (line 701)
        result_sub_228665 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 13), '-', height_228663, y1_228664)
        
        # Assigning a type to the variable 'y1' (line 701)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'y1', result_sub_228665)
        
        # Assigning a BinOp to a Name (line 702):
        
        # Assigning a BinOp to a Name (line 702):
        # Getting the type of 'height' (line 702)
        height_228666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 13), 'height')
        # Getting the type of 'y0' (line 702)
        y0_228667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 22), 'y0')
        # Applying the binary operator '-' (line 702)
        result_sub_228668 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 13), '-', height_228666, y0_228667)
        
        # Assigning a type to the variable 'y0' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'y0', result_sub_228668)
        
        # Assigning a Call to a Name (line 703):
        
        # Assigning a Call to a Name (line 703):
        
        # Call to abs(...): (line 703)
        # Processing the call arguments (line 703)
        # Getting the type of 'x1' (line 703)
        x1_228670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 16), 'x1', False)
        # Getting the type of 'x0' (line 703)
        x0_228671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 21), 'x0', False)
        # Applying the binary operator '-' (line 703)
        result_sub_228672 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 16), '-', x1_228670, x0_228671)
        
        # Processing the call keyword arguments (line 703)
        kwargs_228673 = {}
        # Getting the type of 'abs' (line 703)
        abs_228669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 12), 'abs', False)
        # Calling abs(args, kwargs) (line 703)
        abs_call_result_228674 = invoke(stypy.reporting.localization.Localization(__file__, 703, 12), abs_228669, *[result_sub_228672], **kwargs_228673)
        
        # Assigning a type to the variable 'w' (line 703)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'w', abs_call_result_228674)
        
        # Assigning a Call to a Name (line 704):
        
        # Assigning a Call to a Name (line 704):
        
        # Call to abs(...): (line 704)
        # Processing the call arguments (line 704)
        # Getting the type of 'y1' (line 704)
        y1_228676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 16), 'y1', False)
        # Getting the type of 'y0' (line 704)
        y0_228677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 21), 'y0', False)
        # Applying the binary operator '-' (line 704)
        result_sub_228678 = python_operator(stypy.reporting.localization.Localization(__file__, 704, 16), '-', y1_228676, y0_228677)
        
        # Processing the call keyword arguments (line 704)
        kwargs_228679 = {}
        # Getting the type of 'abs' (line 704)
        abs_228675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 12), 'abs', False)
        # Calling abs(args, kwargs) (line 704)
        abs_call_result_228680 = invoke(stypy.reporting.localization.Localization(__file__, 704, 12), abs_228675, *[result_sub_228678], **kwargs_228679)
        
        # Assigning a type to the variable 'h' (line 704)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'h', abs_call_result_228680)
        
        # Assigning a ListComp to a Name (line 705):
        
        # Assigning a ListComp to a Name (line 705):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 705)
        tuple_228685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 705)
        # Adding element type (line 705)
        
        # Call to min(...): (line 705)
        # Processing the call arguments (line 705)
        # Getting the type of 'x0' (line 705)
        x0_228687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 41), 'x0', False)
        # Getting the type of 'x1' (line 705)
        x1_228688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 45), 'x1', False)
        # Processing the call keyword arguments (line 705)
        kwargs_228689 = {}
        # Getting the type of 'min' (line 705)
        min_228686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 37), 'min', False)
        # Calling min(args, kwargs) (line 705)
        min_call_result_228690 = invoke(stypy.reporting.localization.Localization(__file__, 705, 37), min_228686, *[x0_228687, x1_228688], **kwargs_228689)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 37), tuple_228685, min_call_result_228690)
        # Adding element type (line 705)
        
        # Call to min(...): (line 705)
        # Processing the call arguments (line 705)
        # Getting the type of 'y0' (line 705)
        y0_228692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 54), 'y0', False)
        # Getting the type of 'y1' (line 705)
        y1_228693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 58), 'y1', False)
        # Processing the call keyword arguments (line 705)
        kwargs_228694 = {}
        # Getting the type of 'min' (line 705)
        min_228691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 50), 'min', False)
        # Calling min(args, kwargs) (line 705)
        min_call_result_228695 = invoke(stypy.reporting.localization.Localization(__file__, 705, 50), min_228691, *[y0_228692, y1_228693], **kwargs_228694)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 37), tuple_228685, min_call_result_228695)
        # Adding element type (line 705)
        # Getting the type of 'w' (line 705)
        w_228696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 63), 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 37), tuple_228685, w_228696)
        # Adding element type (line 705)
        # Getting the type of 'h' (line 705)
        h_228697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 66), 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 37), tuple_228685, h_228697)
        
        comprehension_228698 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 16), tuple_228685)
        # Assigning a type to the variable 'val' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 16), 'val', comprehension_228698)
        
        # Call to int(...): (line 705)
        # Processing the call arguments (line 705)
        # Getting the type of 'val' (line 705)
        val_228682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 20), 'val', False)
        # Processing the call keyword arguments (line 705)
        kwargs_228683 = {}
        # Getting the type of 'int' (line 705)
        int_228681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 16), 'int', False)
        # Calling int(args, kwargs) (line 705)
        int_call_result_228684 = invoke(stypy.reporting.localization.Localization(__file__, 705, 16), int_228681, *[val_228682], **kwargs_228683)
        
        list_228699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 705, 16), list_228699, int_call_result_228684)
        # Assigning a type to the variable 'rect' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'rect', list_228699)
        
        # Call to new_path(...): (line 707)
        # Processing the call keyword arguments (line 707)
        kwargs_228703 = {}
        # Getting the type of 'self' (line 707)
        self_228700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 707)
        ctx_228701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 8), self_228700, 'ctx')
        # Obtaining the member 'new_path' of a type (line 707)
        new_path_228702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 8), ctx_228701, 'new_path')
        # Calling new_path(args, kwargs) (line 707)
        new_path_call_result_228704 = invoke(stypy.reporting.localization.Localization(__file__, 707, 8), new_path_228702, *[], **kwargs_228703)
        
        
        # Call to set_line_width(...): (line 708)
        # Processing the call arguments (line 708)
        float_228708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 32), 'float')
        # Processing the call keyword arguments (line 708)
        kwargs_228709 = {}
        # Getting the type of 'self' (line 708)
        self_228705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 708)
        ctx_228706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 8), self_228705, 'ctx')
        # Obtaining the member 'set_line_width' of a type (line 708)
        set_line_width_228707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 8), ctx_228706, 'set_line_width')
        # Calling set_line_width(args, kwargs) (line 708)
        set_line_width_call_result_228710 = invoke(stypy.reporting.localization.Localization(__file__, 708, 8), set_line_width_228707, *[float_228708], **kwargs_228709)
        
        
        # Call to rectangle(...): (line 709)
        # Processing the call arguments (line 709)
        
        # Obtaining the type of the subscript
        int_228714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 32), 'int')
        # Getting the type of 'rect' (line 709)
        rect_228715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 27), 'rect', False)
        # Obtaining the member '__getitem__' of a type (line 709)
        getitem___228716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 27), rect_228715, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 709)
        subscript_call_result_228717 = invoke(stypy.reporting.localization.Localization(__file__, 709, 27), getitem___228716, int_228714)
        
        
        # Obtaining the type of the subscript
        int_228718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 41), 'int')
        # Getting the type of 'rect' (line 709)
        rect_228719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 36), 'rect', False)
        # Obtaining the member '__getitem__' of a type (line 709)
        getitem___228720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 36), rect_228719, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 709)
        subscript_call_result_228721 = invoke(stypy.reporting.localization.Localization(__file__, 709, 36), getitem___228720, int_228718)
        
        
        # Obtaining the type of the subscript
        int_228722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 50), 'int')
        # Getting the type of 'rect' (line 709)
        rect_228723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 45), 'rect', False)
        # Obtaining the member '__getitem__' of a type (line 709)
        getitem___228724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 45), rect_228723, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 709)
        subscript_call_result_228725 = invoke(stypy.reporting.localization.Localization(__file__, 709, 45), getitem___228724, int_228722)
        
        
        # Obtaining the type of the subscript
        int_228726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 59), 'int')
        # Getting the type of 'rect' (line 709)
        rect_228727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 54), 'rect', False)
        # Obtaining the member '__getitem__' of a type (line 709)
        getitem___228728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 54), rect_228727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 709)
        subscript_call_result_228729 = invoke(stypy.reporting.localization.Localization(__file__, 709, 54), getitem___228728, int_228726)
        
        # Processing the call keyword arguments (line 709)
        kwargs_228730 = {}
        # Getting the type of 'self' (line 709)
        self_228711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 709)
        ctx_228712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 8), self_228711, 'ctx')
        # Obtaining the member 'rectangle' of a type (line 709)
        rectangle_228713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 8), ctx_228712, 'rectangle')
        # Calling rectangle(args, kwargs) (line 709)
        rectangle_call_result_228731 = invoke(stypy.reporting.localization.Localization(__file__, 709, 8), rectangle_228713, *[subscript_call_result_228717, subscript_call_result_228721, subscript_call_result_228725, subscript_call_result_228729], **kwargs_228730)
        
        
        # Call to set_source_rgb(...): (line 710)
        # Processing the call arguments (line 710)
        int_228735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 32), 'int')
        int_228736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 35), 'int')
        int_228737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 38), 'int')
        # Processing the call keyword arguments (line 710)
        kwargs_228738 = {}
        # Getting the type of 'self' (line 710)
        self_228732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 710)
        ctx_228733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 8), self_228732, 'ctx')
        # Obtaining the member 'set_source_rgb' of a type (line 710)
        set_source_rgb_228734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 8), ctx_228733, 'set_source_rgb')
        # Calling set_source_rgb(args, kwargs) (line 710)
        set_source_rgb_call_result_228739 = invoke(stypy.reporting.localization.Localization(__file__, 710, 8), set_source_rgb_228734, *[int_228735, int_228736, int_228737], **kwargs_228738)
        
        
        # Call to stroke(...): (line 711)
        # Processing the call keyword arguments (line 711)
        kwargs_228743 = {}
        # Getting the type of 'self' (line 711)
        self_228740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'self', False)
        # Obtaining the member 'ctx' of a type (line 711)
        ctx_228741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 8), self_228740, 'ctx')
        # Obtaining the member 'stroke' of a type (line 711)
        stroke_228742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 8), ctx_228741, 'stroke')
        # Calling stroke(args, kwargs) (line 711)
        stroke_call_result_228744 = invoke(stypy.reporting.localization.Localization(__file__, 711, 8), stroke_228742, *[], **kwargs_228743)
        
        
        # ################# End of 'draw_rubberband(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_rubberband' in the type store
        # Getting the type of 'stypy_return_type' (line 691)
        stypy_return_type_228745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228745)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_rubberband'
        return stypy_return_type_228745


# Assigning a type to the variable 'RubberbandGTK3' (line 686)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 0), 'RubberbandGTK3', RubberbandGTK3)
# Declaration of the 'ToolbarGTK3' class
# Getting the type of 'ToolContainerBase' (line 714)
ToolContainerBase_228746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 18), 'ToolContainerBase')
# Getting the type of 'Gtk' (line 714)
Gtk_228747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 37), 'Gtk')
# Obtaining the member 'Box' of a type (line 714)
Box_228748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 37), Gtk_228747, 'Box')

class ToolbarGTK3(ToolContainerBase_228746, Box_228748, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 715, 4, False)
        # Assigning a type to the variable 'self' (line 716)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolbarGTK3.__init__', ['toolmanager'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['toolmanager'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 716)
        # Processing the call arguments (line 716)
        # Getting the type of 'self' (line 716)
        self_228751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 35), 'self', False)
        # Getting the type of 'toolmanager' (line 716)
        toolmanager_228752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 41), 'toolmanager', False)
        # Processing the call keyword arguments (line 716)
        kwargs_228753 = {}
        # Getting the type of 'ToolContainerBase' (line 716)
        ToolContainerBase_228749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'ToolContainerBase', False)
        # Obtaining the member '__init__' of a type (line 716)
        init___228750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 8), ToolContainerBase_228749, '__init__')
        # Calling __init__(args, kwargs) (line 716)
        init___call_result_228754 = invoke(stypy.reporting.localization.Localization(__file__, 716, 8), init___228750, *[self_228751, toolmanager_228752], **kwargs_228753)
        
        
        # Call to __init__(...): (line 717)
        # Processing the call arguments (line 717)
        # Getting the type of 'self' (line 717)
        self_228758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 25), 'self', False)
        # Processing the call keyword arguments (line 717)
        kwargs_228759 = {}
        # Getting the type of 'Gtk' (line 717)
        Gtk_228755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'Gtk', False)
        # Obtaining the member 'Box' of a type (line 717)
        Box_228756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 8), Gtk_228755, 'Box')
        # Obtaining the member '__init__' of a type (line 717)
        init___228757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 8), Box_228756, '__init__')
        # Calling __init__(args, kwargs) (line 717)
        init___call_result_228760 = invoke(stypy.reporting.localization.Localization(__file__, 717, 8), init___228757, *[self_228758], **kwargs_228759)
        
        
        # Call to set_property(...): (line 718)
        # Processing the call arguments (line 718)
        unicode_228763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 26), 'unicode', u'orientation')
        # Getting the type of 'Gtk' (line 718)
        Gtk_228764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 41), 'Gtk', False)
        # Obtaining the member 'Orientation' of a type (line 718)
        Orientation_228765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 41), Gtk_228764, 'Orientation')
        # Obtaining the member 'VERTICAL' of a type (line 718)
        VERTICAL_228766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 41), Orientation_228765, 'VERTICAL')
        # Processing the call keyword arguments (line 718)
        kwargs_228767 = {}
        # Getting the type of 'self' (line 718)
        self_228761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'self', False)
        # Obtaining the member 'set_property' of a type (line 718)
        set_property_228762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 8), self_228761, 'set_property')
        # Calling set_property(args, kwargs) (line 718)
        set_property_call_result_228768 = invoke(stypy.reporting.localization.Localization(__file__, 718, 8), set_property_228762, *[unicode_228763, VERTICAL_228766], **kwargs_228767)
        
        
        # Assigning a Call to a Attribute (line 720):
        
        # Assigning a Call to a Attribute (line 720):
        
        # Call to Box(...): (line 720)
        # Processing the call keyword arguments (line 720)
        kwargs_228771 = {}
        # Getting the type of 'Gtk' (line 720)
        Gtk_228769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 25), 'Gtk', False)
        # Obtaining the member 'Box' of a type (line 720)
        Box_228770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 25), Gtk_228769, 'Box')
        # Calling Box(args, kwargs) (line 720)
        Box_call_result_228772 = invoke(stypy.reporting.localization.Localization(__file__, 720, 25), Box_228770, *[], **kwargs_228771)
        
        # Getting the type of 'self' (line 720)
        self_228773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'self')
        # Setting the type of the member '_toolarea' of a type (line 720)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 8), self_228773, '_toolarea', Box_call_result_228772)
        
        # Call to set_property(...): (line 721)
        # Processing the call arguments (line 721)
        unicode_228777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 36), 'unicode', u'orientation')
        # Getting the type of 'Gtk' (line 721)
        Gtk_228778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 51), 'Gtk', False)
        # Obtaining the member 'Orientation' of a type (line 721)
        Orientation_228779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 51), Gtk_228778, 'Orientation')
        # Obtaining the member 'HORIZONTAL' of a type (line 721)
        HORIZONTAL_228780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 51), Orientation_228779, 'HORIZONTAL')
        # Processing the call keyword arguments (line 721)
        kwargs_228781 = {}
        # Getting the type of 'self' (line 721)
        self_228774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'self', False)
        # Obtaining the member '_toolarea' of a type (line 721)
        _toolarea_228775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 8), self_228774, '_toolarea')
        # Obtaining the member 'set_property' of a type (line 721)
        set_property_228776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 8), _toolarea_228775, 'set_property')
        # Calling set_property(args, kwargs) (line 721)
        set_property_call_result_228782 = invoke(stypy.reporting.localization.Localization(__file__, 721, 8), set_property_228776, *[unicode_228777, HORIZONTAL_228780], **kwargs_228781)
        
        
        # Call to pack_start(...): (line 722)
        # Processing the call arguments (line 722)
        # Getting the type of 'self' (line 722)
        self_228785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 24), 'self', False)
        # Obtaining the member '_toolarea' of a type (line 722)
        _toolarea_228786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 24), self_228785, '_toolarea')
        # Getting the type of 'False' (line 722)
        False_228787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 40), 'False', False)
        # Getting the type of 'False' (line 722)
        False_228788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 47), 'False', False)
        int_228789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 54), 'int')
        # Processing the call keyword arguments (line 722)
        kwargs_228790 = {}
        # Getting the type of 'self' (line 722)
        self_228783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'self', False)
        # Obtaining the member 'pack_start' of a type (line 722)
        pack_start_228784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 8), self_228783, 'pack_start')
        # Calling pack_start(args, kwargs) (line 722)
        pack_start_call_result_228791 = invoke(stypy.reporting.localization.Localization(__file__, 722, 8), pack_start_228784, *[_toolarea_228786, False_228787, False_228788, int_228789], **kwargs_228790)
        
        
        # Call to show_all(...): (line 723)
        # Processing the call keyword arguments (line 723)
        kwargs_228795 = {}
        # Getting the type of 'self' (line 723)
        self_228792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'self', False)
        # Obtaining the member '_toolarea' of a type (line 723)
        _toolarea_228793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 8), self_228792, '_toolarea')
        # Obtaining the member 'show_all' of a type (line 723)
        show_all_228794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 8), _toolarea_228793, 'show_all')
        # Calling show_all(args, kwargs) (line 723)
        show_all_call_result_228796 = invoke(stypy.reporting.localization.Localization(__file__, 723, 8), show_all_228794, *[], **kwargs_228795)
        
        
        # Assigning a Dict to a Attribute (line 724):
        
        # Assigning a Dict to a Attribute (line 724):
        
        # Obtaining an instance of the builtin type 'dict' (line 724)
        dict_228797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 724)
        
        # Getting the type of 'self' (line 724)
        self_228798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'self')
        # Setting the type of the member '_groups' of a type (line 724)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 8), self_228798, '_groups', dict_228797)
        
        # Assigning a Dict to a Attribute (line 725):
        
        # Assigning a Dict to a Attribute (line 725):
        
        # Obtaining an instance of the builtin type 'dict' (line 725)
        dict_228799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 26), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 725)
        
        # Getting the type of 'self' (line 725)
        self_228800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'self')
        # Setting the type of the member '_toolitems' of a type (line 725)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 8), self_228800, '_toolitems', dict_228799)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def add_toolitem(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_toolitem'
        module_type_store = module_type_store.open_function_context('add_toolitem', 727, 4, False)
        # Assigning a type to the variable 'self' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolbarGTK3.add_toolitem.__dict__.__setitem__('stypy_localization', localization)
        ToolbarGTK3.add_toolitem.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolbarGTK3.add_toolitem.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolbarGTK3.add_toolitem.__dict__.__setitem__('stypy_function_name', 'ToolbarGTK3.add_toolitem')
        ToolbarGTK3.add_toolitem.__dict__.__setitem__('stypy_param_names_list', ['name', 'group', 'position', 'image_file', 'description', 'toggle'])
        ToolbarGTK3.add_toolitem.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolbarGTK3.add_toolitem.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolbarGTK3.add_toolitem.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolbarGTK3.add_toolitem.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolbarGTK3.add_toolitem.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolbarGTK3.add_toolitem.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolbarGTK3.add_toolitem', ['name', 'group', 'position', 'image_file', 'description', 'toggle'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_toolitem', localization, ['name', 'group', 'position', 'image_file', 'description', 'toggle'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_toolitem(...)' code ##################

        
        # Getting the type of 'toggle' (line 729)
        toggle_228801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 11), 'toggle')
        # Testing the type of an if condition (line 729)
        if_condition_228802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 8), toggle_228801)
        # Assigning a type to the variable 'if_condition_228802' (line 729)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 8), 'if_condition_228802', if_condition_228802)
        # SSA begins for if statement (line 729)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 730):
        
        # Assigning a Call to a Name (line 730):
        
        # Call to ToggleToolButton(...): (line 730)
        # Processing the call keyword arguments (line 730)
        kwargs_228805 = {}
        # Getting the type of 'Gtk' (line 730)
        Gtk_228803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 22), 'Gtk', False)
        # Obtaining the member 'ToggleToolButton' of a type (line 730)
        ToggleToolButton_228804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 22), Gtk_228803, 'ToggleToolButton')
        # Calling ToggleToolButton(args, kwargs) (line 730)
        ToggleToolButton_call_result_228806 = invoke(stypy.reporting.localization.Localization(__file__, 730, 22), ToggleToolButton_228804, *[], **kwargs_228805)
        
        # Assigning a type to the variable 'tbutton' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 12), 'tbutton', ToggleToolButton_call_result_228806)
        # SSA branch for the else part of an if statement (line 729)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 732):
        
        # Assigning a Call to a Name (line 732):
        
        # Call to ToolButton(...): (line 732)
        # Processing the call keyword arguments (line 732)
        kwargs_228809 = {}
        # Getting the type of 'Gtk' (line 732)
        Gtk_228807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 22), 'Gtk', False)
        # Obtaining the member 'ToolButton' of a type (line 732)
        ToolButton_228808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 22), Gtk_228807, 'ToolButton')
        # Calling ToolButton(args, kwargs) (line 732)
        ToolButton_call_result_228810 = invoke(stypy.reporting.localization.Localization(__file__, 732, 22), ToolButton_228808, *[], **kwargs_228809)
        
        # Assigning a type to the variable 'tbutton' (line 732)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 12), 'tbutton', ToolButton_call_result_228810)
        # SSA join for if statement (line 729)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_label(...): (line 733)
        # Processing the call arguments (line 733)
        # Getting the type of 'name' (line 733)
        name_228813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 26), 'name', False)
        # Processing the call keyword arguments (line 733)
        kwargs_228814 = {}
        # Getting the type of 'tbutton' (line 733)
        tbutton_228811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'tbutton', False)
        # Obtaining the member 'set_label' of a type (line 733)
        set_label_228812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 733, 8), tbutton_228811, 'set_label')
        # Calling set_label(args, kwargs) (line 733)
        set_label_call_result_228815 = invoke(stypy.reporting.localization.Localization(__file__, 733, 8), set_label_228812, *[name_228813], **kwargs_228814)
        
        
        # Type idiom detected: calculating its left and rigth part (line 735)
        # Getting the type of 'image_file' (line 735)
        image_file_228816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 8), 'image_file')
        # Getting the type of 'None' (line 735)
        None_228817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 29), 'None')
        
        (may_be_228818, more_types_in_union_228819) = may_not_be_none(image_file_228816, None_228817)

        if may_be_228818:

            if more_types_in_union_228819:
                # Runtime conditional SSA (line 735)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 736):
            
            # Assigning a Call to a Name (line 736):
            
            # Call to Image(...): (line 736)
            # Processing the call keyword arguments (line 736)
            kwargs_228822 = {}
            # Getting the type of 'Gtk' (line 736)
            Gtk_228820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 20), 'Gtk', False)
            # Obtaining the member 'Image' of a type (line 736)
            Image_228821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 20), Gtk_228820, 'Image')
            # Calling Image(args, kwargs) (line 736)
            Image_call_result_228823 = invoke(stypy.reporting.localization.Localization(__file__, 736, 20), Image_228821, *[], **kwargs_228822)
            
            # Assigning a type to the variable 'image' (line 736)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 12), 'image', Image_call_result_228823)
            
            # Call to set_from_file(...): (line 737)
            # Processing the call arguments (line 737)
            # Getting the type of 'image_file' (line 737)
            image_file_228826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 32), 'image_file', False)
            # Processing the call keyword arguments (line 737)
            kwargs_228827 = {}
            # Getting the type of 'image' (line 737)
            image_228824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 12), 'image', False)
            # Obtaining the member 'set_from_file' of a type (line 737)
            set_from_file_228825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 12), image_228824, 'set_from_file')
            # Calling set_from_file(args, kwargs) (line 737)
            set_from_file_call_result_228828 = invoke(stypy.reporting.localization.Localization(__file__, 737, 12), set_from_file_228825, *[image_file_228826], **kwargs_228827)
            
            
            # Call to set_icon_widget(...): (line 738)
            # Processing the call arguments (line 738)
            # Getting the type of 'image' (line 738)
            image_228831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 36), 'image', False)
            # Processing the call keyword arguments (line 738)
            kwargs_228832 = {}
            # Getting the type of 'tbutton' (line 738)
            tbutton_228829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'tbutton', False)
            # Obtaining the member 'set_icon_widget' of a type (line 738)
            set_icon_widget_228830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 12), tbutton_228829, 'set_icon_widget')
            # Calling set_icon_widget(args, kwargs) (line 738)
            set_icon_widget_call_result_228833 = invoke(stypy.reporting.localization.Localization(__file__, 738, 12), set_icon_widget_228830, *[image_228831], **kwargs_228832)
            

            if more_types_in_union_228819:
                # SSA join for if statement (line 735)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 740)
        # Getting the type of 'position' (line 740)
        position_228834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 11), 'position')
        # Getting the type of 'None' (line 740)
        None_228835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 23), 'None')
        
        (may_be_228836, more_types_in_union_228837) = may_be_none(position_228834, None_228835)

        if may_be_228836:

            if more_types_in_union_228837:
                # Runtime conditional SSA (line 740)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 741):
            
            # Assigning a Num to a Name (line 741):
            int_228838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 23), 'int')
            # Assigning a type to the variable 'position' (line 741)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 12), 'position', int_228838)

            if more_types_in_union_228837:
                # SSA join for if statement (line 740)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _add_button(...): (line 743)
        # Processing the call arguments (line 743)
        # Getting the type of 'tbutton' (line 743)
        tbutton_228841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 25), 'tbutton', False)
        # Getting the type of 'group' (line 743)
        group_228842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 34), 'group', False)
        # Getting the type of 'position' (line 743)
        position_228843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 41), 'position', False)
        # Processing the call keyword arguments (line 743)
        kwargs_228844 = {}
        # Getting the type of 'self' (line 743)
        self_228839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'self', False)
        # Obtaining the member '_add_button' of a type (line 743)
        _add_button_228840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 8), self_228839, '_add_button')
        # Calling _add_button(args, kwargs) (line 743)
        _add_button_call_result_228845 = invoke(stypy.reporting.localization.Localization(__file__, 743, 8), _add_button_228840, *[tbutton_228841, group_228842, position_228843], **kwargs_228844)
        
        
        # Assigning a Call to a Name (line 744):
        
        # Assigning a Call to a Name (line 744):
        
        # Call to connect(...): (line 744)
        # Processing the call arguments (line 744)
        unicode_228848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 33), 'unicode', u'clicked')
        # Getting the type of 'self' (line 744)
        self_228849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 44), 'self', False)
        # Obtaining the member '_call_tool' of a type (line 744)
        _call_tool_228850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 44), self_228849, '_call_tool')
        # Getting the type of 'name' (line 744)
        name_228851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 61), 'name', False)
        # Processing the call keyword arguments (line 744)
        kwargs_228852 = {}
        # Getting the type of 'tbutton' (line 744)
        tbutton_228846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 17), 'tbutton', False)
        # Obtaining the member 'connect' of a type (line 744)
        connect_228847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 17), tbutton_228846, 'connect')
        # Calling connect(args, kwargs) (line 744)
        connect_call_result_228853 = invoke(stypy.reporting.localization.Localization(__file__, 744, 17), connect_228847, *[unicode_228848, _call_tool_228850, name_228851], **kwargs_228852)
        
        # Assigning a type to the variable 'signal' (line 744)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'signal', connect_call_result_228853)
        
        # Call to set_tooltip_text(...): (line 745)
        # Processing the call arguments (line 745)
        # Getting the type of 'description' (line 745)
        description_228856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 33), 'description', False)
        # Processing the call keyword arguments (line 745)
        kwargs_228857 = {}
        # Getting the type of 'tbutton' (line 745)
        tbutton_228854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'tbutton', False)
        # Obtaining the member 'set_tooltip_text' of a type (line 745)
        set_tooltip_text_228855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 8), tbutton_228854, 'set_tooltip_text')
        # Calling set_tooltip_text(args, kwargs) (line 745)
        set_tooltip_text_call_result_228858 = invoke(stypy.reporting.localization.Localization(__file__, 745, 8), set_tooltip_text_228855, *[description_228856], **kwargs_228857)
        
        
        # Call to show_all(...): (line 746)
        # Processing the call keyword arguments (line 746)
        kwargs_228861 = {}
        # Getting the type of 'tbutton' (line 746)
        tbutton_228859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 8), 'tbutton', False)
        # Obtaining the member 'show_all' of a type (line 746)
        show_all_228860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 8), tbutton_228859, 'show_all')
        # Calling show_all(args, kwargs) (line 746)
        show_all_call_result_228862 = invoke(stypy.reporting.localization.Localization(__file__, 746, 8), show_all_228860, *[], **kwargs_228861)
        
        
        # Call to setdefault(...): (line 747)
        # Processing the call arguments (line 747)
        # Getting the type of 'name' (line 747)
        name_228866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 35), 'name', False)
        
        # Obtaining an instance of the builtin type 'list' (line 747)
        list_228867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 747)
        
        # Processing the call keyword arguments (line 747)
        kwargs_228868 = {}
        # Getting the type of 'self' (line 747)
        self_228863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 8), 'self', False)
        # Obtaining the member '_toolitems' of a type (line 747)
        _toolitems_228864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 8), self_228863, '_toolitems')
        # Obtaining the member 'setdefault' of a type (line 747)
        setdefault_228865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 8), _toolitems_228864, 'setdefault')
        # Calling setdefault(args, kwargs) (line 747)
        setdefault_call_result_228869 = invoke(stypy.reporting.localization.Localization(__file__, 747, 8), setdefault_228865, *[name_228866, list_228867], **kwargs_228868)
        
        
        # Call to append(...): (line 748)
        # Processing the call arguments (line 748)
        
        # Obtaining an instance of the builtin type 'tuple' (line 748)
        tuple_228876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 748)
        # Adding element type (line 748)
        # Getting the type of 'tbutton' (line 748)
        tbutton_228877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 38), 'tbutton', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 38), tuple_228876, tbutton_228877)
        # Adding element type (line 748)
        # Getting the type of 'signal' (line 748)
        signal_228878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 47), 'signal', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 748, 38), tuple_228876, signal_228878)
        
        # Processing the call keyword arguments (line 748)
        kwargs_228879 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 748)
        name_228870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 24), 'name', False)
        # Getting the type of 'self' (line 748)
        self_228871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'self', False)
        # Obtaining the member '_toolitems' of a type (line 748)
        _toolitems_228872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 8), self_228871, '_toolitems')
        # Obtaining the member '__getitem__' of a type (line 748)
        getitem___228873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 8), _toolitems_228872, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 748)
        subscript_call_result_228874 = invoke(stypy.reporting.localization.Localization(__file__, 748, 8), getitem___228873, name_228870)
        
        # Obtaining the member 'append' of a type (line 748)
        append_228875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 8), subscript_call_result_228874, 'append')
        # Calling append(args, kwargs) (line 748)
        append_call_result_228880 = invoke(stypy.reporting.localization.Localization(__file__, 748, 8), append_228875, *[tuple_228876], **kwargs_228879)
        
        
        # ################# End of 'add_toolitem(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_toolitem' in the type store
        # Getting the type of 'stypy_return_type' (line 727)
        stypy_return_type_228881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228881)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_toolitem'
        return stypy_return_type_228881


    @norecursion
    def _add_button(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_button'
        module_type_store = module_type_store.open_function_context('_add_button', 750, 4, False)
        # Assigning a type to the variable 'self' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolbarGTK3._add_button.__dict__.__setitem__('stypy_localization', localization)
        ToolbarGTK3._add_button.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolbarGTK3._add_button.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolbarGTK3._add_button.__dict__.__setitem__('stypy_function_name', 'ToolbarGTK3._add_button')
        ToolbarGTK3._add_button.__dict__.__setitem__('stypy_param_names_list', ['button', 'group', 'position'])
        ToolbarGTK3._add_button.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolbarGTK3._add_button.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolbarGTK3._add_button.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolbarGTK3._add_button.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolbarGTK3._add_button.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolbarGTK3._add_button.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolbarGTK3._add_button', ['button', 'group', 'position'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_button', localization, ['button', 'group', 'position'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_button(...)' code ##################

        
        
        # Getting the type of 'group' (line 751)
        group_228882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 11), 'group')
        # Getting the type of 'self' (line 751)
        self_228883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 24), 'self')
        # Obtaining the member '_groups' of a type (line 751)
        _groups_228884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 24), self_228883, '_groups')
        # Applying the binary operator 'notin' (line 751)
        result_contains_228885 = python_operator(stypy.reporting.localization.Localization(__file__, 751, 11), 'notin', group_228882, _groups_228884)
        
        # Testing the type of an if condition (line 751)
        if_condition_228886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 751, 8), result_contains_228885)
        # Assigning a type to the variable 'if_condition_228886' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'if_condition_228886', if_condition_228886)
        # SSA begins for if statement (line 751)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 752)
        self_228887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 15), 'self')
        # Obtaining the member '_groups' of a type (line 752)
        _groups_228888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 15), self_228887, '_groups')
        # Testing the type of an if condition (line 752)
        if_condition_228889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 752, 12), _groups_228888)
        # Assigning a type to the variable 'if_condition_228889' (line 752)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 12), 'if_condition_228889', if_condition_228889)
        # SSA begins for if statement (line 752)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _add_separator(...): (line 753)
        # Processing the call keyword arguments (line 753)
        kwargs_228892 = {}
        # Getting the type of 'self' (line 753)
        self_228890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 16), 'self', False)
        # Obtaining the member '_add_separator' of a type (line 753)
        _add_separator_228891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 16), self_228890, '_add_separator')
        # Calling _add_separator(args, kwargs) (line 753)
        _add_separator_call_result_228893 = invoke(stypy.reporting.localization.Localization(__file__, 753, 16), _add_separator_228891, *[], **kwargs_228892)
        
        # SSA join for if statement (line 752)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 754):
        
        # Assigning a Call to a Name (line 754):
        
        # Call to Toolbar(...): (line 754)
        # Processing the call keyword arguments (line 754)
        kwargs_228896 = {}
        # Getting the type of 'Gtk' (line 754)
        Gtk_228894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 22), 'Gtk', False)
        # Obtaining the member 'Toolbar' of a type (line 754)
        Toolbar_228895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 22), Gtk_228894, 'Toolbar')
        # Calling Toolbar(args, kwargs) (line 754)
        Toolbar_call_result_228897 = invoke(stypy.reporting.localization.Localization(__file__, 754, 22), Toolbar_228895, *[], **kwargs_228896)
        
        # Assigning a type to the variable 'toolbar' (line 754)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 12), 'toolbar', Toolbar_call_result_228897)
        
        # Call to set_style(...): (line 755)
        # Processing the call arguments (line 755)
        # Getting the type of 'Gtk' (line 755)
        Gtk_228900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 30), 'Gtk', False)
        # Obtaining the member 'ToolbarStyle' of a type (line 755)
        ToolbarStyle_228901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 30), Gtk_228900, 'ToolbarStyle')
        # Obtaining the member 'ICONS' of a type (line 755)
        ICONS_228902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 30), ToolbarStyle_228901, 'ICONS')
        # Processing the call keyword arguments (line 755)
        kwargs_228903 = {}
        # Getting the type of 'toolbar' (line 755)
        toolbar_228898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 12), 'toolbar', False)
        # Obtaining the member 'set_style' of a type (line 755)
        set_style_228899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 12), toolbar_228898, 'set_style')
        # Calling set_style(args, kwargs) (line 755)
        set_style_call_result_228904 = invoke(stypy.reporting.localization.Localization(__file__, 755, 12), set_style_228899, *[ICONS_228902], **kwargs_228903)
        
        
        # Call to pack_start(...): (line 756)
        # Processing the call arguments (line 756)
        # Getting the type of 'toolbar' (line 756)
        toolbar_228908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 38), 'toolbar', False)
        # Getting the type of 'False' (line 756)
        False_228909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 47), 'False', False)
        # Getting the type of 'False' (line 756)
        False_228910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 54), 'False', False)
        int_228911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 61), 'int')
        # Processing the call keyword arguments (line 756)
        kwargs_228912 = {}
        # Getting the type of 'self' (line 756)
        self_228905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 12), 'self', False)
        # Obtaining the member '_toolarea' of a type (line 756)
        _toolarea_228906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 12), self_228905, '_toolarea')
        # Obtaining the member 'pack_start' of a type (line 756)
        pack_start_228907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 12), _toolarea_228906, 'pack_start')
        # Calling pack_start(args, kwargs) (line 756)
        pack_start_call_result_228913 = invoke(stypy.reporting.localization.Localization(__file__, 756, 12), pack_start_228907, *[toolbar_228908, False_228909, False_228910, int_228911], **kwargs_228912)
        
        
        # Call to show_all(...): (line 757)
        # Processing the call keyword arguments (line 757)
        kwargs_228916 = {}
        # Getting the type of 'toolbar' (line 757)
        toolbar_228914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 12), 'toolbar', False)
        # Obtaining the member 'show_all' of a type (line 757)
        show_all_228915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 12), toolbar_228914, 'show_all')
        # Calling show_all(args, kwargs) (line 757)
        show_all_call_result_228917 = invoke(stypy.reporting.localization.Localization(__file__, 757, 12), show_all_228915, *[], **kwargs_228916)
        
        
        # Assigning a Name to a Subscript (line 758):
        
        # Assigning a Name to a Subscript (line 758):
        # Getting the type of 'toolbar' (line 758)
        toolbar_228918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 34), 'toolbar')
        # Getting the type of 'self' (line 758)
        self_228919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 12), 'self')
        # Obtaining the member '_groups' of a type (line 758)
        _groups_228920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 12), self_228919, '_groups')
        # Getting the type of 'group' (line 758)
        group_228921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 25), 'group')
        # Storing an element on a container (line 758)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 758, 12), _groups_228920, (group_228921, toolbar_228918))
        # SSA join for if statement (line 751)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to insert(...): (line 759)
        # Processing the call arguments (line 759)
        # Getting the type of 'button' (line 759)
        button_228928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 35), 'button', False)
        # Getting the type of 'position' (line 759)
        position_228929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 43), 'position', False)
        # Processing the call keyword arguments (line 759)
        kwargs_228930 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'group' (line 759)
        group_228922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 21), 'group', False)
        # Getting the type of 'self' (line 759)
        self_228923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'self', False)
        # Obtaining the member '_groups' of a type (line 759)
        _groups_228924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 8), self_228923, '_groups')
        # Obtaining the member '__getitem__' of a type (line 759)
        getitem___228925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 8), _groups_228924, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 759)
        subscript_call_result_228926 = invoke(stypy.reporting.localization.Localization(__file__, 759, 8), getitem___228925, group_228922)
        
        # Obtaining the member 'insert' of a type (line 759)
        insert_228927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 8), subscript_call_result_228926, 'insert')
        # Calling insert(args, kwargs) (line 759)
        insert_call_result_228931 = invoke(stypy.reporting.localization.Localization(__file__, 759, 8), insert_228927, *[button_228928, position_228929], **kwargs_228930)
        
        
        # ################# End of '_add_button(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_button' in the type store
        # Getting the type of 'stypy_return_type' (line 750)
        stypy_return_type_228932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228932)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_button'
        return stypy_return_type_228932


    @norecursion
    def _call_tool(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_call_tool'
        module_type_store = module_type_store.open_function_context('_call_tool', 761, 4, False)
        # Assigning a type to the variable 'self' (line 762)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolbarGTK3._call_tool.__dict__.__setitem__('stypy_localization', localization)
        ToolbarGTK3._call_tool.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolbarGTK3._call_tool.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolbarGTK3._call_tool.__dict__.__setitem__('stypy_function_name', 'ToolbarGTK3._call_tool')
        ToolbarGTK3._call_tool.__dict__.__setitem__('stypy_param_names_list', ['btn', 'name'])
        ToolbarGTK3._call_tool.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolbarGTK3._call_tool.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolbarGTK3._call_tool.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolbarGTK3._call_tool.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolbarGTK3._call_tool.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolbarGTK3._call_tool.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolbarGTK3._call_tool', ['btn', 'name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_call_tool', localization, ['btn', 'name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_call_tool(...)' code ##################

        
        # Call to trigger_tool(...): (line 762)
        # Processing the call arguments (line 762)
        # Getting the type of 'name' (line 762)
        name_228935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 26), 'name', False)
        # Processing the call keyword arguments (line 762)
        kwargs_228936 = {}
        # Getting the type of 'self' (line 762)
        self_228933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'self', False)
        # Obtaining the member 'trigger_tool' of a type (line 762)
        trigger_tool_228934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 8), self_228933, 'trigger_tool')
        # Calling trigger_tool(args, kwargs) (line 762)
        trigger_tool_call_result_228937 = invoke(stypy.reporting.localization.Localization(__file__, 762, 8), trigger_tool_228934, *[name_228935], **kwargs_228936)
        
        
        # ################# End of '_call_tool(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_call_tool' in the type store
        # Getting the type of 'stypy_return_type' (line 761)
        stypy_return_type_228938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_call_tool'
        return stypy_return_type_228938


    @norecursion
    def toggle_toolitem(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'toggle_toolitem'
        module_type_store = module_type_store.open_function_context('toggle_toolitem', 764, 4, False)
        # Assigning a type to the variable 'self' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolbarGTK3.toggle_toolitem.__dict__.__setitem__('stypy_localization', localization)
        ToolbarGTK3.toggle_toolitem.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolbarGTK3.toggle_toolitem.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolbarGTK3.toggle_toolitem.__dict__.__setitem__('stypy_function_name', 'ToolbarGTK3.toggle_toolitem')
        ToolbarGTK3.toggle_toolitem.__dict__.__setitem__('stypy_param_names_list', ['name', 'toggled'])
        ToolbarGTK3.toggle_toolitem.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolbarGTK3.toggle_toolitem.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolbarGTK3.toggle_toolitem.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolbarGTK3.toggle_toolitem.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolbarGTK3.toggle_toolitem.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolbarGTK3.toggle_toolitem.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolbarGTK3.toggle_toolitem', ['name', 'toggled'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'toggle_toolitem', localization, ['name', 'toggled'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'toggle_toolitem(...)' code ##################

        
        
        # Getting the type of 'name' (line 765)
        name_228939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 11), 'name')
        # Getting the type of 'self' (line 765)
        self_228940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 23), 'self')
        # Obtaining the member '_toolitems' of a type (line 765)
        _toolitems_228941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 23), self_228940, '_toolitems')
        # Applying the binary operator 'notin' (line 765)
        result_contains_228942 = python_operator(stypy.reporting.localization.Localization(__file__, 765, 11), 'notin', name_228939, _toolitems_228941)
        
        # Testing the type of an if condition (line 765)
        if_condition_228943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 765, 8), result_contains_228942)
        # Assigning a type to the variable 'if_condition_228943' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 8), 'if_condition_228943', if_condition_228943)
        # SSA begins for if statement (line 765)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 765)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 767)
        name_228944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 48), 'name')
        # Getting the type of 'self' (line 767)
        self_228945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 32), 'self')
        # Obtaining the member '_toolitems' of a type (line 767)
        _toolitems_228946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 32), self_228945, '_toolitems')
        # Obtaining the member '__getitem__' of a type (line 767)
        getitem___228947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 32), _toolitems_228946, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 767)
        subscript_call_result_228948 = invoke(stypy.reporting.localization.Localization(__file__, 767, 32), getitem___228947, name_228944)
        
        # Testing the type of a for loop iterable (line 767)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 767, 8), subscript_call_result_228948)
        # Getting the type of the for loop variable (line 767)
        for_loop_var_228949 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 767, 8), subscript_call_result_228948)
        # Assigning a type to the variable 'toolitem' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'toolitem', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 8), for_loop_var_228949))
        # Assigning a type to the variable 'signal' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'signal', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 8), for_loop_var_228949))
        # SSA begins for a for statement (line 767)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to handler_block(...): (line 768)
        # Processing the call arguments (line 768)
        # Getting the type of 'signal' (line 768)
        signal_228952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 35), 'signal', False)
        # Processing the call keyword arguments (line 768)
        kwargs_228953 = {}
        # Getting the type of 'toolitem' (line 768)
        toolitem_228950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 12), 'toolitem', False)
        # Obtaining the member 'handler_block' of a type (line 768)
        handler_block_228951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 12), toolitem_228950, 'handler_block')
        # Calling handler_block(args, kwargs) (line 768)
        handler_block_call_result_228954 = invoke(stypy.reporting.localization.Localization(__file__, 768, 12), handler_block_228951, *[signal_228952], **kwargs_228953)
        
        
        # Call to set_active(...): (line 769)
        # Processing the call arguments (line 769)
        # Getting the type of 'toggled' (line 769)
        toggled_228957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 32), 'toggled', False)
        # Processing the call keyword arguments (line 769)
        kwargs_228958 = {}
        # Getting the type of 'toolitem' (line 769)
        toolitem_228955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'toolitem', False)
        # Obtaining the member 'set_active' of a type (line 769)
        set_active_228956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 12), toolitem_228955, 'set_active')
        # Calling set_active(args, kwargs) (line 769)
        set_active_call_result_228959 = invoke(stypy.reporting.localization.Localization(__file__, 769, 12), set_active_228956, *[toggled_228957], **kwargs_228958)
        
        
        # Call to handler_unblock(...): (line 770)
        # Processing the call arguments (line 770)
        # Getting the type of 'signal' (line 770)
        signal_228962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 37), 'signal', False)
        # Processing the call keyword arguments (line 770)
        kwargs_228963 = {}
        # Getting the type of 'toolitem' (line 770)
        toolitem_228960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'toolitem', False)
        # Obtaining the member 'handler_unblock' of a type (line 770)
        handler_unblock_228961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 770, 12), toolitem_228960, 'handler_unblock')
        # Calling handler_unblock(args, kwargs) (line 770)
        handler_unblock_call_result_228964 = invoke(stypy.reporting.localization.Localization(__file__, 770, 12), handler_unblock_228961, *[signal_228962], **kwargs_228963)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'toggle_toolitem(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'toggle_toolitem' in the type store
        # Getting the type of 'stypy_return_type' (line 764)
        stypy_return_type_228965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_228965)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'toggle_toolitem'
        return stypy_return_type_228965


    @norecursion
    def remove_toolitem(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove_toolitem'
        module_type_store = module_type_store.open_function_context('remove_toolitem', 772, 4, False)
        # Assigning a type to the variable 'self' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolbarGTK3.remove_toolitem.__dict__.__setitem__('stypy_localization', localization)
        ToolbarGTK3.remove_toolitem.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolbarGTK3.remove_toolitem.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolbarGTK3.remove_toolitem.__dict__.__setitem__('stypy_function_name', 'ToolbarGTK3.remove_toolitem')
        ToolbarGTK3.remove_toolitem.__dict__.__setitem__('stypy_param_names_list', ['name'])
        ToolbarGTK3.remove_toolitem.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolbarGTK3.remove_toolitem.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolbarGTK3.remove_toolitem.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolbarGTK3.remove_toolitem.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolbarGTK3.remove_toolitem.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolbarGTK3.remove_toolitem.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolbarGTK3.remove_toolitem', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove_toolitem', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove_toolitem(...)' code ##################

        
        
        # Getting the type of 'name' (line 773)
        name_228966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 11), 'name')
        # Getting the type of 'self' (line 773)
        self_228967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 23), 'self')
        # Obtaining the member '_toolitems' of a type (line 773)
        _toolitems_228968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 23), self_228967, '_toolitems')
        # Applying the binary operator 'notin' (line 773)
        result_contains_228969 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 11), 'notin', name_228966, _toolitems_228968)
        
        # Testing the type of an if condition (line 773)
        if_condition_228970 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 773, 8), result_contains_228969)
        # Assigning a type to the variable 'if_condition_228970' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'if_condition_228970', if_condition_228970)
        # SSA begins for if statement (line 773)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to message_event(...): (line 774)
        # Processing the call arguments (line 774)
        unicode_228974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 43), 'unicode', u'%s Not in toolbar')
        # Getting the type of 'name' (line 774)
        name_228975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 65), 'name', False)
        # Applying the binary operator '%' (line 774)
        result_mod_228976 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 43), '%', unicode_228974, name_228975)
        
        # Getting the type of 'self' (line 774)
        self_228977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 71), 'self', False)
        # Processing the call keyword arguments (line 774)
        kwargs_228978 = {}
        # Getting the type of 'self' (line 774)
        self_228971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 12), 'self', False)
        # Obtaining the member 'toolmanager' of a type (line 774)
        toolmanager_228972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 12), self_228971, 'toolmanager')
        # Obtaining the member 'message_event' of a type (line 774)
        message_event_228973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 774, 12), toolmanager_228972, 'message_event')
        # Calling message_event(args, kwargs) (line 774)
        message_event_call_result_228979 = invoke(stypy.reporting.localization.Localization(__file__, 774, 12), message_event_228973, *[result_mod_228976, self_228977], **kwargs_228978)
        
        # Assigning a type to the variable 'stypy_return_type' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 773)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 777)
        self_228980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 21), 'self')
        # Obtaining the member '_groups' of a type (line 777)
        _groups_228981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 21), self_228980, '_groups')
        # Testing the type of a for loop iterable (line 777)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 777, 8), _groups_228981)
        # Getting the type of the for loop variable (line 777)
        for_loop_var_228982 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 777, 8), _groups_228981)
        # Assigning a type to the variable 'group' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'group', for_loop_var_228982)
        # SSA begins for a for statement (line 777)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 778)
        name_228983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 53), 'name')
        # Getting the type of 'self' (line 778)
        self_228984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 37), 'self')
        # Obtaining the member '_toolitems' of a type (line 778)
        _toolitems_228985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 37), self_228984, '_toolitems')
        # Obtaining the member '__getitem__' of a type (line 778)
        getitem___228986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 37), _toolitems_228985, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 778)
        subscript_call_result_228987 = invoke(stypy.reporting.localization.Localization(__file__, 778, 37), getitem___228986, name_228983)
        
        # Testing the type of a for loop iterable (line 778)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 778, 12), subscript_call_result_228987)
        # Getting the type of the for loop variable (line 778)
        for_loop_var_228988 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 778, 12), subscript_call_result_228987)
        # Assigning a type to the variable 'toolitem' (line 778)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 12), 'toolitem', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 12), for_loop_var_228988))
        # Assigning a type to the variable '_signal' (line 778)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 12), '_signal', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 12), for_loop_var_228988))
        # SSA begins for a for statement (line 778)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'toolitem' (line 779)
        toolitem_228989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 19), 'toolitem')
        
        # Obtaining the type of the subscript
        # Getting the type of 'group' (line 779)
        group_228990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 44), 'group')
        # Getting the type of 'self' (line 779)
        self_228991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 31), 'self')
        # Obtaining the member '_groups' of a type (line 779)
        _groups_228992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 31), self_228991, '_groups')
        # Obtaining the member '__getitem__' of a type (line 779)
        getitem___228993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 31), _groups_228992, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 779)
        subscript_call_result_228994 = invoke(stypy.reporting.localization.Localization(__file__, 779, 31), getitem___228993, group_228990)
        
        # Applying the binary operator 'in' (line 779)
        result_contains_228995 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 19), 'in', toolitem_228989, subscript_call_result_228994)
        
        # Testing the type of an if condition (line 779)
        if_condition_228996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 779, 16), result_contains_228995)
        # Assigning a type to the variable 'if_condition_228996' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 16), 'if_condition_228996', if_condition_228996)
        # SSA begins for if statement (line 779)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove(...): (line 780)
        # Processing the call arguments (line 780)
        # Getting the type of 'toolitem' (line 780)
        toolitem_229003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 47), 'toolitem', False)
        # Processing the call keyword arguments (line 780)
        kwargs_229004 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'group' (line 780)
        group_228997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 33), 'group', False)
        # Getting the type of 'self' (line 780)
        self_228998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 20), 'self', False)
        # Obtaining the member '_groups' of a type (line 780)
        _groups_228999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 20), self_228998, '_groups')
        # Obtaining the member '__getitem__' of a type (line 780)
        getitem___229000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 20), _groups_228999, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 780)
        subscript_call_result_229001 = invoke(stypy.reporting.localization.Localization(__file__, 780, 20), getitem___229000, group_228997)
        
        # Obtaining the member 'remove' of a type (line 780)
        remove_229002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 20), subscript_call_result_229001, 'remove')
        # Calling remove(args, kwargs) (line 780)
        remove_call_result_229005 = invoke(stypy.reporting.localization.Localization(__file__, 780, 20), remove_229002, *[toolitem_229003], **kwargs_229004)
        
        # SSA join for if statement (line 779)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Deleting a member
        # Getting the type of 'self' (line 781)
        self_229006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'self')
        # Obtaining the member '_toolitems' of a type (line 781)
        _toolitems_229007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 12), self_229006, '_toolitems')
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 781)
        name_229008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 28), 'name')
        # Getting the type of 'self' (line 781)
        self_229009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'self')
        # Obtaining the member '_toolitems' of a type (line 781)
        _toolitems_229010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 12), self_229009, '_toolitems')
        # Obtaining the member '__getitem__' of a type (line 781)
        getitem___229011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 12), _toolitems_229010, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 781)
        subscript_call_result_229012 = invoke(stypy.reporting.localization.Localization(__file__, 781, 12), getitem___229011, name_229008)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 781, 8), _toolitems_229007, subscript_call_result_229012)
        
        # ################# End of 'remove_toolitem(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove_toolitem' in the type store
        # Getting the type of 'stypy_return_type' (line 772)
        stypy_return_type_229013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229013)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove_toolitem'
        return stypy_return_type_229013


    @norecursion
    def _add_separator(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_separator'
        module_type_store = module_type_store.open_function_context('_add_separator', 783, 4, False)
        # Assigning a type to the variable 'self' (line 784)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ToolbarGTK3._add_separator.__dict__.__setitem__('stypy_localization', localization)
        ToolbarGTK3._add_separator.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ToolbarGTK3._add_separator.__dict__.__setitem__('stypy_type_store', module_type_store)
        ToolbarGTK3._add_separator.__dict__.__setitem__('stypy_function_name', 'ToolbarGTK3._add_separator')
        ToolbarGTK3._add_separator.__dict__.__setitem__('stypy_param_names_list', [])
        ToolbarGTK3._add_separator.__dict__.__setitem__('stypy_varargs_param_name', None)
        ToolbarGTK3._add_separator.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ToolbarGTK3._add_separator.__dict__.__setitem__('stypy_call_defaults', defaults)
        ToolbarGTK3._add_separator.__dict__.__setitem__('stypy_call_varargs', varargs)
        ToolbarGTK3._add_separator.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ToolbarGTK3._add_separator.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ToolbarGTK3._add_separator', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_add_separator', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_add_separator(...)' code ##################

        
        # Assigning a Call to a Name (line 784):
        
        # Assigning a Call to a Name (line 784):
        
        # Call to Separator(...): (line 784)
        # Processing the call keyword arguments (line 784)
        kwargs_229016 = {}
        # Getting the type of 'Gtk' (line 784)
        Gtk_229014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 14), 'Gtk', False)
        # Obtaining the member 'Separator' of a type (line 784)
        Separator_229015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 14), Gtk_229014, 'Separator')
        # Calling Separator(args, kwargs) (line 784)
        Separator_call_result_229017 = invoke(stypy.reporting.localization.Localization(__file__, 784, 14), Separator_229015, *[], **kwargs_229016)
        
        # Assigning a type to the variable 'sep' (line 784)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 8), 'sep', Separator_call_result_229017)
        
        # Call to set_property(...): (line 785)
        # Processing the call arguments (line 785)
        unicode_229020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 25), 'unicode', u'orientation')
        # Getting the type of 'Gtk' (line 785)
        Gtk_229021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 40), 'Gtk', False)
        # Obtaining the member 'Orientation' of a type (line 785)
        Orientation_229022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 40), Gtk_229021, 'Orientation')
        # Obtaining the member 'VERTICAL' of a type (line 785)
        VERTICAL_229023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 40), Orientation_229022, 'VERTICAL')
        # Processing the call keyword arguments (line 785)
        kwargs_229024 = {}
        # Getting the type of 'sep' (line 785)
        sep_229018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'sep', False)
        # Obtaining the member 'set_property' of a type (line 785)
        set_property_229019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 8), sep_229018, 'set_property')
        # Calling set_property(args, kwargs) (line 785)
        set_property_call_result_229025 = invoke(stypy.reporting.localization.Localization(__file__, 785, 8), set_property_229019, *[unicode_229020, VERTICAL_229023], **kwargs_229024)
        
        
        # Call to pack_start(...): (line 786)
        # Processing the call arguments (line 786)
        # Getting the type of 'sep' (line 786)
        sep_229029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 34), 'sep', False)
        # Getting the type of 'False' (line 786)
        False_229030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 39), 'False', False)
        # Getting the type of 'True' (line 786)
        True_229031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 46), 'True', False)
        int_229032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 52), 'int')
        # Processing the call keyword arguments (line 786)
        kwargs_229033 = {}
        # Getting the type of 'self' (line 786)
        self_229026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'self', False)
        # Obtaining the member '_toolarea' of a type (line 786)
        _toolarea_229027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 8), self_229026, '_toolarea')
        # Obtaining the member 'pack_start' of a type (line 786)
        pack_start_229028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 8), _toolarea_229027, 'pack_start')
        # Calling pack_start(args, kwargs) (line 786)
        pack_start_call_result_229034 = invoke(stypy.reporting.localization.Localization(__file__, 786, 8), pack_start_229028, *[sep_229029, False_229030, True_229031, int_229032], **kwargs_229033)
        
        
        # Call to show_all(...): (line 787)
        # Processing the call keyword arguments (line 787)
        kwargs_229037 = {}
        # Getting the type of 'sep' (line 787)
        sep_229035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'sep', False)
        # Obtaining the member 'show_all' of a type (line 787)
        show_all_229036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 8), sep_229035, 'show_all')
        # Calling show_all(args, kwargs) (line 787)
        show_all_call_result_229038 = invoke(stypy.reporting.localization.Localization(__file__, 787, 8), show_all_229036, *[], **kwargs_229037)
        
        
        # ################# End of '_add_separator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_separator' in the type store
        # Getting the type of 'stypy_return_type' (line 783)
        stypy_return_type_229039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229039)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_separator'
        return stypy_return_type_229039


# Assigning a type to the variable 'ToolbarGTK3' (line 714)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 0), 'ToolbarGTK3', ToolbarGTK3)
# Declaration of the 'StatusbarGTK3' class
# Getting the type of 'StatusbarBase' (line 790)
StatusbarBase_229040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 20), 'StatusbarBase')
# Getting the type of 'Gtk' (line 790)
Gtk_229041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 35), 'Gtk')
# Obtaining the member 'Statusbar' of a type (line 790)
Statusbar_229042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 35), Gtk_229041, 'Statusbar')

class StatusbarGTK3(StatusbarBase_229040, Statusbar_229042, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 791, 4, False)
        # Assigning a type to the variable 'self' (line 792)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StatusbarGTK3.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 792)
        # Processing the call arguments (line 792)
        # Getting the type of 'self' (line 792)
        self_229045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 31), 'self', False)
        # Getting the type of 'args' (line 792)
        args_229046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 38), 'args', False)
        # Processing the call keyword arguments (line 792)
        # Getting the type of 'kwargs' (line 792)
        kwargs_229047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 46), 'kwargs', False)
        kwargs_229048 = {'kwargs_229047': kwargs_229047}
        # Getting the type of 'StatusbarBase' (line 792)
        StatusbarBase_229043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'StatusbarBase', False)
        # Obtaining the member '__init__' of a type (line 792)
        init___229044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 8), StatusbarBase_229043, '__init__')
        # Calling __init__(args, kwargs) (line 792)
        init___call_result_229049 = invoke(stypy.reporting.localization.Localization(__file__, 792, 8), init___229044, *[self_229045, args_229046], **kwargs_229048)
        
        
        # Call to __init__(...): (line 793)
        # Processing the call arguments (line 793)
        # Getting the type of 'self' (line 793)
        self_229053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 31), 'self', False)
        # Processing the call keyword arguments (line 793)
        kwargs_229054 = {}
        # Getting the type of 'Gtk' (line 793)
        Gtk_229050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), 'Gtk', False)
        # Obtaining the member 'Statusbar' of a type (line 793)
        Statusbar_229051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 8), Gtk_229050, 'Statusbar')
        # Obtaining the member '__init__' of a type (line 793)
        init___229052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 8), Statusbar_229051, '__init__')
        # Calling __init__(args, kwargs) (line 793)
        init___call_result_229055 = invoke(stypy.reporting.localization.Localization(__file__, 793, 8), init___229052, *[self_229053], **kwargs_229054)
        
        
        # Assigning a Call to a Attribute (line 794):
        
        # Assigning a Call to a Attribute (line 794):
        
        # Call to get_context_id(...): (line 794)
        # Processing the call arguments (line 794)
        unicode_229058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 44), 'unicode', u'message')
        # Processing the call keyword arguments (line 794)
        kwargs_229059 = {}
        # Getting the type of 'self' (line 794)
        self_229056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 24), 'self', False)
        # Obtaining the member 'get_context_id' of a type (line 794)
        get_context_id_229057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 24), self_229056, 'get_context_id')
        # Calling get_context_id(args, kwargs) (line 794)
        get_context_id_call_result_229060 = invoke(stypy.reporting.localization.Localization(__file__, 794, 24), get_context_id_229057, *[unicode_229058], **kwargs_229059)
        
        # Getting the type of 'self' (line 794)
        self_229061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'self')
        # Setting the type of the member '_context' of a type (line 794)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 8), self_229061, '_context', get_context_id_call_result_229060)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_message'
        module_type_store = module_type_store.open_function_context('set_message', 796, 4, False)
        # Assigning a type to the variable 'self' (line 797)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StatusbarGTK3.set_message.__dict__.__setitem__('stypy_localization', localization)
        StatusbarGTK3.set_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StatusbarGTK3.set_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        StatusbarGTK3.set_message.__dict__.__setitem__('stypy_function_name', 'StatusbarGTK3.set_message')
        StatusbarGTK3.set_message.__dict__.__setitem__('stypy_param_names_list', ['s'])
        StatusbarGTK3.set_message.__dict__.__setitem__('stypy_varargs_param_name', None)
        StatusbarGTK3.set_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StatusbarGTK3.set_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        StatusbarGTK3.set_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        StatusbarGTK3.set_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StatusbarGTK3.set_message.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StatusbarGTK3.set_message', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_message', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_message(...)' code ##################

        
        # Call to pop(...): (line 797)
        # Processing the call arguments (line 797)
        # Getting the type of 'self' (line 797)
        self_229064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 17), 'self', False)
        # Obtaining the member '_context' of a type (line 797)
        _context_229065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 17), self_229064, '_context')
        # Processing the call keyword arguments (line 797)
        kwargs_229066 = {}
        # Getting the type of 'self' (line 797)
        self_229062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'self', False)
        # Obtaining the member 'pop' of a type (line 797)
        pop_229063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 8), self_229062, 'pop')
        # Calling pop(args, kwargs) (line 797)
        pop_call_result_229067 = invoke(stypy.reporting.localization.Localization(__file__, 797, 8), pop_229063, *[_context_229065], **kwargs_229066)
        
        
        # Call to push(...): (line 798)
        # Processing the call arguments (line 798)
        # Getting the type of 'self' (line 798)
        self_229070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 18), 'self', False)
        # Obtaining the member '_context' of a type (line 798)
        _context_229071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 18), self_229070, '_context')
        # Getting the type of 's' (line 798)
        s_229072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 33), 's', False)
        # Processing the call keyword arguments (line 798)
        kwargs_229073 = {}
        # Getting the type of 'self' (line 798)
        self_229068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 8), 'self', False)
        # Obtaining the member 'push' of a type (line 798)
        push_229069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 8), self_229068, 'push')
        # Calling push(args, kwargs) (line 798)
        push_call_result_229074 = invoke(stypy.reporting.localization.Localization(__file__, 798, 8), push_229069, *[_context_229071, s_229072], **kwargs_229073)
        
        
        # ################# End of 'set_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_message' in the type store
        # Getting the type of 'stypy_return_type' (line 796)
        stypy_return_type_229075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229075)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_message'
        return stypy_return_type_229075


# Assigning a type to the variable 'StatusbarGTK3' (line 790)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 0), 'StatusbarGTK3', StatusbarGTK3)
# Declaration of the 'SaveFigureGTK3' class
# Getting the type of 'backend_tools' (line 801)
backend_tools_229076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 21), 'backend_tools')
# Obtaining the member 'SaveFigureBase' of a type (line 801)
SaveFigureBase_229077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 21), backend_tools_229076, 'SaveFigureBase')

class SaveFigureGTK3(SaveFigureBase_229077, ):

    @norecursion
    def get_filechooser(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_filechooser'
        module_type_store = module_type_store.open_function_context('get_filechooser', 803, 4, False)
        # Assigning a type to the variable 'self' (line 804)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SaveFigureGTK3.get_filechooser.__dict__.__setitem__('stypy_localization', localization)
        SaveFigureGTK3.get_filechooser.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SaveFigureGTK3.get_filechooser.__dict__.__setitem__('stypy_type_store', module_type_store)
        SaveFigureGTK3.get_filechooser.__dict__.__setitem__('stypy_function_name', 'SaveFigureGTK3.get_filechooser')
        SaveFigureGTK3.get_filechooser.__dict__.__setitem__('stypy_param_names_list', [])
        SaveFigureGTK3.get_filechooser.__dict__.__setitem__('stypy_varargs_param_name', None)
        SaveFigureGTK3.get_filechooser.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SaveFigureGTK3.get_filechooser.__dict__.__setitem__('stypy_call_defaults', defaults)
        SaveFigureGTK3.get_filechooser.__dict__.__setitem__('stypy_call_varargs', varargs)
        SaveFigureGTK3.get_filechooser.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SaveFigureGTK3.get_filechooser.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SaveFigureGTK3.get_filechooser', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_filechooser', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_filechooser(...)' code ##################

        
        # Assigning a Call to a Name (line 804):
        
        # Assigning a Call to a Name (line 804):
        
        # Call to FileChooserDialog(...): (line 804)
        # Processing the call keyword arguments (line 804)
        unicode_229079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 18), 'unicode', u'Save the figure')
        keyword_229080 = unicode_229079
        # Getting the type of 'self' (line 806)
        self_229081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 19), 'self', False)
        # Obtaining the member 'figure' of a type (line 806)
        figure_229082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 19), self_229081, 'figure')
        # Obtaining the member 'canvas' of a type (line 806)
        canvas_229083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 19), figure_229082, 'canvas')
        # Obtaining the member 'manager' of a type (line 806)
        manager_229084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 19), canvas_229083, 'manager')
        # Obtaining the member 'window' of a type (line 806)
        window_229085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 19), manager_229084, 'window')
        keyword_229086 = window_229085
        
        # Call to expanduser(...): (line 807)
        # Processing the call arguments (line 807)
        
        # Obtaining the type of the subscript
        unicode_229090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 45), 'unicode', u'savefig.directory')
        # Getting the type of 'rcParams' (line 807)
        rcParams_229091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 36), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 807)
        getitem___229092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 36), rcParams_229091, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 807)
        subscript_call_result_229093 = invoke(stypy.reporting.localization.Localization(__file__, 807, 36), getitem___229092, unicode_229090)
        
        # Processing the call keyword arguments (line 807)
        kwargs_229094 = {}
        # Getting the type of 'os' (line 807)
        os_229087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 807)
        path_229088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 17), os_229087, 'path')
        # Obtaining the member 'expanduser' of a type (line 807)
        expanduser_229089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 17), path_229088, 'expanduser')
        # Calling expanduser(args, kwargs) (line 807)
        expanduser_call_result_229095 = invoke(stypy.reporting.localization.Localization(__file__, 807, 17), expanduser_229089, *[subscript_call_result_229093], **kwargs_229094)
        
        keyword_229096 = expanduser_call_result_229095
        
        # Call to get_supported_filetypes(...): (line 808)
        # Processing the call keyword arguments (line 808)
        kwargs_229101 = {}
        # Getting the type of 'self' (line 808)
        self_229097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 22), 'self', False)
        # Obtaining the member 'figure' of a type (line 808)
        figure_229098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 22), self_229097, 'figure')
        # Obtaining the member 'canvas' of a type (line 808)
        canvas_229099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 22), figure_229098, 'canvas')
        # Obtaining the member 'get_supported_filetypes' of a type (line 808)
        get_supported_filetypes_229100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 22), canvas_229099, 'get_supported_filetypes')
        # Calling get_supported_filetypes(args, kwargs) (line 808)
        get_supported_filetypes_call_result_229102 = invoke(stypy.reporting.localization.Localization(__file__, 808, 22), get_supported_filetypes_229100, *[], **kwargs_229101)
        
        keyword_229103 = get_supported_filetypes_call_result_229102
        
        # Call to get_default_filetype(...): (line 809)
        # Processing the call keyword arguments (line 809)
        kwargs_229108 = {}
        # Getting the type of 'self' (line 809)
        self_229104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 29), 'self', False)
        # Obtaining the member 'figure' of a type (line 809)
        figure_229105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 29), self_229104, 'figure')
        # Obtaining the member 'canvas' of a type (line 809)
        canvas_229106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 29), figure_229105, 'canvas')
        # Obtaining the member 'get_default_filetype' of a type (line 809)
        get_default_filetype_229107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 29), canvas_229106, 'get_default_filetype')
        # Calling get_default_filetype(args, kwargs) (line 809)
        get_default_filetype_call_result_229109 = invoke(stypy.reporting.localization.Localization(__file__, 809, 29), get_default_filetype_229107, *[], **kwargs_229108)
        
        keyword_229110 = get_default_filetype_call_result_229109
        kwargs_229111 = {'default_filetype': keyword_229110, 'path': keyword_229096, 'filetypes': keyword_229103, 'parent': keyword_229086, 'title': keyword_229080}
        # Getting the type of 'FileChooserDialog' (line 804)
        FileChooserDialog_229078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 13), 'FileChooserDialog', False)
        # Calling FileChooserDialog(args, kwargs) (line 804)
        FileChooserDialog_call_result_229112 = invoke(stypy.reporting.localization.Localization(__file__, 804, 13), FileChooserDialog_229078, *[], **kwargs_229111)
        
        # Assigning a type to the variable 'fc' (line 804)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 8), 'fc', FileChooserDialog_call_result_229112)
        
        # Call to set_current_name(...): (line 810)
        # Processing the call arguments (line 810)
        
        # Call to get_default_filename(...): (line 810)
        # Processing the call keyword arguments (line 810)
        kwargs_229119 = {}
        # Getting the type of 'self' (line 810)
        self_229115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 28), 'self', False)
        # Obtaining the member 'figure' of a type (line 810)
        figure_229116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 28), self_229115, 'figure')
        # Obtaining the member 'canvas' of a type (line 810)
        canvas_229117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 28), figure_229116, 'canvas')
        # Obtaining the member 'get_default_filename' of a type (line 810)
        get_default_filename_229118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 28), canvas_229117, 'get_default_filename')
        # Calling get_default_filename(args, kwargs) (line 810)
        get_default_filename_call_result_229120 = invoke(stypy.reporting.localization.Localization(__file__, 810, 28), get_default_filename_229118, *[], **kwargs_229119)
        
        # Processing the call keyword arguments (line 810)
        kwargs_229121 = {}
        # Getting the type of 'fc' (line 810)
        fc_229113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'fc', False)
        # Obtaining the member 'set_current_name' of a type (line 810)
        set_current_name_229114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 8), fc_229113, 'set_current_name')
        # Calling set_current_name(args, kwargs) (line 810)
        set_current_name_call_result_229122 = invoke(stypy.reporting.localization.Localization(__file__, 810, 8), set_current_name_229114, *[get_default_filename_call_result_229120], **kwargs_229121)
        
        # Getting the type of 'fc' (line 811)
        fc_229123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 15), 'fc')
        # Assigning a type to the variable 'stypy_return_type' (line 811)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 8), 'stypy_return_type', fc_229123)
        
        # ################# End of 'get_filechooser(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_filechooser' in the type store
        # Getting the type of 'stypy_return_type' (line 803)
        stypy_return_type_229124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229124)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_filechooser'
        return stypy_return_type_229124


    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 813, 4, False)
        # Assigning a type to the variable 'self' (line 814)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SaveFigureGTK3.trigger.__dict__.__setitem__('stypy_localization', localization)
        SaveFigureGTK3.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SaveFigureGTK3.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        SaveFigureGTK3.trigger.__dict__.__setitem__('stypy_function_name', 'SaveFigureGTK3.trigger')
        SaveFigureGTK3.trigger.__dict__.__setitem__('stypy_param_names_list', [])
        SaveFigureGTK3.trigger.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        SaveFigureGTK3.trigger.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        SaveFigureGTK3.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        SaveFigureGTK3.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        SaveFigureGTK3.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SaveFigureGTK3.trigger.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SaveFigureGTK3.trigger', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger(...)' code ##################

        
        # Assigning a Call to a Name (line 814):
        
        # Assigning a Call to a Name (line 814):
        
        # Call to get_filechooser(...): (line 814)
        # Processing the call keyword arguments (line 814)
        kwargs_229127 = {}
        # Getting the type of 'self' (line 814)
        self_229125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 18), 'self', False)
        # Obtaining the member 'get_filechooser' of a type (line 814)
        get_filechooser_229126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 18), self_229125, 'get_filechooser')
        # Calling get_filechooser(args, kwargs) (line 814)
        get_filechooser_call_result_229128 = invoke(stypy.reporting.localization.Localization(__file__, 814, 18), get_filechooser_229126, *[], **kwargs_229127)
        
        # Assigning a type to the variable 'chooser' (line 814)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 8), 'chooser', get_filechooser_call_result_229128)
        
        # Assigning a Call to a Tuple (line 815):
        
        # Assigning a Call to a Name:
        
        # Call to get_filename_from_user(...): (line 815)
        # Processing the call keyword arguments (line 815)
        kwargs_229131 = {}
        # Getting the type of 'chooser' (line 815)
        chooser_229129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 25), 'chooser', False)
        # Obtaining the member 'get_filename_from_user' of a type (line 815)
        get_filename_from_user_229130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 25), chooser_229129, 'get_filename_from_user')
        # Calling get_filename_from_user(args, kwargs) (line 815)
        get_filename_from_user_call_result_229132 = invoke(stypy.reporting.localization.Localization(__file__, 815, 25), get_filename_from_user_229130, *[], **kwargs_229131)
        
        # Assigning a type to the variable 'call_assignment_226584' (line 815)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'call_assignment_226584', get_filename_from_user_call_result_229132)
        
        # Assigning a Call to a Name (line 815):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_229135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 8), 'int')
        # Processing the call keyword arguments
        kwargs_229136 = {}
        # Getting the type of 'call_assignment_226584' (line 815)
        call_assignment_226584_229133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'call_assignment_226584', False)
        # Obtaining the member '__getitem__' of a type (line 815)
        getitem___229134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 8), call_assignment_226584_229133, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_229137 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___229134, *[int_229135], **kwargs_229136)
        
        # Assigning a type to the variable 'call_assignment_226585' (line 815)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'call_assignment_226585', getitem___call_result_229137)
        
        # Assigning a Name to a Name (line 815):
        # Getting the type of 'call_assignment_226585' (line 815)
        call_assignment_226585_229138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'call_assignment_226585')
        # Assigning a type to the variable 'fname' (line 815)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'fname', call_assignment_226585_229138)
        
        # Assigning a Call to a Name (line 815):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_229141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 8), 'int')
        # Processing the call keyword arguments
        kwargs_229142 = {}
        # Getting the type of 'call_assignment_226584' (line 815)
        call_assignment_226584_229139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'call_assignment_226584', False)
        # Obtaining the member '__getitem__' of a type (line 815)
        getitem___229140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 8), call_assignment_226584_229139, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_229143 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___229140, *[int_229141], **kwargs_229142)
        
        # Assigning a type to the variable 'call_assignment_226586' (line 815)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'call_assignment_226586', getitem___call_result_229143)
        
        # Assigning a Name to a Name (line 815):
        # Getting the type of 'call_assignment_226586' (line 815)
        call_assignment_226586_229144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 8), 'call_assignment_226586')
        # Assigning a type to the variable 'format_' (line 815)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 15), 'format_', call_assignment_226586_229144)
        
        # Call to destroy(...): (line 816)
        # Processing the call keyword arguments (line 816)
        kwargs_229147 = {}
        # Getting the type of 'chooser' (line 816)
        chooser_229145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 8), 'chooser', False)
        # Obtaining the member 'destroy' of a type (line 816)
        destroy_229146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 8), chooser_229145, 'destroy')
        # Calling destroy(args, kwargs) (line 816)
        destroy_call_result_229148 = invoke(stypy.reporting.localization.Localization(__file__, 816, 8), destroy_229146, *[], **kwargs_229147)
        
        
        # Getting the type of 'fname' (line 817)
        fname_229149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 11), 'fname')
        # Testing the type of an if condition (line 817)
        if_condition_229150 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 817, 8), fname_229149)
        # Assigning a type to the variable 'if_condition_229150' (line 817)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 8), 'if_condition_229150', if_condition_229150)
        # SSA begins for if statement (line 817)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 818):
        
        # Assigning a Call to a Name (line 818):
        
        # Call to expanduser(...): (line 818)
        # Processing the call arguments (line 818)
        
        # Obtaining the type of the subscript
        unicode_229154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 52), 'unicode', u'savefig.directory')
        # Getting the type of 'rcParams' (line 818)
        rcParams_229155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 43), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 818)
        getitem___229156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 43), rcParams_229155, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 818)
        subscript_call_result_229157 = invoke(stypy.reporting.localization.Localization(__file__, 818, 43), getitem___229156, unicode_229154)
        
        # Processing the call keyword arguments (line 818)
        kwargs_229158 = {}
        # Getting the type of 'os' (line 818)
        os_229151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 818)
        path_229152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 24), os_229151, 'path')
        # Obtaining the member 'expanduser' of a type (line 818)
        expanduser_229153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 24), path_229152, 'expanduser')
        # Calling expanduser(args, kwargs) (line 818)
        expanduser_call_result_229159 = invoke(stypy.reporting.localization.Localization(__file__, 818, 24), expanduser_229153, *[subscript_call_result_229157], **kwargs_229158)
        
        # Assigning a type to the variable 'startpath' (line 818)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'startpath', expanduser_call_result_229159)
        
        
        # Getting the type of 'startpath' (line 819)
        startpath_229160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 15), 'startpath')
        unicode_229161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 28), 'unicode', u'')
        # Applying the binary operator '==' (line 819)
        result_eq_229162 = python_operator(stypy.reporting.localization.Localization(__file__, 819, 15), '==', startpath_229160, unicode_229161)
        
        # Testing the type of an if condition (line 819)
        if_condition_229163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 819, 12), result_eq_229162)
        # Assigning a type to the variable 'if_condition_229163' (line 819)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 12), 'if_condition_229163', if_condition_229163)
        # SSA begins for if statement (line 819)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 821):
        
        # Assigning a Name to a Subscript (line 821):
        # Getting the type of 'startpath' (line 821)
        startpath_229164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 48), 'startpath')
        # Getting the type of 'rcParams' (line 821)
        rcParams_229165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 16), 'rcParams')
        unicode_229166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 25), 'unicode', u'savefig.directory')
        # Storing an element on a container (line 821)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 16), rcParams_229165, (unicode_229166, startpath_229164))
        # SSA branch for the else part of an if statement (line 819)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Subscript (line 824):
        
        # Assigning a Call to a Subscript (line 824):
        
        # Call to dirname(...): (line 824)
        # Processing the call arguments (line 824)
        
        # Call to text_type(...): (line 825)
        # Processing the call arguments (line 825)
        # Getting the type of 'fname' (line 825)
        fname_229172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 34), 'fname', False)
        # Processing the call keyword arguments (line 825)
        kwargs_229173 = {}
        # Getting the type of 'six' (line 825)
        six_229170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 20), 'six', False)
        # Obtaining the member 'text_type' of a type (line 825)
        text_type_229171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 20), six_229170, 'text_type')
        # Calling text_type(args, kwargs) (line 825)
        text_type_call_result_229174 = invoke(stypy.reporting.localization.Localization(__file__, 825, 20), text_type_229171, *[fname_229172], **kwargs_229173)
        
        # Processing the call keyword arguments (line 824)
        kwargs_229175 = {}
        # Getting the type of 'os' (line 824)
        os_229167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 48), 'os', False)
        # Obtaining the member 'path' of a type (line 824)
        path_229168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 48), os_229167, 'path')
        # Obtaining the member 'dirname' of a type (line 824)
        dirname_229169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 48), path_229168, 'dirname')
        # Calling dirname(args, kwargs) (line 824)
        dirname_call_result_229176 = invoke(stypy.reporting.localization.Localization(__file__, 824, 48), dirname_229169, *[text_type_call_result_229174], **kwargs_229175)
        
        # Getting the type of 'rcParams' (line 824)
        rcParams_229177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 16), 'rcParams')
        unicode_229178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 25), 'unicode', u'savefig.directory')
        # Storing an element on a container (line 824)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 824, 16), rcParams_229177, (unicode_229178, dirname_call_result_229176))
        # SSA join for if statement (line 819)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 826)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to print_figure(...): (line 827)
        # Processing the call arguments (line 827)
        # Getting the type of 'fname' (line 827)
        fname_229183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 48), 'fname', False)
        # Processing the call keyword arguments (line 827)
        # Getting the type of 'format_' (line 827)
        format__229184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 62), 'format_', False)
        keyword_229185 = format__229184
        kwargs_229186 = {'format': keyword_229185}
        # Getting the type of 'self' (line 827)
        self_229179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 16), 'self', False)
        # Obtaining the member 'figure' of a type (line 827)
        figure_229180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 16), self_229179, 'figure')
        # Obtaining the member 'canvas' of a type (line 827)
        canvas_229181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 16), figure_229180, 'canvas')
        # Obtaining the member 'print_figure' of a type (line 827)
        print_figure_229182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 16), canvas_229181, 'print_figure')
        # Calling print_figure(args, kwargs) (line 827)
        print_figure_call_result_229187 = invoke(stypy.reporting.localization.Localization(__file__, 827, 16), print_figure_229182, *[fname_229183], **kwargs_229186)
        
        # SSA branch for the except part of a try statement (line 826)
        # SSA branch for the except 'Exception' branch of a try statement (line 826)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 828)
        Exception_229188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 19), 'Exception')
        # Assigning a type to the variable 'e' (line 828)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 12), 'e', Exception_229188)
        
        # Call to error_msg_gtk(...): (line 829)
        # Processing the call arguments (line 829)
        
        # Call to str(...): (line 829)
        # Processing the call arguments (line 829)
        # Getting the type of 'e' (line 829)
        e_229191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 34), 'e', False)
        # Processing the call keyword arguments (line 829)
        kwargs_229192 = {}
        # Getting the type of 'str' (line 829)
        str_229190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 30), 'str', False)
        # Calling str(args, kwargs) (line 829)
        str_call_result_229193 = invoke(stypy.reporting.localization.Localization(__file__, 829, 30), str_229190, *[e_229191], **kwargs_229192)
        
        # Processing the call keyword arguments (line 829)
        # Getting the type of 'self' (line 829)
        self_229194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 45), 'self', False)
        keyword_229195 = self_229194
        kwargs_229196 = {'parent': keyword_229195}
        # Getting the type of 'error_msg_gtk' (line 829)
        error_msg_gtk_229189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 16), 'error_msg_gtk', False)
        # Calling error_msg_gtk(args, kwargs) (line 829)
        error_msg_gtk_call_result_229197 = invoke(stypy.reporting.localization.Localization(__file__, 829, 16), error_msg_gtk_229189, *[str_call_result_229193], **kwargs_229196)
        
        # SSA join for try-except statement (line 826)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 817)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 813)
        stypy_return_type_229198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229198)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_229198


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 801, 0, False)
        # Assigning a type to the variable 'self' (line 802)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SaveFigureGTK3.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SaveFigureGTK3' (line 801)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 0), 'SaveFigureGTK3', SaveFigureGTK3)
# Declaration of the 'SetCursorGTK3' class
# Getting the type of 'backend_tools' (line 832)
backend_tools_229199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 20), 'backend_tools')
# Obtaining the member 'SetCursorBase' of a type (line 832)
SetCursorBase_229200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 20), backend_tools_229199, 'SetCursorBase')

class SetCursorGTK3(SetCursorBase_229200, ):

    @norecursion
    def set_cursor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_cursor'
        module_type_store = module_type_store.open_function_context('set_cursor', 833, 4, False)
        # Assigning a type to the variable 'self' (line 834)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SetCursorGTK3.set_cursor.__dict__.__setitem__('stypy_localization', localization)
        SetCursorGTK3.set_cursor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SetCursorGTK3.set_cursor.__dict__.__setitem__('stypy_type_store', module_type_store)
        SetCursorGTK3.set_cursor.__dict__.__setitem__('stypy_function_name', 'SetCursorGTK3.set_cursor')
        SetCursorGTK3.set_cursor.__dict__.__setitem__('stypy_param_names_list', ['cursor'])
        SetCursorGTK3.set_cursor.__dict__.__setitem__('stypy_varargs_param_name', None)
        SetCursorGTK3.set_cursor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SetCursorGTK3.set_cursor.__dict__.__setitem__('stypy_call_defaults', defaults)
        SetCursorGTK3.set_cursor.__dict__.__setitem__('stypy_call_varargs', varargs)
        SetCursorGTK3.set_cursor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SetCursorGTK3.set_cursor.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SetCursorGTK3.set_cursor', ['cursor'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_cursor(...): (line 834)
        # Processing the call arguments (line 834)
        
        # Obtaining the type of the subscript
        # Getting the type of 'cursor' (line 834)
        cursor_229209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 69), 'cursor', False)
        # Getting the type of 'cursord' (line 834)
        cursord_229210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 61), 'cursord', False)
        # Obtaining the member '__getitem__' of a type (line 834)
        getitem___229211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 61), cursord_229210, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 834)
        subscript_call_result_229212 = invoke(stypy.reporting.localization.Localization(__file__, 834, 61), getitem___229211, cursor_229209)
        
        # Processing the call keyword arguments (line 834)
        kwargs_229213 = {}
        
        # Call to get_property(...): (line 834)
        # Processing the call arguments (line 834)
        unicode_229205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 40), 'unicode', u'window')
        # Processing the call keyword arguments (line 834)
        kwargs_229206 = {}
        # Getting the type of 'self' (line 834)
        self_229201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 834)
        figure_229202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 8), self_229201, 'figure')
        # Obtaining the member 'canvas' of a type (line 834)
        canvas_229203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 8), figure_229202, 'canvas')
        # Obtaining the member 'get_property' of a type (line 834)
        get_property_229204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 8), canvas_229203, 'get_property')
        # Calling get_property(args, kwargs) (line 834)
        get_property_call_result_229207 = invoke(stypy.reporting.localization.Localization(__file__, 834, 8), get_property_229204, *[unicode_229205], **kwargs_229206)
        
        # Obtaining the member 'set_cursor' of a type (line 834)
        set_cursor_229208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 8), get_property_call_result_229207, 'set_cursor')
        # Calling set_cursor(args, kwargs) (line 834)
        set_cursor_call_result_229214 = invoke(stypy.reporting.localization.Localization(__file__, 834, 8), set_cursor_229208, *[subscript_call_result_229212], **kwargs_229213)
        
        
        # ################# End of 'set_cursor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_cursor' in the type store
        # Getting the type of 'stypy_return_type' (line 833)
        stypy_return_type_229215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229215)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_cursor'
        return stypy_return_type_229215


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 832, 0, False)
        # Assigning a type to the variable 'self' (line 833)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SetCursorGTK3.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SetCursorGTK3' (line 832)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 0), 'SetCursorGTK3', SetCursorGTK3)
# Declaration of the 'ConfigureSubplotsGTK3' class
# Getting the type of 'backend_tools' (line 837)
backend_tools_229216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 28), 'backend_tools')
# Obtaining the member 'ConfigureSubplotsBase' of a type (line 837)
ConfigureSubplotsBase_229217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 28), backend_tools_229216, 'ConfigureSubplotsBase')
# Getting the type of 'Gtk' (line 837)
Gtk_229218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 65), 'Gtk')
# Obtaining the member 'Window' of a type (line 837)
Window_229219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 65), Gtk_229218, 'Window')

class ConfigureSubplotsGTK3(ConfigureSubplotsBase_229217, Window_229219, ):

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigureSubplotsGTK3.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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
        self_229223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 53), 'self', False)
        # Getting the type of 'args' (line 839)
        args_229224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 60), 'args', False)
        # Processing the call keyword arguments (line 839)
        # Getting the type of 'kwargs' (line 839)
        kwargs_229225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 68), 'kwargs', False)
        kwargs_229226 = {'kwargs_229225': kwargs_229225}
        # Getting the type of 'backend_tools' (line 839)
        backend_tools_229220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 8), 'backend_tools', False)
        # Obtaining the member 'ConfigureSubplotsBase' of a type (line 839)
        ConfigureSubplotsBase_229221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 8), backend_tools_229220, 'ConfigureSubplotsBase')
        # Obtaining the member '__init__' of a type (line 839)
        init___229222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 8), ConfigureSubplotsBase_229221, '__init__')
        # Calling __init__(args, kwargs) (line 839)
        init___call_result_229227 = invoke(stypy.reporting.localization.Localization(__file__, 839, 8), init___229222, *[self_229223, args_229224], **kwargs_229226)
        
        
        # Assigning a Name to a Attribute (line 840):
        
        # Assigning a Name to a Attribute (line 840):
        # Getting the type of 'None' (line 840)
        None_229228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 22), 'None')
        # Getting the type of 'self' (line 840)
        self_229229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'self')
        # Setting the type of the member 'window' of a type (line 840)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 8), self_229229, 'window', None_229228)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def init_window(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'init_window'
        module_type_store = module_type_store.open_function_context('init_window', 842, 4, False)
        # Assigning a type to the variable 'self' (line 843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConfigureSubplotsGTK3.init_window.__dict__.__setitem__('stypy_localization', localization)
        ConfigureSubplotsGTK3.init_window.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConfigureSubplotsGTK3.init_window.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConfigureSubplotsGTK3.init_window.__dict__.__setitem__('stypy_function_name', 'ConfigureSubplotsGTK3.init_window')
        ConfigureSubplotsGTK3.init_window.__dict__.__setitem__('stypy_param_names_list', [])
        ConfigureSubplotsGTK3.init_window.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConfigureSubplotsGTK3.init_window.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConfigureSubplotsGTK3.init_window.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConfigureSubplotsGTK3.init_window.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConfigureSubplotsGTK3.init_window.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConfigureSubplotsGTK3.init_window.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigureSubplotsGTK3.init_window', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'init_window', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'init_window(...)' code ##################

        
        # Getting the type of 'self' (line 843)
        self_229230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 11), 'self')
        # Obtaining the member 'window' of a type (line 843)
        window_229231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 11), self_229230, 'window')
        # Testing the type of an if condition (line 843)
        if_condition_229232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 843, 8), window_229231)
        # Assigning a type to the variable 'if_condition_229232' (line 843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 8), 'if_condition_229232', if_condition_229232)
        # SSA begins for if statement (line 843)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 844)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 843)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 845):
        
        # Assigning a Call to a Attribute (line 845):
        
        # Call to Window(...): (line 845)
        # Processing the call keyword arguments (line 845)
        unicode_229235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 39), 'unicode', u'Subplot Configuration Tool')
        keyword_229236 = unicode_229235
        kwargs_229237 = {'title': keyword_229236}
        # Getting the type of 'Gtk' (line 845)
        Gtk_229233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 22), 'Gtk', False)
        # Obtaining the member 'Window' of a type (line 845)
        Window_229234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 22), Gtk_229233, 'Window')
        # Calling Window(args, kwargs) (line 845)
        Window_call_result_229238 = invoke(stypy.reporting.localization.Localization(__file__, 845, 22), Window_229234, *[], **kwargs_229237)
        
        # Getting the type of 'self' (line 845)
        self_229239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 8), 'self')
        # Setting the type of the member 'window' of a type (line 845)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 8), self_229239, 'window', Window_call_result_229238)
        
        
        # SSA begins for try-except statement (line 847)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to set_icon_from_file(...): (line 848)
        # Processing the call arguments (line 848)
        # Getting the type of 'window_icon' (line 848)
        window_icon_229244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 50), 'window_icon', False)
        # Processing the call keyword arguments (line 848)
        kwargs_229245 = {}
        # Getting the type of 'self' (line 848)
        self_229240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 12), 'self', False)
        # Obtaining the member 'window' of a type (line 848)
        window_229241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 12), self_229240, 'window')
        # Obtaining the member 'window' of a type (line 848)
        window_229242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 12), window_229241, 'window')
        # Obtaining the member 'set_icon_from_file' of a type (line 848)
        set_icon_from_file_229243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 12), window_229242, 'set_icon_from_file')
        # Calling set_icon_from_file(args, kwargs) (line 848)
        set_icon_from_file_call_result_229246 = invoke(stypy.reporting.localization.Localization(__file__, 848, 12), set_icon_from_file_229243, *[window_icon_229244], **kwargs_229245)
        
        # SSA branch for the except part of a try statement (line 847)
        # SSA branch for the except 'Tuple' branch of a try statement (line 847)
        module_type_store.open_ssa_branch('except')
        # SSA branch for the except '<any exception>' branch of a try statement (line 847)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 847)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 857):
        
        # Assigning a Call to a Attribute (line 857):
        
        # Call to Box(...): (line 857)
        # Processing the call keyword arguments (line 857)
        kwargs_229249 = {}
        # Getting the type of 'Gtk' (line 857)
        Gtk_229247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 20), 'Gtk', False)
        # Obtaining the member 'Box' of a type (line 857)
        Box_229248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 857, 20), Gtk_229247, 'Box')
        # Calling Box(args, kwargs) (line 857)
        Box_call_result_229250 = invoke(stypy.reporting.localization.Localization(__file__, 857, 20), Box_229248, *[], **kwargs_229249)
        
        # Getting the type of 'self' (line 857)
        self_229251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 8), 'self')
        # Setting the type of the member 'vbox' of a type (line 857)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 857, 8), self_229251, 'vbox', Box_call_result_229250)
        
        # Call to set_property(...): (line 858)
        # Processing the call arguments (line 858)
        unicode_229255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 858, 31), 'unicode', u'orientation')
        # Getting the type of 'Gtk' (line 858)
        Gtk_229256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 46), 'Gtk', False)
        # Obtaining the member 'Orientation' of a type (line 858)
        Orientation_229257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 858, 46), Gtk_229256, 'Orientation')
        # Obtaining the member 'VERTICAL' of a type (line 858)
        VERTICAL_229258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 858, 46), Orientation_229257, 'VERTICAL')
        # Processing the call keyword arguments (line 858)
        kwargs_229259 = {}
        # Getting the type of 'self' (line 858)
        self_229252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 8), 'self', False)
        # Obtaining the member 'vbox' of a type (line 858)
        vbox_229253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 858, 8), self_229252, 'vbox')
        # Obtaining the member 'set_property' of a type (line 858)
        set_property_229254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 858, 8), vbox_229253, 'set_property')
        # Calling set_property(args, kwargs) (line 858)
        set_property_call_result_229260 = invoke(stypy.reporting.localization.Localization(__file__, 858, 8), set_property_229254, *[unicode_229255, VERTICAL_229258], **kwargs_229259)
        
        
        # Call to add(...): (line 859)
        # Processing the call arguments (line 859)
        # Getting the type of 'self' (line 859)
        self_229264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 24), 'self', False)
        # Obtaining the member 'vbox' of a type (line 859)
        vbox_229265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 24), self_229264, 'vbox')
        # Processing the call keyword arguments (line 859)
        kwargs_229266 = {}
        # Getting the type of 'self' (line 859)
        self_229261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 859)
        window_229262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 8), self_229261, 'window')
        # Obtaining the member 'add' of a type (line 859)
        add_229263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 8), window_229262, 'add')
        # Calling add(args, kwargs) (line 859)
        add_call_result_229267 = invoke(stypy.reporting.localization.Localization(__file__, 859, 8), add_229263, *[vbox_229265], **kwargs_229266)
        
        
        # Call to show(...): (line 860)
        # Processing the call keyword arguments (line 860)
        kwargs_229271 = {}
        # Getting the type of 'self' (line 860)
        self_229268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 8), 'self', False)
        # Obtaining the member 'vbox' of a type (line 860)
        vbox_229269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 8), self_229268, 'vbox')
        # Obtaining the member 'show' of a type (line 860)
        show_229270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 8), vbox_229269, 'show')
        # Calling show(args, kwargs) (line 860)
        show_call_result_229272 = invoke(stypy.reporting.localization.Localization(__file__, 860, 8), show_229270, *[], **kwargs_229271)
        
        
        # Call to connect(...): (line 861)
        # Processing the call arguments (line 861)
        unicode_229276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 28), 'unicode', u'destroy')
        # Getting the type of 'self' (line 861)
        self_229277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 39), 'self', False)
        # Obtaining the member 'destroy' of a type (line 861)
        destroy_229278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 39), self_229277, 'destroy')
        # Processing the call keyword arguments (line 861)
        kwargs_229279 = {}
        # Getting the type of 'self' (line 861)
        self_229273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 861)
        window_229274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 8), self_229273, 'window')
        # Obtaining the member 'connect' of a type (line 861)
        connect_229275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 8), window_229274, 'connect')
        # Calling connect(args, kwargs) (line 861)
        connect_call_result_229280 = invoke(stypy.reporting.localization.Localization(__file__, 861, 8), connect_229275, *[unicode_229276, destroy_229278], **kwargs_229279)
        
        
        # Assigning a Call to a Name (line 863):
        
        # Assigning a Call to a Name (line 863):
        
        # Call to Figure(...): (line 863)
        # Processing the call keyword arguments (line 863)
        
        # Obtaining an instance of the builtin type 'tuple' (line 863)
        tuple_229282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 863)
        # Adding element type (line 863)
        int_229283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 863, 34), tuple_229282, int_229283)
        # Adding element type (line 863)
        int_229284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 863, 34), tuple_229282, int_229284)
        
        keyword_229285 = tuple_229282
        kwargs_229286 = {'figsize': keyword_229285}
        # Getting the type of 'Figure' (line 863)
        Figure_229281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 18), 'Figure', False)
        # Calling Figure(args, kwargs) (line 863)
        Figure_call_result_229287 = invoke(stypy.reporting.localization.Localization(__file__, 863, 18), Figure_229281, *[], **kwargs_229286)
        
        # Assigning a type to the variable 'toolfig' (line 863)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 8), 'toolfig', Figure_call_result_229287)
        
        # Assigning a Call to a Name (line 864):
        
        # Assigning a Call to a Name (line 864):
        
        # Call to __class__(...): (line 864)
        # Processing the call arguments (line 864)
        # Getting the type of 'toolfig' (line 864)
        toolfig_229292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 46), 'toolfig', False)
        # Processing the call keyword arguments (line 864)
        kwargs_229293 = {}
        # Getting the type of 'self' (line 864)
        self_229288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 17), 'self', False)
        # Obtaining the member 'figure' of a type (line 864)
        figure_229289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 17), self_229288, 'figure')
        # Obtaining the member 'canvas' of a type (line 864)
        canvas_229290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 17), figure_229289, 'canvas')
        # Obtaining the member '__class__' of a type (line 864)
        class___229291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 17), canvas_229290, '__class__')
        # Calling __class__(args, kwargs) (line 864)
        class___call_result_229294 = invoke(stypy.reporting.localization.Localization(__file__, 864, 17), class___229291, *[toolfig_229292], **kwargs_229293)
        
        # Assigning a type to the variable 'canvas' (line 864)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 8), 'canvas', class___call_result_229294)
        
        # Call to subplots_adjust(...): (line 866)
        # Processing the call keyword arguments (line 866)
        float_229297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, 36), 'float')
        keyword_229298 = float_229297
        kwargs_229299 = {'top': keyword_229298}
        # Getting the type of 'toolfig' (line 866)
        toolfig_229295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 8), 'toolfig', False)
        # Obtaining the member 'subplots_adjust' of a type (line 866)
        subplots_adjust_229296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 8), toolfig_229295, 'subplots_adjust')
        # Calling subplots_adjust(args, kwargs) (line 866)
        subplots_adjust_call_result_229300 = invoke(stypy.reporting.localization.Localization(__file__, 866, 8), subplots_adjust_229296, *[], **kwargs_229299)
        
        
        # Call to SubplotTool(...): (line 867)
        # Processing the call arguments (line 867)
        # Getting the type of 'self' (line 867)
        self_229302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 20), 'self', False)
        # Obtaining the member 'figure' of a type (line 867)
        figure_229303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 20), self_229302, 'figure')
        # Getting the type of 'toolfig' (line 867)
        toolfig_229304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 33), 'toolfig', False)
        # Processing the call keyword arguments (line 867)
        kwargs_229305 = {}
        # Getting the type of 'SubplotTool' (line 867)
        SubplotTool_229301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 8), 'SubplotTool', False)
        # Calling SubplotTool(args, kwargs) (line 867)
        SubplotTool_call_result_229306 = invoke(stypy.reporting.localization.Localization(__file__, 867, 8), SubplotTool_229301, *[figure_229303, toolfig_229304], **kwargs_229305)
        
        
        # Assigning a Call to a Name (line 869):
        
        # Assigning a Call to a Name (line 869):
        
        # Call to int(...): (line 869)
        # Processing the call arguments (line 869)
        # Getting the type of 'toolfig' (line 869)
        toolfig_229308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 16), 'toolfig', False)
        # Obtaining the member 'bbox' of a type (line 869)
        bbox_229309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 16), toolfig_229308, 'bbox')
        # Obtaining the member 'width' of a type (line 869)
        width_229310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 16), bbox_229309, 'width')
        # Processing the call keyword arguments (line 869)
        kwargs_229311 = {}
        # Getting the type of 'int' (line 869)
        int_229307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 12), 'int', False)
        # Calling int(args, kwargs) (line 869)
        int_call_result_229312 = invoke(stypy.reporting.localization.Localization(__file__, 869, 12), int_229307, *[width_229310], **kwargs_229311)
        
        # Assigning a type to the variable 'w' (line 869)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 8), 'w', int_call_result_229312)
        
        # Assigning a Call to a Name (line 870):
        
        # Assigning a Call to a Name (line 870):
        
        # Call to int(...): (line 870)
        # Processing the call arguments (line 870)
        # Getting the type of 'toolfig' (line 870)
        toolfig_229314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 16), 'toolfig', False)
        # Obtaining the member 'bbox' of a type (line 870)
        bbox_229315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 16), toolfig_229314, 'bbox')
        # Obtaining the member 'height' of a type (line 870)
        height_229316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 16), bbox_229315, 'height')
        # Processing the call keyword arguments (line 870)
        kwargs_229317 = {}
        # Getting the type of 'int' (line 870)
        int_229313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 12), 'int', False)
        # Calling int(args, kwargs) (line 870)
        int_call_result_229318 = invoke(stypy.reporting.localization.Localization(__file__, 870, 12), int_229313, *[height_229316], **kwargs_229317)
        
        # Assigning a type to the variable 'h' (line 870)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'h', int_call_result_229318)
        
        # Call to set_default_size(...): (line 872)
        # Processing the call arguments (line 872)
        # Getting the type of 'w' (line 872)
        w_229322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 37), 'w', False)
        # Getting the type of 'h' (line 872)
        h_229323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 40), 'h', False)
        # Processing the call keyword arguments (line 872)
        kwargs_229324 = {}
        # Getting the type of 'self' (line 872)
        self_229319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 872)
        window_229320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 8), self_229319, 'window')
        # Obtaining the member 'set_default_size' of a type (line 872)
        set_default_size_229321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 8), window_229320, 'set_default_size')
        # Calling set_default_size(args, kwargs) (line 872)
        set_default_size_call_result_229325 = invoke(stypy.reporting.localization.Localization(__file__, 872, 8), set_default_size_229321, *[w_229322, h_229323], **kwargs_229324)
        
        
        # Call to show(...): (line 874)
        # Processing the call keyword arguments (line 874)
        kwargs_229328 = {}
        # Getting the type of 'canvas' (line 874)
        canvas_229326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 8), 'canvas', False)
        # Obtaining the member 'show' of a type (line 874)
        show_229327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 8), canvas_229326, 'show')
        # Calling show(args, kwargs) (line 874)
        show_call_result_229329 = invoke(stypy.reporting.localization.Localization(__file__, 874, 8), show_229327, *[], **kwargs_229328)
        
        
        # Call to pack_start(...): (line 875)
        # Processing the call arguments (line 875)
        # Getting the type of 'canvas' (line 875)
        canvas_229333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 29), 'canvas', False)
        # Getting the type of 'True' (line 875)
        True_229334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 37), 'True', False)
        # Getting the type of 'True' (line 875)
        True_229335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 43), 'True', False)
        int_229336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 49), 'int')
        # Processing the call keyword arguments (line 875)
        kwargs_229337 = {}
        # Getting the type of 'self' (line 875)
        self_229330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 8), 'self', False)
        # Obtaining the member 'vbox' of a type (line 875)
        vbox_229331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 8), self_229330, 'vbox')
        # Obtaining the member 'pack_start' of a type (line 875)
        pack_start_229332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 8), vbox_229331, 'pack_start')
        # Calling pack_start(args, kwargs) (line 875)
        pack_start_call_result_229338 = invoke(stypy.reporting.localization.Localization(__file__, 875, 8), pack_start_229332, *[canvas_229333, True_229334, True_229335, int_229336], **kwargs_229337)
        
        
        # Call to show(...): (line 876)
        # Processing the call keyword arguments (line 876)
        kwargs_229342 = {}
        # Getting the type of 'self' (line 876)
        self_229339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 876)
        window_229340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 8), self_229339, 'window')
        # Obtaining the member 'show' of a type (line 876)
        show_229341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 8), window_229340, 'show')
        # Calling show(args, kwargs) (line 876)
        show_call_result_229343 = invoke(stypy.reporting.localization.Localization(__file__, 876, 8), show_229341, *[], **kwargs_229342)
        
        
        # ################# End of 'init_window(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'init_window' in the type store
        # Getting the type of 'stypy_return_type' (line 842)
        stypy_return_type_229344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229344)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'init_window'
        return stypy_return_type_229344


    @norecursion
    def destroy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'destroy'
        module_type_store = module_type_store.open_function_context('destroy', 878, 4, False)
        # Assigning a type to the variable 'self' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConfigureSubplotsGTK3.destroy.__dict__.__setitem__('stypy_localization', localization)
        ConfigureSubplotsGTK3.destroy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConfigureSubplotsGTK3.destroy.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConfigureSubplotsGTK3.destroy.__dict__.__setitem__('stypy_function_name', 'ConfigureSubplotsGTK3.destroy')
        ConfigureSubplotsGTK3.destroy.__dict__.__setitem__('stypy_param_names_list', [])
        ConfigureSubplotsGTK3.destroy.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        ConfigureSubplotsGTK3.destroy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConfigureSubplotsGTK3.destroy.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConfigureSubplotsGTK3.destroy.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConfigureSubplotsGTK3.destroy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConfigureSubplotsGTK3.destroy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigureSubplotsGTK3.destroy', [], 'args', None, defaults, varargs, kwargs)

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

        
        # Call to destroy(...): (line 879)
        # Processing the call keyword arguments (line 879)
        kwargs_229348 = {}
        # Getting the type of 'self' (line 879)
        self_229345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 879)
        window_229346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), self_229345, 'window')
        # Obtaining the member 'destroy' of a type (line 879)
        destroy_229347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 8), window_229346, 'destroy')
        # Calling destroy(args, kwargs) (line 879)
        destroy_call_result_229349 = invoke(stypy.reporting.localization.Localization(__file__, 879, 8), destroy_229347, *[], **kwargs_229348)
        
        
        # Assigning a Name to a Attribute (line 880):
        
        # Assigning a Name to a Attribute (line 880):
        # Getting the type of 'None' (line 880)
        None_229350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 22), 'None')
        # Getting the type of 'self' (line 880)
        self_229351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 8), 'self')
        # Setting the type of the member 'window' of a type (line 880)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 880, 8), self_229351, 'window', None_229350)
        
        # ################# End of 'destroy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'destroy' in the type store
        # Getting the type of 'stypy_return_type' (line 878)
        stypy_return_type_229352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229352)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'destroy'
        return stypy_return_type_229352


    @norecursion
    def _get_canvas(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_canvas'
        module_type_store = module_type_store.open_function_context('_get_canvas', 882, 4, False)
        # Assigning a type to the variable 'self' (line 883)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 883, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConfigureSubplotsGTK3._get_canvas.__dict__.__setitem__('stypy_localization', localization)
        ConfigureSubplotsGTK3._get_canvas.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConfigureSubplotsGTK3._get_canvas.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConfigureSubplotsGTK3._get_canvas.__dict__.__setitem__('stypy_function_name', 'ConfigureSubplotsGTK3._get_canvas')
        ConfigureSubplotsGTK3._get_canvas.__dict__.__setitem__('stypy_param_names_list', ['fig'])
        ConfigureSubplotsGTK3._get_canvas.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConfigureSubplotsGTK3._get_canvas.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConfigureSubplotsGTK3._get_canvas.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConfigureSubplotsGTK3._get_canvas.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConfigureSubplotsGTK3._get_canvas.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConfigureSubplotsGTK3._get_canvas.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigureSubplotsGTK3._get_canvas', ['fig'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_canvas', localization, ['fig'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_canvas(...)' code ##################

        
        # Call to __class__(...): (line 883)
        # Processing the call arguments (line 883)
        # Getting the type of 'fig' (line 883)
        fig_229356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 37), 'fig', False)
        # Processing the call keyword arguments (line 883)
        kwargs_229357 = {}
        # Getting the type of 'self' (line 883)
        self_229353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 15), 'self', False)
        # Obtaining the member 'canvas' of a type (line 883)
        canvas_229354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 15), self_229353, 'canvas')
        # Obtaining the member '__class__' of a type (line 883)
        class___229355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 15), canvas_229354, '__class__')
        # Calling __class__(args, kwargs) (line 883)
        class___call_result_229358 = invoke(stypy.reporting.localization.Localization(__file__, 883, 15), class___229355, *[fig_229356], **kwargs_229357)
        
        # Assigning a type to the variable 'stypy_return_type' (line 883)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 883, 8), 'stypy_return_type', class___call_result_229358)
        
        # ################# End of '_get_canvas(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_canvas' in the type store
        # Getting the type of 'stypy_return_type' (line 882)
        stypy_return_type_229359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229359)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_canvas'
        return stypy_return_type_229359


    @norecursion
    def trigger(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 885)
        None_229360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 42), 'None')
        defaults = [None_229360]
        # Create a new context for function 'trigger'
        module_type_store = module_type_store.open_function_context('trigger', 885, 4, False)
        # Assigning a type to the variable 'self' (line 886)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ConfigureSubplotsGTK3.trigger.__dict__.__setitem__('stypy_localization', localization)
        ConfigureSubplotsGTK3.trigger.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ConfigureSubplotsGTK3.trigger.__dict__.__setitem__('stypy_type_store', module_type_store)
        ConfigureSubplotsGTK3.trigger.__dict__.__setitem__('stypy_function_name', 'ConfigureSubplotsGTK3.trigger')
        ConfigureSubplotsGTK3.trigger.__dict__.__setitem__('stypy_param_names_list', ['sender', 'event', 'data'])
        ConfigureSubplotsGTK3.trigger.__dict__.__setitem__('stypy_varargs_param_name', None)
        ConfigureSubplotsGTK3.trigger.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ConfigureSubplotsGTK3.trigger.__dict__.__setitem__('stypy_call_defaults', defaults)
        ConfigureSubplotsGTK3.trigger.__dict__.__setitem__('stypy_call_varargs', varargs)
        ConfigureSubplotsGTK3.trigger.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ConfigureSubplotsGTK3.trigger.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ConfigureSubplotsGTK3.trigger', ['sender', 'event', 'data'], None, None, defaults, varargs, kwargs)

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

        
        # Call to init_window(...): (line 886)
        # Processing the call keyword arguments (line 886)
        kwargs_229363 = {}
        # Getting the type of 'self' (line 886)
        self_229361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 8), 'self', False)
        # Obtaining the member 'init_window' of a type (line 886)
        init_window_229362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 8), self_229361, 'init_window')
        # Calling init_window(args, kwargs) (line 886)
        init_window_call_result_229364 = invoke(stypy.reporting.localization.Localization(__file__, 886, 8), init_window_229362, *[], **kwargs_229363)
        
        
        # Call to present(...): (line 887)
        # Processing the call keyword arguments (line 887)
        kwargs_229368 = {}
        # Getting the type of 'self' (line 887)
        self_229365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 887)
        window_229366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 8), self_229365, 'window')
        # Obtaining the member 'present' of a type (line 887)
        present_229367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 8), window_229366, 'present')
        # Calling present(args, kwargs) (line 887)
        present_call_result_229369 = invoke(stypy.reporting.localization.Localization(__file__, 887, 8), present_229367, *[], **kwargs_229368)
        
        
        # ################# End of 'trigger(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger' in the type store
        # Getting the type of 'stypy_return_type' (line 885)
        stypy_return_type_229370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229370)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger'
        return stypy_return_type_229370


# Assigning a type to the variable 'ConfigureSubplotsGTK3' (line 837)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 0), 'ConfigureSubplotsGTK3', ConfigureSubplotsGTK3)


# Getting the type of 'sys' (line 891)
sys_229371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 3), 'sys')
# Obtaining the member 'platform' of a type (line 891)
platform_229372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 3), sys_229371, 'platform')
unicode_229373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 19), 'unicode', u'win32')
# Applying the binary operator '==' (line 891)
result_eq_229374 = python_operator(stypy.reporting.localization.Localization(__file__, 891, 3), '==', platform_229372, unicode_229373)

# Testing the type of an if condition (line 891)
if_condition_229375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 891, 0), result_eq_229374)
# Assigning a type to the variable 'if_condition_229375' (line 891)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 0), 'if_condition_229375', if_condition_229375)
# SSA begins for if statement (line 891)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 892):

# Assigning a Str to a Name (line 892):
unicode_229376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 20), 'unicode', u'matplotlib.png')
# Assigning a type to the variable 'icon_filename' (line 892)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 4), 'icon_filename', unicode_229376)
# SSA branch for the else part of an if statement (line 891)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 894):

# Assigning a Str to a Name (line 894):
unicode_229377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 20), 'unicode', u'matplotlib.svg')
# Assigning a type to the variable 'icon_filename' (line 894)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 4), 'icon_filename', unicode_229377)
# SSA join for if statement (line 891)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 895):

# Assigning a Call to a Name (line 895):

# Call to join(...): (line 895)
# Processing the call arguments (line 895)

# Obtaining the type of the subscript
unicode_229381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 896, 24), 'unicode', u'datapath')
# Getting the type of 'matplotlib' (line 896)
matplotlib_229382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 4), 'matplotlib', False)
# Obtaining the member 'rcParams' of a type (line 896)
rcParams_229383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 4), matplotlib_229382, 'rcParams')
# Obtaining the member '__getitem__' of a type (line 896)
getitem___229384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 4), rcParams_229383, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 896)
subscript_call_result_229385 = invoke(stypy.reporting.localization.Localization(__file__, 896, 4), getitem___229384, unicode_229381)

unicode_229386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 896, 37), 'unicode', u'images')
# Getting the type of 'icon_filename' (line 896)
icon_filename_229387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 47), 'icon_filename', False)
# Processing the call keyword arguments (line 895)
kwargs_229388 = {}
# Getting the type of 'os' (line 895)
os_229378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 14), 'os', False)
# Obtaining the member 'path' of a type (line 895)
path_229379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 14), os_229378, 'path')
# Obtaining the member 'join' of a type (line 895)
join_229380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 14), path_229379, 'join')
# Calling join(args, kwargs) (line 895)
join_call_result_229389 = invoke(stypy.reporting.localization.Localization(__file__, 895, 14), join_229380, *[subscript_call_result_229385, unicode_229386, icon_filename_229387], **kwargs_229388)

# Assigning a type to the variable 'window_icon' (line 895)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 0), 'window_icon', join_call_result_229389)

@norecursion
def error_msg_gtk(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 899)
    None_229390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 30), 'None')
    defaults = [None_229390]
    # Create a new context for function 'error_msg_gtk'
    module_type_store = module_type_store.open_function_context('error_msg_gtk', 899, 0, False)
    
    # Passed parameters checking function
    error_msg_gtk.stypy_localization = localization
    error_msg_gtk.stypy_type_of_self = None
    error_msg_gtk.stypy_type_store = module_type_store
    error_msg_gtk.stypy_function_name = 'error_msg_gtk'
    error_msg_gtk.stypy_param_names_list = ['msg', 'parent']
    error_msg_gtk.stypy_varargs_param_name = None
    error_msg_gtk.stypy_kwargs_param_name = None
    error_msg_gtk.stypy_call_defaults = defaults
    error_msg_gtk.stypy_call_varargs = varargs
    error_msg_gtk.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'error_msg_gtk', ['msg', 'parent'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'error_msg_gtk', localization, ['msg', 'parent'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'error_msg_gtk(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 900)
    # Getting the type of 'parent' (line 900)
    parent_229391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 4), 'parent')
    # Getting the type of 'None' (line 900)
    None_229392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 21), 'None')
    
    (may_be_229393, more_types_in_union_229394) = may_not_be_none(parent_229391, None_229392)

    if may_be_229393:

        if more_types_in_union_229394:
            # Runtime conditional SSA (line 900)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 901):
        
        # Assigning a Call to a Name (line 901):
        
        # Call to get_toplevel(...): (line 901)
        # Processing the call keyword arguments (line 901)
        kwargs_229397 = {}
        # Getting the type of 'parent' (line 901)
        parent_229395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 17), 'parent', False)
        # Obtaining the member 'get_toplevel' of a type (line 901)
        get_toplevel_229396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 17), parent_229395, 'get_toplevel')
        # Calling get_toplevel(args, kwargs) (line 901)
        get_toplevel_call_result_229398 = invoke(stypy.reporting.localization.Localization(__file__, 901, 17), get_toplevel_229396, *[], **kwargs_229397)
        
        # Assigning a type to the variable 'parent' (line 901)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'parent', get_toplevel_call_result_229398)
        
        
        
        # Call to is_toplevel(...): (line 902)
        # Processing the call keyword arguments (line 902)
        kwargs_229401 = {}
        # Getting the type of 'parent' (line 902)
        parent_229399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 15), 'parent', False)
        # Obtaining the member 'is_toplevel' of a type (line 902)
        is_toplevel_229400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 15), parent_229399, 'is_toplevel')
        # Calling is_toplevel(args, kwargs) (line 902)
        is_toplevel_call_result_229402 = invoke(stypy.reporting.localization.Localization(__file__, 902, 15), is_toplevel_229400, *[], **kwargs_229401)
        
        # Applying the 'not' unary operator (line 902)
        result_not__229403 = python_operator(stypy.reporting.localization.Localization(__file__, 902, 11), 'not', is_toplevel_call_result_229402)
        
        # Testing the type of an if condition (line 902)
        if_condition_229404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 902, 8), result_not__229403)
        # Assigning a type to the variable 'if_condition_229404' (line 902)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 8), 'if_condition_229404', if_condition_229404)
        # SSA begins for if statement (line 902)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 903):
        
        # Assigning a Name to a Name (line 903):
        # Getting the type of 'None' (line 903)
        None_229405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 21), 'None')
        # Assigning a type to the variable 'parent' (line 903)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 12), 'parent', None_229405)
        # SSA join for if statement (line 902)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_229394:
            # SSA join for if statement (line 900)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to isinstance(...): (line 905)
    # Processing the call arguments (line 905)
    # Getting the type of 'msg' (line 905)
    msg_229407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 22), 'msg', False)
    # Getting the type of 'six' (line 905)
    six_229408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 27), 'six', False)
    # Obtaining the member 'string_types' of a type (line 905)
    string_types_229409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 27), six_229408, 'string_types')
    # Processing the call keyword arguments (line 905)
    kwargs_229410 = {}
    # Getting the type of 'isinstance' (line 905)
    isinstance_229406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 905)
    isinstance_call_result_229411 = invoke(stypy.reporting.localization.Localization(__file__, 905, 11), isinstance_229406, *[msg_229407, string_types_229409], **kwargs_229410)
    
    # Applying the 'not' unary operator (line 905)
    result_not__229412 = python_operator(stypy.reporting.localization.Localization(__file__, 905, 7), 'not', isinstance_call_result_229411)
    
    # Testing the type of an if condition (line 905)
    if_condition_229413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 905, 4), result_not__229412)
    # Assigning a type to the variable 'if_condition_229413' (line 905)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 4), 'if_condition_229413', if_condition_229413)
    # SSA begins for if statement (line 905)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 906):
    
    # Assigning a Call to a Name (line 906):
    
    # Call to join(...): (line 906)
    # Processing the call arguments (line 906)
    
    # Call to map(...): (line 906)
    # Processing the call arguments (line 906)
    # Getting the type of 'str' (line 906)
    str_229417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 27), 'str', False)
    # Getting the type of 'msg' (line 906)
    msg_229418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 32), 'msg', False)
    # Processing the call keyword arguments (line 906)
    kwargs_229419 = {}
    # Getting the type of 'map' (line 906)
    map_229416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 23), 'map', False)
    # Calling map(args, kwargs) (line 906)
    map_call_result_229420 = invoke(stypy.reporting.localization.Localization(__file__, 906, 23), map_229416, *[str_229417, msg_229418], **kwargs_229419)
    
    # Processing the call keyword arguments (line 906)
    kwargs_229421 = {}
    unicode_229414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 14), 'unicode', u',')
    # Obtaining the member 'join' of a type (line 906)
    join_229415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 14), unicode_229414, 'join')
    # Calling join(args, kwargs) (line 906)
    join_call_result_229422 = invoke(stypy.reporting.localization.Localization(__file__, 906, 14), join_229415, *[map_call_result_229420], **kwargs_229421)
    
    # Assigning a type to the variable 'msg' (line 906)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 906, 8), 'msg', join_call_result_229422)
    # SSA join for if statement (line 905)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 908):
    
    # Assigning a Call to a Name (line 908):
    
    # Call to MessageDialog(...): (line 908)
    # Processing the call keyword arguments (line 908)
    # Getting the type of 'parent' (line 909)
    parent_229425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 25), 'parent', False)
    keyword_229426 = parent_229425
    # Getting the type of 'Gtk' (line 910)
    Gtk_229427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 25), 'Gtk', False)
    # Obtaining the member 'MessageType' of a type (line 910)
    MessageType_229428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 910, 25), Gtk_229427, 'MessageType')
    # Obtaining the member 'ERROR' of a type (line 910)
    ERROR_229429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 910, 25), MessageType_229428, 'ERROR')
    keyword_229430 = ERROR_229429
    # Getting the type of 'Gtk' (line 911)
    Gtk_229431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 25), 'Gtk', False)
    # Obtaining the member 'ButtonsType' of a type (line 911)
    ButtonsType_229432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 911, 25), Gtk_229431, 'ButtonsType')
    # Obtaining the member 'OK' of a type (line 911)
    OK_229433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 911, 25), ButtonsType_229432, 'OK')
    keyword_229434 = OK_229433
    # Getting the type of 'msg' (line 912)
    msg_229435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 25), 'msg', False)
    keyword_229436 = msg_229435
    kwargs_229437 = {'buttons': keyword_229434, 'type': keyword_229430, 'parent': keyword_229426, 'message_format': keyword_229436}
    # Getting the type of 'Gtk' (line 908)
    Gtk_229423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 13), 'Gtk', False)
    # Obtaining the member 'MessageDialog' of a type (line 908)
    MessageDialog_229424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 13), Gtk_229423, 'MessageDialog')
    # Calling MessageDialog(args, kwargs) (line 908)
    MessageDialog_call_result_229438 = invoke(stypy.reporting.localization.Localization(__file__, 908, 13), MessageDialog_229424, *[], **kwargs_229437)
    
    # Assigning a type to the variable 'dialog' (line 908)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 4), 'dialog', MessageDialog_call_result_229438)
    
    # Call to run(...): (line 913)
    # Processing the call keyword arguments (line 913)
    kwargs_229441 = {}
    # Getting the type of 'dialog' (line 913)
    dialog_229439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 4), 'dialog', False)
    # Obtaining the member 'run' of a type (line 913)
    run_229440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 4), dialog_229439, 'run')
    # Calling run(args, kwargs) (line 913)
    run_call_result_229442 = invoke(stypy.reporting.localization.Localization(__file__, 913, 4), run_229440, *[], **kwargs_229441)
    
    
    # Call to destroy(...): (line 914)
    # Processing the call keyword arguments (line 914)
    kwargs_229445 = {}
    # Getting the type of 'dialog' (line 914)
    dialog_229443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 4), 'dialog', False)
    # Obtaining the member 'destroy' of a type (line 914)
    destroy_229444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 4), dialog_229443, 'destroy')
    # Calling destroy(args, kwargs) (line 914)
    destroy_call_result_229446 = invoke(stypy.reporting.localization.Localization(__file__, 914, 4), destroy_229444, *[], **kwargs_229445)
    
    
    # ################# End of 'error_msg_gtk(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'error_msg_gtk' in the type store
    # Getting the type of 'stypy_return_type' (line 899)
    stypy_return_type_229447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_229447)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'error_msg_gtk'
    return stypy_return_type_229447

# Assigning a type to the variable 'error_msg_gtk' (line 899)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 0), 'error_msg_gtk', error_msg_gtk)

# Assigning a Name to a Attribute (line 917):

# Assigning a Name to a Attribute (line 917):
# Getting the type of 'SaveFigureGTK3' (line 917)
SaveFigureGTK3_229448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 31), 'SaveFigureGTK3')
# Getting the type of 'backend_tools' (line 917)
backend_tools_229449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 0), 'backend_tools')
# Setting the type of the member 'ToolSaveFigure' of a type (line 917)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 0), backend_tools_229449, 'ToolSaveFigure', SaveFigureGTK3_229448)

# Assigning a Name to a Attribute (line 918):

# Assigning a Name to a Attribute (line 918):
# Getting the type of 'ConfigureSubplotsGTK3' (line 918)
ConfigureSubplotsGTK3_229450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 38), 'ConfigureSubplotsGTK3')
# Getting the type of 'backend_tools' (line 918)
backend_tools_229451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 0), 'backend_tools')
# Setting the type of the member 'ToolConfigureSubplots' of a type (line 918)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 0), backend_tools_229451, 'ToolConfigureSubplots', ConfigureSubplotsGTK3_229450)

# Assigning a Name to a Attribute (line 919):

# Assigning a Name to a Attribute (line 919):
# Getting the type of 'SetCursorGTK3' (line 919)
SetCursorGTK3_229452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 30), 'SetCursorGTK3')
# Getting the type of 'backend_tools' (line 919)
backend_tools_229453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 0), 'backend_tools')
# Setting the type of the member 'ToolSetCursor' of a type (line 919)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 0), backend_tools_229453, 'ToolSetCursor', SetCursorGTK3_229452)

# Assigning a Name to a Attribute (line 920):

# Assigning a Name to a Attribute (line 920):
# Getting the type of 'RubberbandGTK3' (line 920)
RubberbandGTK3_229454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 31), 'RubberbandGTK3')
# Getting the type of 'backend_tools' (line 920)
backend_tools_229455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 0), 'backend_tools')
# Setting the type of the member 'ToolRubberband' of a type (line 920)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 0), backend_tools_229455, 'ToolRubberband', RubberbandGTK3_229454)

# Assigning a Name to a Name (line 922):

# Assigning a Name to a Name (line 922):
# Getting the type of 'ToolbarGTK3' (line 922)
ToolbarGTK3_229456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 10), 'ToolbarGTK3')
# Assigning a type to the variable 'Toolbar' (line 922)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 0), 'Toolbar', ToolbarGTK3_229456)
# Declaration of the '_BackendGTK3' class
# Getting the type of '_Backend' (line 926)
_Backend_229457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 19), '_Backend')

class _BackendGTK3(_Backend_229457, ):
    
    # Assigning a Name to a Name (line 927):
    
    # Assigning a Name to a Name (line 928):

    @staticmethod
    @norecursion
    def trigger_manager_draw(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'trigger_manager_draw'
        module_type_store = module_type_store.open_function_context('trigger_manager_draw', 930, 4, False)
        
        # Passed parameters checking function
        _BackendGTK3.trigger_manager_draw.__dict__.__setitem__('stypy_localization', localization)
        _BackendGTK3.trigger_manager_draw.__dict__.__setitem__('stypy_type_of_self', None)
        _BackendGTK3.trigger_manager_draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        _BackendGTK3.trigger_manager_draw.__dict__.__setitem__('stypy_function_name', 'trigger_manager_draw')
        _BackendGTK3.trigger_manager_draw.__dict__.__setitem__('stypy_param_names_list', ['manager'])
        _BackendGTK3.trigger_manager_draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        _BackendGTK3.trigger_manager_draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _BackendGTK3.trigger_manager_draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        _BackendGTK3.trigger_manager_draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        _BackendGTK3.trigger_manager_draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _BackendGTK3.trigger_manager_draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'trigger_manager_draw', ['manager'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trigger_manager_draw', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trigger_manager_draw(...)' code ##################

        
        # Call to draw_idle(...): (line 932)
        # Processing the call keyword arguments (line 932)
        kwargs_229461 = {}
        # Getting the type of 'manager' (line 932)
        manager_229458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 8), 'manager', False)
        # Obtaining the member 'canvas' of a type (line 932)
        canvas_229459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 932, 8), manager_229458, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 932)
        draw_idle_229460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 932, 8), canvas_229459, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 932)
        draw_idle_call_result_229462 = invoke(stypy.reporting.localization.Localization(__file__, 932, 8), draw_idle_229460, *[], **kwargs_229461)
        
        
        # ################# End of 'trigger_manager_draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger_manager_draw' in the type store
        # Getting the type of 'stypy_return_type' (line 930)
        stypy_return_type_229463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229463)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger_manager_draw'
        return stypy_return_type_229463


    @staticmethod
    @norecursion
    def mainloop(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mainloop'
        module_type_store = module_type_store.open_function_context('mainloop', 934, 4, False)
        
        # Passed parameters checking function
        _BackendGTK3.mainloop.__dict__.__setitem__('stypy_localization', localization)
        _BackendGTK3.mainloop.__dict__.__setitem__('stypy_type_of_self', None)
        _BackendGTK3.mainloop.__dict__.__setitem__('stypy_type_store', module_type_store)
        _BackendGTK3.mainloop.__dict__.__setitem__('stypy_function_name', 'mainloop')
        _BackendGTK3.mainloop.__dict__.__setitem__('stypy_param_names_list', [])
        _BackendGTK3.mainloop.__dict__.__setitem__('stypy_varargs_param_name', None)
        _BackendGTK3.mainloop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _BackendGTK3.mainloop.__dict__.__setitem__('stypy_call_defaults', defaults)
        _BackendGTK3.mainloop.__dict__.__setitem__('stypy_call_varargs', varargs)
        _BackendGTK3.mainloop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _BackendGTK3.mainloop.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'mainloop', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mainloop', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mainloop(...)' code ##################

        
        
        
        # Call to main_level(...): (line 936)
        # Processing the call keyword arguments (line 936)
        kwargs_229466 = {}
        # Getting the type of 'Gtk' (line 936)
        Gtk_229464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 11), 'Gtk', False)
        # Obtaining the member 'main_level' of a type (line 936)
        main_level_229465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 936, 11), Gtk_229464, 'main_level')
        # Calling main_level(args, kwargs) (line 936)
        main_level_call_result_229467 = invoke(stypy.reporting.localization.Localization(__file__, 936, 11), main_level_229465, *[], **kwargs_229466)
        
        int_229468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 31), 'int')
        # Applying the binary operator '==' (line 936)
        result_eq_229469 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 11), '==', main_level_call_result_229467, int_229468)
        
        # Testing the type of an if condition (line 936)
        if_condition_229470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 936, 8), result_eq_229469)
        # Assigning a type to the variable 'if_condition_229470' (line 936)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 8), 'if_condition_229470', if_condition_229470)
        # SSA begins for if statement (line 936)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to main(...): (line 937)
        # Processing the call keyword arguments (line 937)
        kwargs_229473 = {}
        # Getting the type of 'Gtk' (line 937)
        Gtk_229471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 12), 'Gtk', False)
        # Obtaining the member 'main' of a type (line 937)
        main_229472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 937, 12), Gtk_229471, 'main')
        # Calling main(args, kwargs) (line 937)
        main_call_result_229474 = invoke(stypy.reporting.localization.Localization(__file__, 937, 12), main_229472, *[], **kwargs_229473)
        
        # SSA join for if statement (line 936)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'mainloop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mainloop' in the type store
        # Getting the type of 'stypy_return_type' (line 934)
        stypy_return_type_229475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_229475)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mainloop'
        return stypy_return_type_229475


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 925, 0, False)
        # Assigning a type to the variable 'self' (line 926)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendGTK3.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendGTK3' (line 925)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 0), '_BackendGTK3', _BackendGTK3)

# Assigning a Name to a Name (line 927):
# Getting the type of 'FigureCanvasGTK3' (line 927)
FigureCanvasGTK3_229476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 19), 'FigureCanvasGTK3')
# Getting the type of '_BackendGTK3'
_BackendGTK3_229477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGTK3')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGTK3_229477, 'FigureCanvas', FigureCanvasGTK3_229476)

# Assigning a Name to a Name (line 928):
# Getting the type of 'FigureManagerGTK3' (line 928)
FigureManagerGTK3_229478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 20), 'FigureManagerGTK3')
# Getting the type of '_BackendGTK3'
_BackendGTK3_229479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendGTK3')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendGTK3_229479, 'FigureManager', FigureManagerGTK3_229478)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
