
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: import six
4: 
5: import functools
6: import os
7: import re
8: import signal
9: import sys
10: from six import unichr
11: 
12: import matplotlib
13: 
14: from matplotlib._pylab_helpers import Gcf
15: from matplotlib.backend_bases import (
16:     _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
17:     TimerBase, cursors)
18: import matplotlib.backends.qt_editor.figureoptions as figureoptions
19: from matplotlib.backends.qt_editor.formsubplottool import UiSubplotTool
20: from matplotlib.figure import Figure
21: 
22: from .qt_compat import (
23:     QtCore, QtGui, QtWidgets, _getSaveFileName, is_pyqt5, __version__, QT_API)
24: 
25: backend_version = __version__
26: 
27: # SPECIAL_KEYS are keys that do *not* return their unicode name
28: # instead they have manually specified names
29: SPECIAL_KEYS = {QtCore.Qt.Key_Control: 'control',
30:                 QtCore.Qt.Key_Shift: 'shift',
31:                 QtCore.Qt.Key_Alt: 'alt',
32:                 QtCore.Qt.Key_Meta: 'super',
33:                 QtCore.Qt.Key_Return: 'enter',
34:                 QtCore.Qt.Key_Left: 'left',
35:                 QtCore.Qt.Key_Up: 'up',
36:                 QtCore.Qt.Key_Right: 'right',
37:                 QtCore.Qt.Key_Down: 'down',
38:                 QtCore.Qt.Key_Escape: 'escape',
39:                 QtCore.Qt.Key_F1: 'f1',
40:                 QtCore.Qt.Key_F2: 'f2',
41:                 QtCore.Qt.Key_F3: 'f3',
42:                 QtCore.Qt.Key_F4: 'f4',
43:                 QtCore.Qt.Key_F5: 'f5',
44:                 QtCore.Qt.Key_F6: 'f6',
45:                 QtCore.Qt.Key_F7: 'f7',
46:                 QtCore.Qt.Key_F8: 'f8',
47:                 QtCore.Qt.Key_F9: 'f9',
48:                 QtCore.Qt.Key_F10: 'f10',
49:                 QtCore.Qt.Key_F11: 'f11',
50:                 QtCore.Qt.Key_F12: 'f12',
51:                 QtCore.Qt.Key_Home: 'home',
52:                 QtCore.Qt.Key_End: 'end',
53:                 QtCore.Qt.Key_PageUp: 'pageup',
54:                 QtCore.Qt.Key_PageDown: 'pagedown',
55:                 QtCore.Qt.Key_Tab: 'tab',
56:                 QtCore.Qt.Key_Backspace: 'backspace',
57:                 QtCore.Qt.Key_Enter: 'enter',
58:                 QtCore.Qt.Key_Insert: 'insert',
59:                 QtCore.Qt.Key_Delete: 'delete',
60:                 QtCore.Qt.Key_Pause: 'pause',
61:                 QtCore.Qt.Key_SysReq: 'sysreq',
62:                 QtCore.Qt.Key_Clear: 'clear', }
63: 
64: # define which modifier keys are collected on keyboard events.
65: # elements are (mpl names, Modifier Flag, Qt Key) tuples
66: SUPER = 0
67: ALT = 1
68: CTRL = 2
69: SHIFT = 3
70: MODIFIER_KEYS = [('super', QtCore.Qt.MetaModifier, QtCore.Qt.Key_Meta),
71:                  ('alt', QtCore.Qt.AltModifier, QtCore.Qt.Key_Alt),
72:                  ('ctrl', QtCore.Qt.ControlModifier, QtCore.Qt.Key_Control),
73:                  ('shift', QtCore.Qt.ShiftModifier, QtCore.Qt.Key_Shift),
74:                  ]
75: 
76: if sys.platform == 'darwin':
77:     # in OSX, the control and super (aka cmd/apple) keys are switched, so
78:     # switch them back.
79:     SPECIAL_KEYS.update({QtCore.Qt.Key_Control: 'super',  # cmd/apple key
80:                          QtCore.Qt.Key_Meta: 'control',
81:                          })
82:     MODIFIER_KEYS[0] = ('super', QtCore.Qt.ControlModifier,
83:                         QtCore.Qt.Key_Control)
84:     MODIFIER_KEYS[2] = ('ctrl', QtCore.Qt.MetaModifier,
85:                         QtCore.Qt.Key_Meta)
86: 
87: 
88: cursord = {
89:     cursors.MOVE: QtCore.Qt.SizeAllCursor,
90:     cursors.HAND: QtCore.Qt.PointingHandCursor,
91:     cursors.POINTER: QtCore.Qt.ArrowCursor,
92:     cursors.SELECT_REGION: QtCore.Qt.CrossCursor,
93:     cursors.WAIT: QtCore.Qt.WaitCursor,
94:     }
95: 
96: 
97: # make place holder
98: qApp = None
99: 
100: 
101: def _create_qApp():
102:     '''
103:     Only one qApp can exist at a time, so check before creating one.
104:     '''
105:     global qApp
106: 
107:     if qApp is None:
108:         app = QtWidgets.QApplication.instance()
109:         if app is None:
110:             # check for DISPLAY env variable on X11 build of Qt
111:             if is_pyqt5():
112:                 try:
113:                     from PyQt5 import QtX11Extras
114:                     is_x11_build = True
115:                 except ImportError:
116:                     is_x11_build = False
117:             else:
118:                 is_x11_build = hasattr(QtGui, "QX11Info")
119:             if is_x11_build:
120:                 display = os.environ.get('DISPLAY')
121:                 if display is None or not re.search(r':\d', display):
122:                     raise RuntimeError('Invalid DISPLAY variable')
123: 
124:             qApp = QtWidgets.QApplication([b"matplotlib"])
125:             qApp.lastWindowClosed.connect(qApp.quit)
126:         else:
127:             qApp = app
128: 
129:     if is_pyqt5():
130:         try:
131:             qApp.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
132:             qApp.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
133:         except AttributeError:
134:             pass
135: 
136: 
137: def _allow_super_init(__init__):
138:     '''
139:     Decorator for ``__init__`` to allow ``super().__init__`` on PyQt4/PySide2.
140:     '''
141: 
142:     if QT_API == "PyQt5":
143: 
144:         return __init__
145: 
146:     else:
147:         # To work around lack of cooperative inheritance in PyQt4, PySide,
148:         # and PySide2, when calling FigureCanvasQT.__init__, we temporarily
149:         # patch QWidget.__init__ by a cooperative version, that first calls
150:         # QWidget.__init__ with no additional arguments, and then finds the
151:         # next class in the MRO with an __init__ that does support cooperative
152:         # inheritance (i.e., not defined by the PyQt4, PySide, PySide2, sip
153:         # or Shiboken packages), and manually call its `__init__`, once again
154:         # passing the additional arguments.
155: 
156:         qwidget_init = QtWidgets.QWidget.__init__
157: 
158:         def cooperative_qwidget_init(self, *args, **kwargs):
159:             qwidget_init(self)
160:             mro = type(self).__mro__
161:             next_coop_init = next(
162:                 cls for cls in mro[mro.index(QtWidgets.QWidget) + 1:]
163:                 if cls.__module__.split(".")[0] not in [
164:                     "PyQt4", "sip", "PySide", "PySide2", "Shiboken"])
165:             next_coop_init.__init__(self, *args, **kwargs)
166: 
167:         @functools.wraps(__init__)
168:         def wrapper(self, **kwargs):
169:             try:
170:                 QtWidgets.QWidget.__init__ = cooperative_qwidget_init
171:                 __init__(self, **kwargs)
172:             finally:
173:                 # Restore __init__
174:                 QtWidgets.QWidget.__init__ = qwidget_init
175: 
176:         return wrapper
177: 
178: 
179: class TimerQT(TimerBase):
180:     '''
181:     Subclass of :class:`backend_bases.TimerBase` that uses Qt timer events.
182: 
183:     Attributes
184:     ----------
185:     interval : int
186:         The time between timer events in milliseconds. Default is 1000 ms.
187:     single_shot : bool
188:         Boolean flag indicating whether this timer should
189:         operate as single shot (run once and then stop). Defaults to False.
190:     callbacks : list
191:         Stores list of (func, args) tuples that will be called upon timer
192:         events. This list can be manipulated directly, or the functions
193:         `add_callback` and `remove_callback` can be used.
194: 
195:     '''
196: 
197:     def __init__(self, *args, **kwargs):
198:         TimerBase.__init__(self, *args, **kwargs)
199: 
200:         # Create a new timer and connect the timeout() signal to the
201:         # _on_timer method.
202:         self._timer = QtCore.QTimer()
203:         self._timer.timeout.connect(self._on_timer)
204:         self._timer_set_interval()
205: 
206:     def _timer_set_single_shot(self):
207:         self._timer.setSingleShot(self._single)
208: 
209:     def _timer_set_interval(self):
210:         self._timer.setInterval(self._interval)
211: 
212:     def _timer_start(self):
213:         self._timer.start()
214: 
215:     def _timer_stop(self):
216:         self._timer.stop()
217: 
218: 
219: class FigureCanvasQT(QtWidgets.QWidget, FigureCanvasBase):
220: 
221:     # map Qt button codes to MouseEvent's ones:
222:     buttond = {QtCore.Qt.LeftButton: 1,
223:                QtCore.Qt.MidButton: 2,
224:                QtCore.Qt.RightButton: 3,
225:                # QtCore.Qt.XButton1: None,
226:                # QtCore.Qt.XButton2: None,
227:                }
228: 
229:     def _update_figure_dpi(self):
230:         dpi = self._dpi_ratio * self.figure._original_dpi
231:         self.figure._set_dpi(dpi, forward=False)
232: 
233:     @_allow_super_init
234:     def __init__(self, figure):
235:         _create_qApp()
236:         figure._original_dpi = figure.dpi
237: 
238:         super(FigureCanvasQT, self).__init__(figure=figure)
239: 
240:         self.figure = figure
241:         self._update_figure_dpi()
242: 
243:         w, h = self.get_width_height()
244:         self.resize(w, h)
245: 
246:         self.setMouseTracking(True)
247:         # Key auto-repeat enabled by default
248:         self._keyautorepeat = True
249: 
250:         # In cases with mixed resolution displays, we need to be careful if the
251:         # dpi_ratio changes - in this case we need to resize the canvas
252:         # accordingly. We could watch for screenChanged events from Qt, but
253:         # the issue is that we can't guarantee this will be emitted *before*
254:         # the first paintEvent for the canvas, so instead we keep track of the
255:         # dpi_ratio value here and in paintEvent we resize the canvas if
256:         # needed.
257:         self._dpi_ratio_prev = None
258: 
259:     @property
260:     def _dpi_ratio(self):
261:         # Not available on Qt4 or some older Qt5.
262:         try:
263:             return self.devicePixelRatio()
264:         except AttributeError:
265:             return 1
266: 
267:     def get_width_height(self):
268:         w, h = FigureCanvasBase.get_width_height(self)
269:         return int(w / self._dpi_ratio), int(h / self._dpi_ratio)
270: 
271:     def enterEvent(self, event):
272:         FigureCanvasBase.enter_notify_event(self, guiEvent=event)
273: 
274:     def leaveEvent(self, event):
275:         QtWidgets.QApplication.restoreOverrideCursor()
276:         FigureCanvasBase.leave_notify_event(self, guiEvent=event)
277: 
278:     def mouseEventCoords(self, pos):
279:         '''Calculate mouse coordinates in physical pixels
280: 
281:         Qt5 use logical pixels, but the figure is scaled to physical
282:         pixels for rendering.   Transform to physical pixels so that
283:         all of the down-stream transforms work as expected.
284: 
285:         Also, the origin is different and needs to be corrected.
286: 
287:         '''
288:         dpi_ratio = self._dpi_ratio
289:         x = pos.x()
290:         # flip y so y=0 is bottom of canvas
291:         y = self.figure.bbox.height / dpi_ratio - pos.y()
292:         return x * dpi_ratio, y * dpi_ratio
293: 
294:     def mousePressEvent(self, event):
295:         x, y = self.mouseEventCoords(event.pos())
296:         button = self.buttond.get(event.button())
297:         if button is not None:
298:             FigureCanvasBase.button_press_event(self, x, y, button,
299:                                                 guiEvent=event)
300: 
301:     def mouseDoubleClickEvent(self, event):
302:         x, y = self.mouseEventCoords(event.pos())
303:         button = self.buttond.get(event.button())
304:         if button is not None:
305:             FigureCanvasBase.button_press_event(self, x, y,
306:                                                 button, dblclick=True,
307:                                                 guiEvent=event)
308: 
309:     def mouseMoveEvent(self, event):
310:         x, y = self.mouseEventCoords(event)
311:         FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=event)
312: 
313:     def mouseReleaseEvent(self, event):
314:         x, y = self.mouseEventCoords(event)
315:         button = self.buttond.get(event.button())
316:         if button is not None:
317:             FigureCanvasBase.button_release_event(self, x, y, button,
318:                                                   guiEvent=event)
319: 
320:     def wheelEvent(self, event):
321:         x, y = self.mouseEventCoords(event)
322:         # from QWheelEvent::delta doc
323:         if event.pixelDelta().x() == 0 and event.pixelDelta().y() == 0:
324:             steps = event.angleDelta().y() / 120
325:         else:
326:             steps = event.pixelDelta().y()
327:         if steps:
328:             FigureCanvasBase.scroll_event(self, x, y, steps, guiEvent=event)
329: 
330:     def keyPressEvent(self, event):
331:         key = self._get_key(event)
332:         if key is not None:
333:             FigureCanvasBase.key_press_event(self, key, guiEvent=event)
334: 
335:     def keyReleaseEvent(self, event):
336:         key = self._get_key(event)
337:         if key is not None:
338:             FigureCanvasBase.key_release_event(self, key, guiEvent=event)
339: 
340:     @property
341:     def keyAutoRepeat(self):
342:         '''
343:         If True, enable auto-repeat for key events.
344:         '''
345:         return self._keyautorepeat
346: 
347:     @keyAutoRepeat.setter
348:     def keyAutoRepeat(self, val):
349:         self._keyautorepeat = bool(val)
350: 
351:     def resizeEvent(self, event):
352:         # _dpi_ratio_prev will be set the first time the canvas is painted, and
353:         # the rendered buffer is useless before anyways.
354:         if self._dpi_ratio_prev is None:
355:             return
356:         w = event.size().width() * self._dpi_ratio
357:         h = event.size().height() * self._dpi_ratio
358:         dpival = self.figure.dpi
359:         winch = w / dpival
360:         hinch = h / dpival
361:         self.figure.set_size_inches(winch, hinch, forward=False)
362:         # pass back into Qt to let it finish
363:         QtWidgets.QWidget.resizeEvent(self, event)
364:         # emit our resize events
365:         FigureCanvasBase.resize_event(self)
366: 
367:     def sizeHint(self):
368:         w, h = self.get_width_height()
369:         return QtCore.QSize(w, h)
370: 
371:     def minumumSizeHint(self):
372:         return QtCore.QSize(10, 10)
373: 
374:     def _get_key(self, event):
375:         if not self._keyautorepeat and event.isAutoRepeat():
376:             return None
377: 
378:         event_key = event.key()
379:         event_mods = int(event.modifiers())  # actually a bitmask
380: 
381:         # get names of the pressed modifier keys
382:         # bit twiddling to pick out modifier keys from event_mods bitmask,
383:         # if event_key is a MODIFIER, it should not be duplicated in mods
384:         mods = [name for name, mod_key, qt_key in MODIFIER_KEYS
385:                 if event_key != qt_key and (event_mods & mod_key) == mod_key]
386:         try:
387:             # for certain keys (enter, left, backspace, etc) use a word for the
388:             # key, rather than unicode
389:             key = SPECIAL_KEYS[event_key]
390:         except KeyError:
391:             # unicode defines code points up to 0x0010ffff
392:             # QT will use Key_Codes larger than that for keyboard keys that are
393:             # are not unicode characters (like multimedia keys)
394:             # skip these
395:             # if you really want them, you should add them to SPECIAL_KEYS
396:             MAX_UNICODE = 0x10ffff
397:             if event_key > MAX_UNICODE:
398:                 return None
399: 
400:             key = unichr(event_key)
401:             # qt delivers capitalized letters.  fix capitalization
402:             # note that capslock is ignored
403:             if 'shift' in mods:
404:                 mods.remove('shift')
405:             else:
406:                 key = key.lower()
407: 
408:         mods.reverse()
409:         return '+'.join(mods + [key])
410: 
411:     def new_timer(self, *args, **kwargs):
412:         '''
413:         Creates a new backend-specific subclass of
414:         :class:`backend_bases.Timer`.  This is useful for getting
415:         periodic events through the backend's native event
416:         loop. Implemented only for backends with GUIs.
417: 
418:         Other Parameters
419:         ----------------
420:         interval : scalar
421:             Timer interval in milliseconds
422: 
423:         callbacks : list
424:             Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``
425:             will be executed by the timer every *interval*.
426: 
427:         '''
428:         return TimerQT(*args, **kwargs)
429: 
430:     def flush_events(self):
431:         global qApp
432:         qApp.processEvents()
433: 
434:     def start_event_loop(self, timeout=0):
435:         if hasattr(self, "_event_loop") and self._event_loop.isRunning():
436:             raise RuntimeError("Event loop already running")
437:         self._event_loop = event_loop = QtCore.QEventLoop()
438:         if timeout:
439:             timer = QtCore.QTimer.singleShot(timeout * 1000, event_loop.quit)
440:         event_loop.exec_()
441: 
442:     def stop_event_loop(self, event=None):
443:         if hasattr(self, "_event_loop"):
444:             self._event_loop.quit()
445: 
446: 
447: class MainWindow(QtWidgets.QMainWindow):
448:     closing = QtCore.Signal()
449: 
450:     def closeEvent(self, event):
451:         self.closing.emit()
452:         QtWidgets.QMainWindow.closeEvent(self, event)
453: 
454: 
455: class FigureManagerQT(FigureManagerBase):
456:     '''
457:     Attributes
458:     ----------
459:     canvas : `FigureCanvas`
460:         The FigureCanvas instance
461:     num : int or str
462:         The Figure number
463:     toolbar : qt.QToolBar
464:         The qt.QToolBar
465:     window : qt.QMainWindow
466:         The qt.QMainWindow
467: 
468:     '''
469: 
470:     def __init__(self, canvas, num):
471:         FigureManagerBase.__init__(self, canvas, num)
472:         self.canvas = canvas
473:         self.window = MainWindow()
474:         self.window.closing.connect(canvas.close_event)
475:         self.window.closing.connect(self._widgetclosed)
476: 
477:         self.window.setWindowTitle("Figure %d" % num)
478:         image = os.path.join(matplotlib.rcParams['datapath'],
479:                              'images', 'matplotlib.svg')
480:         self.window.setWindowIcon(QtGui.QIcon(image))
481: 
482:         # Give the keyboard focus to the figure instead of the
483:         # manager; StrongFocus accepts both tab and click to focus and
484:         # will enable the canvas to process event w/o clicking.
485:         # ClickFocus only takes the focus is the window has been
486:         # clicked
487:         # on. http://qt-project.org/doc/qt-4.8/qt.html#FocusPolicy-enum or
488:         # http://doc.qt.digia.com/qt/qt.html#FocusPolicy-enum
489:         self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
490:         self.canvas.setFocus()
491: 
492:         self.window._destroying = False
493: 
494:         # add text label to status bar
495:         self.statusbar_label = QtWidgets.QLabel()
496:         self.window.statusBar().addWidget(self.statusbar_label)
497: 
498:         self.toolbar = self._get_toolbar(self.canvas, self.window)
499:         if self.toolbar is not None:
500:             self.window.addToolBar(self.toolbar)
501:             self.toolbar.message.connect(self.statusbar_label.setText)
502:             tbs_height = self.toolbar.sizeHint().height()
503:         else:
504:             tbs_height = 0
505: 
506:         # resize the main window so it will display the canvas with the
507:         # requested size:
508:         cs = canvas.sizeHint()
509:         sbs = self.window.statusBar().sizeHint()
510:         self._status_and_tool_height = tbs_height + sbs.height()
511:         height = cs.height() + self._status_and_tool_height
512:         self.window.resize(cs.width(), height)
513: 
514:         self.window.setCentralWidget(self.canvas)
515: 
516:         if matplotlib.is_interactive():
517:             self.window.show()
518:             self.canvas.draw_idle()
519: 
520:         def notify_axes_change(fig):
521:             # This will be called whenever the current axes is changed
522:             if self.toolbar is not None:
523:                 self.toolbar.update()
524:         self.canvas.figure.add_axobserver(notify_axes_change)
525:         self.window.raise_()
526: 
527:     def full_screen_toggle(self):
528:         if self.window.isFullScreen():
529:             self.window.showNormal()
530:         else:
531:             self.window.showFullScreen()
532: 
533:     def _widgetclosed(self):
534:         if self.window._destroying:
535:             return
536:         self.window._destroying = True
537:         try:
538:             Gcf.destroy(self.num)
539:         except AttributeError:
540:             pass
541:             # It seems that when the python session is killed,
542:             # Gcf can get destroyed before the Gcf.destroy
543:             # line is run, leading to a useless AttributeError.
544: 
545:     def _get_toolbar(self, canvas, parent):
546:         # must be inited after the window, drawingArea and figure
547:         # attrs are set
548:         if matplotlib.rcParams['toolbar'] == 'toolbar2':
549:             toolbar = NavigationToolbar2QT(canvas, parent, False)
550:         else:
551:             toolbar = None
552:         return toolbar
553: 
554:     def resize(self, width, height):
555:         'set the canvas size in pixels'
556:         self.window.resize(width, height + self._status_and_tool_height)
557: 
558:     def show(self):
559:         self.window.show()
560:         self.window.activateWindow()
561:         self.window.raise_()
562: 
563:     def destroy(self, *args):
564:         # check for qApp first, as PySide deletes it in its atexit handler
565:         if QtWidgets.QApplication.instance() is None:
566:             return
567:         if self.window._destroying:
568:             return
569:         self.window._destroying = True
570:         self.window.destroyed.connect(self._widgetclosed)
571:         if self.toolbar:
572:             self.toolbar.destroy()
573:         self.window.close()
574: 
575:     def get_window_title(self):
576:         return six.text_type(self.window.windowTitle())
577: 
578:     def set_window_title(self, title):
579:         self.window.setWindowTitle(title)
580: 
581: 
582: class NavigationToolbar2QT(NavigationToolbar2, QtWidgets.QToolBar):
583:     message = QtCore.Signal(str)
584: 
585:     def __init__(self, canvas, parent, coordinates=True):
586:         ''' coordinates: should we show the coordinates on the right? '''
587:         self.canvas = canvas
588:         self.parent = parent
589:         self.coordinates = coordinates
590:         self._actions = {}
591:         '''A mapping of toolitem method names to their QActions'''
592: 
593:         QtWidgets.QToolBar.__init__(self, parent)
594:         NavigationToolbar2.__init__(self, canvas)
595: 
596:     def _icon(self, name):
597:         if is_pyqt5():
598:             name = name.replace('.png', '_large.png')
599:         pm = QtGui.QPixmap(os.path.join(self.basedir, name))
600:         if hasattr(pm, 'setDevicePixelRatio'):
601:             pm.setDevicePixelRatio(self.canvas._dpi_ratio)
602:         return QtGui.QIcon(pm)
603: 
604:     def _init_toolbar(self):
605:         self.basedir = os.path.join(matplotlib.rcParams['datapath'], 'images')
606: 
607:         for text, tooltip_text, image_file, callback in self.toolitems:
608:             if text is None:
609:                 self.addSeparator()
610:             else:
611:                 a = self.addAction(self._icon(image_file + '.png'),
612:                                    text, getattr(self, callback))
613:                 self._actions[callback] = a
614:                 if callback in ['zoom', 'pan']:
615:                     a.setCheckable(True)
616:                 if tooltip_text is not None:
617:                     a.setToolTip(tooltip_text)
618:                 if text == 'Subplots':
619:                     a = self.addAction(self._icon("qt4_editor_options.png"),
620:                                        'Customize', self.edit_parameters)
621:                     a.setToolTip('Edit axis, curve and image parameters')
622: 
623:         self.buttons = {}
624: 
625:         # Add the x,y location widget at the right side of the toolbar
626:         # The stretch factor is 1 which means any resizing of the toolbar
627:         # will resize this label instead of the buttons.
628:         if self.coordinates:
629:             self.locLabel = QtWidgets.QLabel("", self)
630:             self.locLabel.setAlignment(
631:                     QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
632:             self.locLabel.setSizePolicy(
633:                 QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
634:                                       QtWidgets.QSizePolicy.Ignored))
635:             labelAction = self.addWidget(self.locLabel)
636:             labelAction.setVisible(True)
637: 
638:         # reference holder for subplots_adjust window
639:         self.adj_window = None
640: 
641:         # Esthetic adjustments - we need to set these explicitly in PyQt5
642:         # otherwise the layout looks different - but we don't want to set it if
643:         # not using HiDPI icons otherwise they look worse than before.
644:         if is_pyqt5():
645:             self.setIconSize(QtCore.QSize(24, 24))
646:             self.layout().setSpacing(12)
647: 
648:     if is_pyqt5():
649:         # For some reason, self.setMinimumHeight doesn't seem to carry over to
650:         # the actual sizeHint, so override it instead in order to make the
651:         # aesthetic adjustments noted above.
652:         def sizeHint(self):
653:             size = super(NavigationToolbar2QT, self).sizeHint()
654:             size.setHeight(max(48, size.height()))
655:             return size
656: 
657:     def edit_parameters(self):
658:         allaxes = self.canvas.figure.get_axes()
659:         if not allaxes:
660:             QtWidgets.QMessageBox.warning(
661:                 self.parent, "Error", "There are no axes to edit.")
662:             return
663:         elif len(allaxes) == 1:
664:             axes, = allaxes
665:         else:
666:             titles = []
667:             for axes in allaxes:
668:                 name = (axes.get_title() or
669:                         " - ".join(filter(None, [axes.get_xlabel(),
670:                                                  axes.get_ylabel()])) or
671:                         "<anonymous {} (id: {:#x})>".format(
672:                             type(axes).__name__, id(axes)))
673:                 titles.append(name)
674:             item, ok = QtWidgets.QInputDialog.getItem(
675:                 self.parent, 'Customize', 'Select axes:', titles, 0, False)
676:             if ok:
677:                 axes = allaxes[titles.index(six.text_type(item))]
678:             else:
679:                 return
680: 
681:         figureoptions.figure_edit(axes, self)
682: 
683:     def _update_buttons_checked(self):
684:         # sync button checkstates to match active mode
685:         self._actions['pan'].setChecked(self._active == 'PAN')
686:         self._actions['zoom'].setChecked(self._active == 'ZOOM')
687: 
688:     def pan(self, *args):
689:         super(NavigationToolbar2QT, self).pan(*args)
690:         self._update_buttons_checked()
691: 
692:     def zoom(self, *args):
693:         super(NavigationToolbar2QT, self).zoom(*args)
694:         self._update_buttons_checked()
695: 
696:     def set_message(self, s):
697:         self.message.emit(s)
698:         if self.coordinates:
699:             self.locLabel.setText(s)
700: 
701:     def set_cursor(self, cursor):
702:         self.canvas.setCursor(cursord[cursor])
703: 
704:     def draw_rubberband(self, event, x0, y0, x1, y1):
705:         height = self.canvas.figure.bbox.height
706:         y1 = height - y1
707:         y0 = height - y0
708:         rect = [int(val) for val in (x0, y0, x1 - x0, y1 - y0)]
709:         self.canvas.drawRectangle(rect)
710: 
711:     def remove_rubberband(self):
712:         self.canvas.drawRectangle(None)
713: 
714:     def configure_subplots(self):
715:         image = os.path.join(matplotlib.rcParams['datapath'],
716:                              'images', 'matplotlib.png')
717:         dia = SubplotToolQt(self.canvas.figure, self.parent)
718:         dia.setWindowIcon(QtGui.QIcon(image))
719:         dia.exec_()
720: 
721:     def save_figure(self, *args):
722:         filetypes = self.canvas.get_supported_filetypes_grouped()
723:         sorted_filetypes = sorted(six.iteritems(filetypes))
724:         default_filetype = self.canvas.get_default_filetype()
725: 
726:         startpath = os.path.expanduser(
727:             matplotlib.rcParams['savefig.directory'])
728:         start = os.path.join(startpath, self.canvas.get_default_filename())
729:         filters = []
730:         selectedFilter = None
731:         for name, exts in sorted_filetypes:
732:             exts_list = " ".join(['*.%s' % ext for ext in exts])
733:             filter = '%s (%s)' % (name, exts_list)
734:             if default_filetype in exts:
735:                 selectedFilter = filter
736:             filters.append(filter)
737:         filters = ';;'.join(filters)
738: 
739:         fname, filter = _getSaveFileName(self.parent,
740:                                          "Choose a filename to save to",
741:                                          start, filters, selectedFilter)
742:         if fname:
743:             # Save dir for next time, unless empty str (i.e., use cwd).
744:             if startpath != "":
745:                 matplotlib.rcParams['savefig.directory'] = (
746:                     os.path.dirname(six.text_type(fname)))
747:             try:
748:                 self.canvas.figure.savefig(six.text_type(fname))
749:             except Exception as e:
750:                 QtWidgets.QMessageBox.critical(
751:                     self, "Error saving file", six.text_type(e),
752:                     QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.NoButton)
753: 
754: 
755: class SubplotToolQt(UiSubplotTool):
756:     def __init__(self, targetfig, parent):
757:         UiSubplotTool.__init__(self, None)
758: 
759:         self._figure = targetfig
760: 
761:         for lower, higher in [("bottom", "top"), ("left", "right")]:
762:             self._widgets[lower].valueChanged.connect(
763:                 lambda val: self._widgets[higher].setMinimum(val + .001))
764:             self._widgets[higher].valueChanged.connect(
765:                 lambda val: self._widgets[lower].setMaximum(val - .001))
766: 
767:         self._attrs = ["top", "bottom", "left", "right", "hspace", "wspace"]
768:         self._defaults = {attr: vars(self._figure.subplotpars)[attr]
769:                           for attr in self._attrs}
770: 
771:         # Set values after setting the range callbacks, but before setting up
772:         # the redraw callbacks.
773:         self._reset()
774: 
775:         for attr in self._attrs:
776:             self._widgets[attr].valueChanged.connect(self._on_value_changed)
777:         for action, method in [("Export values", self._export_values),
778:                                ("Tight layout", self._tight_layout),
779:                                ("Reset", self._reset),
780:                                ("Close", self.close)]:
781:             self._widgets[action].clicked.connect(method)
782: 
783:     def _export_values(self):
784:         # Explicitly round to 3 decimals (which is also the spinbox precision)
785:         # to avoid numbers of the form 0.100...001.
786:         dialog = QtWidgets.QDialog()
787:         layout = QtWidgets.QVBoxLayout()
788:         dialog.setLayout(layout)
789:         text = QtWidgets.QPlainTextEdit()
790:         text.setReadOnly(True)
791:         layout.addWidget(text)
792:         text.setPlainText(
793:             ",\n".join("{}={:.3}".format(attr, self._widgets[attr].value())
794:                        for attr in self._attrs))
795:         # Adjust the height of the text widget to fit the whole text, plus
796:         # some padding.
797:         size = text.maximumSize()
798:         size.setHeight(
799:             QtGui.QFontMetrics(text.document().defaultFont())
800:             .size(0, text.toPlainText()).height() + 20)
801:         text.setMaximumSize(size)
802:         dialog.exec_()
803: 
804:     def _on_value_changed(self):
805:         self._figure.subplots_adjust(**{attr: self._widgets[attr].value()
806:                                         for attr in self._attrs})
807:         self._figure.canvas.draw_idle()
808: 
809:     def _tight_layout(self):
810:         self._figure.tight_layout()
811:         for attr in self._attrs:
812:             widget = self._widgets[attr]
813:             widget.blockSignals(True)
814:             widget.setValue(vars(self._figure.subplotpars)[attr])
815:             widget.blockSignals(False)
816:         self._figure.canvas.draw_idle()
817: 
818:     def _reset(self):
819:         for attr, value in self._defaults.items():
820:             self._widgets[attr].setValue(value)
821: 
822: 
823: def error_msg_qt(msg, parent=None):
824:     if not isinstance(msg, six.string_types):
825:         msg = ','.join(map(str, msg))
826: 
827:     QtWidgets.QMessageBox.warning(None, "Matplotlib",
828:                                   msg, QtGui.QMessageBox.Ok)
829: 
830: 
831: def exception_handler(type, value, tb):
832:     '''Handle uncaught exceptions
833:     It does not catch SystemExit
834:     '''
835:     msg = ''
836:     # get the filename attribute if available (for IOError)
837:     if hasattr(value, 'filename') and value.filename is not None:
838:         msg = value.filename + ': '
839:     if hasattr(value, 'strerror') and value.strerror is not None:
840:         msg += value.strerror
841:     else:
842:         msg += six.text_type(value)
843: 
844:     if len(msg):
845:         error_msg_qt(msg)
846: 
847: 
848: @_Backend.export
849: class _BackendQT5(_Backend):
850:     FigureCanvas = FigureCanvasQT
851:     FigureManager = FigureManagerQT
852: 
853:     @staticmethod
854:     def trigger_manager_draw(manager):
855:         manager.canvas.draw_idle()
856: 
857:     @staticmethod
858:     def mainloop():
859:         # allow KeyboardInterrupt exceptions to close the plot window.
860:         signal.signal(signal.SIGINT, signal.SIG_DFL)
861:         global qApp
862:         qApp.exec_()
863: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import six' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249727 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'six')

if (type(import_249727) is not StypyTypeError):

    if (import_249727 != 'pyd_module'):
        __import__(import_249727)
        sys_modules_249728 = sys.modules[import_249727]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'six', sys_modules_249728.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'six', import_249727)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import functools' statement (line 5)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'functools', functools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import os' statement (line 6)
import os

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import re' statement (line 7)
import re

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import signal' statement (line 8)
import signal

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'signal', signal, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sys' statement (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from six import unichr' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249729 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six')

if (type(import_249729) is not StypyTypeError):

    if (import_249729 != 'pyd_module'):
        __import__(import_249729)
        sys_modules_249730 = sys.modules[import_249729]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six', sys_modules_249730.module_type_store, module_type_store, ['unichr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_249730, sys_modules_249730.module_type_store, module_type_store)
    else:
        from six import unichr

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'six', None, module_type_store, ['unichr'], [unichr])

else:
    # Assigning a type to the variable 'six' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'six', import_249729)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import matplotlib' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249731 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib')

if (type(import_249731) is not StypyTypeError):

    if (import_249731 != 'pyd_module'):
        __import__(import_249731)
        sys_modules_249732 = sys.modules[import_249731]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib', sys_modules_249732.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib', import_249731)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib._pylab_helpers import Gcf' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249733 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib._pylab_helpers')

if (type(import_249733) is not StypyTypeError):

    if (import_249733 != 'pyd_module'):
        __import__(import_249733)
        sys_modules_249734 = sys.modules[import_249733]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib._pylab_helpers', sys_modules_249734.module_type_store, module_type_store, ['Gcf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_249734, sys_modules_249734.module_type_store, module_type_store)
    else:
        from matplotlib._pylab_helpers import Gcf

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib._pylab_helpers', None, module_type_store, ['Gcf'], [Gcf])

else:
    # Assigning a type to the variable 'matplotlib._pylab_helpers' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib._pylab_helpers', import_249733)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase, cursors' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249735 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backend_bases')

if (type(import_249735) is not StypyTypeError):

    if (import_249735 != 'pyd_module'):
        __import__(import_249735)
        sys_modules_249736 = sys.modules[import_249735]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backend_bases', sys_modules_249736.module_type_store, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'NavigationToolbar2', 'TimerBase', 'cursors'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_249736, sys_modules_249736.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase, cursors

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backend_bases', None, module_type_store, ['_Backend', 'FigureCanvasBase', 'FigureManagerBase', 'NavigationToolbar2', 'TimerBase', 'cursors'], [_Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase, cursors])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'matplotlib.backend_bases', import_249735)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import matplotlib.backends.qt_editor.figureoptions' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249737 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.backends.qt_editor.figureoptions')

if (type(import_249737) is not StypyTypeError):

    if (import_249737 != 'pyd_module'):
        __import__(import_249737)
        sys_modules_249738 = sys.modules[import_249737]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'figureoptions', sys_modules_249738.module_type_store, module_type_store)
    else:
        import matplotlib.backends.qt_editor.figureoptions as figureoptions

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'figureoptions', matplotlib.backends.qt_editor.figureoptions, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.backends.qt_editor.figureoptions' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.backends.qt_editor.figureoptions', import_249737)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from matplotlib.backends.qt_editor.formsubplottool import UiSubplotTool' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249739 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends.qt_editor.formsubplottool')

if (type(import_249739) is not StypyTypeError):

    if (import_249739 != 'pyd_module'):
        __import__(import_249739)
        sys_modules_249740 = sys.modules[import_249739]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends.qt_editor.formsubplottool', sys_modules_249740.module_type_store, module_type_store, ['UiSubplotTool'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_249740, sys_modules_249740.module_type_store, module_type_store)
    else:
        from matplotlib.backends.qt_editor.formsubplottool import UiSubplotTool

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends.qt_editor.formsubplottool', None, module_type_store, ['UiSubplotTool'], [UiSubplotTool])

else:
    # Assigning a type to the variable 'matplotlib.backends.qt_editor.formsubplottool' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'matplotlib.backends.qt_editor.formsubplottool', import_249739)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from matplotlib.figure import Figure' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249741 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.figure')

if (type(import_249741) is not StypyTypeError):

    if (import_249741 != 'pyd_module'):
        __import__(import_249741)
        sys_modules_249742 = sys.modules[import_249741]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.figure', sys_modules_249742.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_249742, sys_modules_249742.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'matplotlib.figure', import_249741)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets, _getSaveFileName, is_pyqt5, __version__, QT_API' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_249743 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.backends.qt_compat')

if (type(import_249743) is not StypyTypeError):

    if (import_249743 != 'pyd_module'):
        __import__(import_249743)
        sys_modules_249744 = sys.modules[import_249743]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.backends.qt_compat', sys_modules_249744.module_type_store, module_type_store, ['QtCore', 'QtGui', 'QtWidgets', '_getSaveFileName', 'is_pyqt5', '__version__', 'QT_API'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_249744, sys_modules_249744.module_type_store, module_type_store)
    else:
        from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets, _getSaveFileName, is_pyqt5, __version__, QT_API

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.backends.qt_compat', None, module_type_store, ['QtCore', 'QtGui', 'QtWidgets', '_getSaveFileName', 'is_pyqt5', '__version__', 'QT_API'], [QtCore, QtGui, QtWidgets, _getSaveFileName, is_pyqt5, __version__, QT_API])

else:
    # Assigning a type to the variable 'matplotlib.backends.qt_compat' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'matplotlib.backends.qt_compat', import_249743)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Name to a Name (line 25):

# Assigning a Name to a Name (line 25):
# Getting the type of '__version__' (line 25)
version___249745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), '__version__')
# Assigning a type to the variable 'backend_version' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'backend_version', version___249745)

# Assigning a Dict to a Name (line 29):

# Assigning a Dict to a Name (line 29):

# Obtaining an instance of the builtin type 'dict' (line 29)
dict_249746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 29)
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 29)
QtCore_249747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 29)
Qt_249748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 16), QtCore_249747, 'Qt')
# Obtaining the member 'Key_Control' of a type (line 29)
Key_Control_249749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 16), Qt_249748, 'Key_Control')
unicode_249750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 39), 'unicode', u'control')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Control_249749, unicode_249750))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 30)
QtCore_249751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 30)
Qt_249752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 16), QtCore_249751, 'Qt')
# Obtaining the member 'Key_Shift' of a type (line 30)
Key_Shift_249753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 16), Qt_249752, 'Key_Shift')
unicode_249754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 37), 'unicode', u'shift')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Shift_249753, unicode_249754))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 31)
QtCore_249755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 31)
Qt_249756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), QtCore_249755, 'Qt')
# Obtaining the member 'Key_Alt' of a type (line 31)
Key_Alt_249757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), Qt_249756, 'Key_Alt')
unicode_249758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'unicode', u'alt')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Alt_249757, unicode_249758))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 32)
QtCore_249759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 32)
Qt_249760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), QtCore_249759, 'Qt')
# Obtaining the member 'Key_Meta' of a type (line 32)
Key_Meta_249761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), Qt_249760, 'Key_Meta')
unicode_249762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 36), 'unicode', u'super')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Meta_249761, unicode_249762))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 33)
QtCore_249763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 33)
Qt_249764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 16), QtCore_249763, 'Qt')
# Obtaining the member 'Key_Return' of a type (line 33)
Key_Return_249765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 16), Qt_249764, 'Key_Return')
unicode_249766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 38), 'unicode', u'enter')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Return_249765, unicode_249766))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 34)
QtCore_249767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 34)
Qt_249768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), QtCore_249767, 'Qt')
# Obtaining the member 'Key_Left' of a type (line 34)
Key_Left_249769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), Qt_249768, 'Key_Left')
unicode_249770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 36), 'unicode', u'left')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Left_249769, unicode_249770))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 35)
QtCore_249771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 35)
Qt_249772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 16), QtCore_249771, 'Qt')
# Obtaining the member 'Key_Up' of a type (line 35)
Key_Up_249773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 16), Qt_249772, 'Key_Up')
unicode_249774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 34), 'unicode', u'up')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Up_249773, unicode_249774))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 36)
QtCore_249775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 36)
Qt_249776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), QtCore_249775, 'Qt')
# Obtaining the member 'Key_Right' of a type (line 36)
Key_Right_249777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), Qt_249776, 'Key_Right')
unicode_249778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 37), 'unicode', u'right')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Right_249777, unicode_249778))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 37)
QtCore_249779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 37)
Qt_249780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 16), QtCore_249779, 'Qt')
# Obtaining the member 'Key_Down' of a type (line 37)
Key_Down_249781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 16), Qt_249780, 'Key_Down')
unicode_249782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 36), 'unicode', u'down')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Down_249781, unicode_249782))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 38)
QtCore_249783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 38)
Qt_249784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), QtCore_249783, 'Qt')
# Obtaining the member 'Key_Escape' of a type (line 38)
Key_Escape_249785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), Qt_249784, 'Key_Escape')
unicode_249786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 38), 'unicode', u'escape')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Escape_249785, unicode_249786))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 39)
QtCore_249787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 39)
Qt_249788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), QtCore_249787, 'Qt')
# Obtaining the member 'Key_F1' of a type (line 39)
Key_F1_249789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), Qt_249788, 'Key_F1')
unicode_249790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 34), 'unicode', u'f1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F1_249789, unicode_249790))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 40)
QtCore_249791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 40)
Qt_249792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), QtCore_249791, 'Qt')
# Obtaining the member 'Key_F2' of a type (line 40)
Key_F2_249793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), Qt_249792, 'Key_F2')
unicode_249794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 34), 'unicode', u'f2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F2_249793, unicode_249794))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 41)
QtCore_249795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 41)
Qt_249796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 16), QtCore_249795, 'Qt')
# Obtaining the member 'Key_F3' of a type (line 41)
Key_F3_249797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 16), Qt_249796, 'Key_F3')
unicode_249798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 34), 'unicode', u'f3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F3_249797, unicode_249798))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 42)
QtCore_249799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 42)
Qt_249800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), QtCore_249799, 'Qt')
# Obtaining the member 'Key_F4' of a type (line 42)
Key_F4_249801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), Qt_249800, 'Key_F4')
unicode_249802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 34), 'unicode', u'f4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F4_249801, unicode_249802))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 43)
QtCore_249803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 43)
Qt_249804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 16), QtCore_249803, 'Qt')
# Obtaining the member 'Key_F5' of a type (line 43)
Key_F5_249805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 16), Qt_249804, 'Key_F5')
unicode_249806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'unicode', u'f5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F5_249805, unicode_249806))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 44)
QtCore_249807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 44)
Qt_249808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 16), QtCore_249807, 'Qt')
# Obtaining the member 'Key_F6' of a type (line 44)
Key_F6_249809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 16), Qt_249808, 'Key_F6')
unicode_249810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'unicode', u'f6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F6_249809, unicode_249810))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 45)
QtCore_249811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 45)
Qt_249812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 16), QtCore_249811, 'Qt')
# Obtaining the member 'Key_F7' of a type (line 45)
Key_F7_249813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 16), Qt_249812, 'Key_F7')
unicode_249814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 34), 'unicode', u'f7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F7_249813, unicode_249814))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 46)
QtCore_249815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 46)
Qt_249816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 16), QtCore_249815, 'Qt')
# Obtaining the member 'Key_F8' of a type (line 46)
Key_F8_249817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 16), Qt_249816, 'Key_F8')
unicode_249818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 34), 'unicode', u'f8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F8_249817, unicode_249818))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 47)
QtCore_249819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 47)
Qt_249820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), QtCore_249819, 'Qt')
# Obtaining the member 'Key_F9' of a type (line 47)
Key_F9_249821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), Qt_249820, 'Key_F9')
unicode_249822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 34), 'unicode', u'f9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F9_249821, unicode_249822))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 48)
QtCore_249823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 48)
Qt_249824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), QtCore_249823, 'Qt')
# Obtaining the member 'Key_F10' of a type (line 48)
Key_F10_249825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), Qt_249824, 'Key_F10')
unicode_249826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 35), 'unicode', u'f10')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F10_249825, unicode_249826))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 49)
QtCore_249827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 49)
Qt_249828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), QtCore_249827, 'Qt')
# Obtaining the member 'Key_F11' of a type (line 49)
Key_F11_249829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), Qt_249828, 'Key_F11')
unicode_249830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 35), 'unicode', u'f11')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F11_249829, unicode_249830))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 50)
QtCore_249831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 50)
Qt_249832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 16), QtCore_249831, 'Qt')
# Obtaining the member 'Key_F12' of a type (line 50)
Key_F12_249833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 16), Qt_249832, 'Key_F12')
unicode_249834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 35), 'unicode', u'f12')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_F12_249833, unicode_249834))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 51)
QtCore_249835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 51)
Qt_249836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), QtCore_249835, 'Qt')
# Obtaining the member 'Key_Home' of a type (line 51)
Key_Home_249837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), Qt_249836, 'Key_Home')
unicode_249838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 36), 'unicode', u'home')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Home_249837, unicode_249838))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 52)
QtCore_249839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 52)
Qt_249840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), QtCore_249839, 'Qt')
# Obtaining the member 'Key_End' of a type (line 52)
Key_End_249841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), Qt_249840, 'Key_End')
unicode_249842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 35), 'unicode', u'end')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_End_249841, unicode_249842))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 53)
QtCore_249843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 53)
Qt_249844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 16), QtCore_249843, 'Qt')
# Obtaining the member 'Key_PageUp' of a type (line 53)
Key_PageUp_249845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 16), Qt_249844, 'Key_PageUp')
unicode_249846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'unicode', u'pageup')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_PageUp_249845, unicode_249846))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 54)
QtCore_249847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 54)
Qt_249848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), QtCore_249847, 'Qt')
# Obtaining the member 'Key_PageDown' of a type (line 54)
Key_PageDown_249849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), Qt_249848, 'Key_PageDown')
unicode_249850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 40), 'unicode', u'pagedown')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_PageDown_249849, unicode_249850))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 55)
QtCore_249851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 55)
Qt_249852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), QtCore_249851, 'Qt')
# Obtaining the member 'Key_Tab' of a type (line 55)
Key_Tab_249853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), Qt_249852, 'Key_Tab')
unicode_249854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 35), 'unicode', u'tab')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Tab_249853, unicode_249854))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 56)
QtCore_249855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 56)
Qt_249856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), QtCore_249855, 'Qt')
# Obtaining the member 'Key_Backspace' of a type (line 56)
Key_Backspace_249857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), Qt_249856, 'Key_Backspace')
unicode_249858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 41), 'unicode', u'backspace')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Backspace_249857, unicode_249858))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 57)
QtCore_249859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 57)
Qt_249860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), QtCore_249859, 'Qt')
# Obtaining the member 'Key_Enter' of a type (line 57)
Key_Enter_249861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), Qt_249860, 'Key_Enter')
unicode_249862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 37), 'unicode', u'enter')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Enter_249861, unicode_249862))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 58)
QtCore_249863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 58)
Qt_249864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), QtCore_249863, 'Qt')
# Obtaining the member 'Key_Insert' of a type (line 58)
Key_Insert_249865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), Qt_249864, 'Key_Insert')
unicode_249866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 38), 'unicode', u'insert')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Insert_249865, unicode_249866))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 59)
QtCore_249867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 59)
Qt_249868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), QtCore_249867, 'Qt')
# Obtaining the member 'Key_Delete' of a type (line 59)
Key_Delete_249869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), Qt_249868, 'Key_Delete')
unicode_249870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 38), 'unicode', u'delete')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Delete_249869, unicode_249870))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 60)
QtCore_249871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 60)
Qt_249872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), QtCore_249871, 'Qt')
# Obtaining the member 'Key_Pause' of a type (line 60)
Key_Pause_249873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), Qt_249872, 'Key_Pause')
unicode_249874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 37), 'unicode', u'pause')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Pause_249873, unicode_249874))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 61)
QtCore_249875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 61)
Qt_249876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), QtCore_249875, 'Qt')
# Obtaining the member 'Key_SysReq' of a type (line 61)
Key_SysReq_249877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), Qt_249876, 'Key_SysReq')
unicode_249878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 38), 'unicode', u'sysreq')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_SysReq_249877, unicode_249878))
# Adding element type (key, value) (line 29)
# Getting the type of 'QtCore' (line 62)
QtCore_249879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'QtCore')
# Obtaining the member 'Qt' of a type (line 62)
Qt_249880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), QtCore_249879, 'Qt')
# Obtaining the member 'Key_Clear' of a type (line 62)
Key_Clear_249881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), Qt_249880, 'Key_Clear')
unicode_249882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 37), 'unicode', u'clear')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), dict_249746, (Key_Clear_249881, unicode_249882))

# Assigning a type to the variable 'SPECIAL_KEYS' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'SPECIAL_KEYS', dict_249746)

# Assigning a Num to a Name (line 66):

# Assigning a Num to a Name (line 66):
int_249883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 8), 'int')
# Assigning a type to the variable 'SUPER' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'SUPER', int_249883)

# Assigning a Num to a Name (line 67):

# Assigning a Num to a Name (line 67):
int_249884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 6), 'int')
# Assigning a type to the variable 'ALT' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'ALT', int_249884)

# Assigning a Num to a Name (line 68):

# Assigning a Num to a Name (line 68):
int_249885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 7), 'int')
# Assigning a type to the variable 'CTRL' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'CTRL', int_249885)

# Assigning a Num to a Name (line 69):

# Assigning a Num to a Name (line 69):
int_249886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'int')
# Assigning a type to the variable 'SHIFT' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'SHIFT', int_249886)

# Assigning a List to a Name (line 70):

# Assigning a List to a Name (line 70):

# Obtaining an instance of the builtin type 'list' (line 70)
list_249887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 70)
# Adding element type (line 70)

# Obtaining an instance of the builtin type 'tuple' (line 70)
tuple_249888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 70)
# Adding element type (line 70)
unicode_249889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'unicode', u'super')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 18), tuple_249888, unicode_249889)
# Adding element type (line 70)
# Getting the type of 'QtCore' (line 70)
QtCore_249890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 27), 'QtCore')
# Obtaining the member 'Qt' of a type (line 70)
Qt_249891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 27), QtCore_249890, 'Qt')
# Obtaining the member 'MetaModifier' of a type (line 70)
MetaModifier_249892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 27), Qt_249891, 'MetaModifier')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 18), tuple_249888, MetaModifier_249892)
# Adding element type (line 70)
# Getting the type of 'QtCore' (line 70)
QtCore_249893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 51), 'QtCore')
# Obtaining the member 'Qt' of a type (line 70)
Qt_249894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 51), QtCore_249893, 'Qt')
# Obtaining the member 'Key_Meta' of a type (line 70)
Key_Meta_249895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 51), Qt_249894, 'Key_Meta')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 18), tuple_249888, Key_Meta_249895)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 16), list_249887, tuple_249888)
# Adding element type (line 70)

# Obtaining an instance of the builtin type 'tuple' (line 71)
tuple_249896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 71)
# Adding element type (line 71)
unicode_249897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 18), 'unicode', u'alt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), tuple_249896, unicode_249897)
# Adding element type (line 71)
# Getting the type of 'QtCore' (line 71)
QtCore_249898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'QtCore')
# Obtaining the member 'Qt' of a type (line 71)
Qt_249899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 25), QtCore_249898, 'Qt')
# Obtaining the member 'AltModifier' of a type (line 71)
AltModifier_249900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 25), Qt_249899, 'AltModifier')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), tuple_249896, AltModifier_249900)
# Adding element type (line 71)
# Getting the type of 'QtCore' (line 71)
QtCore_249901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 48), 'QtCore')
# Obtaining the member 'Qt' of a type (line 71)
Qt_249902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 48), QtCore_249901, 'Qt')
# Obtaining the member 'Key_Alt' of a type (line 71)
Key_Alt_249903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 48), Qt_249902, 'Key_Alt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 18), tuple_249896, Key_Alt_249903)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 16), list_249887, tuple_249896)
# Adding element type (line 70)

# Obtaining an instance of the builtin type 'tuple' (line 72)
tuple_249904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 72)
# Adding element type (line 72)
unicode_249905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 18), 'unicode', u'ctrl')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), tuple_249904, unicode_249905)
# Adding element type (line 72)
# Getting the type of 'QtCore' (line 72)
QtCore_249906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'QtCore')
# Obtaining the member 'Qt' of a type (line 72)
Qt_249907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 26), QtCore_249906, 'Qt')
# Obtaining the member 'ControlModifier' of a type (line 72)
ControlModifier_249908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 26), Qt_249907, 'ControlModifier')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), tuple_249904, ControlModifier_249908)
# Adding element type (line 72)
# Getting the type of 'QtCore' (line 72)
QtCore_249909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 53), 'QtCore')
# Obtaining the member 'Qt' of a type (line 72)
Qt_249910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 53), QtCore_249909, 'Qt')
# Obtaining the member 'Key_Control' of a type (line 72)
Key_Control_249911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 53), Qt_249910, 'Key_Control')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), tuple_249904, Key_Control_249911)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 16), list_249887, tuple_249904)
# Adding element type (line 70)

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_249912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)
# Adding element type (line 73)
unicode_249913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 18), 'unicode', u'shift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 18), tuple_249912, unicode_249913)
# Adding element type (line 73)
# Getting the type of 'QtCore' (line 73)
QtCore_249914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 27), 'QtCore')
# Obtaining the member 'Qt' of a type (line 73)
Qt_249915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 27), QtCore_249914, 'Qt')
# Obtaining the member 'ShiftModifier' of a type (line 73)
ShiftModifier_249916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 27), Qt_249915, 'ShiftModifier')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 18), tuple_249912, ShiftModifier_249916)
# Adding element type (line 73)
# Getting the type of 'QtCore' (line 73)
QtCore_249917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 52), 'QtCore')
# Obtaining the member 'Qt' of a type (line 73)
Qt_249918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 52), QtCore_249917, 'Qt')
# Obtaining the member 'Key_Shift' of a type (line 73)
Key_Shift_249919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 52), Qt_249918, 'Key_Shift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 18), tuple_249912, Key_Shift_249919)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 16), list_249887, tuple_249912)

# Assigning a type to the variable 'MODIFIER_KEYS' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'MODIFIER_KEYS', list_249887)


# Getting the type of 'sys' (line 76)
sys_249920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 3), 'sys')
# Obtaining the member 'platform' of a type (line 76)
platform_249921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 3), sys_249920, 'platform')
unicode_249922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 19), 'unicode', u'darwin')
# Applying the binary operator '==' (line 76)
result_eq_249923 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 3), '==', platform_249921, unicode_249922)

# Testing the type of an if condition (line 76)
if_condition_249924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 0), result_eq_249923)
# Assigning a type to the variable 'if_condition_249924' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'if_condition_249924', if_condition_249924)
# SSA begins for if statement (line 76)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to update(...): (line 79)
# Processing the call arguments (line 79)

# Obtaining an instance of the builtin type 'dict' (line 79)
dict_249927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 24), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 79)
# Adding element type (key, value) (line 79)
# Getting the type of 'QtCore' (line 79)
QtCore_249928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'QtCore', False)
# Obtaining the member 'Qt' of a type (line 79)
Qt_249929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), QtCore_249928, 'Qt')
# Obtaining the member 'Key_Control' of a type (line 79)
Key_Control_249930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), Qt_249929, 'Key_Control')
unicode_249931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 48), 'unicode', u'super')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 24), dict_249927, (Key_Control_249930, unicode_249931))
# Adding element type (key, value) (line 79)
# Getting the type of 'QtCore' (line 80)
QtCore_249932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'QtCore', False)
# Obtaining the member 'Qt' of a type (line 80)
Qt_249933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 25), QtCore_249932, 'Qt')
# Obtaining the member 'Key_Meta' of a type (line 80)
Key_Meta_249934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 25), Qt_249933, 'Key_Meta')
unicode_249935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 45), 'unicode', u'control')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 24), dict_249927, (Key_Meta_249934, unicode_249935))

# Processing the call keyword arguments (line 79)
kwargs_249936 = {}
# Getting the type of 'SPECIAL_KEYS' (line 79)
SPECIAL_KEYS_249925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'SPECIAL_KEYS', False)
# Obtaining the member 'update' of a type (line 79)
update_249926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 4), SPECIAL_KEYS_249925, 'update')
# Calling update(args, kwargs) (line 79)
update_call_result_249937 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), update_249926, *[dict_249927], **kwargs_249936)


# Assigning a Tuple to a Subscript (line 82):

# Assigning a Tuple to a Subscript (line 82):

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_249938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)
# Adding element type (line 82)
unicode_249939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 24), 'unicode', u'super')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 24), tuple_249938, unicode_249939)
# Adding element type (line 82)
# Getting the type of 'QtCore' (line 82)
QtCore_249940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 33), 'QtCore')
# Obtaining the member 'Qt' of a type (line 82)
Qt_249941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 33), QtCore_249940, 'Qt')
# Obtaining the member 'ControlModifier' of a type (line 82)
ControlModifier_249942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 33), Qt_249941, 'ControlModifier')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 24), tuple_249938, ControlModifier_249942)
# Adding element type (line 82)
# Getting the type of 'QtCore' (line 83)
QtCore_249943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'QtCore')
# Obtaining the member 'Qt' of a type (line 83)
Qt_249944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 24), QtCore_249943, 'Qt')
# Obtaining the member 'Key_Control' of a type (line 83)
Key_Control_249945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 24), Qt_249944, 'Key_Control')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 24), tuple_249938, Key_Control_249945)

# Getting the type of 'MODIFIER_KEYS' (line 82)
MODIFIER_KEYS_249946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'MODIFIER_KEYS')
int_249947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'int')
# Storing an element on a container (line 82)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 4), MODIFIER_KEYS_249946, (int_249947, tuple_249938))

# Assigning a Tuple to a Subscript (line 84):

# Assigning a Tuple to a Subscript (line 84):

# Obtaining an instance of the builtin type 'tuple' (line 84)
tuple_249948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 84)
# Adding element type (line 84)
unicode_249949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 24), 'unicode', u'ctrl')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 24), tuple_249948, unicode_249949)
# Adding element type (line 84)
# Getting the type of 'QtCore' (line 84)
QtCore_249950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'QtCore')
# Obtaining the member 'Qt' of a type (line 84)
Qt_249951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 32), QtCore_249950, 'Qt')
# Obtaining the member 'MetaModifier' of a type (line 84)
MetaModifier_249952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 32), Qt_249951, 'MetaModifier')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 24), tuple_249948, MetaModifier_249952)
# Adding element type (line 84)
# Getting the type of 'QtCore' (line 85)
QtCore_249953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'QtCore')
# Obtaining the member 'Qt' of a type (line 85)
Qt_249954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 24), QtCore_249953, 'Qt')
# Obtaining the member 'Key_Meta' of a type (line 85)
Key_Meta_249955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 24), Qt_249954, 'Key_Meta')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 24), tuple_249948, Key_Meta_249955)

# Getting the type of 'MODIFIER_KEYS' (line 84)
MODIFIER_KEYS_249956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'MODIFIER_KEYS')
int_249957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 18), 'int')
# Storing an element on a container (line 84)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 4), MODIFIER_KEYS_249956, (int_249957, tuple_249948))
# SSA join for if statement (line 76)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Dict to a Name (line 88):

# Assigning a Dict to a Name (line 88):

# Obtaining an instance of the builtin type 'dict' (line 88)
dict_249958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 88)
# Adding element type (key, value) (line 88)
# Getting the type of 'cursors' (line 89)
cursors_249959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'cursors')
# Obtaining the member 'MOVE' of a type (line 89)
MOVE_249960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), cursors_249959, 'MOVE')
# Getting the type of 'QtCore' (line 89)
QtCore_249961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'QtCore')
# Obtaining the member 'Qt' of a type (line 89)
Qt_249962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 18), QtCore_249961, 'Qt')
# Obtaining the member 'SizeAllCursor' of a type (line 89)
SizeAllCursor_249963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 18), Qt_249962, 'SizeAllCursor')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), dict_249958, (MOVE_249960, SizeAllCursor_249963))
# Adding element type (key, value) (line 88)
# Getting the type of 'cursors' (line 90)
cursors_249964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'cursors')
# Obtaining the member 'HAND' of a type (line 90)
HAND_249965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 4), cursors_249964, 'HAND')
# Getting the type of 'QtCore' (line 90)
QtCore_249966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'QtCore')
# Obtaining the member 'Qt' of a type (line 90)
Qt_249967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 18), QtCore_249966, 'Qt')
# Obtaining the member 'PointingHandCursor' of a type (line 90)
PointingHandCursor_249968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 18), Qt_249967, 'PointingHandCursor')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), dict_249958, (HAND_249965, PointingHandCursor_249968))
# Adding element type (key, value) (line 88)
# Getting the type of 'cursors' (line 91)
cursors_249969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'cursors')
# Obtaining the member 'POINTER' of a type (line 91)
POINTER_249970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 4), cursors_249969, 'POINTER')
# Getting the type of 'QtCore' (line 91)
QtCore_249971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'QtCore')
# Obtaining the member 'Qt' of a type (line 91)
Qt_249972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 21), QtCore_249971, 'Qt')
# Obtaining the member 'ArrowCursor' of a type (line 91)
ArrowCursor_249973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 21), Qt_249972, 'ArrowCursor')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), dict_249958, (POINTER_249970, ArrowCursor_249973))
# Adding element type (key, value) (line 88)
# Getting the type of 'cursors' (line 92)
cursors_249974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'cursors')
# Obtaining the member 'SELECT_REGION' of a type (line 92)
SELECT_REGION_249975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 4), cursors_249974, 'SELECT_REGION')
# Getting the type of 'QtCore' (line 92)
QtCore_249976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'QtCore')
# Obtaining the member 'Qt' of a type (line 92)
Qt_249977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 27), QtCore_249976, 'Qt')
# Obtaining the member 'CrossCursor' of a type (line 92)
CrossCursor_249978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 27), Qt_249977, 'CrossCursor')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), dict_249958, (SELECT_REGION_249975, CrossCursor_249978))
# Adding element type (key, value) (line 88)
# Getting the type of 'cursors' (line 93)
cursors_249979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'cursors')
# Obtaining the member 'WAIT' of a type (line 93)
WAIT_249980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), cursors_249979, 'WAIT')
# Getting the type of 'QtCore' (line 93)
QtCore_249981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'QtCore')
# Obtaining the member 'Qt' of a type (line 93)
Qt_249982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 18), QtCore_249981, 'Qt')
# Obtaining the member 'WaitCursor' of a type (line 93)
WaitCursor_249983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 18), Qt_249982, 'WaitCursor')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), dict_249958, (WAIT_249980, WaitCursor_249983))

# Assigning a type to the variable 'cursord' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'cursord', dict_249958)

# Assigning a Name to a Name (line 98):

# Assigning a Name to a Name (line 98):
# Getting the type of 'None' (line 98)
None_249984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 'None')
# Assigning a type to the variable 'qApp' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'qApp', None_249984)

@norecursion
def _create_qApp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_create_qApp'
    module_type_store = module_type_store.open_function_context('_create_qApp', 101, 0, False)
    
    # Passed parameters checking function
    _create_qApp.stypy_localization = localization
    _create_qApp.stypy_type_of_self = None
    _create_qApp.stypy_type_store = module_type_store
    _create_qApp.stypy_function_name = '_create_qApp'
    _create_qApp.stypy_param_names_list = []
    _create_qApp.stypy_varargs_param_name = None
    _create_qApp.stypy_kwargs_param_name = None
    _create_qApp.stypy_call_defaults = defaults
    _create_qApp.stypy_call_varargs = varargs
    _create_qApp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_create_qApp', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_create_qApp', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_create_qApp(...)' code ##################

    unicode_249985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, (-1)), 'unicode', u'\n    Only one qApp can exist at a time, so check before creating one.\n    ')
    # Marking variables as global (line 105)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 105, 4), 'qApp')
    
    # Type idiom detected: calculating its left and rigth part (line 107)
    # Getting the type of 'qApp' (line 107)
    qApp_249986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 7), 'qApp')
    # Getting the type of 'None' (line 107)
    None_249987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'None')
    
    (may_be_249988, more_types_in_union_249989) = may_be_none(qApp_249986, None_249987)

    if may_be_249988:

        if more_types_in_union_249989:
            # Runtime conditional SSA (line 107)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to instance(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_249993 = {}
        # Getting the type of 'QtWidgets' (line 108)
        QtWidgets_249990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 'QtWidgets', False)
        # Obtaining the member 'QApplication' of a type (line 108)
        QApplication_249991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), QtWidgets_249990, 'QApplication')
        # Obtaining the member 'instance' of a type (line 108)
        instance_249992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), QApplication_249991, 'instance')
        # Calling instance(args, kwargs) (line 108)
        instance_call_result_249994 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), instance_249992, *[], **kwargs_249993)
        
        # Assigning a type to the variable 'app' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'app', instance_call_result_249994)
        
        # Type idiom detected: calculating its left and rigth part (line 109)
        # Getting the type of 'app' (line 109)
        app_249995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'app')
        # Getting the type of 'None' (line 109)
        None_249996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'None')
        
        (may_be_249997, more_types_in_union_249998) = may_be_none(app_249995, None_249996)

        if may_be_249997:

            if more_types_in_union_249998:
                # Runtime conditional SSA (line 109)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Call to is_pyqt5(...): (line 111)
            # Processing the call keyword arguments (line 111)
            kwargs_250000 = {}
            # Getting the type of 'is_pyqt5' (line 111)
            is_pyqt5_249999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'is_pyqt5', False)
            # Calling is_pyqt5(args, kwargs) (line 111)
            is_pyqt5_call_result_250001 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), is_pyqt5_249999, *[], **kwargs_250000)
            
            # Testing the type of an if condition (line 111)
            if_condition_250002 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 12), is_pyqt5_call_result_250001)
            # Assigning a type to the variable 'if_condition_250002' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'if_condition_250002', if_condition_250002)
            # SSA begins for if statement (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # SSA begins for try-except statement (line 112)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 113, 20))
            
            # 'from PyQt5 import QtX11Extras' statement (line 113)
            update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
            import_250003 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 113, 20), 'PyQt5')

            if (type(import_250003) is not StypyTypeError):

                if (import_250003 != 'pyd_module'):
                    __import__(import_250003)
                    sys_modules_250004 = sys.modules[import_250003]
                    import_from_module(stypy.reporting.localization.Localization(__file__, 113, 20), 'PyQt5', sys_modules_250004.module_type_store, module_type_store, ['QtX11Extras'])
                    nest_module(stypy.reporting.localization.Localization(__file__, 113, 20), __file__, sys_modules_250004, sys_modules_250004.module_type_store, module_type_store)
                else:
                    from PyQt5 import QtX11Extras

                    import_from_module(stypy.reporting.localization.Localization(__file__, 113, 20), 'PyQt5', None, module_type_store, ['QtX11Extras'], [QtX11Extras])

            else:
                # Assigning a type to the variable 'PyQt5' (line 113)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'PyQt5', import_250003)

            remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')
            
            
            # Assigning a Name to a Name (line 114):
            
            # Assigning a Name to a Name (line 114):
            # Getting the type of 'True' (line 114)
            True_250005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 35), 'True')
            # Assigning a type to the variable 'is_x11_build' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'is_x11_build', True_250005)
            # SSA branch for the except part of a try statement (line 112)
            # SSA branch for the except 'ImportError' branch of a try statement (line 112)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Name to a Name (line 116):
            
            # Assigning a Name to a Name (line 116):
            # Getting the type of 'False' (line 116)
            False_250006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 35), 'False')
            # Assigning a type to the variable 'is_x11_build' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'is_x11_build', False_250006)
            # SSA join for try-except statement (line 112)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 111)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 118):
            
            # Assigning a Call to a Name (line 118):
            
            # Call to hasattr(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'QtGui' (line 118)
            QtGui_250008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 39), 'QtGui', False)
            unicode_250009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 46), 'unicode', u'QX11Info')
            # Processing the call keyword arguments (line 118)
            kwargs_250010 = {}
            # Getting the type of 'hasattr' (line 118)
            hasattr_250007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'hasattr', False)
            # Calling hasattr(args, kwargs) (line 118)
            hasattr_call_result_250011 = invoke(stypy.reporting.localization.Localization(__file__, 118, 31), hasattr_250007, *[QtGui_250008, unicode_250009], **kwargs_250010)
            
            # Assigning a type to the variable 'is_x11_build' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'is_x11_build', hasattr_call_result_250011)
            # SSA join for if statement (line 111)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'is_x11_build' (line 119)
            is_x11_build_250012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'is_x11_build')
            # Testing the type of an if condition (line 119)
            if_condition_250013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 12), is_x11_build_250012)
            # Assigning a type to the variable 'if_condition_250013' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'if_condition_250013', if_condition_250013)
            # SSA begins for if statement (line 119)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 120):
            
            # Assigning a Call to a Name (line 120):
            
            # Call to get(...): (line 120)
            # Processing the call arguments (line 120)
            unicode_250017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 41), 'unicode', u'DISPLAY')
            # Processing the call keyword arguments (line 120)
            kwargs_250018 = {}
            # Getting the type of 'os' (line 120)
            os_250014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 26), 'os', False)
            # Obtaining the member 'environ' of a type (line 120)
            environ_250015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 26), os_250014, 'environ')
            # Obtaining the member 'get' of a type (line 120)
            get_250016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 26), environ_250015, 'get')
            # Calling get(args, kwargs) (line 120)
            get_call_result_250019 = invoke(stypy.reporting.localization.Localization(__file__, 120, 26), get_250016, *[unicode_250017], **kwargs_250018)
            
            # Assigning a type to the variable 'display' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'display', get_call_result_250019)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'display' (line 121)
            display_250020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'display')
            # Getting the type of 'None' (line 121)
            None_250021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), 'None')
            # Applying the binary operator 'is' (line 121)
            result_is__250022 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 19), 'is', display_250020, None_250021)
            
            
            
            # Call to search(...): (line 121)
            # Processing the call arguments (line 121)
            unicode_250025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 52), 'unicode', u':\\d')
            # Getting the type of 'display' (line 121)
            display_250026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 60), 'display', False)
            # Processing the call keyword arguments (line 121)
            kwargs_250027 = {}
            # Getting the type of 're' (line 121)
            re_250023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 42), 're', False)
            # Obtaining the member 'search' of a type (line 121)
            search_250024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 42), re_250023, 'search')
            # Calling search(args, kwargs) (line 121)
            search_call_result_250028 = invoke(stypy.reporting.localization.Localization(__file__, 121, 42), search_250024, *[unicode_250025, display_250026], **kwargs_250027)
            
            # Applying the 'not' unary operator (line 121)
            result_not__250029 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 38), 'not', search_call_result_250028)
            
            # Applying the binary operator 'or' (line 121)
            result_or_keyword_250030 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 19), 'or', result_is__250022, result_not__250029)
            
            # Testing the type of an if condition (line 121)
            if_condition_250031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 16), result_or_keyword_250030)
            # Assigning a type to the variable 'if_condition_250031' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'if_condition_250031', if_condition_250031)
            # SSA begins for if statement (line 121)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to RuntimeError(...): (line 122)
            # Processing the call arguments (line 122)
            unicode_250033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 39), 'unicode', u'Invalid DISPLAY variable')
            # Processing the call keyword arguments (line 122)
            kwargs_250034 = {}
            # Getting the type of 'RuntimeError' (line 122)
            RuntimeError_250032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 26), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 122)
            RuntimeError_call_result_250035 = invoke(stypy.reporting.localization.Localization(__file__, 122, 26), RuntimeError_250032, *[unicode_250033], **kwargs_250034)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 122, 20), RuntimeError_call_result_250035, 'raise parameter', BaseException)
            # SSA join for if statement (line 121)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 119)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 124):
            
            # Assigning a Call to a Name (line 124):
            
            # Call to QApplication(...): (line 124)
            # Processing the call arguments (line 124)
            
            # Obtaining an instance of the builtin type 'list' (line 124)
            list_250038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 42), 'list')
            # Adding type elements to the builtin type 'list' instance (line 124)
            # Adding element type (line 124)
            str_250039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 43), 'str', 'matplotlib')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 42), list_250038, str_250039)
            
            # Processing the call keyword arguments (line 124)
            kwargs_250040 = {}
            # Getting the type of 'QtWidgets' (line 124)
            QtWidgets_250036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'QtWidgets', False)
            # Obtaining the member 'QApplication' of a type (line 124)
            QApplication_250037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 19), QtWidgets_250036, 'QApplication')
            # Calling QApplication(args, kwargs) (line 124)
            QApplication_call_result_250041 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), QApplication_250037, *[list_250038], **kwargs_250040)
            
            # Assigning a type to the variable 'qApp' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'qApp', QApplication_call_result_250041)
            
            # Call to connect(...): (line 125)
            # Processing the call arguments (line 125)
            # Getting the type of 'qApp' (line 125)
            qApp_250045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 42), 'qApp', False)
            # Obtaining the member 'quit' of a type (line 125)
            quit_250046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 42), qApp_250045, 'quit')
            # Processing the call keyword arguments (line 125)
            kwargs_250047 = {}
            # Getting the type of 'qApp' (line 125)
            qApp_250042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'qApp', False)
            # Obtaining the member 'lastWindowClosed' of a type (line 125)
            lastWindowClosed_250043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), qApp_250042, 'lastWindowClosed')
            # Obtaining the member 'connect' of a type (line 125)
            connect_250044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), lastWindowClosed_250043, 'connect')
            # Calling connect(args, kwargs) (line 125)
            connect_call_result_250048 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), connect_250044, *[quit_250046], **kwargs_250047)
            

            if more_types_in_union_249998:
                # Runtime conditional SSA for else branch (line 109)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_249997) or more_types_in_union_249998):
            
            # Assigning a Name to a Name (line 127):
            
            # Assigning a Name to a Name (line 127):
            # Getting the type of 'app' (line 127)
            app_250049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'app')
            # Assigning a type to the variable 'qApp' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'qApp', app_250049)

            if (may_be_249997 and more_types_in_union_249998):
                # SSA join for if statement (line 109)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_249989:
            # SSA join for if statement (line 107)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to is_pyqt5(...): (line 129)
    # Processing the call keyword arguments (line 129)
    kwargs_250051 = {}
    # Getting the type of 'is_pyqt5' (line 129)
    is_pyqt5_250050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 7), 'is_pyqt5', False)
    # Calling is_pyqt5(args, kwargs) (line 129)
    is_pyqt5_call_result_250052 = invoke(stypy.reporting.localization.Localization(__file__, 129, 7), is_pyqt5_250050, *[], **kwargs_250051)
    
    # Testing the type of an if condition (line 129)
    if_condition_250053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 4), is_pyqt5_call_result_250052)
    # Assigning a type to the variable 'if_condition_250053' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'if_condition_250053', if_condition_250053)
    # SSA begins for if statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to setAttribute(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'QtCore' (line 131)
    QtCore_250056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'QtCore', False)
    # Obtaining the member 'Qt' of a type (line 131)
    Qt_250057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 30), QtCore_250056, 'Qt')
    # Obtaining the member 'AA_UseHighDpiPixmaps' of a type (line 131)
    AA_UseHighDpiPixmaps_250058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 30), Qt_250057, 'AA_UseHighDpiPixmaps')
    # Processing the call keyword arguments (line 131)
    kwargs_250059 = {}
    # Getting the type of 'qApp' (line 131)
    qApp_250054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'qApp', False)
    # Obtaining the member 'setAttribute' of a type (line 131)
    setAttribute_250055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), qApp_250054, 'setAttribute')
    # Calling setAttribute(args, kwargs) (line 131)
    setAttribute_call_result_250060 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), setAttribute_250055, *[AA_UseHighDpiPixmaps_250058], **kwargs_250059)
    
    
    # Call to setAttribute(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'QtCore' (line 132)
    QtCore_250063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 30), 'QtCore', False)
    # Obtaining the member 'Qt' of a type (line 132)
    Qt_250064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 30), QtCore_250063, 'Qt')
    # Obtaining the member 'AA_EnableHighDpiScaling' of a type (line 132)
    AA_EnableHighDpiScaling_250065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 30), Qt_250064, 'AA_EnableHighDpiScaling')
    # Processing the call keyword arguments (line 132)
    kwargs_250066 = {}
    # Getting the type of 'qApp' (line 132)
    qApp_250061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'qApp', False)
    # Obtaining the member 'setAttribute' of a type (line 132)
    setAttribute_250062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), qApp_250061, 'setAttribute')
    # Calling setAttribute(args, kwargs) (line 132)
    setAttribute_call_result_250067 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), setAttribute_250062, *[AA_EnableHighDpiScaling_250065], **kwargs_250066)
    
    # SSA branch for the except part of a try statement (line 130)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 130)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 129)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_create_qApp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_create_qApp' in the type store
    # Getting the type of 'stypy_return_type' (line 101)
    stypy_return_type_250068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_250068)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_create_qApp'
    return stypy_return_type_250068

# Assigning a type to the variable '_create_qApp' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), '_create_qApp', _create_qApp)

@norecursion
def _allow_super_init(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_allow_super_init'
    module_type_store = module_type_store.open_function_context('_allow_super_init', 137, 0, False)
    
    # Passed parameters checking function
    _allow_super_init.stypy_localization = localization
    _allow_super_init.stypy_type_of_self = None
    _allow_super_init.stypy_type_store = module_type_store
    _allow_super_init.stypy_function_name = '_allow_super_init'
    _allow_super_init.stypy_param_names_list = ['__init__']
    _allow_super_init.stypy_varargs_param_name = None
    _allow_super_init.stypy_kwargs_param_name = None
    _allow_super_init.stypy_call_defaults = defaults
    _allow_super_init.stypy_call_varargs = varargs
    _allow_super_init.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_allow_super_init', ['__init__'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_allow_super_init', localization, ['__init__'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_allow_super_init(...)' code ##################

    unicode_250069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, (-1)), 'unicode', u'\n    Decorator for ``__init__`` to allow ``super().__init__`` on PyQt4/PySide2.\n    ')
    
    
    # Getting the type of 'QT_API' (line 142)
    QT_API_250070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 7), 'QT_API')
    unicode_250071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 17), 'unicode', u'PyQt5')
    # Applying the binary operator '==' (line 142)
    result_eq_250072 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 7), '==', QT_API_250070, unicode_250071)
    
    # Testing the type of an if condition (line 142)
    if_condition_250073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 4), result_eq_250072)
    # Assigning a type to the variable 'if_condition_250073' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'if_condition_250073', if_condition_250073)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '__init__' (line 144)
    init___250074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), '__init__')
    # Assigning a type to the variable 'stypy_return_type' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'stypy_return_type', init___250074)
    # SSA branch for the else part of an if statement (line 142)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 156):
    
    # Assigning a Attribute to a Name (line 156):
    # Getting the type of 'QtWidgets' (line 156)
    QtWidgets_250075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'QtWidgets')
    # Obtaining the member 'QWidget' of a type (line 156)
    QWidget_250076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 23), QtWidgets_250075, 'QWidget')
    # Obtaining the member '__init__' of a type (line 156)
    init___250077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 23), QWidget_250076, '__init__')
    # Assigning a type to the variable 'qwidget_init' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'qwidget_init', init___250077)

    @norecursion
    def cooperative_qwidget_init(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cooperative_qwidget_init'
        module_type_store = module_type_store.open_function_context('cooperative_qwidget_init', 158, 8, False)
        
        # Passed parameters checking function
        cooperative_qwidget_init.stypy_localization = localization
        cooperative_qwidget_init.stypy_type_of_self = None
        cooperative_qwidget_init.stypy_type_store = module_type_store
        cooperative_qwidget_init.stypy_function_name = 'cooperative_qwidget_init'
        cooperative_qwidget_init.stypy_param_names_list = ['self']
        cooperative_qwidget_init.stypy_varargs_param_name = 'args'
        cooperative_qwidget_init.stypy_kwargs_param_name = 'kwargs'
        cooperative_qwidget_init.stypy_call_defaults = defaults
        cooperative_qwidget_init.stypy_call_varargs = varargs
        cooperative_qwidget_init.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'cooperative_qwidget_init', ['self'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cooperative_qwidget_init', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cooperative_qwidget_init(...)' code ##################

        
        # Call to qwidget_init(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'self' (line 159)
        self_250079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'self', False)
        # Processing the call keyword arguments (line 159)
        kwargs_250080 = {}
        # Getting the type of 'qwidget_init' (line 159)
        qwidget_init_250078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'qwidget_init', False)
        # Calling qwidget_init(args, kwargs) (line 159)
        qwidget_init_call_result_250081 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), qwidget_init_250078, *[self_250079], **kwargs_250080)
        
        
        # Assigning a Attribute to a Name (line 160):
        
        # Assigning a Attribute to a Name (line 160):
        
        # Call to type(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'self' (line 160)
        self_250083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 23), 'self', False)
        # Processing the call keyword arguments (line 160)
        kwargs_250084 = {}
        # Getting the type of 'type' (line 160)
        type_250082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 18), 'type', False)
        # Calling type(args, kwargs) (line 160)
        type_call_result_250085 = invoke(stypy.reporting.localization.Localization(__file__, 160, 18), type_250082, *[self_250083], **kwargs_250084)
        
        # Obtaining the member '__mro__' of a type (line 160)
        mro___250086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 18), type_call_result_250085, '__mro__')
        # Assigning a type to the variable 'mro' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'mro', mro___250086)
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to next(...): (line 161)
        # Processing the call arguments (line 161)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 162, 16, True)
        # Calculating comprehension expression
        
        # Obtaining the type of the subscript
        
        # Call to index(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'QtWidgets' (line 162)
        QtWidgets_250107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 45), 'QtWidgets', False)
        # Obtaining the member 'QWidget' of a type (line 162)
        QWidget_250108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 45), QtWidgets_250107, 'QWidget')
        # Processing the call keyword arguments (line 162)
        kwargs_250109 = {}
        # Getting the type of 'mro' (line 162)
        mro_250105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 35), 'mro', False)
        # Obtaining the member 'index' of a type (line 162)
        index_250106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 35), mro_250105, 'index')
        # Calling index(args, kwargs) (line 162)
        index_call_result_250110 = invoke(stypy.reporting.localization.Localization(__file__, 162, 35), index_250106, *[QWidget_250108], **kwargs_250109)
        
        int_250111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 66), 'int')
        # Applying the binary operator '+' (line 162)
        result_add_250112 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 35), '+', index_call_result_250110, int_250111)
        
        slice_250113 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 162, 31), result_add_250112, None, None)
        # Getting the type of 'mro' (line 162)
        mro_250114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 31), 'mro', False)
        # Obtaining the member '__getitem__' of a type (line 162)
        getitem___250115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 31), mro_250114, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 162)
        subscript_call_result_250116 = invoke(stypy.reporting.localization.Localization(__file__, 162, 31), getitem___250115, slice_250113)
        
        comprehension_250117 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 16), subscript_call_result_250116)
        # Assigning a type to the variable 'cls' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'cls', comprehension_250117)
        
        
        # Obtaining the type of the subscript
        int_250089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 45), 'int')
        
        # Call to split(...): (line 163)
        # Processing the call arguments (line 163)
        unicode_250093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 40), 'unicode', u'.')
        # Processing the call keyword arguments (line 163)
        kwargs_250094 = {}
        # Getting the type of 'cls' (line 163)
        cls_250090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'cls', False)
        # Obtaining the member '__module__' of a type (line 163)
        module___250091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), cls_250090, '__module__')
        # Obtaining the member 'split' of a type (line 163)
        split_250092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), module___250091, 'split')
        # Calling split(args, kwargs) (line 163)
        split_call_result_250095 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), split_250092, *[unicode_250093], **kwargs_250094)
        
        # Obtaining the member '__getitem__' of a type (line 163)
        getitem___250096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), split_call_result_250095, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 163)
        subscript_call_result_250097 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), getitem___250096, int_250089)
        
        
        # Obtaining an instance of the builtin type 'list' (line 163)
        list_250098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 163)
        # Adding element type (line 163)
        unicode_250099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 20), 'unicode', u'PyQt4')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 55), list_250098, unicode_250099)
        # Adding element type (line 163)
        unicode_250100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 29), 'unicode', u'sip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 55), list_250098, unicode_250100)
        # Adding element type (line 163)
        unicode_250101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 36), 'unicode', u'PySide')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 55), list_250098, unicode_250101)
        # Adding element type (line 163)
        unicode_250102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 46), 'unicode', u'PySide2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 55), list_250098, unicode_250102)
        # Adding element type (line 163)
        unicode_250103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 57), 'unicode', u'Shiboken')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 55), list_250098, unicode_250103)
        
        # Applying the binary operator 'notin' (line 163)
        result_contains_250104 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 19), 'notin', subscript_call_result_250097, list_250098)
        
        # Getting the type of 'cls' (line 162)
        cls_250088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'cls', False)
        list_250118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 16), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 16), list_250118, cls_250088)
        # Processing the call keyword arguments (line 161)
        kwargs_250119 = {}
        # Getting the type of 'next' (line 161)
        next_250087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'next', False)
        # Calling next(args, kwargs) (line 161)
        next_call_result_250120 = invoke(stypy.reporting.localization.Localization(__file__, 161, 29), next_250087, *[list_250118], **kwargs_250119)
        
        # Assigning a type to the variable 'next_coop_init' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'next_coop_init', next_call_result_250120)
        
        # Call to __init__(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'self' (line 165)
        self_250123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 36), 'self', False)
        # Getting the type of 'args' (line 165)
        args_250124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 43), 'args', False)
        # Processing the call keyword arguments (line 165)
        # Getting the type of 'kwargs' (line 165)
        kwargs_250125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 51), 'kwargs', False)
        kwargs_250126 = {'kwargs_250125': kwargs_250125}
        # Getting the type of 'next_coop_init' (line 165)
        next_coop_init_250121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'next_coop_init', False)
        # Obtaining the member '__init__' of a type (line 165)
        init___250122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), next_coop_init_250121, '__init__')
        # Calling __init__(args, kwargs) (line 165)
        init___call_result_250127 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), init___250122, *[self_250123, args_250124], **kwargs_250126)
        
        
        # ################# End of 'cooperative_qwidget_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cooperative_qwidget_init' in the type store
        # Getting the type of 'stypy_return_type' (line 158)
        stypy_return_type_250128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250128)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cooperative_qwidget_init'
        return stypy_return_type_250128

    # Assigning a type to the variable 'cooperative_qwidget_init' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'cooperative_qwidget_init', cooperative_qwidget_init)

    @norecursion
    def wrapper(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrapper'
        module_type_store = module_type_store.open_function_context('wrapper', 167, 8, False)
        
        # Passed parameters checking function
        wrapper.stypy_localization = localization
        wrapper.stypy_type_of_self = None
        wrapper.stypy_type_store = module_type_store
        wrapper.stypy_function_name = 'wrapper'
        wrapper.stypy_param_names_list = ['self']
        wrapper.stypy_varargs_param_name = None
        wrapper.stypy_kwargs_param_name = 'kwargs'
        wrapper.stypy_call_defaults = defaults
        wrapper.stypy_call_varargs = varargs
        wrapper.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrapper', ['self'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrapper', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrapper(...)' code ##################

        
        # Try-finally block (line 169)
        
        # Assigning a Name to a Attribute (line 170):
        
        # Assigning a Name to a Attribute (line 170):
        # Getting the type of 'cooperative_qwidget_init' (line 170)
        cooperative_qwidget_init_250129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 45), 'cooperative_qwidget_init')
        # Getting the type of 'QtWidgets' (line 170)
        QtWidgets_250130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'QtWidgets')
        # Obtaining the member 'QWidget' of a type (line 170)
        QWidget_250131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), QtWidgets_250130, 'QWidget')
        # Setting the type of the member '__init__' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), QWidget_250131, '__init__', cooperative_qwidget_init_250129)
        
        # Call to __init__(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'self' (line 171)
        self_250133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 25), 'self', False)
        # Processing the call keyword arguments (line 171)
        # Getting the type of 'kwargs' (line 171)
        kwargs_250134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 33), 'kwargs', False)
        kwargs_250135 = {'kwargs_250134': kwargs_250134}
        # Getting the type of '__init__' (line 171)
        init___250132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), '__init__', False)
        # Calling __init__(args, kwargs) (line 171)
        init___call_result_250136 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), init___250132, *[self_250133], **kwargs_250135)
        
        
        # finally branch of the try-finally block (line 169)
        
        # Assigning a Name to a Attribute (line 174):
        
        # Assigning a Name to a Attribute (line 174):
        # Getting the type of 'qwidget_init' (line 174)
        qwidget_init_250137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 45), 'qwidget_init')
        # Getting the type of 'QtWidgets' (line 174)
        QtWidgets_250138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'QtWidgets')
        # Obtaining the member 'QWidget' of a type (line 174)
        QWidget_250139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), QtWidgets_250138, 'QWidget')
        # Setting the type of the member '__init__' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), QWidget_250139, '__init__', qwidget_init_250137)
        
        
        # ################# End of 'wrapper(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrapper' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_250140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250140)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrapper'
        return stypy_return_type_250140

    # Assigning a type to the variable 'wrapper' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'wrapper', wrapper)
    # Getting the type of 'wrapper' (line 176)
    wrapper_250141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'wrapper')
    # Assigning a type to the variable 'stypy_return_type' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', wrapper_250141)
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_allow_super_init(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_allow_super_init' in the type store
    # Getting the type of 'stypy_return_type' (line 137)
    stypy_return_type_250142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_250142)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_allow_super_init'
    return stypy_return_type_250142

# Assigning a type to the variable '_allow_super_init' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), '_allow_super_init', _allow_super_init)
# Declaration of the 'TimerQT' class
# Getting the type of 'TimerBase' (line 179)
TimerBase_250143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 14), 'TimerBase')

class TimerQT(TimerBase_250143, ):
    unicode_250144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, (-1)), 'unicode', u'\n    Subclass of :class:`backend_bases.TimerBase` that uses Qt timer events.\n\n    Attributes\n    ----------\n    interval : int\n        The time between timer events in milliseconds. Default is 1000 ms.\n    single_shot : bool\n        Boolean flag indicating whether this timer should\n        operate as single shot (run once and then stop). Defaults to False.\n    callbacks : list\n        Stores list of (func, args) tuples that will be called upon timer\n        events. This list can be manipulated directly, or the functions\n        `add_callback` and `remove_callback` can be used.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerQT.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'self' (line 198)
        self_250147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'self', False)
        # Getting the type of 'args' (line 198)
        args_250148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 34), 'args', False)
        # Processing the call keyword arguments (line 198)
        # Getting the type of 'kwargs' (line 198)
        kwargs_250149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 42), 'kwargs', False)
        kwargs_250150 = {'kwargs_250149': kwargs_250149}
        # Getting the type of 'TimerBase' (line 198)
        TimerBase_250145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'TimerBase', False)
        # Obtaining the member '__init__' of a type (line 198)
        init___250146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), TimerBase_250145, '__init__')
        # Calling __init__(args, kwargs) (line 198)
        init___call_result_250151 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), init___250146, *[self_250147, args_250148], **kwargs_250150)
        
        
        # Assigning a Call to a Attribute (line 202):
        
        # Assigning a Call to a Attribute (line 202):
        
        # Call to QTimer(...): (line 202)
        # Processing the call keyword arguments (line 202)
        kwargs_250154 = {}
        # Getting the type of 'QtCore' (line 202)
        QtCore_250152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'QtCore', False)
        # Obtaining the member 'QTimer' of a type (line 202)
        QTimer_250153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 22), QtCore_250152, 'QTimer')
        # Calling QTimer(args, kwargs) (line 202)
        QTimer_call_result_250155 = invoke(stypy.reporting.localization.Localization(__file__, 202, 22), QTimer_250153, *[], **kwargs_250154)
        
        # Getting the type of 'self' (line 202)
        self_250156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self')
        # Setting the type of the member '_timer' of a type (line 202)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_250156, '_timer', QTimer_call_result_250155)
        
        # Call to connect(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'self' (line 203)
        self_250161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 36), 'self', False)
        # Obtaining the member '_on_timer' of a type (line 203)
        _on_timer_250162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 36), self_250161, '_on_timer')
        # Processing the call keyword arguments (line 203)
        kwargs_250163 = {}
        # Getting the type of 'self' (line 203)
        self_250157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self', False)
        # Obtaining the member '_timer' of a type (line 203)
        _timer_250158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_250157, '_timer')
        # Obtaining the member 'timeout' of a type (line 203)
        timeout_250159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), _timer_250158, 'timeout')
        # Obtaining the member 'connect' of a type (line 203)
        connect_250160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), timeout_250159, 'connect')
        # Calling connect(args, kwargs) (line 203)
        connect_call_result_250164 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), connect_250160, *[_on_timer_250162], **kwargs_250163)
        
        
        # Call to _timer_set_interval(...): (line 204)
        # Processing the call keyword arguments (line 204)
        kwargs_250167 = {}
        # Getting the type of 'self' (line 204)
        self_250165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self', False)
        # Obtaining the member '_timer_set_interval' of a type (line 204)
        _timer_set_interval_250166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_250165, '_timer_set_interval')
        # Calling _timer_set_interval(args, kwargs) (line 204)
        _timer_set_interval_call_result_250168 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), _timer_set_interval_250166, *[], **kwargs_250167)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _timer_set_single_shot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_timer_set_single_shot'
        module_type_store = module_type_store.open_function_context('_timer_set_single_shot', 206, 4, False)
        # Assigning a type to the variable 'self' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimerQT._timer_set_single_shot.__dict__.__setitem__('stypy_localization', localization)
        TimerQT._timer_set_single_shot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimerQT._timer_set_single_shot.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimerQT._timer_set_single_shot.__dict__.__setitem__('stypy_function_name', 'TimerQT._timer_set_single_shot')
        TimerQT._timer_set_single_shot.__dict__.__setitem__('stypy_param_names_list', [])
        TimerQT._timer_set_single_shot.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimerQT._timer_set_single_shot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimerQT._timer_set_single_shot.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimerQT._timer_set_single_shot.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimerQT._timer_set_single_shot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimerQT._timer_set_single_shot.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerQT._timer_set_single_shot', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_timer_set_single_shot', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_timer_set_single_shot(...)' code ##################

        
        # Call to setSingleShot(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'self' (line 207)
        self_250172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 34), 'self', False)
        # Obtaining the member '_single' of a type (line 207)
        _single_250173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 34), self_250172, '_single')
        # Processing the call keyword arguments (line 207)
        kwargs_250174 = {}
        # Getting the type of 'self' (line 207)
        self_250169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self', False)
        # Obtaining the member '_timer' of a type (line 207)
        _timer_250170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_250169, '_timer')
        # Obtaining the member 'setSingleShot' of a type (line 207)
        setSingleShot_250171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), _timer_250170, 'setSingleShot')
        # Calling setSingleShot(args, kwargs) (line 207)
        setSingleShot_call_result_250175 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), setSingleShot_250171, *[_single_250173], **kwargs_250174)
        
        
        # ################# End of '_timer_set_single_shot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_timer_set_single_shot' in the type store
        # Getting the type of 'stypy_return_type' (line 206)
        stypy_return_type_250176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250176)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_timer_set_single_shot'
        return stypy_return_type_250176


    @norecursion
    def _timer_set_interval(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_timer_set_interval'
        module_type_store = module_type_store.open_function_context('_timer_set_interval', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimerQT._timer_set_interval.__dict__.__setitem__('stypy_localization', localization)
        TimerQT._timer_set_interval.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimerQT._timer_set_interval.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimerQT._timer_set_interval.__dict__.__setitem__('stypy_function_name', 'TimerQT._timer_set_interval')
        TimerQT._timer_set_interval.__dict__.__setitem__('stypy_param_names_list', [])
        TimerQT._timer_set_interval.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimerQT._timer_set_interval.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimerQT._timer_set_interval.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimerQT._timer_set_interval.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimerQT._timer_set_interval.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimerQT._timer_set_interval.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerQT._timer_set_interval', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to setInterval(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'self' (line 210)
        self_250180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 32), 'self', False)
        # Obtaining the member '_interval' of a type (line 210)
        _interval_250181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 32), self_250180, '_interval')
        # Processing the call keyword arguments (line 210)
        kwargs_250182 = {}
        # Getting the type of 'self' (line 210)
        self_250177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'self', False)
        # Obtaining the member '_timer' of a type (line 210)
        _timer_250178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), self_250177, '_timer')
        # Obtaining the member 'setInterval' of a type (line 210)
        setInterval_250179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), _timer_250178, 'setInterval')
        # Calling setInterval(args, kwargs) (line 210)
        setInterval_call_result_250183 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), setInterval_250179, *[_interval_250181], **kwargs_250182)
        
        
        # ################# End of '_timer_set_interval(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_timer_set_interval' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_250184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250184)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_timer_set_interval'
        return stypy_return_type_250184


    @norecursion
    def _timer_start(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_timer_start'
        module_type_store = module_type_store.open_function_context('_timer_start', 212, 4, False)
        # Assigning a type to the variable 'self' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimerQT._timer_start.__dict__.__setitem__('stypy_localization', localization)
        TimerQT._timer_start.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimerQT._timer_start.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimerQT._timer_start.__dict__.__setitem__('stypy_function_name', 'TimerQT._timer_start')
        TimerQT._timer_start.__dict__.__setitem__('stypy_param_names_list', [])
        TimerQT._timer_start.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimerQT._timer_start.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimerQT._timer_start.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimerQT._timer_start.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimerQT._timer_start.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimerQT._timer_start.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerQT._timer_start', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to start(...): (line 213)
        # Processing the call keyword arguments (line 213)
        kwargs_250188 = {}
        # Getting the type of 'self' (line 213)
        self_250185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'self', False)
        # Obtaining the member '_timer' of a type (line 213)
        _timer_250186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), self_250185, '_timer')
        # Obtaining the member 'start' of a type (line 213)
        start_250187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), _timer_250186, 'start')
        # Calling start(args, kwargs) (line 213)
        start_call_result_250189 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), start_250187, *[], **kwargs_250188)
        
        
        # ################# End of '_timer_start(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_timer_start' in the type store
        # Getting the type of 'stypy_return_type' (line 212)
        stypy_return_type_250190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250190)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_timer_start'
        return stypy_return_type_250190


    @norecursion
    def _timer_stop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_timer_stop'
        module_type_store = module_type_store.open_function_context('_timer_stop', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimerQT._timer_stop.__dict__.__setitem__('stypy_localization', localization)
        TimerQT._timer_stop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimerQT._timer_stop.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimerQT._timer_stop.__dict__.__setitem__('stypy_function_name', 'TimerQT._timer_stop')
        TimerQT._timer_stop.__dict__.__setitem__('stypy_param_names_list', [])
        TimerQT._timer_stop.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimerQT._timer_stop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimerQT._timer_stop.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimerQT._timer_stop.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimerQT._timer_stop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimerQT._timer_stop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerQT._timer_stop', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to stop(...): (line 216)
        # Processing the call keyword arguments (line 216)
        kwargs_250194 = {}
        # Getting the type of 'self' (line 216)
        self_250191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'self', False)
        # Obtaining the member '_timer' of a type (line 216)
        _timer_250192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), self_250191, '_timer')
        # Obtaining the member 'stop' of a type (line 216)
        stop_250193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), _timer_250192, 'stop')
        # Calling stop(args, kwargs) (line 216)
        stop_call_result_250195 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), stop_250193, *[], **kwargs_250194)
        
        
        # ################# End of '_timer_stop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_timer_stop' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_250196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_timer_stop'
        return stypy_return_type_250196


# Assigning a type to the variable 'TimerQT' (line 179)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'TimerQT', TimerQT)
# Declaration of the 'FigureCanvasQT' class
# Getting the type of 'QtWidgets' (line 219)
QtWidgets_250197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'QtWidgets')
# Obtaining the member 'QWidget' of a type (line 219)
QWidget_250198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 21), QtWidgets_250197, 'QWidget')
# Getting the type of 'FigureCanvasBase' (line 219)
FigureCanvasBase_250199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 40), 'FigureCanvasBase')

class FigureCanvasQT(QWidget_250198, FigureCanvasBase_250199, ):
    
    # Assigning a Dict to a Name (line 222):

    @norecursion
    def _update_figure_dpi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update_figure_dpi'
        module_type_store = module_type_store.open_function_context('_update_figure_dpi', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT._update_figure_dpi.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT._update_figure_dpi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT._update_figure_dpi.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT._update_figure_dpi.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT._update_figure_dpi')
        FigureCanvasQT._update_figure_dpi.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQT._update_figure_dpi.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT._update_figure_dpi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT._update_figure_dpi.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT._update_figure_dpi.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT._update_figure_dpi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT._update_figure_dpi.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT._update_figure_dpi', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update_figure_dpi', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update_figure_dpi(...)' code ##################

        
        # Assigning a BinOp to a Name (line 230):
        
        # Assigning a BinOp to a Name (line 230):
        # Getting the type of 'self' (line 230)
        self_250200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 14), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 230)
        _dpi_ratio_250201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 14), self_250200, '_dpi_ratio')
        # Getting the type of 'self' (line 230)
        self_250202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 32), 'self')
        # Obtaining the member 'figure' of a type (line 230)
        figure_250203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 32), self_250202, 'figure')
        # Obtaining the member '_original_dpi' of a type (line 230)
        _original_dpi_250204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 32), figure_250203, '_original_dpi')
        # Applying the binary operator '*' (line 230)
        result_mul_250205 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 14), '*', _dpi_ratio_250201, _original_dpi_250204)
        
        # Assigning a type to the variable 'dpi' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'dpi', result_mul_250205)
        
        # Call to _set_dpi(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'dpi' (line 231)
        dpi_250209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 29), 'dpi', False)
        # Processing the call keyword arguments (line 231)
        # Getting the type of 'False' (line 231)
        False_250210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 42), 'False', False)
        keyword_250211 = False_250210
        kwargs_250212 = {'forward': keyword_250211}
        # Getting the type of 'self' (line 231)
        self_250206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 231)
        figure_250207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), self_250206, 'figure')
        # Obtaining the member '_set_dpi' of a type (line 231)
        _set_dpi_250208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), figure_250207, '_set_dpi')
        # Calling _set_dpi(args, kwargs) (line 231)
        _set_dpi_call_result_250213 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), _set_dpi_250208, *[dpi_250209], **kwargs_250212)
        
        
        # ################# End of '_update_figure_dpi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update_figure_dpi' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_250214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250214)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update_figure_dpi'
        return stypy_return_type_250214


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.__init__', ['figure'], None, None, defaults, varargs, kwargs)

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

        
        # Call to _create_qApp(...): (line 235)
        # Processing the call keyword arguments (line 235)
        kwargs_250216 = {}
        # Getting the type of '_create_qApp' (line 235)
        _create_qApp_250215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), '_create_qApp', False)
        # Calling _create_qApp(args, kwargs) (line 235)
        _create_qApp_call_result_250217 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), _create_qApp_250215, *[], **kwargs_250216)
        
        
        # Assigning a Attribute to a Attribute (line 236):
        
        # Assigning a Attribute to a Attribute (line 236):
        # Getting the type of 'figure' (line 236)
        figure_250218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 31), 'figure')
        # Obtaining the member 'dpi' of a type (line 236)
        dpi_250219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 31), figure_250218, 'dpi')
        # Getting the type of 'figure' (line 236)
        figure_250220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'figure')
        # Setting the type of the member '_original_dpi' of a type (line 236)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), figure_250220, '_original_dpi', dpi_250219)
        
        # Call to __init__(...): (line 238)
        # Processing the call keyword arguments (line 238)
        # Getting the type of 'figure' (line 238)
        figure_250227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 52), 'figure', False)
        keyword_250228 = figure_250227
        kwargs_250229 = {'figure': keyword_250228}
        
        # Call to super(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'FigureCanvasQT' (line 238)
        FigureCanvasQT_250222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 14), 'FigureCanvasQT', False)
        # Getting the type of 'self' (line 238)
        self_250223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 30), 'self', False)
        # Processing the call keyword arguments (line 238)
        kwargs_250224 = {}
        # Getting the type of 'super' (line 238)
        super_250221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'super', False)
        # Calling super(args, kwargs) (line 238)
        super_call_result_250225 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), super_250221, *[FigureCanvasQT_250222, self_250223], **kwargs_250224)
        
        # Obtaining the member '__init__' of a type (line 238)
        init___250226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), super_call_result_250225, '__init__')
        # Calling __init__(args, kwargs) (line 238)
        init___call_result_250230 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), init___250226, *[], **kwargs_250229)
        
        
        # Assigning a Name to a Attribute (line 240):
        
        # Assigning a Name to a Attribute (line 240):
        # Getting the type of 'figure' (line 240)
        figure_250231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 22), 'figure')
        # Getting the type of 'self' (line 240)
        self_250232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'self')
        # Setting the type of the member 'figure' of a type (line 240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), self_250232, 'figure', figure_250231)
        
        # Call to _update_figure_dpi(...): (line 241)
        # Processing the call keyword arguments (line 241)
        kwargs_250235 = {}
        # Getting the type of 'self' (line 241)
        self_250233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'self', False)
        # Obtaining the member '_update_figure_dpi' of a type (line 241)
        _update_figure_dpi_250234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), self_250233, '_update_figure_dpi')
        # Calling _update_figure_dpi(args, kwargs) (line 241)
        _update_figure_dpi_call_result_250236 = invoke(stypy.reporting.localization.Localization(__file__, 241, 8), _update_figure_dpi_250234, *[], **kwargs_250235)
        
        
        # Assigning a Call to a Tuple (line 243):
        
        # Assigning a Call to a Name:
        
        # Call to get_width_height(...): (line 243)
        # Processing the call keyword arguments (line 243)
        kwargs_250239 = {}
        # Getting the type of 'self' (line 243)
        self_250237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'self', False)
        # Obtaining the member 'get_width_height' of a type (line 243)
        get_width_height_250238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 15), self_250237, 'get_width_height')
        # Calling get_width_height(args, kwargs) (line 243)
        get_width_height_call_result_250240 = invoke(stypy.reporting.localization.Localization(__file__, 243, 15), get_width_height_250238, *[], **kwargs_250239)
        
        # Assigning a type to the variable 'call_assignment_249696' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'call_assignment_249696', get_width_height_call_result_250240)
        
        # Assigning a Call to a Name (line 243):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250244 = {}
        # Getting the type of 'call_assignment_249696' (line 243)
        call_assignment_249696_250241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'call_assignment_249696', False)
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___250242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), call_assignment_249696_250241, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250245 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250242, *[int_250243], **kwargs_250244)
        
        # Assigning a type to the variable 'call_assignment_249697' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'call_assignment_249697', getitem___call_result_250245)
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'call_assignment_249697' (line 243)
        call_assignment_249697_250246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'call_assignment_249697')
        # Assigning a type to the variable 'w' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'w', call_assignment_249697_250246)
        
        # Assigning a Call to a Name (line 243):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250250 = {}
        # Getting the type of 'call_assignment_249696' (line 243)
        call_assignment_249696_250247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'call_assignment_249696', False)
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___250248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), call_assignment_249696_250247, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250251 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250248, *[int_250249], **kwargs_250250)
        
        # Assigning a type to the variable 'call_assignment_249698' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'call_assignment_249698', getitem___call_result_250251)
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'call_assignment_249698' (line 243)
        call_assignment_249698_250252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'call_assignment_249698')
        # Assigning a type to the variable 'h' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'h', call_assignment_249698_250252)
        
        # Call to resize(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'w' (line 244)
        w_250255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'w', False)
        # Getting the type of 'h' (line 244)
        h_250256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 23), 'h', False)
        # Processing the call keyword arguments (line 244)
        kwargs_250257 = {}
        # Getting the type of 'self' (line 244)
        self_250253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self', False)
        # Obtaining the member 'resize' of a type (line 244)
        resize_250254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_250253, 'resize')
        # Calling resize(args, kwargs) (line 244)
        resize_call_result_250258 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), resize_250254, *[w_250255, h_250256], **kwargs_250257)
        
        
        # Call to setMouseTracking(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'True' (line 246)
        True_250261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 30), 'True', False)
        # Processing the call keyword arguments (line 246)
        kwargs_250262 = {}
        # Getting the type of 'self' (line 246)
        self_250259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'self', False)
        # Obtaining the member 'setMouseTracking' of a type (line 246)
        setMouseTracking_250260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), self_250259, 'setMouseTracking')
        # Calling setMouseTracking(args, kwargs) (line 246)
        setMouseTracking_call_result_250263 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), setMouseTracking_250260, *[True_250261], **kwargs_250262)
        
        
        # Assigning a Name to a Attribute (line 248):
        
        # Assigning a Name to a Attribute (line 248):
        # Getting the type of 'True' (line 248)
        True_250264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 30), 'True')
        # Getting the type of 'self' (line 248)
        self_250265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'self')
        # Setting the type of the member '_keyautorepeat' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), self_250265, '_keyautorepeat', True_250264)
        
        # Assigning a Name to a Attribute (line 257):
        
        # Assigning a Name to a Attribute (line 257):
        # Getting the type of 'None' (line 257)
        None_250266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 31), 'None')
        # Getting the type of 'self' (line 257)
        self_250267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self')
        # Setting the type of the member '_dpi_ratio_prev' of a type (line 257)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_250267, '_dpi_ratio_prev', None_250266)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _dpi_ratio(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dpi_ratio'
        module_type_store = module_type_store.open_function_context('_dpi_ratio', 259, 4, False)
        # Assigning a type to the variable 'self' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT._dpi_ratio.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT._dpi_ratio.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT._dpi_ratio.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT._dpi_ratio.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT._dpi_ratio')
        FigureCanvasQT._dpi_ratio.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQT._dpi_ratio.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT._dpi_ratio.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT._dpi_ratio.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT._dpi_ratio.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT._dpi_ratio.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT._dpi_ratio.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT._dpi_ratio', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dpi_ratio', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dpi_ratio(...)' code ##################

        
        
        # SSA begins for try-except statement (line 262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to devicePixelRatio(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_250270 = {}
        # Getting the type of 'self' (line 263)
        self_250268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'self', False)
        # Obtaining the member 'devicePixelRatio' of a type (line 263)
        devicePixelRatio_250269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 19), self_250268, 'devicePixelRatio')
        # Calling devicePixelRatio(args, kwargs) (line 263)
        devicePixelRatio_call_result_250271 = invoke(stypy.reporting.localization.Localization(__file__, 263, 19), devicePixelRatio_250269, *[], **kwargs_250270)
        
        # Assigning a type to the variable 'stypy_return_type' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'stypy_return_type', devicePixelRatio_call_result_250271)
        # SSA branch for the except part of a try statement (line 262)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 262)
        module_type_store.open_ssa_branch('except')
        int_250272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'stypy_return_type', int_250272)
        # SSA join for try-except statement (line 262)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_dpi_ratio(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dpi_ratio' in the type store
        # Getting the type of 'stypy_return_type' (line 259)
        stypy_return_type_250273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250273)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dpi_ratio'
        return stypy_return_type_250273


    @norecursion
    def get_width_height(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_width_height'
        module_type_store = module_type_store.open_function_context('get_width_height', 267, 4, False)
        # Assigning a type to the variable 'self' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.get_width_height.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.get_width_height.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.get_width_height.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.get_width_height.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.get_width_height')
        FigureCanvasQT.get_width_height.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQT.get_width_height.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.get_width_height.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.get_width_height.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.get_width_height.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.get_width_height.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.get_width_height.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.get_width_height', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_width_height', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_width_height(...)' code ##################

        
        # Assigning a Call to a Tuple (line 268):
        
        # Assigning a Call to a Name:
        
        # Call to get_width_height(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'self' (line 268)
        self_250276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 49), 'self', False)
        # Processing the call keyword arguments (line 268)
        kwargs_250277 = {}
        # Getting the type of 'FigureCanvasBase' (line 268)
        FigureCanvasBase_250274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'FigureCanvasBase', False)
        # Obtaining the member 'get_width_height' of a type (line 268)
        get_width_height_250275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 15), FigureCanvasBase_250274, 'get_width_height')
        # Calling get_width_height(args, kwargs) (line 268)
        get_width_height_call_result_250278 = invoke(stypy.reporting.localization.Localization(__file__, 268, 15), get_width_height_250275, *[self_250276], **kwargs_250277)
        
        # Assigning a type to the variable 'call_assignment_249699' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'call_assignment_249699', get_width_height_call_result_250278)
        
        # Assigning a Call to a Name (line 268):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250282 = {}
        # Getting the type of 'call_assignment_249699' (line 268)
        call_assignment_249699_250279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'call_assignment_249699', False)
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___250280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), call_assignment_249699_250279, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250283 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250280, *[int_250281], **kwargs_250282)
        
        # Assigning a type to the variable 'call_assignment_249700' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'call_assignment_249700', getitem___call_result_250283)
        
        # Assigning a Name to a Name (line 268):
        # Getting the type of 'call_assignment_249700' (line 268)
        call_assignment_249700_250284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'call_assignment_249700')
        # Assigning a type to the variable 'w' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'w', call_assignment_249700_250284)
        
        # Assigning a Call to a Name (line 268):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250288 = {}
        # Getting the type of 'call_assignment_249699' (line 268)
        call_assignment_249699_250285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'call_assignment_249699', False)
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___250286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), call_assignment_249699_250285, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250289 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250286, *[int_250287], **kwargs_250288)
        
        # Assigning a type to the variable 'call_assignment_249701' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'call_assignment_249701', getitem___call_result_250289)
        
        # Assigning a Name to a Name (line 268):
        # Getting the type of 'call_assignment_249701' (line 268)
        call_assignment_249701_250290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'call_assignment_249701')
        # Assigning a type to the variable 'h' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), 'h', call_assignment_249701_250290)
        
        # Obtaining an instance of the builtin type 'tuple' (line 269)
        tuple_250291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 269)
        # Adding element type (line 269)
        
        # Call to int(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'w' (line 269)
        w_250293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 19), 'w', False)
        # Getting the type of 'self' (line 269)
        self_250294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 23), 'self', False)
        # Obtaining the member '_dpi_ratio' of a type (line 269)
        _dpi_ratio_250295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 23), self_250294, '_dpi_ratio')
        # Applying the binary operator 'div' (line 269)
        result_div_250296 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 19), 'div', w_250293, _dpi_ratio_250295)
        
        # Processing the call keyword arguments (line 269)
        kwargs_250297 = {}
        # Getting the type of 'int' (line 269)
        int_250292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'int', False)
        # Calling int(args, kwargs) (line 269)
        int_call_result_250298 = invoke(stypy.reporting.localization.Localization(__file__, 269, 15), int_250292, *[result_div_250296], **kwargs_250297)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 15), tuple_250291, int_call_result_250298)
        # Adding element type (line 269)
        
        # Call to int(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'h' (line 269)
        h_250300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'h', False)
        # Getting the type of 'self' (line 269)
        self_250301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 49), 'self', False)
        # Obtaining the member '_dpi_ratio' of a type (line 269)
        _dpi_ratio_250302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 49), self_250301, '_dpi_ratio')
        # Applying the binary operator 'div' (line 269)
        result_div_250303 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 45), 'div', h_250300, _dpi_ratio_250302)
        
        # Processing the call keyword arguments (line 269)
        kwargs_250304 = {}
        # Getting the type of 'int' (line 269)
        int_250299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 41), 'int', False)
        # Calling int(args, kwargs) (line 269)
        int_call_result_250305 = invoke(stypy.reporting.localization.Localization(__file__, 269, 41), int_250299, *[result_div_250303], **kwargs_250304)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 15), tuple_250291, int_call_result_250305)
        
        # Assigning a type to the variable 'stypy_return_type' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'stypy_return_type', tuple_250291)
        
        # ################# End of 'get_width_height(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_width_height' in the type store
        # Getting the type of 'stypy_return_type' (line 267)
        stypy_return_type_250306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250306)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_width_height'
        return stypy_return_type_250306


    @norecursion
    def enterEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'enterEvent'
        module_type_store = module_type_store.open_function_context('enterEvent', 271, 4, False)
        # Assigning a type to the variable 'self' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.enterEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.enterEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.enterEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.enterEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.enterEvent')
        FigureCanvasQT.enterEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.enterEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.enterEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.enterEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.enterEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.enterEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.enterEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.enterEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'enterEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'enterEvent(...)' code ##################

        
        # Call to enter_notify_event(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'self' (line 272)
        self_250309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 44), 'self', False)
        # Processing the call keyword arguments (line 272)
        # Getting the type of 'event' (line 272)
        event_250310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 59), 'event', False)
        keyword_250311 = event_250310
        kwargs_250312 = {'guiEvent': keyword_250311}
        # Getting the type of 'FigureCanvasBase' (line 272)
        FigureCanvasBase_250307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'enter_notify_event' of a type (line 272)
        enter_notify_event_250308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), FigureCanvasBase_250307, 'enter_notify_event')
        # Calling enter_notify_event(args, kwargs) (line 272)
        enter_notify_event_call_result_250313 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), enter_notify_event_250308, *[self_250309], **kwargs_250312)
        
        
        # ################# End of 'enterEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'enterEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 271)
        stypy_return_type_250314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250314)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'enterEvent'
        return stypy_return_type_250314


    @norecursion
    def leaveEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'leaveEvent'
        module_type_store = module_type_store.open_function_context('leaveEvent', 274, 4, False)
        # Assigning a type to the variable 'self' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.leaveEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.leaveEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.leaveEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.leaveEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.leaveEvent')
        FigureCanvasQT.leaveEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.leaveEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.leaveEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.leaveEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.leaveEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.leaveEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.leaveEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.leaveEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'leaveEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'leaveEvent(...)' code ##################

        
        # Call to restoreOverrideCursor(...): (line 275)
        # Processing the call keyword arguments (line 275)
        kwargs_250318 = {}
        # Getting the type of 'QtWidgets' (line 275)
        QtWidgets_250315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'QtWidgets', False)
        # Obtaining the member 'QApplication' of a type (line 275)
        QApplication_250316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), QtWidgets_250315, 'QApplication')
        # Obtaining the member 'restoreOverrideCursor' of a type (line 275)
        restoreOverrideCursor_250317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), QApplication_250316, 'restoreOverrideCursor')
        # Calling restoreOverrideCursor(args, kwargs) (line 275)
        restoreOverrideCursor_call_result_250319 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), restoreOverrideCursor_250317, *[], **kwargs_250318)
        
        
        # Call to leave_notify_event(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'self' (line 276)
        self_250322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 44), 'self', False)
        # Processing the call keyword arguments (line 276)
        # Getting the type of 'event' (line 276)
        event_250323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 59), 'event', False)
        keyword_250324 = event_250323
        kwargs_250325 = {'guiEvent': keyword_250324}
        # Getting the type of 'FigureCanvasBase' (line 276)
        FigureCanvasBase_250320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'leave_notify_event' of a type (line 276)
        leave_notify_event_250321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), FigureCanvasBase_250320, 'leave_notify_event')
        # Calling leave_notify_event(args, kwargs) (line 276)
        leave_notify_event_call_result_250326 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), leave_notify_event_250321, *[self_250322], **kwargs_250325)
        
        
        # ################# End of 'leaveEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'leaveEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_250327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250327)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'leaveEvent'
        return stypy_return_type_250327


    @norecursion
    def mouseEventCoords(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mouseEventCoords'
        module_type_store = module_type_store.open_function_context('mouseEventCoords', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.mouseEventCoords.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.mouseEventCoords.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.mouseEventCoords.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.mouseEventCoords.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.mouseEventCoords')
        FigureCanvasQT.mouseEventCoords.__dict__.__setitem__('stypy_param_names_list', ['pos'])
        FigureCanvasQT.mouseEventCoords.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.mouseEventCoords.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.mouseEventCoords.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.mouseEventCoords.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.mouseEventCoords.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.mouseEventCoords.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.mouseEventCoords', ['pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mouseEventCoords', localization, ['pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mouseEventCoords(...)' code ##################

        unicode_250328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, (-1)), 'unicode', u'Calculate mouse coordinates in physical pixels\n\n        Qt5 use logical pixels, but the figure is scaled to physical\n        pixels for rendering.   Transform to physical pixels so that\n        all of the down-stream transforms work as expected.\n\n        Also, the origin is different and needs to be corrected.\n\n        ')
        
        # Assigning a Attribute to a Name (line 288):
        
        # Assigning a Attribute to a Name (line 288):
        # Getting the type of 'self' (line 288)
        self_250329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 20), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 288)
        _dpi_ratio_250330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 20), self_250329, '_dpi_ratio')
        # Assigning a type to the variable 'dpi_ratio' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'dpi_ratio', _dpi_ratio_250330)
        
        # Assigning a Call to a Name (line 289):
        
        # Assigning a Call to a Name (line 289):
        
        # Call to x(...): (line 289)
        # Processing the call keyword arguments (line 289)
        kwargs_250333 = {}
        # Getting the type of 'pos' (line 289)
        pos_250331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'pos', False)
        # Obtaining the member 'x' of a type (line 289)
        x_250332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 12), pos_250331, 'x')
        # Calling x(args, kwargs) (line 289)
        x_call_result_250334 = invoke(stypy.reporting.localization.Localization(__file__, 289, 12), x_250332, *[], **kwargs_250333)
        
        # Assigning a type to the variable 'x' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'x', x_call_result_250334)
        
        # Assigning a BinOp to a Name (line 291):
        
        # Assigning a BinOp to a Name (line 291):
        # Getting the type of 'self' (line 291)
        self_250335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'self')
        # Obtaining the member 'figure' of a type (line 291)
        figure_250336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 12), self_250335, 'figure')
        # Obtaining the member 'bbox' of a type (line 291)
        bbox_250337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 12), figure_250336, 'bbox')
        # Obtaining the member 'height' of a type (line 291)
        height_250338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 12), bbox_250337, 'height')
        # Getting the type of 'dpi_ratio' (line 291)
        dpi_ratio_250339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 38), 'dpi_ratio')
        # Applying the binary operator 'div' (line 291)
        result_div_250340 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 12), 'div', height_250338, dpi_ratio_250339)
        
        
        # Call to y(...): (line 291)
        # Processing the call keyword arguments (line 291)
        kwargs_250343 = {}
        # Getting the type of 'pos' (line 291)
        pos_250341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 50), 'pos', False)
        # Obtaining the member 'y' of a type (line 291)
        y_250342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 50), pos_250341, 'y')
        # Calling y(args, kwargs) (line 291)
        y_call_result_250344 = invoke(stypy.reporting.localization.Localization(__file__, 291, 50), y_250342, *[], **kwargs_250343)
        
        # Applying the binary operator '-' (line 291)
        result_sub_250345 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 12), '-', result_div_250340, y_call_result_250344)
        
        # Assigning a type to the variable 'y' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'y', result_sub_250345)
        
        # Obtaining an instance of the builtin type 'tuple' (line 292)
        tuple_250346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 292)
        # Adding element type (line 292)
        # Getting the type of 'x' (line 292)
        x_250347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'x')
        # Getting the type of 'dpi_ratio' (line 292)
        dpi_ratio_250348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 19), 'dpi_ratio')
        # Applying the binary operator '*' (line 292)
        result_mul_250349 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 15), '*', x_250347, dpi_ratio_250348)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 15), tuple_250346, result_mul_250349)
        # Adding element type (line 292)
        # Getting the type of 'y' (line 292)
        y_250350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'y')
        # Getting the type of 'dpi_ratio' (line 292)
        dpi_ratio_250351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 34), 'dpi_ratio')
        # Applying the binary operator '*' (line 292)
        result_mul_250352 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 30), '*', y_250350, dpi_ratio_250351)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 15), tuple_250346, result_mul_250352)
        
        # Assigning a type to the variable 'stypy_return_type' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'stypy_return_type', tuple_250346)
        
        # ################# End of 'mouseEventCoords(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mouseEventCoords' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_250353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mouseEventCoords'
        return stypy_return_type_250353


    @norecursion
    def mousePressEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mousePressEvent'
        module_type_store = module_type_store.open_function_context('mousePressEvent', 294, 4, False)
        # Assigning a type to the variable 'self' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.mousePressEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.mousePressEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.mousePressEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.mousePressEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.mousePressEvent')
        FigureCanvasQT.mousePressEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.mousePressEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.mousePressEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.mousePressEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.mousePressEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.mousePressEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.mousePressEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.mousePressEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mousePressEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mousePressEvent(...)' code ##################

        
        # Assigning a Call to a Tuple (line 295):
        
        # Assigning a Call to a Name:
        
        # Call to mouseEventCoords(...): (line 295)
        # Processing the call arguments (line 295)
        
        # Call to pos(...): (line 295)
        # Processing the call keyword arguments (line 295)
        kwargs_250358 = {}
        # Getting the type of 'event' (line 295)
        event_250356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 37), 'event', False)
        # Obtaining the member 'pos' of a type (line 295)
        pos_250357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 37), event_250356, 'pos')
        # Calling pos(args, kwargs) (line 295)
        pos_call_result_250359 = invoke(stypy.reporting.localization.Localization(__file__, 295, 37), pos_250357, *[], **kwargs_250358)
        
        # Processing the call keyword arguments (line 295)
        kwargs_250360 = {}
        # Getting the type of 'self' (line 295)
        self_250354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'self', False)
        # Obtaining the member 'mouseEventCoords' of a type (line 295)
        mouseEventCoords_250355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 15), self_250354, 'mouseEventCoords')
        # Calling mouseEventCoords(args, kwargs) (line 295)
        mouseEventCoords_call_result_250361 = invoke(stypy.reporting.localization.Localization(__file__, 295, 15), mouseEventCoords_250355, *[pos_call_result_250359], **kwargs_250360)
        
        # Assigning a type to the variable 'call_assignment_249702' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_249702', mouseEventCoords_call_result_250361)
        
        # Assigning a Call to a Name (line 295):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250365 = {}
        # Getting the type of 'call_assignment_249702' (line 295)
        call_assignment_249702_250362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_249702', False)
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___250363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), call_assignment_249702_250362, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250366 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250363, *[int_250364], **kwargs_250365)
        
        # Assigning a type to the variable 'call_assignment_249703' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_249703', getitem___call_result_250366)
        
        # Assigning a Name to a Name (line 295):
        # Getting the type of 'call_assignment_249703' (line 295)
        call_assignment_249703_250367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_249703')
        # Assigning a type to the variable 'x' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'x', call_assignment_249703_250367)
        
        # Assigning a Call to a Name (line 295):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250371 = {}
        # Getting the type of 'call_assignment_249702' (line 295)
        call_assignment_249702_250368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_249702', False)
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___250369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), call_assignment_249702_250368, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250372 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250369, *[int_250370], **kwargs_250371)
        
        # Assigning a type to the variable 'call_assignment_249704' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_249704', getitem___call_result_250372)
        
        # Assigning a Name to a Name (line 295):
        # Getting the type of 'call_assignment_249704' (line 295)
        call_assignment_249704_250373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'call_assignment_249704')
        # Assigning a type to the variable 'y' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 11), 'y', call_assignment_249704_250373)
        
        # Assigning a Call to a Name (line 296):
        
        # Assigning a Call to a Name (line 296):
        
        # Call to get(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Call to button(...): (line 296)
        # Processing the call keyword arguments (line 296)
        kwargs_250379 = {}
        # Getting the type of 'event' (line 296)
        event_250377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 34), 'event', False)
        # Obtaining the member 'button' of a type (line 296)
        button_250378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 34), event_250377, 'button')
        # Calling button(args, kwargs) (line 296)
        button_call_result_250380 = invoke(stypy.reporting.localization.Localization(__file__, 296, 34), button_250378, *[], **kwargs_250379)
        
        # Processing the call keyword arguments (line 296)
        kwargs_250381 = {}
        # Getting the type of 'self' (line 296)
        self_250374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 17), 'self', False)
        # Obtaining the member 'buttond' of a type (line 296)
        buttond_250375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 17), self_250374, 'buttond')
        # Obtaining the member 'get' of a type (line 296)
        get_250376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 17), buttond_250375, 'get')
        # Calling get(args, kwargs) (line 296)
        get_call_result_250382 = invoke(stypy.reporting.localization.Localization(__file__, 296, 17), get_250376, *[button_call_result_250380], **kwargs_250381)
        
        # Assigning a type to the variable 'button' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'button', get_call_result_250382)
        
        # Type idiom detected: calculating its left and rigth part (line 297)
        # Getting the type of 'button' (line 297)
        button_250383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'button')
        # Getting the type of 'None' (line 297)
        None_250384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 25), 'None')
        
        (may_be_250385, more_types_in_union_250386) = may_not_be_none(button_250383, None_250384)

        if may_be_250385:

            if more_types_in_union_250386:
                # Runtime conditional SSA (line 297)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to button_press_event(...): (line 298)
            # Processing the call arguments (line 298)
            # Getting the type of 'self' (line 298)
            self_250389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 48), 'self', False)
            # Getting the type of 'x' (line 298)
            x_250390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 54), 'x', False)
            # Getting the type of 'y' (line 298)
            y_250391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 57), 'y', False)
            # Getting the type of 'button' (line 298)
            button_250392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 60), 'button', False)
            # Processing the call keyword arguments (line 298)
            # Getting the type of 'event' (line 299)
            event_250393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 57), 'event', False)
            keyword_250394 = event_250393
            kwargs_250395 = {'guiEvent': keyword_250394}
            # Getting the type of 'FigureCanvasBase' (line 298)
            FigureCanvasBase_250387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'FigureCanvasBase', False)
            # Obtaining the member 'button_press_event' of a type (line 298)
            button_press_event_250388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 12), FigureCanvasBase_250387, 'button_press_event')
            # Calling button_press_event(args, kwargs) (line 298)
            button_press_event_call_result_250396 = invoke(stypy.reporting.localization.Localization(__file__, 298, 12), button_press_event_250388, *[self_250389, x_250390, y_250391, button_250392], **kwargs_250395)
            

            if more_types_in_union_250386:
                # SSA join for if statement (line 297)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'mousePressEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mousePressEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 294)
        stypy_return_type_250397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250397)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mousePressEvent'
        return stypy_return_type_250397


    @norecursion
    def mouseDoubleClickEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mouseDoubleClickEvent'
        module_type_store = module_type_store.open_function_context('mouseDoubleClickEvent', 301, 4, False)
        # Assigning a type to the variable 'self' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.mouseDoubleClickEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.mouseDoubleClickEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.mouseDoubleClickEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.mouseDoubleClickEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.mouseDoubleClickEvent')
        FigureCanvasQT.mouseDoubleClickEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.mouseDoubleClickEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.mouseDoubleClickEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.mouseDoubleClickEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.mouseDoubleClickEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.mouseDoubleClickEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.mouseDoubleClickEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.mouseDoubleClickEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mouseDoubleClickEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mouseDoubleClickEvent(...)' code ##################

        
        # Assigning a Call to a Tuple (line 302):
        
        # Assigning a Call to a Name:
        
        # Call to mouseEventCoords(...): (line 302)
        # Processing the call arguments (line 302)
        
        # Call to pos(...): (line 302)
        # Processing the call keyword arguments (line 302)
        kwargs_250402 = {}
        # Getting the type of 'event' (line 302)
        event_250400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 37), 'event', False)
        # Obtaining the member 'pos' of a type (line 302)
        pos_250401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 37), event_250400, 'pos')
        # Calling pos(args, kwargs) (line 302)
        pos_call_result_250403 = invoke(stypy.reporting.localization.Localization(__file__, 302, 37), pos_250401, *[], **kwargs_250402)
        
        # Processing the call keyword arguments (line 302)
        kwargs_250404 = {}
        # Getting the type of 'self' (line 302)
        self_250398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'self', False)
        # Obtaining the member 'mouseEventCoords' of a type (line 302)
        mouseEventCoords_250399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 15), self_250398, 'mouseEventCoords')
        # Calling mouseEventCoords(args, kwargs) (line 302)
        mouseEventCoords_call_result_250405 = invoke(stypy.reporting.localization.Localization(__file__, 302, 15), mouseEventCoords_250399, *[pos_call_result_250403], **kwargs_250404)
        
        # Assigning a type to the variable 'call_assignment_249705' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'call_assignment_249705', mouseEventCoords_call_result_250405)
        
        # Assigning a Call to a Name (line 302):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250409 = {}
        # Getting the type of 'call_assignment_249705' (line 302)
        call_assignment_249705_250406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'call_assignment_249705', False)
        # Obtaining the member '__getitem__' of a type (line 302)
        getitem___250407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), call_assignment_249705_250406, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250410 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250407, *[int_250408], **kwargs_250409)
        
        # Assigning a type to the variable 'call_assignment_249706' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'call_assignment_249706', getitem___call_result_250410)
        
        # Assigning a Name to a Name (line 302):
        # Getting the type of 'call_assignment_249706' (line 302)
        call_assignment_249706_250411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'call_assignment_249706')
        # Assigning a type to the variable 'x' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'x', call_assignment_249706_250411)
        
        # Assigning a Call to a Name (line 302):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250415 = {}
        # Getting the type of 'call_assignment_249705' (line 302)
        call_assignment_249705_250412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'call_assignment_249705', False)
        # Obtaining the member '__getitem__' of a type (line 302)
        getitem___250413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), call_assignment_249705_250412, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250416 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250413, *[int_250414], **kwargs_250415)
        
        # Assigning a type to the variable 'call_assignment_249707' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'call_assignment_249707', getitem___call_result_250416)
        
        # Assigning a Name to a Name (line 302):
        # Getting the type of 'call_assignment_249707' (line 302)
        call_assignment_249707_250417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'call_assignment_249707')
        # Assigning a type to the variable 'y' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 11), 'y', call_assignment_249707_250417)
        
        # Assigning a Call to a Name (line 303):
        
        # Assigning a Call to a Name (line 303):
        
        # Call to get(...): (line 303)
        # Processing the call arguments (line 303)
        
        # Call to button(...): (line 303)
        # Processing the call keyword arguments (line 303)
        kwargs_250423 = {}
        # Getting the type of 'event' (line 303)
        event_250421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 34), 'event', False)
        # Obtaining the member 'button' of a type (line 303)
        button_250422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 34), event_250421, 'button')
        # Calling button(args, kwargs) (line 303)
        button_call_result_250424 = invoke(stypy.reporting.localization.Localization(__file__, 303, 34), button_250422, *[], **kwargs_250423)
        
        # Processing the call keyword arguments (line 303)
        kwargs_250425 = {}
        # Getting the type of 'self' (line 303)
        self_250418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 17), 'self', False)
        # Obtaining the member 'buttond' of a type (line 303)
        buttond_250419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 17), self_250418, 'buttond')
        # Obtaining the member 'get' of a type (line 303)
        get_250420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 17), buttond_250419, 'get')
        # Calling get(args, kwargs) (line 303)
        get_call_result_250426 = invoke(stypy.reporting.localization.Localization(__file__, 303, 17), get_250420, *[button_call_result_250424], **kwargs_250425)
        
        # Assigning a type to the variable 'button' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'button', get_call_result_250426)
        
        # Type idiom detected: calculating its left and rigth part (line 304)
        # Getting the type of 'button' (line 304)
        button_250427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'button')
        # Getting the type of 'None' (line 304)
        None_250428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'None')
        
        (may_be_250429, more_types_in_union_250430) = may_not_be_none(button_250427, None_250428)

        if may_be_250429:

            if more_types_in_union_250430:
                # Runtime conditional SSA (line 304)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to button_press_event(...): (line 305)
            # Processing the call arguments (line 305)
            # Getting the type of 'self' (line 305)
            self_250433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 48), 'self', False)
            # Getting the type of 'x' (line 305)
            x_250434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 54), 'x', False)
            # Getting the type of 'y' (line 305)
            y_250435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 57), 'y', False)
            # Getting the type of 'button' (line 306)
            button_250436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 48), 'button', False)
            # Processing the call keyword arguments (line 305)
            # Getting the type of 'True' (line 306)
            True_250437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 65), 'True', False)
            keyword_250438 = True_250437
            # Getting the type of 'event' (line 307)
            event_250439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 57), 'event', False)
            keyword_250440 = event_250439
            kwargs_250441 = {'guiEvent': keyword_250440, 'dblclick': keyword_250438}
            # Getting the type of 'FigureCanvasBase' (line 305)
            FigureCanvasBase_250431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'FigureCanvasBase', False)
            # Obtaining the member 'button_press_event' of a type (line 305)
            button_press_event_250432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 12), FigureCanvasBase_250431, 'button_press_event')
            # Calling button_press_event(args, kwargs) (line 305)
            button_press_event_call_result_250442 = invoke(stypy.reporting.localization.Localization(__file__, 305, 12), button_press_event_250432, *[self_250433, x_250434, y_250435, button_250436], **kwargs_250441)
            

            if more_types_in_union_250430:
                # SSA join for if statement (line 304)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'mouseDoubleClickEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mouseDoubleClickEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 301)
        stypy_return_type_250443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250443)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mouseDoubleClickEvent'
        return stypy_return_type_250443


    @norecursion
    def mouseMoveEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mouseMoveEvent'
        module_type_store = module_type_store.open_function_context('mouseMoveEvent', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.mouseMoveEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.mouseMoveEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.mouseMoveEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.mouseMoveEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.mouseMoveEvent')
        FigureCanvasQT.mouseMoveEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.mouseMoveEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.mouseMoveEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.mouseMoveEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.mouseMoveEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.mouseMoveEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.mouseMoveEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.mouseMoveEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mouseMoveEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mouseMoveEvent(...)' code ##################

        
        # Assigning a Call to a Tuple (line 310):
        
        # Assigning a Call to a Name:
        
        # Call to mouseEventCoords(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'event' (line 310)
        event_250446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 37), 'event', False)
        # Processing the call keyword arguments (line 310)
        kwargs_250447 = {}
        # Getting the type of 'self' (line 310)
        self_250444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'self', False)
        # Obtaining the member 'mouseEventCoords' of a type (line 310)
        mouseEventCoords_250445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 15), self_250444, 'mouseEventCoords')
        # Calling mouseEventCoords(args, kwargs) (line 310)
        mouseEventCoords_call_result_250448 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), mouseEventCoords_250445, *[event_250446], **kwargs_250447)
        
        # Assigning a type to the variable 'call_assignment_249708' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'call_assignment_249708', mouseEventCoords_call_result_250448)
        
        # Assigning a Call to a Name (line 310):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250452 = {}
        # Getting the type of 'call_assignment_249708' (line 310)
        call_assignment_249708_250449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'call_assignment_249708', False)
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___250450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), call_assignment_249708_250449, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250453 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250450, *[int_250451], **kwargs_250452)
        
        # Assigning a type to the variable 'call_assignment_249709' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'call_assignment_249709', getitem___call_result_250453)
        
        # Assigning a Name to a Name (line 310):
        # Getting the type of 'call_assignment_249709' (line 310)
        call_assignment_249709_250454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'call_assignment_249709')
        # Assigning a type to the variable 'x' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'x', call_assignment_249709_250454)
        
        # Assigning a Call to a Name (line 310):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250458 = {}
        # Getting the type of 'call_assignment_249708' (line 310)
        call_assignment_249708_250455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'call_assignment_249708', False)
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___250456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), call_assignment_249708_250455, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250459 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250456, *[int_250457], **kwargs_250458)
        
        # Assigning a type to the variable 'call_assignment_249710' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'call_assignment_249710', getitem___call_result_250459)
        
        # Assigning a Name to a Name (line 310):
        # Getting the type of 'call_assignment_249710' (line 310)
        call_assignment_249710_250460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'call_assignment_249710')
        # Assigning a type to the variable 'y' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'y', call_assignment_249710_250460)
        
        # Call to motion_notify_event(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'self' (line 311)
        self_250463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 45), 'self', False)
        # Getting the type of 'x' (line 311)
        x_250464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 51), 'x', False)
        # Getting the type of 'y' (line 311)
        y_250465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 54), 'y', False)
        # Processing the call keyword arguments (line 311)
        # Getting the type of 'event' (line 311)
        event_250466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 66), 'event', False)
        keyword_250467 = event_250466
        kwargs_250468 = {'guiEvent': keyword_250467}
        # Getting the type of 'FigureCanvasBase' (line 311)
        FigureCanvasBase_250461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'motion_notify_event' of a type (line 311)
        motion_notify_event_250462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), FigureCanvasBase_250461, 'motion_notify_event')
        # Calling motion_notify_event(args, kwargs) (line 311)
        motion_notify_event_call_result_250469 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), motion_notify_event_250462, *[self_250463, x_250464, y_250465], **kwargs_250468)
        
        
        # ################# End of 'mouseMoveEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mouseMoveEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_250470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250470)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mouseMoveEvent'
        return stypy_return_type_250470


    @norecursion
    def mouseReleaseEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mouseReleaseEvent'
        module_type_store = module_type_store.open_function_context('mouseReleaseEvent', 313, 4, False)
        # Assigning a type to the variable 'self' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.mouseReleaseEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.mouseReleaseEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.mouseReleaseEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.mouseReleaseEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.mouseReleaseEvent')
        FigureCanvasQT.mouseReleaseEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.mouseReleaseEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.mouseReleaseEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.mouseReleaseEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.mouseReleaseEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.mouseReleaseEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.mouseReleaseEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.mouseReleaseEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mouseReleaseEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mouseReleaseEvent(...)' code ##################

        
        # Assigning a Call to a Tuple (line 314):
        
        # Assigning a Call to a Name:
        
        # Call to mouseEventCoords(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'event' (line 314)
        event_250473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 37), 'event', False)
        # Processing the call keyword arguments (line 314)
        kwargs_250474 = {}
        # Getting the type of 'self' (line 314)
        self_250471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), 'self', False)
        # Obtaining the member 'mouseEventCoords' of a type (line 314)
        mouseEventCoords_250472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 15), self_250471, 'mouseEventCoords')
        # Calling mouseEventCoords(args, kwargs) (line 314)
        mouseEventCoords_call_result_250475 = invoke(stypy.reporting.localization.Localization(__file__, 314, 15), mouseEventCoords_250472, *[event_250473], **kwargs_250474)
        
        # Assigning a type to the variable 'call_assignment_249711' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'call_assignment_249711', mouseEventCoords_call_result_250475)
        
        # Assigning a Call to a Name (line 314):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250479 = {}
        # Getting the type of 'call_assignment_249711' (line 314)
        call_assignment_249711_250476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'call_assignment_249711', False)
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___250477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), call_assignment_249711_250476, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250480 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250477, *[int_250478], **kwargs_250479)
        
        # Assigning a type to the variable 'call_assignment_249712' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'call_assignment_249712', getitem___call_result_250480)
        
        # Assigning a Name to a Name (line 314):
        # Getting the type of 'call_assignment_249712' (line 314)
        call_assignment_249712_250481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'call_assignment_249712')
        # Assigning a type to the variable 'x' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'x', call_assignment_249712_250481)
        
        # Assigning a Call to a Name (line 314):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250485 = {}
        # Getting the type of 'call_assignment_249711' (line 314)
        call_assignment_249711_250482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'call_assignment_249711', False)
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___250483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), call_assignment_249711_250482, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250486 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250483, *[int_250484], **kwargs_250485)
        
        # Assigning a type to the variable 'call_assignment_249713' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'call_assignment_249713', getitem___call_result_250486)
        
        # Assigning a Name to a Name (line 314):
        # Getting the type of 'call_assignment_249713' (line 314)
        call_assignment_249713_250487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'call_assignment_249713')
        # Assigning a type to the variable 'y' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'y', call_assignment_249713_250487)
        
        # Assigning a Call to a Name (line 315):
        
        # Assigning a Call to a Name (line 315):
        
        # Call to get(...): (line 315)
        # Processing the call arguments (line 315)
        
        # Call to button(...): (line 315)
        # Processing the call keyword arguments (line 315)
        kwargs_250493 = {}
        # Getting the type of 'event' (line 315)
        event_250491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 34), 'event', False)
        # Obtaining the member 'button' of a type (line 315)
        button_250492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 34), event_250491, 'button')
        # Calling button(args, kwargs) (line 315)
        button_call_result_250494 = invoke(stypy.reporting.localization.Localization(__file__, 315, 34), button_250492, *[], **kwargs_250493)
        
        # Processing the call keyword arguments (line 315)
        kwargs_250495 = {}
        # Getting the type of 'self' (line 315)
        self_250488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 17), 'self', False)
        # Obtaining the member 'buttond' of a type (line 315)
        buttond_250489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 17), self_250488, 'buttond')
        # Obtaining the member 'get' of a type (line 315)
        get_250490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 17), buttond_250489, 'get')
        # Calling get(args, kwargs) (line 315)
        get_call_result_250496 = invoke(stypy.reporting.localization.Localization(__file__, 315, 17), get_250490, *[button_call_result_250494], **kwargs_250495)
        
        # Assigning a type to the variable 'button' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'button', get_call_result_250496)
        
        # Type idiom detected: calculating its left and rigth part (line 316)
        # Getting the type of 'button' (line 316)
        button_250497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'button')
        # Getting the type of 'None' (line 316)
        None_250498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 25), 'None')
        
        (may_be_250499, more_types_in_union_250500) = may_not_be_none(button_250497, None_250498)

        if may_be_250499:

            if more_types_in_union_250500:
                # Runtime conditional SSA (line 316)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to button_release_event(...): (line 317)
            # Processing the call arguments (line 317)
            # Getting the type of 'self' (line 317)
            self_250503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 50), 'self', False)
            # Getting the type of 'x' (line 317)
            x_250504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 56), 'x', False)
            # Getting the type of 'y' (line 317)
            y_250505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 59), 'y', False)
            # Getting the type of 'button' (line 317)
            button_250506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 62), 'button', False)
            # Processing the call keyword arguments (line 317)
            # Getting the type of 'event' (line 318)
            event_250507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 59), 'event', False)
            keyword_250508 = event_250507
            kwargs_250509 = {'guiEvent': keyword_250508}
            # Getting the type of 'FigureCanvasBase' (line 317)
            FigureCanvasBase_250501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'FigureCanvasBase', False)
            # Obtaining the member 'button_release_event' of a type (line 317)
            button_release_event_250502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), FigureCanvasBase_250501, 'button_release_event')
            # Calling button_release_event(args, kwargs) (line 317)
            button_release_event_call_result_250510 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), button_release_event_250502, *[self_250503, x_250504, y_250505, button_250506], **kwargs_250509)
            

            if more_types_in_union_250500:
                # SSA join for if statement (line 316)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'mouseReleaseEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mouseReleaseEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 313)
        stypy_return_type_250511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250511)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mouseReleaseEvent'
        return stypy_return_type_250511


    @norecursion
    def wheelEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wheelEvent'
        module_type_store = module_type_store.open_function_context('wheelEvent', 320, 4, False)
        # Assigning a type to the variable 'self' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.wheelEvent')
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.wheelEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.wheelEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wheelEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wheelEvent(...)' code ##################

        
        # Assigning a Call to a Tuple (line 321):
        
        # Assigning a Call to a Name:
        
        # Call to mouseEventCoords(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'event' (line 321)
        event_250514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 37), 'event', False)
        # Processing the call keyword arguments (line 321)
        kwargs_250515 = {}
        # Getting the type of 'self' (line 321)
        self_250512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 15), 'self', False)
        # Obtaining the member 'mouseEventCoords' of a type (line 321)
        mouseEventCoords_250513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 15), self_250512, 'mouseEventCoords')
        # Calling mouseEventCoords(args, kwargs) (line 321)
        mouseEventCoords_call_result_250516 = invoke(stypy.reporting.localization.Localization(__file__, 321, 15), mouseEventCoords_250513, *[event_250514], **kwargs_250515)
        
        # Assigning a type to the variable 'call_assignment_249714' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'call_assignment_249714', mouseEventCoords_call_result_250516)
        
        # Assigning a Call to a Name (line 321):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250520 = {}
        # Getting the type of 'call_assignment_249714' (line 321)
        call_assignment_249714_250517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'call_assignment_249714', False)
        # Obtaining the member '__getitem__' of a type (line 321)
        getitem___250518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), call_assignment_249714_250517, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250521 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250518, *[int_250519], **kwargs_250520)
        
        # Assigning a type to the variable 'call_assignment_249715' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'call_assignment_249715', getitem___call_result_250521)
        
        # Assigning a Name to a Name (line 321):
        # Getting the type of 'call_assignment_249715' (line 321)
        call_assignment_249715_250522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'call_assignment_249715')
        # Assigning a type to the variable 'x' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'x', call_assignment_249715_250522)
        
        # Assigning a Call to a Name (line 321):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250526 = {}
        # Getting the type of 'call_assignment_249714' (line 321)
        call_assignment_249714_250523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'call_assignment_249714', False)
        # Obtaining the member '__getitem__' of a type (line 321)
        getitem___250524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), call_assignment_249714_250523, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250527 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250524, *[int_250525], **kwargs_250526)
        
        # Assigning a type to the variable 'call_assignment_249716' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'call_assignment_249716', getitem___call_result_250527)
        
        # Assigning a Name to a Name (line 321):
        # Getting the type of 'call_assignment_249716' (line 321)
        call_assignment_249716_250528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'call_assignment_249716')
        # Assigning a type to the variable 'y' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 11), 'y', call_assignment_249716_250528)
        
        
        # Evaluating a boolean operation
        
        
        # Call to x(...): (line 323)
        # Processing the call keyword arguments (line 323)
        kwargs_250534 = {}
        
        # Call to pixelDelta(...): (line 323)
        # Processing the call keyword arguments (line 323)
        kwargs_250531 = {}
        # Getting the type of 'event' (line 323)
        event_250529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 11), 'event', False)
        # Obtaining the member 'pixelDelta' of a type (line 323)
        pixelDelta_250530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 11), event_250529, 'pixelDelta')
        # Calling pixelDelta(args, kwargs) (line 323)
        pixelDelta_call_result_250532 = invoke(stypy.reporting.localization.Localization(__file__, 323, 11), pixelDelta_250530, *[], **kwargs_250531)
        
        # Obtaining the member 'x' of a type (line 323)
        x_250533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 11), pixelDelta_call_result_250532, 'x')
        # Calling x(args, kwargs) (line 323)
        x_call_result_250535 = invoke(stypy.reporting.localization.Localization(__file__, 323, 11), x_250533, *[], **kwargs_250534)
        
        int_250536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 37), 'int')
        # Applying the binary operator '==' (line 323)
        result_eq_250537 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 11), '==', x_call_result_250535, int_250536)
        
        
        
        # Call to y(...): (line 323)
        # Processing the call keyword arguments (line 323)
        kwargs_250543 = {}
        
        # Call to pixelDelta(...): (line 323)
        # Processing the call keyword arguments (line 323)
        kwargs_250540 = {}
        # Getting the type of 'event' (line 323)
        event_250538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 43), 'event', False)
        # Obtaining the member 'pixelDelta' of a type (line 323)
        pixelDelta_250539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 43), event_250538, 'pixelDelta')
        # Calling pixelDelta(args, kwargs) (line 323)
        pixelDelta_call_result_250541 = invoke(stypy.reporting.localization.Localization(__file__, 323, 43), pixelDelta_250539, *[], **kwargs_250540)
        
        # Obtaining the member 'y' of a type (line 323)
        y_250542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 43), pixelDelta_call_result_250541, 'y')
        # Calling y(args, kwargs) (line 323)
        y_call_result_250544 = invoke(stypy.reporting.localization.Localization(__file__, 323, 43), y_250542, *[], **kwargs_250543)
        
        int_250545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 69), 'int')
        # Applying the binary operator '==' (line 323)
        result_eq_250546 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 43), '==', y_call_result_250544, int_250545)
        
        # Applying the binary operator 'and' (line 323)
        result_and_keyword_250547 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 11), 'and', result_eq_250537, result_eq_250546)
        
        # Testing the type of an if condition (line 323)
        if_condition_250548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 8), result_and_keyword_250547)
        # Assigning a type to the variable 'if_condition_250548' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'if_condition_250548', if_condition_250548)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 324):
        
        # Assigning a BinOp to a Name (line 324):
        
        # Call to y(...): (line 324)
        # Processing the call keyword arguments (line 324)
        kwargs_250554 = {}
        
        # Call to angleDelta(...): (line 324)
        # Processing the call keyword arguments (line 324)
        kwargs_250551 = {}
        # Getting the type of 'event' (line 324)
        event_250549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 'event', False)
        # Obtaining the member 'angleDelta' of a type (line 324)
        angleDelta_250550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 20), event_250549, 'angleDelta')
        # Calling angleDelta(args, kwargs) (line 324)
        angleDelta_call_result_250552 = invoke(stypy.reporting.localization.Localization(__file__, 324, 20), angleDelta_250550, *[], **kwargs_250551)
        
        # Obtaining the member 'y' of a type (line 324)
        y_250553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 20), angleDelta_call_result_250552, 'y')
        # Calling y(args, kwargs) (line 324)
        y_call_result_250555 = invoke(stypy.reporting.localization.Localization(__file__, 324, 20), y_250553, *[], **kwargs_250554)
        
        int_250556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 45), 'int')
        # Applying the binary operator 'div' (line 324)
        result_div_250557 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 20), 'div', y_call_result_250555, int_250556)
        
        # Assigning a type to the variable 'steps' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'steps', result_div_250557)
        # SSA branch for the else part of an if statement (line 323)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 326):
        
        # Assigning a Call to a Name (line 326):
        
        # Call to y(...): (line 326)
        # Processing the call keyword arguments (line 326)
        kwargs_250563 = {}
        
        # Call to pixelDelta(...): (line 326)
        # Processing the call keyword arguments (line 326)
        kwargs_250560 = {}
        # Getting the type of 'event' (line 326)
        event_250558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 20), 'event', False)
        # Obtaining the member 'pixelDelta' of a type (line 326)
        pixelDelta_250559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 20), event_250558, 'pixelDelta')
        # Calling pixelDelta(args, kwargs) (line 326)
        pixelDelta_call_result_250561 = invoke(stypy.reporting.localization.Localization(__file__, 326, 20), pixelDelta_250559, *[], **kwargs_250560)
        
        # Obtaining the member 'y' of a type (line 326)
        y_250562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 20), pixelDelta_call_result_250561, 'y')
        # Calling y(args, kwargs) (line 326)
        y_call_result_250564 = invoke(stypy.reporting.localization.Localization(__file__, 326, 20), y_250562, *[], **kwargs_250563)
        
        # Assigning a type to the variable 'steps' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'steps', y_call_result_250564)
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'steps' (line 327)
        steps_250565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 11), 'steps')
        # Testing the type of an if condition (line 327)
        if_condition_250566 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 8), steps_250565)
        # Assigning a type to the variable 'if_condition_250566' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'if_condition_250566', if_condition_250566)
        # SSA begins for if statement (line 327)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to scroll_event(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'self' (line 328)
        self_250569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 42), 'self', False)
        # Getting the type of 'x' (line 328)
        x_250570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 48), 'x', False)
        # Getting the type of 'y' (line 328)
        y_250571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 51), 'y', False)
        # Getting the type of 'steps' (line 328)
        steps_250572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 54), 'steps', False)
        # Processing the call keyword arguments (line 328)
        # Getting the type of 'event' (line 328)
        event_250573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 70), 'event', False)
        keyword_250574 = event_250573
        kwargs_250575 = {'guiEvent': keyword_250574}
        # Getting the type of 'FigureCanvasBase' (line 328)
        FigureCanvasBase_250567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'FigureCanvasBase', False)
        # Obtaining the member 'scroll_event' of a type (line 328)
        scroll_event_250568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 12), FigureCanvasBase_250567, 'scroll_event')
        # Calling scroll_event(args, kwargs) (line 328)
        scroll_event_call_result_250576 = invoke(stypy.reporting.localization.Localization(__file__, 328, 12), scroll_event_250568, *[self_250569, x_250570, y_250571, steps_250572], **kwargs_250575)
        
        # SSA join for if statement (line 327)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'wheelEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wheelEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 320)
        stypy_return_type_250577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250577)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wheelEvent'
        return stypy_return_type_250577


    @norecursion
    def keyPressEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'keyPressEvent'
        module_type_store = module_type_store.open_function_context('keyPressEvent', 330, 4, False)
        # Assigning a type to the variable 'self' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.keyPressEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.keyPressEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.keyPressEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.keyPressEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.keyPressEvent')
        FigureCanvasQT.keyPressEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.keyPressEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.keyPressEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.keyPressEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.keyPressEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.keyPressEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.keyPressEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.keyPressEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'keyPressEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'keyPressEvent(...)' code ##################

        
        # Assigning a Call to a Name (line 331):
        
        # Assigning a Call to a Name (line 331):
        
        # Call to _get_key(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'event' (line 331)
        event_250580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 28), 'event', False)
        # Processing the call keyword arguments (line 331)
        kwargs_250581 = {}
        # Getting the type of 'self' (line 331)
        self_250578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 14), 'self', False)
        # Obtaining the member '_get_key' of a type (line 331)
        _get_key_250579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 14), self_250578, '_get_key')
        # Calling _get_key(args, kwargs) (line 331)
        _get_key_call_result_250582 = invoke(stypy.reporting.localization.Localization(__file__, 331, 14), _get_key_250579, *[event_250580], **kwargs_250581)
        
        # Assigning a type to the variable 'key' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'key', _get_key_call_result_250582)
        
        # Type idiom detected: calculating its left and rigth part (line 332)
        # Getting the type of 'key' (line 332)
        key_250583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'key')
        # Getting the type of 'None' (line 332)
        None_250584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 22), 'None')
        
        (may_be_250585, more_types_in_union_250586) = may_not_be_none(key_250583, None_250584)

        if may_be_250585:

            if more_types_in_union_250586:
                # Runtime conditional SSA (line 332)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to key_press_event(...): (line 333)
            # Processing the call arguments (line 333)
            # Getting the type of 'self' (line 333)
            self_250589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 45), 'self', False)
            # Getting the type of 'key' (line 333)
            key_250590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 51), 'key', False)
            # Processing the call keyword arguments (line 333)
            # Getting the type of 'event' (line 333)
            event_250591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 65), 'event', False)
            keyword_250592 = event_250591
            kwargs_250593 = {'guiEvent': keyword_250592}
            # Getting the type of 'FigureCanvasBase' (line 333)
            FigureCanvasBase_250587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'FigureCanvasBase', False)
            # Obtaining the member 'key_press_event' of a type (line 333)
            key_press_event_250588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), FigureCanvasBase_250587, 'key_press_event')
            # Calling key_press_event(args, kwargs) (line 333)
            key_press_event_call_result_250594 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), key_press_event_250588, *[self_250589, key_250590], **kwargs_250593)
            

            if more_types_in_union_250586:
                # SSA join for if statement (line 332)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'keyPressEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'keyPressEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 330)
        stypy_return_type_250595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250595)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'keyPressEvent'
        return stypy_return_type_250595


    @norecursion
    def keyReleaseEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'keyReleaseEvent'
        module_type_store = module_type_store.open_function_context('keyReleaseEvent', 335, 4, False)
        # Assigning a type to the variable 'self' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.keyReleaseEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.keyReleaseEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.keyReleaseEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.keyReleaseEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.keyReleaseEvent')
        FigureCanvasQT.keyReleaseEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.keyReleaseEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.keyReleaseEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.keyReleaseEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.keyReleaseEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.keyReleaseEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.keyReleaseEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.keyReleaseEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'keyReleaseEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'keyReleaseEvent(...)' code ##################

        
        # Assigning a Call to a Name (line 336):
        
        # Assigning a Call to a Name (line 336):
        
        # Call to _get_key(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'event' (line 336)
        event_250598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 28), 'event', False)
        # Processing the call keyword arguments (line 336)
        kwargs_250599 = {}
        # Getting the type of 'self' (line 336)
        self_250596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 14), 'self', False)
        # Obtaining the member '_get_key' of a type (line 336)
        _get_key_250597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 14), self_250596, '_get_key')
        # Calling _get_key(args, kwargs) (line 336)
        _get_key_call_result_250600 = invoke(stypy.reporting.localization.Localization(__file__, 336, 14), _get_key_250597, *[event_250598], **kwargs_250599)
        
        # Assigning a type to the variable 'key' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'key', _get_key_call_result_250600)
        
        # Type idiom detected: calculating its left and rigth part (line 337)
        # Getting the type of 'key' (line 337)
        key_250601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'key')
        # Getting the type of 'None' (line 337)
        None_250602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 22), 'None')
        
        (may_be_250603, more_types_in_union_250604) = may_not_be_none(key_250601, None_250602)

        if may_be_250603:

            if more_types_in_union_250604:
                # Runtime conditional SSA (line 337)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to key_release_event(...): (line 338)
            # Processing the call arguments (line 338)
            # Getting the type of 'self' (line 338)
            self_250607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 47), 'self', False)
            # Getting the type of 'key' (line 338)
            key_250608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 53), 'key', False)
            # Processing the call keyword arguments (line 338)
            # Getting the type of 'event' (line 338)
            event_250609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 67), 'event', False)
            keyword_250610 = event_250609
            kwargs_250611 = {'guiEvent': keyword_250610}
            # Getting the type of 'FigureCanvasBase' (line 338)
            FigureCanvasBase_250605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'FigureCanvasBase', False)
            # Obtaining the member 'key_release_event' of a type (line 338)
            key_release_event_250606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), FigureCanvasBase_250605, 'key_release_event')
            # Calling key_release_event(args, kwargs) (line 338)
            key_release_event_call_result_250612 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), key_release_event_250606, *[self_250607, key_250608], **kwargs_250611)
            

            if more_types_in_union_250604:
                # SSA join for if statement (line 337)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'keyReleaseEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'keyReleaseEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 335)
        stypy_return_type_250613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250613)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'keyReleaseEvent'
        return stypy_return_type_250613


    @norecursion
    def keyAutoRepeat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'keyAutoRepeat'
        module_type_store = module_type_store.open_function_context('keyAutoRepeat', 340, 4, False)
        # Assigning a type to the variable 'self' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.keyAutoRepeat')
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.keyAutoRepeat', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'keyAutoRepeat', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'keyAutoRepeat(...)' code ##################

        unicode_250614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, (-1)), 'unicode', u'\n        If True, enable auto-repeat for key events.\n        ')
        # Getting the type of 'self' (line 345)
        self_250615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 15), 'self')
        # Obtaining the member '_keyautorepeat' of a type (line 345)
        _keyautorepeat_250616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 15), self_250615, '_keyautorepeat')
        # Assigning a type to the variable 'stypy_return_type' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'stypy_return_type', _keyautorepeat_250616)
        
        # ################# End of 'keyAutoRepeat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'keyAutoRepeat' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_250617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250617)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'keyAutoRepeat'
        return stypy_return_type_250617


    @norecursion
    def keyAutoRepeat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'keyAutoRepeat'
        module_type_store = module_type_store.open_function_context('keyAutoRepeat', 347, 4, False)
        # Assigning a type to the variable 'self' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.keyAutoRepeat')
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_param_names_list', ['val'])
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.keyAutoRepeat.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.keyAutoRepeat', ['val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'keyAutoRepeat', localization, ['val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'keyAutoRepeat(...)' code ##################

        
        # Assigning a Call to a Attribute (line 349):
        
        # Assigning a Call to a Attribute (line 349):
        
        # Call to bool(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'val' (line 349)
        val_250619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 35), 'val', False)
        # Processing the call keyword arguments (line 349)
        kwargs_250620 = {}
        # Getting the type of 'bool' (line 349)
        bool_250618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 30), 'bool', False)
        # Calling bool(args, kwargs) (line 349)
        bool_call_result_250621 = invoke(stypy.reporting.localization.Localization(__file__, 349, 30), bool_250618, *[val_250619], **kwargs_250620)
        
        # Getting the type of 'self' (line 349)
        self_250622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'self')
        # Setting the type of the member '_keyautorepeat' of a type (line 349)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), self_250622, '_keyautorepeat', bool_call_result_250621)
        
        # ################# End of 'keyAutoRepeat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'keyAutoRepeat' in the type store
        # Getting the type of 'stypy_return_type' (line 347)
        stypy_return_type_250623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250623)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'keyAutoRepeat'
        return stypy_return_type_250623


    @norecursion
    def resizeEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'resizeEvent'
        module_type_store = module_type_store.open_function_context('resizeEvent', 351, 4, False)
        # Assigning a type to the variable 'self' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.resizeEvent.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.resizeEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.resizeEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.resizeEvent.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.resizeEvent')
        FigureCanvasQT.resizeEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.resizeEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.resizeEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.resizeEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.resizeEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.resizeEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.resizeEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.resizeEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'resizeEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'resizeEvent(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 354)
        # Getting the type of 'self' (line 354)
        self_250624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'self')
        # Obtaining the member '_dpi_ratio_prev' of a type (line 354)
        _dpi_ratio_prev_250625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 11), self_250624, '_dpi_ratio_prev')
        # Getting the type of 'None' (line 354)
        None_250626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 35), 'None')
        
        (may_be_250627, more_types_in_union_250628) = may_be_none(_dpi_ratio_prev_250625, None_250626)

        if may_be_250627:

            if more_types_in_union_250628:
                # Runtime conditional SSA (line 354)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 355)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_250628:
                # SSA join for if statement (line 354)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 356):
        
        # Assigning a BinOp to a Name (line 356):
        
        # Call to width(...): (line 356)
        # Processing the call keyword arguments (line 356)
        kwargs_250634 = {}
        
        # Call to size(...): (line 356)
        # Processing the call keyword arguments (line 356)
        kwargs_250631 = {}
        # Getting the type of 'event' (line 356)
        event_250629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'event', False)
        # Obtaining the member 'size' of a type (line 356)
        size_250630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 12), event_250629, 'size')
        # Calling size(args, kwargs) (line 356)
        size_call_result_250632 = invoke(stypy.reporting.localization.Localization(__file__, 356, 12), size_250630, *[], **kwargs_250631)
        
        # Obtaining the member 'width' of a type (line 356)
        width_250633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 12), size_call_result_250632, 'width')
        # Calling width(args, kwargs) (line 356)
        width_call_result_250635 = invoke(stypy.reporting.localization.Localization(__file__, 356, 12), width_250633, *[], **kwargs_250634)
        
        # Getting the type of 'self' (line 356)
        self_250636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 35), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 356)
        _dpi_ratio_250637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 35), self_250636, '_dpi_ratio')
        # Applying the binary operator '*' (line 356)
        result_mul_250638 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 12), '*', width_call_result_250635, _dpi_ratio_250637)
        
        # Assigning a type to the variable 'w' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'w', result_mul_250638)
        
        # Assigning a BinOp to a Name (line 357):
        
        # Assigning a BinOp to a Name (line 357):
        
        # Call to height(...): (line 357)
        # Processing the call keyword arguments (line 357)
        kwargs_250644 = {}
        
        # Call to size(...): (line 357)
        # Processing the call keyword arguments (line 357)
        kwargs_250641 = {}
        # Getting the type of 'event' (line 357)
        event_250639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'event', False)
        # Obtaining the member 'size' of a type (line 357)
        size_250640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 12), event_250639, 'size')
        # Calling size(args, kwargs) (line 357)
        size_call_result_250642 = invoke(stypy.reporting.localization.Localization(__file__, 357, 12), size_250640, *[], **kwargs_250641)
        
        # Obtaining the member 'height' of a type (line 357)
        height_250643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 12), size_call_result_250642, 'height')
        # Calling height(args, kwargs) (line 357)
        height_call_result_250645 = invoke(stypy.reporting.localization.Localization(__file__, 357, 12), height_250643, *[], **kwargs_250644)
        
        # Getting the type of 'self' (line 357)
        self_250646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 36), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 357)
        _dpi_ratio_250647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 36), self_250646, '_dpi_ratio')
        # Applying the binary operator '*' (line 357)
        result_mul_250648 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 12), '*', height_call_result_250645, _dpi_ratio_250647)
        
        # Assigning a type to the variable 'h' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'h', result_mul_250648)
        
        # Assigning a Attribute to a Name (line 358):
        
        # Assigning a Attribute to a Name (line 358):
        # Getting the type of 'self' (line 358)
        self_250649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 17), 'self')
        # Obtaining the member 'figure' of a type (line 358)
        figure_250650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 17), self_250649, 'figure')
        # Obtaining the member 'dpi' of a type (line 358)
        dpi_250651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 17), figure_250650, 'dpi')
        # Assigning a type to the variable 'dpival' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'dpival', dpi_250651)
        
        # Assigning a BinOp to a Name (line 359):
        
        # Assigning a BinOp to a Name (line 359):
        # Getting the type of 'w' (line 359)
        w_250652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'w')
        # Getting the type of 'dpival' (line 359)
        dpival_250653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 20), 'dpival')
        # Applying the binary operator 'div' (line 359)
        result_div_250654 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 16), 'div', w_250652, dpival_250653)
        
        # Assigning a type to the variable 'winch' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'winch', result_div_250654)
        
        # Assigning a BinOp to a Name (line 360):
        
        # Assigning a BinOp to a Name (line 360):
        # Getting the type of 'h' (line 360)
        h_250655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'h')
        # Getting the type of 'dpival' (line 360)
        dpival_250656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'dpival')
        # Applying the binary operator 'div' (line 360)
        result_div_250657 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 16), 'div', h_250655, dpival_250656)
        
        # Assigning a type to the variable 'hinch' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'hinch', result_div_250657)
        
        # Call to set_size_inches(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'winch' (line 361)
        winch_250661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 36), 'winch', False)
        # Getting the type of 'hinch' (line 361)
        hinch_250662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 43), 'hinch', False)
        # Processing the call keyword arguments (line 361)
        # Getting the type of 'False' (line 361)
        False_250663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 58), 'False', False)
        keyword_250664 = False_250663
        kwargs_250665 = {'forward': keyword_250664}
        # Getting the type of 'self' (line 361)
        self_250658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'self', False)
        # Obtaining the member 'figure' of a type (line 361)
        figure_250659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), self_250658, 'figure')
        # Obtaining the member 'set_size_inches' of a type (line 361)
        set_size_inches_250660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), figure_250659, 'set_size_inches')
        # Calling set_size_inches(args, kwargs) (line 361)
        set_size_inches_call_result_250666 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), set_size_inches_250660, *[winch_250661, hinch_250662], **kwargs_250665)
        
        
        # Call to resizeEvent(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'self' (line 363)
        self_250670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 38), 'self', False)
        # Getting the type of 'event' (line 363)
        event_250671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 44), 'event', False)
        # Processing the call keyword arguments (line 363)
        kwargs_250672 = {}
        # Getting the type of 'QtWidgets' (line 363)
        QtWidgets_250667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'QtWidgets', False)
        # Obtaining the member 'QWidget' of a type (line 363)
        QWidget_250668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), QtWidgets_250667, 'QWidget')
        # Obtaining the member 'resizeEvent' of a type (line 363)
        resizeEvent_250669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), QWidget_250668, 'resizeEvent')
        # Calling resizeEvent(args, kwargs) (line 363)
        resizeEvent_call_result_250673 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), resizeEvent_250669, *[self_250670, event_250671], **kwargs_250672)
        
        
        # Call to resize_event(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'self' (line 365)
        self_250676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 38), 'self', False)
        # Processing the call keyword arguments (line 365)
        kwargs_250677 = {}
        # Getting the type of 'FigureCanvasBase' (line 365)
        FigureCanvasBase_250674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'FigureCanvasBase', False)
        # Obtaining the member 'resize_event' of a type (line 365)
        resize_event_250675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 8), FigureCanvasBase_250674, 'resize_event')
        # Calling resize_event(args, kwargs) (line 365)
        resize_event_call_result_250678 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), resize_event_250675, *[self_250676], **kwargs_250677)
        
        
        # ################# End of 'resizeEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'resizeEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 351)
        stypy_return_type_250679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'resizeEvent'
        return stypy_return_type_250679


    @norecursion
    def sizeHint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sizeHint'
        module_type_store = module_type_store.open_function_context('sizeHint', 367, 4, False)
        # Assigning a type to the variable 'self' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.sizeHint.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.sizeHint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.sizeHint.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.sizeHint.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.sizeHint')
        FigureCanvasQT.sizeHint.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQT.sizeHint.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.sizeHint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.sizeHint.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.sizeHint.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.sizeHint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.sizeHint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.sizeHint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sizeHint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sizeHint(...)' code ##################

        
        # Assigning a Call to a Tuple (line 368):
        
        # Assigning a Call to a Name:
        
        # Call to get_width_height(...): (line 368)
        # Processing the call keyword arguments (line 368)
        kwargs_250682 = {}
        # Getting the type of 'self' (line 368)
        self_250680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 15), 'self', False)
        # Obtaining the member 'get_width_height' of a type (line 368)
        get_width_height_250681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 15), self_250680, 'get_width_height')
        # Calling get_width_height(args, kwargs) (line 368)
        get_width_height_call_result_250683 = invoke(stypy.reporting.localization.Localization(__file__, 368, 15), get_width_height_250681, *[], **kwargs_250682)
        
        # Assigning a type to the variable 'call_assignment_249717' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_249717', get_width_height_call_result_250683)
        
        # Assigning a Call to a Name (line 368):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250687 = {}
        # Getting the type of 'call_assignment_249717' (line 368)
        call_assignment_249717_250684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_249717', False)
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___250685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), call_assignment_249717_250684, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250688 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250685, *[int_250686], **kwargs_250687)
        
        # Assigning a type to the variable 'call_assignment_249718' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_249718', getitem___call_result_250688)
        
        # Assigning a Name to a Name (line 368):
        # Getting the type of 'call_assignment_249718' (line 368)
        call_assignment_249718_250689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_249718')
        # Assigning a type to the variable 'w' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'w', call_assignment_249718_250689)
        
        # Assigning a Call to a Name (line 368):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_250692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 8), 'int')
        # Processing the call keyword arguments
        kwargs_250693 = {}
        # Getting the type of 'call_assignment_249717' (line 368)
        call_assignment_249717_250690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_249717', False)
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___250691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), call_assignment_249717_250690, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_250694 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___250691, *[int_250692], **kwargs_250693)
        
        # Assigning a type to the variable 'call_assignment_249719' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_249719', getitem___call_result_250694)
        
        # Assigning a Name to a Name (line 368):
        # Getting the type of 'call_assignment_249719' (line 368)
        call_assignment_249719_250695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'call_assignment_249719')
        # Assigning a type to the variable 'h' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 11), 'h', call_assignment_249719_250695)
        
        # Call to QSize(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'w' (line 369)
        w_250698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 28), 'w', False)
        # Getting the type of 'h' (line 369)
        h_250699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 31), 'h', False)
        # Processing the call keyword arguments (line 369)
        kwargs_250700 = {}
        # Getting the type of 'QtCore' (line 369)
        QtCore_250696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 15), 'QtCore', False)
        # Obtaining the member 'QSize' of a type (line 369)
        QSize_250697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 15), QtCore_250696, 'QSize')
        # Calling QSize(args, kwargs) (line 369)
        QSize_call_result_250701 = invoke(stypy.reporting.localization.Localization(__file__, 369, 15), QSize_250697, *[w_250698, h_250699], **kwargs_250700)
        
        # Assigning a type to the variable 'stypy_return_type' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'stypy_return_type', QSize_call_result_250701)
        
        # ################# End of 'sizeHint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sizeHint' in the type store
        # Getting the type of 'stypy_return_type' (line 367)
        stypy_return_type_250702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250702)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sizeHint'
        return stypy_return_type_250702


    @norecursion
    def minumumSizeHint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'minumumSizeHint'
        module_type_store = module_type_store.open_function_context('minumumSizeHint', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.minumumSizeHint.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.minumumSizeHint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.minumumSizeHint.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.minumumSizeHint.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.minumumSizeHint')
        FigureCanvasQT.minumumSizeHint.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQT.minumumSizeHint.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.minumumSizeHint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.minumumSizeHint.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.minumumSizeHint.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.minumumSizeHint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.minumumSizeHint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.minumumSizeHint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'minumumSizeHint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'minumumSizeHint(...)' code ##################

        
        # Call to QSize(...): (line 372)
        # Processing the call arguments (line 372)
        int_250705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 28), 'int')
        int_250706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 32), 'int')
        # Processing the call keyword arguments (line 372)
        kwargs_250707 = {}
        # Getting the type of 'QtCore' (line 372)
        QtCore_250703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 15), 'QtCore', False)
        # Obtaining the member 'QSize' of a type (line 372)
        QSize_250704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 15), QtCore_250703, 'QSize')
        # Calling QSize(args, kwargs) (line 372)
        QSize_call_result_250708 = invoke(stypy.reporting.localization.Localization(__file__, 372, 15), QSize_250704, *[int_250705, int_250706], **kwargs_250707)
        
        # Assigning a type to the variable 'stypy_return_type' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'stypy_return_type', QSize_call_result_250708)
        
        # ################# End of 'minumumSizeHint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'minumumSizeHint' in the type store
        # Getting the type of 'stypy_return_type' (line 371)
        stypy_return_type_250709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250709)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'minumumSizeHint'
        return stypy_return_type_250709


    @norecursion
    def _get_key(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_key'
        module_type_store = module_type_store.open_function_context('_get_key', 374, 4, False)
        # Assigning a type to the variable 'self' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT._get_key.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT._get_key.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT._get_key.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT._get_key.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT._get_key')
        FigureCanvasQT._get_key.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT._get_key.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT._get_key.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT._get_key.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT._get_key.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT._get_key.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT._get_key.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT._get_key', ['event'], None, None, defaults, varargs, kwargs)

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

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 375)
        self_250710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 15), 'self')
        # Obtaining the member '_keyautorepeat' of a type (line 375)
        _keyautorepeat_250711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 15), self_250710, '_keyautorepeat')
        # Applying the 'not' unary operator (line 375)
        result_not__250712 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 11), 'not', _keyautorepeat_250711)
        
        
        # Call to isAutoRepeat(...): (line 375)
        # Processing the call keyword arguments (line 375)
        kwargs_250715 = {}
        # Getting the type of 'event' (line 375)
        event_250713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 39), 'event', False)
        # Obtaining the member 'isAutoRepeat' of a type (line 375)
        isAutoRepeat_250714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 39), event_250713, 'isAutoRepeat')
        # Calling isAutoRepeat(args, kwargs) (line 375)
        isAutoRepeat_call_result_250716 = invoke(stypy.reporting.localization.Localization(__file__, 375, 39), isAutoRepeat_250714, *[], **kwargs_250715)
        
        # Applying the binary operator 'and' (line 375)
        result_and_keyword_250717 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 11), 'and', result_not__250712, isAutoRepeat_call_result_250716)
        
        # Testing the type of an if condition (line 375)
        if_condition_250718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 8), result_and_keyword_250717)
        # Assigning a type to the variable 'if_condition_250718' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'if_condition_250718', if_condition_250718)
        # SSA begins for if statement (line 375)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 376)
        None_250719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'stypy_return_type', None_250719)
        # SSA join for if statement (line 375)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 378):
        
        # Assigning a Call to a Name (line 378):
        
        # Call to key(...): (line 378)
        # Processing the call keyword arguments (line 378)
        kwargs_250722 = {}
        # Getting the type of 'event' (line 378)
        event_250720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 20), 'event', False)
        # Obtaining the member 'key' of a type (line 378)
        key_250721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 20), event_250720, 'key')
        # Calling key(args, kwargs) (line 378)
        key_call_result_250723 = invoke(stypy.reporting.localization.Localization(__file__, 378, 20), key_250721, *[], **kwargs_250722)
        
        # Assigning a type to the variable 'event_key' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'event_key', key_call_result_250723)
        
        # Assigning a Call to a Name (line 379):
        
        # Assigning a Call to a Name (line 379):
        
        # Call to int(...): (line 379)
        # Processing the call arguments (line 379)
        
        # Call to modifiers(...): (line 379)
        # Processing the call keyword arguments (line 379)
        kwargs_250727 = {}
        # Getting the type of 'event' (line 379)
        event_250725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 25), 'event', False)
        # Obtaining the member 'modifiers' of a type (line 379)
        modifiers_250726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 25), event_250725, 'modifiers')
        # Calling modifiers(args, kwargs) (line 379)
        modifiers_call_result_250728 = invoke(stypy.reporting.localization.Localization(__file__, 379, 25), modifiers_250726, *[], **kwargs_250727)
        
        # Processing the call keyword arguments (line 379)
        kwargs_250729 = {}
        # Getting the type of 'int' (line 379)
        int_250724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 21), 'int', False)
        # Calling int(args, kwargs) (line 379)
        int_call_result_250730 = invoke(stypy.reporting.localization.Localization(__file__, 379, 21), int_250724, *[modifiers_call_result_250728], **kwargs_250729)
        
        # Assigning a type to the variable 'event_mods' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'event_mods', int_call_result_250730)
        
        # Assigning a ListComp to a Name (line 384):
        
        # Assigning a ListComp to a Name (line 384):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'MODIFIER_KEYS' (line 384)
        MODIFIER_KEYS_250741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 50), 'MODIFIER_KEYS')
        comprehension_250742 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 16), MODIFIER_KEYS_250741)
        # Assigning a type to the variable 'name' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 16), comprehension_250742))
        # Assigning a type to the variable 'mod_key' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'mod_key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 16), comprehension_250742))
        # Assigning a type to the variable 'qt_key' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'qt_key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 16), comprehension_250742))
        
        # Evaluating a boolean operation
        
        # Getting the type of 'event_key' (line 385)
        event_key_250732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 19), 'event_key')
        # Getting the type of 'qt_key' (line 385)
        qt_key_250733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 32), 'qt_key')
        # Applying the binary operator '!=' (line 385)
        result_ne_250734 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 19), '!=', event_key_250732, qt_key_250733)
        
        
        # Getting the type of 'event_mods' (line 385)
        event_mods_250735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 44), 'event_mods')
        # Getting the type of 'mod_key' (line 385)
        mod_key_250736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 57), 'mod_key')
        # Applying the binary operator '&' (line 385)
        result_and__250737 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 44), '&', event_mods_250735, mod_key_250736)
        
        # Getting the type of 'mod_key' (line 385)
        mod_key_250738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 69), 'mod_key')
        # Applying the binary operator '==' (line 385)
        result_eq_250739 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 43), '==', result_and__250737, mod_key_250738)
        
        # Applying the binary operator 'and' (line 385)
        result_and_keyword_250740 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 19), 'and', result_ne_250734, result_eq_250739)
        
        # Getting the type of 'name' (line 384)
        name_250731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'name')
        list_250743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 16), list_250743, name_250731)
        # Assigning a type to the variable 'mods' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'mods', list_250743)
        
        
        # SSA begins for try-except statement (line 386)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 389):
        
        # Assigning a Subscript to a Name (line 389):
        
        # Obtaining the type of the subscript
        # Getting the type of 'event_key' (line 389)
        event_key_250744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 31), 'event_key')
        # Getting the type of 'SPECIAL_KEYS' (line 389)
        SPECIAL_KEYS_250745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 18), 'SPECIAL_KEYS')
        # Obtaining the member '__getitem__' of a type (line 389)
        getitem___250746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 18), SPECIAL_KEYS_250745, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 389)
        subscript_call_result_250747 = invoke(stypy.reporting.localization.Localization(__file__, 389, 18), getitem___250746, event_key_250744)
        
        # Assigning a type to the variable 'key' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'key', subscript_call_result_250747)
        # SSA branch for the except part of a try statement (line 386)
        # SSA branch for the except 'KeyError' branch of a try statement (line 386)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Num to a Name (line 396):
        
        # Assigning a Num to a Name (line 396):
        int_250748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 26), 'int')
        # Assigning a type to the variable 'MAX_UNICODE' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'MAX_UNICODE', int_250748)
        
        
        # Getting the type of 'event_key' (line 397)
        event_key_250749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 15), 'event_key')
        # Getting the type of 'MAX_UNICODE' (line 397)
        MAX_UNICODE_250750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 27), 'MAX_UNICODE')
        # Applying the binary operator '>' (line 397)
        result_gt_250751 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 15), '>', event_key_250749, MAX_UNICODE_250750)
        
        # Testing the type of an if condition (line 397)
        if_condition_250752 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 397, 12), result_gt_250751)
        # Assigning a type to the variable 'if_condition_250752' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'if_condition_250752', if_condition_250752)
        # SSA begins for if statement (line 397)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 398)
        None_250753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 23), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 16), 'stypy_return_type', None_250753)
        # SSA join for if statement (line 397)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 400):
        
        # Assigning a Call to a Name (line 400):
        
        # Call to unichr(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'event_key' (line 400)
        event_key_250755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 25), 'event_key', False)
        # Processing the call keyword arguments (line 400)
        kwargs_250756 = {}
        # Getting the type of 'unichr' (line 400)
        unichr_250754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 18), 'unichr', False)
        # Calling unichr(args, kwargs) (line 400)
        unichr_call_result_250757 = invoke(stypy.reporting.localization.Localization(__file__, 400, 18), unichr_250754, *[event_key_250755], **kwargs_250756)
        
        # Assigning a type to the variable 'key' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'key', unichr_call_result_250757)
        
        
        unicode_250758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 15), 'unicode', u'shift')
        # Getting the type of 'mods' (line 403)
        mods_250759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 26), 'mods')
        # Applying the binary operator 'in' (line 403)
        result_contains_250760 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 15), 'in', unicode_250758, mods_250759)
        
        # Testing the type of an if condition (line 403)
        if_condition_250761 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 12), result_contains_250760)
        # Assigning a type to the variable 'if_condition_250761' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'if_condition_250761', if_condition_250761)
        # SSA begins for if statement (line 403)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove(...): (line 404)
        # Processing the call arguments (line 404)
        unicode_250764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 28), 'unicode', u'shift')
        # Processing the call keyword arguments (line 404)
        kwargs_250765 = {}
        # Getting the type of 'mods' (line 404)
        mods_250762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 16), 'mods', False)
        # Obtaining the member 'remove' of a type (line 404)
        remove_250763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 16), mods_250762, 'remove')
        # Calling remove(args, kwargs) (line 404)
        remove_call_result_250766 = invoke(stypy.reporting.localization.Localization(__file__, 404, 16), remove_250763, *[unicode_250764], **kwargs_250765)
        
        # SSA branch for the else part of an if statement (line 403)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 406):
        
        # Assigning a Call to a Name (line 406):
        
        # Call to lower(...): (line 406)
        # Processing the call keyword arguments (line 406)
        kwargs_250769 = {}
        # Getting the type of 'key' (line 406)
        key_250767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 22), 'key', False)
        # Obtaining the member 'lower' of a type (line 406)
        lower_250768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 22), key_250767, 'lower')
        # Calling lower(args, kwargs) (line 406)
        lower_call_result_250770 = invoke(stypy.reporting.localization.Localization(__file__, 406, 22), lower_250768, *[], **kwargs_250769)
        
        # Assigning a type to the variable 'key' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'key', lower_call_result_250770)
        # SSA join for if statement (line 403)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 386)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to reverse(...): (line 408)
        # Processing the call keyword arguments (line 408)
        kwargs_250773 = {}
        # Getting the type of 'mods' (line 408)
        mods_250771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'mods', False)
        # Obtaining the member 'reverse' of a type (line 408)
        reverse_250772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 8), mods_250771, 'reverse')
        # Calling reverse(args, kwargs) (line 408)
        reverse_call_result_250774 = invoke(stypy.reporting.localization.Localization(__file__, 408, 8), reverse_250772, *[], **kwargs_250773)
        
        
        # Call to join(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'mods' (line 409)
        mods_250777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 24), 'mods', False)
        
        # Obtaining an instance of the builtin type 'list' (line 409)
        list_250778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 409)
        # Adding element type (line 409)
        # Getting the type of 'key' (line 409)
        key_250779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 32), 'key', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 31), list_250778, key_250779)
        
        # Applying the binary operator '+' (line 409)
        result_add_250780 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 24), '+', mods_250777, list_250778)
        
        # Processing the call keyword arguments (line 409)
        kwargs_250781 = {}
        unicode_250775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 15), 'unicode', u'+')
        # Obtaining the member 'join' of a type (line 409)
        join_250776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 15), unicode_250775, 'join')
        # Calling join(args, kwargs) (line 409)
        join_call_result_250782 = invoke(stypy.reporting.localization.Localization(__file__, 409, 15), join_250776, *[result_add_250780], **kwargs_250781)
        
        # Assigning a type to the variable 'stypy_return_type' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'stypy_return_type', join_call_result_250782)
        
        # ################# End of '_get_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_key' in the type store
        # Getting the type of 'stypy_return_type' (line 374)
        stypy_return_type_250783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250783)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_key'
        return stypy_return_type_250783


    @norecursion
    def new_timer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_timer'
        module_type_store = module_type_store.open_function_context('new_timer', 411, 4, False)
        # Assigning a type to the variable 'self' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.new_timer.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.new_timer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.new_timer.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.new_timer.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.new_timer')
        FigureCanvasQT.new_timer.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQT.new_timer.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasQT.new_timer.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasQT.new_timer.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.new_timer.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.new_timer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.new_timer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.new_timer', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        unicode_250784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, (-1)), 'unicode', u"\n        Creates a new backend-specific subclass of\n        :class:`backend_bases.Timer`.  This is useful for getting\n        periodic events through the backend's native event\n        loop. Implemented only for backends with GUIs.\n\n        Other Parameters\n        ----------------\n        interval : scalar\n            Timer interval in milliseconds\n\n        callbacks : list\n            Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``\n            will be executed by the timer every *interval*.\n\n        ")
        
        # Call to TimerQT(...): (line 428)
        # Getting the type of 'args' (line 428)
        args_250786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 24), 'args', False)
        # Processing the call keyword arguments (line 428)
        # Getting the type of 'kwargs' (line 428)
        kwargs_250787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 32), 'kwargs', False)
        kwargs_250788 = {'kwargs_250787': kwargs_250787}
        # Getting the type of 'TimerQT' (line 428)
        TimerQT_250785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 15), 'TimerQT', False)
        # Calling TimerQT(args, kwargs) (line 428)
        TimerQT_call_result_250789 = invoke(stypy.reporting.localization.Localization(__file__, 428, 15), TimerQT_250785, *[args_250786], **kwargs_250788)
        
        # Assigning a type to the variable 'stypy_return_type' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'stypy_return_type', TimerQT_call_result_250789)
        
        # ################# End of 'new_timer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_timer' in the type store
        # Getting the type of 'stypy_return_type' (line 411)
        stypy_return_type_250790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250790)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_timer'
        return stypy_return_type_250790


    @norecursion
    def flush_events(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'flush_events'
        module_type_store = module_type_store.open_function_context('flush_events', 430, 4, False)
        # Assigning a type to the variable 'self' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.flush_events.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.flush_events.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.flush_events.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.flush_events.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.flush_events')
        FigureCanvasQT.flush_events.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasQT.flush_events.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.flush_events.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.flush_events.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.flush_events.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.flush_events.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.flush_events.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.flush_events', [], None, None, defaults, varargs, kwargs)

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

        # Marking variables as global (line 431)
        module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 431, 8), 'qApp')
        
        # Call to processEvents(...): (line 432)
        # Processing the call keyword arguments (line 432)
        kwargs_250793 = {}
        # Getting the type of 'qApp' (line 432)
        qApp_250791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'qApp', False)
        # Obtaining the member 'processEvents' of a type (line 432)
        processEvents_250792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), qApp_250791, 'processEvents')
        # Calling processEvents(args, kwargs) (line 432)
        processEvents_call_result_250794 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), processEvents_250792, *[], **kwargs_250793)
        
        
        # ################# End of 'flush_events(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'flush_events' in the type store
        # Getting the type of 'stypy_return_type' (line 430)
        stypy_return_type_250795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250795)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'flush_events'
        return stypy_return_type_250795


    @norecursion
    def start_event_loop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_250796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 39), 'int')
        defaults = [int_250796]
        # Create a new context for function 'start_event_loop'
        module_type_store = module_type_store.open_function_context('start_event_loop', 434, 4, False)
        # Assigning a type to the variable 'self' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.start_event_loop.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.start_event_loop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.start_event_loop.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.start_event_loop.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.start_event_loop')
        FigureCanvasQT.start_event_loop.__dict__.__setitem__('stypy_param_names_list', ['timeout'])
        FigureCanvasQT.start_event_loop.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.start_event_loop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.start_event_loop.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.start_event_loop.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.start_event_loop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.start_event_loop.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.start_event_loop', ['timeout'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'start_event_loop', localization, ['timeout'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'start_event_loop(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'self' (line 435)
        self_250798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 19), 'self', False)
        unicode_250799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 25), 'unicode', u'_event_loop')
        # Processing the call keyword arguments (line 435)
        kwargs_250800 = {}
        # Getting the type of 'hasattr' (line 435)
        hasattr_250797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 11), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 435)
        hasattr_call_result_250801 = invoke(stypy.reporting.localization.Localization(__file__, 435, 11), hasattr_250797, *[self_250798, unicode_250799], **kwargs_250800)
        
        
        # Call to isRunning(...): (line 435)
        # Processing the call keyword arguments (line 435)
        kwargs_250805 = {}
        # Getting the type of 'self' (line 435)
        self_250802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 44), 'self', False)
        # Obtaining the member '_event_loop' of a type (line 435)
        _event_loop_250803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 44), self_250802, '_event_loop')
        # Obtaining the member 'isRunning' of a type (line 435)
        isRunning_250804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 44), _event_loop_250803, 'isRunning')
        # Calling isRunning(args, kwargs) (line 435)
        isRunning_call_result_250806 = invoke(stypy.reporting.localization.Localization(__file__, 435, 44), isRunning_250804, *[], **kwargs_250805)
        
        # Applying the binary operator 'and' (line 435)
        result_and_keyword_250807 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 11), 'and', hasattr_call_result_250801, isRunning_call_result_250806)
        
        # Testing the type of an if condition (line 435)
        if_condition_250808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 8), result_and_keyword_250807)
        # Assigning a type to the variable 'if_condition_250808' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'if_condition_250808', if_condition_250808)
        # SSA begins for if statement (line 435)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 436)
        # Processing the call arguments (line 436)
        unicode_250810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 31), 'unicode', u'Event loop already running')
        # Processing the call keyword arguments (line 436)
        kwargs_250811 = {}
        # Getting the type of 'RuntimeError' (line 436)
        RuntimeError_250809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 436)
        RuntimeError_call_result_250812 = invoke(stypy.reporting.localization.Localization(__file__, 436, 18), RuntimeError_250809, *[unicode_250810], **kwargs_250811)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 436, 12), RuntimeError_call_result_250812, 'raise parameter', BaseException)
        # SSA join for if statement (line 435)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Name (line 437):
        
        # Call to QEventLoop(...): (line 437)
        # Processing the call keyword arguments (line 437)
        kwargs_250815 = {}
        # Getting the type of 'QtCore' (line 437)
        QtCore_250813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 40), 'QtCore', False)
        # Obtaining the member 'QEventLoop' of a type (line 437)
        QEventLoop_250814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 40), QtCore_250813, 'QEventLoop')
        # Calling QEventLoop(args, kwargs) (line 437)
        QEventLoop_call_result_250816 = invoke(stypy.reporting.localization.Localization(__file__, 437, 40), QEventLoop_250814, *[], **kwargs_250815)
        
        # Assigning a type to the variable 'event_loop' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 27), 'event_loop', QEventLoop_call_result_250816)
        
        # Assigning a Name to a Attribute (line 437):
        # Getting the type of 'event_loop' (line 437)
        event_loop_250817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 27), 'event_loop')
        # Getting the type of 'self' (line 437)
        self_250818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'self')
        # Setting the type of the member '_event_loop' of a type (line 437)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), self_250818, '_event_loop', event_loop_250817)
        
        # Getting the type of 'timeout' (line 438)
        timeout_250819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'timeout')
        # Testing the type of an if condition (line 438)
        if_condition_250820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 8), timeout_250819)
        # Assigning a type to the variable 'if_condition_250820' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'if_condition_250820', if_condition_250820)
        # SSA begins for if statement (line 438)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 439):
        
        # Assigning a Call to a Name (line 439):
        
        # Call to singleShot(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'timeout' (line 439)
        timeout_250824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 45), 'timeout', False)
        int_250825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 55), 'int')
        # Applying the binary operator '*' (line 439)
        result_mul_250826 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 45), '*', timeout_250824, int_250825)
        
        # Getting the type of 'event_loop' (line 439)
        event_loop_250827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 61), 'event_loop', False)
        # Obtaining the member 'quit' of a type (line 439)
        quit_250828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 61), event_loop_250827, 'quit')
        # Processing the call keyword arguments (line 439)
        kwargs_250829 = {}
        # Getting the type of 'QtCore' (line 439)
        QtCore_250821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 20), 'QtCore', False)
        # Obtaining the member 'QTimer' of a type (line 439)
        QTimer_250822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 20), QtCore_250821, 'QTimer')
        # Obtaining the member 'singleShot' of a type (line 439)
        singleShot_250823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 20), QTimer_250822, 'singleShot')
        # Calling singleShot(args, kwargs) (line 439)
        singleShot_call_result_250830 = invoke(stypy.reporting.localization.Localization(__file__, 439, 20), singleShot_250823, *[result_mul_250826, quit_250828], **kwargs_250829)
        
        # Assigning a type to the variable 'timer' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'timer', singleShot_call_result_250830)
        # SSA join for if statement (line 438)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to exec_(...): (line 440)
        # Processing the call keyword arguments (line 440)
        kwargs_250833 = {}
        # Getting the type of 'event_loop' (line 440)
        event_loop_250831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'event_loop', False)
        # Obtaining the member 'exec_' of a type (line 440)
        exec__250832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 8), event_loop_250831, 'exec_')
        # Calling exec_(args, kwargs) (line 440)
        exec__call_result_250834 = invoke(stypy.reporting.localization.Localization(__file__, 440, 8), exec__250832, *[], **kwargs_250833)
        
        
        # ################# End of 'start_event_loop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'start_event_loop' in the type store
        # Getting the type of 'stypy_return_type' (line 434)
        stypy_return_type_250835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250835)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'start_event_loop'
        return stypy_return_type_250835


    @norecursion
    def stop_event_loop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 442)
        None_250836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 36), 'None')
        defaults = [None_250836]
        # Create a new context for function 'stop_event_loop'
        module_type_store = module_type_store.open_function_context('stop_event_loop', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasQT.stop_event_loop.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasQT.stop_event_loop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasQT.stop_event_loop.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasQT.stop_event_loop.__dict__.__setitem__('stypy_function_name', 'FigureCanvasQT.stop_event_loop')
        FigureCanvasQT.stop_event_loop.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasQT.stop_event_loop.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasQT.stop_event_loop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasQT.stop_event_loop.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasQT.stop_event_loop.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasQT.stop_event_loop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasQT.stop_event_loop.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasQT.stop_event_loop', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'stop_event_loop', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'stop_event_loop(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 443)
        unicode_250837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 25), 'unicode', u'_event_loop')
        # Getting the type of 'self' (line 443)
        self_250838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 19), 'self')
        
        (may_be_250839, more_types_in_union_250840) = may_provide_member(unicode_250837, self_250838)

        if may_be_250839:

            if more_types_in_union_250840:
                # Runtime conditional SSA (line 443)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'self' (line 443)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'self', remove_not_member_provider_from_union(self_250838, u'_event_loop'))
            
            # Call to quit(...): (line 444)
            # Processing the call keyword arguments (line 444)
            kwargs_250844 = {}
            # Getting the type of 'self' (line 444)
            self_250841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'self', False)
            # Obtaining the member '_event_loop' of a type (line 444)
            _event_loop_250842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), self_250841, '_event_loop')
            # Obtaining the member 'quit' of a type (line 444)
            quit_250843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), _event_loop_250842, 'quit')
            # Calling quit(args, kwargs) (line 444)
            quit_call_result_250845 = invoke(stypy.reporting.localization.Localization(__file__, 444, 12), quit_250843, *[], **kwargs_250844)
            

            if more_types_in_union_250840:
                # SSA join for if statement (line 443)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'stop_event_loop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'stop_event_loop' in the type store
        # Getting the type of 'stypy_return_type' (line 442)
        stypy_return_type_250846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250846)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'stop_event_loop'
        return stypy_return_type_250846


# Assigning a type to the variable 'FigureCanvasQT' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'FigureCanvasQT', FigureCanvasQT)

# Assigning a Dict to a Name (line 222):

# Obtaining an instance of the builtin type 'dict' (line 222)
dict_250847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 222)
# Adding element type (key, value) (line 222)
# Getting the type of 'QtCore' (line 222)
QtCore_250848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'QtCore')
# Obtaining the member 'Qt' of a type (line 222)
Qt_250849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 15), QtCore_250848, 'Qt')
# Obtaining the member 'LeftButton' of a type (line 222)
LeftButton_250850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 15), Qt_250849, 'LeftButton')
int_250851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 37), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 14), dict_250847, (LeftButton_250850, int_250851))
# Adding element type (key, value) (line 222)
# Getting the type of 'QtCore' (line 223)
QtCore_250852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'QtCore')
# Obtaining the member 'Qt' of a type (line 223)
Qt_250853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), QtCore_250852, 'Qt')
# Obtaining the member 'MidButton' of a type (line 223)
MidButton_250854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), Qt_250853, 'MidButton')
int_250855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 36), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 14), dict_250847, (MidButton_250854, int_250855))
# Adding element type (key, value) (line 222)
# Getting the type of 'QtCore' (line 224)
QtCore_250856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'QtCore')
# Obtaining the member 'Qt' of a type (line 224)
Qt_250857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 15), QtCore_250856, 'Qt')
# Obtaining the member 'RightButton' of a type (line 224)
RightButton_250858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 15), Qt_250857, 'RightButton')
int_250859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 38), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 14), dict_250847, (RightButton_250858, int_250859))

# Getting the type of 'FigureCanvasQT'
FigureCanvasQT_250860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasQT')
# Setting the type of the member 'buttond' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasQT_250860, 'buttond', dict_250847)
# Declaration of the 'MainWindow' class
# Getting the type of 'QtWidgets' (line 447)
QtWidgets_250861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 17), 'QtWidgets')
# Obtaining the member 'QMainWindow' of a type (line 447)
QMainWindow_250862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 17), QtWidgets_250861, 'QMainWindow')

class MainWindow(QMainWindow_250862, ):
    
    # Assigning a Call to a Name (line 448):

    @norecursion
    def closeEvent(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'closeEvent'
        module_type_store = module_type_store.open_function_context('closeEvent', 450, 4, False)
        # Assigning a type to the variable 'self' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MainWindow.closeEvent.__dict__.__setitem__('stypy_localization', localization)
        MainWindow.closeEvent.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MainWindow.closeEvent.__dict__.__setitem__('stypy_type_store', module_type_store)
        MainWindow.closeEvent.__dict__.__setitem__('stypy_function_name', 'MainWindow.closeEvent')
        MainWindow.closeEvent.__dict__.__setitem__('stypy_param_names_list', ['event'])
        MainWindow.closeEvent.__dict__.__setitem__('stypy_varargs_param_name', None)
        MainWindow.closeEvent.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MainWindow.closeEvent.__dict__.__setitem__('stypy_call_defaults', defaults)
        MainWindow.closeEvent.__dict__.__setitem__('stypy_call_varargs', varargs)
        MainWindow.closeEvent.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MainWindow.closeEvent.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MainWindow.closeEvent', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'closeEvent', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'closeEvent(...)' code ##################

        
        # Call to emit(...): (line 451)
        # Processing the call keyword arguments (line 451)
        kwargs_250866 = {}
        # Getting the type of 'self' (line 451)
        self_250863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'self', False)
        # Obtaining the member 'closing' of a type (line 451)
        closing_250864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), self_250863, 'closing')
        # Obtaining the member 'emit' of a type (line 451)
        emit_250865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), closing_250864, 'emit')
        # Calling emit(args, kwargs) (line 451)
        emit_call_result_250867 = invoke(stypy.reporting.localization.Localization(__file__, 451, 8), emit_250865, *[], **kwargs_250866)
        
        
        # Call to closeEvent(...): (line 452)
        # Processing the call arguments (line 452)
        # Getting the type of 'self' (line 452)
        self_250871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 41), 'self', False)
        # Getting the type of 'event' (line 452)
        event_250872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 47), 'event', False)
        # Processing the call keyword arguments (line 452)
        kwargs_250873 = {}
        # Getting the type of 'QtWidgets' (line 452)
        QtWidgets_250868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'QtWidgets', False)
        # Obtaining the member 'QMainWindow' of a type (line 452)
        QMainWindow_250869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), QtWidgets_250868, 'QMainWindow')
        # Obtaining the member 'closeEvent' of a type (line 452)
        closeEvent_250870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), QMainWindow_250869, 'closeEvent')
        # Calling closeEvent(args, kwargs) (line 452)
        closeEvent_call_result_250874 = invoke(stypy.reporting.localization.Localization(__file__, 452, 8), closeEvent_250870, *[self_250871, event_250872], **kwargs_250873)
        
        
        # ################# End of 'closeEvent(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'closeEvent' in the type store
        # Getting the type of 'stypy_return_type' (line 450)
        stypy_return_type_250875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_250875)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'closeEvent'
        return stypy_return_type_250875


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 447, 0, False)
        # Assigning a type to the variable 'self' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MainWindow.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MainWindow' (line 447)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 0), 'MainWindow', MainWindow)

# Assigning a Call to a Name (line 448):

# Call to Signal(...): (line 448)
# Processing the call keyword arguments (line 448)
kwargs_250878 = {}
# Getting the type of 'QtCore' (line 448)
QtCore_250876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 14), 'QtCore', False)
# Obtaining the member 'Signal' of a type (line 448)
Signal_250877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 14), QtCore_250876, 'Signal')
# Calling Signal(args, kwargs) (line 448)
Signal_call_result_250879 = invoke(stypy.reporting.localization.Localization(__file__, 448, 14), Signal_250877, *[], **kwargs_250878)

# Getting the type of 'MainWindow'
MainWindow_250880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MainWindow')
# Setting the type of the member 'closing' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MainWindow_250880, 'closing', Signal_call_result_250879)
# Declaration of the 'FigureManagerQT' class
# Getting the type of 'FigureManagerBase' (line 455)
FigureManagerBase_250881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 22), 'FigureManagerBase')

class FigureManagerQT(FigureManagerBase_250881, ):
    unicode_250882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, (-1)), 'unicode', u'\n    Attributes\n    ----------\n    canvas : `FigureCanvas`\n        The FigureCanvas instance\n    num : int or str\n        The Figure number\n    toolbar : qt.QToolBar\n        The qt.QToolBar\n    window : qt.QMainWindow\n        The qt.QMainWindow\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 470, 4, False)
        # Assigning a type to the variable 'self' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerQT.__init__', ['canvas', 'num'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'self' (line 471)
        self_250885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 35), 'self', False)
        # Getting the type of 'canvas' (line 471)
        canvas_250886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 41), 'canvas', False)
        # Getting the type of 'num' (line 471)
        num_250887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 49), 'num', False)
        # Processing the call keyword arguments (line 471)
        kwargs_250888 = {}
        # Getting the type of 'FigureManagerBase' (line 471)
        FigureManagerBase_250883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'FigureManagerBase', False)
        # Obtaining the member '__init__' of a type (line 471)
        init___250884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 8), FigureManagerBase_250883, '__init__')
        # Calling __init__(args, kwargs) (line 471)
        init___call_result_250889 = invoke(stypy.reporting.localization.Localization(__file__, 471, 8), init___250884, *[self_250885, canvas_250886, num_250887], **kwargs_250888)
        
        
        # Assigning a Name to a Attribute (line 472):
        
        # Assigning a Name to a Attribute (line 472):
        # Getting the type of 'canvas' (line 472)
        canvas_250890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 22), 'canvas')
        # Getting the type of 'self' (line 472)
        self_250891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'self')
        # Setting the type of the member 'canvas' of a type (line 472)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), self_250891, 'canvas', canvas_250890)
        
        # Assigning a Call to a Attribute (line 473):
        
        # Assigning a Call to a Attribute (line 473):
        
        # Call to MainWindow(...): (line 473)
        # Processing the call keyword arguments (line 473)
        kwargs_250893 = {}
        # Getting the type of 'MainWindow' (line 473)
        MainWindow_250892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 22), 'MainWindow', False)
        # Calling MainWindow(args, kwargs) (line 473)
        MainWindow_call_result_250894 = invoke(stypy.reporting.localization.Localization(__file__, 473, 22), MainWindow_250892, *[], **kwargs_250893)
        
        # Getting the type of 'self' (line 473)
        self_250895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'self')
        # Setting the type of the member 'window' of a type (line 473)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 8), self_250895, 'window', MainWindow_call_result_250894)
        
        # Call to connect(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'canvas' (line 474)
        canvas_250900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 36), 'canvas', False)
        # Obtaining the member 'close_event' of a type (line 474)
        close_event_250901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 36), canvas_250900, 'close_event')
        # Processing the call keyword arguments (line 474)
        kwargs_250902 = {}
        # Getting the type of 'self' (line 474)
        self_250896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 474)
        window_250897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), self_250896, 'window')
        # Obtaining the member 'closing' of a type (line 474)
        closing_250898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), window_250897, 'closing')
        # Obtaining the member 'connect' of a type (line 474)
        connect_250899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), closing_250898, 'connect')
        # Calling connect(args, kwargs) (line 474)
        connect_call_result_250903 = invoke(stypy.reporting.localization.Localization(__file__, 474, 8), connect_250899, *[close_event_250901], **kwargs_250902)
        
        
        # Call to connect(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'self' (line 475)
        self_250908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 36), 'self', False)
        # Obtaining the member '_widgetclosed' of a type (line 475)
        _widgetclosed_250909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 36), self_250908, '_widgetclosed')
        # Processing the call keyword arguments (line 475)
        kwargs_250910 = {}
        # Getting the type of 'self' (line 475)
        self_250904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 475)
        window_250905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), self_250904, 'window')
        # Obtaining the member 'closing' of a type (line 475)
        closing_250906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), window_250905, 'closing')
        # Obtaining the member 'connect' of a type (line 475)
        connect_250907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), closing_250906, 'connect')
        # Calling connect(args, kwargs) (line 475)
        connect_call_result_250911 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), connect_250907, *[_widgetclosed_250909], **kwargs_250910)
        
        
        # Call to setWindowTitle(...): (line 477)
        # Processing the call arguments (line 477)
        unicode_250915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 35), 'unicode', u'Figure %d')
        # Getting the type of 'num' (line 477)
        num_250916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 49), 'num', False)
        # Applying the binary operator '%' (line 477)
        result_mod_250917 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 35), '%', unicode_250915, num_250916)
        
        # Processing the call keyword arguments (line 477)
        kwargs_250918 = {}
        # Getting the type of 'self' (line 477)
        self_250912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 477)
        window_250913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), self_250912, 'window')
        # Obtaining the member 'setWindowTitle' of a type (line 477)
        setWindowTitle_250914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), window_250913, 'setWindowTitle')
        # Calling setWindowTitle(args, kwargs) (line 477)
        setWindowTitle_call_result_250919 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), setWindowTitle_250914, *[result_mod_250917], **kwargs_250918)
        
        
        # Assigning a Call to a Name (line 478):
        
        # Assigning a Call to a Name (line 478):
        
        # Call to join(...): (line 478)
        # Processing the call arguments (line 478)
        
        # Obtaining the type of the subscript
        unicode_250923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 49), 'unicode', u'datapath')
        # Getting the type of 'matplotlib' (line 478)
        matplotlib_250924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 29), 'matplotlib', False)
        # Obtaining the member 'rcParams' of a type (line 478)
        rcParams_250925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 29), matplotlib_250924, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 478)
        getitem___250926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 29), rcParams_250925, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 478)
        subscript_call_result_250927 = invoke(stypy.reporting.localization.Localization(__file__, 478, 29), getitem___250926, unicode_250923)
        
        unicode_250928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 29), 'unicode', u'images')
        unicode_250929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 39), 'unicode', u'matplotlib.svg')
        # Processing the call keyword arguments (line 478)
        kwargs_250930 = {}
        # Getting the type of 'os' (line 478)
        os_250920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 478)
        path_250921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 16), os_250920, 'path')
        # Obtaining the member 'join' of a type (line 478)
        join_250922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 16), path_250921, 'join')
        # Calling join(args, kwargs) (line 478)
        join_call_result_250931 = invoke(stypy.reporting.localization.Localization(__file__, 478, 16), join_250922, *[subscript_call_result_250927, unicode_250928, unicode_250929], **kwargs_250930)
        
        # Assigning a type to the variable 'image' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'image', join_call_result_250931)
        
        # Call to setWindowIcon(...): (line 480)
        # Processing the call arguments (line 480)
        
        # Call to QIcon(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'image' (line 480)
        image_250937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 46), 'image', False)
        # Processing the call keyword arguments (line 480)
        kwargs_250938 = {}
        # Getting the type of 'QtGui' (line 480)
        QtGui_250935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 34), 'QtGui', False)
        # Obtaining the member 'QIcon' of a type (line 480)
        QIcon_250936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 34), QtGui_250935, 'QIcon')
        # Calling QIcon(args, kwargs) (line 480)
        QIcon_call_result_250939 = invoke(stypy.reporting.localization.Localization(__file__, 480, 34), QIcon_250936, *[image_250937], **kwargs_250938)
        
        # Processing the call keyword arguments (line 480)
        kwargs_250940 = {}
        # Getting the type of 'self' (line 480)
        self_250932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 480)
        window_250933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), self_250932, 'window')
        # Obtaining the member 'setWindowIcon' of a type (line 480)
        setWindowIcon_250934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), window_250933, 'setWindowIcon')
        # Calling setWindowIcon(args, kwargs) (line 480)
        setWindowIcon_call_result_250941 = invoke(stypy.reporting.localization.Localization(__file__, 480, 8), setWindowIcon_250934, *[QIcon_call_result_250939], **kwargs_250940)
        
        
        # Call to setFocusPolicy(...): (line 489)
        # Processing the call arguments (line 489)
        # Getting the type of 'QtCore' (line 489)
        QtCore_250945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 35), 'QtCore', False)
        # Obtaining the member 'Qt' of a type (line 489)
        Qt_250946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 35), QtCore_250945, 'Qt')
        # Obtaining the member 'StrongFocus' of a type (line 489)
        StrongFocus_250947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 35), Qt_250946, 'StrongFocus')
        # Processing the call keyword arguments (line 489)
        kwargs_250948 = {}
        # Getting the type of 'self' (line 489)
        self_250942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 489)
        canvas_250943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 8), self_250942, 'canvas')
        # Obtaining the member 'setFocusPolicy' of a type (line 489)
        setFocusPolicy_250944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 8), canvas_250943, 'setFocusPolicy')
        # Calling setFocusPolicy(args, kwargs) (line 489)
        setFocusPolicy_call_result_250949 = invoke(stypy.reporting.localization.Localization(__file__, 489, 8), setFocusPolicy_250944, *[StrongFocus_250947], **kwargs_250948)
        
        
        # Call to setFocus(...): (line 490)
        # Processing the call keyword arguments (line 490)
        kwargs_250953 = {}
        # Getting the type of 'self' (line 490)
        self_250950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 490)
        canvas_250951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), self_250950, 'canvas')
        # Obtaining the member 'setFocus' of a type (line 490)
        setFocus_250952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), canvas_250951, 'setFocus')
        # Calling setFocus(args, kwargs) (line 490)
        setFocus_call_result_250954 = invoke(stypy.reporting.localization.Localization(__file__, 490, 8), setFocus_250952, *[], **kwargs_250953)
        
        
        # Assigning a Name to a Attribute (line 492):
        
        # Assigning a Name to a Attribute (line 492):
        # Getting the type of 'False' (line 492)
        False_250955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 34), 'False')
        # Getting the type of 'self' (line 492)
        self_250956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'self')
        # Obtaining the member 'window' of a type (line 492)
        window_250957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 8), self_250956, 'window')
        # Setting the type of the member '_destroying' of a type (line 492)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 8), window_250957, '_destroying', False_250955)
        
        # Assigning a Call to a Attribute (line 495):
        
        # Assigning a Call to a Attribute (line 495):
        
        # Call to QLabel(...): (line 495)
        # Processing the call keyword arguments (line 495)
        kwargs_250960 = {}
        # Getting the type of 'QtWidgets' (line 495)
        QtWidgets_250958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 31), 'QtWidgets', False)
        # Obtaining the member 'QLabel' of a type (line 495)
        QLabel_250959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 31), QtWidgets_250958, 'QLabel')
        # Calling QLabel(args, kwargs) (line 495)
        QLabel_call_result_250961 = invoke(stypy.reporting.localization.Localization(__file__, 495, 31), QLabel_250959, *[], **kwargs_250960)
        
        # Getting the type of 'self' (line 495)
        self_250962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'self')
        # Setting the type of the member 'statusbar_label' of a type (line 495)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 8), self_250962, 'statusbar_label', QLabel_call_result_250961)
        
        # Call to addWidget(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'self' (line 496)
        self_250969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 42), 'self', False)
        # Obtaining the member 'statusbar_label' of a type (line 496)
        statusbar_label_250970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 42), self_250969, 'statusbar_label')
        # Processing the call keyword arguments (line 496)
        kwargs_250971 = {}
        
        # Call to statusBar(...): (line 496)
        # Processing the call keyword arguments (line 496)
        kwargs_250966 = {}
        # Getting the type of 'self' (line 496)
        self_250963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 496)
        window_250964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 8), self_250963, 'window')
        # Obtaining the member 'statusBar' of a type (line 496)
        statusBar_250965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 8), window_250964, 'statusBar')
        # Calling statusBar(args, kwargs) (line 496)
        statusBar_call_result_250967 = invoke(stypy.reporting.localization.Localization(__file__, 496, 8), statusBar_250965, *[], **kwargs_250966)
        
        # Obtaining the member 'addWidget' of a type (line 496)
        addWidget_250968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 8), statusBar_call_result_250967, 'addWidget')
        # Calling addWidget(args, kwargs) (line 496)
        addWidget_call_result_250972 = invoke(stypy.reporting.localization.Localization(__file__, 496, 8), addWidget_250968, *[statusbar_label_250970], **kwargs_250971)
        
        
        # Assigning a Call to a Attribute (line 498):
        
        # Assigning a Call to a Attribute (line 498):
        
        # Call to _get_toolbar(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'self' (line 498)
        self_250975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 41), 'self', False)
        # Obtaining the member 'canvas' of a type (line 498)
        canvas_250976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 41), self_250975, 'canvas')
        # Getting the type of 'self' (line 498)
        self_250977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 54), 'self', False)
        # Obtaining the member 'window' of a type (line 498)
        window_250978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 54), self_250977, 'window')
        # Processing the call keyword arguments (line 498)
        kwargs_250979 = {}
        # Getting the type of 'self' (line 498)
        self_250973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 23), 'self', False)
        # Obtaining the member '_get_toolbar' of a type (line 498)
        _get_toolbar_250974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 23), self_250973, '_get_toolbar')
        # Calling _get_toolbar(args, kwargs) (line 498)
        _get_toolbar_call_result_250980 = invoke(stypy.reporting.localization.Localization(__file__, 498, 23), _get_toolbar_250974, *[canvas_250976, window_250978], **kwargs_250979)
        
        # Getting the type of 'self' (line 498)
        self_250981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'self')
        # Setting the type of the member 'toolbar' of a type (line 498)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), self_250981, 'toolbar', _get_toolbar_call_result_250980)
        
        
        # Getting the type of 'self' (line 499)
        self_250982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 11), 'self')
        # Obtaining the member 'toolbar' of a type (line 499)
        toolbar_250983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 11), self_250982, 'toolbar')
        # Getting the type of 'None' (line 499)
        None_250984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 31), 'None')
        # Applying the binary operator 'isnot' (line 499)
        result_is_not_250985 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 11), 'isnot', toolbar_250983, None_250984)
        
        # Testing the type of an if condition (line 499)
        if_condition_250986 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 8), result_is_not_250985)
        # Assigning a type to the variable 'if_condition_250986' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'if_condition_250986', if_condition_250986)
        # SSA begins for if statement (line 499)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to addToolBar(...): (line 500)
        # Processing the call arguments (line 500)
        # Getting the type of 'self' (line 500)
        self_250990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 35), 'self', False)
        # Obtaining the member 'toolbar' of a type (line 500)
        toolbar_250991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 35), self_250990, 'toolbar')
        # Processing the call keyword arguments (line 500)
        kwargs_250992 = {}
        # Getting the type of 'self' (line 500)
        self_250987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'self', False)
        # Obtaining the member 'window' of a type (line 500)
        window_250988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 12), self_250987, 'window')
        # Obtaining the member 'addToolBar' of a type (line 500)
        addToolBar_250989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 12), window_250988, 'addToolBar')
        # Calling addToolBar(args, kwargs) (line 500)
        addToolBar_call_result_250993 = invoke(stypy.reporting.localization.Localization(__file__, 500, 12), addToolBar_250989, *[toolbar_250991], **kwargs_250992)
        
        
        # Call to connect(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of 'self' (line 501)
        self_250998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 41), 'self', False)
        # Obtaining the member 'statusbar_label' of a type (line 501)
        statusbar_label_250999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 41), self_250998, 'statusbar_label')
        # Obtaining the member 'setText' of a type (line 501)
        setText_251000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 41), statusbar_label_250999, 'setText')
        # Processing the call keyword arguments (line 501)
        kwargs_251001 = {}
        # Getting the type of 'self' (line 501)
        self_250994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'self', False)
        # Obtaining the member 'toolbar' of a type (line 501)
        toolbar_250995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 12), self_250994, 'toolbar')
        # Obtaining the member 'message' of a type (line 501)
        message_250996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 12), toolbar_250995, 'message')
        # Obtaining the member 'connect' of a type (line 501)
        connect_250997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 12), message_250996, 'connect')
        # Calling connect(args, kwargs) (line 501)
        connect_call_result_251002 = invoke(stypy.reporting.localization.Localization(__file__, 501, 12), connect_250997, *[setText_251000], **kwargs_251001)
        
        
        # Assigning a Call to a Name (line 502):
        
        # Assigning a Call to a Name (line 502):
        
        # Call to height(...): (line 502)
        # Processing the call keyword arguments (line 502)
        kwargs_251009 = {}
        
        # Call to sizeHint(...): (line 502)
        # Processing the call keyword arguments (line 502)
        kwargs_251006 = {}
        # Getting the type of 'self' (line 502)
        self_251003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 25), 'self', False)
        # Obtaining the member 'toolbar' of a type (line 502)
        toolbar_251004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 25), self_251003, 'toolbar')
        # Obtaining the member 'sizeHint' of a type (line 502)
        sizeHint_251005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 25), toolbar_251004, 'sizeHint')
        # Calling sizeHint(args, kwargs) (line 502)
        sizeHint_call_result_251007 = invoke(stypy.reporting.localization.Localization(__file__, 502, 25), sizeHint_251005, *[], **kwargs_251006)
        
        # Obtaining the member 'height' of a type (line 502)
        height_251008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 25), sizeHint_call_result_251007, 'height')
        # Calling height(args, kwargs) (line 502)
        height_call_result_251010 = invoke(stypy.reporting.localization.Localization(__file__, 502, 25), height_251008, *[], **kwargs_251009)
        
        # Assigning a type to the variable 'tbs_height' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'tbs_height', height_call_result_251010)
        # SSA branch for the else part of an if statement (line 499)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 504):
        
        # Assigning a Num to a Name (line 504):
        int_251011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 25), 'int')
        # Assigning a type to the variable 'tbs_height' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'tbs_height', int_251011)
        # SSA join for if statement (line 499)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 508):
        
        # Assigning a Call to a Name (line 508):
        
        # Call to sizeHint(...): (line 508)
        # Processing the call keyword arguments (line 508)
        kwargs_251014 = {}
        # Getting the type of 'canvas' (line 508)
        canvas_251012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 13), 'canvas', False)
        # Obtaining the member 'sizeHint' of a type (line 508)
        sizeHint_251013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 13), canvas_251012, 'sizeHint')
        # Calling sizeHint(args, kwargs) (line 508)
        sizeHint_call_result_251015 = invoke(stypy.reporting.localization.Localization(__file__, 508, 13), sizeHint_251013, *[], **kwargs_251014)
        
        # Assigning a type to the variable 'cs' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'cs', sizeHint_call_result_251015)
        
        # Assigning a Call to a Name (line 509):
        
        # Assigning a Call to a Name (line 509):
        
        # Call to sizeHint(...): (line 509)
        # Processing the call keyword arguments (line 509)
        kwargs_251022 = {}
        
        # Call to statusBar(...): (line 509)
        # Processing the call keyword arguments (line 509)
        kwargs_251019 = {}
        # Getting the type of 'self' (line 509)
        self_251016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 14), 'self', False)
        # Obtaining the member 'window' of a type (line 509)
        window_251017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 14), self_251016, 'window')
        # Obtaining the member 'statusBar' of a type (line 509)
        statusBar_251018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 14), window_251017, 'statusBar')
        # Calling statusBar(args, kwargs) (line 509)
        statusBar_call_result_251020 = invoke(stypy.reporting.localization.Localization(__file__, 509, 14), statusBar_251018, *[], **kwargs_251019)
        
        # Obtaining the member 'sizeHint' of a type (line 509)
        sizeHint_251021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 14), statusBar_call_result_251020, 'sizeHint')
        # Calling sizeHint(args, kwargs) (line 509)
        sizeHint_call_result_251023 = invoke(stypy.reporting.localization.Localization(__file__, 509, 14), sizeHint_251021, *[], **kwargs_251022)
        
        # Assigning a type to the variable 'sbs' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'sbs', sizeHint_call_result_251023)
        
        # Assigning a BinOp to a Attribute (line 510):
        
        # Assigning a BinOp to a Attribute (line 510):
        # Getting the type of 'tbs_height' (line 510)
        tbs_height_251024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 39), 'tbs_height')
        
        # Call to height(...): (line 510)
        # Processing the call keyword arguments (line 510)
        kwargs_251027 = {}
        # Getting the type of 'sbs' (line 510)
        sbs_251025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 52), 'sbs', False)
        # Obtaining the member 'height' of a type (line 510)
        height_251026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 52), sbs_251025, 'height')
        # Calling height(args, kwargs) (line 510)
        height_call_result_251028 = invoke(stypy.reporting.localization.Localization(__file__, 510, 52), height_251026, *[], **kwargs_251027)
        
        # Applying the binary operator '+' (line 510)
        result_add_251029 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 39), '+', tbs_height_251024, height_call_result_251028)
        
        # Getting the type of 'self' (line 510)
        self_251030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'self')
        # Setting the type of the member '_status_and_tool_height' of a type (line 510)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), self_251030, '_status_and_tool_height', result_add_251029)
        
        # Assigning a BinOp to a Name (line 511):
        
        # Assigning a BinOp to a Name (line 511):
        
        # Call to height(...): (line 511)
        # Processing the call keyword arguments (line 511)
        kwargs_251033 = {}
        # Getting the type of 'cs' (line 511)
        cs_251031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 17), 'cs', False)
        # Obtaining the member 'height' of a type (line 511)
        height_251032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 17), cs_251031, 'height')
        # Calling height(args, kwargs) (line 511)
        height_call_result_251034 = invoke(stypy.reporting.localization.Localization(__file__, 511, 17), height_251032, *[], **kwargs_251033)
        
        # Getting the type of 'self' (line 511)
        self_251035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 31), 'self')
        # Obtaining the member '_status_and_tool_height' of a type (line 511)
        _status_and_tool_height_251036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 31), self_251035, '_status_and_tool_height')
        # Applying the binary operator '+' (line 511)
        result_add_251037 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 17), '+', height_call_result_251034, _status_and_tool_height_251036)
        
        # Assigning a type to the variable 'height' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'height', result_add_251037)
        
        # Call to resize(...): (line 512)
        # Processing the call arguments (line 512)
        
        # Call to width(...): (line 512)
        # Processing the call keyword arguments (line 512)
        kwargs_251043 = {}
        # Getting the type of 'cs' (line 512)
        cs_251041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 27), 'cs', False)
        # Obtaining the member 'width' of a type (line 512)
        width_251042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 27), cs_251041, 'width')
        # Calling width(args, kwargs) (line 512)
        width_call_result_251044 = invoke(stypy.reporting.localization.Localization(__file__, 512, 27), width_251042, *[], **kwargs_251043)
        
        # Getting the type of 'height' (line 512)
        height_251045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 39), 'height', False)
        # Processing the call keyword arguments (line 512)
        kwargs_251046 = {}
        # Getting the type of 'self' (line 512)
        self_251038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 512)
        window_251039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), self_251038, 'window')
        # Obtaining the member 'resize' of a type (line 512)
        resize_251040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), window_251039, 'resize')
        # Calling resize(args, kwargs) (line 512)
        resize_call_result_251047 = invoke(stypy.reporting.localization.Localization(__file__, 512, 8), resize_251040, *[width_call_result_251044, height_251045], **kwargs_251046)
        
        
        # Call to setCentralWidget(...): (line 514)
        # Processing the call arguments (line 514)
        # Getting the type of 'self' (line 514)
        self_251051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 37), 'self', False)
        # Obtaining the member 'canvas' of a type (line 514)
        canvas_251052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 37), self_251051, 'canvas')
        # Processing the call keyword arguments (line 514)
        kwargs_251053 = {}
        # Getting the type of 'self' (line 514)
        self_251048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 514)
        window_251049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 8), self_251048, 'window')
        # Obtaining the member 'setCentralWidget' of a type (line 514)
        setCentralWidget_251050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 8), window_251049, 'setCentralWidget')
        # Calling setCentralWidget(args, kwargs) (line 514)
        setCentralWidget_call_result_251054 = invoke(stypy.reporting.localization.Localization(__file__, 514, 8), setCentralWidget_251050, *[canvas_251052], **kwargs_251053)
        
        
        
        # Call to is_interactive(...): (line 516)
        # Processing the call keyword arguments (line 516)
        kwargs_251057 = {}
        # Getting the type of 'matplotlib' (line 516)
        matplotlib_251055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'matplotlib', False)
        # Obtaining the member 'is_interactive' of a type (line 516)
        is_interactive_251056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 11), matplotlib_251055, 'is_interactive')
        # Calling is_interactive(args, kwargs) (line 516)
        is_interactive_call_result_251058 = invoke(stypy.reporting.localization.Localization(__file__, 516, 11), is_interactive_251056, *[], **kwargs_251057)
        
        # Testing the type of an if condition (line 516)
        if_condition_251059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 516, 8), is_interactive_call_result_251058)
        # Assigning a type to the variable 'if_condition_251059' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'if_condition_251059', if_condition_251059)
        # SSA begins for if statement (line 516)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to show(...): (line 517)
        # Processing the call keyword arguments (line 517)
        kwargs_251063 = {}
        # Getting the type of 'self' (line 517)
        self_251060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'self', False)
        # Obtaining the member 'window' of a type (line 517)
        window_251061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 12), self_251060, 'window')
        # Obtaining the member 'show' of a type (line 517)
        show_251062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 12), window_251061, 'show')
        # Calling show(args, kwargs) (line 517)
        show_call_result_251064 = invoke(stypy.reporting.localization.Localization(__file__, 517, 12), show_251062, *[], **kwargs_251063)
        
        
        # Call to draw_idle(...): (line 518)
        # Processing the call keyword arguments (line 518)
        kwargs_251068 = {}
        # Getting the type of 'self' (line 518)
        self_251065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'self', False)
        # Obtaining the member 'canvas' of a type (line 518)
        canvas_251066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), self_251065, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 518)
        draw_idle_251067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), canvas_251066, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 518)
        draw_idle_call_result_251069 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), draw_idle_251067, *[], **kwargs_251068)
        
        # SSA join for if statement (line 516)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def notify_axes_change(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'notify_axes_change'
            module_type_store = module_type_store.open_function_context('notify_axes_change', 520, 8, False)
            
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

            
            
            # Getting the type of 'self' (line 522)
            self_251070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'self')
            # Obtaining the member 'toolbar' of a type (line 522)
            toolbar_251071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 15), self_251070, 'toolbar')
            # Getting the type of 'None' (line 522)
            None_251072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 35), 'None')
            # Applying the binary operator 'isnot' (line 522)
            result_is_not_251073 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 15), 'isnot', toolbar_251071, None_251072)
            
            # Testing the type of an if condition (line 522)
            if_condition_251074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 12), result_is_not_251073)
            # Assigning a type to the variable 'if_condition_251074' (line 522)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'if_condition_251074', if_condition_251074)
            # SSA begins for if statement (line 522)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to update(...): (line 523)
            # Processing the call keyword arguments (line 523)
            kwargs_251078 = {}
            # Getting the type of 'self' (line 523)
            self_251075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'self', False)
            # Obtaining the member 'toolbar' of a type (line 523)
            toolbar_251076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 16), self_251075, 'toolbar')
            # Obtaining the member 'update' of a type (line 523)
            update_251077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 16), toolbar_251076, 'update')
            # Calling update(args, kwargs) (line 523)
            update_call_result_251079 = invoke(stypy.reporting.localization.Localization(__file__, 523, 16), update_251077, *[], **kwargs_251078)
            
            # SSA join for if statement (line 522)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'notify_axes_change(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'notify_axes_change' in the type store
            # Getting the type of 'stypy_return_type' (line 520)
            stypy_return_type_251080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_251080)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'notify_axes_change'
            return stypy_return_type_251080

        # Assigning a type to the variable 'notify_axes_change' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'notify_axes_change', notify_axes_change)
        
        # Call to add_axobserver(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'notify_axes_change' (line 524)
        notify_axes_change_251085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 42), 'notify_axes_change', False)
        # Processing the call keyword arguments (line 524)
        kwargs_251086 = {}
        # Getting the type of 'self' (line 524)
        self_251081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 524)
        canvas_251082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 8), self_251081, 'canvas')
        # Obtaining the member 'figure' of a type (line 524)
        figure_251083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 8), canvas_251082, 'figure')
        # Obtaining the member 'add_axobserver' of a type (line 524)
        add_axobserver_251084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 8), figure_251083, 'add_axobserver')
        # Calling add_axobserver(args, kwargs) (line 524)
        add_axobserver_call_result_251087 = invoke(stypy.reporting.localization.Localization(__file__, 524, 8), add_axobserver_251084, *[notify_axes_change_251085], **kwargs_251086)
        
        
        # Call to raise_(...): (line 525)
        # Processing the call keyword arguments (line 525)
        kwargs_251091 = {}
        # Getting the type of 'self' (line 525)
        self_251088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 525)
        window_251089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 8), self_251088, 'window')
        # Obtaining the member 'raise_' of a type (line 525)
        raise__251090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 8), window_251089, 'raise_')
        # Calling raise_(args, kwargs) (line 525)
        raise__call_result_251092 = invoke(stypy.reporting.localization.Localization(__file__, 525, 8), raise__251090, *[], **kwargs_251091)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def full_screen_toggle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'full_screen_toggle'
        module_type_store = module_type_store.open_function_context('full_screen_toggle', 527, 4, False)
        # Assigning a type to the variable 'self' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerQT.full_screen_toggle.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerQT.full_screen_toggle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerQT.full_screen_toggle.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerQT.full_screen_toggle.__dict__.__setitem__('stypy_function_name', 'FigureManagerQT.full_screen_toggle')
        FigureManagerQT.full_screen_toggle.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerQT.full_screen_toggle.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerQT.full_screen_toggle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerQT.full_screen_toggle.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerQT.full_screen_toggle.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerQT.full_screen_toggle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerQT.full_screen_toggle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerQT.full_screen_toggle', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to isFullScreen(...): (line 528)
        # Processing the call keyword arguments (line 528)
        kwargs_251096 = {}
        # Getting the type of 'self' (line 528)
        self_251093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 11), 'self', False)
        # Obtaining the member 'window' of a type (line 528)
        window_251094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 11), self_251093, 'window')
        # Obtaining the member 'isFullScreen' of a type (line 528)
        isFullScreen_251095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 11), window_251094, 'isFullScreen')
        # Calling isFullScreen(args, kwargs) (line 528)
        isFullScreen_call_result_251097 = invoke(stypy.reporting.localization.Localization(__file__, 528, 11), isFullScreen_251095, *[], **kwargs_251096)
        
        # Testing the type of an if condition (line 528)
        if_condition_251098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 528, 8), isFullScreen_call_result_251097)
        # Assigning a type to the variable 'if_condition_251098' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'if_condition_251098', if_condition_251098)
        # SSA begins for if statement (line 528)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to showNormal(...): (line 529)
        # Processing the call keyword arguments (line 529)
        kwargs_251102 = {}
        # Getting the type of 'self' (line 529)
        self_251099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'self', False)
        # Obtaining the member 'window' of a type (line 529)
        window_251100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 12), self_251099, 'window')
        # Obtaining the member 'showNormal' of a type (line 529)
        showNormal_251101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 12), window_251100, 'showNormal')
        # Calling showNormal(args, kwargs) (line 529)
        showNormal_call_result_251103 = invoke(stypy.reporting.localization.Localization(__file__, 529, 12), showNormal_251101, *[], **kwargs_251102)
        
        # SSA branch for the else part of an if statement (line 528)
        module_type_store.open_ssa_branch('else')
        
        # Call to showFullScreen(...): (line 531)
        # Processing the call keyword arguments (line 531)
        kwargs_251107 = {}
        # Getting the type of 'self' (line 531)
        self_251104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'self', False)
        # Obtaining the member 'window' of a type (line 531)
        window_251105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 12), self_251104, 'window')
        # Obtaining the member 'showFullScreen' of a type (line 531)
        showFullScreen_251106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 12), window_251105, 'showFullScreen')
        # Calling showFullScreen(args, kwargs) (line 531)
        showFullScreen_call_result_251108 = invoke(stypy.reporting.localization.Localization(__file__, 531, 12), showFullScreen_251106, *[], **kwargs_251107)
        
        # SSA join for if statement (line 528)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'full_screen_toggle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'full_screen_toggle' in the type store
        # Getting the type of 'stypy_return_type' (line 527)
        stypy_return_type_251109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251109)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'full_screen_toggle'
        return stypy_return_type_251109


    @norecursion
    def _widgetclosed(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_widgetclosed'
        module_type_store = module_type_store.open_function_context('_widgetclosed', 533, 4, False)
        # Assigning a type to the variable 'self' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerQT._widgetclosed.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerQT._widgetclosed.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerQT._widgetclosed.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerQT._widgetclosed.__dict__.__setitem__('stypy_function_name', 'FigureManagerQT._widgetclosed')
        FigureManagerQT._widgetclosed.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerQT._widgetclosed.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerQT._widgetclosed.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerQT._widgetclosed.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerQT._widgetclosed.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerQT._widgetclosed.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerQT._widgetclosed.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerQT._widgetclosed', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_widgetclosed', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_widgetclosed(...)' code ##################

        
        # Getting the type of 'self' (line 534)
        self_251110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 11), 'self')
        # Obtaining the member 'window' of a type (line 534)
        window_251111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 11), self_251110, 'window')
        # Obtaining the member '_destroying' of a type (line 534)
        _destroying_251112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 11), window_251111, '_destroying')
        # Testing the type of an if condition (line 534)
        if_condition_251113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 534, 8), _destroying_251112)
        # Assigning a type to the variable 'if_condition_251113' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'if_condition_251113', if_condition_251113)
        # SSA begins for if statement (line 534)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 534)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 536):
        
        # Assigning a Name to a Attribute (line 536):
        # Getting the type of 'True' (line 536)
        True_251114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 34), 'True')
        # Getting the type of 'self' (line 536)
        self_251115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'self')
        # Obtaining the member 'window' of a type (line 536)
        window_251116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), self_251115, 'window')
        # Setting the type of the member '_destroying' of a type (line 536)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), window_251116, '_destroying', True_251114)
        
        
        # SSA begins for try-except statement (line 537)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to destroy(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'self' (line 538)
        self_251119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 24), 'self', False)
        # Obtaining the member 'num' of a type (line 538)
        num_251120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 24), self_251119, 'num')
        # Processing the call keyword arguments (line 538)
        kwargs_251121 = {}
        # Getting the type of 'Gcf' (line 538)
        Gcf_251117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'Gcf', False)
        # Obtaining the member 'destroy' of a type (line 538)
        destroy_251118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 12), Gcf_251117, 'destroy')
        # Calling destroy(args, kwargs) (line 538)
        destroy_call_result_251122 = invoke(stypy.reporting.localization.Localization(__file__, 538, 12), destroy_251118, *[num_251120], **kwargs_251121)
        
        # SSA branch for the except part of a try statement (line 537)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 537)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 537)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_widgetclosed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_widgetclosed' in the type store
        # Getting the type of 'stypy_return_type' (line 533)
        stypy_return_type_251123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251123)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_widgetclosed'
        return stypy_return_type_251123


    @norecursion
    def _get_toolbar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_toolbar'
        module_type_store = module_type_store.open_function_context('_get_toolbar', 545, 4, False)
        # Assigning a type to the variable 'self' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerQT._get_toolbar.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerQT._get_toolbar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerQT._get_toolbar.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerQT._get_toolbar.__dict__.__setitem__('stypy_function_name', 'FigureManagerQT._get_toolbar')
        FigureManagerQT._get_toolbar.__dict__.__setitem__('stypy_param_names_list', ['canvas', 'parent'])
        FigureManagerQT._get_toolbar.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerQT._get_toolbar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerQT._get_toolbar.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerQT._get_toolbar.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerQT._get_toolbar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerQT._get_toolbar.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerQT._get_toolbar', ['canvas', 'parent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_toolbar', localization, ['canvas', 'parent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_toolbar(...)' code ##################

        
        
        
        # Obtaining the type of the subscript
        unicode_251124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 31), 'unicode', u'toolbar')
        # Getting the type of 'matplotlib' (line 548)
        matplotlib_251125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 11), 'matplotlib')
        # Obtaining the member 'rcParams' of a type (line 548)
        rcParams_251126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 11), matplotlib_251125, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 548)
        getitem___251127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 11), rcParams_251126, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 548)
        subscript_call_result_251128 = invoke(stypy.reporting.localization.Localization(__file__, 548, 11), getitem___251127, unicode_251124)
        
        unicode_251129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 45), 'unicode', u'toolbar2')
        # Applying the binary operator '==' (line 548)
        result_eq_251130 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 11), '==', subscript_call_result_251128, unicode_251129)
        
        # Testing the type of an if condition (line 548)
        if_condition_251131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 8), result_eq_251130)
        # Assigning a type to the variable 'if_condition_251131' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'if_condition_251131', if_condition_251131)
        # SSA begins for if statement (line 548)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 549):
        
        # Assigning a Call to a Name (line 549):
        
        # Call to NavigationToolbar2QT(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'canvas' (line 549)
        canvas_251133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 43), 'canvas', False)
        # Getting the type of 'parent' (line 549)
        parent_251134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 51), 'parent', False)
        # Getting the type of 'False' (line 549)
        False_251135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 59), 'False', False)
        # Processing the call keyword arguments (line 549)
        kwargs_251136 = {}
        # Getting the type of 'NavigationToolbar2QT' (line 549)
        NavigationToolbar2QT_251132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 22), 'NavigationToolbar2QT', False)
        # Calling NavigationToolbar2QT(args, kwargs) (line 549)
        NavigationToolbar2QT_call_result_251137 = invoke(stypy.reporting.localization.Localization(__file__, 549, 22), NavigationToolbar2QT_251132, *[canvas_251133, parent_251134, False_251135], **kwargs_251136)
        
        # Assigning a type to the variable 'toolbar' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'toolbar', NavigationToolbar2QT_call_result_251137)
        # SSA branch for the else part of an if statement (line 548)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 551):
        
        # Assigning a Name to a Name (line 551):
        # Getting the type of 'None' (line 551)
        None_251138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 22), 'None')
        # Assigning a type to the variable 'toolbar' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'toolbar', None_251138)
        # SSA join for if statement (line 548)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'toolbar' (line 552)
        toolbar_251139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 15), 'toolbar')
        # Assigning a type to the variable 'stypy_return_type' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'stypy_return_type', toolbar_251139)
        
        # ################# End of '_get_toolbar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_toolbar' in the type store
        # Getting the type of 'stypy_return_type' (line 545)
        stypy_return_type_251140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251140)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_toolbar'
        return stypy_return_type_251140


    @norecursion
    def resize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'resize'
        module_type_store = module_type_store.open_function_context('resize', 554, 4, False)
        # Assigning a type to the variable 'self' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerQT.resize.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerQT.resize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerQT.resize.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerQT.resize.__dict__.__setitem__('stypy_function_name', 'FigureManagerQT.resize')
        FigureManagerQT.resize.__dict__.__setitem__('stypy_param_names_list', ['width', 'height'])
        FigureManagerQT.resize.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerQT.resize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerQT.resize.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerQT.resize.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerQT.resize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerQT.resize.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerQT.resize', ['width', 'height'], None, None, defaults, varargs, kwargs)

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

        unicode_251141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 8), 'unicode', u'set the canvas size in pixels')
        
        # Call to resize(...): (line 556)
        # Processing the call arguments (line 556)
        # Getting the type of 'width' (line 556)
        width_251145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 27), 'width', False)
        # Getting the type of 'height' (line 556)
        height_251146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 34), 'height', False)
        # Getting the type of 'self' (line 556)
        self_251147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 43), 'self', False)
        # Obtaining the member '_status_and_tool_height' of a type (line 556)
        _status_and_tool_height_251148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 43), self_251147, '_status_and_tool_height')
        # Applying the binary operator '+' (line 556)
        result_add_251149 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 34), '+', height_251146, _status_and_tool_height_251148)
        
        # Processing the call keyword arguments (line 556)
        kwargs_251150 = {}
        # Getting the type of 'self' (line 556)
        self_251142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 556)
        window_251143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 8), self_251142, 'window')
        # Obtaining the member 'resize' of a type (line 556)
        resize_251144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 8), window_251143, 'resize')
        # Calling resize(args, kwargs) (line 556)
        resize_call_result_251151 = invoke(stypy.reporting.localization.Localization(__file__, 556, 8), resize_251144, *[width_251145, result_add_251149], **kwargs_251150)
        
        
        # ################# End of 'resize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'resize' in the type store
        # Getting the type of 'stypy_return_type' (line 554)
        stypy_return_type_251152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251152)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'resize'
        return stypy_return_type_251152


    @norecursion
    def show(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'show'
        module_type_store = module_type_store.open_function_context('show', 558, 4, False)
        # Assigning a type to the variable 'self' (line 559)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerQT.show.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerQT.show.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerQT.show.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerQT.show.__dict__.__setitem__('stypy_function_name', 'FigureManagerQT.show')
        FigureManagerQT.show.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerQT.show.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerQT.show.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerQT.show.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerQT.show.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerQT.show.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerQT.show.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerQT.show', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to show(...): (line 559)
        # Processing the call keyword arguments (line 559)
        kwargs_251156 = {}
        # Getting the type of 'self' (line 559)
        self_251153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 559)
        window_251154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 8), self_251153, 'window')
        # Obtaining the member 'show' of a type (line 559)
        show_251155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 8), window_251154, 'show')
        # Calling show(args, kwargs) (line 559)
        show_call_result_251157 = invoke(stypy.reporting.localization.Localization(__file__, 559, 8), show_251155, *[], **kwargs_251156)
        
        
        # Call to activateWindow(...): (line 560)
        # Processing the call keyword arguments (line 560)
        kwargs_251161 = {}
        # Getting the type of 'self' (line 560)
        self_251158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 560)
        window_251159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 8), self_251158, 'window')
        # Obtaining the member 'activateWindow' of a type (line 560)
        activateWindow_251160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 560, 8), window_251159, 'activateWindow')
        # Calling activateWindow(args, kwargs) (line 560)
        activateWindow_call_result_251162 = invoke(stypy.reporting.localization.Localization(__file__, 560, 8), activateWindow_251160, *[], **kwargs_251161)
        
        
        # Call to raise_(...): (line 561)
        # Processing the call keyword arguments (line 561)
        kwargs_251166 = {}
        # Getting the type of 'self' (line 561)
        self_251163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 561)
        window_251164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 8), self_251163, 'window')
        # Obtaining the member 'raise_' of a type (line 561)
        raise__251165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 8), window_251164, 'raise_')
        # Calling raise_(args, kwargs) (line 561)
        raise__call_result_251167 = invoke(stypy.reporting.localization.Localization(__file__, 561, 8), raise__251165, *[], **kwargs_251166)
        
        
        # ################# End of 'show(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'show' in the type store
        # Getting the type of 'stypy_return_type' (line 558)
        stypy_return_type_251168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251168)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'show'
        return stypy_return_type_251168


    @norecursion
    def destroy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'destroy'
        module_type_store = module_type_store.open_function_context('destroy', 563, 4, False)
        # Assigning a type to the variable 'self' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerQT.destroy.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerQT.destroy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerQT.destroy.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerQT.destroy.__dict__.__setitem__('stypy_function_name', 'FigureManagerQT.destroy')
        FigureManagerQT.destroy.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerQT.destroy.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureManagerQT.destroy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerQT.destroy.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerQT.destroy.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerQT.destroy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerQT.destroy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerQT.destroy', [], 'args', None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 565)
        
        # Call to instance(...): (line 565)
        # Processing the call keyword arguments (line 565)
        kwargs_251172 = {}
        # Getting the type of 'QtWidgets' (line 565)
        QtWidgets_251169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 11), 'QtWidgets', False)
        # Obtaining the member 'QApplication' of a type (line 565)
        QApplication_251170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 11), QtWidgets_251169, 'QApplication')
        # Obtaining the member 'instance' of a type (line 565)
        instance_251171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 11), QApplication_251170, 'instance')
        # Calling instance(args, kwargs) (line 565)
        instance_call_result_251173 = invoke(stypy.reporting.localization.Localization(__file__, 565, 11), instance_251171, *[], **kwargs_251172)
        
        # Getting the type of 'None' (line 565)
        None_251174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 48), 'None')
        
        (may_be_251175, more_types_in_union_251176) = may_be_none(instance_call_result_251173, None_251174)

        if may_be_251175:

            if more_types_in_union_251176:
                # Runtime conditional SSA (line 565)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 566)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_251176:
                # SSA join for if statement (line 565)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 567)
        self_251177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 11), 'self')
        # Obtaining the member 'window' of a type (line 567)
        window_251178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 11), self_251177, 'window')
        # Obtaining the member '_destroying' of a type (line 567)
        _destroying_251179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 11), window_251178, '_destroying')
        # Testing the type of an if condition (line 567)
        if_condition_251180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 567, 8), _destroying_251179)
        # Assigning a type to the variable 'if_condition_251180' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'if_condition_251180', if_condition_251180)
        # SSA begins for if statement (line 567)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 567)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 569):
        
        # Assigning a Name to a Attribute (line 569):
        # Getting the type of 'True' (line 569)
        True_251181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 34), 'True')
        # Getting the type of 'self' (line 569)
        self_251182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'self')
        # Obtaining the member 'window' of a type (line 569)
        window_251183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 8), self_251182, 'window')
        # Setting the type of the member '_destroying' of a type (line 569)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 8), window_251183, '_destroying', True_251181)
        
        # Call to connect(...): (line 570)
        # Processing the call arguments (line 570)
        # Getting the type of 'self' (line 570)
        self_251188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 38), 'self', False)
        # Obtaining the member '_widgetclosed' of a type (line 570)
        _widgetclosed_251189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 38), self_251188, '_widgetclosed')
        # Processing the call keyword arguments (line 570)
        kwargs_251190 = {}
        # Getting the type of 'self' (line 570)
        self_251184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 570)
        window_251185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), self_251184, 'window')
        # Obtaining the member 'destroyed' of a type (line 570)
        destroyed_251186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), window_251185, 'destroyed')
        # Obtaining the member 'connect' of a type (line 570)
        connect_251187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), destroyed_251186, 'connect')
        # Calling connect(args, kwargs) (line 570)
        connect_call_result_251191 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), connect_251187, *[_widgetclosed_251189], **kwargs_251190)
        
        
        # Getting the type of 'self' (line 571)
        self_251192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 11), 'self')
        # Obtaining the member 'toolbar' of a type (line 571)
        toolbar_251193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 11), self_251192, 'toolbar')
        # Testing the type of an if condition (line 571)
        if_condition_251194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 571, 8), toolbar_251193)
        # Assigning a type to the variable 'if_condition_251194' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'if_condition_251194', if_condition_251194)
        # SSA begins for if statement (line 571)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to destroy(...): (line 572)
        # Processing the call keyword arguments (line 572)
        kwargs_251198 = {}
        # Getting the type of 'self' (line 572)
        self_251195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'self', False)
        # Obtaining the member 'toolbar' of a type (line 572)
        toolbar_251196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 12), self_251195, 'toolbar')
        # Obtaining the member 'destroy' of a type (line 572)
        destroy_251197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 12), toolbar_251196, 'destroy')
        # Calling destroy(args, kwargs) (line 572)
        destroy_call_result_251199 = invoke(stypy.reporting.localization.Localization(__file__, 572, 12), destroy_251197, *[], **kwargs_251198)
        
        # SSA join for if statement (line 571)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to close(...): (line 573)
        # Processing the call keyword arguments (line 573)
        kwargs_251203 = {}
        # Getting the type of 'self' (line 573)
        self_251200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 573)
        window_251201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 8), self_251200, 'window')
        # Obtaining the member 'close' of a type (line 573)
        close_251202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 8), window_251201, 'close')
        # Calling close(args, kwargs) (line 573)
        close_call_result_251204 = invoke(stypy.reporting.localization.Localization(__file__, 573, 8), close_251202, *[], **kwargs_251203)
        
        
        # ################# End of 'destroy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'destroy' in the type store
        # Getting the type of 'stypy_return_type' (line 563)
        stypy_return_type_251205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251205)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'destroy'
        return stypy_return_type_251205


    @norecursion
    def get_window_title(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_window_title'
        module_type_store = module_type_store.open_function_context('get_window_title', 575, 4, False)
        # Assigning a type to the variable 'self' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerQT.get_window_title.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerQT.get_window_title.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerQT.get_window_title.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerQT.get_window_title.__dict__.__setitem__('stypy_function_name', 'FigureManagerQT.get_window_title')
        FigureManagerQT.get_window_title.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerQT.get_window_title.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerQT.get_window_title.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerQT.get_window_title.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerQT.get_window_title.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerQT.get_window_title.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerQT.get_window_title.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerQT.get_window_title', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to text_type(...): (line 576)
        # Processing the call arguments (line 576)
        
        # Call to windowTitle(...): (line 576)
        # Processing the call keyword arguments (line 576)
        kwargs_251211 = {}
        # Getting the type of 'self' (line 576)
        self_251208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 29), 'self', False)
        # Obtaining the member 'window' of a type (line 576)
        window_251209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 29), self_251208, 'window')
        # Obtaining the member 'windowTitle' of a type (line 576)
        windowTitle_251210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 29), window_251209, 'windowTitle')
        # Calling windowTitle(args, kwargs) (line 576)
        windowTitle_call_result_251212 = invoke(stypy.reporting.localization.Localization(__file__, 576, 29), windowTitle_251210, *[], **kwargs_251211)
        
        # Processing the call keyword arguments (line 576)
        kwargs_251213 = {}
        # Getting the type of 'six' (line 576)
        six_251206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 15), 'six', False)
        # Obtaining the member 'text_type' of a type (line 576)
        text_type_251207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 15), six_251206, 'text_type')
        # Calling text_type(args, kwargs) (line 576)
        text_type_call_result_251214 = invoke(stypy.reporting.localization.Localization(__file__, 576, 15), text_type_251207, *[windowTitle_call_result_251212], **kwargs_251213)
        
        # Assigning a type to the variable 'stypy_return_type' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'stypy_return_type', text_type_call_result_251214)
        
        # ################# End of 'get_window_title(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_window_title' in the type store
        # Getting the type of 'stypy_return_type' (line 575)
        stypy_return_type_251215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251215)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_window_title'
        return stypy_return_type_251215


    @norecursion
    def set_window_title(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_window_title'
        module_type_store = module_type_store.open_function_context('set_window_title', 578, 4, False)
        # Assigning a type to the variable 'self' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerQT.set_window_title.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerQT.set_window_title.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerQT.set_window_title.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerQT.set_window_title.__dict__.__setitem__('stypy_function_name', 'FigureManagerQT.set_window_title')
        FigureManagerQT.set_window_title.__dict__.__setitem__('stypy_param_names_list', ['title'])
        FigureManagerQT.set_window_title.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerQT.set_window_title.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerQT.set_window_title.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerQT.set_window_title.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerQT.set_window_title.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerQT.set_window_title.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerQT.set_window_title', ['title'], None, None, defaults, varargs, kwargs)

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

        
        # Call to setWindowTitle(...): (line 579)
        # Processing the call arguments (line 579)
        # Getting the type of 'title' (line 579)
        title_251219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 35), 'title', False)
        # Processing the call keyword arguments (line 579)
        kwargs_251220 = {}
        # Getting the type of 'self' (line 579)
        self_251216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'self', False)
        # Obtaining the member 'window' of a type (line 579)
        window_251217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 8), self_251216, 'window')
        # Obtaining the member 'setWindowTitle' of a type (line 579)
        setWindowTitle_251218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 8), window_251217, 'setWindowTitle')
        # Calling setWindowTitle(args, kwargs) (line 579)
        setWindowTitle_call_result_251221 = invoke(stypy.reporting.localization.Localization(__file__, 579, 8), setWindowTitle_251218, *[title_251219], **kwargs_251220)
        
        
        # ################# End of 'set_window_title(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_window_title' in the type store
        # Getting the type of 'stypy_return_type' (line 578)
        stypy_return_type_251222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251222)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_window_title'
        return stypy_return_type_251222


# Assigning a type to the variable 'FigureManagerQT' (line 455)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 0), 'FigureManagerQT', FigureManagerQT)
# Declaration of the 'NavigationToolbar2QT' class
# Getting the type of 'NavigationToolbar2' (line 582)
NavigationToolbar2_251223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 27), 'NavigationToolbar2')
# Getting the type of 'QtWidgets' (line 582)
QtWidgets_251224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 47), 'QtWidgets')
# Obtaining the member 'QToolBar' of a type (line 582)
QToolBar_251225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 47), QtWidgets_251224, 'QToolBar')

class NavigationToolbar2QT(NavigationToolbar2_251223, QToolBar_251225, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 585)
        True_251226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 51), 'True')
        defaults = [True_251226]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 585, 4, False)
        # Assigning a type to the variable 'self' (line 586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT.__init__', ['canvas', 'parent', 'coordinates'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['canvas', 'parent', 'coordinates'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_251227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 8), 'unicode', u' coordinates: should we show the coordinates on the right? ')
        
        # Assigning a Name to a Attribute (line 587):
        
        # Assigning a Name to a Attribute (line 587):
        # Getting the type of 'canvas' (line 587)
        canvas_251228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 22), 'canvas')
        # Getting the type of 'self' (line 587)
        self_251229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'self')
        # Setting the type of the member 'canvas' of a type (line 587)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 8), self_251229, 'canvas', canvas_251228)
        
        # Assigning a Name to a Attribute (line 588):
        
        # Assigning a Name to a Attribute (line 588):
        # Getting the type of 'parent' (line 588)
        parent_251230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 22), 'parent')
        # Getting the type of 'self' (line 588)
        self_251231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 8), 'self')
        # Setting the type of the member 'parent' of a type (line 588)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 8), self_251231, 'parent', parent_251230)
        
        # Assigning a Name to a Attribute (line 589):
        
        # Assigning a Name to a Attribute (line 589):
        # Getting the type of 'coordinates' (line 589)
        coordinates_251232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 27), 'coordinates')
        # Getting the type of 'self' (line 589)
        self_251233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'self')
        # Setting the type of the member 'coordinates' of a type (line 589)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 8), self_251233, 'coordinates', coordinates_251232)
        
        # Assigning a Dict to a Attribute (line 590):
        
        # Assigning a Dict to a Attribute (line 590):
        
        # Obtaining an instance of the builtin type 'dict' (line 590)
        dict_251234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 590)
        
        # Getting the type of 'self' (line 590)
        self_251235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'self')
        # Setting the type of the member '_actions' of a type (line 590)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 8), self_251235, '_actions', dict_251234)
        unicode_251236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 8), 'unicode', u'A mapping of toolitem method names to their QActions')
        
        # Call to __init__(...): (line 593)
        # Processing the call arguments (line 593)
        # Getting the type of 'self' (line 593)
        self_251240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 36), 'self', False)
        # Getting the type of 'parent' (line 593)
        parent_251241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 42), 'parent', False)
        # Processing the call keyword arguments (line 593)
        kwargs_251242 = {}
        # Getting the type of 'QtWidgets' (line 593)
        QtWidgets_251237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'QtWidgets', False)
        # Obtaining the member 'QToolBar' of a type (line 593)
        QToolBar_251238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), QtWidgets_251237, 'QToolBar')
        # Obtaining the member '__init__' of a type (line 593)
        init___251239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), QToolBar_251238, '__init__')
        # Calling __init__(args, kwargs) (line 593)
        init___call_result_251243 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), init___251239, *[self_251240, parent_251241], **kwargs_251242)
        
        
        # Call to __init__(...): (line 594)
        # Processing the call arguments (line 594)
        # Getting the type of 'self' (line 594)
        self_251246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 36), 'self', False)
        # Getting the type of 'canvas' (line 594)
        canvas_251247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 42), 'canvas', False)
        # Processing the call keyword arguments (line 594)
        kwargs_251248 = {}
        # Getting the type of 'NavigationToolbar2' (line 594)
        NavigationToolbar2_251244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'NavigationToolbar2', False)
        # Obtaining the member '__init__' of a type (line 594)
        init___251245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 8), NavigationToolbar2_251244, '__init__')
        # Calling __init__(args, kwargs) (line 594)
        init___call_result_251249 = invoke(stypy.reporting.localization.Localization(__file__, 594, 8), init___251245, *[self_251246, canvas_251247], **kwargs_251248)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _icon(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_icon'
        module_type_store = module_type_store.open_function_context('_icon', 596, 4, False)
        # Assigning a type to the variable 'self' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT._icon.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT._icon.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT._icon.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT._icon.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT._icon')
        NavigationToolbar2QT._icon.__dict__.__setitem__('stypy_param_names_list', ['name'])
        NavigationToolbar2QT._icon.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2QT._icon.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT._icon.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT._icon.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT._icon.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT._icon.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT._icon', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_icon', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_icon(...)' code ##################

        
        
        # Call to is_pyqt5(...): (line 597)
        # Processing the call keyword arguments (line 597)
        kwargs_251251 = {}
        # Getting the type of 'is_pyqt5' (line 597)
        is_pyqt5_251250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 11), 'is_pyqt5', False)
        # Calling is_pyqt5(args, kwargs) (line 597)
        is_pyqt5_call_result_251252 = invoke(stypy.reporting.localization.Localization(__file__, 597, 11), is_pyqt5_251250, *[], **kwargs_251251)
        
        # Testing the type of an if condition (line 597)
        if_condition_251253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 8), is_pyqt5_call_result_251252)
        # Assigning a type to the variable 'if_condition_251253' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'if_condition_251253', if_condition_251253)
        # SSA begins for if statement (line 597)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 598):
        
        # Assigning a Call to a Name (line 598):
        
        # Call to replace(...): (line 598)
        # Processing the call arguments (line 598)
        unicode_251256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 32), 'unicode', u'.png')
        unicode_251257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 40), 'unicode', u'_large.png')
        # Processing the call keyword arguments (line 598)
        kwargs_251258 = {}
        # Getting the type of 'name' (line 598)
        name_251254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 19), 'name', False)
        # Obtaining the member 'replace' of a type (line 598)
        replace_251255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 19), name_251254, 'replace')
        # Calling replace(args, kwargs) (line 598)
        replace_call_result_251259 = invoke(stypy.reporting.localization.Localization(__file__, 598, 19), replace_251255, *[unicode_251256, unicode_251257], **kwargs_251258)
        
        # Assigning a type to the variable 'name' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 12), 'name', replace_call_result_251259)
        # SSA join for if statement (line 597)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 599):
        
        # Assigning a Call to a Name (line 599):
        
        # Call to QPixmap(...): (line 599)
        # Processing the call arguments (line 599)
        
        # Call to join(...): (line 599)
        # Processing the call arguments (line 599)
        # Getting the type of 'self' (line 599)
        self_251265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 40), 'self', False)
        # Obtaining the member 'basedir' of a type (line 599)
        basedir_251266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 40), self_251265, 'basedir')
        # Getting the type of 'name' (line 599)
        name_251267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 54), 'name', False)
        # Processing the call keyword arguments (line 599)
        kwargs_251268 = {}
        # Getting the type of 'os' (line 599)
        os_251262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 599)
        path_251263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 27), os_251262, 'path')
        # Obtaining the member 'join' of a type (line 599)
        join_251264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 27), path_251263, 'join')
        # Calling join(args, kwargs) (line 599)
        join_call_result_251269 = invoke(stypy.reporting.localization.Localization(__file__, 599, 27), join_251264, *[basedir_251266, name_251267], **kwargs_251268)
        
        # Processing the call keyword arguments (line 599)
        kwargs_251270 = {}
        # Getting the type of 'QtGui' (line 599)
        QtGui_251260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 13), 'QtGui', False)
        # Obtaining the member 'QPixmap' of a type (line 599)
        QPixmap_251261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 13), QtGui_251260, 'QPixmap')
        # Calling QPixmap(args, kwargs) (line 599)
        QPixmap_call_result_251271 = invoke(stypy.reporting.localization.Localization(__file__, 599, 13), QPixmap_251261, *[join_call_result_251269], **kwargs_251270)
        
        # Assigning a type to the variable 'pm' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'pm', QPixmap_call_result_251271)
        
        # Type idiom detected: calculating its left and rigth part (line 600)
        unicode_251272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 23), 'unicode', u'setDevicePixelRatio')
        # Getting the type of 'pm' (line 600)
        pm_251273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 19), 'pm')
        
        (may_be_251274, more_types_in_union_251275) = may_provide_member(unicode_251272, pm_251273)

        if may_be_251274:

            if more_types_in_union_251275:
                # Runtime conditional SSA (line 600)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'pm' (line 600)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'pm', remove_not_member_provider_from_union(pm_251273, u'setDevicePixelRatio'))
            
            # Call to setDevicePixelRatio(...): (line 601)
            # Processing the call arguments (line 601)
            # Getting the type of 'self' (line 601)
            self_251278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 35), 'self', False)
            # Obtaining the member 'canvas' of a type (line 601)
            canvas_251279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 35), self_251278, 'canvas')
            # Obtaining the member '_dpi_ratio' of a type (line 601)
            _dpi_ratio_251280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 35), canvas_251279, '_dpi_ratio')
            # Processing the call keyword arguments (line 601)
            kwargs_251281 = {}
            # Getting the type of 'pm' (line 601)
            pm_251276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'pm', False)
            # Obtaining the member 'setDevicePixelRatio' of a type (line 601)
            setDevicePixelRatio_251277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 12), pm_251276, 'setDevicePixelRatio')
            # Calling setDevicePixelRatio(args, kwargs) (line 601)
            setDevicePixelRatio_call_result_251282 = invoke(stypy.reporting.localization.Localization(__file__, 601, 12), setDevicePixelRatio_251277, *[_dpi_ratio_251280], **kwargs_251281)
            

            if more_types_in_union_251275:
                # SSA join for if statement (line 600)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to QIcon(...): (line 602)
        # Processing the call arguments (line 602)
        # Getting the type of 'pm' (line 602)
        pm_251285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 27), 'pm', False)
        # Processing the call keyword arguments (line 602)
        kwargs_251286 = {}
        # Getting the type of 'QtGui' (line 602)
        QtGui_251283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 15), 'QtGui', False)
        # Obtaining the member 'QIcon' of a type (line 602)
        QIcon_251284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 15), QtGui_251283, 'QIcon')
        # Calling QIcon(args, kwargs) (line 602)
        QIcon_call_result_251287 = invoke(stypy.reporting.localization.Localization(__file__, 602, 15), QIcon_251284, *[pm_251285], **kwargs_251286)
        
        # Assigning a type to the variable 'stypy_return_type' (line 602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'stypy_return_type', QIcon_call_result_251287)
        
        # ################# End of '_icon(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_icon' in the type store
        # Getting the type of 'stypy_return_type' (line 596)
        stypy_return_type_251288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251288)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_icon'
        return stypy_return_type_251288


    @norecursion
    def _init_toolbar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init_toolbar'
        module_type_store = module_type_store.open_function_context('_init_toolbar', 604, 4, False)
        # Assigning a type to the variable 'self' (line 605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT._init_toolbar.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT._init_toolbar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT._init_toolbar.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT._init_toolbar.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT._init_toolbar')
        NavigationToolbar2QT._init_toolbar.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2QT._init_toolbar.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2QT._init_toolbar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT._init_toolbar.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT._init_toolbar.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT._init_toolbar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT._init_toolbar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT._init_toolbar', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 605):
        
        # Assigning a Call to a Attribute (line 605):
        
        # Call to join(...): (line 605)
        # Processing the call arguments (line 605)
        
        # Obtaining the type of the subscript
        unicode_251292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 56), 'unicode', u'datapath')
        # Getting the type of 'matplotlib' (line 605)
        matplotlib_251293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 36), 'matplotlib', False)
        # Obtaining the member 'rcParams' of a type (line 605)
        rcParams_251294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 36), matplotlib_251293, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 605)
        getitem___251295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 36), rcParams_251294, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 605)
        subscript_call_result_251296 = invoke(stypy.reporting.localization.Localization(__file__, 605, 36), getitem___251295, unicode_251292)
        
        unicode_251297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 69), 'unicode', u'images')
        # Processing the call keyword arguments (line 605)
        kwargs_251298 = {}
        # Getting the type of 'os' (line 605)
        os_251289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 605)
        path_251290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 23), os_251289, 'path')
        # Obtaining the member 'join' of a type (line 605)
        join_251291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 23), path_251290, 'join')
        # Calling join(args, kwargs) (line 605)
        join_call_result_251299 = invoke(stypy.reporting.localization.Localization(__file__, 605, 23), join_251291, *[subscript_call_result_251296, unicode_251297], **kwargs_251298)
        
        # Getting the type of 'self' (line 605)
        self_251300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'self')
        # Setting the type of the member 'basedir' of a type (line 605)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 8), self_251300, 'basedir', join_call_result_251299)
        
        # Getting the type of 'self' (line 607)
        self_251301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 56), 'self')
        # Obtaining the member 'toolitems' of a type (line 607)
        toolitems_251302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 56), self_251301, 'toolitems')
        # Testing the type of a for loop iterable (line 607)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 607, 8), toolitems_251302)
        # Getting the type of the for loop variable (line 607)
        for_loop_var_251303 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 607, 8), toolitems_251302)
        # Assigning a type to the variable 'text' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'text', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 8), for_loop_var_251303))
        # Assigning a type to the variable 'tooltip_text' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'tooltip_text', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 8), for_loop_var_251303))
        # Assigning a type to the variable 'image_file' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'image_file', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 8), for_loop_var_251303))
        # Assigning a type to the variable 'callback' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'callback', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 607, 8), for_loop_var_251303))
        # SSA begins for a for statement (line 607)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 608)
        # Getting the type of 'text' (line 608)
        text_251304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 15), 'text')
        # Getting the type of 'None' (line 608)
        None_251305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 23), 'None')
        
        (may_be_251306, more_types_in_union_251307) = may_be_none(text_251304, None_251305)

        if may_be_251306:

            if more_types_in_union_251307:
                # Runtime conditional SSA (line 608)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to addSeparator(...): (line 609)
            # Processing the call keyword arguments (line 609)
            kwargs_251310 = {}
            # Getting the type of 'self' (line 609)
            self_251308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 16), 'self', False)
            # Obtaining the member 'addSeparator' of a type (line 609)
            addSeparator_251309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 16), self_251308, 'addSeparator')
            # Calling addSeparator(args, kwargs) (line 609)
            addSeparator_call_result_251311 = invoke(stypy.reporting.localization.Localization(__file__, 609, 16), addSeparator_251309, *[], **kwargs_251310)
            

            if more_types_in_union_251307:
                # Runtime conditional SSA for else branch (line 608)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_251306) or more_types_in_union_251307):
            
            # Assigning a Call to a Name (line 611):
            
            # Assigning a Call to a Name (line 611):
            
            # Call to addAction(...): (line 611)
            # Processing the call arguments (line 611)
            
            # Call to _icon(...): (line 611)
            # Processing the call arguments (line 611)
            # Getting the type of 'image_file' (line 611)
            image_file_251316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 46), 'image_file', False)
            unicode_251317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 59), 'unicode', u'.png')
            # Applying the binary operator '+' (line 611)
            result_add_251318 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 46), '+', image_file_251316, unicode_251317)
            
            # Processing the call keyword arguments (line 611)
            kwargs_251319 = {}
            # Getting the type of 'self' (line 611)
            self_251314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 35), 'self', False)
            # Obtaining the member '_icon' of a type (line 611)
            _icon_251315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 35), self_251314, '_icon')
            # Calling _icon(args, kwargs) (line 611)
            _icon_call_result_251320 = invoke(stypy.reporting.localization.Localization(__file__, 611, 35), _icon_251315, *[result_add_251318], **kwargs_251319)
            
            # Getting the type of 'text' (line 612)
            text_251321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 35), 'text', False)
            
            # Call to getattr(...): (line 612)
            # Processing the call arguments (line 612)
            # Getting the type of 'self' (line 612)
            self_251323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 49), 'self', False)
            # Getting the type of 'callback' (line 612)
            callback_251324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 55), 'callback', False)
            # Processing the call keyword arguments (line 612)
            kwargs_251325 = {}
            # Getting the type of 'getattr' (line 612)
            getattr_251322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 41), 'getattr', False)
            # Calling getattr(args, kwargs) (line 612)
            getattr_call_result_251326 = invoke(stypy.reporting.localization.Localization(__file__, 612, 41), getattr_251322, *[self_251323, callback_251324], **kwargs_251325)
            
            # Processing the call keyword arguments (line 611)
            kwargs_251327 = {}
            # Getting the type of 'self' (line 611)
            self_251312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 20), 'self', False)
            # Obtaining the member 'addAction' of a type (line 611)
            addAction_251313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 20), self_251312, 'addAction')
            # Calling addAction(args, kwargs) (line 611)
            addAction_call_result_251328 = invoke(stypy.reporting.localization.Localization(__file__, 611, 20), addAction_251313, *[_icon_call_result_251320, text_251321, getattr_call_result_251326], **kwargs_251327)
            
            # Assigning a type to the variable 'a' (line 611)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 16), 'a', addAction_call_result_251328)
            
            # Assigning a Name to a Subscript (line 613):
            
            # Assigning a Name to a Subscript (line 613):
            # Getting the type of 'a' (line 613)
            a_251329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 42), 'a')
            # Getting the type of 'self' (line 613)
            self_251330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 16), 'self')
            # Obtaining the member '_actions' of a type (line 613)
            _actions_251331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 16), self_251330, '_actions')
            # Getting the type of 'callback' (line 613)
            callback_251332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 30), 'callback')
            # Storing an element on a container (line 613)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 16), _actions_251331, (callback_251332, a_251329))
            
            
            # Getting the type of 'callback' (line 614)
            callback_251333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 19), 'callback')
            
            # Obtaining an instance of the builtin type 'list' (line 614)
            list_251334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 614)
            # Adding element type (line 614)
            unicode_251335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 32), 'unicode', u'zoom')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 31), list_251334, unicode_251335)
            # Adding element type (line 614)
            unicode_251336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 40), 'unicode', u'pan')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 31), list_251334, unicode_251336)
            
            # Applying the binary operator 'in' (line 614)
            result_contains_251337 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 19), 'in', callback_251333, list_251334)
            
            # Testing the type of an if condition (line 614)
            if_condition_251338 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 614, 16), result_contains_251337)
            # Assigning a type to the variable 'if_condition_251338' (line 614)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 16), 'if_condition_251338', if_condition_251338)
            # SSA begins for if statement (line 614)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to setCheckable(...): (line 615)
            # Processing the call arguments (line 615)
            # Getting the type of 'True' (line 615)
            True_251341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 35), 'True', False)
            # Processing the call keyword arguments (line 615)
            kwargs_251342 = {}
            # Getting the type of 'a' (line 615)
            a_251339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 20), 'a', False)
            # Obtaining the member 'setCheckable' of a type (line 615)
            setCheckable_251340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 20), a_251339, 'setCheckable')
            # Calling setCheckable(args, kwargs) (line 615)
            setCheckable_call_result_251343 = invoke(stypy.reporting.localization.Localization(__file__, 615, 20), setCheckable_251340, *[True_251341], **kwargs_251342)
            
            # SSA join for if statement (line 614)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Type idiom detected: calculating its left and rigth part (line 616)
            # Getting the type of 'tooltip_text' (line 616)
            tooltip_text_251344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'tooltip_text')
            # Getting the type of 'None' (line 616)
            None_251345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 39), 'None')
            
            (may_be_251346, more_types_in_union_251347) = may_not_be_none(tooltip_text_251344, None_251345)

            if may_be_251346:

                if more_types_in_union_251347:
                    # Runtime conditional SSA (line 616)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to setToolTip(...): (line 617)
                # Processing the call arguments (line 617)
                # Getting the type of 'tooltip_text' (line 617)
                tooltip_text_251350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 33), 'tooltip_text', False)
                # Processing the call keyword arguments (line 617)
                kwargs_251351 = {}
                # Getting the type of 'a' (line 617)
                a_251348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 20), 'a', False)
                # Obtaining the member 'setToolTip' of a type (line 617)
                setToolTip_251349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 20), a_251348, 'setToolTip')
                # Calling setToolTip(args, kwargs) (line 617)
                setToolTip_call_result_251352 = invoke(stypy.reporting.localization.Localization(__file__, 617, 20), setToolTip_251349, *[tooltip_text_251350], **kwargs_251351)
                

                if more_types_in_union_251347:
                    # SSA join for if statement (line 616)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Getting the type of 'text' (line 618)
            text_251353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 19), 'text')
            unicode_251354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 27), 'unicode', u'Subplots')
            # Applying the binary operator '==' (line 618)
            result_eq_251355 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 19), '==', text_251353, unicode_251354)
            
            # Testing the type of an if condition (line 618)
            if_condition_251356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 618, 16), result_eq_251355)
            # Assigning a type to the variable 'if_condition_251356' (line 618)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 16), 'if_condition_251356', if_condition_251356)
            # SSA begins for if statement (line 618)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 619):
            
            # Assigning a Call to a Name (line 619):
            
            # Call to addAction(...): (line 619)
            # Processing the call arguments (line 619)
            
            # Call to _icon(...): (line 619)
            # Processing the call arguments (line 619)
            unicode_251361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 50), 'unicode', u'qt4_editor_options.png')
            # Processing the call keyword arguments (line 619)
            kwargs_251362 = {}
            # Getting the type of 'self' (line 619)
            self_251359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 39), 'self', False)
            # Obtaining the member '_icon' of a type (line 619)
            _icon_251360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 39), self_251359, '_icon')
            # Calling _icon(args, kwargs) (line 619)
            _icon_call_result_251363 = invoke(stypy.reporting.localization.Localization(__file__, 619, 39), _icon_251360, *[unicode_251361], **kwargs_251362)
            
            unicode_251364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 39), 'unicode', u'Customize')
            # Getting the type of 'self' (line 620)
            self_251365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 52), 'self', False)
            # Obtaining the member 'edit_parameters' of a type (line 620)
            edit_parameters_251366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 52), self_251365, 'edit_parameters')
            # Processing the call keyword arguments (line 619)
            kwargs_251367 = {}
            # Getting the type of 'self' (line 619)
            self_251357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 24), 'self', False)
            # Obtaining the member 'addAction' of a type (line 619)
            addAction_251358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 24), self_251357, 'addAction')
            # Calling addAction(args, kwargs) (line 619)
            addAction_call_result_251368 = invoke(stypy.reporting.localization.Localization(__file__, 619, 24), addAction_251358, *[_icon_call_result_251363, unicode_251364, edit_parameters_251366], **kwargs_251367)
            
            # Assigning a type to the variable 'a' (line 619)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 20), 'a', addAction_call_result_251368)
            
            # Call to setToolTip(...): (line 621)
            # Processing the call arguments (line 621)
            unicode_251371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 33), 'unicode', u'Edit axis, curve and image parameters')
            # Processing the call keyword arguments (line 621)
            kwargs_251372 = {}
            # Getting the type of 'a' (line 621)
            a_251369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 20), 'a', False)
            # Obtaining the member 'setToolTip' of a type (line 621)
            setToolTip_251370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 20), a_251369, 'setToolTip')
            # Calling setToolTip(args, kwargs) (line 621)
            setToolTip_call_result_251373 = invoke(stypy.reporting.localization.Localization(__file__, 621, 20), setToolTip_251370, *[unicode_251371], **kwargs_251372)
            
            # SSA join for if statement (line 618)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_251306 and more_types_in_union_251307):
                # SSA join for if statement (line 608)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Dict to a Attribute (line 623):
        
        # Assigning a Dict to a Attribute (line 623):
        
        # Obtaining an instance of the builtin type 'dict' (line 623)
        dict_251374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 623)
        
        # Getting the type of 'self' (line 623)
        self_251375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'self')
        # Setting the type of the member 'buttons' of a type (line 623)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 8), self_251375, 'buttons', dict_251374)
        
        # Getting the type of 'self' (line 628)
        self_251376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 11), 'self')
        # Obtaining the member 'coordinates' of a type (line 628)
        coordinates_251377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 11), self_251376, 'coordinates')
        # Testing the type of an if condition (line 628)
        if_condition_251378 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 628, 8), coordinates_251377)
        # Assigning a type to the variable 'if_condition_251378' (line 628)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'if_condition_251378', if_condition_251378)
        # SSA begins for if statement (line 628)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 629):
        
        # Assigning a Call to a Attribute (line 629):
        
        # Call to QLabel(...): (line 629)
        # Processing the call arguments (line 629)
        unicode_251381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 45), 'unicode', u'')
        # Getting the type of 'self' (line 629)
        self_251382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 49), 'self', False)
        # Processing the call keyword arguments (line 629)
        kwargs_251383 = {}
        # Getting the type of 'QtWidgets' (line 629)
        QtWidgets_251379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 28), 'QtWidgets', False)
        # Obtaining the member 'QLabel' of a type (line 629)
        QLabel_251380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 28), QtWidgets_251379, 'QLabel')
        # Calling QLabel(args, kwargs) (line 629)
        QLabel_call_result_251384 = invoke(stypy.reporting.localization.Localization(__file__, 629, 28), QLabel_251380, *[unicode_251381, self_251382], **kwargs_251383)
        
        # Getting the type of 'self' (line 629)
        self_251385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 12), 'self')
        # Setting the type of the member 'locLabel' of a type (line 629)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 12), self_251385, 'locLabel', QLabel_call_result_251384)
        
        # Call to setAlignment(...): (line 630)
        # Processing the call arguments (line 630)
        # Getting the type of 'QtCore' (line 631)
        QtCore_251389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 20), 'QtCore', False)
        # Obtaining the member 'Qt' of a type (line 631)
        Qt_251390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 20), QtCore_251389, 'Qt')
        # Obtaining the member 'AlignRight' of a type (line 631)
        AlignRight_251391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 20), Qt_251390, 'AlignRight')
        # Getting the type of 'QtCore' (line 631)
        QtCore_251392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 43), 'QtCore', False)
        # Obtaining the member 'Qt' of a type (line 631)
        Qt_251393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 43), QtCore_251392, 'Qt')
        # Obtaining the member 'AlignTop' of a type (line 631)
        AlignTop_251394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 43), Qt_251393, 'AlignTop')
        # Applying the binary operator '|' (line 631)
        result_or__251395 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 20), '|', AlignRight_251391, AlignTop_251394)
        
        # Processing the call keyword arguments (line 630)
        kwargs_251396 = {}
        # Getting the type of 'self' (line 630)
        self_251386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'self', False)
        # Obtaining the member 'locLabel' of a type (line 630)
        locLabel_251387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 12), self_251386, 'locLabel')
        # Obtaining the member 'setAlignment' of a type (line 630)
        setAlignment_251388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 12), locLabel_251387, 'setAlignment')
        # Calling setAlignment(args, kwargs) (line 630)
        setAlignment_call_result_251397 = invoke(stypy.reporting.localization.Localization(__file__, 630, 12), setAlignment_251388, *[result_or__251395], **kwargs_251396)
        
        
        # Call to setSizePolicy(...): (line 632)
        # Processing the call arguments (line 632)
        
        # Call to QSizePolicy(...): (line 633)
        # Processing the call arguments (line 633)
        # Getting the type of 'QtWidgets' (line 633)
        QtWidgets_251403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 38), 'QtWidgets', False)
        # Obtaining the member 'QSizePolicy' of a type (line 633)
        QSizePolicy_251404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 38), QtWidgets_251403, 'QSizePolicy')
        # Obtaining the member 'Expanding' of a type (line 633)
        Expanding_251405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 38), QSizePolicy_251404, 'Expanding')
        # Getting the type of 'QtWidgets' (line 634)
        QtWidgets_251406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 38), 'QtWidgets', False)
        # Obtaining the member 'QSizePolicy' of a type (line 634)
        QSizePolicy_251407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 38), QtWidgets_251406, 'QSizePolicy')
        # Obtaining the member 'Ignored' of a type (line 634)
        Ignored_251408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 38), QSizePolicy_251407, 'Ignored')
        # Processing the call keyword arguments (line 633)
        kwargs_251409 = {}
        # Getting the type of 'QtWidgets' (line 633)
        QtWidgets_251401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 16), 'QtWidgets', False)
        # Obtaining the member 'QSizePolicy' of a type (line 633)
        QSizePolicy_251402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 16), QtWidgets_251401, 'QSizePolicy')
        # Calling QSizePolicy(args, kwargs) (line 633)
        QSizePolicy_call_result_251410 = invoke(stypy.reporting.localization.Localization(__file__, 633, 16), QSizePolicy_251402, *[Expanding_251405, Ignored_251408], **kwargs_251409)
        
        # Processing the call keyword arguments (line 632)
        kwargs_251411 = {}
        # Getting the type of 'self' (line 632)
        self_251398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'self', False)
        # Obtaining the member 'locLabel' of a type (line 632)
        locLabel_251399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 12), self_251398, 'locLabel')
        # Obtaining the member 'setSizePolicy' of a type (line 632)
        setSizePolicy_251400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 12), locLabel_251399, 'setSizePolicy')
        # Calling setSizePolicy(args, kwargs) (line 632)
        setSizePolicy_call_result_251412 = invoke(stypy.reporting.localization.Localization(__file__, 632, 12), setSizePolicy_251400, *[QSizePolicy_call_result_251410], **kwargs_251411)
        
        
        # Assigning a Call to a Name (line 635):
        
        # Assigning a Call to a Name (line 635):
        
        # Call to addWidget(...): (line 635)
        # Processing the call arguments (line 635)
        # Getting the type of 'self' (line 635)
        self_251415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 41), 'self', False)
        # Obtaining the member 'locLabel' of a type (line 635)
        locLabel_251416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 41), self_251415, 'locLabel')
        # Processing the call keyword arguments (line 635)
        kwargs_251417 = {}
        # Getting the type of 'self' (line 635)
        self_251413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 26), 'self', False)
        # Obtaining the member 'addWidget' of a type (line 635)
        addWidget_251414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 26), self_251413, 'addWidget')
        # Calling addWidget(args, kwargs) (line 635)
        addWidget_call_result_251418 = invoke(stypy.reporting.localization.Localization(__file__, 635, 26), addWidget_251414, *[locLabel_251416], **kwargs_251417)
        
        # Assigning a type to the variable 'labelAction' (line 635)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 12), 'labelAction', addWidget_call_result_251418)
        
        # Call to setVisible(...): (line 636)
        # Processing the call arguments (line 636)
        # Getting the type of 'True' (line 636)
        True_251421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 35), 'True', False)
        # Processing the call keyword arguments (line 636)
        kwargs_251422 = {}
        # Getting the type of 'labelAction' (line 636)
        labelAction_251419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'labelAction', False)
        # Obtaining the member 'setVisible' of a type (line 636)
        setVisible_251420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 12), labelAction_251419, 'setVisible')
        # Calling setVisible(args, kwargs) (line 636)
        setVisible_call_result_251423 = invoke(stypy.reporting.localization.Localization(__file__, 636, 12), setVisible_251420, *[True_251421], **kwargs_251422)
        
        # SSA join for if statement (line 628)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 639):
        
        # Assigning a Name to a Attribute (line 639):
        # Getting the type of 'None' (line 639)
        None_251424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 26), 'None')
        # Getting the type of 'self' (line 639)
        self_251425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 8), 'self')
        # Setting the type of the member 'adj_window' of a type (line 639)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 8), self_251425, 'adj_window', None_251424)
        
        
        # Call to is_pyqt5(...): (line 644)
        # Processing the call keyword arguments (line 644)
        kwargs_251427 = {}
        # Getting the type of 'is_pyqt5' (line 644)
        is_pyqt5_251426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 11), 'is_pyqt5', False)
        # Calling is_pyqt5(args, kwargs) (line 644)
        is_pyqt5_call_result_251428 = invoke(stypy.reporting.localization.Localization(__file__, 644, 11), is_pyqt5_251426, *[], **kwargs_251427)
        
        # Testing the type of an if condition (line 644)
        if_condition_251429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 644, 8), is_pyqt5_call_result_251428)
        # Assigning a type to the variable 'if_condition_251429' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'if_condition_251429', if_condition_251429)
        # SSA begins for if statement (line 644)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setIconSize(...): (line 645)
        # Processing the call arguments (line 645)
        
        # Call to QSize(...): (line 645)
        # Processing the call arguments (line 645)
        int_251434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 42), 'int')
        int_251435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 46), 'int')
        # Processing the call keyword arguments (line 645)
        kwargs_251436 = {}
        # Getting the type of 'QtCore' (line 645)
        QtCore_251432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 29), 'QtCore', False)
        # Obtaining the member 'QSize' of a type (line 645)
        QSize_251433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 29), QtCore_251432, 'QSize')
        # Calling QSize(args, kwargs) (line 645)
        QSize_call_result_251437 = invoke(stypy.reporting.localization.Localization(__file__, 645, 29), QSize_251433, *[int_251434, int_251435], **kwargs_251436)
        
        # Processing the call keyword arguments (line 645)
        kwargs_251438 = {}
        # Getting the type of 'self' (line 645)
        self_251430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 12), 'self', False)
        # Obtaining the member 'setIconSize' of a type (line 645)
        setIconSize_251431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 12), self_251430, 'setIconSize')
        # Calling setIconSize(args, kwargs) (line 645)
        setIconSize_call_result_251439 = invoke(stypy.reporting.localization.Localization(__file__, 645, 12), setIconSize_251431, *[QSize_call_result_251437], **kwargs_251438)
        
        
        # Call to setSpacing(...): (line 646)
        # Processing the call arguments (line 646)
        int_251445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 37), 'int')
        # Processing the call keyword arguments (line 646)
        kwargs_251446 = {}
        
        # Call to layout(...): (line 646)
        # Processing the call keyword arguments (line 646)
        kwargs_251442 = {}
        # Getting the type of 'self' (line 646)
        self_251440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 12), 'self', False)
        # Obtaining the member 'layout' of a type (line 646)
        layout_251441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 12), self_251440, 'layout')
        # Calling layout(args, kwargs) (line 646)
        layout_call_result_251443 = invoke(stypy.reporting.localization.Localization(__file__, 646, 12), layout_251441, *[], **kwargs_251442)
        
        # Obtaining the member 'setSpacing' of a type (line 646)
        setSpacing_251444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 12), layout_call_result_251443, 'setSpacing')
        # Calling setSpacing(args, kwargs) (line 646)
        setSpacing_call_result_251447 = invoke(stypy.reporting.localization.Localization(__file__, 646, 12), setSpacing_251444, *[int_251445], **kwargs_251446)
        
        # SSA join for if statement (line 644)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_init_toolbar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_toolbar' in the type store
        # Getting the type of 'stypy_return_type' (line 604)
        stypy_return_type_251448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251448)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_toolbar'
        return stypy_return_type_251448


    @norecursion
    def edit_parameters(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'edit_parameters'
        module_type_store = module_type_store.open_function_context('edit_parameters', 657, 4, False)
        # Assigning a type to the variable 'self' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT.edit_parameters.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT.edit_parameters.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT.edit_parameters.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT.edit_parameters.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT.edit_parameters')
        NavigationToolbar2QT.edit_parameters.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2QT.edit_parameters.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2QT.edit_parameters.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT.edit_parameters.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT.edit_parameters.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT.edit_parameters.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT.edit_parameters.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT.edit_parameters', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'edit_parameters', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'edit_parameters(...)' code ##################

        
        # Assigning a Call to a Name (line 658):
        
        # Assigning a Call to a Name (line 658):
        
        # Call to get_axes(...): (line 658)
        # Processing the call keyword arguments (line 658)
        kwargs_251453 = {}
        # Getting the type of 'self' (line 658)
        self_251449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 18), 'self', False)
        # Obtaining the member 'canvas' of a type (line 658)
        canvas_251450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 18), self_251449, 'canvas')
        # Obtaining the member 'figure' of a type (line 658)
        figure_251451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 18), canvas_251450, 'figure')
        # Obtaining the member 'get_axes' of a type (line 658)
        get_axes_251452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 18), figure_251451, 'get_axes')
        # Calling get_axes(args, kwargs) (line 658)
        get_axes_call_result_251454 = invoke(stypy.reporting.localization.Localization(__file__, 658, 18), get_axes_251452, *[], **kwargs_251453)
        
        # Assigning a type to the variable 'allaxes' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 8), 'allaxes', get_axes_call_result_251454)
        
        
        # Getting the type of 'allaxes' (line 659)
        allaxes_251455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 15), 'allaxes')
        # Applying the 'not' unary operator (line 659)
        result_not__251456 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 11), 'not', allaxes_251455)
        
        # Testing the type of an if condition (line 659)
        if_condition_251457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 659, 8), result_not__251456)
        # Assigning a type to the variable 'if_condition_251457' (line 659)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 8), 'if_condition_251457', if_condition_251457)
        # SSA begins for if statement (line 659)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warning(...): (line 660)
        # Processing the call arguments (line 660)
        # Getting the type of 'self' (line 661)
        self_251461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'self', False)
        # Obtaining the member 'parent' of a type (line 661)
        parent_251462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 16), self_251461, 'parent')
        unicode_251463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 29), 'unicode', u'Error')
        unicode_251464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 38), 'unicode', u'There are no axes to edit.')
        # Processing the call keyword arguments (line 660)
        kwargs_251465 = {}
        # Getting the type of 'QtWidgets' (line 660)
        QtWidgets_251458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 12), 'QtWidgets', False)
        # Obtaining the member 'QMessageBox' of a type (line 660)
        QMessageBox_251459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 12), QtWidgets_251458, 'QMessageBox')
        # Obtaining the member 'warning' of a type (line 660)
        warning_251460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 12), QMessageBox_251459, 'warning')
        # Calling warning(args, kwargs) (line 660)
        warning_call_result_251466 = invoke(stypy.reporting.localization.Localization(__file__, 660, 12), warning_251460, *[parent_251462, unicode_251463, unicode_251464], **kwargs_251465)
        
        # Assigning a type to the variable 'stypy_return_type' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 12), 'stypy_return_type', types.NoneType)
        # SSA branch for the else part of an if statement (line 659)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to len(...): (line 663)
        # Processing the call arguments (line 663)
        # Getting the type of 'allaxes' (line 663)
        allaxes_251468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 17), 'allaxes', False)
        # Processing the call keyword arguments (line 663)
        kwargs_251469 = {}
        # Getting the type of 'len' (line 663)
        len_251467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 13), 'len', False)
        # Calling len(args, kwargs) (line 663)
        len_call_result_251470 = invoke(stypy.reporting.localization.Localization(__file__, 663, 13), len_251467, *[allaxes_251468], **kwargs_251469)
        
        int_251471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 29), 'int')
        # Applying the binary operator '==' (line 663)
        result_eq_251472 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 13), '==', len_call_result_251470, int_251471)
        
        # Testing the type of an if condition (line 663)
        if_condition_251473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 663, 13), result_eq_251472)
        # Assigning a type to the variable 'if_condition_251473' (line 663)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 13), 'if_condition_251473', if_condition_251473)
        # SSA begins for if statement (line 663)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 664):
        
        # Assigning a Subscript to a Name (line 664):
        
        # Obtaining the type of the subscript
        int_251474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 12), 'int')
        # Getting the type of 'allaxes' (line 664)
        allaxes_251475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 20), 'allaxes')
        # Obtaining the member '__getitem__' of a type (line 664)
        getitem___251476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 12), allaxes_251475, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 664)
        subscript_call_result_251477 = invoke(stypy.reporting.localization.Localization(__file__, 664, 12), getitem___251476, int_251474)
        
        # Assigning a type to the variable 'tuple_var_assignment_249720' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'tuple_var_assignment_249720', subscript_call_result_251477)
        
        # Assigning a Name to a Name (line 664):
        # Getting the type of 'tuple_var_assignment_249720' (line 664)
        tuple_var_assignment_249720_251478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'tuple_var_assignment_249720')
        # Assigning a type to the variable 'axes' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'axes', tuple_var_assignment_249720_251478)
        # SSA branch for the else part of an if statement (line 663)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 666):
        
        # Assigning a List to a Name (line 666):
        
        # Obtaining an instance of the builtin type 'list' (line 666)
        list_251479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 666)
        
        # Assigning a type to the variable 'titles' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'titles', list_251479)
        
        # Getting the type of 'allaxes' (line 667)
        allaxes_251480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 24), 'allaxes')
        # Testing the type of a for loop iterable (line 667)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 667, 12), allaxes_251480)
        # Getting the type of the for loop variable (line 667)
        for_loop_var_251481 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 667, 12), allaxes_251480)
        # Assigning a type to the variable 'axes' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), 'axes', for_loop_var_251481)
        # SSA begins for a for statement (line 667)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BoolOp to a Name (line 668):
        
        # Assigning a BoolOp to a Name (line 668):
        
        # Evaluating a boolean operation
        
        # Call to get_title(...): (line 668)
        # Processing the call keyword arguments (line 668)
        kwargs_251484 = {}
        # Getting the type of 'axes' (line 668)
        axes_251482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 24), 'axes', False)
        # Obtaining the member 'get_title' of a type (line 668)
        get_title_251483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 24), axes_251482, 'get_title')
        # Calling get_title(args, kwargs) (line 668)
        get_title_call_result_251485 = invoke(stypy.reporting.localization.Localization(__file__, 668, 24), get_title_251483, *[], **kwargs_251484)
        
        
        # Call to join(...): (line 669)
        # Processing the call arguments (line 669)
        
        # Call to filter(...): (line 669)
        # Processing the call arguments (line 669)
        # Getting the type of 'None' (line 669)
        None_251489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 42), 'None', False)
        
        # Obtaining an instance of the builtin type 'list' (line 669)
        list_251490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 669)
        # Adding element type (line 669)
        
        # Call to get_xlabel(...): (line 669)
        # Processing the call keyword arguments (line 669)
        kwargs_251493 = {}
        # Getting the type of 'axes' (line 669)
        axes_251491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 49), 'axes', False)
        # Obtaining the member 'get_xlabel' of a type (line 669)
        get_xlabel_251492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 49), axes_251491, 'get_xlabel')
        # Calling get_xlabel(args, kwargs) (line 669)
        get_xlabel_call_result_251494 = invoke(stypy.reporting.localization.Localization(__file__, 669, 49), get_xlabel_251492, *[], **kwargs_251493)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 48), list_251490, get_xlabel_call_result_251494)
        # Adding element type (line 669)
        
        # Call to get_ylabel(...): (line 670)
        # Processing the call keyword arguments (line 670)
        kwargs_251497 = {}
        # Getting the type of 'axes' (line 670)
        axes_251495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 49), 'axes', False)
        # Obtaining the member 'get_ylabel' of a type (line 670)
        get_ylabel_251496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 49), axes_251495, 'get_ylabel')
        # Calling get_ylabel(args, kwargs) (line 670)
        get_ylabel_call_result_251498 = invoke(stypy.reporting.localization.Localization(__file__, 670, 49), get_ylabel_251496, *[], **kwargs_251497)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 669, 48), list_251490, get_ylabel_call_result_251498)
        
        # Processing the call keyword arguments (line 669)
        kwargs_251499 = {}
        # Getting the type of 'filter' (line 669)
        filter_251488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 35), 'filter', False)
        # Calling filter(args, kwargs) (line 669)
        filter_call_result_251500 = invoke(stypy.reporting.localization.Localization(__file__, 669, 35), filter_251488, *[None_251489, list_251490], **kwargs_251499)
        
        # Processing the call keyword arguments (line 669)
        kwargs_251501 = {}
        unicode_251486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 24), 'unicode', u' - ')
        # Obtaining the member 'join' of a type (line 669)
        join_251487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 24), unicode_251486, 'join')
        # Calling join(args, kwargs) (line 669)
        join_call_result_251502 = invoke(stypy.reporting.localization.Localization(__file__, 669, 24), join_251487, *[filter_call_result_251500], **kwargs_251501)
        
        # Applying the binary operator 'or' (line 668)
        result_or_keyword_251503 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 24), 'or', get_title_call_result_251485, join_call_result_251502)
        
        # Call to format(...): (line 671)
        # Processing the call arguments (line 671)
        
        # Call to type(...): (line 672)
        # Processing the call arguments (line 672)
        # Getting the type of 'axes' (line 672)
        axes_251507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 33), 'axes', False)
        # Processing the call keyword arguments (line 672)
        kwargs_251508 = {}
        # Getting the type of 'type' (line 672)
        type_251506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 28), 'type', False)
        # Calling type(args, kwargs) (line 672)
        type_call_result_251509 = invoke(stypy.reporting.localization.Localization(__file__, 672, 28), type_251506, *[axes_251507], **kwargs_251508)
        
        # Obtaining the member '__name__' of a type (line 672)
        name___251510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 28), type_call_result_251509, '__name__')
        
        # Call to id(...): (line 672)
        # Processing the call arguments (line 672)
        # Getting the type of 'axes' (line 672)
        axes_251512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 52), 'axes', False)
        # Processing the call keyword arguments (line 672)
        kwargs_251513 = {}
        # Getting the type of 'id' (line 672)
        id_251511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 49), 'id', False)
        # Calling id(args, kwargs) (line 672)
        id_call_result_251514 = invoke(stypy.reporting.localization.Localization(__file__, 672, 49), id_251511, *[axes_251512], **kwargs_251513)
        
        # Processing the call keyword arguments (line 671)
        kwargs_251515 = {}
        unicode_251504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 24), 'unicode', u'<anonymous {} (id: {:#x})>')
        # Obtaining the member 'format' of a type (line 671)
        format_251505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 24), unicode_251504, 'format')
        # Calling format(args, kwargs) (line 671)
        format_call_result_251516 = invoke(stypy.reporting.localization.Localization(__file__, 671, 24), format_251505, *[name___251510, id_call_result_251514], **kwargs_251515)
        
        # Applying the binary operator 'or' (line 668)
        result_or_keyword_251517 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 24), 'or', result_or_keyword_251503, format_call_result_251516)
        
        # Assigning a type to the variable 'name' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 16), 'name', result_or_keyword_251517)
        
        # Call to append(...): (line 673)
        # Processing the call arguments (line 673)
        # Getting the type of 'name' (line 673)
        name_251520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 30), 'name', False)
        # Processing the call keyword arguments (line 673)
        kwargs_251521 = {}
        # Getting the type of 'titles' (line 673)
        titles_251518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 16), 'titles', False)
        # Obtaining the member 'append' of a type (line 673)
        append_251519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 16), titles_251518, 'append')
        # Calling append(args, kwargs) (line 673)
        append_call_result_251522 = invoke(stypy.reporting.localization.Localization(__file__, 673, 16), append_251519, *[name_251520], **kwargs_251521)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 674):
        
        # Assigning a Call to a Name:
        
        # Call to getItem(...): (line 674)
        # Processing the call arguments (line 674)
        # Getting the type of 'self' (line 675)
        self_251526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'self', False)
        # Obtaining the member 'parent' of a type (line 675)
        parent_251527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 16), self_251526, 'parent')
        unicode_251528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 29), 'unicode', u'Customize')
        unicode_251529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 42), 'unicode', u'Select axes:')
        # Getting the type of 'titles' (line 675)
        titles_251530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 58), 'titles', False)
        int_251531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 66), 'int')
        # Getting the type of 'False' (line 675)
        False_251532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 69), 'False', False)
        # Processing the call keyword arguments (line 674)
        kwargs_251533 = {}
        # Getting the type of 'QtWidgets' (line 674)
        QtWidgets_251523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 23), 'QtWidgets', False)
        # Obtaining the member 'QInputDialog' of a type (line 674)
        QInputDialog_251524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 23), QtWidgets_251523, 'QInputDialog')
        # Obtaining the member 'getItem' of a type (line 674)
        getItem_251525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 23), QInputDialog_251524, 'getItem')
        # Calling getItem(args, kwargs) (line 674)
        getItem_call_result_251534 = invoke(stypy.reporting.localization.Localization(__file__, 674, 23), getItem_251525, *[parent_251527, unicode_251528, unicode_251529, titles_251530, int_251531, False_251532], **kwargs_251533)
        
        # Assigning a type to the variable 'call_assignment_249721' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_249721', getItem_call_result_251534)
        
        # Assigning a Call to a Name (line 674):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_251537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 12), 'int')
        # Processing the call keyword arguments
        kwargs_251538 = {}
        # Getting the type of 'call_assignment_249721' (line 674)
        call_assignment_249721_251535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_249721', False)
        # Obtaining the member '__getitem__' of a type (line 674)
        getitem___251536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 12), call_assignment_249721_251535, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_251539 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___251536, *[int_251537], **kwargs_251538)
        
        # Assigning a type to the variable 'call_assignment_249722' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_249722', getitem___call_result_251539)
        
        # Assigning a Name to a Name (line 674):
        # Getting the type of 'call_assignment_249722' (line 674)
        call_assignment_249722_251540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_249722')
        # Assigning a type to the variable 'item' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'item', call_assignment_249722_251540)
        
        # Assigning a Call to a Name (line 674):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_251543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 12), 'int')
        # Processing the call keyword arguments
        kwargs_251544 = {}
        # Getting the type of 'call_assignment_249721' (line 674)
        call_assignment_249721_251541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_249721', False)
        # Obtaining the member '__getitem__' of a type (line 674)
        getitem___251542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 12), call_assignment_249721_251541, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_251545 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___251542, *[int_251543], **kwargs_251544)
        
        # Assigning a type to the variable 'call_assignment_249723' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_249723', getitem___call_result_251545)
        
        # Assigning a Name to a Name (line 674):
        # Getting the type of 'call_assignment_249723' (line 674)
        call_assignment_249723_251546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 12), 'call_assignment_249723')
        # Assigning a type to the variable 'ok' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 18), 'ok', call_assignment_249723_251546)
        
        # Getting the type of 'ok' (line 676)
        ok_251547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 15), 'ok')
        # Testing the type of an if condition (line 676)
        if_condition_251548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 676, 12), ok_251547)
        # Assigning a type to the variable 'if_condition_251548' (line 676)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 12), 'if_condition_251548', if_condition_251548)
        # SSA begins for if statement (line 676)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 677):
        
        # Assigning a Subscript to a Name (line 677):
        
        # Obtaining the type of the subscript
        
        # Call to index(...): (line 677)
        # Processing the call arguments (line 677)
        
        # Call to text_type(...): (line 677)
        # Processing the call arguments (line 677)
        # Getting the type of 'item' (line 677)
        item_251553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 58), 'item', False)
        # Processing the call keyword arguments (line 677)
        kwargs_251554 = {}
        # Getting the type of 'six' (line 677)
        six_251551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 44), 'six', False)
        # Obtaining the member 'text_type' of a type (line 677)
        text_type_251552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 44), six_251551, 'text_type')
        # Calling text_type(args, kwargs) (line 677)
        text_type_call_result_251555 = invoke(stypy.reporting.localization.Localization(__file__, 677, 44), text_type_251552, *[item_251553], **kwargs_251554)
        
        # Processing the call keyword arguments (line 677)
        kwargs_251556 = {}
        # Getting the type of 'titles' (line 677)
        titles_251549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 31), 'titles', False)
        # Obtaining the member 'index' of a type (line 677)
        index_251550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 31), titles_251549, 'index')
        # Calling index(args, kwargs) (line 677)
        index_call_result_251557 = invoke(stypy.reporting.localization.Localization(__file__, 677, 31), index_251550, *[text_type_call_result_251555], **kwargs_251556)
        
        # Getting the type of 'allaxes' (line 677)
        allaxes_251558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 23), 'allaxes')
        # Obtaining the member '__getitem__' of a type (line 677)
        getitem___251559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 23), allaxes_251558, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 677)
        subscript_call_result_251560 = invoke(stypy.reporting.localization.Localization(__file__, 677, 23), getitem___251559, index_call_result_251557)
        
        # Assigning a type to the variable 'axes' (line 677)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'axes', subscript_call_result_251560)
        # SSA branch for the else part of an if statement (line 676)
        module_type_store.open_ssa_branch('else')
        # Assigning a type to the variable 'stypy_return_type' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 676)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 663)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 659)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to figure_edit(...): (line 681)
        # Processing the call arguments (line 681)
        # Getting the type of 'axes' (line 681)
        axes_251563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 34), 'axes', False)
        # Getting the type of 'self' (line 681)
        self_251564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 40), 'self', False)
        # Processing the call keyword arguments (line 681)
        kwargs_251565 = {}
        # Getting the type of 'figureoptions' (line 681)
        figureoptions_251561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'figureoptions', False)
        # Obtaining the member 'figure_edit' of a type (line 681)
        figure_edit_251562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 8), figureoptions_251561, 'figure_edit')
        # Calling figure_edit(args, kwargs) (line 681)
        figure_edit_call_result_251566 = invoke(stypy.reporting.localization.Localization(__file__, 681, 8), figure_edit_251562, *[axes_251563, self_251564], **kwargs_251565)
        
        
        # ################# End of 'edit_parameters(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'edit_parameters' in the type store
        # Getting the type of 'stypy_return_type' (line 657)
        stypy_return_type_251567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251567)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'edit_parameters'
        return stypy_return_type_251567


    @norecursion
    def _update_buttons_checked(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update_buttons_checked'
        module_type_store = module_type_store.open_function_context('_update_buttons_checked', 683, 4, False)
        # Assigning a type to the variable 'self' (line 684)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT._update_buttons_checked.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT._update_buttons_checked.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT._update_buttons_checked.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT._update_buttons_checked.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT._update_buttons_checked')
        NavigationToolbar2QT._update_buttons_checked.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2QT._update_buttons_checked.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2QT._update_buttons_checked.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT._update_buttons_checked.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT._update_buttons_checked.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT._update_buttons_checked.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT._update_buttons_checked.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT._update_buttons_checked', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update_buttons_checked', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update_buttons_checked(...)' code ##################

        
        # Call to setChecked(...): (line 685)
        # Processing the call arguments (line 685)
        
        # Getting the type of 'self' (line 685)
        self_251574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 40), 'self', False)
        # Obtaining the member '_active' of a type (line 685)
        _active_251575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 40), self_251574, '_active')
        unicode_251576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 56), 'unicode', u'PAN')
        # Applying the binary operator '==' (line 685)
        result_eq_251577 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 40), '==', _active_251575, unicode_251576)
        
        # Processing the call keyword arguments (line 685)
        kwargs_251578 = {}
        
        # Obtaining the type of the subscript
        unicode_251568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 22), 'unicode', u'pan')
        # Getting the type of 'self' (line 685)
        self_251569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 8), 'self', False)
        # Obtaining the member '_actions' of a type (line 685)
        _actions_251570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 8), self_251569, '_actions')
        # Obtaining the member '__getitem__' of a type (line 685)
        getitem___251571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 8), _actions_251570, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 685)
        subscript_call_result_251572 = invoke(stypy.reporting.localization.Localization(__file__, 685, 8), getitem___251571, unicode_251568)
        
        # Obtaining the member 'setChecked' of a type (line 685)
        setChecked_251573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 8), subscript_call_result_251572, 'setChecked')
        # Calling setChecked(args, kwargs) (line 685)
        setChecked_call_result_251579 = invoke(stypy.reporting.localization.Localization(__file__, 685, 8), setChecked_251573, *[result_eq_251577], **kwargs_251578)
        
        
        # Call to setChecked(...): (line 686)
        # Processing the call arguments (line 686)
        
        # Getting the type of 'self' (line 686)
        self_251586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 41), 'self', False)
        # Obtaining the member '_active' of a type (line 686)
        _active_251587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 41), self_251586, '_active')
        unicode_251588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 57), 'unicode', u'ZOOM')
        # Applying the binary operator '==' (line 686)
        result_eq_251589 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 41), '==', _active_251587, unicode_251588)
        
        # Processing the call keyword arguments (line 686)
        kwargs_251590 = {}
        
        # Obtaining the type of the subscript
        unicode_251580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 22), 'unicode', u'zoom')
        # Getting the type of 'self' (line 686)
        self_251581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'self', False)
        # Obtaining the member '_actions' of a type (line 686)
        _actions_251582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 8), self_251581, '_actions')
        # Obtaining the member '__getitem__' of a type (line 686)
        getitem___251583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 8), _actions_251582, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 686)
        subscript_call_result_251584 = invoke(stypy.reporting.localization.Localization(__file__, 686, 8), getitem___251583, unicode_251580)
        
        # Obtaining the member 'setChecked' of a type (line 686)
        setChecked_251585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 8), subscript_call_result_251584, 'setChecked')
        # Calling setChecked(args, kwargs) (line 686)
        setChecked_call_result_251591 = invoke(stypy.reporting.localization.Localization(__file__, 686, 8), setChecked_251585, *[result_eq_251589], **kwargs_251590)
        
        
        # ################# End of '_update_buttons_checked(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update_buttons_checked' in the type store
        # Getting the type of 'stypy_return_type' (line 683)
        stypy_return_type_251592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251592)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update_buttons_checked'
        return stypy_return_type_251592


    @norecursion
    def pan(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pan'
        module_type_store = module_type_store.open_function_context('pan', 688, 4, False)
        # Assigning a type to the variable 'self' (line 689)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT.pan.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT.pan.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT.pan.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT.pan.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT.pan')
        NavigationToolbar2QT.pan.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2QT.pan.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        NavigationToolbar2QT.pan.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT.pan.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT.pan.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT.pan.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT.pan.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT.pan', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pan', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pan(...)' code ##################

        
        # Call to pan(...): (line 689)
        # Getting the type of 'args' (line 689)
        args_251599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 47), 'args', False)
        # Processing the call keyword arguments (line 689)
        kwargs_251600 = {}
        
        # Call to super(...): (line 689)
        # Processing the call arguments (line 689)
        # Getting the type of 'NavigationToolbar2QT' (line 689)
        NavigationToolbar2QT_251594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 14), 'NavigationToolbar2QT', False)
        # Getting the type of 'self' (line 689)
        self_251595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 36), 'self', False)
        # Processing the call keyword arguments (line 689)
        kwargs_251596 = {}
        # Getting the type of 'super' (line 689)
        super_251593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'super', False)
        # Calling super(args, kwargs) (line 689)
        super_call_result_251597 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), super_251593, *[NavigationToolbar2QT_251594, self_251595], **kwargs_251596)
        
        # Obtaining the member 'pan' of a type (line 689)
        pan_251598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), super_call_result_251597, 'pan')
        # Calling pan(args, kwargs) (line 689)
        pan_call_result_251601 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), pan_251598, *[args_251599], **kwargs_251600)
        
        
        # Call to _update_buttons_checked(...): (line 690)
        # Processing the call keyword arguments (line 690)
        kwargs_251604 = {}
        # Getting the type of 'self' (line 690)
        self_251602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'self', False)
        # Obtaining the member '_update_buttons_checked' of a type (line 690)
        _update_buttons_checked_251603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 8), self_251602, '_update_buttons_checked')
        # Calling _update_buttons_checked(args, kwargs) (line 690)
        _update_buttons_checked_call_result_251605 = invoke(stypy.reporting.localization.Localization(__file__, 690, 8), _update_buttons_checked_251603, *[], **kwargs_251604)
        
        
        # ################# End of 'pan(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pan' in the type store
        # Getting the type of 'stypy_return_type' (line 688)
        stypy_return_type_251606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251606)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pan'
        return stypy_return_type_251606


    @norecursion
    def zoom(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'zoom'
        module_type_store = module_type_store.open_function_context('zoom', 692, 4, False)
        # Assigning a type to the variable 'self' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT.zoom.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT.zoom.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT.zoom.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT.zoom.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT.zoom')
        NavigationToolbar2QT.zoom.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2QT.zoom.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        NavigationToolbar2QT.zoom.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT.zoom.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT.zoom.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT.zoom.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT.zoom.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT.zoom', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'zoom', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'zoom(...)' code ##################

        
        # Call to zoom(...): (line 693)
        # Getting the type of 'args' (line 693)
        args_251613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 48), 'args', False)
        # Processing the call keyword arguments (line 693)
        kwargs_251614 = {}
        
        # Call to super(...): (line 693)
        # Processing the call arguments (line 693)
        # Getting the type of 'NavigationToolbar2QT' (line 693)
        NavigationToolbar2QT_251608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 14), 'NavigationToolbar2QT', False)
        # Getting the type of 'self' (line 693)
        self_251609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 36), 'self', False)
        # Processing the call keyword arguments (line 693)
        kwargs_251610 = {}
        # Getting the type of 'super' (line 693)
        super_251607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'super', False)
        # Calling super(args, kwargs) (line 693)
        super_call_result_251611 = invoke(stypy.reporting.localization.Localization(__file__, 693, 8), super_251607, *[NavigationToolbar2QT_251608, self_251609], **kwargs_251610)
        
        # Obtaining the member 'zoom' of a type (line 693)
        zoom_251612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 8), super_call_result_251611, 'zoom')
        # Calling zoom(args, kwargs) (line 693)
        zoom_call_result_251615 = invoke(stypy.reporting.localization.Localization(__file__, 693, 8), zoom_251612, *[args_251613], **kwargs_251614)
        
        
        # Call to _update_buttons_checked(...): (line 694)
        # Processing the call keyword arguments (line 694)
        kwargs_251618 = {}
        # Getting the type of 'self' (line 694)
        self_251616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'self', False)
        # Obtaining the member '_update_buttons_checked' of a type (line 694)
        _update_buttons_checked_251617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 8), self_251616, '_update_buttons_checked')
        # Calling _update_buttons_checked(args, kwargs) (line 694)
        _update_buttons_checked_call_result_251619 = invoke(stypy.reporting.localization.Localization(__file__, 694, 8), _update_buttons_checked_251617, *[], **kwargs_251618)
        
        
        # ################# End of 'zoom(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'zoom' in the type store
        # Getting the type of 'stypy_return_type' (line 692)
        stypy_return_type_251620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251620)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'zoom'
        return stypy_return_type_251620


    @norecursion
    def set_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_message'
        module_type_store = module_type_store.open_function_context('set_message', 696, 4, False)
        # Assigning a type to the variable 'self' (line 697)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT.set_message.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT.set_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT.set_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT.set_message.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT.set_message')
        NavigationToolbar2QT.set_message.__dict__.__setitem__('stypy_param_names_list', ['s'])
        NavigationToolbar2QT.set_message.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2QT.set_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT.set_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT.set_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT.set_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT.set_message.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT.set_message', ['s'], None, None, defaults, varargs, kwargs)

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

        
        # Call to emit(...): (line 697)
        # Processing the call arguments (line 697)
        # Getting the type of 's' (line 697)
        s_251624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 26), 's', False)
        # Processing the call keyword arguments (line 697)
        kwargs_251625 = {}
        # Getting the type of 'self' (line 697)
        self_251621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'self', False)
        # Obtaining the member 'message' of a type (line 697)
        message_251622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), self_251621, 'message')
        # Obtaining the member 'emit' of a type (line 697)
        emit_251623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), message_251622, 'emit')
        # Calling emit(args, kwargs) (line 697)
        emit_call_result_251626 = invoke(stypy.reporting.localization.Localization(__file__, 697, 8), emit_251623, *[s_251624], **kwargs_251625)
        
        
        # Getting the type of 'self' (line 698)
        self_251627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 11), 'self')
        # Obtaining the member 'coordinates' of a type (line 698)
        coordinates_251628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 11), self_251627, 'coordinates')
        # Testing the type of an if condition (line 698)
        if_condition_251629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 698, 8), coordinates_251628)
        # Assigning a type to the variable 'if_condition_251629' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'if_condition_251629', if_condition_251629)
        # SSA begins for if statement (line 698)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setText(...): (line 699)
        # Processing the call arguments (line 699)
        # Getting the type of 's' (line 699)
        s_251633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 34), 's', False)
        # Processing the call keyword arguments (line 699)
        kwargs_251634 = {}
        # Getting the type of 'self' (line 699)
        self_251630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'self', False)
        # Obtaining the member 'locLabel' of a type (line 699)
        locLabel_251631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 12), self_251630, 'locLabel')
        # Obtaining the member 'setText' of a type (line 699)
        setText_251632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 12), locLabel_251631, 'setText')
        # Calling setText(args, kwargs) (line 699)
        setText_call_result_251635 = invoke(stypy.reporting.localization.Localization(__file__, 699, 12), setText_251632, *[s_251633], **kwargs_251634)
        
        # SSA join for if statement (line 698)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_message' in the type store
        # Getting the type of 'stypy_return_type' (line 696)
        stypy_return_type_251636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251636)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_message'
        return stypy_return_type_251636


    @norecursion
    def set_cursor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_cursor'
        module_type_store = module_type_store.open_function_context('set_cursor', 701, 4, False)
        # Assigning a type to the variable 'self' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT.set_cursor.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT.set_cursor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT.set_cursor.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT.set_cursor.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT.set_cursor')
        NavigationToolbar2QT.set_cursor.__dict__.__setitem__('stypy_param_names_list', ['cursor'])
        NavigationToolbar2QT.set_cursor.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2QT.set_cursor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT.set_cursor.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT.set_cursor.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT.set_cursor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT.set_cursor.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT.set_cursor', ['cursor'], None, None, defaults, varargs, kwargs)

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

        
        # Call to setCursor(...): (line 702)
        # Processing the call arguments (line 702)
        
        # Obtaining the type of the subscript
        # Getting the type of 'cursor' (line 702)
        cursor_251640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 38), 'cursor', False)
        # Getting the type of 'cursord' (line 702)
        cursord_251641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 30), 'cursord', False)
        # Obtaining the member '__getitem__' of a type (line 702)
        getitem___251642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 30), cursord_251641, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 702)
        subscript_call_result_251643 = invoke(stypy.reporting.localization.Localization(__file__, 702, 30), getitem___251642, cursor_251640)
        
        # Processing the call keyword arguments (line 702)
        kwargs_251644 = {}
        # Getting the type of 'self' (line 702)
        self_251637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 702)
        canvas_251638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 8), self_251637, 'canvas')
        # Obtaining the member 'setCursor' of a type (line 702)
        setCursor_251639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 8), canvas_251638, 'setCursor')
        # Calling setCursor(args, kwargs) (line 702)
        setCursor_call_result_251645 = invoke(stypy.reporting.localization.Localization(__file__, 702, 8), setCursor_251639, *[subscript_call_result_251643], **kwargs_251644)
        
        
        # ################# End of 'set_cursor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_cursor' in the type store
        # Getting the type of 'stypy_return_type' (line 701)
        stypy_return_type_251646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251646)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_cursor'
        return stypy_return_type_251646


    @norecursion
    def draw_rubberband(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_rubberband'
        module_type_store = module_type_store.open_function_context('draw_rubberband', 704, 4, False)
        # Assigning a type to the variable 'self' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT.draw_rubberband.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT.draw_rubberband.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT.draw_rubberband.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT.draw_rubberband.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT.draw_rubberband')
        NavigationToolbar2QT.draw_rubberband.__dict__.__setitem__('stypy_param_names_list', ['event', 'x0', 'y0', 'x1', 'y1'])
        NavigationToolbar2QT.draw_rubberband.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2QT.draw_rubberband.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT.draw_rubberband.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT.draw_rubberband.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT.draw_rubberband.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT.draw_rubberband.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT.draw_rubberband', ['event', 'x0', 'y0', 'x1', 'y1'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Name (line 705):
        
        # Assigning a Attribute to a Name (line 705):
        # Getting the type of 'self' (line 705)
        self_251647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 17), 'self')
        # Obtaining the member 'canvas' of a type (line 705)
        canvas_251648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 17), self_251647, 'canvas')
        # Obtaining the member 'figure' of a type (line 705)
        figure_251649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 17), canvas_251648, 'figure')
        # Obtaining the member 'bbox' of a type (line 705)
        bbox_251650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 17), figure_251649, 'bbox')
        # Obtaining the member 'height' of a type (line 705)
        height_251651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 17), bbox_251650, 'height')
        # Assigning a type to the variable 'height' (line 705)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'height', height_251651)
        
        # Assigning a BinOp to a Name (line 706):
        
        # Assigning a BinOp to a Name (line 706):
        # Getting the type of 'height' (line 706)
        height_251652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 13), 'height')
        # Getting the type of 'y1' (line 706)
        y1_251653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 22), 'y1')
        # Applying the binary operator '-' (line 706)
        result_sub_251654 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 13), '-', height_251652, y1_251653)
        
        # Assigning a type to the variable 'y1' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'y1', result_sub_251654)
        
        # Assigning a BinOp to a Name (line 707):
        
        # Assigning a BinOp to a Name (line 707):
        # Getting the type of 'height' (line 707)
        height_251655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 13), 'height')
        # Getting the type of 'y0' (line 707)
        y0_251656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 22), 'y0')
        # Applying the binary operator '-' (line 707)
        result_sub_251657 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 13), '-', height_251655, y0_251656)
        
        # Assigning a type to the variable 'y0' (line 707)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'y0', result_sub_251657)
        
        # Assigning a ListComp to a Name (line 708):
        
        # Assigning a ListComp to a Name (line 708):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'tuple' (line 708)
        tuple_251662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 708)
        # Adding element type (line 708)
        # Getting the type of 'x0' (line 708)
        x0_251663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 37), 'x0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 37), tuple_251662, x0_251663)
        # Adding element type (line 708)
        # Getting the type of 'y0' (line 708)
        y0_251664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 41), 'y0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 37), tuple_251662, y0_251664)
        # Adding element type (line 708)
        # Getting the type of 'x1' (line 708)
        x1_251665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 45), 'x1')
        # Getting the type of 'x0' (line 708)
        x0_251666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 50), 'x0')
        # Applying the binary operator '-' (line 708)
        result_sub_251667 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 45), '-', x1_251665, x0_251666)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 37), tuple_251662, result_sub_251667)
        # Adding element type (line 708)
        # Getting the type of 'y1' (line 708)
        y1_251668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 54), 'y1')
        # Getting the type of 'y0' (line 708)
        y0_251669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 59), 'y0')
        # Applying the binary operator '-' (line 708)
        result_sub_251670 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 54), '-', y1_251668, y0_251669)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 37), tuple_251662, result_sub_251670)
        
        comprehension_251671 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 16), tuple_251662)
        # Assigning a type to the variable 'val' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 16), 'val', comprehension_251671)
        
        # Call to int(...): (line 708)
        # Processing the call arguments (line 708)
        # Getting the type of 'val' (line 708)
        val_251659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 20), 'val', False)
        # Processing the call keyword arguments (line 708)
        kwargs_251660 = {}
        # Getting the type of 'int' (line 708)
        int_251658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 16), 'int', False)
        # Calling int(args, kwargs) (line 708)
        int_call_result_251661 = invoke(stypy.reporting.localization.Localization(__file__, 708, 16), int_251658, *[val_251659], **kwargs_251660)
        
        list_251672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 16), list_251672, int_call_result_251661)
        # Assigning a type to the variable 'rect' (line 708)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'rect', list_251672)
        
        # Call to drawRectangle(...): (line 709)
        # Processing the call arguments (line 709)
        # Getting the type of 'rect' (line 709)
        rect_251676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 34), 'rect', False)
        # Processing the call keyword arguments (line 709)
        kwargs_251677 = {}
        # Getting the type of 'self' (line 709)
        self_251673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 709)
        canvas_251674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 8), self_251673, 'canvas')
        # Obtaining the member 'drawRectangle' of a type (line 709)
        drawRectangle_251675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 8), canvas_251674, 'drawRectangle')
        # Calling drawRectangle(args, kwargs) (line 709)
        drawRectangle_call_result_251678 = invoke(stypy.reporting.localization.Localization(__file__, 709, 8), drawRectangle_251675, *[rect_251676], **kwargs_251677)
        
        
        # ################# End of 'draw_rubberband(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_rubberband' in the type store
        # Getting the type of 'stypy_return_type' (line 704)
        stypy_return_type_251679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_rubberband'
        return stypy_return_type_251679


    @norecursion
    def remove_rubberband(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove_rubberband'
        module_type_store = module_type_store.open_function_context('remove_rubberband', 711, 4, False)
        # Assigning a type to the variable 'self' (line 712)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT.remove_rubberband.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT.remove_rubberband.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT.remove_rubberband.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT.remove_rubberband.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT.remove_rubberband')
        NavigationToolbar2QT.remove_rubberband.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2QT.remove_rubberband.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2QT.remove_rubberband.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT.remove_rubberband.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT.remove_rubberband.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT.remove_rubberband.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT.remove_rubberband.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT.remove_rubberband', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to drawRectangle(...): (line 712)
        # Processing the call arguments (line 712)
        # Getting the type of 'None' (line 712)
        None_251683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 34), 'None', False)
        # Processing the call keyword arguments (line 712)
        kwargs_251684 = {}
        # Getting the type of 'self' (line 712)
        self_251680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 712)
        canvas_251681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 8), self_251680, 'canvas')
        # Obtaining the member 'drawRectangle' of a type (line 712)
        drawRectangle_251682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 8), canvas_251681, 'drawRectangle')
        # Calling drawRectangle(args, kwargs) (line 712)
        drawRectangle_call_result_251685 = invoke(stypy.reporting.localization.Localization(__file__, 712, 8), drawRectangle_251682, *[None_251683], **kwargs_251684)
        
        
        # ################# End of 'remove_rubberband(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove_rubberband' in the type store
        # Getting the type of 'stypy_return_type' (line 711)
        stypy_return_type_251686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251686)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove_rubberband'
        return stypy_return_type_251686


    @norecursion
    def configure_subplots(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'configure_subplots'
        module_type_store = module_type_store.open_function_context('configure_subplots', 714, 4, False)
        # Assigning a type to the variable 'self' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT.configure_subplots.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT.configure_subplots.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT.configure_subplots.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT.configure_subplots.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT.configure_subplots')
        NavigationToolbar2QT.configure_subplots.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2QT.configure_subplots.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2QT.configure_subplots.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT.configure_subplots.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT.configure_subplots.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT.configure_subplots.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT.configure_subplots.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT.configure_subplots', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'configure_subplots', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'configure_subplots(...)' code ##################

        
        # Assigning a Call to a Name (line 715):
        
        # Assigning a Call to a Name (line 715):
        
        # Call to join(...): (line 715)
        # Processing the call arguments (line 715)
        
        # Obtaining the type of the subscript
        unicode_251690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 49), 'unicode', u'datapath')
        # Getting the type of 'matplotlib' (line 715)
        matplotlib_251691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 29), 'matplotlib', False)
        # Obtaining the member 'rcParams' of a type (line 715)
        rcParams_251692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 29), matplotlib_251691, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 715)
        getitem___251693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 29), rcParams_251692, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 715)
        subscript_call_result_251694 = invoke(stypy.reporting.localization.Localization(__file__, 715, 29), getitem___251693, unicode_251690)
        
        unicode_251695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 29), 'unicode', u'images')
        unicode_251696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 39), 'unicode', u'matplotlib.png')
        # Processing the call keyword arguments (line 715)
        kwargs_251697 = {}
        # Getting the type of 'os' (line 715)
        os_251687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 715)
        path_251688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 16), os_251687, 'path')
        # Obtaining the member 'join' of a type (line 715)
        join_251689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 16), path_251688, 'join')
        # Calling join(args, kwargs) (line 715)
        join_call_result_251698 = invoke(stypy.reporting.localization.Localization(__file__, 715, 16), join_251689, *[subscript_call_result_251694, unicode_251695, unicode_251696], **kwargs_251697)
        
        # Assigning a type to the variable 'image' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'image', join_call_result_251698)
        
        # Assigning a Call to a Name (line 717):
        
        # Assigning a Call to a Name (line 717):
        
        # Call to SubplotToolQt(...): (line 717)
        # Processing the call arguments (line 717)
        # Getting the type of 'self' (line 717)
        self_251700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 28), 'self', False)
        # Obtaining the member 'canvas' of a type (line 717)
        canvas_251701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 28), self_251700, 'canvas')
        # Obtaining the member 'figure' of a type (line 717)
        figure_251702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 28), canvas_251701, 'figure')
        # Getting the type of 'self' (line 717)
        self_251703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 48), 'self', False)
        # Obtaining the member 'parent' of a type (line 717)
        parent_251704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 48), self_251703, 'parent')
        # Processing the call keyword arguments (line 717)
        kwargs_251705 = {}
        # Getting the type of 'SubplotToolQt' (line 717)
        SubplotToolQt_251699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 14), 'SubplotToolQt', False)
        # Calling SubplotToolQt(args, kwargs) (line 717)
        SubplotToolQt_call_result_251706 = invoke(stypy.reporting.localization.Localization(__file__, 717, 14), SubplotToolQt_251699, *[figure_251702, parent_251704], **kwargs_251705)
        
        # Assigning a type to the variable 'dia' (line 717)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'dia', SubplotToolQt_call_result_251706)
        
        # Call to setWindowIcon(...): (line 718)
        # Processing the call arguments (line 718)
        
        # Call to QIcon(...): (line 718)
        # Processing the call arguments (line 718)
        # Getting the type of 'image' (line 718)
        image_251711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 38), 'image', False)
        # Processing the call keyword arguments (line 718)
        kwargs_251712 = {}
        # Getting the type of 'QtGui' (line 718)
        QtGui_251709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 26), 'QtGui', False)
        # Obtaining the member 'QIcon' of a type (line 718)
        QIcon_251710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 26), QtGui_251709, 'QIcon')
        # Calling QIcon(args, kwargs) (line 718)
        QIcon_call_result_251713 = invoke(stypy.reporting.localization.Localization(__file__, 718, 26), QIcon_251710, *[image_251711], **kwargs_251712)
        
        # Processing the call keyword arguments (line 718)
        kwargs_251714 = {}
        # Getting the type of 'dia' (line 718)
        dia_251707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'dia', False)
        # Obtaining the member 'setWindowIcon' of a type (line 718)
        setWindowIcon_251708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 8), dia_251707, 'setWindowIcon')
        # Calling setWindowIcon(args, kwargs) (line 718)
        setWindowIcon_call_result_251715 = invoke(stypy.reporting.localization.Localization(__file__, 718, 8), setWindowIcon_251708, *[QIcon_call_result_251713], **kwargs_251714)
        
        
        # Call to exec_(...): (line 719)
        # Processing the call keyword arguments (line 719)
        kwargs_251718 = {}
        # Getting the type of 'dia' (line 719)
        dia_251716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'dia', False)
        # Obtaining the member 'exec_' of a type (line 719)
        exec__251717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 8), dia_251716, 'exec_')
        # Calling exec_(args, kwargs) (line 719)
        exec__call_result_251719 = invoke(stypy.reporting.localization.Localization(__file__, 719, 8), exec__251717, *[], **kwargs_251718)
        
        
        # ################# End of 'configure_subplots(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'configure_subplots' in the type store
        # Getting the type of 'stypy_return_type' (line 714)
        stypy_return_type_251720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251720)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'configure_subplots'
        return stypy_return_type_251720


    @norecursion
    def save_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'save_figure'
        module_type_store = module_type_store.open_function_context('save_figure', 721, 4, False)
        # Assigning a type to the variable 'self' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2QT.save_figure.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2QT.save_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2QT.save_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2QT.save_figure.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2QT.save_figure')
        NavigationToolbar2QT.save_figure.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2QT.save_figure.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        NavigationToolbar2QT.save_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2QT.save_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2QT.save_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2QT.save_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2QT.save_figure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2QT.save_figure', [], 'args', None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 722):
        
        # Assigning a Call to a Name (line 722):
        
        # Call to get_supported_filetypes_grouped(...): (line 722)
        # Processing the call keyword arguments (line 722)
        kwargs_251724 = {}
        # Getting the type of 'self' (line 722)
        self_251721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 20), 'self', False)
        # Obtaining the member 'canvas' of a type (line 722)
        canvas_251722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 20), self_251721, 'canvas')
        # Obtaining the member 'get_supported_filetypes_grouped' of a type (line 722)
        get_supported_filetypes_grouped_251723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 20), canvas_251722, 'get_supported_filetypes_grouped')
        # Calling get_supported_filetypes_grouped(args, kwargs) (line 722)
        get_supported_filetypes_grouped_call_result_251725 = invoke(stypy.reporting.localization.Localization(__file__, 722, 20), get_supported_filetypes_grouped_251723, *[], **kwargs_251724)
        
        # Assigning a type to the variable 'filetypes' (line 722)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'filetypes', get_supported_filetypes_grouped_call_result_251725)
        
        # Assigning a Call to a Name (line 723):
        
        # Assigning a Call to a Name (line 723):
        
        # Call to sorted(...): (line 723)
        # Processing the call arguments (line 723)
        
        # Call to iteritems(...): (line 723)
        # Processing the call arguments (line 723)
        # Getting the type of 'filetypes' (line 723)
        filetypes_251729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 48), 'filetypes', False)
        # Processing the call keyword arguments (line 723)
        kwargs_251730 = {}
        # Getting the type of 'six' (line 723)
        six_251727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 34), 'six', False)
        # Obtaining the member 'iteritems' of a type (line 723)
        iteritems_251728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 34), six_251727, 'iteritems')
        # Calling iteritems(args, kwargs) (line 723)
        iteritems_call_result_251731 = invoke(stypy.reporting.localization.Localization(__file__, 723, 34), iteritems_251728, *[filetypes_251729], **kwargs_251730)
        
        # Processing the call keyword arguments (line 723)
        kwargs_251732 = {}
        # Getting the type of 'sorted' (line 723)
        sorted_251726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 27), 'sorted', False)
        # Calling sorted(args, kwargs) (line 723)
        sorted_call_result_251733 = invoke(stypy.reporting.localization.Localization(__file__, 723, 27), sorted_251726, *[iteritems_call_result_251731], **kwargs_251732)
        
        # Assigning a type to the variable 'sorted_filetypes' (line 723)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'sorted_filetypes', sorted_call_result_251733)
        
        # Assigning a Call to a Name (line 724):
        
        # Assigning a Call to a Name (line 724):
        
        # Call to get_default_filetype(...): (line 724)
        # Processing the call keyword arguments (line 724)
        kwargs_251737 = {}
        # Getting the type of 'self' (line 724)
        self_251734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 27), 'self', False)
        # Obtaining the member 'canvas' of a type (line 724)
        canvas_251735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 27), self_251734, 'canvas')
        # Obtaining the member 'get_default_filetype' of a type (line 724)
        get_default_filetype_251736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 724, 27), canvas_251735, 'get_default_filetype')
        # Calling get_default_filetype(args, kwargs) (line 724)
        get_default_filetype_call_result_251738 = invoke(stypy.reporting.localization.Localization(__file__, 724, 27), get_default_filetype_251736, *[], **kwargs_251737)
        
        # Assigning a type to the variable 'default_filetype' (line 724)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'default_filetype', get_default_filetype_call_result_251738)
        
        # Assigning a Call to a Name (line 726):
        
        # Assigning a Call to a Name (line 726):
        
        # Call to expanduser(...): (line 726)
        # Processing the call arguments (line 726)
        
        # Obtaining the type of the subscript
        unicode_251742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 32), 'unicode', u'savefig.directory')
        # Getting the type of 'matplotlib' (line 727)
        matplotlib_251743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 12), 'matplotlib', False)
        # Obtaining the member 'rcParams' of a type (line 727)
        rcParams_251744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 12), matplotlib_251743, 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 727)
        getitem___251745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 12), rcParams_251744, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 727)
        subscript_call_result_251746 = invoke(stypy.reporting.localization.Localization(__file__, 727, 12), getitem___251745, unicode_251742)
        
        # Processing the call keyword arguments (line 726)
        kwargs_251747 = {}
        # Getting the type of 'os' (line 726)
        os_251739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 726)
        path_251740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 20), os_251739, 'path')
        # Obtaining the member 'expanduser' of a type (line 726)
        expanduser_251741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 20), path_251740, 'expanduser')
        # Calling expanduser(args, kwargs) (line 726)
        expanduser_call_result_251748 = invoke(stypy.reporting.localization.Localization(__file__, 726, 20), expanduser_251741, *[subscript_call_result_251746], **kwargs_251747)
        
        # Assigning a type to the variable 'startpath' (line 726)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'startpath', expanduser_call_result_251748)
        
        # Assigning a Call to a Name (line 728):
        
        # Assigning a Call to a Name (line 728):
        
        # Call to join(...): (line 728)
        # Processing the call arguments (line 728)
        # Getting the type of 'startpath' (line 728)
        startpath_251752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 29), 'startpath', False)
        
        # Call to get_default_filename(...): (line 728)
        # Processing the call keyword arguments (line 728)
        kwargs_251756 = {}
        # Getting the type of 'self' (line 728)
        self_251753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 40), 'self', False)
        # Obtaining the member 'canvas' of a type (line 728)
        canvas_251754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 40), self_251753, 'canvas')
        # Obtaining the member 'get_default_filename' of a type (line 728)
        get_default_filename_251755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 40), canvas_251754, 'get_default_filename')
        # Calling get_default_filename(args, kwargs) (line 728)
        get_default_filename_call_result_251757 = invoke(stypy.reporting.localization.Localization(__file__, 728, 40), get_default_filename_251755, *[], **kwargs_251756)
        
        # Processing the call keyword arguments (line 728)
        kwargs_251758 = {}
        # Getting the type of 'os' (line 728)
        os_251749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 728)
        path_251750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 16), os_251749, 'path')
        # Obtaining the member 'join' of a type (line 728)
        join_251751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 16), path_251750, 'join')
        # Calling join(args, kwargs) (line 728)
        join_call_result_251759 = invoke(stypy.reporting.localization.Localization(__file__, 728, 16), join_251751, *[startpath_251752, get_default_filename_call_result_251757], **kwargs_251758)
        
        # Assigning a type to the variable 'start' (line 728)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 8), 'start', join_call_result_251759)
        
        # Assigning a List to a Name (line 729):
        
        # Assigning a List to a Name (line 729):
        
        # Obtaining an instance of the builtin type 'list' (line 729)
        list_251760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 729)
        
        # Assigning a type to the variable 'filters' (line 729)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 8), 'filters', list_251760)
        
        # Assigning a Name to a Name (line 730):
        
        # Assigning a Name to a Name (line 730):
        # Getting the type of 'None' (line 730)
        None_251761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 25), 'None')
        # Assigning a type to the variable 'selectedFilter' (line 730)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'selectedFilter', None_251761)
        
        # Getting the type of 'sorted_filetypes' (line 731)
        sorted_filetypes_251762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 26), 'sorted_filetypes')
        # Testing the type of a for loop iterable (line 731)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 731, 8), sorted_filetypes_251762)
        # Getting the type of the for loop variable (line 731)
        for_loop_var_251763 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 731, 8), sorted_filetypes_251762)
        # Assigning a type to the variable 'name' (line 731)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 8), for_loop_var_251763))
        # Assigning a type to the variable 'exts' (line 731)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 8), 'exts', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 8), for_loop_var_251763))
        # SSA begins for a for statement (line 731)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 732):
        
        # Assigning a Call to a Name (line 732):
        
        # Call to join(...): (line 732)
        # Processing the call arguments (line 732)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'exts' (line 732)
        exts_251769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 58), 'exts', False)
        comprehension_251770 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 34), exts_251769)
        # Assigning a type to the variable 'ext' (line 732)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 34), 'ext', comprehension_251770)
        unicode_251766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 34), 'unicode', u'*.%s')
        # Getting the type of 'ext' (line 732)
        ext_251767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 43), 'ext', False)
        # Applying the binary operator '%' (line 732)
        result_mod_251768 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 34), '%', unicode_251766, ext_251767)
        
        list_251771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 34), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 34), list_251771, result_mod_251768)
        # Processing the call keyword arguments (line 732)
        kwargs_251772 = {}
        unicode_251764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 24), 'unicode', u' ')
        # Obtaining the member 'join' of a type (line 732)
        join_251765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 24), unicode_251764, 'join')
        # Calling join(args, kwargs) (line 732)
        join_call_result_251773 = invoke(stypy.reporting.localization.Localization(__file__, 732, 24), join_251765, *[list_251771], **kwargs_251772)
        
        # Assigning a type to the variable 'exts_list' (line 732)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 12), 'exts_list', join_call_result_251773)
        
        # Assigning a BinOp to a Name (line 733):
        
        # Assigning a BinOp to a Name (line 733):
        unicode_251774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 21), 'unicode', u'%s (%s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 733)
        tuple_251775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 733)
        # Adding element type (line 733)
        # Getting the type of 'name' (line 733)
        name_251776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 34), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 34), tuple_251775, name_251776)
        # Adding element type (line 733)
        # Getting the type of 'exts_list' (line 733)
        exts_list_251777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 40), 'exts_list')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 34), tuple_251775, exts_list_251777)
        
        # Applying the binary operator '%' (line 733)
        result_mod_251778 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 21), '%', unicode_251774, tuple_251775)
        
        # Assigning a type to the variable 'filter' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 12), 'filter', result_mod_251778)
        
        
        # Getting the type of 'default_filetype' (line 734)
        default_filetype_251779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 15), 'default_filetype')
        # Getting the type of 'exts' (line 734)
        exts_251780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 35), 'exts')
        # Applying the binary operator 'in' (line 734)
        result_contains_251781 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 15), 'in', default_filetype_251779, exts_251780)
        
        # Testing the type of an if condition (line 734)
        if_condition_251782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 734, 12), result_contains_251781)
        # Assigning a type to the variable 'if_condition_251782' (line 734)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 12), 'if_condition_251782', if_condition_251782)
        # SSA begins for if statement (line 734)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 735):
        
        # Assigning a Name to a Name (line 735):
        # Getting the type of 'filter' (line 735)
        filter_251783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 33), 'filter')
        # Assigning a type to the variable 'selectedFilter' (line 735)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 16), 'selectedFilter', filter_251783)
        # SSA join for if statement (line 734)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 736)
        # Processing the call arguments (line 736)
        # Getting the type of 'filter' (line 736)
        filter_251786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 27), 'filter', False)
        # Processing the call keyword arguments (line 736)
        kwargs_251787 = {}
        # Getting the type of 'filters' (line 736)
        filters_251784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 12), 'filters', False)
        # Obtaining the member 'append' of a type (line 736)
        append_251785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 12), filters_251784, 'append')
        # Calling append(args, kwargs) (line 736)
        append_call_result_251788 = invoke(stypy.reporting.localization.Localization(__file__, 736, 12), append_251785, *[filter_251786], **kwargs_251787)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 737):
        
        # Assigning a Call to a Name (line 737):
        
        # Call to join(...): (line 737)
        # Processing the call arguments (line 737)
        # Getting the type of 'filters' (line 737)
        filters_251791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 28), 'filters', False)
        # Processing the call keyword arguments (line 737)
        kwargs_251792 = {}
        unicode_251789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 18), 'unicode', u';;')
        # Obtaining the member 'join' of a type (line 737)
        join_251790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 18), unicode_251789, 'join')
        # Calling join(args, kwargs) (line 737)
        join_call_result_251793 = invoke(stypy.reporting.localization.Localization(__file__, 737, 18), join_251790, *[filters_251791], **kwargs_251792)
        
        # Assigning a type to the variable 'filters' (line 737)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 8), 'filters', join_call_result_251793)
        
        # Assigning a Call to a Tuple (line 739):
        
        # Assigning a Call to a Name:
        
        # Call to _getSaveFileName(...): (line 739)
        # Processing the call arguments (line 739)
        # Getting the type of 'self' (line 739)
        self_251795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 41), 'self', False)
        # Obtaining the member 'parent' of a type (line 739)
        parent_251796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 41), self_251795, 'parent')
        unicode_251797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 41), 'unicode', u'Choose a filename to save to')
        # Getting the type of 'start' (line 741)
        start_251798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 41), 'start', False)
        # Getting the type of 'filters' (line 741)
        filters_251799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 48), 'filters', False)
        # Getting the type of 'selectedFilter' (line 741)
        selectedFilter_251800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 57), 'selectedFilter', False)
        # Processing the call keyword arguments (line 739)
        kwargs_251801 = {}
        # Getting the type of '_getSaveFileName' (line 739)
        _getSaveFileName_251794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 24), '_getSaveFileName', False)
        # Calling _getSaveFileName(args, kwargs) (line 739)
        _getSaveFileName_call_result_251802 = invoke(stypy.reporting.localization.Localization(__file__, 739, 24), _getSaveFileName_251794, *[parent_251796, unicode_251797, start_251798, filters_251799, selectedFilter_251800], **kwargs_251801)
        
        # Assigning a type to the variable 'call_assignment_249724' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'call_assignment_249724', _getSaveFileName_call_result_251802)
        
        # Assigning a Call to a Name (line 739):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_251805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 8), 'int')
        # Processing the call keyword arguments
        kwargs_251806 = {}
        # Getting the type of 'call_assignment_249724' (line 739)
        call_assignment_249724_251803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'call_assignment_249724', False)
        # Obtaining the member '__getitem__' of a type (line 739)
        getitem___251804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 8), call_assignment_249724_251803, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_251807 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___251804, *[int_251805], **kwargs_251806)
        
        # Assigning a type to the variable 'call_assignment_249725' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'call_assignment_249725', getitem___call_result_251807)
        
        # Assigning a Name to a Name (line 739):
        # Getting the type of 'call_assignment_249725' (line 739)
        call_assignment_249725_251808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'call_assignment_249725')
        # Assigning a type to the variable 'fname' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'fname', call_assignment_249725_251808)
        
        # Assigning a Call to a Name (line 739):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_251811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 8), 'int')
        # Processing the call keyword arguments
        kwargs_251812 = {}
        # Getting the type of 'call_assignment_249724' (line 739)
        call_assignment_249724_251809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'call_assignment_249724', False)
        # Obtaining the member '__getitem__' of a type (line 739)
        getitem___251810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 8), call_assignment_249724_251809, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_251813 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___251810, *[int_251811], **kwargs_251812)
        
        # Assigning a type to the variable 'call_assignment_249726' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'call_assignment_249726', getitem___call_result_251813)
        
        # Assigning a Name to a Name (line 739):
        # Getting the type of 'call_assignment_249726' (line 739)
        call_assignment_249726_251814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'call_assignment_249726')
        # Assigning a type to the variable 'filter' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 15), 'filter', call_assignment_249726_251814)
        
        # Getting the type of 'fname' (line 742)
        fname_251815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 11), 'fname')
        # Testing the type of an if condition (line 742)
        if_condition_251816 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 742, 8), fname_251815)
        # Assigning a type to the variable 'if_condition_251816' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'if_condition_251816', if_condition_251816)
        # SSA begins for if statement (line 742)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'startpath' (line 744)
        startpath_251817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 15), 'startpath')
        unicode_251818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 28), 'unicode', u'')
        # Applying the binary operator '!=' (line 744)
        result_ne_251819 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 15), '!=', startpath_251817, unicode_251818)
        
        # Testing the type of an if condition (line 744)
        if_condition_251820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 744, 12), result_ne_251819)
        # Assigning a type to the variable 'if_condition_251820' (line 744)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 12), 'if_condition_251820', if_condition_251820)
        # SSA begins for if statement (line 744)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 745):
        
        # Assigning a Call to a Subscript (line 745):
        
        # Call to dirname(...): (line 746)
        # Processing the call arguments (line 746)
        
        # Call to text_type(...): (line 746)
        # Processing the call arguments (line 746)
        # Getting the type of 'fname' (line 746)
        fname_251826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 50), 'fname', False)
        # Processing the call keyword arguments (line 746)
        kwargs_251827 = {}
        # Getting the type of 'six' (line 746)
        six_251824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 36), 'six', False)
        # Obtaining the member 'text_type' of a type (line 746)
        text_type_251825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 36), six_251824, 'text_type')
        # Calling text_type(args, kwargs) (line 746)
        text_type_call_result_251828 = invoke(stypy.reporting.localization.Localization(__file__, 746, 36), text_type_251825, *[fname_251826], **kwargs_251827)
        
        # Processing the call keyword arguments (line 746)
        kwargs_251829 = {}
        # Getting the type of 'os' (line 746)
        os_251821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 746)
        path_251822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 20), os_251821, 'path')
        # Obtaining the member 'dirname' of a type (line 746)
        dirname_251823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 20), path_251822, 'dirname')
        # Calling dirname(args, kwargs) (line 746)
        dirname_call_result_251830 = invoke(stypy.reporting.localization.Localization(__file__, 746, 20), dirname_251823, *[text_type_call_result_251828], **kwargs_251829)
        
        # Getting the type of 'matplotlib' (line 745)
        matplotlib_251831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 16), 'matplotlib')
        # Obtaining the member 'rcParams' of a type (line 745)
        rcParams_251832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 16), matplotlib_251831, 'rcParams')
        unicode_251833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 36), 'unicode', u'savefig.directory')
        # Storing an element on a container (line 745)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 16), rcParams_251832, (unicode_251833, dirname_call_result_251830))
        # SSA join for if statement (line 744)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 747)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to savefig(...): (line 748)
        # Processing the call arguments (line 748)
        
        # Call to text_type(...): (line 748)
        # Processing the call arguments (line 748)
        # Getting the type of 'fname' (line 748)
        fname_251840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 57), 'fname', False)
        # Processing the call keyword arguments (line 748)
        kwargs_251841 = {}
        # Getting the type of 'six' (line 748)
        six_251838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 43), 'six', False)
        # Obtaining the member 'text_type' of a type (line 748)
        text_type_251839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 43), six_251838, 'text_type')
        # Calling text_type(args, kwargs) (line 748)
        text_type_call_result_251842 = invoke(stypy.reporting.localization.Localization(__file__, 748, 43), text_type_251839, *[fname_251840], **kwargs_251841)
        
        # Processing the call keyword arguments (line 748)
        kwargs_251843 = {}
        # Getting the type of 'self' (line 748)
        self_251834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 16), 'self', False)
        # Obtaining the member 'canvas' of a type (line 748)
        canvas_251835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 16), self_251834, 'canvas')
        # Obtaining the member 'figure' of a type (line 748)
        figure_251836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 16), canvas_251835, 'figure')
        # Obtaining the member 'savefig' of a type (line 748)
        savefig_251837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 16), figure_251836, 'savefig')
        # Calling savefig(args, kwargs) (line 748)
        savefig_call_result_251844 = invoke(stypy.reporting.localization.Localization(__file__, 748, 16), savefig_251837, *[text_type_call_result_251842], **kwargs_251843)
        
        # SSA branch for the except part of a try statement (line 747)
        # SSA branch for the except 'Exception' branch of a try statement (line 747)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 749)
        Exception_251845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 19), 'Exception')
        # Assigning a type to the variable 'e' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 12), 'e', Exception_251845)
        
        # Call to critical(...): (line 750)
        # Processing the call arguments (line 750)
        # Getting the type of 'self' (line 751)
        self_251849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 20), 'self', False)
        unicode_251850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 26), 'unicode', u'Error saving file')
        
        # Call to text_type(...): (line 751)
        # Processing the call arguments (line 751)
        # Getting the type of 'e' (line 751)
        e_251853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 61), 'e', False)
        # Processing the call keyword arguments (line 751)
        kwargs_251854 = {}
        # Getting the type of 'six' (line 751)
        six_251851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 47), 'six', False)
        # Obtaining the member 'text_type' of a type (line 751)
        text_type_251852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 47), six_251851, 'text_type')
        # Calling text_type(args, kwargs) (line 751)
        text_type_call_result_251855 = invoke(stypy.reporting.localization.Localization(__file__, 751, 47), text_type_251852, *[e_251853], **kwargs_251854)
        
        # Getting the type of 'QtWidgets' (line 752)
        QtWidgets_251856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 20), 'QtWidgets', False)
        # Obtaining the member 'QMessageBox' of a type (line 752)
        QMessageBox_251857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 20), QtWidgets_251856, 'QMessageBox')
        # Obtaining the member 'Ok' of a type (line 752)
        Ok_251858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 20), QMessageBox_251857, 'Ok')
        # Getting the type of 'QtWidgets' (line 752)
        QtWidgets_251859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 46), 'QtWidgets', False)
        # Obtaining the member 'QMessageBox' of a type (line 752)
        QMessageBox_251860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 46), QtWidgets_251859, 'QMessageBox')
        # Obtaining the member 'NoButton' of a type (line 752)
        NoButton_251861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 46), QMessageBox_251860, 'NoButton')
        # Processing the call keyword arguments (line 750)
        kwargs_251862 = {}
        # Getting the type of 'QtWidgets' (line 750)
        QtWidgets_251846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 16), 'QtWidgets', False)
        # Obtaining the member 'QMessageBox' of a type (line 750)
        QMessageBox_251847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 16), QtWidgets_251846, 'QMessageBox')
        # Obtaining the member 'critical' of a type (line 750)
        critical_251848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 16), QMessageBox_251847, 'critical')
        # Calling critical(args, kwargs) (line 750)
        critical_call_result_251863 = invoke(stypy.reporting.localization.Localization(__file__, 750, 16), critical_251848, *[self_251849, unicode_251850, text_type_call_result_251855, Ok_251858, NoButton_251861], **kwargs_251862)
        
        # SSA join for try-except statement (line 747)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 742)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'save_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'save_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 721)
        stypy_return_type_251864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_251864)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'save_figure'
        return stypy_return_type_251864


# Assigning a type to the variable 'NavigationToolbar2QT' (line 582)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 0), 'NavigationToolbar2QT', NavigationToolbar2QT)

# Assigning a Call to a Name (line 583):

# Call to Signal(...): (line 583)
# Processing the call arguments (line 583)
# Getting the type of 'str' (line 583)
str_251867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 28), 'str', False)
# Processing the call keyword arguments (line 583)
kwargs_251868 = {}
# Getting the type of 'QtCore' (line 583)
QtCore_251865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 14), 'QtCore', False)
# Obtaining the member 'Signal' of a type (line 583)
Signal_251866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 14), QtCore_251865, 'Signal')
# Calling Signal(args, kwargs) (line 583)
Signal_call_result_251869 = invoke(stypy.reporting.localization.Localization(__file__, 583, 14), Signal_251866, *[str_251867], **kwargs_251868)

# Getting the type of 'NavigationToolbar2QT'
NavigationToolbar2QT_251870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NavigationToolbar2QT')
# Setting the type of the member 'message' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NavigationToolbar2QT_251870, 'message', Signal_call_result_251869)

# Assigning a Call to a Name (line 583):


# Call to is_pyqt5(...): (line 648)
# Processing the call keyword arguments (line 648)
kwargs_251872 = {}
# Getting the type of 'is_pyqt5' (line 648)
is_pyqt5_251871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 7), 'is_pyqt5', False)
# Calling is_pyqt5(args, kwargs) (line 648)
is_pyqt5_call_result_251873 = invoke(stypy.reporting.localization.Localization(__file__, 648, 7), is_pyqt5_251871, *[], **kwargs_251872)

# Testing the type of an if condition (line 648)
if_condition_251874 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 648, 4), is_pyqt5_call_result_251873)
# Assigning a type to the variable 'if_condition_251874' (line 648)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'if_condition_251874', if_condition_251874)
# SSA begins for if statement (line 648)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def sizeHint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sizeHint'
    module_type_store = module_type_store.open_function_context('sizeHint', 652, 8, False)
    
    # Passed parameters checking function
    sizeHint.stypy_localization = localization
    sizeHint.stypy_type_of_self = None
    sizeHint.stypy_type_store = module_type_store
    sizeHint.stypy_function_name = 'sizeHint'
    sizeHint.stypy_param_names_list = ['self']
    sizeHint.stypy_varargs_param_name = None
    sizeHint.stypy_kwargs_param_name = None
    sizeHint.stypy_call_defaults = defaults
    sizeHint.stypy_call_varargs = varargs
    sizeHint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sizeHint', ['self'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sizeHint', localization, ['self'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sizeHint(...)' code ##################

    
    # Assigning a Call to a Name (line 653):
    
    # Assigning a Call to a Name (line 653):
    
    # Call to sizeHint(...): (line 653)
    # Processing the call keyword arguments (line 653)
    kwargs_251881 = {}
    
    # Call to super(...): (line 653)
    # Processing the call arguments (line 653)
    # Getting the type of 'NavigationToolbar2QT' (line 653)
    NavigationToolbar2QT_251876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 25), 'NavigationToolbar2QT', False)
    # Getting the type of 'self' (line 653)
    self_251877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 47), 'self', False)
    # Processing the call keyword arguments (line 653)
    kwargs_251878 = {}
    # Getting the type of 'super' (line 653)
    super_251875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 19), 'super', False)
    # Calling super(args, kwargs) (line 653)
    super_call_result_251879 = invoke(stypy.reporting.localization.Localization(__file__, 653, 19), super_251875, *[NavigationToolbar2QT_251876, self_251877], **kwargs_251878)
    
    # Obtaining the member 'sizeHint' of a type (line 653)
    sizeHint_251880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 19), super_call_result_251879, 'sizeHint')
    # Calling sizeHint(args, kwargs) (line 653)
    sizeHint_call_result_251882 = invoke(stypy.reporting.localization.Localization(__file__, 653, 19), sizeHint_251880, *[], **kwargs_251881)
    
    # Assigning a type to the variable 'size' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 12), 'size', sizeHint_call_result_251882)
    
    # Call to setHeight(...): (line 654)
    # Processing the call arguments (line 654)
    
    # Call to max(...): (line 654)
    # Processing the call arguments (line 654)
    int_251886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 31), 'int')
    
    # Call to height(...): (line 654)
    # Processing the call keyword arguments (line 654)
    kwargs_251889 = {}
    # Getting the type of 'size' (line 654)
    size_251887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 35), 'size', False)
    # Obtaining the member 'height' of a type (line 654)
    height_251888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 35), size_251887, 'height')
    # Calling height(args, kwargs) (line 654)
    height_call_result_251890 = invoke(stypy.reporting.localization.Localization(__file__, 654, 35), height_251888, *[], **kwargs_251889)
    
    # Processing the call keyword arguments (line 654)
    kwargs_251891 = {}
    # Getting the type of 'max' (line 654)
    max_251885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 27), 'max', False)
    # Calling max(args, kwargs) (line 654)
    max_call_result_251892 = invoke(stypy.reporting.localization.Localization(__file__, 654, 27), max_251885, *[int_251886, height_call_result_251890], **kwargs_251891)
    
    # Processing the call keyword arguments (line 654)
    kwargs_251893 = {}
    # Getting the type of 'size' (line 654)
    size_251883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'size', False)
    # Obtaining the member 'setHeight' of a type (line 654)
    setHeight_251884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 12), size_251883, 'setHeight')
    # Calling setHeight(args, kwargs) (line 654)
    setHeight_call_result_251894 = invoke(stypy.reporting.localization.Localization(__file__, 654, 12), setHeight_251884, *[max_call_result_251892], **kwargs_251893)
    
    # Getting the type of 'size' (line 655)
    size_251895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 19), 'size')
    # Assigning a type to the variable 'stypy_return_type' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 12), 'stypy_return_type', size_251895)
    
    # ################# End of 'sizeHint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sizeHint' in the type store
    # Getting the type of 'stypy_return_type' (line 652)
    stypy_return_type_251896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_251896)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sizeHint'
    return stypy_return_type_251896

# Assigning a type to the variable 'sizeHint' (line 652)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'sizeHint', sizeHint)
# SSA join for if statement (line 648)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'SubplotToolQt' class
# Getting the type of 'UiSubplotTool' (line 755)
UiSubplotTool_251897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 20), 'UiSubplotTool')

class SubplotToolQt(UiSubplotTool_251897, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 756, 4, False)
        # Assigning a type to the variable 'self' (line 757)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotToolQt.__init__', ['targetfig', 'parent'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['targetfig', 'parent'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 757)
        # Processing the call arguments (line 757)
        # Getting the type of 'self' (line 757)
        self_251900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 31), 'self', False)
        # Getting the type of 'None' (line 757)
        None_251901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 37), 'None', False)
        # Processing the call keyword arguments (line 757)
        kwargs_251902 = {}
        # Getting the type of 'UiSubplotTool' (line 757)
        UiSubplotTool_251898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 8), 'UiSubplotTool', False)
        # Obtaining the member '__init__' of a type (line 757)
        init___251899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 8), UiSubplotTool_251898, '__init__')
        # Calling __init__(args, kwargs) (line 757)
        init___call_result_251903 = invoke(stypy.reporting.localization.Localization(__file__, 757, 8), init___251899, *[self_251900, None_251901], **kwargs_251902)
        
        
        # Assigning a Name to a Attribute (line 759):
        
        # Assigning a Name to a Attribute (line 759):
        # Getting the type of 'targetfig' (line 759)
        targetfig_251904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 23), 'targetfig')
        # Getting the type of 'self' (line 759)
        self_251905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'self')
        # Setting the type of the member '_figure' of a type (line 759)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 8), self_251905, '_figure', targetfig_251904)
        
        
        # Obtaining an instance of the builtin type 'list' (line 761)
        list_251906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 761)
        # Adding element type (line 761)
        
        # Obtaining an instance of the builtin type 'tuple' (line 761)
        tuple_251907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 761)
        # Adding element type (line 761)
        unicode_251908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 31), 'unicode', u'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 31), tuple_251907, unicode_251908)
        # Adding element type (line 761)
        unicode_251909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 41), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 31), tuple_251907, unicode_251909)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 29), list_251906, tuple_251907)
        # Adding element type (line 761)
        
        # Obtaining an instance of the builtin type 'tuple' (line 761)
        tuple_251910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 761)
        # Adding element type (line 761)
        unicode_251911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 50), 'unicode', u'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 50), tuple_251910, unicode_251911)
        # Adding element type (line 761)
        unicode_251912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 58), 'unicode', u'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 50), tuple_251910, unicode_251912)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 29), list_251906, tuple_251910)
        
        # Testing the type of a for loop iterable (line 761)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 761, 8), list_251906)
        # Getting the type of the for loop variable (line 761)
        for_loop_var_251913 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 761, 8), list_251906)
        # Assigning a type to the variable 'lower' (line 761)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'lower', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 8), for_loop_var_251913))
        # Assigning a type to the variable 'higher' (line 761)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'higher', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 8), for_loop_var_251913))
        # SSA begins for a for statement (line 761)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to connect(...): (line 762)
        # Processing the call arguments (line 762)

        @norecursion
        def _stypy_temp_lambda_107(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_107'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_107', 763, 16, True)
            # Passed parameters checking function
            _stypy_temp_lambda_107.stypy_localization = localization
            _stypy_temp_lambda_107.stypy_type_of_self = None
            _stypy_temp_lambda_107.stypy_type_store = module_type_store
            _stypy_temp_lambda_107.stypy_function_name = '_stypy_temp_lambda_107'
            _stypy_temp_lambda_107.stypy_param_names_list = ['val']
            _stypy_temp_lambda_107.stypy_varargs_param_name = None
            _stypy_temp_lambda_107.stypy_kwargs_param_name = None
            _stypy_temp_lambda_107.stypy_call_defaults = defaults
            _stypy_temp_lambda_107.stypy_call_varargs = varargs
            _stypy_temp_lambda_107.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_107', ['val'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_107', ['val'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to setMinimum(...): (line 763)
            # Processing the call arguments (line 763)
            # Getting the type of 'val' (line 763)
            val_251927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 61), 'val', False)
            float_251928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 67), 'float')
            # Applying the binary operator '+' (line 763)
            result_add_251929 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 61), '+', val_251927, float_251928)
            
            # Processing the call keyword arguments (line 763)
            kwargs_251930 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'higher' (line 763)
            higher_251921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 42), 'higher', False)
            # Getting the type of 'self' (line 763)
            self_251922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 28), 'self', False)
            # Obtaining the member '_widgets' of a type (line 763)
            _widgets_251923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 28), self_251922, '_widgets')
            # Obtaining the member '__getitem__' of a type (line 763)
            getitem___251924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 28), _widgets_251923, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 763)
            subscript_call_result_251925 = invoke(stypy.reporting.localization.Localization(__file__, 763, 28), getitem___251924, higher_251921)
            
            # Obtaining the member 'setMinimum' of a type (line 763)
            setMinimum_251926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 28), subscript_call_result_251925, 'setMinimum')
            # Calling setMinimum(args, kwargs) (line 763)
            setMinimum_call_result_251931 = invoke(stypy.reporting.localization.Localization(__file__, 763, 28), setMinimum_251926, *[result_add_251929], **kwargs_251930)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 763)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), 'stypy_return_type', setMinimum_call_result_251931)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_107' in the type store
            # Getting the type of 'stypy_return_type' (line 763)
            stypy_return_type_251932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_251932)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_107'
            return stypy_return_type_251932

        # Assigning a type to the variable '_stypy_temp_lambda_107' (line 763)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), '_stypy_temp_lambda_107', _stypy_temp_lambda_107)
        # Getting the type of '_stypy_temp_lambda_107' (line 763)
        _stypy_temp_lambda_107_251933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), '_stypy_temp_lambda_107')
        # Processing the call keyword arguments (line 762)
        kwargs_251934 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'lower' (line 762)
        lower_251914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 26), 'lower', False)
        # Getting the type of 'self' (line 762)
        self_251915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 12), 'self', False)
        # Obtaining the member '_widgets' of a type (line 762)
        _widgets_251916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 12), self_251915, '_widgets')
        # Obtaining the member '__getitem__' of a type (line 762)
        getitem___251917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 12), _widgets_251916, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 762)
        subscript_call_result_251918 = invoke(stypy.reporting.localization.Localization(__file__, 762, 12), getitem___251917, lower_251914)
        
        # Obtaining the member 'valueChanged' of a type (line 762)
        valueChanged_251919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 12), subscript_call_result_251918, 'valueChanged')
        # Obtaining the member 'connect' of a type (line 762)
        connect_251920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 12), valueChanged_251919, 'connect')
        # Calling connect(args, kwargs) (line 762)
        connect_call_result_251935 = invoke(stypy.reporting.localization.Localization(__file__, 762, 12), connect_251920, *[_stypy_temp_lambda_107_251933], **kwargs_251934)
        
        
        # Call to connect(...): (line 764)
        # Processing the call arguments (line 764)

        @norecursion
        def _stypy_temp_lambda_108(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_108'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_108', 765, 16, True)
            # Passed parameters checking function
            _stypy_temp_lambda_108.stypy_localization = localization
            _stypy_temp_lambda_108.stypy_type_of_self = None
            _stypy_temp_lambda_108.stypy_type_store = module_type_store
            _stypy_temp_lambda_108.stypy_function_name = '_stypy_temp_lambda_108'
            _stypy_temp_lambda_108.stypy_param_names_list = ['val']
            _stypy_temp_lambda_108.stypy_varargs_param_name = None
            _stypy_temp_lambda_108.stypy_kwargs_param_name = None
            _stypy_temp_lambda_108.stypy_call_defaults = defaults
            _stypy_temp_lambda_108.stypy_call_varargs = varargs
            _stypy_temp_lambda_108.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_108', ['val'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_108', ['val'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to setMaximum(...): (line 765)
            # Processing the call arguments (line 765)
            # Getting the type of 'val' (line 765)
            val_251949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 60), 'val', False)
            float_251950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 66), 'float')
            # Applying the binary operator '-' (line 765)
            result_sub_251951 = python_operator(stypy.reporting.localization.Localization(__file__, 765, 60), '-', val_251949, float_251950)
            
            # Processing the call keyword arguments (line 765)
            kwargs_251952 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'lower' (line 765)
            lower_251943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 42), 'lower', False)
            # Getting the type of 'self' (line 765)
            self_251944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 28), 'self', False)
            # Obtaining the member '_widgets' of a type (line 765)
            _widgets_251945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 28), self_251944, '_widgets')
            # Obtaining the member '__getitem__' of a type (line 765)
            getitem___251946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 28), _widgets_251945, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 765)
            subscript_call_result_251947 = invoke(stypy.reporting.localization.Localization(__file__, 765, 28), getitem___251946, lower_251943)
            
            # Obtaining the member 'setMaximum' of a type (line 765)
            setMaximum_251948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 28), subscript_call_result_251947, 'setMaximum')
            # Calling setMaximum(args, kwargs) (line 765)
            setMaximum_call_result_251953 = invoke(stypy.reporting.localization.Localization(__file__, 765, 28), setMaximum_251948, *[result_sub_251951], **kwargs_251952)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 765)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'stypy_return_type', setMaximum_call_result_251953)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_108' in the type store
            # Getting the type of 'stypy_return_type' (line 765)
            stypy_return_type_251954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_251954)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_108'
            return stypy_return_type_251954

        # Assigning a type to the variable '_stypy_temp_lambda_108' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), '_stypy_temp_lambda_108', _stypy_temp_lambda_108)
        # Getting the type of '_stypy_temp_lambda_108' (line 765)
        _stypy_temp_lambda_108_251955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), '_stypy_temp_lambda_108')
        # Processing the call keyword arguments (line 764)
        kwargs_251956 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'higher' (line 764)
        higher_251936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 26), 'higher', False)
        # Getting the type of 'self' (line 764)
        self_251937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'self', False)
        # Obtaining the member '_widgets' of a type (line 764)
        _widgets_251938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 12), self_251937, '_widgets')
        # Obtaining the member '__getitem__' of a type (line 764)
        getitem___251939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 12), _widgets_251938, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 764)
        subscript_call_result_251940 = invoke(stypy.reporting.localization.Localization(__file__, 764, 12), getitem___251939, higher_251936)
        
        # Obtaining the member 'valueChanged' of a type (line 764)
        valueChanged_251941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 12), subscript_call_result_251940, 'valueChanged')
        # Obtaining the member 'connect' of a type (line 764)
        connect_251942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 764, 12), valueChanged_251941, 'connect')
        # Calling connect(args, kwargs) (line 764)
        connect_call_result_251957 = invoke(stypy.reporting.localization.Localization(__file__, 764, 12), connect_251942, *[_stypy_temp_lambda_108_251955], **kwargs_251956)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 767):
        
        # Assigning a List to a Attribute (line 767):
        
        # Obtaining an instance of the builtin type 'list' (line 767)
        list_251958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 767)
        # Adding element type (line 767)
        unicode_251959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 23), 'unicode', u'top')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 22), list_251958, unicode_251959)
        # Adding element type (line 767)
        unicode_251960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 30), 'unicode', u'bottom')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 22), list_251958, unicode_251960)
        # Adding element type (line 767)
        unicode_251961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 40), 'unicode', u'left')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 22), list_251958, unicode_251961)
        # Adding element type (line 767)
        unicode_251962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 48), 'unicode', u'right')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 22), list_251958, unicode_251962)
        # Adding element type (line 767)
        unicode_251963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 57), 'unicode', u'hspace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 22), list_251958, unicode_251963)
        # Adding element type (line 767)
        unicode_251964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 67), 'unicode', u'wspace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 22), list_251958, unicode_251964)
        
        # Getting the type of 'self' (line 767)
        self_251965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'self')
        # Setting the type of the member '_attrs' of a type (line 767)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 8), self_251965, '_attrs', list_251958)
        
        # Assigning a DictComp to a Attribute (line 768):
        
        # Assigning a DictComp to a Attribute (line 768):
        # Calculating dict comprehension
        module_type_store = module_type_store.open_function_context('dict comprehension expression', 768, 26, True)
        # Calculating comprehension expression
        # Getting the type of 'self' (line 769)
        self_251976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 38), 'self')
        # Obtaining the member '_attrs' of a type (line 769)
        _attrs_251977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 38), self_251976, '_attrs')
        comprehension_251978 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 26), _attrs_251977)
        # Assigning a type to the variable 'attr' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 26), 'attr', comprehension_251978)
        # Getting the type of 'attr' (line 768)
        attr_251966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 26), 'attr')
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 768)
        attr_251967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 63), 'attr')
        
        # Call to vars(...): (line 768)
        # Processing the call arguments (line 768)
        # Getting the type of 'self' (line 768)
        self_251969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 37), 'self', False)
        # Obtaining the member '_figure' of a type (line 768)
        _figure_251970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 37), self_251969, '_figure')
        # Obtaining the member 'subplotpars' of a type (line 768)
        subplotpars_251971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 37), _figure_251970, 'subplotpars')
        # Processing the call keyword arguments (line 768)
        kwargs_251972 = {}
        # Getting the type of 'vars' (line 768)
        vars_251968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 32), 'vars', False)
        # Calling vars(args, kwargs) (line 768)
        vars_call_result_251973 = invoke(stypy.reporting.localization.Localization(__file__, 768, 32), vars_251968, *[subplotpars_251971], **kwargs_251972)
        
        # Obtaining the member '__getitem__' of a type (line 768)
        getitem___251974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 32), vars_call_result_251973, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 768)
        subscript_call_result_251975 = invoke(stypy.reporting.localization.Localization(__file__, 768, 32), getitem___251974, attr_251967)
        
        dict_251979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 26), 'dict')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 26), dict_251979, (attr_251966, subscript_call_result_251975))
        # Getting the type of 'self' (line 768)
        self_251980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'self')
        # Setting the type of the member '_defaults' of a type (line 768)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 768, 8), self_251980, '_defaults', dict_251979)
        
        # Call to _reset(...): (line 773)
        # Processing the call keyword arguments (line 773)
        kwargs_251983 = {}
        # Getting the type of 'self' (line 773)
        self_251981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'self', False)
        # Obtaining the member '_reset' of a type (line 773)
        _reset_251982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 8), self_251981, '_reset')
        # Calling _reset(args, kwargs) (line 773)
        _reset_call_result_251984 = invoke(stypy.reporting.localization.Localization(__file__, 773, 8), _reset_251982, *[], **kwargs_251983)
        
        
        # Getting the type of 'self' (line 775)
        self_251985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 20), 'self')
        # Obtaining the member '_attrs' of a type (line 775)
        _attrs_251986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 20), self_251985, '_attrs')
        # Testing the type of a for loop iterable (line 775)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 775, 8), _attrs_251986)
        # Getting the type of the for loop variable (line 775)
        for_loop_var_251987 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 775, 8), _attrs_251986)
        # Assigning a type to the variable 'attr' (line 775)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'attr', for_loop_var_251987)
        # SSA begins for a for statement (line 775)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to connect(...): (line 776)
        # Processing the call arguments (line 776)
        # Getting the type of 'self' (line 776)
        self_251995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 53), 'self', False)
        # Obtaining the member '_on_value_changed' of a type (line 776)
        _on_value_changed_251996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 53), self_251995, '_on_value_changed')
        # Processing the call keyword arguments (line 776)
        kwargs_251997 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 776)
        attr_251988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 26), 'attr', False)
        # Getting the type of 'self' (line 776)
        self_251989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 12), 'self', False)
        # Obtaining the member '_widgets' of a type (line 776)
        _widgets_251990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 12), self_251989, '_widgets')
        # Obtaining the member '__getitem__' of a type (line 776)
        getitem___251991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 12), _widgets_251990, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 776)
        subscript_call_result_251992 = invoke(stypy.reporting.localization.Localization(__file__, 776, 12), getitem___251991, attr_251988)
        
        # Obtaining the member 'valueChanged' of a type (line 776)
        valueChanged_251993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 12), subscript_call_result_251992, 'valueChanged')
        # Obtaining the member 'connect' of a type (line 776)
        connect_251994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 12), valueChanged_251993, 'connect')
        # Calling connect(args, kwargs) (line 776)
        connect_call_result_251998 = invoke(stypy.reporting.localization.Localization(__file__, 776, 12), connect_251994, *[_on_value_changed_251996], **kwargs_251997)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 777)
        list_251999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 777)
        # Adding element type (line 777)
        
        # Obtaining an instance of the builtin type 'tuple' (line 777)
        tuple_252000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 777)
        # Adding element type (line 777)
        unicode_252001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 32), 'unicode', u'Export values')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 32), tuple_252000, unicode_252001)
        # Adding element type (line 777)
        # Getting the type of 'self' (line 777)
        self_252002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 49), 'self')
        # Obtaining the member '_export_values' of a type (line 777)
        _export_values_252003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 49), self_252002, '_export_values')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 32), tuple_252000, _export_values_252003)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 30), list_251999, tuple_252000)
        # Adding element type (line 777)
        
        # Obtaining an instance of the builtin type 'tuple' (line 778)
        tuple_252004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 778)
        # Adding element type (line 778)
        unicode_252005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 32), 'unicode', u'Tight layout')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 32), tuple_252004, unicode_252005)
        # Adding element type (line 778)
        # Getting the type of 'self' (line 778)
        self_252006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 48), 'self')
        # Obtaining the member '_tight_layout' of a type (line 778)
        _tight_layout_252007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 48), self_252006, '_tight_layout')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 778, 32), tuple_252004, _tight_layout_252007)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 30), list_251999, tuple_252004)
        # Adding element type (line 777)
        
        # Obtaining an instance of the builtin type 'tuple' (line 779)
        tuple_252008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 779)
        # Adding element type (line 779)
        unicode_252009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 32), 'unicode', u'Reset')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 779, 32), tuple_252008, unicode_252009)
        # Adding element type (line 779)
        # Getting the type of 'self' (line 779)
        self_252010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 41), 'self')
        # Obtaining the member '_reset' of a type (line 779)
        _reset_252011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 41), self_252010, '_reset')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 779, 32), tuple_252008, _reset_252011)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 30), list_251999, tuple_252008)
        # Adding element type (line 777)
        
        # Obtaining an instance of the builtin type 'tuple' (line 780)
        tuple_252012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 780)
        # Adding element type (line 780)
        unicode_252013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 32), 'unicode', u'Close')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 780, 32), tuple_252012, unicode_252013)
        # Adding element type (line 780)
        # Getting the type of 'self' (line 780)
        self_252014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 41), 'self')
        # Obtaining the member 'close' of a type (line 780)
        close_252015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 41), self_252014, 'close')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 780, 32), tuple_252012, close_252015)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 30), list_251999, tuple_252012)
        
        # Testing the type of a for loop iterable (line 777)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 777, 8), list_251999)
        # Getting the type of the for loop variable (line 777)
        for_loop_var_252016 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 777, 8), list_251999)
        # Assigning a type to the variable 'action' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'action', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 8), for_loop_var_252016))
        # Assigning a type to the variable 'method' (line 777)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'method', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 777, 8), for_loop_var_252016))
        # SSA begins for a for statement (line 777)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to connect(...): (line 781)
        # Processing the call arguments (line 781)
        # Getting the type of 'method' (line 781)
        method_252024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 50), 'method', False)
        # Processing the call keyword arguments (line 781)
        kwargs_252025 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'action' (line 781)
        action_252017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 26), 'action', False)
        # Getting the type of 'self' (line 781)
        self_252018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 12), 'self', False)
        # Obtaining the member '_widgets' of a type (line 781)
        _widgets_252019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 12), self_252018, '_widgets')
        # Obtaining the member '__getitem__' of a type (line 781)
        getitem___252020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 12), _widgets_252019, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 781)
        subscript_call_result_252021 = invoke(stypy.reporting.localization.Localization(__file__, 781, 12), getitem___252020, action_252017)
        
        # Obtaining the member 'clicked' of a type (line 781)
        clicked_252022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 12), subscript_call_result_252021, 'clicked')
        # Obtaining the member 'connect' of a type (line 781)
        connect_252023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 12), clicked_252022, 'connect')
        # Calling connect(args, kwargs) (line 781)
        connect_call_result_252026 = invoke(stypy.reporting.localization.Localization(__file__, 781, 12), connect_252023, *[method_252024], **kwargs_252025)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _export_values(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_export_values'
        module_type_store = module_type_store.open_function_context('_export_values', 783, 4, False)
        # Assigning a type to the variable 'self' (line 784)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotToolQt._export_values.__dict__.__setitem__('stypy_localization', localization)
        SubplotToolQt._export_values.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotToolQt._export_values.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotToolQt._export_values.__dict__.__setitem__('stypy_function_name', 'SubplotToolQt._export_values')
        SubplotToolQt._export_values.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotToolQt._export_values.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotToolQt._export_values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotToolQt._export_values.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotToolQt._export_values.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotToolQt._export_values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotToolQt._export_values.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotToolQt._export_values', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_export_values', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_export_values(...)' code ##################

        
        # Assigning a Call to a Name (line 786):
        
        # Assigning a Call to a Name (line 786):
        
        # Call to QDialog(...): (line 786)
        # Processing the call keyword arguments (line 786)
        kwargs_252029 = {}
        # Getting the type of 'QtWidgets' (line 786)
        QtWidgets_252027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 17), 'QtWidgets', False)
        # Obtaining the member 'QDialog' of a type (line 786)
        QDialog_252028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 17), QtWidgets_252027, 'QDialog')
        # Calling QDialog(args, kwargs) (line 786)
        QDialog_call_result_252030 = invoke(stypy.reporting.localization.Localization(__file__, 786, 17), QDialog_252028, *[], **kwargs_252029)
        
        # Assigning a type to the variable 'dialog' (line 786)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'dialog', QDialog_call_result_252030)
        
        # Assigning a Call to a Name (line 787):
        
        # Assigning a Call to a Name (line 787):
        
        # Call to QVBoxLayout(...): (line 787)
        # Processing the call keyword arguments (line 787)
        kwargs_252033 = {}
        # Getting the type of 'QtWidgets' (line 787)
        QtWidgets_252031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 17), 'QtWidgets', False)
        # Obtaining the member 'QVBoxLayout' of a type (line 787)
        QVBoxLayout_252032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 17), QtWidgets_252031, 'QVBoxLayout')
        # Calling QVBoxLayout(args, kwargs) (line 787)
        QVBoxLayout_call_result_252034 = invoke(stypy.reporting.localization.Localization(__file__, 787, 17), QVBoxLayout_252032, *[], **kwargs_252033)
        
        # Assigning a type to the variable 'layout' (line 787)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'layout', QVBoxLayout_call_result_252034)
        
        # Call to setLayout(...): (line 788)
        # Processing the call arguments (line 788)
        # Getting the type of 'layout' (line 788)
        layout_252037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 25), 'layout', False)
        # Processing the call keyword arguments (line 788)
        kwargs_252038 = {}
        # Getting the type of 'dialog' (line 788)
        dialog_252035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'dialog', False)
        # Obtaining the member 'setLayout' of a type (line 788)
        setLayout_252036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 788, 8), dialog_252035, 'setLayout')
        # Calling setLayout(args, kwargs) (line 788)
        setLayout_call_result_252039 = invoke(stypy.reporting.localization.Localization(__file__, 788, 8), setLayout_252036, *[layout_252037], **kwargs_252038)
        
        
        # Assigning a Call to a Name (line 789):
        
        # Assigning a Call to a Name (line 789):
        
        # Call to QPlainTextEdit(...): (line 789)
        # Processing the call keyword arguments (line 789)
        kwargs_252042 = {}
        # Getting the type of 'QtWidgets' (line 789)
        QtWidgets_252040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 15), 'QtWidgets', False)
        # Obtaining the member 'QPlainTextEdit' of a type (line 789)
        QPlainTextEdit_252041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 15), QtWidgets_252040, 'QPlainTextEdit')
        # Calling QPlainTextEdit(args, kwargs) (line 789)
        QPlainTextEdit_call_result_252043 = invoke(stypy.reporting.localization.Localization(__file__, 789, 15), QPlainTextEdit_252041, *[], **kwargs_252042)
        
        # Assigning a type to the variable 'text' (line 789)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 8), 'text', QPlainTextEdit_call_result_252043)
        
        # Call to setReadOnly(...): (line 790)
        # Processing the call arguments (line 790)
        # Getting the type of 'True' (line 790)
        True_252046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 25), 'True', False)
        # Processing the call keyword arguments (line 790)
        kwargs_252047 = {}
        # Getting the type of 'text' (line 790)
        text_252044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'text', False)
        # Obtaining the member 'setReadOnly' of a type (line 790)
        setReadOnly_252045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 8), text_252044, 'setReadOnly')
        # Calling setReadOnly(args, kwargs) (line 790)
        setReadOnly_call_result_252048 = invoke(stypy.reporting.localization.Localization(__file__, 790, 8), setReadOnly_252045, *[True_252046], **kwargs_252047)
        
        
        # Call to addWidget(...): (line 791)
        # Processing the call arguments (line 791)
        # Getting the type of 'text' (line 791)
        text_252051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 25), 'text', False)
        # Processing the call keyword arguments (line 791)
        kwargs_252052 = {}
        # Getting the type of 'layout' (line 791)
        layout_252049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 8), 'layout', False)
        # Obtaining the member 'addWidget' of a type (line 791)
        addWidget_252050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 8), layout_252049, 'addWidget')
        # Calling addWidget(args, kwargs) (line 791)
        addWidget_call_result_252053 = invoke(stypy.reporting.localization.Localization(__file__, 791, 8), addWidget_252050, *[text_252051], **kwargs_252052)
        
        
        # Call to setPlainText(...): (line 792)
        # Processing the call arguments (line 792)
        
        # Call to join(...): (line 793)
        # Processing the call arguments (line 793)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 793, 23, True)
        # Calculating comprehension expression
        # Getting the type of 'self' (line 794)
        self_252071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 35), 'self', False)
        # Obtaining the member '_attrs' of a type (line 794)
        _attrs_252072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 35), self_252071, '_attrs')
        comprehension_252073 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 793, 23), _attrs_252072)
        # Assigning a type to the variable 'attr' (line 793)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 23), 'attr', comprehension_252073)
        
        # Call to format(...): (line 793)
        # Processing the call arguments (line 793)
        # Getting the type of 'attr' (line 793)
        attr_252060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 41), 'attr', False)
        
        # Call to value(...): (line 793)
        # Processing the call keyword arguments (line 793)
        kwargs_252067 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 793)
        attr_252061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 61), 'attr', False)
        # Getting the type of 'self' (line 793)
        self_252062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 47), 'self', False)
        # Obtaining the member '_widgets' of a type (line 793)
        _widgets_252063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 47), self_252062, '_widgets')
        # Obtaining the member '__getitem__' of a type (line 793)
        getitem___252064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 47), _widgets_252063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 793)
        subscript_call_result_252065 = invoke(stypy.reporting.localization.Localization(__file__, 793, 47), getitem___252064, attr_252061)
        
        # Obtaining the member 'value' of a type (line 793)
        value_252066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 47), subscript_call_result_252065, 'value')
        # Calling value(args, kwargs) (line 793)
        value_call_result_252068 = invoke(stypy.reporting.localization.Localization(__file__, 793, 47), value_252066, *[], **kwargs_252067)
        
        # Processing the call keyword arguments (line 793)
        kwargs_252069 = {}
        unicode_252058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 23), 'unicode', u'{}={:.3}')
        # Obtaining the member 'format' of a type (line 793)
        format_252059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 23), unicode_252058, 'format')
        # Calling format(args, kwargs) (line 793)
        format_call_result_252070 = invoke(stypy.reporting.localization.Localization(__file__, 793, 23), format_252059, *[attr_252060, value_call_result_252068], **kwargs_252069)
        
        list_252074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 23), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 793, 23), list_252074, format_call_result_252070)
        # Processing the call keyword arguments (line 793)
        kwargs_252075 = {}
        unicode_252056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 12), 'unicode', u',\n')
        # Obtaining the member 'join' of a type (line 793)
        join_252057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 12), unicode_252056, 'join')
        # Calling join(args, kwargs) (line 793)
        join_call_result_252076 = invoke(stypy.reporting.localization.Localization(__file__, 793, 12), join_252057, *[list_252074], **kwargs_252075)
        
        # Processing the call keyword arguments (line 792)
        kwargs_252077 = {}
        # Getting the type of 'text' (line 792)
        text_252054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'text', False)
        # Obtaining the member 'setPlainText' of a type (line 792)
        setPlainText_252055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 8), text_252054, 'setPlainText')
        # Calling setPlainText(args, kwargs) (line 792)
        setPlainText_call_result_252078 = invoke(stypy.reporting.localization.Localization(__file__, 792, 8), setPlainText_252055, *[join_call_result_252076], **kwargs_252077)
        
        
        # Assigning a Call to a Name (line 797):
        
        # Assigning a Call to a Name (line 797):
        
        # Call to maximumSize(...): (line 797)
        # Processing the call keyword arguments (line 797)
        kwargs_252081 = {}
        # Getting the type of 'text' (line 797)
        text_252079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 15), 'text', False)
        # Obtaining the member 'maximumSize' of a type (line 797)
        maximumSize_252080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 15), text_252079, 'maximumSize')
        # Calling maximumSize(args, kwargs) (line 797)
        maximumSize_call_result_252082 = invoke(stypy.reporting.localization.Localization(__file__, 797, 15), maximumSize_252080, *[], **kwargs_252081)
        
        # Assigning a type to the variable 'size' (line 797)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'size', maximumSize_call_result_252082)
        
        # Call to setHeight(...): (line 798)
        # Processing the call arguments (line 798)
        
        # Call to height(...): (line 799)
        # Processing the call keyword arguments (line 799)
        kwargs_252105 = {}
        
        # Call to size(...): (line 799)
        # Processing the call arguments (line 799)
        int_252097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 18), 'int')
        
        # Call to toPlainText(...): (line 800)
        # Processing the call keyword arguments (line 800)
        kwargs_252100 = {}
        # Getting the type of 'text' (line 800)
        text_252098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 21), 'text', False)
        # Obtaining the member 'toPlainText' of a type (line 800)
        toPlainText_252099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 21), text_252098, 'toPlainText')
        # Calling toPlainText(args, kwargs) (line 800)
        toPlainText_call_result_252101 = invoke(stypy.reporting.localization.Localization(__file__, 800, 21), toPlainText_252099, *[], **kwargs_252100)
        
        # Processing the call keyword arguments (line 799)
        kwargs_252102 = {}
        
        # Call to QFontMetrics(...): (line 799)
        # Processing the call arguments (line 799)
        
        # Call to defaultFont(...): (line 799)
        # Processing the call keyword arguments (line 799)
        kwargs_252092 = {}
        
        # Call to document(...): (line 799)
        # Processing the call keyword arguments (line 799)
        kwargs_252089 = {}
        # Getting the type of 'text' (line 799)
        text_252087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 31), 'text', False)
        # Obtaining the member 'document' of a type (line 799)
        document_252088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 31), text_252087, 'document')
        # Calling document(args, kwargs) (line 799)
        document_call_result_252090 = invoke(stypy.reporting.localization.Localization(__file__, 799, 31), document_252088, *[], **kwargs_252089)
        
        # Obtaining the member 'defaultFont' of a type (line 799)
        defaultFont_252091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 31), document_call_result_252090, 'defaultFont')
        # Calling defaultFont(args, kwargs) (line 799)
        defaultFont_call_result_252093 = invoke(stypy.reporting.localization.Localization(__file__, 799, 31), defaultFont_252091, *[], **kwargs_252092)
        
        # Processing the call keyword arguments (line 799)
        kwargs_252094 = {}
        # Getting the type of 'QtGui' (line 799)
        QtGui_252085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 12), 'QtGui', False)
        # Obtaining the member 'QFontMetrics' of a type (line 799)
        QFontMetrics_252086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 12), QtGui_252085, 'QFontMetrics')
        # Calling QFontMetrics(args, kwargs) (line 799)
        QFontMetrics_call_result_252095 = invoke(stypy.reporting.localization.Localization(__file__, 799, 12), QFontMetrics_252086, *[defaultFont_call_result_252093], **kwargs_252094)
        
        # Obtaining the member 'size' of a type (line 799)
        size_252096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 12), QFontMetrics_call_result_252095, 'size')
        # Calling size(args, kwargs) (line 799)
        size_call_result_252103 = invoke(stypy.reporting.localization.Localization(__file__, 799, 12), size_252096, *[int_252097, toPlainText_call_result_252101], **kwargs_252102)
        
        # Obtaining the member 'height' of a type (line 799)
        height_252104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 12), size_call_result_252103, 'height')
        # Calling height(args, kwargs) (line 799)
        height_call_result_252106 = invoke(stypy.reporting.localization.Localization(__file__, 799, 12), height_252104, *[], **kwargs_252105)
        
        int_252107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 52), 'int')
        # Applying the binary operator '+' (line 799)
        result_add_252108 = python_operator(stypy.reporting.localization.Localization(__file__, 799, 12), '+', height_call_result_252106, int_252107)
        
        # Processing the call keyword arguments (line 798)
        kwargs_252109 = {}
        # Getting the type of 'size' (line 798)
        size_252083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 8), 'size', False)
        # Obtaining the member 'setHeight' of a type (line 798)
        setHeight_252084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 8), size_252083, 'setHeight')
        # Calling setHeight(args, kwargs) (line 798)
        setHeight_call_result_252110 = invoke(stypy.reporting.localization.Localization(__file__, 798, 8), setHeight_252084, *[result_add_252108], **kwargs_252109)
        
        
        # Call to setMaximumSize(...): (line 801)
        # Processing the call arguments (line 801)
        # Getting the type of 'size' (line 801)
        size_252113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 28), 'size', False)
        # Processing the call keyword arguments (line 801)
        kwargs_252114 = {}
        # Getting the type of 'text' (line 801)
        text_252111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 8), 'text', False)
        # Obtaining the member 'setMaximumSize' of a type (line 801)
        setMaximumSize_252112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 8), text_252111, 'setMaximumSize')
        # Calling setMaximumSize(args, kwargs) (line 801)
        setMaximumSize_call_result_252115 = invoke(stypy.reporting.localization.Localization(__file__, 801, 8), setMaximumSize_252112, *[size_252113], **kwargs_252114)
        
        
        # Call to exec_(...): (line 802)
        # Processing the call keyword arguments (line 802)
        kwargs_252118 = {}
        # Getting the type of 'dialog' (line 802)
        dialog_252116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 8), 'dialog', False)
        # Obtaining the member 'exec_' of a type (line 802)
        exec__252117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 8), dialog_252116, 'exec_')
        # Calling exec_(args, kwargs) (line 802)
        exec__call_result_252119 = invoke(stypy.reporting.localization.Localization(__file__, 802, 8), exec__252117, *[], **kwargs_252118)
        
        
        # ################# End of '_export_values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_export_values' in the type store
        # Getting the type of 'stypy_return_type' (line 783)
        stypy_return_type_252120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252120)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_export_values'
        return stypy_return_type_252120


    @norecursion
    def _on_value_changed(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_on_value_changed'
        module_type_store = module_type_store.open_function_context('_on_value_changed', 804, 4, False)
        # Assigning a type to the variable 'self' (line 805)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotToolQt._on_value_changed.__dict__.__setitem__('stypy_localization', localization)
        SubplotToolQt._on_value_changed.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotToolQt._on_value_changed.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotToolQt._on_value_changed.__dict__.__setitem__('stypy_function_name', 'SubplotToolQt._on_value_changed')
        SubplotToolQt._on_value_changed.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotToolQt._on_value_changed.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotToolQt._on_value_changed.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotToolQt._on_value_changed.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotToolQt._on_value_changed.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotToolQt._on_value_changed.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotToolQt._on_value_changed.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotToolQt._on_value_changed', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_on_value_changed', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_on_value_changed(...)' code ##################

        
        # Call to subplots_adjust(...): (line 805)
        # Processing the call keyword arguments (line 805)
        # Calculating dict comprehension
        module_type_store = module_type_store.open_function_context('dict comprehension expression', 805, 40, True)
        # Calculating comprehension expression
        # Getting the type of 'self' (line 806)
        self_252133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 52), 'self', False)
        # Obtaining the member '_attrs' of a type (line 806)
        _attrs_252134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 52), self_252133, '_attrs')
        comprehension_252135 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 805, 40), _attrs_252134)
        # Assigning a type to the variable 'attr' (line 805)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 40), 'attr', comprehension_252135)
        # Getting the type of 'attr' (line 805)
        attr_252124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 40), 'attr', False)
        
        # Call to value(...): (line 805)
        # Processing the call keyword arguments (line 805)
        kwargs_252131 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 805)
        attr_252125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 60), 'attr', False)
        # Getting the type of 'self' (line 805)
        self_252126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 46), 'self', False)
        # Obtaining the member '_widgets' of a type (line 805)
        _widgets_252127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 46), self_252126, '_widgets')
        # Obtaining the member '__getitem__' of a type (line 805)
        getitem___252128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 46), _widgets_252127, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 805)
        subscript_call_result_252129 = invoke(stypy.reporting.localization.Localization(__file__, 805, 46), getitem___252128, attr_252125)
        
        # Obtaining the member 'value' of a type (line 805)
        value_252130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 46), subscript_call_result_252129, 'value')
        # Calling value(args, kwargs) (line 805)
        value_call_result_252132 = invoke(stypy.reporting.localization.Localization(__file__, 805, 46), value_252130, *[], **kwargs_252131)
        
        dict_252136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 40), 'dict')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 805, 40), dict_252136, (attr_252124, value_call_result_252132))
        kwargs_252137 = {'dict_252136': dict_252136}
        # Getting the type of 'self' (line 805)
        self_252121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 8), 'self', False)
        # Obtaining the member '_figure' of a type (line 805)
        _figure_252122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 8), self_252121, '_figure')
        # Obtaining the member 'subplots_adjust' of a type (line 805)
        subplots_adjust_252123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 8), _figure_252122, 'subplots_adjust')
        # Calling subplots_adjust(args, kwargs) (line 805)
        subplots_adjust_call_result_252138 = invoke(stypy.reporting.localization.Localization(__file__, 805, 8), subplots_adjust_252123, *[], **kwargs_252137)
        
        
        # Call to draw_idle(...): (line 807)
        # Processing the call keyword arguments (line 807)
        kwargs_252143 = {}
        # Getting the type of 'self' (line 807)
        self_252139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'self', False)
        # Obtaining the member '_figure' of a type (line 807)
        _figure_252140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 8), self_252139, '_figure')
        # Obtaining the member 'canvas' of a type (line 807)
        canvas_252141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 8), _figure_252140, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 807)
        draw_idle_252142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 8), canvas_252141, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 807)
        draw_idle_call_result_252144 = invoke(stypy.reporting.localization.Localization(__file__, 807, 8), draw_idle_252142, *[], **kwargs_252143)
        
        
        # ################# End of '_on_value_changed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_on_value_changed' in the type store
        # Getting the type of 'stypy_return_type' (line 804)
        stypy_return_type_252145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252145)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_on_value_changed'
        return stypy_return_type_252145


    @norecursion
    def _tight_layout(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_tight_layout'
        module_type_store = module_type_store.open_function_context('_tight_layout', 809, 4, False)
        # Assigning a type to the variable 'self' (line 810)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotToolQt._tight_layout.__dict__.__setitem__('stypy_localization', localization)
        SubplotToolQt._tight_layout.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotToolQt._tight_layout.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotToolQt._tight_layout.__dict__.__setitem__('stypy_function_name', 'SubplotToolQt._tight_layout')
        SubplotToolQt._tight_layout.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotToolQt._tight_layout.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotToolQt._tight_layout.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotToolQt._tight_layout.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotToolQt._tight_layout.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotToolQt._tight_layout.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotToolQt._tight_layout.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotToolQt._tight_layout', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_tight_layout', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_tight_layout(...)' code ##################

        
        # Call to tight_layout(...): (line 810)
        # Processing the call keyword arguments (line 810)
        kwargs_252149 = {}
        # Getting the type of 'self' (line 810)
        self_252146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'self', False)
        # Obtaining the member '_figure' of a type (line 810)
        _figure_252147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 8), self_252146, '_figure')
        # Obtaining the member 'tight_layout' of a type (line 810)
        tight_layout_252148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 8), _figure_252147, 'tight_layout')
        # Calling tight_layout(args, kwargs) (line 810)
        tight_layout_call_result_252150 = invoke(stypy.reporting.localization.Localization(__file__, 810, 8), tight_layout_252148, *[], **kwargs_252149)
        
        
        # Getting the type of 'self' (line 811)
        self_252151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 20), 'self')
        # Obtaining the member '_attrs' of a type (line 811)
        _attrs_252152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 20), self_252151, '_attrs')
        # Testing the type of a for loop iterable (line 811)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 811, 8), _attrs_252152)
        # Getting the type of the for loop variable (line 811)
        for_loop_var_252153 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 811, 8), _attrs_252152)
        # Assigning a type to the variable 'attr' (line 811)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 8), 'attr', for_loop_var_252153)
        # SSA begins for a for statement (line 811)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 812):
        
        # Assigning a Subscript to a Name (line 812):
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 812)
        attr_252154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 35), 'attr')
        # Getting the type of 'self' (line 812)
        self_252155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 21), 'self')
        # Obtaining the member '_widgets' of a type (line 812)
        _widgets_252156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 21), self_252155, '_widgets')
        # Obtaining the member '__getitem__' of a type (line 812)
        getitem___252157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 21), _widgets_252156, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 812)
        subscript_call_result_252158 = invoke(stypy.reporting.localization.Localization(__file__, 812, 21), getitem___252157, attr_252154)
        
        # Assigning a type to the variable 'widget' (line 812)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 12), 'widget', subscript_call_result_252158)
        
        # Call to blockSignals(...): (line 813)
        # Processing the call arguments (line 813)
        # Getting the type of 'True' (line 813)
        True_252161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 32), 'True', False)
        # Processing the call keyword arguments (line 813)
        kwargs_252162 = {}
        # Getting the type of 'widget' (line 813)
        widget_252159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 12), 'widget', False)
        # Obtaining the member 'blockSignals' of a type (line 813)
        blockSignals_252160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 12), widget_252159, 'blockSignals')
        # Calling blockSignals(args, kwargs) (line 813)
        blockSignals_call_result_252163 = invoke(stypy.reporting.localization.Localization(__file__, 813, 12), blockSignals_252160, *[True_252161], **kwargs_252162)
        
        
        # Call to setValue(...): (line 814)
        # Processing the call arguments (line 814)
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 814)
        attr_252166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 59), 'attr', False)
        
        # Call to vars(...): (line 814)
        # Processing the call arguments (line 814)
        # Getting the type of 'self' (line 814)
        self_252168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 33), 'self', False)
        # Obtaining the member '_figure' of a type (line 814)
        _figure_252169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 33), self_252168, '_figure')
        # Obtaining the member 'subplotpars' of a type (line 814)
        subplotpars_252170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 33), _figure_252169, 'subplotpars')
        # Processing the call keyword arguments (line 814)
        kwargs_252171 = {}
        # Getting the type of 'vars' (line 814)
        vars_252167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 28), 'vars', False)
        # Calling vars(args, kwargs) (line 814)
        vars_call_result_252172 = invoke(stypy.reporting.localization.Localization(__file__, 814, 28), vars_252167, *[subplotpars_252170], **kwargs_252171)
        
        # Obtaining the member '__getitem__' of a type (line 814)
        getitem___252173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 28), vars_call_result_252172, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 814)
        subscript_call_result_252174 = invoke(stypy.reporting.localization.Localization(__file__, 814, 28), getitem___252173, attr_252166)
        
        # Processing the call keyword arguments (line 814)
        kwargs_252175 = {}
        # Getting the type of 'widget' (line 814)
        widget_252164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 12), 'widget', False)
        # Obtaining the member 'setValue' of a type (line 814)
        setValue_252165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 12), widget_252164, 'setValue')
        # Calling setValue(args, kwargs) (line 814)
        setValue_call_result_252176 = invoke(stypy.reporting.localization.Localization(__file__, 814, 12), setValue_252165, *[subscript_call_result_252174], **kwargs_252175)
        
        
        # Call to blockSignals(...): (line 815)
        # Processing the call arguments (line 815)
        # Getting the type of 'False' (line 815)
        False_252179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 32), 'False', False)
        # Processing the call keyword arguments (line 815)
        kwargs_252180 = {}
        # Getting the type of 'widget' (line 815)
        widget_252177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 12), 'widget', False)
        # Obtaining the member 'blockSignals' of a type (line 815)
        blockSignals_252178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 12), widget_252177, 'blockSignals')
        # Calling blockSignals(args, kwargs) (line 815)
        blockSignals_call_result_252181 = invoke(stypy.reporting.localization.Localization(__file__, 815, 12), blockSignals_252178, *[False_252179], **kwargs_252180)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to draw_idle(...): (line 816)
        # Processing the call keyword arguments (line 816)
        kwargs_252186 = {}
        # Getting the type of 'self' (line 816)
        self_252182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 8), 'self', False)
        # Obtaining the member '_figure' of a type (line 816)
        _figure_252183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 8), self_252182, '_figure')
        # Obtaining the member 'canvas' of a type (line 816)
        canvas_252184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 8), _figure_252183, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 816)
        draw_idle_252185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 8), canvas_252184, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 816)
        draw_idle_call_result_252187 = invoke(stypy.reporting.localization.Localization(__file__, 816, 8), draw_idle_252185, *[], **kwargs_252186)
        
        
        # ################# End of '_tight_layout(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_tight_layout' in the type store
        # Getting the type of 'stypy_return_type' (line 809)
        stypy_return_type_252188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252188)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_tight_layout'
        return stypy_return_type_252188


    @norecursion
    def _reset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_reset'
        module_type_store = module_type_store.open_function_context('_reset', 818, 4, False)
        # Assigning a type to the variable 'self' (line 819)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SubplotToolQt._reset.__dict__.__setitem__('stypy_localization', localization)
        SubplotToolQt._reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SubplotToolQt._reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        SubplotToolQt._reset.__dict__.__setitem__('stypy_function_name', 'SubplotToolQt._reset')
        SubplotToolQt._reset.__dict__.__setitem__('stypy_param_names_list', [])
        SubplotToolQt._reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        SubplotToolQt._reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SubplotToolQt._reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        SubplotToolQt._reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        SubplotToolQt._reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SubplotToolQt._reset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SubplotToolQt._reset', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_reset', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_reset(...)' code ##################

        
        
        # Call to items(...): (line 819)
        # Processing the call keyword arguments (line 819)
        kwargs_252192 = {}
        # Getting the type of 'self' (line 819)
        self_252189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 27), 'self', False)
        # Obtaining the member '_defaults' of a type (line 819)
        _defaults_252190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 27), self_252189, '_defaults')
        # Obtaining the member 'items' of a type (line 819)
        items_252191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 27), _defaults_252190, 'items')
        # Calling items(args, kwargs) (line 819)
        items_call_result_252193 = invoke(stypy.reporting.localization.Localization(__file__, 819, 27), items_252191, *[], **kwargs_252192)
        
        # Testing the type of a for loop iterable (line 819)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 819, 8), items_call_result_252193)
        # Getting the type of the for loop variable (line 819)
        for_loop_var_252194 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 819, 8), items_call_result_252193)
        # Assigning a type to the variable 'attr' (line 819)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 8), 'attr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 819, 8), for_loop_var_252194))
        # Assigning a type to the variable 'value' (line 819)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 819, 8), for_loop_var_252194))
        # SSA begins for a for statement (line 819)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setValue(...): (line 820)
        # Processing the call arguments (line 820)
        # Getting the type of 'value' (line 820)
        value_252201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 41), 'value', False)
        # Processing the call keyword arguments (line 820)
        kwargs_252202 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'attr' (line 820)
        attr_252195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 26), 'attr', False)
        # Getting the type of 'self' (line 820)
        self_252196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 12), 'self', False)
        # Obtaining the member '_widgets' of a type (line 820)
        _widgets_252197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 12), self_252196, '_widgets')
        # Obtaining the member '__getitem__' of a type (line 820)
        getitem___252198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 12), _widgets_252197, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 820)
        subscript_call_result_252199 = invoke(stypy.reporting.localization.Localization(__file__, 820, 12), getitem___252198, attr_252195)
        
        # Obtaining the member 'setValue' of a type (line 820)
        setValue_252200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 12), subscript_call_result_252199, 'setValue')
        # Calling setValue(args, kwargs) (line 820)
        setValue_call_result_252203 = invoke(stypy.reporting.localization.Localization(__file__, 820, 12), setValue_252200, *[value_252201], **kwargs_252202)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_reset' in the type store
        # Getting the type of 'stypy_return_type' (line 818)
        stypy_return_type_252204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252204)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_reset'
        return stypy_return_type_252204


# Assigning a type to the variable 'SubplotToolQt' (line 755)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 0), 'SubplotToolQt', SubplotToolQt)

@norecursion
def error_msg_qt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 823)
    None_252205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 29), 'None')
    defaults = [None_252205]
    # Create a new context for function 'error_msg_qt'
    module_type_store = module_type_store.open_function_context('error_msg_qt', 823, 0, False)
    
    # Passed parameters checking function
    error_msg_qt.stypy_localization = localization
    error_msg_qt.stypy_type_of_self = None
    error_msg_qt.stypy_type_store = module_type_store
    error_msg_qt.stypy_function_name = 'error_msg_qt'
    error_msg_qt.stypy_param_names_list = ['msg', 'parent']
    error_msg_qt.stypy_varargs_param_name = None
    error_msg_qt.stypy_kwargs_param_name = None
    error_msg_qt.stypy_call_defaults = defaults
    error_msg_qt.stypy_call_varargs = varargs
    error_msg_qt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'error_msg_qt', ['msg', 'parent'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'error_msg_qt', localization, ['msg', 'parent'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'error_msg_qt(...)' code ##################

    
    
    
    # Call to isinstance(...): (line 824)
    # Processing the call arguments (line 824)
    # Getting the type of 'msg' (line 824)
    msg_252207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 22), 'msg', False)
    # Getting the type of 'six' (line 824)
    six_252208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 27), 'six', False)
    # Obtaining the member 'string_types' of a type (line 824)
    string_types_252209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 27), six_252208, 'string_types')
    # Processing the call keyword arguments (line 824)
    kwargs_252210 = {}
    # Getting the type of 'isinstance' (line 824)
    isinstance_252206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 824)
    isinstance_call_result_252211 = invoke(stypy.reporting.localization.Localization(__file__, 824, 11), isinstance_252206, *[msg_252207, string_types_252209], **kwargs_252210)
    
    # Applying the 'not' unary operator (line 824)
    result_not__252212 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 7), 'not', isinstance_call_result_252211)
    
    # Testing the type of an if condition (line 824)
    if_condition_252213 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 824, 4), result_not__252212)
    # Assigning a type to the variable 'if_condition_252213' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'if_condition_252213', if_condition_252213)
    # SSA begins for if statement (line 824)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 825):
    
    # Assigning a Call to a Name (line 825):
    
    # Call to join(...): (line 825)
    # Processing the call arguments (line 825)
    
    # Call to map(...): (line 825)
    # Processing the call arguments (line 825)
    # Getting the type of 'str' (line 825)
    str_252217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 27), 'str', False)
    # Getting the type of 'msg' (line 825)
    msg_252218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 32), 'msg', False)
    # Processing the call keyword arguments (line 825)
    kwargs_252219 = {}
    # Getting the type of 'map' (line 825)
    map_252216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 23), 'map', False)
    # Calling map(args, kwargs) (line 825)
    map_call_result_252220 = invoke(stypy.reporting.localization.Localization(__file__, 825, 23), map_252216, *[str_252217, msg_252218], **kwargs_252219)
    
    # Processing the call keyword arguments (line 825)
    kwargs_252221 = {}
    unicode_252214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 14), 'unicode', u',')
    # Obtaining the member 'join' of a type (line 825)
    join_252215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 14), unicode_252214, 'join')
    # Calling join(args, kwargs) (line 825)
    join_call_result_252222 = invoke(stypy.reporting.localization.Localization(__file__, 825, 14), join_252215, *[map_call_result_252220], **kwargs_252221)
    
    # Assigning a type to the variable 'msg' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 8), 'msg', join_call_result_252222)
    # SSA join for if statement (line 824)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to warning(...): (line 827)
    # Processing the call arguments (line 827)
    # Getting the type of 'None' (line 827)
    None_252226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 34), 'None', False)
    unicode_252227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 40), 'unicode', u'Matplotlib')
    # Getting the type of 'msg' (line 828)
    msg_252228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 34), 'msg', False)
    # Getting the type of 'QtGui' (line 828)
    QtGui_252229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 39), 'QtGui', False)
    # Obtaining the member 'QMessageBox' of a type (line 828)
    QMessageBox_252230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 39), QtGui_252229, 'QMessageBox')
    # Obtaining the member 'Ok' of a type (line 828)
    Ok_252231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 39), QMessageBox_252230, 'Ok')
    # Processing the call keyword arguments (line 827)
    kwargs_252232 = {}
    # Getting the type of 'QtWidgets' (line 827)
    QtWidgets_252223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 4), 'QtWidgets', False)
    # Obtaining the member 'QMessageBox' of a type (line 827)
    QMessageBox_252224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 4), QtWidgets_252223, 'QMessageBox')
    # Obtaining the member 'warning' of a type (line 827)
    warning_252225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 4), QMessageBox_252224, 'warning')
    # Calling warning(args, kwargs) (line 827)
    warning_call_result_252233 = invoke(stypy.reporting.localization.Localization(__file__, 827, 4), warning_252225, *[None_252226, unicode_252227, msg_252228, Ok_252231], **kwargs_252232)
    
    
    # ################# End of 'error_msg_qt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'error_msg_qt' in the type store
    # Getting the type of 'stypy_return_type' (line 823)
    stypy_return_type_252234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_252234)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'error_msg_qt'
    return stypy_return_type_252234

# Assigning a type to the variable 'error_msg_qt' (line 823)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 823, 0), 'error_msg_qt', error_msg_qt)

@norecursion
def exception_handler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'exception_handler'
    module_type_store = module_type_store.open_function_context('exception_handler', 831, 0, False)
    
    # Passed parameters checking function
    exception_handler.stypy_localization = localization
    exception_handler.stypy_type_of_self = None
    exception_handler.stypy_type_store = module_type_store
    exception_handler.stypy_function_name = 'exception_handler'
    exception_handler.stypy_param_names_list = ['type', 'value', 'tb']
    exception_handler.stypy_varargs_param_name = None
    exception_handler.stypy_kwargs_param_name = None
    exception_handler.stypy_call_defaults = defaults
    exception_handler.stypy_call_varargs = varargs
    exception_handler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exception_handler', ['type', 'value', 'tb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exception_handler', localization, ['type', 'value', 'tb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exception_handler(...)' code ##################

    unicode_252235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, (-1)), 'unicode', u'Handle uncaught exceptions\n    It does not catch SystemExit\n    ')
    
    # Assigning a Str to a Name (line 835):
    
    # Assigning a Str to a Name (line 835):
    unicode_252236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 10), 'unicode', u'')
    # Assigning a type to the variable 'msg' (line 835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 4), 'msg', unicode_252236)
    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 837)
    # Processing the call arguments (line 837)
    # Getting the type of 'value' (line 837)
    value_252238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 15), 'value', False)
    unicode_252239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 22), 'unicode', u'filename')
    # Processing the call keyword arguments (line 837)
    kwargs_252240 = {}
    # Getting the type of 'hasattr' (line 837)
    hasattr_252237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 7), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 837)
    hasattr_call_result_252241 = invoke(stypy.reporting.localization.Localization(__file__, 837, 7), hasattr_252237, *[value_252238, unicode_252239], **kwargs_252240)
    
    
    # Getting the type of 'value' (line 837)
    value_252242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 38), 'value')
    # Obtaining the member 'filename' of a type (line 837)
    filename_252243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 38), value_252242, 'filename')
    # Getting the type of 'None' (line 837)
    None_252244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 60), 'None')
    # Applying the binary operator 'isnot' (line 837)
    result_is_not_252245 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 38), 'isnot', filename_252243, None_252244)
    
    # Applying the binary operator 'and' (line 837)
    result_and_keyword_252246 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 7), 'and', hasattr_call_result_252241, result_is_not_252245)
    
    # Testing the type of an if condition (line 837)
    if_condition_252247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 837, 4), result_and_keyword_252246)
    # Assigning a type to the variable 'if_condition_252247' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'if_condition_252247', if_condition_252247)
    # SSA begins for if statement (line 837)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 838):
    
    # Assigning a BinOp to a Name (line 838):
    # Getting the type of 'value' (line 838)
    value_252248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 14), 'value')
    # Obtaining the member 'filename' of a type (line 838)
    filename_252249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 14), value_252248, 'filename')
    unicode_252250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 31), 'unicode', u': ')
    # Applying the binary operator '+' (line 838)
    result_add_252251 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 14), '+', filename_252249, unicode_252250)
    
    # Assigning a type to the variable 'msg' (line 838)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'msg', result_add_252251)
    # SSA join for if statement (line 837)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 839)
    # Processing the call arguments (line 839)
    # Getting the type of 'value' (line 839)
    value_252253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 15), 'value', False)
    unicode_252254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 22), 'unicode', u'strerror')
    # Processing the call keyword arguments (line 839)
    kwargs_252255 = {}
    # Getting the type of 'hasattr' (line 839)
    hasattr_252252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 7), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 839)
    hasattr_call_result_252256 = invoke(stypy.reporting.localization.Localization(__file__, 839, 7), hasattr_252252, *[value_252253, unicode_252254], **kwargs_252255)
    
    
    # Getting the type of 'value' (line 839)
    value_252257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 38), 'value')
    # Obtaining the member 'strerror' of a type (line 839)
    strerror_252258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 38), value_252257, 'strerror')
    # Getting the type of 'None' (line 839)
    None_252259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 60), 'None')
    # Applying the binary operator 'isnot' (line 839)
    result_is_not_252260 = python_operator(stypy.reporting.localization.Localization(__file__, 839, 38), 'isnot', strerror_252258, None_252259)
    
    # Applying the binary operator 'and' (line 839)
    result_and_keyword_252261 = python_operator(stypy.reporting.localization.Localization(__file__, 839, 7), 'and', hasattr_call_result_252256, result_is_not_252260)
    
    # Testing the type of an if condition (line 839)
    if_condition_252262 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 839, 4), result_and_keyword_252261)
    # Assigning a type to the variable 'if_condition_252262' (line 839)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 4), 'if_condition_252262', if_condition_252262)
    # SSA begins for if statement (line 839)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'msg' (line 840)
    msg_252263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'msg')
    # Getting the type of 'value' (line 840)
    value_252264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 15), 'value')
    # Obtaining the member 'strerror' of a type (line 840)
    strerror_252265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 15), value_252264, 'strerror')
    # Applying the binary operator '+=' (line 840)
    result_iadd_252266 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 8), '+=', msg_252263, strerror_252265)
    # Assigning a type to the variable 'msg' (line 840)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'msg', result_iadd_252266)
    
    # SSA branch for the else part of an if statement (line 839)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'msg' (line 842)
    msg_252267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 8), 'msg')
    
    # Call to text_type(...): (line 842)
    # Processing the call arguments (line 842)
    # Getting the type of 'value' (line 842)
    value_252270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 29), 'value', False)
    # Processing the call keyword arguments (line 842)
    kwargs_252271 = {}
    # Getting the type of 'six' (line 842)
    six_252268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 15), 'six', False)
    # Obtaining the member 'text_type' of a type (line 842)
    text_type_252269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 15), six_252268, 'text_type')
    # Calling text_type(args, kwargs) (line 842)
    text_type_call_result_252272 = invoke(stypy.reporting.localization.Localization(__file__, 842, 15), text_type_252269, *[value_252270], **kwargs_252271)
    
    # Applying the binary operator '+=' (line 842)
    result_iadd_252273 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 8), '+=', msg_252267, text_type_call_result_252272)
    # Assigning a type to the variable 'msg' (line 842)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 8), 'msg', result_iadd_252273)
    
    # SSA join for if statement (line 839)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to len(...): (line 844)
    # Processing the call arguments (line 844)
    # Getting the type of 'msg' (line 844)
    msg_252275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 11), 'msg', False)
    # Processing the call keyword arguments (line 844)
    kwargs_252276 = {}
    # Getting the type of 'len' (line 844)
    len_252274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 7), 'len', False)
    # Calling len(args, kwargs) (line 844)
    len_call_result_252277 = invoke(stypy.reporting.localization.Localization(__file__, 844, 7), len_252274, *[msg_252275], **kwargs_252276)
    
    # Testing the type of an if condition (line 844)
    if_condition_252278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 844, 4), len_call_result_252277)
    # Assigning a type to the variable 'if_condition_252278' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 4), 'if_condition_252278', if_condition_252278)
    # SSA begins for if statement (line 844)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to error_msg_qt(...): (line 845)
    # Processing the call arguments (line 845)
    # Getting the type of 'msg' (line 845)
    msg_252280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 21), 'msg', False)
    # Processing the call keyword arguments (line 845)
    kwargs_252281 = {}
    # Getting the type of 'error_msg_qt' (line 845)
    error_msg_qt_252279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 8), 'error_msg_qt', False)
    # Calling error_msg_qt(args, kwargs) (line 845)
    error_msg_qt_call_result_252282 = invoke(stypy.reporting.localization.Localization(__file__, 845, 8), error_msg_qt_252279, *[msg_252280], **kwargs_252281)
    
    # SSA join for if statement (line 844)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'exception_handler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exception_handler' in the type store
    # Getting the type of 'stypy_return_type' (line 831)
    stypy_return_type_252283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_252283)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exception_handler'
    return stypy_return_type_252283

# Assigning a type to the variable 'exception_handler' (line 831)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 0), 'exception_handler', exception_handler)
# Declaration of the '_BackendQT5' class
# Getting the type of '_Backend' (line 849)
_Backend_252284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 18), '_Backend')

class _BackendQT5(_Backend_252284, ):
    
    # Assigning a Name to a Name (line 850):
    
    # Assigning a Name to a Name (line 851):

    @staticmethod
    @norecursion
    def trigger_manager_draw(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'trigger_manager_draw'
        module_type_store = module_type_store.open_function_context('trigger_manager_draw', 853, 4, False)
        
        # Passed parameters checking function
        _BackendQT5.trigger_manager_draw.__dict__.__setitem__('stypy_localization', localization)
        _BackendQT5.trigger_manager_draw.__dict__.__setitem__('stypy_type_of_self', None)
        _BackendQT5.trigger_manager_draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        _BackendQT5.trigger_manager_draw.__dict__.__setitem__('stypy_function_name', 'trigger_manager_draw')
        _BackendQT5.trigger_manager_draw.__dict__.__setitem__('stypy_param_names_list', ['manager'])
        _BackendQT5.trigger_manager_draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        _BackendQT5.trigger_manager_draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _BackendQT5.trigger_manager_draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        _BackendQT5.trigger_manager_draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        _BackendQT5.trigger_manager_draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _BackendQT5.trigger_manager_draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
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

        
        # Call to draw_idle(...): (line 855)
        # Processing the call keyword arguments (line 855)
        kwargs_252288 = {}
        # Getting the type of 'manager' (line 855)
        manager_252285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 8), 'manager', False)
        # Obtaining the member 'canvas' of a type (line 855)
        canvas_252286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 8), manager_252285, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 855)
        draw_idle_252287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 855, 8), canvas_252286, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 855)
        draw_idle_call_result_252289 = invoke(stypy.reporting.localization.Localization(__file__, 855, 8), draw_idle_252287, *[], **kwargs_252288)
        
        
        # ################# End of 'trigger_manager_draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger_manager_draw' in the type store
        # Getting the type of 'stypy_return_type' (line 853)
        stypy_return_type_252290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252290)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger_manager_draw'
        return stypy_return_type_252290


    @staticmethod
    @norecursion
    def mainloop(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mainloop'
        module_type_store = module_type_store.open_function_context('mainloop', 857, 4, False)
        
        # Passed parameters checking function
        _BackendQT5.mainloop.__dict__.__setitem__('stypy_localization', localization)
        _BackendQT5.mainloop.__dict__.__setitem__('stypy_type_of_self', None)
        _BackendQT5.mainloop.__dict__.__setitem__('stypy_type_store', module_type_store)
        _BackendQT5.mainloop.__dict__.__setitem__('stypy_function_name', 'mainloop')
        _BackendQT5.mainloop.__dict__.__setitem__('stypy_param_names_list', [])
        _BackendQT5.mainloop.__dict__.__setitem__('stypy_varargs_param_name', None)
        _BackendQT5.mainloop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _BackendQT5.mainloop.__dict__.__setitem__('stypy_call_defaults', defaults)
        _BackendQT5.mainloop.__dict__.__setitem__('stypy_call_varargs', varargs)
        _BackendQT5.mainloop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _BackendQT5.mainloop.__dict__.__setitem__('stypy_declared_arg_number', 0)
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

        
        # Call to signal(...): (line 860)
        # Processing the call arguments (line 860)
        # Getting the type of 'signal' (line 860)
        signal_252293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 22), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 860)
        SIGINT_252294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 22), signal_252293, 'SIGINT')
        # Getting the type of 'signal' (line 860)
        signal_252295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 37), 'signal', False)
        # Obtaining the member 'SIG_DFL' of a type (line 860)
        SIG_DFL_252296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 37), signal_252295, 'SIG_DFL')
        # Processing the call keyword arguments (line 860)
        kwargs_252297 = {}
        # Getting the type of 'signal' (line 860)
        signal_252291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 8), 'signal', False)
        # Obtaining the member 'signal' of a type (line 860)
        signal_252292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 860, 8), signal_252291, 'signal')
        # Calling signal(args, kwargs) (line 860)
        signal_call_result_252298 = invoke(stypy.reporting.localization.Localization(__file__, 860, 8), signal_252292, *[SIGINT_252294, SIG_DFL_252296], **kwargs_252297)
        
        # Marking variables as global (line 861)
        module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 861, 8), 'qApp')
        
        # Call to exec_(...): (line 862)
        # Processing the call keyword arguments (line 862)
        kwargs_252301 = {}
        # Getting the type of 'qApp' (line 862)
        qApp_252299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'qApp', False)
        # Obtaining the member 'exec_' of a type (line 862)
        exec__252300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 8), qApp_252299, 'exec_')
        # Calling exec_(args, kwargs) (line 862)
        exec__call_result_252302 = invoke(stypy.reporting.localization.Localization(__file__, 862, 8), exec__252300, *[], **kwargs_252301)
        
        
        # ################# End of 'mainloop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mainloop' in the type store
        # Getting the type of 'stypy_return_type' (line 857)
        stypy_return_type_252303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_252303)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mainloop'
        return stypy_return_type_252303


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 848, 0, False)
        # Assigning a type to the variable 'self' (line 849)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendQT5.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendQT5' (line 848)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 0), '_BackendQT5', _BackendQT5)

# Assigning a Name to a Name (line 850):
# Getting the type of 'FigureCanvasQT' (line 850)
FigureCanvasQT_252304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 19), 'FigureCanvasQT')
# Getting the type of '_BackendQT5'
_BackendQT5_252305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendQT5')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendQT5_252305, 'FigureCanvas', FigureCanvasQT_252304)

# Assigning a Name to a Name (line 851):
# Getting the type of 'FigureManagerQT' (line 851)
FigureManagerQT_252306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 20), 'FigureManagerQT')
# Getting the type of '_BackendQT5'
_BackendQT5_252307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendQT5')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendQT5_252307, 'FigureManager', FigureManagerQT_252306)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
