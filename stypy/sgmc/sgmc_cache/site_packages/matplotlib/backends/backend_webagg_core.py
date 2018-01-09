
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Displays Agg images in the browser, with interactivity
3: '''
4: # The WebAgg backend is divided into two modules:
5: #
6: # - `backend_webagg_core.py` contains code necessary to embed a WebAgg
7: #   plot inside of a web application, and communicate in an abstract
8: #   way over a web socket.
9: #
10: # - `backend_webagg.py` contains a concrete implementation of a basic
11: #   application, implemented with tornado.
12: 
13: from __future__ import (absolute_import, division, print_function,
14:                         unicode_literals)
15: 
16: import six
17: 
18: import io
19: import json
20: import os
21: import time
22: import warnings
23: 
24: import numpy as np
25: import tornado
26: import datetime
27: 
28: from matplotlib.backends import backend_agg
29: from matplotlib.backend_bases import _Backend
30: from matplotlib.figure import Figure
31: from matplotlib import backend_bases
32: from matplotlib import _png
33: 
34: 
35: # http://www.cambiaresearch.com/articles/15/javascript-char-codes-key-codes
36: _SHIFT_LUT = {59: ':',
37:               61: '+',
38:               173: '_',
39:               186: ':',
40:               187: '+',
41:               188: '<',
42:               189: '_',
43:               190: '>',
44:               191: '?',
45:               192: '~',
46:               219: '{',
47:               220: '|',
48:               221: '}',
49:               222: '"'}
50: 
51: _LUT = {8: 'backspace',
52:         9: 'tab',
53:         13: 'enter',
54:         16: 'shift',
55:         17: 'control',
56:         18: 'alt',
57:         19: 'pause',
58:         20: 'caps',
59:         27: 'escape',
60:         32: ' ',
61:         33: 'pageup',
62:         34: 'pagedown',
63:         35: 'end',
64:         36: 'home',
65:         37: 'left',
66:         38: 'up',
67:         39: 'right',
68:         40: 'down',
69:         45: 'insert',
70:         46: 'delete',
71:         91: 'super',
72:         92: 'super',
73:         93: 'select',
74:         106: '*',
75:         107: '+',
76:         109: '-',
77:         110: '.',
78:         111: '/',
79:         144: 'num_lock',
80:         145: 'scroll_lock',
81:         186: ':',
82:         187: '=',
83:         188: ',',
84:         189: '-',
85:         190: '.',
86:         191: '/',
87:         192: '`',
88:         219: '[',
89:         220: '\\',
90:         221: ']',
91:         222: "'"}
92: 
93: 
94: def _handle_key(key):
95:     '''Handle key codes'''
96:     code = int(key[key.index('k') + 1:])
97:     value = chr(code)
98:     # letter keys
99:     if code >= 65 and code <= 90:
100:         if 'shift+' in key:
101:             key = key.replace('shift+', '')
102:         else:
103:             value = value.lower()
104:     # number keys
105:     elif code >= 48 and code <= 57:
106:         if 'shift+' in key:
107:             value = ')!@#$%^&*('[int(value)]
108:             key = key.replace('shift+', '')
109:     # function keys
110:     elif code >= 112 and code <= 123:
111:         value = 'f%s' % (code - 111)
112:     # number pad keys
113:     elif code >= 96 and code <= 105:
114:         value = '%s' % (code - 96)
115:     # keys with shift alternatives
116:     elif code in _SHIFT_LUT and 'shift+' in key:
117:         key = key.replace('shift+', '')
118:         value = _SHIFT_LUT[code]
119:     elif code in _LUT:
120:         value = _LUT[code]
121:     key = key[:key.index('k')] + value
122:     return key
123: 
124: 
125: class FigureCanvasWebAggCore(backend_agg.FigureCanvasAgg):
126:     supports_blit = False
127: 
128:     def __init__(self, *args, **kwargs):
129:         backend_agg.FigureCanvasAgg.__init__(self, *args, **kwargs)
130: 
131:         # Set to True when the renderer contains data that is newer
132:         # than the PNG buffer.
133:         self._png_is_old = True
134: 
135:         # Set to True by the `refresh` message so that the next frame
136:         # sent to the clients will be a full frame.
137:         self._force_full = True
138: 
139:         # Store the current image mode so that at any point, clients can
140:         # request the information. This should be changed by calling
141:         # self.set_image_mode(mode) so that the notification can be given
142:         # to the connected clients.
143:         self._current_image_mode = 'full'
144: 
145:         # Store the DPI ratio of the browser.  This is the scaling that
146:         # occurs automatically for all images on a HiDPI display.
147:         self._dpi_ratio = 1
148: 
149:     def show(self):
150:         # show the figure window
151:         from matplotlib.pyplot import show
152:         show()
153: 
154:     def draw(self):
155:         renderer = self.get_renderer(cleared=True)
156: 
157:         self._png_is_old = True
158: 
159:         backend_agg.RendererAgg.lock.acquire()
160:         try:
161:             self.figure.draw(renderer)
162:         finally:
163:             backend_agg.RendererAgg.lock.release()
164:             # Swap the frames
165:             self.manager.refresh_all()
166: 
167:     def draw_idle(self):
168:         self.send_event("draw")
169: 
170:     def set_image_mode(self, mode):
171:         '''
172:         Set the image mode for any subsequent images which will be sent
173:         to the clients. The modes may currently be either 'full' or 'diff'.
174: 
175:         Note: diff images may not contain transparency, therefore upon
176:         draw this mode may be changed if the resulting image has any
177:         transparent component.
178: 
179:         '''
180:         if mode not in ['full', 'diff']:
181:             raise ValueError('image mode must be either full or diff.')
182:         if self._current_image_mode != mode:
183:             self._current_image_mode = mode
184:             self.handle_send_image_mode(None)
185: 
186:     def get_diff_image(self):
187:         if self._png_is_old:
188:             renderer = self.get_renderer()
189: 
190:             # The buffer is created as type uint32 so that entire
191:             # pixels can be compared in one numpy call, rather than
192:             # needing to compare each plane separately.
193:             buff = (np.frombuffer(renderer.buffer_rgba(), dtype=np.uint32)
194:                     .reshape((renderer.height, renderer.width)))
195: 
196:             # If any pixels have transparency, we need to force a full
197:             # draw as we cannot overlay new on top of old.
198:             pixels = buff.view(dtype=np.uint8).reshape(buff.shape + (4,))
199: 
200:             if self._force_full or np.any(pixels[:, :, 3] != 255):
201:                 self.set_image_mode('full')
202:                 output = buff
203:             else:
204:                 self.set_image_mode('diff')
205:                 last_buffer = (np.frombuffer(self._last_renderer.buffer_rgba(),
206:                                              dtype=np.uint32)
207:                                .reshape((renderer.height, renderer.width)))
208:                 diff = buff != last_buffer
209:                 output = np.where(diff, buff, 0)
210: 
211:             # TODO: We should write a new version of write_png that
212:             # handles the differencing inline
213:             buff = _png.write_png(
214:                 output.view(dtype=np.uint8).reshape(output.shape + (4,)),
215:                 None, compression=6, filter=_png.PNG_FILTER_NONE)
216: 
217:             # Swap the renderer frames
218:             self._renderer, self._last_renderer = (
219:                 self._last_renderer, renderer)
220:             self._force_full = False
221:             self._png_is_old = False
222:             return buff
223: 
224:     def get_renderer(self, cleared=None):
225:         # Mirrors super.get_renderer, but caches the old one
226:         # so that we can do things such as produce a diff image
227:         # in get_diff_image
228:         _, _, w, h = self.figure.bbox.bounds
229:         w, h = int(w), int(h)
230:         key = w, h, self.figure.dpi
231:         try:
232:             self._lastKey, self._renderer
233:         except AttributeError:
234:             need_new_renderer = True
235:         else:
236:             need_new_renderer = (self._lastKey != key)
237: 
238:         if need_new_renderer:
239:             self._renderer = backend_agg.RendererAgg(
240:                 w, h, self.figure.dpi)
241:             self._last_renderer = backend_agg.RendererAgg(
242:                 w, h, self.figure.dpi)
243:             self._lastKey = key
244: 
245:         elif cleared:
246:             self._renderer.clear()
247: 
248:         return self._renderer
249: 
250:     def handle_event(self, event):
251:         e_type = event['type']
252:         handler = getattr(self, 'handle_{0}'.format(e_type),
253:                           self.handle_unknown_event)
254:         return handler(event)
255: 
256:     def handle_unknown_event(self, event):
257:         warnings.warn('Unhandled message type {0}. {1}'.format(
258:             event['type'], event))
259: 
260:     def handle_ack(self, event):
261:         # Network latency tends to decrease if traffic is flowing
262:         # in both directions.  Therefore, the browser sends back
263:         # an "ack" message after each image frame is received.
264:         # This could also be used as a simple sanity check in the
265:         # future, but for now the performance increase is enough
266:         # to justify it, even if the server does nothing with it.
267:         pass
268: 
269:     def handle_draw(self, event):
270:         self.draw()
271: 
272:     def _handle_mouse(self, event):
273:         x = event['x']
274:         y = event['y']
275:         y = self.get_renderer().height - y
276: 
277:         # Javascript button numbers and matplotlib button numbers are
278:         # off by 1
279:         button = event['button'] + 1
280: 
281:         # The right mouse button pops up a context menu, which
282:         # doesn't work very well, so use the middle mouse button
283:         # instead.  It doesn't seem that it's possible to disable
284:         # the context menu in recent versions of Chrome.  If this
285:         # is resolved, please also adjust the docstring in MouseEvent.
286:         if button == 2:
287:             button = 3
288: 
289:         e_type = event['type']
290:         guiEvent = event.get('guiEvent', None)
291:         if e_type == 'button_press':
292:             self.button_press_event(x, y, button, guiEvent=guiEvent)
293:         elif e_type == 'button_release':
294:             self.button_release_event(x, y, button, guiEvent=guiEvent)
295:         elif e_type == 'motion_notify':
296:             self.motion_notify_event(x, y, guiEvent=guiEvent)
297:         elif e_type == 'figure_enter':
298:             self.enter_notify_event(xy=(x, y), guiEvent=guiEvent)
299:         elif e_type == 'figure_leave':
300:             self.leave_notify_event()
301:         elif e_type == 'scroll':
302:             self.scroll_event(x, y, event['step'], guiEvent=guiEvent)
303:     handle_button_press = handle_button_release = handle_motion_notify = \
304:         handle_figure_enter = handle_figure_leave = handle_scroll = \
305:         _handle_mouse
306: 
307:     def _handle_key(self, event):
308:         key = _handle_key(event['key'])
309:         e_type = event['type']
310:         guiEvent = event.get('guiEvent', None)
311:         if e_type == 'key_press':
312:             self.key_press_event(key, guiEvent=guiEvent)
313:         elif e_type == 'key_release':
314:             self.key_release_event(key, guiEvent=guiEvent)
315:     handle_key_press = handle_key_release = _handle_key
316: 
317:     def handle_toolbar_button(self, event):
318:         # TODO: Be more suspicious of the input
319:         getattr(self.toolbar, event['name'])()
320: 
321:     def handle_refresh(self, event):
322:         figure_label = self.figure.get_label()
323:         if not figure_label:
324:             figure_label = "Figure {0}".format(self.manager.num)
325:         self.send_event('figure_label', label=figure_label)
326:         self._force_full = True
327:         self.draw_idle()
328: 
329:     def handle_resize(self, event):
330:         x, y = event.get('width', 800), event.get('height', 800)
331:         x, y = int(x) * self._dpi_ratio, int(y) * self._dpi_ratio
332:         fig = self.figure
333:         # An attempt at approximating the figure size in pixels.
334:         fig.set_size_inches(x / fig.dpi, y / fig.dpi, forward=False)
335: 
336:         _, _, w, h = self.figure.bbox.bounds
337:         # Acknowledge the resize, and force the viewer to update the
338:         # canvas size to the figure's new size (which is hopefully
339:         # identical or within a pixel or so).
340:         self._png_is_old = True
341:         self.manager.resize(w, h)
342:         self.resize_event()
343: 
344:     def handle_send_image_mode(self, event):
345:         # The client requests notification of what the current image mode is.
346:         self.send_event('image_mode', mode=self._current_image_mode)
347: 
348:     def handle_set_dpi_ratio(self, event):
349:         dpi_ratio = event.get('dpi_ratio', 1)
350:         if dpi_ratio != self._dpi_ratio:
351:             # We don't want to scale up the figure dpi more than once.
352:             if not hasattr(self.figure, '_original_dpi'):
353:                 self.figure._original_dpi = self.figure.dpi
354:             self.figure.dpi = dpi_ratio * self.figure._original_dpi
355:             self._dpi_ratio = dpi_ratio
356:             self._force_full = True
357:             self.draw_idle()
358: 
359:     def send_event(self, event_type, **kwargs):
360:         self.manager._send_event(event_type, **kwargs)
361: 
362: 
363: _JQUERY_ICON_CLASSES = {
364:     'home': 'ui-icon ui-icon-home',
365:     'back': 'ui-icon ui-icon-circle-arrow-w',
366:     'forward': 'ui-icon ui-icon-circle-arrow-e',
367:     'zoom_to_rect': 'ui-icon ui-icon-search',
368:     'move': 'ui-icon ui-icon-arrow-4',
369:     'download': 'ui-icon ui-icon-disk',
370:     None: None,
371: }
372: 
373: 
374: class NavigationToolbar2WebAgg(backend_bases.NavigationToolbar2):
375: 
376:     # Use the standard toolbar items + download button
377:     toolitems = [(text, tooltip_text, _JQUERY_ICON_CLASSES[image_file],
378:                   name_of_method)
379:                  for text, tooltip_text, image_file, name_of_method
380:                  in (backend_bases.NavigationToolbar2.toolitems +
381:                      (('Download', 'Download plot', 'download', 'download'),))
382:                  if image_file in _JQUERY_ICON_CLASSES]
383: 
384:     def _init_toolbar(self):
385:         self.message = ''
386:         self.cursor = 0
387: 
388:     def set_message(self, message):
389:         if message != self.message:
390:             self.canvas.send_event("message", message=message)
391:         self.message = message
392: 
393:     def set_cursor(self, cursor):
394:         if cursor != self.cursor:
395:             self.canvas.send_event("cursor", cursor=cursor)
396:         self.cursor = cursor
397: 
398:     def draw_rubberband(self, event, x0, y0, x1, y1):
399:         self.canvas.send_event(
400:             "rubberband", x0=x0, y0=y0, x1=x1, y1=y1)
401: 
402:     def release_zoom(self, event):
403:         backend_bases.NavigationToolbar2.release_zoom(self, event)
404:         self.canvas.send_event(
405:             "rubberband", x0=-1, y0=-1, x1=-1, y1=-1)
406: 
407:     def save_figure(self, *args):
408:         '''Save the current figure'''
409:         self.canvas.send_event('save')
410: 
411: 
412: class FigureManagerWebAgg(backend_bases.FigureManagerBase):
413:     ToolbarCls = NavigationToolbar2WebAgg
414: 
415:     def __init__(self, canvas, num):
416:         backend_bases.FigureManagerBase.__init__(self, canvas, num)
417: 
418:         self.web_sockets = set()
419: 
420:         self.toolbar = self._get_toolbar(canvas)
421: 
422:     def show(self):
423:         pass
424: 
425:     def _get_toolbar(self, canvas):
426:         toolbar = self.ToolbarCls(canvas)
427:         return toolbar
428: 
429:     def resize(self, w, h):
430:         self._send_event(
431:             'resize',
432:             size=(w / self.canvas._dpi_ratio, h / self.canvas._dpi_ratio))
433: 
434:     def set_window_title(self, title):
435:         self._send_event('figure_label', label=title)
436: 
437:     # The following methods are specific to FigureManagerWebAgg
438: 
439:     def add_web_socket(self, web_socket):
440:         assert hasattr(web_socket, 'send_binary')
441:         assert hasattr(web_socket, 'send_json')
442: 
443:         self.web_sockets.add(web_socket)
444: 
445:         _, _, w, h = self.canvas.figure.bbox.bounds
446:         self.resize(w, h)
447:         self._send_event('refresh')
448: 
449:     def remove_web_socket(self, web_socket):
450:         self.web_sockets.remove(web_socket)
451: 
452:     def handle_json(self, content):
453:         self.canvas.handle_event(content)
454: 
455:     def refresh_all(self):
456:         if self.web_sockets:
457:             diff = self.canvas.get_diff_image()
458:             if diff is not None:
459:                 for s in self.web_sockets:
460:                     s.send_binary(diff)
461: 
462:     @classmethod
463:     def get_javascript(cls, stream=None):
464:         if stream is None:
465:             output = io.StringIO()
466:         else:
467:             output = stream
468: 
469:         with io.open(os.path.join(
470:                 os.path.dirname(__file__),
471:                 "web_backend",
472:                 "mpl.js"), encoding='utf8') as fd:
473:             output.write(fd.read())
474: 
475:         toolitems = []
476:         for name, tooltip, image, method in cls.ToolbarCls.toolitems:
477:             if name is None:
478:                 toolitems.append(['', '', '', ''])
479:             else:
480:                 toolitems.append([name, tooltip, image, method])
481:         output.write("mpl.toolbar_items = {0};\n\n".format(
482:             json.dumps(toolitems)))
483: 
484:         extensions = []
485:         for filetype, ext in sorted(FigureCanvasWebAggCore.
486:                                     get_supported_filetypes_grouped().
487:                                     items()):
488:             if not ext[0] == 'pgf':  # pgf does not support BytesIO
489:                 extensions.append(ext[0])
490:         output.write("mpl.extensions = {0};\n\n".format(
491:             json.dumps(extensions)))
492: 
493:         output.write("mpl.default_extension = {0};".format(
494:             json.dumps(FigureCanvasWebAggCore.get_default_filetype())))
495: 
496:         if stream is None:
497:             return output.getvalue()
498: 
499:     @classmethod
500:     def get_static_file_path(cls):
501:         return os.path.join(os.path.dirname(__file__), 'web_backend')
502: 
503:     def _send_event(self, event_type, **kwargs):
504:         payload = {'type': event_type}
505:         payload.update(kwargs)
506:         for s in self.web_sockets:
507:             s.send_json(payload)
508: 
509: 
510: class TimerTornado(backend_bases.TimerBase):
511:     def _timer_start(self):
512:         self._timer_stop()
513:         if self._single:
514:             ioloop = tornado.ioloop.IOLoop.instance()
515:             self._timer = ioloop.add_timeout(
516:                 datetime.timedelta(milliseconds=self.interval),
517:                 self._on_timer)
518:         else:
519:             self._timer = tornado.ioloop.PeriodicCallback(
520:                 self._on_timer,
521:                 self.interval)
522:             self._timer.start()
523: 
524:     def _timer_stop(self):
525:         if self._timer is None:
526:             return
527:         elif self._single:
528:             ioloop = tornado.ioloop.IOLoop.instance()
529:             ioloop.remove_timeout(self._timer)
530:         else:
531:             self._timer.stop()
532: 
533:         self._timer = None
534: 
535:     def _timer_set_interval(self):
536:         # Only stop and restart it if the timer has already been started
537:         if self._timer is not None:
538:             self._timer_stop()
539:             self._timer_start()
540: 
541: 
542: @_Backend.export
543: class _BackendWebAggCoreAgg(_Backend):
544:     FigureCanvas = FigureCanvasWebAggCore
545:     FigureManager = FigureManagerWebAgg
546: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_261468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nDisplays Agg images in the browser, with interactivity\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import six' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_261469 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six')

if (type(import_261469) is not StypyTypeError):

    if (import_261469 != 'pyd_module'):
        __import__(import_261469)
        sys_modules_261470 = sys.modules[import_261469]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', sys_modules_261470.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', import_261469)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import io' statement (line 18)
import io

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'io', io, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import json' statement (line 19)
import json

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'json', json, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import os' statement (line 20)
import os

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import time' statement (line 21)
import time

import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import warnings' statement (line 22)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import numpy' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_261471 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy')

if (type(import_261471) is not StypyTypeError):

    if (import_261471 != 'pyd_module'):
        __import__(import_261471)
        sys_modules_261472 = sys.modules[import_261471]
        import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'np', sys_modules_261472.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy', import_261471)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'import tornado' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_261473 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'tornado')

if (type(import_261473) is not StypyTypeError):

    if (import_261473 != 'pyd_module'):
        __import__(import_261473)
        sys_modules_261474 = sys.modules[import_261473]
        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'tornado', sys_modules_261474.module_type_store, module_type_store)
    else:
        import tornado

        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'tornado', tornado, module_type_store)

else:
    # Assigning a type to the variable 'tornado' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'tornado', import_261473)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'import datetime' statement (line 26)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'datetime', datetime, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from matplotlib.backends import backend_agg' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_261475 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.backends')

if (type(import_261475) is not StypyTypeError):

    if (import_261475 != 'pyd_module'):
        __import__(import_261475)
        sys_modules_261476 = sys.modules[import_261475]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.backends', sys_modules_261476.module_type_store, module_type_store, ['backend_agg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_261476, sys_modules_261476.module_type_store, module_type_store)
    else:
        from matplotlib.backends import backend_agg

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.backends', None, module_type_store, ['backend_agg'], [backend_agg])

else:
    # Assigning a type to the variable 'matplotlib.backends' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'matplotlib.backends', import_261475)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from matplotlib.backend_bases import _Backend' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_261477 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.backend_bases')

if (type(import_261477) is not StypyTypeError):

    if (import_261477 != 'pyd_module'):
        __import__(import_261477)
        sys_modules_261478 = sys.modules[import_261477]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.backend_bases', sys_modules_261478.module_type_store, module_type_store, ['_Backend'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_261478, sys_modules_261478.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import _Backend

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.backend_bases', None, module_type_store, ['_Backend'], [_Backend])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.backend_bases', import_261477)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from matplotlib.figure import Figure' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_261479 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.figure')

if (type(import_261479) is not StypyTypeError):

    if (import_261479 != 'pyd_module'):
        __import__(import_261479)
        sys_modules_261480 = sys.modules[import_261479]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.figure', sys_modules_261480.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_261480, sys_modules_261480.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'matplotlib.figure', import_261479)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from matplotlib import backend_bases' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_261481 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib')

if (type(import_261481) is not StypyTypeError):

    if (import_261481 != 'pyd_module'):
        __import__(import_261481)
        sys_modules_261482 = sys.modules[import_261481]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib', sys_modules_261482.module_type_store, module_type_store, ['backend_bases'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_261482, sys_modules_261482.module_type_store, module_type_store)
    else:
        from matplotlib import backend_bases

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib', None, module_type_store, ['backend_bases'], [backend_bases])

else:
    # Assigning a type to the variable 'matplotlib' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib', import_261481)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from matplotlib import _png' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_261483 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib')

if (type(import_261483) is not StypyTypeError):

    if (import_261483 != 'pyd_module'):
        __import__(import_261483)
        sys_modules_261484 = sys.modules[import_261483]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib', sys_modules_261484.module_type_store, module_type_store, ['_png'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_261484, sys_modules_261484.module_type_store, module_type_store)
    else:
        from matplotlib import _png

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib', None, module_type_store, ['_png'], [_png])

else:
    # Assigning a type to the variable 'matplotlib' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib', import_261483)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


# Assigning a Dict to a Name (line 36):

# Assigning a Dict to a Name (line 36):

# Obtaining an instance of the builtin type 'dict' (line 36)
dict_261485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 36)
# Adding element type (key, value) (line 36)
int_261486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'int')
unicode_261487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'unicode', u':')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261486, unicode_261487))
# Adding element type (key, value) (line 36)
int_261488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 14), 'int')
unicode_261489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'unicode', u'+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261488, unicode_261489))
# Adding element type (key, value) (line 36)
int_261490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 14), 'int')
unicode_261491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 19), 'unicode', u'_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261490, unicode_261491))
# Adding element type (key, value) (line 36)
int_261492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 14), 'int')
unicode_261493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'unicode', u':')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261492, unicode_261493))
# Adding element type (key, value) (line 36)
int_261494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 14), 'int')
unicode_261495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'unicode', u'+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261494, unicode_261495))
# Adding element type (key, value) (line 36)
int_261496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'int')
unicode_261497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'unicode', u'<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261496, unicode_261497))
# Adding element type (key, value) (line 36)
int_261498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 14), 'int')
unicode_261499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'unicode', u'_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261498, unicode_261499))
# Adding element type (key, value) (line 36)
int_261500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 14), 'int')
unicode_261501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'unicode', u'>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261500, unicode_261501))
# Adding element type (key, value) (line 36)
int_261502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 14), 'int')
unicode_261503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'unicode', u'?')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261502, unicode_261503))
# Adding element type (key, value) (line 36)
int_261504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 14), 'int')
unicode_261505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'unicode', u'~')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261504, unicode_261505))
# Adding element type (key, value) (line 36)
int_261506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 14), 'int')
unicode_261507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'unicode', u'{')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261506, unicode_261507))
# Adding element type (key, value) (line 36)
int_261508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 14), 'int')
unicode_261509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'unicode', u'|')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261508, unicode_261509))
# Adding element type (key, value) (line 36)
int_261510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 14), 'int')
unicode_261511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'unicode', u'}')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261510, unicode_261511))
# Adding element type (key, value) (line 36)
int_261512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 14), 'int')
unicode_261513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 19), 'unicode', u'"')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 13), dict_261485, (int_261512, unicode_261513))

# Assigning a type to the variable '_SHIFT_LUT' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), '_SHIFT_LUT', dict_261485)

# Assigning a Dict to a Name (line 51):

# Assigning a Dict to a Name (line 51):

# Obtaining an instance of the builtin type 'dict' (line 51)
dict_261514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 7), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 51)
# Adding element type (key, value) (line 51)
int_261515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'int')
unicode_261516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 11), 'unicode', u'backspace')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261515, unicode_261516))
# Adding element type (key, value) (line 51)
int_261517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'int')
unicode_261518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 11), 'unicode', u'tab')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261517, unicode_261518))
# Adding element type (key, value) (line 51)
int_261519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 8), 'int')
unicode_261520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 12), 'unicode', u'enter')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261519, unicode_261520))
# Adding element type (key, value) (line 51)
int_261521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
unicode_261522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'unicode', u'shift')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261521, unicode_261522))
# Adding element type (key, value) (line 51)
int_261523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'int')
unicode_261524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 12), 'unicode', u'control')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261523, unicode_261524))
# Adding element type (key, value) (line 51)
int_261525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
unicode_261526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'unicode', u'alt')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261525, unicode_261526))
# Adding element type (key, value) (line 51)
int_261527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 8), 'int')
unicode_261528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'unicode', u'pause')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261527, unicode_261528))
# Adding element type (key, value) (line 51)
int_261529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'int')
unicode_261530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 12), 'unicode', u'caps')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261529, unicode_261530))
# Adding element type (key, value) (line 51)
int_261531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'int')
unicode_261532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 12), 'unicode', u'escape')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261531, unicode_261532))
# Adding element type (key, value) (line 51)
int_261533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
unicode_261534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'unicode', u' ')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261533, unicode_261534))
# Adding element type (key, value) (line 51)
int_261535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
unicode_261536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 12), 'unicode', u'pageup')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261535, unicode_261536))
# Adding element type (key, value) (line 51)
int_261537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'int')
unicode_261538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'unicode', u'pagedown')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261537, unicode_261538))
# Adding element type (key, value) (line 51)
int_261539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'int')
unicode_261540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 12), 'unicode', u'end')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261539, unicode_261540))
# Adding element type (key, value) (line 51)
int_261541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
unicode_261542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 12), 'unicode', u'home')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261541, unicode_261542))
# Adding element type (key, value) (line 51)
int_261543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 8), 'int')
unicode_261544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 12), 'unicode', u'left')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261543, unicode_261544))
# Adding element type (key, value) (line 51)
int_261545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 8), 'int')
unicode_261546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 12), 'unicode', u'up')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261545, unicode_261546))
# Adding element type (key, value) (line 51)
int_261547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 8), 'int')
unicode_261548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 12), 'unicode', u'right')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261547, unicode_261548))
# Adding element type (key, value) (line 51)
int_261549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 8), 'int')
unicode_261550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 12), 'unicode', u'down')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261549, unicode_261550))
# Adding element type (key, value) (line 51)
int_261551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'int')
unicode_261552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 12), 'unicode', u'insert')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261551, unicode_261552))
# Adding element type (key, value) (line 51)
int_261553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 8), 'int')
unicode_261554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'unicode', u'delete')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261553, unicode_261554))
# Adding element type (key, value) (line 51)
int_261555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 8), 'int')
unicode_261556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 12), 'unicode', u'super')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261555, unicode_261556))
# Adding element type (key, value) (line 51)
int_261557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 8), 'int')
unicode_261558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 12), 'unicode', u'super')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261557, unicode_261558))
# Adding element type (key, value) (line 51)
int_261559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'int')
unicode_261560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 12), 'unicode', u'select')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261559, unicode_261560))
# Adding element type (key, value) (line 51)
int_261561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 8), 'int')
unicode_261562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 13), 'unicode', u'*')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261561, unicode_261562))
# Adding element type (key, value) (line 51)
int_261563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'int')
unicode_261564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 13), 'unicode', u'+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261563, unicode_261564))
# Adding element type (key, value) (line 51)
int_261565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'int')
unicode_261566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 13), 'unicode', u'-')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261565, unicode_261566))
# Adding element type (key, value) (line 51)
int_261567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 8), 'int')
unicode_261568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 13), 'unicode', u'.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261567, unicode_261568))
# Adding element type (key, value) (line 51)
int_261569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 8), 'int')
unicode_261570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 13), 'unicode', u'/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261569, unicode_261570))
# Adding element type (key, value) (line 51)
int_261571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 8), 'int')
unicode_261572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 13), 'unicode', u'num_lock')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261571, unicode_261572))
# Adding element type (key, value) (line 51)
int_261573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'int')
unicode_261574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 13), 'unicode', u'scroll_lock')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261573, unicode_261574))
# Adding element type (key, value) (line 51)
int_261575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 8), 'int')
unicode_261576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 13), 'unicode', u':')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261575, unicode_261576))
# Adding element type (key, value) (line 51)
int_261577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
unicode_261578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 13), 'unicode', u'=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261577, unicode_261578))
# Adding element type (key, value) (line 51)
int_261579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'int')
unicode_261580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 13), 'unicode', u',')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261579, unicode_261580))
# Adding element type (key, value) (line 51)
int_261581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
unicode_261582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 13), 'unicode', u'-')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261581, unicode_261582))
# Adding element type (key, value) (line 51)
int_261583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
unicode_261584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 13), 'unicode', u'.')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261583, unicode_261584))
# Adding element type (key, value) (line 51)
int_261585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
unicode_261586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 13), 'unicode', u'/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261585, unicode_261586))
# Adding element type (key, value) (line 51)
int_261587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'int')
unicode_261588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 13), 'unicode', u'`')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261587, unicode_261588))
# Adding element type (key, value) (line 51)
int_261589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 8), 'int')
unicode_261590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'unicode', u'[')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261589, unicode_261590))
# Adding element type (key, value) (line 51)
int_261591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'int')
unicode_261592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 13), 'unicode', u'\\')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261591, unicode_261592))
# Adding element type (key, value) (line 51)
int_261593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'int')
unicode_261594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 13), 'unicode', u']')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261593, unicode_261594))
# Adding element type (key, value) (line 51)
int_261595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
unicode_261596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 13), 'unicode', u"'")
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 7), dict_261514, (int_261595, unicode_261596))

# Assigning a type to the variable '_LUT' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '_LUT', dict_261514)

@norecursion
def _handle_key(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_handle_key'
    module_type_store = module_type_store.open_function_context('_handle_key', 94, 0, False)
    
    # Passed parameters checking function
    _handle_key.stypy_localization = localization
    _handle_key.stypy_type_of_self = None
    _handle_key.stypy_type_store = module_type_store
    _handle_key.stypy_function_name = '_handle_key'
    _handle_key.stypy_param_names_list = ['key']
    _handle_key.stypy_varargs_param_name = None
    _handle_key.stypy_kwargs_param_name = None
    _handle_key.stypy_call_defaults = defaults
    _handle_key.stypy_call_varargs = varargs
    _handle_key.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_handle_key', ['key'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_handle_key', localization, ['key'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_handle_key(...)' code ##################

    unicode_261597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 4), 'unicode', u'Handle key codes')
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to int(...): (line 96)
    # Processing the call arguments (line 96)
    
    # Obtaining the type of the subscript
    
    # Call to index(...): (line 96)
    # Processing the call arguments (line 96)
    unicode_261601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 29), 'unicode', u'k')
    # Processing the call keyword arguments (line 96)
    kwargs_261602 = {}
    # Getting the type of 'key' (line 96)
    key_261599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'key', False)
    # Obtaining the member 'index' of a type (line 96)
    index_261600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 19), key_261599, 'index')
    # Calling index(args, kwargs) (line 96)
    index_call_result_261603 = invoke(stypy.reporting.localization.Localization(__file__, 96, 19), index_261600, *[unicode_261601], **kwargs_261602)
    
    int_261604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 36), 'int')
    # Applying the binary operator '+' (line 96)
    result_add_261605 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 19), '+', index_call_result_261603, int_261604)
    
    slice_261606 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 96, 15), result_add_261605, None, None)
    # Getting the type of 'key' (line 96)
    key_261607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'key', False)
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___261608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 15), key_261607, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_261609 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), getitem___261608, slice_261606)
    
    # Processing the call keyword arguments (line 96)
    kwargs_261610 = {}
    # Getting the type of 'int' (line 96)
    int_261598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'int', False)
    # Calling int(args, kwargs) (line 96)
    int_call_result_261611 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), int_261598, *[subscript_call_result_261609], **kwargs_261610)
    
    # Assigning a type to the variable 'code' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'code', int_call_result_261611)
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to chr(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'code' (line 97)
    code_261613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'code', False)
    # Processing the call keyword arguments (line 97)
    kwargs_261614 = {}
    # Getting the type of 'chr' (line 97)
    chr_261612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'chr', False)
    # Calling chr(args, kwargs) (line 97)
    chr_call_result_261615 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), chr_261612, *[code_261613], **kwargs_261614)
    
    # Assigning a type to the variable 'value' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'value', chr_call_result_261615)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'code' (line 99)
    code_261616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 7), 'code')
    int_261617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 15), 'int')
    # Applying the binary operator '>=' (line 99)
    result_ge_261618 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), '>=', code_261616, int_261617)
    
    
    # Getting the type of 'code' (line 99)
    code_261619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'code')
    int_261620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 30), 'int')
    # Applying the binary operator '<=' (line 99)
    result_le_261621 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 22), '<=', code_261619, int_261620)
    
    # Applying the binary operator 'and' (line 99)
    result_and_keyword_261622 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), 'and', result_ge_261618, result_le_261621)
    
    # Testing the type of an if condition (line 99)
    if_condition_261623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 4), result_and_keyword_261622)
    # Assigning a type to the variable 'if_condition_261623' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'if_condition_261623', if_condition_261623)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    unicode_261624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 11), 'unicode', u'shift+')
    # Getting the type of 'key' (line 100)
    key_261625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'key')
    # Applying the binary operator 'in' (line 100)
    result_contains_261626 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 11), 'in', unicode_261624, key_261625)
    
    # Testing the type of an if condition (line 100)
    if_condition_261627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 8), result_contains_261626)
    # Assigning a type to the variable 'if_condition_261627' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'if_condition_261627', if_condition_261627)
    # SSA begins for if statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to replace(...): (line 101)
    # Processing the call arguments (line 101)
    unicode_261630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 30), 'unicode', u'shift+')
    unicode_261631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 40), 'unicode', u'')
    # Processing the call keyword arguments (line 101)
    kwargs_261632 = {}
    # Getting the type of 'key' (line 101)
    key_261628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'key', False)
    # Obtaining the member 'replace' of a type (line 101)
    replace_261629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 18), key_261628, 'replace')
    # Calling replace(args, kwargs) (line 101)
    replace_call_result_261633 = invoke(stypy.reporting.localization.Localization(__file__, 101, 18), replace_261629, *[unicode_261630, unicode_261631], **kwargs_261632)
    
    # Assigning a type to the variable 'key' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'key', replace_call_result_261633)
    # SSA branch for the else part of an if statement (line 100)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to lower(...): (line 103)
    # Processing the call keyword arguments (line 103)
    kwargs_261636 = {}
    # Getting the type of 'value' (line 103)
    value_261634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'value', False)
    # Obtaining the member 'lower' of a type (line 103)
    lower_261635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 20), value_261634, 'lower')
    # Calling lower(args, kwargs) (line 103)
    lower_call_result_261637 = invoke(stypy.reporting.localization.Localization(__file__, 103, 20), lower_261635, *[], **kwargs_261636)
    
    # Assigning a type to the variable 'value' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'value', lower_call_result_261637)
    # SSA join for if statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 99)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'code' (line 105)
    code_261638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 9), 'code')
    int_261639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 17), 'int')
    # Applying the binary operator '>=' (line 105)
    result_ge_261640 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 9), '>=', code_261638, int_261639)
    
    
    # Getting the type of 'code' (line 105)
    code_261641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'code')
    int_261642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 32), 'int')
    # Applying the binary operator '<=' (line 105)
    result_le_261643 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 24), '<=', code_261641, int_261642)
    
    # Applying the binary operator 'and' (line 105)
    result_and_keyword_261644 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 9), 'and', result_ge_261640, result_le_261643)
    
    # Testing the type of an if condition (line 105)
    if_condition_261645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 9), result_and_keyword_261644)
    # Assigning a type to the variable 'if_condition_261645' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 9), 'if_condition_261645', if_condition_261645)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    unicode_261646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 11), 'unicode', u'shift+')
    # Getting the type of 'key' (line 106)
    key_261647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'key')
    # Applying the binary operator 'in' (line 106)
    result_contains_261648 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), 'in', unicode_261646, key_261647)
    
    # Testing the type of an if condition (line 106)
    if_condition_261649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), result_contains_261648)
    # Assigning a type to the variable 'if_condition_261649' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_261649', if_condition_261649)
    # SSA begins for if statement (line 106)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 107):
    
    # Assigning a Subscript to a Name (line 107):
    
    # Obtaining the type of the subscript
    
    # Call to int(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'value' (line 107)
    value_261651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 37), 'value', False)
    # Processing the call keyword arguments (line 107)
    kwargs_261652 = {}
    # Getting the type of 'int' (line 107)
    int_261650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 33), 'int', False)
    # Calling int(args, kwargs) (line 107)
    int_call_result_261653 = invoke(stypy.reporting.localization.Localization(__file__, 107, 33), int_261650, *[value_261651], **kwargs_261652)
    
    unicode_261654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 20), 'unicode', u')!@#$%^&*(')
    # Obtaining the member '__getitem__' of a type (line 107)
    getitem___261655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 20), unicode_261654, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 107)
    subscript_call_result_261656 = invoke(stypy.reporting.localization.Localization(__file__, 107, 20), getitem___261655, int_call_result_261653)
    
    # Assigning a type to the variable 'value' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'value', subscript_call_result_261656)
    
    # Assigning a Call to a Name (line 108):
    
    # Assigning a Call to a Name (line 108):
    
    # Call to replace(...): (line 108)
    # Processing the call arguments (line 108)
    unicode_261659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 30), 'unicode', u'shift+')
    unicode_261660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 40), 'unicode', u'')
    # Processing the call keyword arguments (line 108)
    kwargs_261661 = {}
    # Getting the type of 'key' (line 108)
    key_261657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'key', False)
    # Obtaining the member 'replace' of a type (line 108)
    replace_261658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 18), key_261657, 'replace')
    # Calling replace(args, kwargs) (line 108)
    replace_call_result_261662 = invoke(stypy.reporting.localization.Localization(__file__, 108, 18), replace_261658, *[unicode_261659, unicode_261660], **kwargs_261661)
    
    # Assigning a type to the variable 'key' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'key', replace_call_result_261662)
    # SSA join for if statement (line 106)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 105)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'code' (line 110)
    code_261663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 9), 'code')
    int_261664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 17), 'int')
    # Applying the binary operator '>=' (line 110)
    result_ge_261665 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 9), '>=', code_261663, int_261664)
    
    
    # Getting the type of 'code' (line 110)
    code_261666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'code')
    int_261667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 33), 'int')
    # Applying the binary operator '<=' (line 110)
    result_le_261668 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 25), '<=', code_261666, int_261667)
    
    # Applying the binary operator 'and' (line 110)
    result_and_keyword_261669 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 9), 'and', result_ge_261665, result_le_261668)
    
    # Testing the type of an if condition (line 110)
    if_condition_261670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 9), result_and_keyword_261669)
    # Assigning a type to the variable 'if_condition_261670' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 9), 'if_condition_261670', if_condition_261670)
    # SSA begins for if statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 111):
    
    # Assigning a BinOp to a Name (line 111):
    unicode_261671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 16), 'unicode', u'f%s')
    # Getting the type of 'code' (line 111)
    code_261672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'code')
    int_261673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 32), 'int')
    # Applying the binary operator '-' (line 111)
    result_sub_261674 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 25), '-', code_261672, int_261673)
    
    # Applying the binary operator '%' (line 111)
    result_mod_261675 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 16), '%', unicode_261671, result_sub_261674)
    
    # Assigning a type to the variable 'value' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'value', result_mod_261675)
    # SSA branch for the else part of an if statement (line 110)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'code' (line 113)
    code_261676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 9), 'code')
    int_261677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 17), 'int')
    # Applying the binary operator '>=' (line 113)
    result_ge_261678 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 9), '>=', code_261676, int_261677)
    
    
    # Getting the type of 'code' (line 113)
    code_261679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'code')
    int_261680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 32), 'int')
    # Applying the binary operator '<=' (line 113)
    result_le_261681 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 24), '<=', code_261679, int_261680)
    
    # Applying the binary operator 'and' (line 113)
    result_and_keyword_261682 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 9), 'and', result_ge_261678, result_le_261681)
    
    # Testing the type of an if condition (line 113)
    if_condition_261683 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 9), result_and_keyword_261682)
    # Assigning a type to the variable 'if_condition_261683' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 9), 'if_condition_261683', if_condition_261683)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 114):
    
    # Assigning a BinOp to a Name (line 114):
    unicode_261684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 16), 'unicode', u'%s')
    # Getting the type of 'code' (line 114)
    code_261685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'code')
    int_261686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 31), 'int')
    # Applying the binary operator '-' (line 114)
    result_sub_261687 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 24), '-', code_261685, int_261686)
    
    # Applying the binary operator '%' (line 114)
    result_mod_261688 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 16), '%', unicode_261684, result_sub_261687)
    
    # Assigning a type to the variable 'value' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'value', result_mod_261688)
    # SSA branch for the else part of an if statement (line 113)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'code' (line 116)
    code_261689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'code')
    # Getting the type of '_SHIFT_LUT' (line 116)
    _SHIFT_LUT_261690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), '_SHIFT_LUT')
    # Applying the binary operator 'in' (line 116)
    result_contains_261691 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 9), 'in', code_261689, _SHIFT_LUT_261690)
    
    
    unicode_261692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 32), 'unicode', u'shift+')
    # Getting the type of 'key' (line 116)
    key_261693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 44), 'key')
    # Applying the binary operator 'in' (line 116)
    result_contains_261694 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 32), 'in', unicode_261692, key_261693)
    
    # Applying the binary operator 'and' (line 116)
    result_and_keyword_261695 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 9), 'and', result_contains_261691, result_contains_261694)
    
    # Testing the type of an if condition (line 116)
    if_condition_261696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 9), result_and_keyword_261695)
    # Assigning a type to the variable 'if_condition_261696' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'if_condition_261696', if_condition_261696)
    # SSA begins for if statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 117):
    
    # Assigning a Call to a Name (line 117):
    
    # Call to replace(...): (line 117)
    # Processing the call arguments (line 117)
    unicode_261699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 26), 'unicode', u'shift+')
    unicode_261700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 36), 'unicode', u'')
    # Processing the call keyword arguments (line 117)
    kwargs_261701 = {}
    # Getting the type of 'key' (line 117)
    key_261697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 14), 'key', False)
    # Obtaining the member 'replace' of a type (line 117)
    replace_261698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 14), key_261697, 'replace')
    # Calling replace(args, kwargs) (line 117)
    replace_call_result_261702 = invoke(stypy.reporting.localization.Localization(__file__, 117, 14), replace_261698, *[unicode_261699, unicode_261700], **kwargs_261701)
    
    # Assigning a type to the variable 'key' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'key', replace_call_result_261702)
    
    # Assigning a Subscript to a Name (line 118):
    
    # Assigning a Subscript to a Name (line 118):
    
    # Obtaining the type of the subscript
    # Getting the type of 'code' (line 118)
    code_261703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'code')
    # Getting the type of '_SHIFT_LUT' (line 118)
    _SHIFT_LUT_261704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), '_SHIFT_LUT')
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___261705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 16), _SHIFT_LUT_261704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_261706 = invoke(stypy.reporting.localization.Localization(__file__, 118, 16), getitem___261705, code_261703)
    
    # Assigning a type to the variable 'value' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'value', subscript_call_result_261706)
    # SSA branch for the else part of an if statement (line 116)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'code' (line 119)
    code_261707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 9), 'code')
    # Getting the type of '_LUT' (line 119)
    _LUT_261708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), '_LUT')
    # Applying the binary operator 'in' (line 119)
    result_contains_261709 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 9), 'in', code_261707, _LUT_261708)
    
    # Testing the type of an if condition (line 119)
    if_condition_261710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 9), result_contains_261709)
    # Assigning a type to the variable 'if_condition_261710' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 9), 'if_condition_261710', if_condition_261710)
    # SSA begins for if statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 120):
    
    # Assigning a Subscript to a Name (line 120):
    
    # Obtaining the type of the subscript
    # Getting the type of 'code' (line 120)
    code_261711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'code')
    # Getting the type of '_LUT' (line 120)
    _LUT_261712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), '_LUT')
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___261713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), _LUT_261712, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_261714 = invoke(stypy.reporting.localization.Localization(__file__, 120, 16), getitem___261713, code_261711)
    
    # Assigning a type to the variable 'value' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'value', subscript_call_result_261714)
    # SSA join for if statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 110)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 121):
    
    # Assigning a BinOp to a Name (line 121):
    
    # Obtaining the type of the subscript
    
    # Call to index(...): (line 121)
    # Processing the call arguments (line 121)
    unicode_261717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 25), 'unicode', u'k')
    # Processing the call keyword arguments (line 121)
    kwargs_261718 = {}
    # Getting the type of 'key' (line 121)
    key_261715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'key', False)
    # Obtaining the member 'index' of a type (line 121)
    index_261716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), key_261715, 'index')
    # Calling index(args, kwargs) (line 121)
    index_call_result_261719 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), index_261716, *[unicode_261717], **kwargs_261718)
    
    slice_261720 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 121, 10), None, index_call_result_261719, None)
    # Getting the type of 'key' (line 121)
    key_261721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 10), 'key')
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___261722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 10), key_261721, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_261723 = invoke(stypy.reporting.localization.Localization(__file__, 121, 10), getitem___261722, slice_261720)
    
    # Getting the type of 'value' (line 121)
    value_261724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 33), 'value')
    # Applying the binary operator '+' (line 121)
    result_add_261725 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 10), '+', subscript_call_result_261723, value_261724)
    
    # Assigning a type to the variable 'key' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'key', result_add_261725)
    # Getting the type of 'key' (line 122)
    key_261726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'key')
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type', key_261726)
    
    # ################# End of '_handle_key(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_handle_key' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_261727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_261727)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_handle_key'
    return stypy_return_type_261727

# Assigning a type to the variable '_handle_key' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), '_handle_key', _handle_key)
# Declaration of the 'FigureCanvasWebAggCore' class
# Getting the type of 'backend_agg' (line 125)
backend_agg_261728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), 'backend_agg')
# Obtaining the member 'FigureCanvasAgg' of a type (line 125)
FigureCanvasAgg_261729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 29), backend_agg_261728, 'FigureCanvasAgg')

class FigureCanvasWebAggCore(FigureCanvasAgg_261729, ):
    
    # Assigning a Name to a Name (line 126):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'self' (line 129)
        self_261733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'self', False)
        # Getting the type of 'args' (line 129)
        args_261734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 52), 'args', False)
        # Processing the call keyword arguments (line 129)
        # Getting the type of 'kwargs' (line 129)
        kwargs_261735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 60), 'kwargs', False)
        kwargs_261736 = {'kwargs_261735': kwargs_261735}
        # Getting the type of 'backend_agg' (line 129)
        backend_agg_261730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'backend_agg', False)
        # Obtaining the member 'FigureCanvasAgg' of a type (line 129)
        FigureCanvasAgg_261731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), backend_agg_261730, 'FigureCanvasAgg')
        # Obtaining the member '__init__' of a type (line 129)
        init___261732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), FigureCanvasAgg_261731, '__init__')
        # Calling __init__(args, kwargs) (line 129)
        init___call_result_261737 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), init___261732, *[self_261733, args_261734], **kwargs_261736)
        
        
        # Assigning a Name to a Attribute (line 133):
        
        # Assigning a Name to a Attribute (line 133):
        # Getting the type of 'True' (line 133)
        True_261738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 27), 'True')
        # Getting the type of 'self' (line 133)
        self_261739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'self')
        # Setting the type of the member '_png_is_old' of a type (line 133)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), self_261739, '_png_is_old', True_261738)
        
        # Assigning a Name to a Attribute (line 137):
        
        # Assigning a Name to a Attribute (line 137):
        # Getting the type of 'True' (line 137)
        True_261740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'True')
        # Getting the type of 'self' (line 137)
        self_261741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self')
        # Setting the type of the member '_force_full' of a type (line 137)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_261741, '_force_full', True_261740)
        
        # Assigning a Str to a Attribute (line 143):
        
        # Assigning a Str to a Attribute (line 143):
        unicode_261742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 35), 'unicode', u'full')
        # Getting the type of 'self' (line 143)
        self_261743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
        # Setting the type of the member '_current_image_mode' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_261743, '_current_image_mode', unicode_261742)
        
        # Assigning a Num to a Attribute (line 147):
        
        # Assigning a Num to a Attribute (line 147):
        int_261744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 26), 'int')
        # Getting the type of 'self' (line 147)
        self_261745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self')
        # Setting the type of the member '_dpi_ratio' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_261745, '_dpi_ratio', int_261744)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def show(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'show'
        module_type_store = module_type_store.open_function_context('show', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.show.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.show.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.show.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.show.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.show')
        FigureCanvasWebAggCore.show.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasWebAggCore.show.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.show.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.show.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.show.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.show.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.show.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.show', [], None, None, defaults, varargs, kwargs)

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

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 151, 8))
        
        # 'from matplotlib.pyplot import show' statement (line 151)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
        import_261746 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 151, 8), 'matplotlib.pyplot')

        if (type(import_261746) is not StypyTypeError):

            if (import_261746 != 'pyd_module'):
                __import__(import_261746)
                sys_modules_261747 = sys.modules[import_261746]
                import_from_module(stypy.reporting.localization.Localization(__file__, 151, 8), 'matplotlib.pyplot', sys_modules_261747.module_type_store, module_type_store, ['show'])
                nest_module(stypy.reporting.localization.Localization(__file__, 151, 8), __file__, sys_modules_261747, sys_modules_261747.module_type_store, module_type_store)
            else:
                from matplotlib.pyplot import show

                import_from_module(stypy.reporting.localization.Localization(__file__, 151, 8), 'matplotlib.pyplot', None, module_type_store, ['show'], [show])

        else:
            # Assigning a type to the variable 'matplotlib.pyplot' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'matplotlib.pyplot', import_261746)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')
        
        
        # Call to show(...): (line 152)
        # Processing the call keyword arguments (line 152)
        kwargs_261749 = {}
        # Getting the type of 'show' (line 152)
        show_261748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'show', False)
        # Calling show(args, kwargs) (line 152)
        show_call_result_261750 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), show_261748, *[], **kwargs_261749)
        
        
        # ################# End of 'show(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'show' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_261751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_261751)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'show'
        return stypy_return_type_261751


    @norecursion
    def draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw'
        module_type_store = module_type_store.open_function_context('draw', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.draw.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.draw.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.draw')
        FigureCanvasWebAggCore.draw.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasWebAggCore.draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.draw', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 155):
        
        # Assigning a Call to a Name (line 155):
        
        # Call to get_renderer(...): (line 155)
        # Processing the call keyword arguments (line 155)
        # Getting the type of 'True' (line 155)
        True_261754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 45), 'True', False)
        keyword_261755 = True_261754
        kwargs_261756 = {'cleared': keyword_261755}
        # Getting the type of 'self' (line 155)
        self_261752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 155)
        get_renderer_261753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 19), self_261752, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 155)
        get_renderer_call_result_261757 = invoke(stypy.reporting.localization.Localization(__file__, 155, 19), get_renderer_261753, *[], **kwargs_261756)
        
        # Assigning a type to the variable 'renderer' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'renderer', get_renderer_call_result_261757)
        
        # Assigning a Name to a Attribute (line 157):
        
        # Assigning a Name to a Attribute (line 157):
        # Getting the type of 'True' (line 157)
        True_261758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'True')
        # Getting the type of 'self' (line 157)
        self_261759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member '_png_is_old' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_261759, '_png_is_old', True_261758)
        
        # Call to acquire(...): (line 159)
        # Processing the call keyword arguments (line 159)
        kwargs_261764 = {}
        # Getting the type of 'backend_agg' (line 159)
        backend_agg_261760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'backend_agg', False)
        # Obtaining the member 'RendererAgg' of a type (line 159)
        RendererAgg_261761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), backend_agg_261760, 'RendererAgg')
        # Obtaining the member 'lock' of a type (line 159)
        lock_261762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), RendererAgg_261761, 'lock')
        # Obtaining the member 'acquire' of a type (line 159)
        acquire_261763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), lock_261762, 'acquire')
        # Calling acquire(args, kwargs) (line 159)
        acquire_call_result_261765 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), acquire_261763, *[], **kwargs_261764)
        
        
        # Try-finally block (line 160)
        
        # Call to draw(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'renderer' (line 161)
        renderer_261769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'renderer', False)
        # Processing the call keyword arguments (line 161)
        kwargs_261770 = {}
        # Getting the type of 'self' (line 161)
        self_261766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'self', False)
        # Obtaining the member 'figure' of a type (line 161)
        figure_261767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), self_261766, 'figure')
        # Obtaining the member 'draw' of a type (line 161)
        draw_261768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), figure_261767, 'draw')
        # Calling draw(args, kwargs) (line 161)
        draw_call_result_261771 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), draw_261768, *[renderer_261769], **kwargs_261770)
        
        
        # finally branch of the try-finally block (line 160)
        
        # Call to release(...): (line 163)
        # Processing the call keyword arguments (line 163)
        kwargs_261776 = {}
        # Getting the type of 'backend_agg' (line 163)
        backend_agg_261772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'backend_agg', False)
        # Obtaining the member 'RendererAgg' of a type (line 163)
        RendererAgg_261773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), backend_agg_261772, 'RendererAgg')
        # Obtaining the member 'lock' of a type (line 163)
        lock_261774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), RendererAgg_261773, 'lock')
        # Obtaining the member 'release' of a type (line 163)
        release_261775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), lock_261774, 'release')
        # Calling release(args, kwargs) (line 163)
        release_call_result_261777 = invoke(stypy.reporting.localization.Localization(__file__, 163, 12), release_261775, *[], **kwargs_261776)
        
        
        # Call to refresh_all(...): (line 165)
        # Processing the call keyword arguments (line 165)
        kwargs_261781 = {}
        # Getting the type of 'self' (line 165)
        self_261778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'self', False)
        # Obtaining the member 'manager' of a type (line 165)
        manager_261779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), self_261778, 'manager')
        # Obtaining the member 'refresh_all' of a type (line 165)
        refresh_all_261780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), manager_261779, 'refresh_all')
        # Calling refresh_all(args, kwargs) (line 165)
        refresh_all_call_result_261782 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), refresh_all_261780, *[], **kwargs_261781)
        
        
        
        # ################# End of 'draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_261783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_261783)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw'
        return stypy_return_type_261783


    @norecursion
    def draw_idle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_idle'
        module_type_store = module_type_store.open_function_context('draw_idle', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.draw_idle.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.draw_idle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.draw_idle.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.draw_idle.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.draw_idle')
        FigureCanvasWebAggCore.draw_idle.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasWebAggCore.draw_idle.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.draw_idle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.draw_idle.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.draw_idle.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.draw_idle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.draw_idle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.draw_idle', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to send_event(...): (line 168)
        # Processing the call arguments (line 168)
        unicode_261786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 24), 'unicode', u'draw')
        # Processing the call keyword arguments (line 168)
        kwargs_261787 = {}
        # Getting the type of 'self' (line 168)
        self_261784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self', False)
        # Obtaining the member 'send_event' of a type (line 168)
        send_event_261785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_261784, 'send_event')
        # Calling send_event(args, kwargs) (line 168)
        send_event_call_result_261788 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), send_event_261785, *[unicode_261786], **kwargs_261787)
        
        
        # ################# End of 'draw_idle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_idle' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_261789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_261789)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_idle'
        return stypy_return_type_261789


    @norecursion
    def set_image_mode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_image_mode'
        module_type_store = module_type_store.open_function_context('set_image_mode', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.set_image_mode.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.set_image_mode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.set_image_mode.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.set_image_mode.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.set_image_mode')
        FigureCanvasWebAggCore.set_image_mode.__dict__.__setitem__('stypy_param_names_list', ['mode'])
        FigureCanvasWebAggCore.set_image_mode.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.set_image_mode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.set_image_mode.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.set_image_mode.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.set_image_mode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.set_image_mode.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.set_image_mode', ['mode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_image_mode', localization, ['mode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_image_mode(...)' code ##################

        unicode_261790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, (-1)), 'unicode', u"\n        Set the image mode for any subsequent images which will be sent\n        to the clients. The modes may currently be either 'full' or 'diff'.\n\n        Note: diff images may not contain transparency, therefore upon\n        draw this mode may be changed if the resulting image has any\n        transparent component.\n\n        ")
        
        
        # Getting the type of 'mode' (line 180)
        mode_261791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'mode')
        
        # Obtaining an instance of the builtin type 'list' (line 180)
        list_261792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 180)
        # Adding element type (line 180)
        unicode_261793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 24), 'unicode', u'full')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 23), list_261792, unicode_261793)
        # Adding element type (line 180)
        unicode_261794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 32), 'unicode', u'diff')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 23), list_261792, unicode_261794)
        
        # Applying the binary operator 'notin' (line 180)
        result_contains_261795 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 11), 'notin', mode_261791, list_261792)
        
        # Testing the type of an if condition (line 180)
        if_condition_261796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 8), result_contains_261795)
        # Assigning a type to the variable 'if_condition_261796' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'if_condition_261796', if_condition_261796)
        # SSA begins for if statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 181)
        # Processing the call arguments (line 181)
        unicode_261798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 29), 'unicode', u'image mode must be either full or diff.')
        # Processing the call keyword arguments (line 181)
        kwargs_261799 = {}
        # Getting the type of 'ValueError' (line 181)
        ValueError_261797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 181)
        ValueError_call_result_261800 = invoke(stypy.reporting.localization.Localization(__file__, 181, 18), ValueError_261797, *[unicode_261798], **kwargs_261799)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 181, 12), ValueError_call_result_261800, 'raise parameter', BaseException)
        # SSA join for if statement (line 180)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 182)
        self_261801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'self')
        # Obtaining the member '_current_image_mode' of a type (line 182)
        _current_image_mode_261802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 11), self_261801, '_current_image_mode')
        # Getting the type of 'mode' (line 182)
        mode_261803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 39), 'mode')
        # Applying the binary operator '!=' (line 182)
        result_ne_261804 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 11), '!=', _current_image_mode_261802, mode_261803)
        
        # Testing the type of an if condition (line 182)
        if_condition_261805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 8), result_ne_261804)
        # Assigning a type to the variable 'if_condition_261805' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'if_condition_261805', if_condition_261805)
        # SSA begins for if statement (line 182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 183):
        
        # Assigning a Name to a Attribute (line 183):
        # Getting the type of 'mode' (line 183)
        mode_261806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 39), 'mode')
        # Getting the type of 'self' (line 183)
        self_261807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'self')
        # Setting the type of the member '_current_image_mode' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), self_261807, '_current_image_mode', mode_261806)
        
        # Call to handle_send_image_mode(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'None' (line 184)
        None_261810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 40), 'None', False)
        # Processing the call keyword arguments (line 184)
        kwargs_261811 = {}
        # Getting the type of 'self' (line 184)
        self_261808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'self', False)
        # Obtaining the member 'handle_send_image_mode' of a type (line 184)
        handle_send_image_mode_261809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 12), self_261808, 'handle_send_image_mode')
        # Calling handle_send_image_mode(args, kwargs) (line 184)
        handle_send_image_mode_call_result_261812 = invoke(stypy.reporting.localization.Localization(__file__, 184, 12), handle_send_image_mode_261809, *[None_261810], **kwargs_261811)
        
        # SSA join for if statement (line 182)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_image_mode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_image_mode' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_261813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_261813)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_image_mode'
        return stypy_return_type_261813


    @norecursion
    def get_diff_image(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_diff_image'
        module_type_store = module_type_store.open_function_context('get_diff_image', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.get_diff_image.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.get_diff_image.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.get_diff_image.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.get_diff_image.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.get_diff_image')
        FigureCanvasWebAggCore.get_diff_image.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasWebAggCore.get_diff_image.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.get_diff_image.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.get_diff_image.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.get_diff_image.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.get_diff_image.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.get_diff_image.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.get_diff_image', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_diff_image', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_diff_image(...)' code ##################

        
        # Getting the type of 'self' (line 187)
        self_261814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'self')
        # Obtaining the member '_png_is_old' of a type (line 187)
        _png_is_old_261815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 11), self_261814, '_png_is_old')
        # Testing the type of an if condition (line 187)
        if_condition_261816 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 8), _png_is_old_261815)
        # Assigning a type to the variable 'if_condition_261816' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'if_condition_261816', if_condition_261816)
        # SSA begins for if statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 188):
        
        # Assigning a Call to a Name (line 188):
        
        # Call to get_renderer(...): (line 188)
        # Processing the call keyword arguments (line 188)
        kwargs_261819 = {}
        # Getting the type of 'self' (line 188)
        self_261817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 188)
        get_renderer_261818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 23), self_261817, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 188)
        get_renderer_call_result_261820 = invoke(stypy.reporting.localization.Localization(__file__, 188, 23), get_renderer_261818, *[], **kwargs_261819)
        
        # Assigning a type to the variable 'renderer' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'renderer', get_renderer_call_result_261820)
        
        # Assigning a Call to a Name (line 193):
        
        # Assigning a Call to a Name (line 193):
        
        # Call to reshape(...): (line 193)
        # Processing the call arguments (line 193)
        
        # Obtaining an instance of the builtin type 'tuple' (line 194)
        tuple_261833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 194)
        # Adding element type (line 194)
        # Getting the type of 'renderer' (line 194)
        renderer_261834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 30), 'renderer', False)
        # Obtaining the member 'height' of a type (line 194)
        height_261835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 30), renderer_261834, 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 30), tuple_261833, height_261835)
        # Adding element type (line 194)
        # Getting the type of 'renderer' (line 194)
        renderer_261836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 47), 'renderer', False)
        # Obtaining the member 'width' of a type (line 194)
        width_261837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 47), renderer_261836, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 30), tuple_261833, width_261837)
        
        # Processing the call keyword arguments (line 193)
        kwargs_261838 = {}
        
        # Call to frombuffer(...): (line 193)
        # Processing the call arguments (line 193)
        
        # Call to buffer_rgba(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_261825 = {}
        # Getting the type of 'renderer' (line 193)
        renderer_261823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 34), 'renderer', False)
        # Obtaining the member 'buffer_rgba' of a type (line 193)
        buffer_rgba_261824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 34), renderer_261823, 'buffer_rgba')
        # Calling buffer_rgba(args, kwargs) (line 193)
        buffer_rgba_call_result_261826 = invoke(stypy.reporting.localization.Localization(__file__, 193, 34), buffer_rgba_261824, *[], **kwargs_261825)
        
        # Processing the call keyword arguments (line 193)
        # Getting the type of 'np' (line 193)
        np_261827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 64), 'np', False)
        # Obtaining the member 'uint32' of a type (line 193)
        uint32_261828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 64), np_261827, 'uint32')
        keyword_261829 = uint32_261828
        kwargs_261830 = {'dtype': keyword_261829}
        # Getting the type of 'np' (line 193)
        np_261821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 20), 'np', False)
        # Obtaining the member 'frombuffer' of a type (line 193)
        frombuffer_261822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 20), np_261821, 'frombuffer')
        # Calling frombuffer(args, kwargs) (line 193)
        frombuffer_call_result_261831 = invoke(stypy.reporting.localization.Localization(__file__, 193, 20), frombuffer_261822, *[buffer_rgba_call_result_261826], **kwargs_261830)
        
        # Obtaining the member 'reshape' of a type (line 193)
        reshape_261832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 20), frombuffer_call_result_261831, 'reshape')
        # Calling reshape(args, kwargs) (line 193)
        reshape_call_result_261839 = invoke(stypy.reporting.localization.Localization(__file__, 193, 20), reshape_261832, *[tuple_261833], **kwargs_261838)
        
        # Assigning a type to the variable 'buff' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'buff', reshape_call_result_261839)
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to reshape(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'buff' (line 198)
        buff_261848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 55), 'buff', False)
        # Obtaining the member 'shape' of a type (line 198)
        shape_261849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 55), buff_261848, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 198)
        tuple_261850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 69), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 198)
        # Adding element type (line 198)
        int_261851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 69), tuple_261850, int_261851)
        
        # Applying the binary operator '+' (line 198)
        result_add_261852 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 55), '+', shape_261849, tuple_261850)
        
        # Processing the call keyword arguments (line 198)
        kwargs_261853 = {}
        
        # Call to view(...): (line 198)
        # Processing the call keyword arguments (line 198)
        # Getting the type of 'np' (line 198)
        np_261842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 37), 'np', False)
        # Obtaining the member 'uint8' of a type (line 198)
        uint8_261843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 37), np_261842, 'uint8')
        keyword_261844 = uint8_261843
        kwargs_261845 = {'dtype': keyword_261844}
        # Getting the type of 'buff' (line 198)
        buff_261840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'buff', False)
        # Obtaining the member 'view' of a type (line 198)
        view_261841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 21), buff_261840, 'view')
        # Calling view(args, kwargs) (line 198)
        view_call_result_261846 = invoke(stypy.reporting.localization.Localization(__file__, 198, 21), view_261841, *[], **kwargs_261845)
        
        # Obtaining the member 'reshape' of a type (line 198)
        reshape_261847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 21), view_call_result_261846, 'reshape')
        # Calling reshape(args, kwargs) (line 198)
        reshape_call_result_261854 = invoke(stypy.reporting.localization.Localization(__file__, 198, 21), reshape_261847, *[result_add_261852], **kwargs_261853)
        
        # Assigning a type to the variable 'pixels' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'pixels', reshape_call_result_261854)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 200)
        self_261855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'self')
        # Obtaining the member '_force_full' of a type (line 200)
        _force_full_261856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), self_261855, '_force_full')
        
        # Call to any(...): (line 200)
        # Processing the call arguments (line 200)
        
        
        # Obtaining the type of the subscript
        slice_261859 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 42), None, None, None)
        slice_261860 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 42), None, None, None)
        int_261861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 55), 'int')
        # Getting the type of 'pixels' (line 200)
        pixels_261862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 42), 'pixels', False)
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___261863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 42), pixels_261862, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_261864 = invoke(stypy.reporting.localization.Localization(__file__, 200, 42), getitem___261863, (slice_261859, slice_261860, int_261861))
        
        int_261865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 61), 'int')
        # Applying the binary operator '!=' (line 200)
        result_ne_261866 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 42), '!=', subscript_call_result_261864, int_261865)
        
        # Processing the call keyword arguments (line 200)
        kwargs_261867 = {}
        # Getting the type of 'np' (line 200)
        np_261857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 35), 'np', False)
        # Obtaining the member 'any' of a type (line 200)
        any_261858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 35), np_261857, 'any')
        # Calling any(args, kwargs) (line 200)
        any_call_result_261868 = invoke(stypy.reporting.localization.Localization(__file__, 200, 35), any_261858, *[result_ne_261866], **kwargs_261867)
        
        # Applying the binary operator 'or' (line 200)
        result_or_keyword_261869 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 15), 'or', _force_full_261856, any_call_result_261868)
        
        # Testing the type of an if condition (line 200)
        if_condition_261870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 12), result_or_keyword_261869)
        # Assigning a type to the variable 'if_condition_261870' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'if_condition_261870', if_condition_261870)
        # SSA begins for if statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_image_mode(...): (line 201)
        # Processing the call arguments (line 201)
        unicode_261873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 36), 'unicode', u'full')
        # Processing the call keyword arguments (line 201)
        kwargs_261874 = {}
        # Getting the type of 'self' (line 201)
        self_261871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'self', False)
        # Obtaining the member 'set_image_mode' of a type (line 201)
        set_image_mode_261872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), self_261871, 'set_image_mode')
        # Calling set_image_mode(args, kwargs) (line 201)
        set_image_mode_call_result_261875 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), set_image_mode_261872, *[unicode_261873], **kwargs_261874)
        
        
        # Assigning a Name to a Name (line 202):
        
        # Assigning a Name to a Name (line 202):
        # Getting the type of 'buff' (line 202)
        buff_261876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 25), 'buff')
        # Assigning a type to the variable 'output' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'output', buff_261876)
        # SSA branch for the else part of an if statement (line 200)
        module_type_store.open_ssa_branch('else')
        
        # Call to set_image_mode(...): (line 204)
        # Processing the call arguments (line 204)
        unicode_261879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 36), 'unicode', u'diff')
        # Processing the call keyword arguments (line 204)
        kwargs_261880 = {}
        # Getting the type of 'self' (line 204)
        self_261877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'self', False)
        # Obtaining the member 'set_image_mode' of a type (line 204)
        set_image_mode_261878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 16), self_261877, 'set_image_mode')
        # Calling set_image_mode(args, kwargs) (line 204)
        set_image_mode_call_result_261881 = invoke(stypy.reporting.localization.Localization(__file__, 204, 16), set_image_mode_261878, *[unicode_261879], **kwargs_261880)
        
        
        # Assigning a Call to a Name (line 205):
        
        # Assigning a Call to a Name (line 205):
        
        # Call to reshape(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Obtaining an instance of the builtin type 'tuple' (line 207)
        tuple_261895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 207)
        # Adding element type (line 207)
        # Getting the type of 'renderer' (line 207)
        renderer_261896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 41), 'renderer', False)
        # Obtaining the member 'height' of a type (line 207)
        height_261897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 41), renderer_261896, 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 41), tuple_261895, height_261897)
        # Adding element type (line 207)
        # Getting the type of 'renderer' (line 207)
        renderer_261898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 58), 'renderer', False)
        # Obtaining the member 'width' of a type (line 207)
        width_261899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 58), renderer_261898, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 41), tuple_261895, width_261899)
        
        # Processing the call keyword arguments (line 205)
        kwargs_261900 = {}
        
        # Call to frombuffer(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Call to buffer_rgba(...): (line 205)
        # Processing the call keyword arguments (line 205)
        kwargs_261887 = {}
        # Getting the type of 'self' (line 205)
        self_261884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 45), 'self', False)
        # Obtaining the member '_last_renderer' of a type (line 205)
        _last_renderer_261885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 45), self_261884, '_last_renderer')
        # Obtaining the member 'buffer_rgba' of a type (line 205)
        buffer_rgba_261886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 45), _last_renderer_261885, 'buffer_rgba')
        # Calling buffer_rgba(args, kwargs) (line 205)
        buffer_rgba_call_result_261888 = invoke(stypy.reporting.localization.Localization(__file__, 205, 45), buffer_rgba_261886, *[], **kwargs_261887)
        
        # Processing the call keyword arguments (line 205)
        # Getting the type of 'np' (line 206)
        np_261889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 51), 'np', False)
        # Obtaining the member 'uint32' of a type (line 206)
        uint32_261890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 51), np_261889, 'uint32')
        keyword_261891 = uint32_261890
        kwargs_261892 = {'dtype': keyword_261891}
        # Getting the type of 'np' (line 205)
        np_261882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 31), 'np', False)
        # Obtaining the member 'frombuffer' of a type (line 205)
        frombuffer_261883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 31), np_261882, 'frombuffer')
        # Calling frombuffer(args, kwargs) (line 205)
        frombuffer_call_result_261893 = invoke(stypy.reporting.localization.Localization(__file__, 205, 31), frombuffer_261883, *[buffer_rgba_call_result_261888], **kwargs_261892)
        
        # Obtaining the member 'reshape' of a type (line 205)
        reshape_261894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 31), frombuffer_call_result_261893, 'reshape')
        # Calling reshape(args, kwargs) (line 205)
        reshape_call_result_261901 = invoke(stypy.reporting.localization.Localization(__file__, 205, 31), reshape_261894, *[tuple_261895], **kwargs_261900)
        
        # Assigning a type to the variable 'last_buffer' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'last_buffer', reshape_call_result_261901)
        
        # Assigning a Compare to a Name (line 208):
        
        # Assigning a Compare to a Name (line 208):
        
        # Getting the type of 'buff' (line 208)
        buff_261902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'buff')
        # Getting the type of 'last_buffer' (line 208)
        last_buffer_261903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 31), 'last_buffer')
        # Applying the binary operator '!=' (line 208)
        result_ne_261904 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 23), '!=', buff_261902, last_buffer_261903)
        
        # Assigning a type to the variable 'diff' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'diff', result_ne_261904)
        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to where(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'diff' (line 209)
        diff_261907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 34), 'diff', False)
        # Getting the type of 'buff' (line 209)
        buff_261908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 40), 'buff', False)
        int_261909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 46), 'int')
        # Processing the call keyword arguments (line 209)
        kwargs_261910 = {}
        # Getting the type of 'np' (line 209)
        np_261905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 25), 'np', False)
        # Obtaining the member 'where' of a type (line 209)
        where_261906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 25), np_261905, 'where')
        # Calling where(args, kwargs) (line 209)
        where_call_result_261911 = invoke(stypy.reporting.localization.Localization(__file__, 209, 25), where_261906, *[diff_261907, buff_261908, int_261909], **kwargs_261910)
        
        # Assigning a type to the variable 'output' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'output', where_call_result_261911)
        # SSA join for if statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 213):
        
        # Assigning a Call to a Name (line 213):
        
        # Call to write_png(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to reshape(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'output' (line 214)
        output_261922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 52), 'output', False)
        # Obtaining the member 'shape' of a type (line 214)
        shape_261923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 52), output_261922, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 214)
        tuple_261924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 68), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 214)
        # Adding element type (line 214)
        int_261925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 68), tuple_261924, int_261925)
        
        # Applying the binary operator '+' (line 214)
        result_add_261926 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 52), '+', shape_261923, tuple_261924)
        
        # Processing the call keyword arguments (line 214)
        kwargs_261927 = {}
        
        # Call to view(...): (line 214)
        # Processing the call keyword arguments (line 214)
        # Getting the type of 'np' (line 214)
        np_261916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 34), 'np', False)
        # Obtaining the member 'uint8' of a type (line 214)
        uint8_261917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 34), np_261916, 'uint8')
        keyword_261918 = uint8_261917
        kwargs_261919 = {'dtype': keyword_261918}
        # Getting the type of 'output' (line 214)
        output_261914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'output', False)
        # Obtaining the member 'view' of a type (line 214)
        view_261915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 16), output_261914, 'view')
        # Calling view(args, kwargs) (line 214)
        view_call_result_261920 = invoke(stypy.reporting.localization.Localization(__file__, 214, 16), view_261915, *[], **kwargs_261919)
        
        # Obtaining the member 'reshape' of a type (line 214)
        reshape_261921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 16), view_call_result_261920, 'reshape')
        # Calling reshape(args, kwargs) (line 214)
        reshape_call_result_261928 = invoke(stypy.reporting.localization.Localization(__file__, 214, 16), reshape_261921, *[result_add_261926], **kwargs_261927)
        
        # Getting the type of 'None' (line 215)
        None_261929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'None', False)
        # Processing the call keyword arguments (line 213)
        int_261930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 34), 'int')
        keyword_261931 = int_261930
        # Getting the type of '_png' (line 215)
        _png_261932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 44), '_png', False)
        # Obtaining the member 'PNG_FILTER_NONE' of a type (line 215)
        PNG_FILTER_NONE_261933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 44), _png_261932, 'PNG_FILTER_NONE')
        keyword_261934 = PNG_FILTER_NONE_261933
        kwargs_261935 = {'filter': keyword_261934, 'compression': keyword_261931}
        # Getting the type of '_png' (line 213)
        _png_261912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), '_png', False)
        # Obtaining the member 'write_png' of a type (line 213)
        write_png_261913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 19), _png_261912, 'write_png')
        # Calling write_png(args, kwargs) (line 213)
        write_png_call_result_261936 = invoke(stypy.reporting.localization.Localization(__file__, 213, 19), write_png_261913, *[reshape_call_result_261928, None_261929], **kwargs_261935)
        
        # Assigning a type to the variable 'buff' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'buff', write_png_call_result_261936)
        
        # Assigning a Tuple to a Tuple (line 218):
        
        # Assigning a Attribute to a Name (line 218):
        # Getting the type of 'self' (line 219)
        self_261937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'self')
        # Obtaining the member '_last_renderer' of a type (line 219)
        _last_renderer_261938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 16), self_261937, '_last_renderer')
        # Assigning a type to the variable 'tuple_assignment_261448' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'tuple_assignment_261448', _last_renderer_261938)
        
        # Assigning a Name to a Name (line 218):
        # Getting the type of 'renderer' (line 219)
        renderer_261939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 37), 'renderer')
        # Assigning a type to the variable 'tuple_assignment_261449' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'tuple_assignment_261449', renderer_261939)
        
        # Assigning a Name to a Attribute (line 218):
        # Getting the type of 'tuple_assignment_261448' (line 218)
        tuple_assignment_261448_261940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'tuple_assignment_261448')
        # Getting the type of 'self' (line 218)
        self_261941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'self')
        # Setting the type of the member '_renderer' of a type (line 218)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), self_261941, '_renderer', tuple_assignment_261448_261940)
        
        # Assigning a Name to a Attribute (line 218):
        # Getting the type of 'tuple_assignment_261449' (line 218)
        tuple_assignment_261449_261942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'tuple_assignment_261449')
        # Getting the type of 'self' (line 218)
        self_261943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 28), 'self')
        # Setting the type of the member '_last_renderer' of a type (line 218)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 28), self_261943, '_last_renderer', tuple_assignment_261449_261942)
        
        # Assigning a Name to a Attribute (line 220):
        
        # Assigning a Name to a Attribute (line 220):
        # Getting the type of 'False' (line 220)
        False_261944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 31), 'False')
        # Getting the type of 'self' (line 220)
        self_261945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'self')
        # Setting the type of the member '_force_full' of a type (line 220)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), self_261945, '_force_full', False_261944)
        
        # Assigning a Name to a Attribute (line 221):
        
        # Assigning a Name to a Attribute (line 221):
        # Getting the type of 'False' (line 221)
        False_261946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 31), 'False')
        # Getting the type of 'self' (line 221)
        self_261947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'self')
        # Setting the type of the member '_png_is_old' of a type (line 221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), self_261947, '_png_is_old', False_261946)
        # Getting the type of 'buff' (line 222)
        buff_261948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 19), 'buff')
        # Assigning a type to the variable 'stypy_return_type' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'stypy_return_type', buff_261948)
        # SSA join for if statement (line 187)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_diff_image(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_diff_image' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_261949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_261949)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_diff_image'
        return stypy_return_type_261949


    @norecursion
    def get_renderer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 224)
        None_261950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 35), 'None')
        defaults = [None_261950]
        # Create a new context for function 'get_renderer'
        module_type_store = module_type_store.open_function_context('get_renderer', 224, 4, False)
        # Assigning a type to the variable 'self' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.get_renderer.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.get_renderer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.get_renderer.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.get_renderer.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.get_renderer')
        FigureCanvasWebAggCore.get_renderer.__dict__.__setitem__('stypy_param_names_list', ['cleared'])
        FigureCanvasWebAggCore.get_renderer.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.get_renderer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.get_renderer.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.get_renderer.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.get_renderer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.get_renderer.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.get_renderer', ['cleared'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_renderer', localization, ['cleared'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_renderer(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 228):
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_261951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 8), 'int')
        # Getting the type of 'self' (line 228)
        self_261952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'self')
        # Obtaining the member 'figure' of a type (line 228)
        figure_261953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), self_261952, 'figure')
        # Obtaining the member 'bbox' of a type (line 228)
        bbox_261954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), figure_261953, 'bbox')
        # Obtaining the member 'bounds' of a type (line 228)
        bounds_261955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), bbox_261954, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___261956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), bounds_261955, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_261957 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), getitem___261956, int_261951)
        
        # Assigning a type to the variable 'tuple_var_assignment_261450' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_261450', subscript_call_result_261957)
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_261958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 8), 'int')
        # Getting the type of 'self' (line 228)
        self_261959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'self')
        # Obtaining the member 'figure' of a type (line 228)
        figure_261960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), self_261959, 'figure')
        # Obtaining the member 'bbox' of a type (line 228)
        bbox_261961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), figure_261960, 'bbox')
        # Obtaining the member 'bounds' of a type (line 228)
        bounds_261962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), bbox_261961, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___261963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), bounds_261962, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_261964 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), getitem___261963, int_261958)
        
        # Assigning a type to the variable 'tuple_var_assignment_261451' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_261451', subscript_call_result_261964)
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_261965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 8), 'int')
        # Getting the type of 'self' (line 228)
        self_261966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'self')
        # Obtaining the member 'figure' of a type (line 228)
        figure_261967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), self_261966, 'figure')
        # Obtaining the member 'bbox' of a type (line 228)
        bbox_261968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), figure_261967, 'bbox')
        # Obtaining the member 'bounds' of a type (line 228)
        bounds_261969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), bbox_261968, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___261970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), bounds_261969, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_261971 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), getitem___261970, int_261965)
        
        # Assigning a type to the variable 'tuple_var_assignment_261452' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_261452', subscript_call_result_261971)
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_261972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 8), 'int')
        # Getting the type of 'self' (line 228)
        self_261973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'self')
        # Obtaining the member 'figure' of a type (line 228)
        figure_261974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), self_261973, 'figure')
        # Obtaining the member 'bbox' of a type (line 228)
        bbox_261975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), figure_261974, 'bbox')
        # Obtaining the member 'bounds' of a type (line 228)
        bounds_261976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), bbox_261975, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___261977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), bounds_261976, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_261978 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), getitem___261977, int_261972)
        
        # Assigning a type to the variable 'tuple_var_assignment_261453' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_261453', subscript_call_result_261978)
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'tuple_var_assignment_261450' (line 228)
        tuple_var_assignment_261450_261979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_261450')
        # Assigning a type to the variable '_' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), '_', tuple_var_assignment_261450_261979)
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'tuple_var_assignment_261451' (line 228)
        tuple_var_assignment_261451_261980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_261451')
        # Assigning a type to the variable '_' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), '_', tuple_var_assignment_261451_261980)
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'tuple_var_assignment_261452' (line 228)
        tuple_var_assignment_261452_261981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_261452')
        # Assigning a type to the variable 'w' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 14), 'w', tuple_var_assignment_261452_261981)
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'tuple_var_assignment_261453' (line 228)
        tuple_var_assignment_261453_261982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'tuple_var_assignment_261453')
        # Assigning a type to the variable 'h' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'h', tuple_var_assignment_261453_261982)
        
        # Assigning a Tuple to a Tuple (line 229):
        
        # Assigning a Call to a Name (line 229):
        
        # Call to int(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'w' (line 229)
        w_261984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 19), 'w', False)
        # Processing the call keyword arguments (line 229)
        kwargs_261985 = {}
        # Getting the type of 'int' (line 229)
        int_261983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'int', False)
        # Calling int(args, kwargs) (line 229)
        int_call_result_261986 = invoke(stypy.reporting.localization.Localization(__file__, 229, 15), int_261983, *[w_261984], **kwargs_261985)
        
        # Assigning a type to the variable 'tuple_assignment_261454' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_assignment_261454', int_call_result_261986)
        
        # Assigning a Call to a Name (line 229):
        
        # Call to int(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'h' (line 229)
        h_261988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'h', False)
        # Processing the call keyword arguments (line 229)
        kwargs_261989 = {}
        # Getting the type of 'int' (line 229)
        int_261987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 23), 'int', False)
        # Calling int(args, kwargs) (line 229)
        int_call_result_261990 = invoke(stypy.reporting.localization.Localization(__file__, 229, 23), int_261987, *[h_261988], **kwargs_261989)
        
        # Assigning a type to the variable 'tuple_assignment_261455' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_assignment_261455', int_call_result_261990)
        
        # Assigning a Name to a Name (line 229):
        # Getting the type of 'tuple_assignment_261454' (line 229)
        tuple_assignment_261454_261991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_assignment_261454')
        # Assigning a type to the variable 'w' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'w', tuple_assignment_261454_261991)
        
        # Assigning a Name to a Name (line 229):
        # Getting the type of 'tuple_assignment_261455' (line 229)
        tuple_assignment_261455_261992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_assignment_261455')
        # Assigning a type to the variable 'h' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), 'h', tuple_assignment_261455_261992)
        
        # Assigning a Tuple to a Name (line 230):
        
        # Assigning a Tuple to a Name (line 230):
        
        # Obtaining an instance of the builtin type 'tuple' (line 230)
        tuple_261993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 230)
        # Adding element type (line 230)
        # Getting the type of 'w' (line 230)
        w_261994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 14), 'w')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 14), tuple_261993, w_261994)
        # Adding element type (line 230)
        # Getting the type of 'h' (line 230)
        h_261995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 17), 'h')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 14), tuple_261993, h_261995)
        # Adding element type (line 230)
        # Getting the type of 'self' (line 230)
        self_261996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'self')
        # Obtaining the member 'figure' of a type (line 230)
        figure_261997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 20), self_261996, 'figure')
        # Obtaining the member 'dpi' of a type (line 230)
        dpi_261998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 20), figure_261997, 'dpi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 14), tuple_261993, dpi_261998)
        
        # Assigning a type to the variable 'key' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'key', tuple_261993)
        
        
        # SSA begins for try-except statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining an instance of the builtin type 'tuple' (line 232)
        tuple_261999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 232)
        # Adding element type (line 232)
        # Getting the type of 'self' (line 232)
        self_262000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'self')
        # Obtaining the member '_lastKey' of a type (line 232)
        _lastKey_262001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), self_262000, '_lastKey')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 12), tuple_261999, _lastKey_262001)
        # Adding element type (line 232)
        # Getting the type of 'self' (line 232)
        self_262002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'self')
        # Obtaining the member '_renderer' of a type (line 232)
        _renderer_262003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 27), self_262002, '_renderer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 12), tuple_261999, _renderer_262003)
        
        # SSA branch for the except part of a try statement (line 231)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 231)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Name to a Name (line 234):
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'True' (line 234)
        True_262004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 32), 'True')
        # Assigning a type to the variable 'need_new_renderer' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'need_new_renderer', True_262004)
        # SSA branch for the else branch of a try statement (line 231)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a Compare to a Name (line 236):
        
        # Assigning a Compare to a Name (line 236):
        
        # Getting the type of 'self' (line 236)
        self_262005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 33), 'self')
        # Obtaining the member '_lastKey' of a type (line 236)
        _lastKey_262006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 33), self_262005, '_lastKey')
        # Getting the type of 'key' (line 236)
        key_262007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 50), 'key')
        # Applying the binary operator '!=' (line 236)
        result_ne_262008 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 33), '!=', _lastKey_262006, key_262007)
        
        # Assigning a type to the variable 'need_new_renderer' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'need_new_renderer', result_ne_262008)
        # SSA join for try-except statement (line 231)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'need_new_renderer' (line 238)
        need_new_renderer_262009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'need_new_renderer')
        # Testing the type of an if condition (line 238)
        if_condition_262010 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), need_new_renderer_262009)
        # Assigning a type to the variable 'if_condition_262010' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_262010', if_condition_262010)
        # SSA begins for if statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 239):
        
        # Assigning a Call to a Attribute (line 239):
        
        # Call to RendererAgg(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'w' (line 240)
        w_262013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'w', False)
        # Getting the type of 'h' (line 240)
        h_262014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'h', False)
        # Getting the type of 'self' (line 240)
        self_262015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 22), 'self', False)
        # Obtaining the member 'figure' of a type (line 240)
        figure_262016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 22), self_262015, 'figure')
        # Obtaining the member 'dpi' of a type (line 240)
        dpi_262017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 22), figure_262016, 'dpi')
        # Processing the call keyword arguments (line 239)
        kwargs_262018 = {}
        # Getting the type of 'backend_agg' (line 239)
        backend_agg_262011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 29), 'backend_agg', False)
        # Obtaining the member 'RendererAgg' of a type (line 239)
        RendererAgg_262012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 29), backend_agg_262011, 'RendererAgg')
        # Calling RendererAgg(args, kwargs) (line 239)
        RendererAgg_call_result_262019 = invoke(stypy.reporting.localization.Localization(__file__, 239, 29), RendererAgg_262012, *[w_262013, h_262014, dpi_262017], **kwargs_262018)
        
        # Getting the type of 'self' (line 239)
        self_262020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'self')
        # Setting the type of the member '_renderer' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), self_262020, '_renderer', RendererAgg_call_result_262019)
        
        # Assigning a Call to a Attribute (line 241):
        
        # Assigning a Call to a Attribute (line 241):
        
        # Call to RendererAgg(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'w' (line 242)
        w_262023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'w', False)
        # Getting the type of 'h' (line 242)
        h_262024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'h', False)
        # Getting the type of 'self' (line 242)
        self_262025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 22), 'self', False)
        # Obtaining the member 'figure' of a type (line 242)
        figure_262026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 22), self_262025, 'figure')
        # Obtaining the member 'dpi' of a type (line 242)
        dpi_262027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 22), figure_262026, 'dpi')
        # Processing the call keyword arguments (line 241)
        kwargs_262028 = {}
        # Getting the type of 'backend_agg' (line 241)
        backend_agg_262021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 34), 'backend_agg', False)
        # Obtaining the member 'RendererAgg' of a type (line 241)
        RendererAgg_262022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 34), backend_agg_262021, 'RendererAgg')
        # Calling RendererAgg(args, kwargs) (line 241)
        RendererAgg_call_result_262029 = invoke(stypy.reporting.localization.Localization(__file__, 241, 34), RendererAgg_262022, *[w_262023, h_262024, dpi_262027], **kwargs_262028)
        
        # Getting the type of 'self' (line 241)
        self_262030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'self')
        # Setting the type of the member '_last_renderer' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), self_262030, '_last_renderer', RendererAgg_call_result_262029)
        
        # Assigning a Name to a Attribute (line 243):
        
        # Assigning a Name to a Attribute (line 243):
        # Getting the type of 'key' (line 243)
        key_262031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), 'key')
        # Getting the type of 'self' (line 243)
        self_262032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'self')
        # Setting the type of the member '_lastKey' of a type (line 243)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), self_262032, '_lastKey', key_262031)
        # SSA branch for the else part of an if statement (line 238)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'cleared' (line 245)
        cleared_262033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 13), 'cleared')
        # Testing the type of an if condition (line 245)
        if_condition_262034 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 13), cleared_262033)
        # Assigning a type to the variable 'if_condition_262034' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 13), 'if_condition_262034', if_condition_262034)
        # SSA begins for if statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to clear(...): (line 246)
        # Processing the call keyword arguments (line 246)
        kwargs_262038 = {}
        # Getting the type of 'self' (line 246)
        self_262035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'self', False)
        # Obtaining the member '_renderer' of a type (line 246)
        _renderer_262036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 12), self_262035, '_renderer')
        # Obtaining the member 'clear' of a type (line 246)
        clear_262037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 12), _renderer_262036, 'clear')
        # Calling clear(args, kwargs) (line 246)
        clear_call_result_262039 = invoke(stypy.reporting.localization.Localization(__file__, 246, 12), clear_262037, *[], **kwargs_262038)
        
        # SSA join for if statement (line 245)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 248)
        self_262040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'self')
        # Obtaining the member '_renderer' of a type (line 248)
        _renderer_262041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 15), self_262040, '_renderer')
        # Assigning a type to the variable 'stypy_return_type' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'stypy_return_type', _renderer_262041)
        
        # ################# End of 'get_renderer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_renderer' in the type store
        # Getting the type of 'stypy_return_type' (line 224)
        stypy_return_type_262042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262042)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_renderer'
        return stypy_return_type_262042


    @norecursion
    def handle_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_event'
        module_type_store = module_type_store.open_function_context('handle_event', 250, 4, False)
        # Assigning a type to the variable 'self' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.handle_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.handle_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.handle_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.handle_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.handle_event')
        FigureCanvasWebAggCore.handle_event.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasWebAggCore.handle_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.handle_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.handle_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.handle_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.handle_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.handle_event.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.handle_event', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_event', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_event(...)' code ##################

        
        # Assigning a Subscript to a Name (line 251):
        
        # Assigning a Subscript to a Name (line 251):
        
        # Obtaining the type of the subscript
        unicode_262043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 23), 'unicode', u'type')
        # Getting the type of 'event' (line 251)
        event_262044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 17), 'event')
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___262045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 17), event_262044, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_262046 = invoke(stypy.reporting.localization.Localization(__file__, 251, 17), getitem___262045, unicode_262043)
        
        # Assigning a type to the variable 'e_type' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'e_type', subscript_call_result_262046)
        
        # Assigning a Call to a Name (line 252):
        
        # Assigning a Call to a Name (line 252):
        
        # Call to getattr(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'self' (line 252)
        self_262048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 26), 'self', False)
        
        # Call to format(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'e_type' (line 252)
        e_type_262051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 52), 'e_type', False)
        # Processing the call keyword arguments (line 252)
        kwargs_262052 = {}
        unicode_262049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 32), 'unicode', u'handle_{0}')
        # Obtaining the member 'format' of a type (line 252)
        format_262050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 32), unicode_262049, 'format')
        # Calling format(args, kwargs) (line 252)
        format_call_result_262053 = invoke(stypy.reporting.localization.Localization(__file__, 252, 32), format_262050, *[e_type_262051], **kwargs_262052)
        
        # Getting the type of 'self' (line 253)
        self_262054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'self', False)
        # Obtaining the member 'handle_unknown_event' of a type (line 253)
        handle_unknown_event_262055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 26), self_262054, 'handle_unknown_event')
        # Processing the call keyword arguments (line 252)
        kwargs_262056 = {}
        # Getting the type of 'getattr' (line 252)
        getattr_262047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 252)
        getattr_call_result_262057 = invoke(stypy.reporting.localization.Localization(__file__, 252, 18), getattr_262047, *[self_262048, format_call_result_262053, handle_unknown_event_262055], **kwargs_262056)
        
        # Assigning a type to the variable 'handler' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'handler', getattr_call_result_262057)
        
        # Call to handler(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'event' (line 254)
        event_262059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 23), 'event', False)
        # Processing the call keyword arguments (line 254)
        kwargs_262060 = {}
        # Getting the type of 'handler' (line 254)
        handler_262058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'handler', False)
        # Calling handler(args, kwargs) (line 254)
        handler_call_result_262061 = invoke(stypy.reporting.localization.Localization(__file__, 254, 15), handler_262058, *[event_262059], **kwargs_262060)
        
        # Assigning a type to the variable 'stypy_return_type' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'stypy_return_type', handler_call_result_262061)
        
        # ################# End of 'handle_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_event' in the type store
        # Getting the type of 'stypy_return_type' (line 250)
        stypy_return_type_262062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262062)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_event'
        return stypy_return_type_262062


    @norecursion
    def handle_unknown_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_unknown_event'
        module_type_store = module_type_store.open_function_context('handle_unknown_event', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.handle_unknown_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.handle_unknown_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.handle_unknown_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.handle_unknown_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.handle_unknown_event')
        FigureCanvasWebAggCore.handle_unknown_event.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasWebAggCore.handle_unknown_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.handle_unknown_event.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.handle_unknown_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.handle_unknown_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.handle_unknown_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.handle_unknown_event.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.handle_unknown_event', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_unknown_event', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_unknown_event(...)' code ##################

        
        # Call to warn(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Call to format(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Obtaining the type of the subscript
        unicode_262067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 18), 'unicode', u'type')
        # Getting the type of 'event' (line 258)
        event_262068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'event', False)
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___262069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), event_262068, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_262070 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), getitem___262069, unicode_262067)
        
        # Getting the type of 'event' (line 258)
        event_262071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 27), 'event', False)
        # Processing the call keyword arguments (line 257)
        kwargs_262072 = {}
        unicode_262065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 22), 'unicode', u'Unhandled message type {0}. {1}')
        # Obtaining the member 'format' of a type (line 257)
        format_262066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 22), unicode_262065, 'format')
        # Calling format(args, kwargs) (line 257)
        format_call_result_262073 = invoke(stypy.reporting.localization.Localization(__file__, 257, 22), format_262066, *[subscript_call_result_262070, event_262071], **kwargs_262072)
        
        # Processing the call keyword arguments (line 257)
        kwargs_262074 = {}
        # Getting the type of 'warnings' (line 257)
        warnings_262063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 257)
        warn_262064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), warnings_262063, 'warn')
        # Calling warn(args, kwargs) (line 257)
        warn_call_result_262075 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), warn_262064, *[format_call_result_262073], **kwargs_262074)
        
        
        # ################# End of 'handle_unknown_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_unknown_event' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_262076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262076)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_unknown_event'
        return stypy_return_type_262076


    @norecursion
    def handle_ack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_ack'
        module_type_store = module_type_store.open_function_context('handle_ack', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.handle_ack.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.handle_ack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.handle_ack.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.handle_ack.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.handle_ack')
        FigureCanvasWebAggCore.handle_ack.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasWebAggCore.handle_ack.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.handle_ack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.handle_ack.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.handle_ack.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.handle_ack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.handle_ack.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.handle_ack', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_ack', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_ack(...)' code ##################

        pass
        
        # ################# End of 'handle_ack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_ack' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_262077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262077)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_ack'
        return stypy_return_type_262077


    @norecursion
    def handle_draw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_draw'
        module_type_store = module_type_store.open_function_context('handle_draw', 269, 4, False)
        # Assigning a type to the variable 'self' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.handle_draw.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.handle_draw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.handle_draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.handle_draw.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.handle_draw')
        FigureCanvasWebAggCore.handle_draw.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasWebAggCore.handle_draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.handle_draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.handle_draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.handle_draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.handle_draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.handle_draw.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.handle_draw', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_draw', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_draw(...)' code ##################

        
        # Call to draw(...): (line 270)
        # Processing the call keyword arguments (line 270)
        kwargs_262080 = {}
        # Getting the type of 'self' (line 270)
        self_262078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self', False)
        # Obtaining the member 'draw' of a type (line 270)
        draw_262079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_262078, 'draw')
        # Calling draw(args, kwargs) (line 270)
        draw_call_result_262081 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), draw_262079, *[], **kwargs_262080)
        
        
        # ################# End of 'handle_draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_draw' in the type store
        # Getting the type of 'stypy_return_type' (line 269)
        stypy_return_type_262082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262082)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_draw'
        return stypy_return_type_262082


    @norecursion
    def _handle_mouse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_handle_mouse'
        module_type_store = module_type_store.open_function_context('_handle_mouse', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore._handle_mouse.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore._handle_mouse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore._handle_mouse.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore._handle_mouse.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore._handle_mouse')
        FigureCanvasWebAggCore._handle_mouse.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasWebAggCore._handle_mouse.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore._handle_mouse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore._handle_mouse.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore._handle_mouse.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore._handle_mouse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore._handle_mouse.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore._handle_mouse', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_handle_mouse', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_handle_mouse(...)' code ##################

        
        # Assigning a Subscript to a Name (line 273):
        
        # Assigning a Subscript to a Name (line 273):
        
        # Obtaining the type of the subscript
        unicode_262083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 18), 'unicode', u'x')
        # Getting the type of 'event' (line 273)
        event_262084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'event')
        # Obtaining the member '__getitem__' of a type (line 273)
        getitem___262085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), event_262084, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 273)
        subscript_call_result_262086 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), getitem___262085, unicode_262083)
        
        # Assigning a type to the variable 'x' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'x', subscript_call_result_262086)
        
        # Assigning a Subscript to a Name (line 274):
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        unicode_262087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 18), 'unicode', u'y')
        # Getting the type of 'event' (line 274)
        event_262088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'event')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___262089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 12), event_262088, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_262090 = invoke(stypy.reporting.localization.Localization(__file__, 274, 12), getitem___262089, unicode_262087)
        
        # Assigning a type to the variable 'y' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'y', subscript_call_result_262090)
        
        # Assigning a BinOp to a Name (line 275):
        
        # Assigning a BinOp to a Name (line 275):
        
        # Call to get_renderer(...): (line 275)
        # Processing the call keyword arguments (line 275)
        kwargs_262093 = {}
        # Getting the type of 'self' (line 275)
        self_262091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'self', False)
        # Obtaining the member 'get_renderer' of a type (line 275)
        get_renderer_262092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), self_262091, 'get_renderer')
        # Calling get_renderer(args, kwargs) (line 275)
        get_renderer_call_result_262094 = invoke(stypy.reporting.localization.Localization(__file__, 275, 12), get_renderer_262092, *[], **kwargs_262093)
        
        # Obtaining the member 'height' of a type (line 275)
        height_262095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), get_renderer_call_result_262094, 'height')
        # Getting the type of 'y' (line 275)
        y_262096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 41), 'y')
        # Applying the binary operator '-' (line 275)
        result_sub_262097 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 12), '-', height_262095, y_262096)
        
        # Assigning a type to the variable 'y' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'y', result_sub_262097)
        
        # Assigning a BinOp to a Name (line 279):
        
        # Assigning a BinOp to a Name (line 279):
        
        # Obtaining the type of the subscript
        unicode_262098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 23), 'unicode', u'button')
        # Getting the type of 'event' (line 279)
        event_262099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 17), 'event')
        # Obtaining the member '__getitem__' of a type (line 279)
        getitem___262100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 17), event_262099, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 279)
        subscript_call_result_262101 = invoke(stypy.reporting.localization.Localization(__file__, 279, 17), getitem___262100, unicode_262098)
        
        int_262102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 35), 'int')
        # Applying the binary operator '+' (line 279)
        result_add_262103 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 17), '+', subscript_call_result_262101, int_262102)
        
        # Assigning a type to the variable 'button' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'button', result_add_262103)
        
        
        # Getting the type of 'button' (line 286)
        button_262104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 11), 'button')
        int_262105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 21), 'int')
        # Applying the binary operator '==' (line 286)
        result_eq_262106 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 11), '==', button_262104, int_262105)
        
        # Testing the type of an if condition (line 286)
        if_condition_262107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 8), result_eq_262106)
        # Assigning a type to the variable 'if_condition_262107' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'if_condition_262107', if_condition_262107)
        # SSA begins for if statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 287):
        
        # Assigning a Num to a Name (line 287):
        int_262108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 21), 'int')
        # Assigning a type to the variable 'button' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'button', int_262108)
        # SSA join for if statement (line 286)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 289):
        
        # Assigning a Subscript to a Name (line 289):
        
        # Obtaining the type of the subscript
        unicode_262109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 23), 'unicode', u'type')
        # Getting the type of 'event' (line 289)
        event_262110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 17), 'event')
        # Obtaining the member '__getitem__' of a type (line 289)
        getitem___262111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 17), event_262110, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 289)
        subscript_call_result_262112 = invoke(stypy.reporting.localization.Localization(__file__, 289, 17), getitem___262111, unicode_262109)
        
        # Assigning a type to the variable 'e_type' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'e_type', subscript_call_result_262112)
        
        # Assigning a Call to a Name (line 290):
        
        # Assigning a Call to a Name (line 290):
        
        # Call to get(...): (line 290)
        # Processing the call arguments (line 290)
        unicode_262115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 29), 'unicode', u'guiEvent')
        # Getting the type of 'None' (line 290)
        None_262116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 41), 'None', False)
        # Processing the call keyword arguments (line 290)
        kwargs_262117 = {}
        # Getting the type of 'event' (line 290)
        event_262113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 19), 'event', False)
        # Obtaining the member 'get' of a type (line 290)
        get_262114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 19), event_262113, 'get')
        # Calling get(args, kwargs) (line 290)
        get_call_result_262118 = invoke(stypy.reporting.localization.Localization(__file__, 290, 19), get_262114, *[unicode_262115, None_262116], **kwargs_262117)
        
        # Assigning a type to the variable 'guiEvent' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'guiEvent', get_call_result_262118)
        
        
        # Getting the type of 'e_type' (line 291)
        e_type_262119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 11), 'e_type')
        unicode_262120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 21), 'unicode', u'button_press')
        # Applying the binary operator '==' (line 291)
        result_eq_262121 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 11), '==', e_type_262119, unicode_262120)
        
        # Testing the type of an if condition (line 291)
        if_condition_262122 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 291, 8), result_eq_262121)
        # Assigning a type to the variable 'if_condition_262122' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'if_condition_262122', if_condition_262122)
        # SSA begins for if statement (line 291)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to button_press_event(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'x' (line 292)
        x_262125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 36), 'x', False)
        # Getting the type of 'y' (line 292)
        y_262126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 39), 'y', False)
        # Getting the type of 'button' (line 292)
        button_262127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 42), 'button', False)
        # Processing the call keyword arguments (line 292)
        # Getting the type of 'guiEvent' (line 292)
        guiEvent_262128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 59), 'guiEvent', False)
        keyword_262129 = guiEvent_262128
        kwargs_262130 = {'guiEvent': keyword_262129}
        # Getting the type of 'self' (line 292)
        self_262123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'self', False)
        # Obtaining the member 'button_press_event' of a type (line 292)
        button_press_event_262124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 12), self_262123, 'button_press_event')
        # Calling button_press_event(args, kwargs) (line 292)
        button_press_event_call_result_262131 = invoke(stypy.reporting.localization.Localization(__file__, 292, 12), button_press_event_262124, *[x_262125, y_262126, button_262127], **kwargs_262130)
        
        # SSA branch for the else part of an if statement (line 291)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'e_type' (line 293)
        e_type_262132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'e_type')
        unicode_262133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 23), 'unicode', u'button_release')
        # Applying the binary operator '==' (line 293)
        result_eq_262134 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 13), '==', e_type_262132, unicode_262133)
        
        # Testing the type of an if condition (line 293)
        if_condition_262135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 293, 13), result_eq_262134)
        # Assigning a type to the variable 'if_condition_262135' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'if_condition_262135', if_condition_262135)
        # SSA begins for if statement (line 293)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to button_release_event(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'x' (line 294)
        x_262138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 38), 'x', False)
        # Getting the type of 'y' (line 294)
        y_262139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 41), 'y', False)
        # Getting the type of 'button' (line 294)
        button_262140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 44), 'button', False)
        # Processing the call keyword arguments (line 294)
        # Getting the type of 'guiEvent' (line 294)
        guiEvent_262141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 61), 'guiEvent', False)
        keyword_262142 = guiEvent_262141
        kwargs_262143 = {'guiEvent': keyword_262142}
        # Getting the type of 'self' (line 294)
        self_262136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'self', False)
        # Obtaining the member 'button_release_event' of a type (line 294)
        button_release_event_262137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), self_262136, 'button_release_event')
        # Calling button_release_event(args, kwargs) (line 294)
        button_release_event_call_result_262144 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), button_release_event_262137, *[x_262138, y_262139, button_262140], **kwargs_262143)
        
        # SSA branch for the else part of an if statement (line 293)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'e_type' (line 295)
        e_type_262145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 13), 'e_type')
        unicode_262146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 23), 'unicode', u'motion_notify')
        # Applying the binary operator '==' (line 295)
        result_eq_262147 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 13), '==', e_type_262145, unicode_262146)
        
        # Testing the type of an if condition (line 295)
        if_condition_262148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 13), result_eq_262147)
        # Assigning a type to the variable 'if_condition_262148' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 13), 'if_condition_262148', if_condition_262148)
        # SSA begins for if statement (line 295)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to motion_notify_event(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'x' (line 296)
        x_262151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 37), 'x', False)
        # Getting the type of 'y' (line 296)
        y_262152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 40), 'y', False)
        # Processing the call keyword arguments (line 296)
        # Getting the type of 'guiEvent' (line 296)
        guiEvent_262153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 52), 'guiEvent', False)
        keyword_262154 = guiEvent_262153
        kwargs_262155 = {'guiEvent': keyword_262154}
        # Getting the type of 'self' (line 296)
        self_262149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'self', False)
        # Obtaining the member 'motion_notify_event' of a type (line 296)
        motion_notify_event_262150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), self_262149, 'motion_notify_event')
        # Calling motion_notify_event(args, kwargs) (line 296)
        motion_notify_event_call_result_262156 = invoke(stypy.reporting.localization.Localization(__file__, 296, 12), motion_notify_event_262150, *[x_262151, y_262152], **kwargs_262155)
        
        # SSA branch for the else part of an if statement (line 295)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'e_type' (line 297)
        e_type_262157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 13), 'e_type')
        unicode_262158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 23), 'unicode', u'figure_enter')
        # Applying the binary operator '==' (line 297)
        result_eq_262159 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 13), '==', e_type_262157, unicode_262158)
        
        # Testing the type of an if condition (line 297)
        if_condition_262160 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 13), result_eq_262159)
        # Assigning a type to the variable 'if_condition_262160' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 13), 'if_condition_262160', if_condition_262160)
        # SSA begins for if statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to enter_notify_event(...): (line 298)
        # Processing the call keyword arguments (line 298)
        
        # Obtaining an instance of the builtin type 'tuple' (line 298)
        tuple_262163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 298)
        # Adding element type (line 298)
        # Getting the type of 'x' (line 298)
        x_262164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 40), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 40), tuple_262163, x_262164)
        # Adding element type (line 298)
        # Getting the type of 'y' (line 298)
        y_262165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 43), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 40), tuple_262163, y_262165)
        
        keyword_262166 = tuple_262163
        # Getting the type of 'guiEvent' (line 298)
        guiEvent_262167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 56), 'guiEvent', False)
        keyword_262168 = guiEvent_262167
        kwargs_262169 = {'guiEvent': keyword_262168, 'xy': keyword_262166}
        # Getting the type of 'self' (line 298)
        self_262161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'self', False)
        # Obtaining the member 'enter_notify_event' of a type (line 298)
        enter_notify_event_262162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 12), self_262161, 'enter_notify_event')
        # Calling enter_notify_event(args, kwargs) (line 298)
        enter_notify_event_call_result_262170 = invoke(stypy.reporting.localization.Localization(__file__, 298, 12), enter_notify_event_262162, *[], **kwargs_262169)
        
        # SSA branch for the else part of an if statement (line 297)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'e_type' (line 299)
        e_type_262171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 13), 'e_type')
        unicode_262172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 23), 'unicode', u'figure_leave')
        # Applying the binary operator '==' (line 299)
        result_eq_262173 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 13), '==', e_type_262171, unicode_262172)
        
        # Testing the type of an if condition (line 299)
        if_condition_262174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 13), result_eq_262173)
        # Assigning a type to the variable 'if_condition_262174' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 13), 'if_condition_262174', if_condition_262174)
        # SSA begins for if statement (line 299)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to leave_notify_event(...): (line 300)
        # Processing the call keyword arguments (line 300)
        kwargs_262177 = {}
        # Getting the type of 'self' (line 300)
        self_262175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'self', False)
        # Obtaining the member 'leave_notify_event' of a type (line 300)
        leave_notify_event_262176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), self_262175, 'leave_notify_event')
        # Calling leave_notify_event(args, kwargs) (line 300)
        leave_notify_event_call_result_262178 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), leave_notify_event_262176, *[], **kwargs_262177)
        
        # SSA branch for the else part of an if statement (line 299)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'e_type' (line 301)
        e_type_262179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 13), 'e_type')
        unicode_262180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 23), 'unicode', u'scroll')
        # Applying the binary operator '==' (line 301)
        result_eq_262181 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 13), '==', e_type_262179, unicode_262180)
        
        # Testing the type of an if condition (line 301)
        if_condition_262182 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 13), result_eq_262181)
        # Assigning a type to the variable 'if_condition_262182' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 13), 'if_condition_262182', if_condition_262182)
        # SSA begins for if statement (line 301)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to scroll_event(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'x' (line 302)
        x_262185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 30), 'x', False)
        # Getting the type of 'y' (line 302)
        y_262186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 33), 'y', False)
        
        # Obtaining the type of the subscript
        unicode_262187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 42), 'unicode', u'step')
        # Getting the type of 'event' (line 302)
        event_262188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 36), 'event', False)
        # Obtaining the member '__getitem__' of a type (line 302)
        getitem___262189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 36), event_262188, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 302)
        subscript_call_result_262190 = invoke(stypy.reporting.localization.Localization(__file__, 302, 36), getitem___262189, unicode_262187)
        
        # Processing the call keyword arguments (line 302)
        # Getting the type of 'guiEvent' (line 302)
        guiEvent_262191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 60), 'guiEvent', False)
        keyword_262192 = guiEvent_262191
        kwargs_262193 = {'guiEvent': keyword_262192}
        # Getting the type of 'self' (line 302)
        self_262183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'self', False)
        # Obtaining the member 'scroll_event' of a type (line 302)
        scroll_event_262184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 12), self_262183, 'scroll_event')
        # Calling scroll_event(args, kwargs) (line 302)
        scroll_event_call_result_262194 = invoke(stypy.reporting.localization.Localization(__file__, 302, 12), scroll_event_262184, *[x_262185, y_262186, subscript_call_result_262190], **kwargs_262193)
        
        # SSA join for if statement (line 301)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 299)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 297)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 295)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 293)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 291)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_handle_mouse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_handle_mouse' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_262195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262195)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_handle_mouse'
        return stypy_return_type_262195

    
    # Multiple assignment of 6 elements.

    @norecursion
    def _handle_key(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_handle_key'
        module_type_store = module_type_store.open_function_context('_handle_key', 307, 4, False)
        # Assigning a type to the variable 'self' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore._handle_key.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore._handle_key.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore._handle_key.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore._handle_key.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore._handle_key')
        FigureCanvasWebAggCore._handle_key.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasWebAggCore._handle_key.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore._handle_key.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore._handle_key.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore._handle_key.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore._handle_key.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore._handle_key.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore._handle_key', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_handle_key', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_handle_key(...)' code ##################

        
        # Assigning a Call to a Name (line 308):
        
        # Assigning a Call to a Name (line 308):
        
        # Call to _handle_key(...): (line 308)
        # Processing the call arguments (line 308)
        
        # Obtaining the type of the subscript
        unicode_262197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 32), 'unicode', u'key')
        # Getting the type of 'event' (line 308)
        event_262198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 26), 'event', False)
        # Obtaining the member '__getitem__' of a type (line 308)
        getitem___262199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 26), event_262198, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 308)
        subscript_call_result_262200 = invoke(stypy.reporting.localization.Localization(__file__, 308, 26), getitem___262199, unicode_262197)
        
        # Processing the call keyword arguments (line 308)
        kwargs_262201 = {}
        # Getting the type of '_handle_key' (line 308)
        _handle_key_262196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 14), '_handle_key', False)
        # Calling _handle_key(args, kwargs) (line 308)
        _handle_key_call_result_262202 = invoke(stypy.reporting.localization.Localization(__file__, 308, 14), _handle_key_262196, *[subscript_call_result_262200], **kwargs_262201)
        
        # Assigning a type to the variable 'key' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'key', _handle_key_call_result_262202)
        
        # Assigning a Subscript to a Name (line 309):
        
        # Assigning a Subscript to a Name (line 309):
        
        # Obtaining the type of the subscript
        unicode_262203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 23), 'unicode', u'type')
        # Getting the type of 'event' (line 309)
        event_262204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 17), 'event')
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___262205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 17), event_262204, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_262206 = invoke(stypy.reporting.localization.Localization(__file__, 309, 17), getitem___262205, unicode_262203)
        
        # Assigning a type to the variable 'e_type' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'e_type', subscript_call_result_262206)
        
        # Assigning a Call to a Name (line 310):
        
        # Assigning a Call to a Name (line 310):
        
        # Call to get(...): (line 310)
        # Processing the call arguments (line 310)
        unicode_262209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 29), 'unicode', u'guiEvent')
        # Getting the type of 'None' (line 310)
        None_262210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 41), 'None', False)
        # Processing the call keyword arguments (line 310)
        kwargs_262211 = {}
        # Getting the type of 'event' (line 310)
        event_262207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'event', False)
        # Obtaining the member 'get' of a type (line 310)
        get_262208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 19), event_262207, 'get')
        # Calling get(args, kwargs) (line 310)
        get_call_result_262212 = invoke(stypy.reporting.localization.Localization(__file__, 310, 19), get_262208, *[unicode_262209, None_262210], **kwargs_262211)
        
        # Assigning a type to the variable 'guiEvent' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'guiEvent', get_call_result_262212)
        
        
        # Getting the type of 'e_type' (line 311)
        e_type_262213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 11), 'e_type')
        unicode_262214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 21), 'unicode', u'key_press')
        # Applying the binary operator '==' (line 311)
        result_eq_262215 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 11), '==', e_type_262213, unicode_262214)
        
        # Testing the type of an if condition (line 311)
        if_condition_262216 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 8), result_eq_262215)
        # Assigning a type to the variable 'if_condition_262216' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'if_condition_262216', if_condition_262216)
        # SSA begins for if statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to key_press_event(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'key' (line 312)
        key_262219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 33), 'key', False)
        # Processing the call keyword arguments (line 312)
        # Getting the type of 'guiEvent' (line 312)
        guiEvent_262220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 47), 'guiEvent', False)
        keyword_262221 = guiEvent_262220
        kwargs_262222 = {'guiEvent': keyword_262221}
        # Getting the type of 'self' (line 312)
        self_262217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'self', False)
        # Obtaining the member 'key_press_event' of a type (line 312)
        key_press_event_262218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), self_262217, 'key_press_event')
        # Calling key_press_event(args, kwargs) (line 312)
        key_press_event_call_result_262223 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), key_press_event_262218, *[key_262219], **kwargs_262222)
        
        # SSA branch for the else part of an if statement (line 311)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'e_type' (line 313)
        e_type_262224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'e_type')
        unicode_262225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 23), 'unicode', u'key_release')
        # Applying the binary operator '==' (line 313)
        result_eq_262226 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 13), '==', e_type_262224, unicode_262225)
        
        # Testing the type of an if condition (line 313)
        if_condition_262227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 13), result_eq_262226)
        # Assigning a type to the variable 'if_condition_262227' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'if_condition_262227', if_condition_262227)
        # SSA begins for if statement (line 313)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to key_release_event(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'key' (line 314)
        key_262230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 35), 'key', False)
        # Processing the call keyword arguments (line 314)
        # Getting the type of 'guiEvent' (line 314)
        guiEvent_262231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 49), 'guiEvent', False)
        keyword_262232 = guiEvent_262231
        kwargs_262233 = {'guiEvent': keyword_262232}
        # Getting the type of 'self' (line 314)
        self_262228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'self', False)
        # Obtaining the member 'key_release_event' of a type (line 314)
        key_release_event_262229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 12), self_262228, 'key_release_event')
        # Calling key_release_event(args, kwargs) (line 314)
        key_release_event_call_result_262234 = invoke(stypy.reporting.localization.Localization(__file__, 314, 12), key_release_event_262229, *[key_262230], **kwargs_262233)
        
        # SSA join for if statement (line 313)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 311)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_handle_key(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_handle_key' in the type store
        # Getting the type of 'stypy_return_type' (line 307)
        stypy_return_type_262235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262235)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_handle_key'
        return stypy_return_type_262235

    
    # Multiple assignment of 2 elements.

    @norecursion
    def handle_toolbar_button(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_toolbar_button'
        module_type_store = module_type_store.open_function_context('handle_toolbar_button', 317, 4, False)
        # Assigning a type to the variable 'self' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.handle_toolbar_button.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.handle_toolbar_button.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.handle_toolbar_button.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.handle_toolbar_button.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.handle_toolbar_button')
        FigureCanvasWebAggCore.handle_toolbar_button.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasWebAggCore.handle_toolbar_button.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.handle_toolbar_button.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.handle_toolbar_button.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.handle_toolbar_button.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.handle_toolbar_button.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.handle_toolbar_button.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.handle_toolbar_button', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_toolbar_button', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_toolbar_button(...)' code ##################

        
        # Call to (...): (line 319)
        # Processing the call keyword arguments (line 319)
        kwargs_262245 = {}
        
        # Call to getattr(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'self' (line 319)
        self_262237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 16), 'self', False)
        # Obtaining the member 'toolbar' of a type (line 319)
        toolbar_262238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 16), self_262237, 'toolbar')
        
        # Obtaining the type of the subscript
        unicode_262239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 36), 'unicode', u'name')
        # Getting the type of 'event' (line 319)
        event_262240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 30), 'event', False)
        # Obtaining the member '__getitem__' of a type (line 319)
        getitem___262241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 30), event_262240, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 319)
        subscript_call_result_262242 = invoke(stypy.reporting.localization.Localization(__file__, 319, 30), getitem___262241, unicode_262239)
        
        # Processing the call keyword arguments (line 319)
        kwargs_262243 = {}
        # Getting the type of 'getattr' (line 319)
        getattr_262236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'getattr', False)
        # Calling getattr(args, kwargs) (line 319)
        getattr_call_result_262244 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), getattr_262236, *[toolbar_262238, subscript_call_result_262242], **kwargs_262243)
        
        # Calling (args, kwargs) (line 319)
        _call_result_262246 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), getattr_call_result_262244, *[], **kwargs_262245)
        
        
        # ################# End of 'handle_toolbar_button(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_toolbar_button' in the type store
        # Getting the type of 'stypy_return_type' (line 317)
        stypy_return_type_262247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262247)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_toolbar_button'
        return stypy_return_type_262247


    @norecursion
    def handle_refresh(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_refresh'
        module_type_store = module_type_store.open_function_context('handle_refresh', 321, 4, False)
        # Assigning a type to the variable 'self' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.handle_refresh.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.handle_refresh.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.handle_refresh.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.handle_refresh.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.handle_refresh')
        FigureCanvasWebAggCore.handle_refresh.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasWebAggCore.handle_refresh.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.handle_refresh.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.handle_refresh.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.handle_refresh.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.handle_refresh.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.handle_refresh.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.handle_refresh', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_refresh', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_refresh(...)' code ##################

        
        # Assigning a Call to a Name (line 322):
        
        # Assigning a Call to a Name (line 322):
        
        # Call to get_label(...): (line 322)
        # Processing the call keyword arguments (line 322)
        kwargs_262251 = {}
        # Getting the type of 'self' (line 322)
        self_262248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 23), 'self', False)
        # Obtaining the member 'figure' of a type (line 322)
        figure_262249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 23), self_262248, 'figure')
        # Obtaining the member 'get_label' of a type (line 322)
        get_label_262250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 23), figure_262249, 'get_label')
        # Calling get_label(args, kwargs) (line 322)
        get_label_call_result_262252 = invoke(stypy.reporting.localization.Localization(__file__, 322, 23), get_label_262250, *[], **kwargs_262251)
        
        # Assigning a type to the variable 'figure_label' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'figure_label', get_label_call_result_262252)
        
        
        # Getting the type of 'figure_label' (line 323)
        figure_label_262253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'figure_label')
        # Applying the 'not' unary operator (line 323)
        result_not__262254 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 11), 'not', figure_label_262253)
        
        # Testing the type of an if condition (line 323)
        if_condition_262255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 8), result_not__262254)
        # Assigning a type to the variable 'if_condition_262255' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'if_condition_262255', if_condition_262255)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to format(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'self' (line 324)
        self_262258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 47), 'self', False)
        # Obtaining the member 'manager' of a type (line 324)
        manager_262259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 47), self_262258, 'manager')
        # Obtaining the member 'num' of a type (line 324)
        num_262260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 47), manager_262259, 'num')
        # Processing the call keyword arguments (line 324)
        kwargs_262261 = {}
        unicode_262256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 27), 'unicode', u'Figure {0}')
        # Obtaining the member 'format' of a type (line 324)
        format_262257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 27), unicode_262256, 'format')
        # Calling format(args, kwargs) (line 324)
        format_call_result_262262 = invoke(stypy.reporting.localization.Localization(__file__, 324, 27), format_262257, *[num_262260], **kwargs_262261)
        
        # Assigning a type to the variable 'figure_label' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'figure_label', format_call_result_262262)
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to send_event(...): (line 325)
        # Processing the call arguments (line 325)
        unicode_262265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 24), 'unicode', u'figure_label')
        # Processing the call keyword arguments (line 325)
        # Getting the type of 'figure_label' (line 325)
        figure_label_262266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 46), 'figure_label', False)
        keyword_262267 = figure_label_262266
        kwargs_262268 = {'label': keyword_262267}
        # Getting the type of 'self' (line 325)
        self_262263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'self', False)
        # Obtaining the member 'send_event' of a type (line 325)
        send_event_262264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), self_262263, 'send_event')
        # Calling send_event(args, kwargs) (line 325)
        send_event_call_result_262269 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), send_event_262264, *[unicode_262265], **kwargs_262268)
        
        
        # Assigning a Name to a Attribute (line 326):
        
        # Assigning a Name to a Attribute (line 326):
        # Getting the type of 'True' (line 326)
        True_262270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 27), 'True')
        # Getting the type of 'self' (line 326)
        self_262271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'self')
        # Setting the type of the member '_force_full' of a type (line 326)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), self_262271, '_force_full', True_262270)
        
        # Call to draw_idle(...): (line 327)
        # Processing the call keyword arguments (line 327)
        kwargs_262274 = {}
        # Getting the type of 'self' (line 327)
        self_262272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'self', False)
        # Obtaining the member 'draw_idle' of a type (line 327)
        draw_idle_262273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 8), self_262272, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 327)
        draw_idle_call_result_262275 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), draw_idle_262273, *[], **kwargs_262274)
        
        
        # ################# End of 'handle_refresh(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_refresh' in the type store
        # Getting the type of 'stypy_return_type' (line 321)
        stypy_return_type_262276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_refresh'
        return stypy_return_type_262276


    @norecursion
    def handle_resize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_resize'
        module_type_store = module_type_store.open_function_context('handle_resize', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.handle_resize.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.handle_resize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.handle_resize.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.handle_resize.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.handle_resize')
        FigureCanvasWebAggCore.handle_resize.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasWebAggCore.handle_resize.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.handle_resize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.handle_resize.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.handle_resize.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.handle_resize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.handle_resize.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.handle_resize', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_resize', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_resize(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 330):
        
        # Assigning a Call to a Name (line 330):
        
        # Call to get(...): (line 330)
        # Processing the call arguments (line 330)
        unicode_262279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 25), 'unicode', u'width')
        int_262280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 34), 'int')
        # Processing the call keyword arguments (line 330)
        kwargs_262281 = {}
        # Getting the type of 'event' (line 330)
        event_262277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 15), 'event', False)
        # Obtaining the member 'get' of a type (line 330)
        get_262278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 15), event_262277, 'get')
        # Calling get(args, kwargs) (line 330)
        get_call_result_262282 = invoke(stypy.reporting.localization.Localization(__file__, 330, 15), get_262278, *[unicode_262279, int_262280], **kwargs_262281)
        
        # Assigning a type to the variable 'tuple_assignment_261456' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_assignment_261456', get_call_result_262282)
        
        # Assigning a Call to a Name (line 330):
        
        # Call to get(...): (line 330)
        # Processing the call arguments (line 330)
        unicode_262285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 50), 'unicode', u'height')
        int_262286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 60), 'int')
        # Processing the call keyword arguments (line 330)
        kwargs_262287 = {}
        # Getting the type of 'event' (line 330)
        event_262283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 40), 'event', False)
        # Obtaining the member 'get' of a type (line 330)
        get_262284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 40), event_262283, 'get')
        # Calling get(args, kwargs) (line 330)
        get_call_result_262288 = invoke(stypy.reporting.localization.Localization(__file__, 330, 40), get_262284, *[unicode_262285, int_262286], **kwargs_262287)
        
        # Assigning a type to the variable 'tuple_assignment_261457' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_assignment_261457', get_call_result_262288)
        
        # Assigning a Name to a Name (line 330):
        # Getting the type of 'tuple_assignment_261456' (line 330)
        tuple_assignment_261456_262289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_assignment_261456')
        # Assigning a type to the variable 'x' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'x', tuple_assignment_261456_262289)
        
        # Assigning a Name to a Name (line 330):
        # Getting the type of 'tuple_assignment_261457' (line 330)
        tuple_assignment_261457_262290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'tuple_assignment_261457')
        # Assigning a type to the variable 'y' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'y', tuple_assignment_261457_262290)
        
        # Assigning a Tuple to a Tuple (line 331):
        
        # Assigning a BinOp to a Name (line 331):
        
        # Call to int(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'x' (line 331)
        x_262292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 19), 'x', False)
        # Processing the call keyword arguments (line 331)
        kwargs_262293 = {}
        # Getting the type of 'int' (line 331)
        int_262291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 15), 'int', False)
        # Calling int(args, kwargs) (line 331)
        int_call_result_262294 = invoke(stypy.reporting.localization.Localization(__file__, 331, 15), int_262291, *[x_262292], **kwargs_262293)
        
        # Getting the type of 'self' (line 331)
        self_262295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 331)
        _dpi_ratio_262296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 24), self_262295, '_dpi_ratio')
        # Applying the binary operator '*' (line 331)
        result_mul_262297 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 15), '*', int_call_result_262294, _dpi_ratio_262296)
        
        # Assigning a type to the variable 'tuple_assignment_261458' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'tuple_assignment_261458', result_mul_262297)
        
        # Assigning a BinOp to a Name (line 331):
        
        # Call to int(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'y' (line 331)
        y_262299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 45), 'y', False)
        # Processing the call keyword arguments (line 331)
        kwargs_262300 = {}
        # Getting the type of 'int' (line 331)
        int_262298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 41), 'int', False)
        # Calling int(args, kwargs) (line 331)
        int_call_result_262301 = invoke(stypy.reporting.localization.Localization(__file__, 331, 41), int_262298, *[y_262299], **kwargs_262300)
        
        # Getting the type of 'self' (line 331)
        self_262302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 50), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 331)
        _dpi_ratio_262303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 50), self_262302, '_dpi_ratio')
        # Applying the binary operator '*' (line 331)
        result_mul_262304 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 41), '*', int_call_result_262301, _dpi_ratio_262303)
        
        # Assigning a type to the variable 'tuple_assignment_261459' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'tuple_assignment_261459', result_mul_262304)
        
        # Assigning a Name to a Name (line 331):
        # Getting the type of 'tuple_assignment_261458' (line 331)
        tuple_assignment_261458_262305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'tuple_assignment_261458')
        # Assigning a type to the variable 'x' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'x', tuple_assignment_261458_262305)
        
        # Assigning a Name to a Name (line 331):
        # Getting the type of 'tuple_assignment_261459' (line 331)
        tuple_assignment_261459_262306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'tuple_assignment_261459')
        # Assigning a type to the variable 'y' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'y', tuple_assignment_261459_262306)
        
        # Assigning a Attribute to a Name (line 332):
        
        # Assigning a Attribute to a Name (line 332):
        # Getting the type of 'self' (line 332)
        self_262307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), 'self')
        # Obtaining the member 'figure' of a type (line 332)
        figure_262308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 14), self_262307, 'figure')
        # Assigning a type to the variable 'fig' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'fig', figure_262308)
        
        # Call to set_size_inches(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'x' (line 334)
        x_262311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 28), 'x', False)
        # Getting the type of 'fig' (line 334)
        fig_262312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 32), 'fig', False)
        # Obtaining the member 'dpi' of a type (line 334)
        dpi_262313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 32), fig_262312, 'dpi')
        # Applying the binary operator 'div' (line 334)
        result_div_262314 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 28), 'div', x_262311, dpi_262313)
        
        # Getting the type of 'y' (line 334)
        y_262315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 41), 'y', False)
        # Getting the type of 'fig' (line 334)
        fig_262316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 45), 'fig', False)
        # Obtaining the member 'dpi' of a type (line 334)
        dpi_262317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 45), fig_262316, 'dpi')
        # Applying the binary operator 'div' (line 334)
        result_div_262318 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 41), 'div', y_262315, dpi_262317)
        
        # Processing the call keyword arguments (line 334)
        # Getting the type of 'False' (line 334)
        False_262319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 62), 'False', False)
        keyword_262320 = False_262319
        kwargs_262321 = {'forward': keyword_262320}
        # Getting the type of 'fig' (line 334)
        fig_262309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'fig', False)
        # Obtaining the member 'set_size_inches' of a type (line 334)
        set_size_inches_262310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 8), fig_262309, 'set_size_inches')
        # Calling set_size_inches(args, kwargs) (line 334)
        set_size_inches_call_result_262322 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), set_size_inches_262310, *[result_div_262314, result_div_262318], **kwargs_262321)
        
        
        # Assigning a Attribute to a Tuple (line 336):
        
        # Assigning a Subscript to a Name (line 336):
        
        # Obtaining the type of the subscript
        int_262323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 8), 'int')
        # Getting the type of 'self' (line 336)
        self_262324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 21), 'self')
        # Obtaining the member 'figure' of a type (line 336)
        figure_262325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), self_262324, 'figure')
        # Obtaining the member 'bbox' of a type (line 336)
        bbox_262326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), figure_262325, 'bbox')
        # Obtaining the member 'bounds' of a type (line 336)
        bounds_262327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), bbox_262326, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 336)
        getitem___262328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), bounds_262327, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 336)
        subscript_call_result_262329 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), getitem___262328, int_262323)
        
        # Assigning a type to the variable 'tuple_var_assignment_261460' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_var_assignment_261460', subscript_call_result_262329)
        
        # Assigning a Subscript to a Name (line 336):
        
        # Obtaining the type of the subscript
        int_262330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 8), 'int')
        # Getting the type of 'self' (line 336)
        self_262331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 21), 'self')
        # Obtaining the member 'figure' of a type (line 336)
        figure_262332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), self_262331, 'figure')
        # Obtaining the member 'bbox' of a type (line 336)
        bbox_262333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), figure_262332, 'bbox')
        # Obtaining the member 'bounds' of a type (line 336)
        bounds_262334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), bbox_262333, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 336)
        getitem___262335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), bounds_262334, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 336)
        subscript_call_result_262336 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), getitem___262335, int_262330)
        
        # Assigning a type to the variable 'tuple_var_assignment_261461' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_var_assignment_261461', subscript_call_result_262336)
        
        # Assigning a Subscript to a Name (line 336):
        
        # Obtaining the type of the subscript
        int_262337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 8), 'int')
        # Getting the type of 'self' (line 336)
        self_262338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 21), 'self')
        # Obtaining the member 'figure' of a type (line 336)
        figure_262339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), self_262338, 'figure')
        # Obtaining the member 'bbox' of a type (line 336)
        bbox_262340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), figure_262339, 'bbox')
        # Obtaining the member 'bounds' of a type (line 336)
        bounds_262341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), bbox_262340, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 336)
        getitem___262342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), bounds_262341, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 336)
        subscript_call_result_262343 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), getitem___262342, int_262337)
        
        # Assigning a type to the variable 'tuple_var_assignment_261462' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_var_assignment_261462', subscript_call_result_262343)
        
        # Assigning a Subscript to a Name (line 336):
        
        # Obtaining the type of the subscript
        int_262344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 8), 'int')
        # Getting the type of 'self' (line 336)
        self_262345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 21), 'self')
        # Obtaining the member 'figure' of a type (line 336)
        figure_262346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), self_262345, 'figure')
        # Obtaining the member 'bbox' of a type (line 336)
        bbox_262347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), figure_262346, 'bbox')
        # Obtaining the member 'bounds' of a type (line 336)
        bounds_262348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 21), bbox_262347, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 336)
        getitem___262349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), bounds_262348, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 336)
        subscript_call_result_262350 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), getitem___262349, int_262344)
        
        # Assigning a type to the variable 'tuple_var_assignment_261463' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_var_assignment_261463', subscript_call_result_262350)
        
        # Assigning a Name to a Name (line 336):
        # Getting the type of 'tuple_var_assignment_261460' (line 336)
        tuple_var_assignment_261460_262351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_var_assignment_261460')
        # Assigning a type to the variable '_' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), '_', tuple_var_assignment_261460_262351)
        
        # Assigning a Name to a Name (line 336):
        # Getting the type of 'tuple_var_assignment_261461' (line 336)
        tuple_var_assignment_261461_262352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_var_assignment_261461')
        # Assigning a type to the variable '_' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), '_', tuple_var_assignment_261461_262352)
        
        # Assigning a Name to a Name (line 336):
        # Getting the type of 'tuple_var_assignment_261462' (line 336)
        tuple_var_assignment_261462_262353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_var_assignment_261462')
        # Assigning a type to the variable 'w' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 14), 'w', tuple_var_assignment_261462_262353)
        
        # Assigning a Name to a Name (line 336):
        # Getting the type of 'tuple_var_assignment_261463' (line 336)
        tuple_var_assignment_261463_262354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'tuple_var_assignment_261463')
        # Assigning a type to the variable 'h' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 17), 'h', tuple_var_assignment_261463_262354)
        
        # Assigning a Name to a Attribute (line 340):
        
        # Assigning a Name to a Attribute (line 340):
        # Getting the type of 'True' (line 340)
        True_262355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 27), 'True')
        # Getting the type of 'self' (line 340)
        self_262356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'self')
        # Setting the type of the member '_png_is_old' of a type (line 340)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), self_262356, '_png_is_old', True_262355)
        
        # Call to resize(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'w' (line 341)
        w_262360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 28), 'w', False)
        # Getting the type of 'h' (line 341)
        h_262361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 31), 'h', False)
        # Processing the call keyword arguments (line 341)
        kwargs_262362 = {}
        # Getting the type of 'self' (line 341)
        self_262357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self', False)
        # Obtaining the member 'manager' of a type (line 341)
        manager_262358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_262357, 'manager')
        # Obtaining the member 'resize' of a type (line 341)
        resize_262359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), manager_262358, 'resize')
        # Calling resize(args, kwargs) (line 341)
        resize_call_result_262363 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), resize_262359, *[w_262360, h_262361], **kwargs_262362)
        
        
        # Call to resize_event(...): (line 342)
        # Processing the call keyword arguments (line 342)
        kwargs_262366 = {}
        # Getting the type of 'self' (line 342)
        self_262364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'self', False)
        # Obtaining the member 'resize_event' of a type (line 342)
        resize_event_262365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), self_262364, 'resize_event')
        # Calling resize_event(args, kwargs) (line 342)
        resize_event_call_result_262367 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), resize_event_262365, *[], **kwargs_262366)
        
        
        # ################# End of 'handle_resize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_resize' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_262368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262368)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_resize'
        return stypy_return_type_262368


    @norecursion
    def handle_send_image_mode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_send_image_mode'
        module_type_store = module_type_store.open_function_context('handle_send_image_mode', 344, 4, False)
        # Assigning a type to the variable 'self' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.handle_send_image_mode.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.handle_send_image_mode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.handle_send_image_mode.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.handle_send_image_mode.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.handle_send_image_mode')
        FigureCanvasWebAggCore.handle_send_image_mode.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasWebAggCore.handle_send_image_mode.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.handle_send_image_mode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.handle_send_image_mode.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.handle_send_image_mode.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.handle_send_image_mode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.handle_send_image_mode.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.handle_send_image_mode', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_send_image_mode', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_send_image_mode(...)' code ##################

        
        # Call to send_event(...): (line 346)
        # Processing the call arguments (line 346)
        unicode_262371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 24), 'unicode', u'image_mode')
        # Processing the call keyword arguments (line 346)
        # Getting the type of 'self' (line 346)
        self_262372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 43), 'self', False)
        # Obtaining the member '_current_image_mode' of a type (line 346)
        _current_image_mode_262373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 43), self_262372, '_current_image_mode')
        keyword_262374 = _current_image_mode_262373
        kwargs_262375 = {'mode': keyword_262374}
        # Getting the type of 'self' (line 346)
        self_262369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'self', False)
        # Obtaining the member 'send_event' of a type (line 346)
        send_event_262370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), self_262369, 'send_event')
        # Calling send_event(args, kwargs) (line 346)
        send_event_call_result_262376 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), send_event_262370, *[unicode_262371], **kwargs_262375)
        
        
        # ################# End of 'handle_send_image_mode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_send_image_mode' in the type store
        # Getting the type of 'stypy_return_type' (line 344)
        stypy_return_type_262377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262377)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_send_image_mode'
        return stypy_return_type_262377


    @norecursion
    def handle_set_dpi_ratio(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_set_dpi_ratio'
        module_type_store = module_type_store.open_function_context('handle_set_dpi_ratio', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.handle_set_dpi_ratio.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.handle_set_dpi_ratio.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.handle_set_dpi_ratio.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.handle_set_dpi_ratio.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.handle_set_dpi_ratio')
        FigureCanvasWebAggCore.handle_set_dpi_ratio.__dict__.__setitem__('stypy_param_names_list', ['event'])
        FigureCanvasWebAggCore.handle_set_dpi_ratio.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.handle_set_dpi_ratio.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAggCore.handle_set_dpi_ratio.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.handle_set_dpi_ratio.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.handle_set_dpi_ratio.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.handle_set_dpi_ratio.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.handle_set_dpi_ratio', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_set_dpi_ratio', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_set_dpi_ratio(...)' code ##################

        
        # Assigning a Call to a Name (line 349):
        
        # Assigning a Call to a Name (line 349):
        
        # Call to get(...): (line 349)
        # Processing the call arguments (line 349)
        unicode_262380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 30), 'unicode', u'dpi_ratio')
        int_262381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 43), 'int')
        # Processing the call keyword arguments (line 349)
        kwargs_262382 = {}
        # Getting the type of 'event' (line 349)
        event_262378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 20), 'event', False)
        # Obtaining the member 'get' of a type (line 349)
        get_262379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 20), event_262378, 'get')
        # Calling get(args, kwargs) (line 349)
        get_call_result_262383 = invoke(stypy.reporting.localization.Localization(__file__, 349, 20), get_262379, *[unicode_262380, int_262381], **kwargs_262382)
        
        # Assigning a type to the variable 'dpi_ratio' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'dpi_ratio', get_call_result_262383)
        
        
        # Getting the type of 'dpi_ratio' (line 350)
        dpi_ratio_262384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 11), 'dpi_ratio')
        # Getting the type of 'self' (line 350)
        self_262385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 24), 'self')
        # Obtaining the member '_dpi_ratio' of a type (line 350)
        _dpi_ratio_262386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 24), self_262385, '_dpi_ratio')
        # Applying the binary operator '!=' (line 350)
        result_ne_262387 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 11), '!=', dpi_ratio_262384, _dpi_ratio_262386)
        
        # Testing the type of an if condition (line 350)
        if_condition_262388 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 8), result_ne_262387)
        # Assigning a type to the variable 'if_condition_262388' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'if_condition_262388', if_condition_262388)
        # SSA begins for if statement (line 350)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 352)
        unicode_262389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 40), 'unicode', u'_original_dpi')
        # Getting the type of 'self' (line 352)
        self_262390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 27), 'self')
        # Obtaining the member 'figure' of a type (line 352)
        figure_262391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 27), self_262390, 'figure')
        
        (may_be_262392, more_types_in_union_262393) = may_not_provide_member(unicode_262389, figure_262391)

        if may_be_262392:

            if more_types_in_union_262393:
                # Runtime conditional SSA (line 352)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 352)
            self_262394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'self')
            # Obtaining the member 'figure' of a type (line 352)
            figure_262395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 12), self_262394, 'figure')
            # Setting the type of the member 'figure' of a type (line 352)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 12), self_262394, 'figure', remove_member_provider_from_union(figure_262391, u'_original_dpi'))
            
            # Assigning a Attribute to a Attribute (line 353):
            
            # Assigning a Attribute to a Attribute (line 353):
            # Getting the type of 'self' (line 353)
            self_262396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 44), 'self')
            # Obtaining the member 'figure' of a type (line 353)
            figure_262397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 44), self_262396, 'figure')
            # Obtaining the member 'dpi' of a type (line 353)
            dpi_262398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 44), figure_262397, 'dpi')
            # Getting the type of 'self' (line 353)
            self_262399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'self')
            # Obtaining the member 'figure' of a type (line 353)
            figure_262400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 16), self_262399, 'figure')
            # Setting the type of the member '_original_dpi' of a type (line 353)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 16), figure_262400, '_original_dpi', dpi_262398)

            if more_types_in_union_262393:
                # SSA join for if statement (line 352)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Attribute (line 354):
        
        # Assigning a BinOp to a Attribute (line 354):
        # Getting the type of 'dpi_ratio' (line 354)
        dpi_ratio_262401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 30), 'dpi_ratio')
        # Getting the type of 'self' (line 354)
        self_262402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 42), 'self')
        # Obtaining the member 'figure' of a type (line 354)
        figure_262403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 42), self_262402, 'figure')
        # Obtaining the member '_original_dpi' of a type (line 354)
        _original_dpi_262404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 42), figure_262403, '_original_dpi')
        # Applying the binary operator '*' (line 354)
        result_mul_262405 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 30), '*', dpi_ratio_262401, _original_dpi_262404)
        
        # Getting the type of 'self' (line 354)
        self_262406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'self')
        # Obtaining the member 'figure' of a type (line 354)
        figure_262407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 12), self_262406, 'figure')
        # Setting the type of the member 'dpi' of a type (line 354)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 12), figure_262407, 'dpi', result_mul_262405)
        
        # Assigning a Name to a Attribute (line 355):
        
        # Assigning a Name to a Attribute (line 355):
        # Getting the type of 'dpi_ratio' (line 355)
        dpi_ratio_262408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 30), 'dpi_ratio')
        # Getting the type of 'self' (line 355)
        self_262409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'self')
        # Setting the type of the member '_dpi_ratio' of a type (line 355)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 12), self_262409, '_dpi_ratio', dpi_ratio_262408)
        
        # Assigning a Name to a Attribute (line 356):
        
        # Assigning a Name to a Attribute (line 356):
        # Getting the type of 'True' (line 356)
        True_262410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 31), 'True')
        # Getting the type of 'self' (line 356)
        self_262411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'self')
        # Setting the type of the member '_force_full' of a type (line 356)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 12), self_262411, '_force_full', True_262410)
        
        # Call to draw_idle(...): (line 357)
        # Processing the call keyword arguments (line 357)
        kwargs_262414 = {}
        # Getting the type of 'self' (line 357)
        self_262412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'self', False)
        # Obtaining the member 'draw_idle' of a type (line 357)
        draw_idle_262413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 12), self_262412, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 357)
        draw_idle_call_result_262415 = invoke(stypy.reporting.localization.Localization(__file__, 357, 12), draw_idle_262413, *[], **kwargs_262414)
        
        # SSA join for if statement (line 350)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'handle_set_dpi_ratio(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_set_dpi_ratio' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_262416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262416)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_set_dpi_ratio'
        return stypy_return_type_262416


    @norecursion
    def send_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'send_event'
        module_type_store = module_type_store.open_function_context('send_event', 359, 4, False)
        # Assigning a type to the variable 'self' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAggCore.send_event.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAggCore.send_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAggCore.send_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAggCore.send_event.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAggCore.send_event')
        FigureCanvasWebAggCore.send_event.__dict__.__setitem__('stypy_param_names_list', ['event_type'])
        FigureCanvasWebAggCore.send_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAggCore.send_event.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasWebAggCore.send_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAggCore.send_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAggCore.send_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAggCore.send_event.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAggCore.send_event', ['event_type'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'send_event', localization, ['event_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'send_event(...)' code ##################

        
        # Call to _send_event(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'event_type' (line 360)
        event_type_262420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 33), 'event_type', False)
        # Processing the call keyword arguments (line 360)
        # Getting the type of 'kwargs' (line 360)
        kwargs_262421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 47), 'kwargs', False)
        kwargs_262422 = {'kwargs_262421': kwargs_262421}
        # Getting the type of 'self' (line 360)
        self_262417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'self', False)
        # Obtaining the member 'manager' of a type (line 360)
        manager_262418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 8), self_262417, 'manager')
        # Obtaining the member '_send_event' of a type (line 360)
        _send_event_262419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 8), manager_262418, '_send_event')
        # Calling _send_event(args, kwargs) (line 360)
        _send_event_call_result_262423 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), _send_event_262419, *[event_type_262420], **kwargs_262422)
        
        
        # ################# End of 'send_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'send_event' in the type store
        # Getting the type of 'stypy_return_type' (line 359)
        stypy_return_type_262424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262424)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'send_event'
        return stypy_return_type_262424


# Assigning a type to the variable 'FigureCanvasWebAggCore' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'FigureCanvasWebAggCore', FigureCanvasWebAggCore)

# Assigning a Name to a Name (line 126):
# Getting the type of 'False' (line 126)
False_262425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'False')
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Setting the type of the member 'supports_blit' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262426, 'supports_blit', False_262425)

# Assigning a Name to a Name (line 303):
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Obtaining the member '_handle_mouse' of a type
_handle_mouse_262428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262427, '_handle_mouse')
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Setting the type of the member 'handle_scroll' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262429, 'handle_scroll', _handle_mouse_262428)

# Assigning a Name to a Name (line 303):
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Obtaining the member 'handle_scroll' of a type
handle_scroll_262431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262430, 'handle_scroll')
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Setting the type of the member 'handle_figure_leave' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262432, 'handle_figure_leave', handle_scroll_262431)

# Assigning a Name to a Name (line 303):
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Obtaining the member 'handle_figure_leave' of a type
handle_figure_leave_262434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262433, 'handle_figure_leave')
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Setting the type of the member 'handle_figure_enter' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262435, 'handle_figure_enter', handle_figure_leave_262434)

# Assigning a Name to a Name (line 303):
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Obtaining the member 'handle_figure_enter' of a type
handle_figure_enter_262437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262436, 'handle_figure_enter')
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Setting the type of the member 'handle_motion_notify' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262438, 'handle_motion_notify', handle_figure_enter_262437)

# Assigning a Name to a Name (line 303):
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Obtaining the member 'handle_motion_notify' of a type
handle_motion_notify_262440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262439, 'handle_motion_notify')
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Setting the type of the member 'handle_button_release' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262441, 'handle_button_release', handle_motion_notify_262440)

# Assigning a Name to a Name (line 303):
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Obtaining the member 'handle_button_release' of a type
handle_button_release_262443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262442, 'handle_button_release')
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Setting the type of the member 'handle_button_press' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262444, 'handle_button_press', handle_button_release_262443)

# Assigning a Name to a Name (line 315):
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Obtaining the member '_handle_key' of a type
_handle_key_262446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262445, '_handle_key')
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Setting the type of the member 'handle_key_release' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262447, 'handle_key_release', _handle_key_262446)

# Assigning a Name to a Name (line 315):
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Obtaining the member 'handle_key_release' of a type
handle_key_release_262449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262448, 'handle_key_release')
# Getting the type of 'FigureCanvasWebAggCore'
FigureCanvasWebAggCore_262450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureCanvasWebAggCore')
# Setting the type of the member 'handle_key_press' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureCanvasWebAggCore_262450, 'handle_key_press', handle_key_release_262449)

# Assigning a Dict to a Name (line 363):

# Assigning a Dict to a Name (line 363):

# Obtaining an instance of the builtin type 'dict' (line 363)
dict_262451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 23), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 363)
# Adding element type (key, value) (line 363)
unicode_262452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 4), 'unicode', u'home')
unicode_262453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 12), 'unicode', u'ui-icon ui-icon-home')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 23), dict_262451, (unicode_262452, unicode_262453))
# Adding element type (key, value) (line 363)
unicode_262454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 4), 'unicode', u'back')
unicode_262455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 12), 'unicode', u'ui-icon ui-icon-circle-arrow-w')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 23), dict_262451, (unicode_262454, unicode_262455))
# Adding element type (key, value) (line 363)
unicode_262456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 4), 'unicode', u'forward')
unicode_262457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 15), 'unicode', u'ui-icon ui-icon-circle-arrow-e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 23), dict_262451, (unicode_262456, unicode_262457))
# Adding element type (key, value) (line 363)
unicode_262458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 4), 'unicode', u'zoom_to_rect')
unicode_262459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 20), 'unicode', u'ui-icon ui-icon-search')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 23), dict_262451, (unicode_262458, unicode_262459))
# Adding element type (key, value) (line 363)
unicode_262460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 4), 'unicode', u'move')
unicode_262461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 12), 'unicode', u'ui-icon ui-icon-arrow-4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 23), dict_262451, (unicode_262460, unicode_262461))
# Adding element type (key, value) (line 363)
unicode_262462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 4), 'unicode', u'download')
unicode_262463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 16), 'unicode', u'ui-icon ui-icon-disk')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 23), dict_262451, (unicode_262462, unicode_262463))
# Adding element type (key, value) (line 363)
# Getting the type of 'None' (line 370)
None_262464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'None')
# Getting the type of 'None' (line 370)
None_262465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 10), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 23), dict_262451, (None_262464, None_262465))

# Assigning a type to the variable '_JQUERY_ICON_CLASSES' (line 363)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 0), '_JQUERY_ICON_CLASSES', dict_262451)
# Declaration of the 'NavigationToolbar2WebAgg' class
# Getting the type of 'backend_bases' (line 374)
backend_bases_262466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 31), 'backend_bases')
# Obtaining the member 'NavigationToolbar2' of a type (line 374)
NavigationToolbar2_262467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 31), backend_bases_262466, 'NavigationToolbar2')

class NavigationToolbar2WebAgg(NavigationToolbar2_262467, ):
    
    # Assigning a ListComp to a Name (line 377):

    @norecursion
    def _init_toolbar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init_toolbar'
        module_type_store = module_type_store.open_function_context('_init_toolbar', 384, 4, False)
        # Assigning a type to the variable 'self' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2WebAgg._init_toolbar.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2WebAgg._init_toolbar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2WebAgg._init_toolbar.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2WebAgg._init_toolbar.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2WebAgg._init_toolbar')
        NavigationToolbar2WebAgg._init_toolbar.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2WebAgg._init_toolbar.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2WebAgg._init_toolbar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2WebAgg._init_toolbar.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2WebAgg._init_toolbar.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2WebAgg._init_toolbar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2WebAgg._init_toolbar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2WebAgg._init_toolbar', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 385):
        
        # Assigning a Str to a Attribute (line 385):
        unicode_262468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 23), 'unicode', u'')
        # Getting the type of 'self' (line 385)
        self_262469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'self')
        # Setting the type of the member 'message' of a type (line 385)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 8), self_262469, 'message', unicode_262468)
        
        # Assigning a Num to a Attribute (line 386):
        
        # Assigning a Num to a Attribute (line 386):
        int_262470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 22), 'int')
        # Getting the type of 'self' (line 386)
        self_262471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'self')
        # Setting the type of the member 'cursor' of a type (line 386)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 8), self_262471, 'cursor', int_262470)
        
        # ################# End of '_init_toolbar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_toolbar' in the type store
        # Getting the type of 'stypy_return_type' (line 384)
        stypy_return_type_262472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262472)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_toolbar'
        return stypy_return_type_262472


    @norecursion
    def set_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_message'
        module_type_store = module_type_store.open_function_context('set_message', 388, 4, False)
        # Assigning a type to the variable 'self' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2WebAgg.set_message.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2WebAgg.set_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2WebAgg.set_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2WebAgg.set_message.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2WebAgg.set_message')
        NavigationToolbar2WebAgg.set_message.__dict__.__setitem__('stypy_param_names_list', ['message'])
        NavigationToolbar2WebAgg.set_message.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2WebAgg.set_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2WebAgg.set_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2WebAgg.set_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2WebAgg.set_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2WebAgg.set_message.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2WebAgg.set_message', ['message'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_message', localization, ['message'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_message(...)' code ##################

        
        
        # Getting the type of 'message' (line 389)
        message_262473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), 'message')
        # Getting the type of 'self' (line 389)
        self_262474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 22), 'self')
        # Obtaining the member 'message' of a type (line 389)
        message_262475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 22), self_262474, 'message')
        # Applying the binary operator '!=' (line 389)
        result_ne_262476 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 11), '!=', message_262473, message_262475)
        
        # Testing the type of an if condition (line 389)
        if_condition_262477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 8), result_ne_262476)
        # Assigning a type to the variable 'if_condition_262477' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'if_condition_262477', if_condition_262477)
        # SSA begins for if statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to send_event(...): (line 390)
        # Processing the call arguments (line 390)
        unicode_262481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 35), 'unicode', u'message')
        # Processing the call keyword arguments (line 390)
        # Getting the type of 'message' (line 390)
        message_262482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 54), 'message', False)
        keyword_262483 = message_262482
        kwargs_262484 = {'message': keyword_262483}
        # Getting the type of 'self' (line 390)
        self_262478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'self', False)
        # Obtaining the member 'canvas' of a type (line 390)
        canvas_262479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), self_262478, 'canvas')
        # Obtaining the member 'send_event' of a type (line 390)
        send_event_262480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), canvas_262479, 'send_event')
        # Calling send_event(args, kwargs) (line 390)
        send_event_call_result_262485 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), send_event_262480, *[unicode_262481], **kwargs_262484)
        
        # SSA join for if statement (line 389)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 391):
        
        # Assigning a Name to a Attribute (line 391):
        # Getting the type of 'message' (line 391)
        message_262486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 23), 'message')
        # Getting the type of 'self' (line 391)
        self_262487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'self')
        # Setting the type of the member 'message' of a type (line 391)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), self_262487, 'message', message_262486)
        
        # ################# End of 'set_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_message' in the type store
        # Getting the type of 'stypy_return_type' (line 388)
        stypy_return_type_262488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262488)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_message'
        return stypy_return_type_262488


    @norecursion
    def set_cursor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_cursor'
        module_type_store = module_type_store.open_function_context('set_cursor', 393, 4, False)
        # Assigning a type to the variable 'self' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2WebAgg.set_cursor.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2WebAgg.set_cursor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2WebAgg.set_cursor.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2WebAgg.set_cursor.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2WebAgg.set_cursor')
        NavigationToolbar2WebAgg.set_cursor.__dict__.__setitem__('stypy_param_names_list', ['cursor'])
        NavigationToolbar2WebAgg.set_cursor.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2WebAgg.set_cursor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2WebAgg.set_cursor.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2WebAgg.set_cursor.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2WebAgg.set_cursor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2WebAgg.set_cursor.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2WebAgg.set_cursor', ['cursor'], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'cursor' (line 394)
        cursor_262489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 11), 'cursor')
        # Getting the type of 'self' (line 394)
        self_262490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 21), 'self')
        # Obtaining the member 'cursor' of a type (line 394)
        cursor_262491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 21), self_262490, 'cursor')
        # Applying the binary operator '!=' (line 394)
        result_ne_262492 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 11), '!=', cursor_262489, cursor_262491)
        
        # Testing the type of an if condition (line 394)
        if_condition_262493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 8), result_ne_262492)
        # Assigning a type to the variable 'if_condition_262493' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'if_condition_262493', if_condition_262493)
        # SSA begins for if statement (line 394)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to send_event(...): (line 395)
        # Processing the call arguments (line 395)
        unicode_262497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 35), 'unicode', u'cursor')
        # Processing the call keyword arguments (line 395)
        # Getting the type of 'cursor' (line 395)
        cursor_262498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 52), 'cursor', False)
        keyword_262499 = cursor_262498
        kwargs_262500 = {'cursor': keyword_262499}
        # Getting the type of 'self' (line 395)
        self_262494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'self', False)
        # Obtaining the member 'canvas' of a type (line 395)
        canvas_262495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), self_262494, 'canvas')
        # Obtaining the member 'send_event' of a type (line 395)
        send_event_262496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), canvas_262495, 'send_event')
        # Calling send_event(args, kwargs) (line 395)
        send_event_call_result_262501 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), send_event_262496, *[unicode_262497], **kwargs_262500)
        
        # SSA join for if statement (line 394)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 396):
        
        # Assigning a Name to a Attribute (line 396):
        # Getting the type of 'cursor' (line 396)
        cursor_262502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 22), 'cursor')
        # Getting the type of 'self' (line 396)
        self_262503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'self')
        # Setting the type of the member 'cursor' of a type (line 396)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), self_262503, 'cursor', cursor_262502)
        
        # ################# End of 'set_cursor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_cursor' in the type store
        # Getting the type of 'stypy_return_type' (line 393)
        stypy_return_type_262504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262504)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_cursor'
        return stypy_return_type_262504


    @norecursion
    def draw_rubberband(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'draw_rubberband'
        module_type_store = module_type_store.open_function_context('draw_rubberband', 398, 4, False)
        # Assigning a type to the variable 'self' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2WebAgg.draw_rubberband.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2WebAgg.draw_rubberband.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2WebAgg.draw_rubberband.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2WebAgg.draw_rubberband.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2WebAgg.draw_rubberband')
        NavigationToolbar2WebAgg.draw_rubberband.__dict__.__setitem__('stypy_param_names_list', ['event', 'x0', 'y0', 'x1', 'y1'])
        NavigationToolbar2WebAgg.draw_rubberband.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2WebAgg.draw_rubberband.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2WebAgg.draw_rubberband.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2WebAgg.draw_rubberband.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2WebAgg.draw_rubberband.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2WebAgg.draw_rubberband.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2WebAgg.draw_rubberband', ['event', 'x0', 'y0', 'x1', 'y1'], None, None, defaults, varargs, kwargs)

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

        
        # Call to send_event(...): (line 399)
        # Processing the call arguments (line 399)
        unicode_262508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 12), 'unicode', u'rubberband')
        # Processing the call keyword arguments (line 399)
        # Getting the type of 'x0' (line 400)
        x0_262509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 29), 'x0', False)
        keyword_262510 = x0_262509
        # Getting the type of 'y0' (line 400)
        y0_262511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 36), 'y0', False)
        keyword_262512 = y0_262511
        # Getting the type of 'x1' (line 400)
        x1_262513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 43), 'x1', False)
        keyword_262514 = x1_262513
        # Getting the type of 'y1' (line 400)
        y1_262515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 50), 'y1', False)
        keyword_262516 = y1_262515
        kwargs_262517 = {'y1': keyword_262516, 'y0': keyword_262512, 'x0': keyword_262510, 'x1': keyword_262514}
        # Getting the type of 'self' (line 399)
        self_262505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 399)
        canvas_262506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), self_262505, 'canvas')
        # Obtaining the member 'send_event' of a type (line 399)
        send_event_262507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), canvas_262506, 'send_event')
        # Calling send_event(args, kwargs) (line 399)
        send_event_call_result_262518 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), send_event_262507, *[unicode_262508], **kwargs_262517)
        
        
        # ################# End of 'draw_rubberband(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'draw_rubberband' in the type store
        # Getting the type of 'stypy_return_type' (line 398)
        stypy_return_type_262519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262519)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'draw_rubberband'
        return stypy_return_type_262519


    @norecursion
    def release_zoom(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'release_zoom'
        module_type_store = module_type_store.open_function_context('release_zoom', 402, 4, False)
        # Assigning a type to the variable 'self' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2WebAgg.release_zoom.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2WebAgg.release_zoom.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2WebAgg.release_zoom.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2WebAgg.release_zoom.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2WebAgg.release_zoom')
        NavigationToolbar2WebAgg.release_zoom.__dict__.__setitem__('stypy_param_names_list', ['event'])
        NavigationToolbar2WebAgg.release_zoom.__dict__.__setitem__('stypy_varargs_param_name', None)
        NavigationToolbar2WebAgg.release_zoom.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2WebAgg.release_zoom.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2WebAgg.release_zoom.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2WebAgg.release_zoom.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2WebAgg.release_zoom.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2WebAgg.release_zoom', ['event'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'release_zoom', localization, ['event'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'release_zoom(...)' code ##################

        
        # Call to release_zoom(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'self' (line 403)
        self_262523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 54), 'self', False)
        # Getting the type of 'event' (line 403)
        event_262524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 60), 'event', False)
        # Processing the call keyword arguments (line 403)
        kwargs_262525 = {}
        # Getting the type of 'backend_bases' (line 403)
        backend_bases_262520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'backend_bases', False)
        # Obtaining the member 'NavigationToolbar2' of a type (line 403)
        NavigationToolbar2_262521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), backend_bases_262520, 'NavigationToolbar2')
        # Obtaining the member 'release_zoom' of a type (line 403)
        release_zoom_262522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), NavigationToolbar2_262521, 'release_zoom')
        # Calling release_zoom(args, kwargs) (line 403)
        release_zoom_call_result_262526 = invoke(stypy.reporting.localization.Localization(__file__, 403, 8), release_zoom_262522, *[self_262523, event_262524], **kwargs_262525)
        
        
        # Call to send_event(...): (line 404)
        # Processing the call arguments (line 404)
        unicode_262530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 12), 'unicode', u'rubberband')
        # Processing the call keyword arguments (line 404)
        int_262531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 29), 'int')
        keyword_262532 = int_262531
        int_262533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 36), 'int')
        keyword_262534 = int_262533
        int_262535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 43), 'int')
        keyword_262536 = int_262535
        int_262537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 50), 'int')
        keyword_262538 = int_262537
        kwargs_262539 = {'y1': keyword_262538, 'y0': keyword_262534, 'x0': keyword_262532, 'x1': keyword_262536}
        # Getting the type of 'self' (line 404)
        self_262527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 404)
        canvas_262528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), self_262527, 'canvas')
        # Obtaining the member 'send_event' of a type (line 404)
        send_event_262529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), canvas_262528, 'send_event')
        # Calling send_event(args, kwargs) (line 404)
        send_event_call_result_262540 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), send_event_262529, *[unicode_262530], **kwargs_262539)
        
        
        # ################# End of 'release_zoom(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'release_zoom' in the type store
        # Getting the type of 'stypy_return_type' (line 402)
        stypy_return_type_262541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262541)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'release_zoom'
        return stypy_return_type_262541


    @norecursion
    def save_figure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'save_figure'
        module_type_store = module_type_store.open_function_context('save_figure', 407, 4, False)
        # Assigning a type to the variable 'self' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NavigationToolbar2WebAgg.save_figure.__dict__.__setitem__('stypy_localization', localization)
        NavigationToolbar2WebAgg.save_figure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NavigationToolbar2WebAgg.save_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        NavigationToolbar2WebAgg.save_figure.__dict__.__setitem__('stypy_function_name', 'NavigationToolbar2WebAgg.save_figure')
        NavigationToolbar2WebAgg.save_figure.__dict__.__setitem__('stypy_param_names_list', [])
        NavigationToolbar2WebAgg.save_figure.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        NavigationToolbar2WebAgg.save_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NavigationToolbar2WebAgg.save_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        NavigationToolbar2WebAgg.save_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        NavigationToolbar2WebAgg.save_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NavigationToolbar2WebAgg.save_figure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2WebAgg.save_figure', [], 'args', None, defaults, varargs, kwargs)

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

        unicode_262542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 8), 'unicode', u'Save the current figure')
        
        # Call to send_event(...): (line 409)
        # Processing the call arguments (line 409)
        unicode_262546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 31), 'unicode', u'save')
        # Processing the call keyword arguments (line 409)
        kwargs_262547 = {}
        # Getting the type of 'self' (line 409)
        self_262543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 409)
        canvas_262544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), self_262543, 'canvas')
        # Obtaining the member 'send_event' of a type (line 409)
        send_event_262545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), canvas_262544, 'send_event')
        # Calling send_event(args, kwargs) (line 409)
        send_event_call_result_262548 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), send_event_262545, *[unicode_262546], **kwargs_262547)
        
        
        # ################# End of 'save_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'save_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 407)
        stypy_return_type_262549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262549)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'save_figure'
        return stypy_return_type_262549


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 374, 0, False)
        # Assigning a type to the variable 'self' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationToolbar2WebAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NavigationToolbar2WebAgg' (line 374)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'NavigationToolbar2WebAgg', NavigationToolbar2WebAgg)

# Assigning a ListComp to a Name (line 377):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'backend_bases' (line 380)
backend_bases_262561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), 'backend_bases')
# Obtaining the member 'NavigationToolbar2' of a type (line 380)
NavigationToolbar2_262562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 21), backend_bases_262561, 'NavigationToolbar2')
# Obtaining the member 'toolitems' of a type (line 380)
toolitems_262563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 21), NavigationToolbar2_262562, 'toolitems')

# Obtaining an instance of the builtin type 'tuple' (line 381)
tuple_262564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 381)
# Adding element type (line 381)

# Obtaining an instance of the builtin type 'tuple' (line 381)
tuple_262565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 381)
# Adding element type (line 381)
unicode_262566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 23), 'unicode', u'Download')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_262565, unicode_262566)
# Adding element type (line 381)
unicode_262567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 35), 'unicode', u'Download plot')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_262565, unicode_262567)
# Adding element type (line 381)
unicode_262568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 52), 'unicode', u'download')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_262565, unicode_262568)
# Adding element type (line 381)
unicode_262569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 64), 'unicode', u'download')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 23), tuple_262565, unicode_262569)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 22), tuple_262564, tuple_262565)

# Applying the binary operator '+' (line 380)
result_add_262570 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 21), '+', toolitems_262563, tuple_262564)

comprehension_262571 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 17), result_add_262570)
# Assigning a type to the variable 'text' (line 377)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 17), 'text', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 17), comprehension_262571))
# Assigning a type to the variable 'tooltip_text' (line 377)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 17), 'tooltip_text', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 17), comprehension_262571))
# Assigning a type to the variable 'image_file' (line 377)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 17), 'image_file', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 17), comprehension_262571))
# Assigning a type to the variable 'name_of_method' (line 377)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 17), 'name_of_method', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 17), comprehension_262571))

# Getting the type of 'image_file' (line 382)
image_file_262558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 20), 'image_file')
# Getting the type of '_JQUERY_ICON_CLASSES' (line 382)
_JQUERY_ICON_CLASSES_262559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 34), '_JQUERY_ICON_CLASSES')
# Applying the binary operator 'in' (line 382)
result_contains_262560 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 20), 'in', image_file_262558, _JQUERY_ICON_CLASSES_262559)


# Obtaining an instance of the builtin type 'tuple' (line 377)
tuple_262550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 377)
# Adding element type (line 377)
# Getting the type of 'text' (line 377)
text_262551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 18), 'text')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 18), tuple_262550, text_262551)
# Adding element type (line 377)
# Getting the type of 'tooltip_text' (line 377)
tooltip_text_262552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 24), 'tooltip_text')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 18), tuple_262550, tooltip_text_262552)
# Adding element type (line 377)

# Obtaining the type of the subscript
# Getting the type of 'image_file' (line 377)
image_file_262553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 59), 'image_file')
# Getting the type of '_JQUERY_ICON_CLASSES' (line 377)
_JQUERY_ICON_CLASSES_262554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 38), '_JQUERY_ICON_CLASSES')
# Obtaining the member '__getitem__' of a type (line 377)
getitem___262555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 38), _JQUERY_ICON_CLASSES_262554, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 377)
subscript_call_result_262556 = invoke(stypy.reporting.localization.Localization(__file__, 377, 38), getitem___262555, image_file_262553)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 18), tuple_262550, subscript_call_result_262556)
# Adding element type (line 377)
# Getting the type of 'name_of_method' (line 378)
name_of_method_262557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 18), 'name_of_method')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 18), tuple_262550, name_of_method_262557)

list_262572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 17), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 17), list_262572, tuple_262550)
# Getting the type of 'NavigationToolbar2WebAgg'
NavigationToolbar2WebAgg_262573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NavigationToolbar2WebAgg')
# Setting the type of the member 'toolitems' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NavigationToolbar2WebAgg_262573, 'toolitems', list_262572)
# Declaration of the 'FigureManagerWebAgg' class
# Getting the type of 'backend_bases' (line 412)
backend_bases_262574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 26), 'backend_bases')
# Obtaining the member 'FigureManagerBase' of a type (line 412)
FigureManagerBase_262575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 26), backend_bases_262574, 'FigureManagerBase')

class FigureManagerWebAgg(FigureManagerBase_262575, ):
    
    # Assigning a Name to a Name (line 413):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 415, 4, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg.__init__', ['canvas', 'num'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'self' (line 416)
        self_262579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 49), 'self', False)
        # Getting the type of 'canvas' (line 416)
        canvas_262580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 55), 'canvas', False)
        # Getting the type of 'num' (line 416)
        num_262581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 63), 'num', False)
        # Processing the call keyword arguments (line 416)
        kwargs_262582 = {}
        # Getting the type of 'backend_bases' (line 416)
        backend_bases_262576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'backend_bases', False)
        # Obtaining the member 'FigureManagerBase' of a type (line 416)
        FigureManagerBase_262577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), backend_bases_262576, 'FigureManagerBase')
        # Obtaining the member '__init__' of a type (line 416)
        init___262578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), FigureManagerBase_262577, '__init__')
        # Calling __init__(args, kwargs) (line 416)
        init___call_result_262583 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), init___262578, *[self_262579, canvas_262580, num_262581], **kwargs_262582)
        
        
        # Assigning a Call to a Attribute (line 418):
        
        # Assigning a Call to a Attribute (line 418):
        
        # Call to set(...): (line 418)
        # Processing the call keyword arguments (line 418)
        kwargs_262585 = {}
        # Getting the type of 'set' (line 418)
        set_262584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 27), 'set', False)
        # Calling set(args, kwargs) (line 418)
        set_call_result_262586 = invoke(stypy.reporting.localization.Localization(__file__, 418, 27), set_262584, *[], **kwargs_262585)
        
        # Getting the type of 'self' (line 418)
        self_262587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'self')
        # Setting the type of the member 'web_sockets' of a type (line 418)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), self_262587, 'web_sockets', set_call_result_262586)
        
        # Assigning a Call to a Attribute (line 420):
        
        # Assigning a Call to a Attribute (line 420):
        
        # Call to _get_toolbar(...): (line 420)
        # Processing the call arguments (line 420)
        # Getting the type of 'canvas' (line 420)
        canvas_262590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 41), 'canvas', False)
        # Processing the call keyword arguments (line 420)
        kwargs_262591 = {}
        # Getting the type of 'self' (line 420)
        self_262588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 23), 'self', False)
        # Obtaining the member '_get_toolbar' of a type (line 420)
        _get_toolbar_262589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 23), self_262588, '_get_toolbar')
        # Calling _get_toolbar(args, kwargs) (line 420)
        _get_toolbar_call_result_262592 = invoke(stypy.reporting.localization.Localization(__file__, 420, 23), _get_toolbar_262589, *[canvas_262590], **kwargs_262591)
        
        # Getting the type of 'self' (line 420)
        self_262593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'self')
        # Setting the type of the member 'toolbar' of a type (line 420)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 8), self_262593, 'toolbar', _get_toolbar_call_result_262592)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def show(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'show'
        module_type_store = module_type_store.open_function_context('show', 422, 4, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerWebAgg.show.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerWebAgg.show.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerWebAgg.show.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerWebAgg.show.__dict__.__setitem__('stypy_function_name', 'FigureManagerWebAgg.show')
        FigureManagerWebAgg.show.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerWebAgg.show.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerWebAgg.show.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerWebAgg.show.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerWebAgg.show.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerWebAgg.show.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerWebAgg.show.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg.show', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of 'show(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'show' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_262594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262594)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'show'
        return stypy_return_type_262594


    @norecursion
    def _get_toolbar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_toolbar'
        module_type_store = module_type_store.open_function_context('_get_toolbar', 425, 4, False)
        # Assigning a type to the variable 'self' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerWebAgg._get_toolbar.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerWebAgg._get_toolbar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerWebAgg._get_toolbar.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerWebAgg._get_toolbar.__dict__.__setitem__('stypy_function_name', 'FigureManagerWebAgg._get_toolbar')
        FigureManagerWebAgg._get_toolbar.__dict__.__setitem__('stypy_param_names_list', ['canvas'])
        FigureManagerWebAgg._get_toolbar.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerWebAgg._get_toolbar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerWebAgg._get_toolbar.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerWebAgg._get_toolbar.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerWebAgg._get_toolbar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerWebAgg._get_toolbar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg._get_toolbar', ['canvas'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_toolbar', localization, ['canvas'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_toolbar(...)' code ##################

        
        # Assigning a Call to a Name (line 426):
        
        # Assigning a Call to a Name (line 426):
        
        # Call to ToolbarCls(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'canvas' (line 426)
        canvas_262597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 34), 'canvas', False)
        # Processing the call keyword arguments (line 426)
        kwargs_262598 = {}
        # Getting the type of 'self' (line 426)
        self_262595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 18), 'self', False)
        # Obtaining the member 'ToolbarCls' of a type (line 426)
        ToolbarCls_262596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 18), self_262595, 'ToolbarCls')
        # Calling ToolbarCls(args, kwargs) (line 426)
        ToolbarCls_call_result_262599 = invoke(stypy.reporting.localization.Localization(__file__, 426, 18), ToolbarCls_262596, *[canvas_262597], **kwargs_262598)
        
        # Assigning a type to the variable 'toolbar' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'toolbar', ToolbarCls_call_result_262599)
        # Getting the type of 'toolbar' (line 427)
        toolbar_262600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 15), 'toolbar')
        # Assigning a type to the variable 'stypy_return_type' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'stypy_return_type', toolbar_262600)
        
        # ################# End of '_get_toolbar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_toolbar' in the type store
        # Getting the type of 'stypy_return_type' (line 425)
        stypy_return_type_262601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262601)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_toolbar'
        return stypy_return_type_262601


    @norecursion
    def resize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'resize'
        module_type_store = module_type_store.open_function_context('resize', 429, 4, False)
        # Assigning a type to the variable 'self' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerWebAgg.resize.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerWebAgg.resize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerWebAgg.resize.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerWebAgg.resize.__dict__.__setitem__('stypy_function_name', 'FigureManagerWebAgg.resize')
        FigureManagerWebAgg.resize.__dict__.__setitem__('stypy_param_names_list', ['w', 'h'])
        FigureManagerWebAgg.resize.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerWebAgg.resize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerWebAgg.resize.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerWebAgg.resize.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerWebAgg.resize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerWebAgg.resize.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg.resize', ['w', 'h'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'resize', localization, ['w', 'h'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'resize(...)' code ##################

        
        # Call to _send_event(...): (line 430)
        # Processing the call arguments (line 430)
        unicode_262604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 12), 'unicode', u'resize')
        # Processing the call keyword arguments (line 430)
        
        # Obtaining an instance of the builtin type 'tuple' (line 432)
        tuple_262605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 432)
        # Adding element type (line 432)
        # Getting the type of 'w' (line 432)
        w_262606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 18), 'w', False)
        # Getting the type of 'self' (line 432)
        self_262607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 22), 'self', False)
        # Obtaining the member 'canvas' of a type (line 432)
        canvas_262608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 22), self_262607, 'canvas')
        # Obtaining the member '_dpi_ratio' of a type (line 432)
        _dpi_ratio_262609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 22), canvas_262608, '_dpi_ratio')
        # Applying the binary operator 'div' (line 432)
        result_div_262610 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 18), 'div', w_262606, _dpi_ratio_262609)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 18), tuple_262605, result_div_262610)
        # Adding element type (line 432)
        # Getting the type of 'h' (line 432)
        h_262611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 46), 'h', False)
        # Getting the type of 'self' (line 432)
        self_262612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 50), 'self', False)
        # Obtaining the member 'canvas' of a type (line 432)
        canvas_262613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 50), self_262612, 'canvas')
        # Obtaining the member '_dpi_ratio' of a type (line 432)
        _dpi_ratio_262614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 50), canvas_262613, '_dpi_ratio')
        # Applying the binary operator 'div' (line 432)
        result_div_262615 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 46), 'div', h_262611, _dpi_ratio_262614)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 18), tuple_262605, result_div_262615)
        
        keyword_262616 = tuple_262605
        kwargs_262617 = {'size': keyword_262616}
        # Getting the type of 'self' (line 430)
        self_262602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'self', False)
        # Obtaining the member '_send_event' of a type (line 430)
        _send_event_262603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), self_262602, '_send_event')
        # Calling _send_event(args, kwargs) (line 430)
        _send_event_call_result_262618 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), _send_event_262603, *[unicode_262604], **kwargs_262617)
        
        
        # ################# End of 'resize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'resize' in the type store
        # Getting the type of 'stypy_return_type' (line 429)
        stypy_return_type_262619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262619)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'resize'
        return stypy_return_type_262619


    @norecursion
    def set_window_title(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_window_title'
        module_type_store = module_type_store.open_function_context('set_window_title', 434, 4, False)
        # Assigning a type to the variable 'self' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerWebAgg.set_window_title.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerWebAgg.set_window_title.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerWebAgg.set_window_title.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerWebAgg.set_window_title.__dict__.__setitem__('stypy_function_name', 'FigureManagerWebAgg.set_window_title')
        FigureManagerWebAgg.set_window_title.__dict__.__setitem__('stypy_param_names_list', ['title'])
        FigureManagerWebAgg.set_window_title.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerWebAgg.set_window_title.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerWebAgg.set_window_title.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerWebAgg.set_window_title.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerWebAgg.set_window_title.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerWebAgg.set_window_title.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg.set_window_title', ['title'], None, None, defaults, varargs, kwargs)

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

        
        # Call to _send_event(...): (line 435)
        # Processing the call arguments (line 435)
        unicode_262622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 25), 'unicode', u'figure_label')
        # Processing the call keyword arguments (line 435)
        # Getting the type of 'title' (line 435)
        title_262623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 47), 'title', False)
        keyword_262624 = title_262623
        kwargs_262625 = {'label': keyword_262624}
        # Getting the type of 'self' (line 435)
        self_262620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'self', False)
        # Obtaining the member '_send_event' of a type (line 435)
        _send_event_262621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 8), self_262620, '_send_event')
        # Calling _send_event(args, kwargs) (line 435)
        _send_event_call_result_262626 = invoke(stypy.reporting.localization.Localization(__file__, 435, 8), _send_event_262621, *[unicode_262622], **kwargs_262625)
        
        
        # ################# End of 'set_window_title(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_window_title' in the type store
        # Getting the type of 'stypy_return_type' (line 434)
        stypy_return_type_262627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262627)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_window_title'
        return stypy_return_type_262627


    @norecursion
    def add_web_socket(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_web_socket'
        module_type_store = module_type_store.open_function_context('add_web_socket', 439, 4, False)
        # Assigning a type to the variable 'self' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerWebAgg.add_web_socket.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerWebAgg.add_web_socket.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerWebAgg.add_web_socket.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerWebAgg.add_web_socket.__dict__.__setitem__('stypy_function_name', 'FigureManagerWebAgg.add_web_socket')
        FigureManagerWebAgg.add_web_socket.__dict__.__setitem__('stypy_param_names_list', ['web_socket'])
        FigureManagerWebAgg.add_web_socket.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerWebAgg.add_web_socket.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerWebAgg.add_web_socket.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerWebAgg.add_web_socket.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerWebAgg.add_web_socket.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerWebAgg.add_web_socket.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg.add_web_socket', ['web_socket'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_web_socket', localization, ['web_socket'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_web_socket(...)' code ##################

        # Evaluating assert statement condition
        
        # Call to hasattr(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'web_socket' (line 440)
        web_socket_262629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 23), 'web_socket', False)
        unicode_262630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 35), 'unicode', u'send_binary')
        # Processing the call keyword arguments (line 440)
        kwargs_262631 = {}
        # Getting the type of 'hasattr' (line 440)
        hasattr_262628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 440)
        hasattr_call_result_262632 = invoke(stypy.reporting.localization.Localization(__file__, 440, 15), hasattr_262628, *[web_socket_262629, unicode_262630], **kwargs_262631)
        
        # Evaluating assert statement condition
        
        # Call to hasattr(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'web_socket' (line 441)
        web_socket_262634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 23), 'web_socket', False)
        unicode_262635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 35), 'unicode', u'send_json')
        # Processing the call keyword arguments (line 441)
        kwargs_262636 = {}
        # Getting the type of 'hasattr' (line 441)
        hasattr_262633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 441)
        hasattr_call_result_262637 = invoke(stypy.reporting.localization.Localization(__file__, 441, 15), hasattr_262633, *[web_socket_262634, unicode_262635], **kwargs_262636)
        
        
        # Call to add(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'web_socket' (line 443)
        web_socket_262641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 29), 'web_socket', False)
        # Processing the call keyword arguments (line 443)
        kwargs_262642 = {}
        # Getting the type of 'self' (line 443)
        self_262638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'self', False)
        # Obtaining the member 'web_sockets' of a type (line 443)
        web_sockets_262639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), self_262638, 'web_sockets')
        # Obtaining the member 'add' of a type (line 443)
        add_262640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), web_sockets_262639, 'add')
        # Calling add(args, kwargs) (line 443)
        add_call_result_262643 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), add_262640, *[web_socket_262641], **kwargs_262642)
        
        
        # Assigning a Attribute to a Tuple (line 445):
        
        # Assigning a Subscript to a Name (line 445):
        
        # Obtaining the type of the subscript
        int_262644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 8), 'int')
        # Getting the type of 'self' (line 445)
        self_262645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'self')
        # Obtaining the member 'canvas' of a type (line 445)
        canvas_262646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), self_262645, 'canvas')
        # Obtaining the member 'figure' of a type (line 445)
        figure_262647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), canvas_262646, 'figure')
        # Obtaining the member 'bbox' of a type (line 445)
        bbox_262648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), figure_262647, 'bbox')
        # Obtaining the member 'bounds' of a type (line 445)
        bounds_262649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), bbox_262648, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___262650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), bounds_262649, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_262651 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), getitem___262650, int_262644)
        
        # Assigning a type to the variable 'tuple_var_assignment_261464' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_261464', subscript_call_result_262651)
        
        # Assigning a Subscript to a Name (line 445):
        
        # Obtaining the type of the subscript
        int_262652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 8), 'int')
        # Getting the type of 'self' (line 445)
        self_262653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'self')
        # Obtaining the member 'canvas' of a type (line 445)
        canvas_262654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), self_262653, 'canvas')
        # Obtaining the member 'figure' of a type (line 445)
        figure_262655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), canvas_262654, 'figure')
        # Obtaining the member 'bbox' of a type (line 445)
        bbox_262656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), figure_262655, 'bbox')
        # Obtaining the member 'bounds' of a type (line 445)
        bounds_262657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), bbox_262656, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___262658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), bounds_262657, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_262659 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), getitem___262658, int_262652)
        
        # Assigning a type to the variable 'tuple_var_assignment_261465' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_261465', subscript_call_result_262659)
        
        # Assigning a Subscript to a Name (line 445):
        
        # Obtaining the type of the subscript
        int_262660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 8), 'int')
        # Getting the type of 'self' (line 445)
        self_262661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'self')
        # Obtaining the member 'canvas' of a type (line 445)
        canvas_262662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), self_262661, 'canvas')
        # Obtaining the member 'figure' of a type (line 445)
        figure_262663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), canvas_262662, 'figure')
        # Obtaining the member 'bbox' of a type (line 445)
        bbox_262664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), figure_262663, 'bbox')
        # Obtaining the member 'bounds' of a type (line 445)
        bounds_262665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), bbox_262664, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___262666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), bounds_262665, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_262667 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), getitem___262666, int_262660)
        
        # Assigning a type to the variable 'tuple_var_assignment_261466' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_261466', subscript_call_result_262667)
        
        # Assigning a Subscript to a Name (line 445):
        
        # Obtaining the type of the subscript
        int_262668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 8), 'int')
        # Getting the type of 'self' (line 445)
        self_262669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'self')
        # Obtaining the member 'canvas' of a type (line 445)
        canvas_262670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), self_262669, 'canvas')
        # Obtaining the member 'figure' of a type (line 445)
        figure_262671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), canvas_262670, 'figure')
        # Obtaining the member 'bbox' of a type (line 445)
        bbox_262672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), figure_262671, 'bbox')
        # Obtaining the member 'bounds' of a type (line 445)
        bounds_262673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), bbox_262672, 'bounds')
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___262674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), bounds_262673, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_262675 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), getitem___262674, int_262668)
        
        # Assigning a type to the variable 'tuple_var_assignment_261467' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_261467', subscript_call_result_262675)
        
        # Assigning a Name to a Name (line 445):
        # Getting the type of 'tuple_var_assignment_261464' (line 445)
        tuple_var_assignment_261464_262676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_261464')
        # Assigning a type to the variable '_' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), '_', tuple_var_assignment_261464_262676)
        
        # Assigning a Name to a Name (line 445):
        # Getting the type of 'tuple_var_assignment_261465' (line 445)
        tuple_var_assignment_261465_262677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_261465')
        # Assigning a type to the variable '_' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 11), '_', tuple_var_assignment_261465_262677)
        
        # Assigning a Name to a Name (line 445):
        # Getting the type of 'tuple_var_assignment_261466' (line 445)
        tuple_var_assignment_261466_262678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_261466')
        # Assigning a type to the variable 'w' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 14), 'w', tuple_var_assignment_261466_262678)
        
        # Assigning a Name to a Name (line 445):
        # Getting the type of 'tuple_var_assignment_261467' (line 445)
        tuple_var_assignment_261467_262679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_261467')
        # Assigning a type to the variable 'h' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 17), 'h', tuple_var_assignment_261467_262679)
        
        # Call to resize(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 'w' (line 446)
        w_262682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 20), 'w', False)
        # Getting the type of 'h' (line 446)
        h_262683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 23), 'h', False)
        # Processing the call keyword arguments (line 446)
        kwargs_262684 = {}
        # Getting the type of 'self' (line 446)
        self_262680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'self', False)
        # Obtaining the member 'resize' of a type (line 446)
        resize_262681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), self_262680, 'resize')
        # Calling resize(args, kwargs) (line 446)
        resize_call_result_262685 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), resize_262681, *[w_262682, h_262683], **kwargs_262684)
        
        
        # Call to _send_event(...): (line 447)
        # Processing the call arguments (line 447)
        unicode_262688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 25), 'unicode', u'refresh')
        # Processing the call keyword arguments (line 447)
        kwargs_262689 = {}
        # Getting the type of 'self' (line 447)
        self_262686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'self', False)
        # Obtaining the member '_send_event' of a type (line 447)
        _send_event_262687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 8), self_262686, '_send_event')
        # Calling _send_event(args, kwargs) (line 447)
        _send_event_call_result_262690 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), _send_event_262687, *[unicode_262688], **kwargs_262689)
        
        
        # ################# End of 'add_web_socket(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_web_socket' in the type store
        # Getting the type of 'stypy_return_type' (line 439)
        stypy_return_type_262691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262691)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_web_socket'
        return stypy_return_type_262691


    @norecursion
    def remove_web_socket(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove_web_socket'
        module_type_store = module_type_store.open_function_context('remove_web_socket', 449, 4, False)
        # Assigning a type to the variable 'self' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerWebAgg.remove_web_socket.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerWebAgg.remove_web_socket.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerWebAgg.remove_web_socket.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerWebAgg.remove_web_socket.__dict__.__setitem__('stypy_function_name', 'FigureManagerWebAgg.remove_web_socket')
        FigureManagerWebAgg.remove_web_socket.__dict__.__setitem__('stypy_param_names_list', ['web_socket'])
        FigureManagerWebAgg.remove_web_socket.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerWebAgg.remove_web_socket.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerWebAgg.remove_web_socket.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerWebAgg.remove_web_socket.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerWebAgg.remove_web_socket.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerWebAgg.remove_web_socket.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg.remove_web_socket', ['web_socket'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove_web_socket', localization, ['web_socket'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove_web_socket(...)' code ##################

        
        # Call to remove(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 'web_socket' (line 450)
        web_socket_262695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 32), 'web_socket', False)
        # Processing the call keyword arguments (line 450)
        kwargs_262696 = {}
        # Getting the type of 'self' (line 450)
        self_262692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'self', False)
        # Obtaining the member 'web_sockets' of a type (line 450)
        web_sockets_262693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), self_262692, 'web_sockets')
        # Obtaining the member 'remove' of a type (line 450)
        remove_262694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 8), web_sockets_262693, 'remove')
        # Calling remove(args, kwargs) (line 450)
        remove_call_result_262697 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), remove_262694, *[web_socket_262695], **kwargs_262696)
        
        
        # ################# End of 'remove_web_socket(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove_web_socket' in the type store
        # Getting the type of 'stypy_return_type' (line 449)
        stypy_return_type_262698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262698)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove_web_socket'
        return stypy_return_type_262698


    @norecursion
    def handle_json(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'handle_json'
        module_type_store = module_type_store.open_function_context('handle_json', 452, 4, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerWebAgg.handle_json.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerWebAgg.handle_json.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerWebAgg.handle_json.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerWebAgg.handle_json.__dict__.__setitem__('stypy_function_name', 'FigureManagerWebAgg.handle_json')
        FigureManagerWebAgg.handle_json.__dict__.__setitem__('stypy_param_names_list', ['content'])
        FigureManagerWebAgg.handle_json.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerWebAgg.handle_json.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerWebAgg.handle_json.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerWebAgg.handle_json.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerWebAgg.handle_json.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerWebAgg.handle_json.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg.handle_json', ['content'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'handle_json', localization, ['content'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'handle_json(...)' code ##################

        
        # Call to handle_event(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'content' (line 453)
        content_262702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 33), 'content', False)
        # Processing the call keyword arguments (line 453)
        kwargs_262703 = {}
        # Getting the type of 'self' (line 453)
        self_262699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'self', False)
        # Obtaining the member 'canvas' of a type (line 453)
        canvas_262700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 8), self_262699, 'canvas')
        # Obtaining the member 'handle_event' of a type (line 453)
        handle_event_262701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 8), canvas_262700, 'handle_event')
        # Calling handle_event(args, kwargs) (line 453)
        handle_event_call_result_262704 = invoke(stypy.reporting.localization.Localization(__file__, 453, 8), handle_event_262701, *[content_262702], **kwargs_262703)
        
        
        # ################# End of 'handle_json(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'handle_json' in the type store
        # Getting the type of 'stypy_return_type' (line 452)
        stypy_return_type_262705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262705)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'handle_json'
        return stypy_return_type_262705


    @norecursion
    def refresh_all(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'refresh_all'
        module_type_store = module_type_store.open_function_context('refresh_all', 455, 4, False)
        # Assigning a type to the variable 'self' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerWebAgg.refresh_all.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerWebAgg.refresh_all.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerWebAgg.refresh_all.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerWebAgg.refresh_all.__dict__.__setitem__('stypy_function_name', 'FigureManagerWebAgg.refresh_all')
        FigureManagerWebAgg.refresh_all.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerWebAgg.refresh_all.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerWebAgg.refresh_all.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerWebAgg.refresh_all.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerWebAgg.refresh_all.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerWebAgg.refresh_all.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerWebAgg.refresh_all.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg.refresh_all', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'refresh_all', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'refresh_all(...)' code ##################

        
        # Getting the type of 'self' (line 456)
        self_262706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 11), 'self')
        # Obtaining the member 'web_sockets' of a type (line 456)
        web_sockets_262707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 11), self_262706, 'web_sockets')
        # Testing the type of an if condition (line 456)
        if_condition_262708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 8), web_sockets_262707)
        # Assigning a type to the variable 'if_condition_262708' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'if_condition_262708', if_condition_262708)
        # SSA begins for if statement (line 456)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 457):
        
        # Assigning a Call to a Name (line 457):
        
        # Call to get_diff_image(...): (line 457)
        # Processing the call keyword arguments (line 457)
        kwargs_262712 = {}
        # Getting the type of 'self' (line 457)
        self_262709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 19), 'self', False)
        # Obtaining the member 'canvas' of a type (line 457)
        canvas_262710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 19), self_262709, 'canvas')
        # Obtaining the member 'get_diff_image' of a type (line 457)
        get_diff_image_262711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 19), canvas_262710, 'get_diff_image')
        # Calling get_diff_image(args, kwargs) (line 457)
        get_diff_image_call_result_262713 = invoke(stypy.reporting.localization.Localization(__file__, 457, 19), get_diff_image_262711, *[], **kwargs_262712)
        
        # Assigning a type to the variable 'diff' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'diff', get_diff_image_call_result_262713)
        
        # Type idiom detected: calculating its left and rigth part (line 458)
        # Getting the type of 'diff' (line 458)
        diff_262714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'diff')
        # Getting the type of 'None' (line 458)
        None_262715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 27), 'None')
        
        (may_be_262716, more_types_in_union_262717) = may_not_be_none(diff_262714, None_262715)

        if may_be_262716:

            if more_types_in_union_262717:
                # Runtime conditional SSA (line 458)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'self' (line 459)
            self_262718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 25), 'self')
            # Obtaining the member 'web_sockets' of a type (line 459)
            web_sockets_262719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 25), self_262718, 'web_sockets')
            # Testing the type of a for loop iterable (line 459)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 459, 16), web_sockets_262719)
            # Getting the type of the for loop variable (line 459)
            for_loop_var_262720 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 459, 16), web_sockets_262719)
            # Assigning a type to the variable 's' (line 459)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 16), 's', for_loop_var_262720)
            # SSA begins for a for statement (line 459)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to send_binary(...): (line 460)
            # Processing the call arguments (line 460)
            # Getting the type of 'diff' (line 460)
            diff_262723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 34), 'diff', False)
            # Processing the call keyword arguments (line 460)
            kwargs_262724 = {}
            # Getting the type of 's' (line 460)
            s_262721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 20), 's', False)
            # Obtaining the member 'send_binary' of a type (line 460)
            send_binary_262722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 20), s_262721, 'send_binary')
            # Calling send_binary(args, kwargs) (line 460)
            send_binary_call_result_262725 = invoke(stypy.reporting.localization.Localization(__file__, 460, 20), send_binary_262722, *[diff_262723], **kwargs_262724)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_262717:
                # SSA join for if statement (line 458)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 456)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'refresh_all(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'refresh_all' in the type store
        # Getting the type of 'stypy_return_type' (line 455)
        stypy_return_type_262726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262726)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'refresh_all'
        return stypy_return_type_262726


    @norecursion
    def get_javascript(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 463)
        None_262727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 35), 'None')
        defaults = [None_262727]
        # Create a new context for function 'get_javascript'
        module_type_store = module_type_store.open_function_context('get_javascript', 462, 4, False)
        # Assigning a type to the variable 'self' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerWebAgg.get_javascript.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerWebAgg.get_javascript.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerWebAgg.get_javascript.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerWebAgg.get_javascript.__dict__.__setitem__('stypy_function_name', 'FigureManagerWebAgg.get_javascript')
        FigureManagerWebAgg.get_javascript.__dict__.__setitem__('stypy_param_names_list', ['stream'])
        FigureManagerWebAgg.get_javascript.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerWebAgg.get_javascript.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerWebAgg.get_javascript.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerWebAgg.get_javascript.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerWebAgg.get_javascript.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerWebAgg.get_javascript.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg.get_javascript', ['stream'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_javascript', localization, ['stream'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_javascript(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 464)
        # Getting the type of 'stream' (line 464)
        stream_262728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 11), 'stream')
        # Getting the type of 'None' (line 464)
        None_262729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 21), 'None')
        
        (may_be_262730, more_types_in_union_262731) = may_be_none(stream_262728, None_262729)

        if may_be_262730:

            if more_types_in_union_262731:
                # Runtime conditional SSA (line 464)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 465):
            
            # Assigning a Call to a Name (line 465):
            
            # Call to StringIO(...): (line 465)
            # Processing the call keyword arguments (line 465)
            kwargs_262734 = {}
            # Getting the type of 'io' (line 465)
            io_262732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 21), 'io', False)
            # Obtaining the member 'StringIO' of a type (line 465)
            StringIO_262733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 21), io_262732, 'StringIO')
            # Calling StringIO(args, kwargs) (line 465)
            StringIO_call_result_262735 = invoke(stypy.reporting.localization.Localization(__file__, 465, 21), StringIO_262733, *[], **kwargs_262734)
            
            # Assigning a type to the variable 'output' (line 465)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'output', StringIO_call_result_262735)

            if more_types_in_union_262731:
                # Runtime conditional SSA for else branch (line 464)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_262730) or more_types_in_union_262731):
            
            # Assigning a Name to a Name (line 467):
            
            # Assigning a Name to a Name (line 467):
            # Getting the type of 'stream' (line 467)
            stream_262736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 21), 'stream')
            # Assigning a type to the variable 'output' (line 467)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'output', stream_262736)

            if (may_be_262730 and more_types_in_union_262731):
                # SSA join for if statement (line 464)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to open(...): (line 469)
        # Processing the call arguments (line 469)
        
        # Call to join(...): (line 469)
        # Processing the call arguments (line 469)
        
        # Call to dirname(...): (line 470)
        # Processing the call arguments (line 470)
        # Getting the type of '__file__' (line 470)
        file___262745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 32), '__file__', False)
        # Processing the call keyword arguments (line 470)
        kwargs_262746 = {}
        # Getting the type of 'os' (line 470)
        os_262742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 470)
        path_262743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 16), os_262742, 'path')
        # Obtaining the member 'dirname' of a type (line 470)
        dirname_262744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 16), path_262743, 'dirname')
        # Calling dirname(args, kwargs) (line 470)
        dirname_call_result_262747 = invoke(stypy.reporting.localization.Localization(__file__, 470, 16), dirname_262744, *[file___262745], **kwargs_262746)
        
        unicode_262748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 16), 'unicode', u'web_backend')
        unicode_262749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 16), 'unicode', u'mpl.js')
        # Processing the call keyword arguments (line 469)
        kwargs_262750 = {}
        # Getting the type of 'os' (line 469)
        os_262739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 469)
        path_262740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 21), os_262739, 'path')
        # Obtaining the member 'join' of a type (line 469)
        join_262741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 21), path_262740, 'join')
        # Calling join(args, kwargs) (line 469)
        join_call_result_262751 = invoke(stypy.reporting.localization.Localization(__file__, 469, 21), join_262741, *[dirname_call_result_262747, unicode_262748, unicode_262749], **kwargs_262750)
        
        # Processing the call keyword arguments (line 469)
        unicode_262752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 36), 'unicode', u'utf8')
        keyword_262753 = unicode_262752
        kwargs_262754 = {'encoding': keyword_262753}
        # Getting the type of 'io' (line 469)
        io_262737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 13), 'io', False)
        # Obtaining the member 'open' of a type (line 469)
        open_262738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 13), io_262737, 'open')
        # Calling open(args, kwargs) (line 469)
        open_call_result_262755 = invoke(stypy.reporting.localization.Localization(__file__, 469, 13), open_262738, *[join_call_result_262751], **kwargs_262754)
        
        with_262756 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 469, 13), open_call_result_262755, 'with parameter', '__enter__', '__exit__')

        if with_262756:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 469)
            enter___262757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 13), open_call_result_262755, '__enter__')
            with_enter_262758 = invoke(stypy.reporting.localization.Localization(__file__, 469, 13), enter___262757)
            # Assigning a type to the variable 'fd' (line 469)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 13), 'fd', with_enter_262758)
            
            # Call to write(...): (line 473)
            # Processing the call arguments (line 473)
            
            # Call to read(...): (line 473)
            # Processing the call keyword arguments (line 473)
            kwargs_262763 = {}
            # Getting the type of 'fd' (line 473)
            fd_262761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 25), 'fd', False)
            # Obtaining the member 'read' of a type (line 473)
            read_262762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 25), fd_262761, 'read')
            # Calling read(args, kwargs) (line 473)
            read_call_result_262764 = invoke(stypy.reporting.localization.Localization(__file__, 473, 25), read_262762, *[], **kwargs_262763)
            
            # Processing the call keyword arguments (line 473)
            kwargs_262765 = {}
            # Getting the type of 'output' (line 473)
            output_262759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'output', False)
            # Obtaining the member 'write' of a type (line 473)
            write_262760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 12), output_262759, 'write')
            # Calling write(args, kwargs) (line 473)
            write_call_result_262766 = invoke(stypy.reporting.localization.Localization(__file__, 473, 12), write_262760, *[read_call_result_262764], **kwargs_262765)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 469)
            exit___262767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 13), open_call_result_262755, '__exit__')
            with_exit_262768 = invoke(stypy.reporting.localization.Localization(__file__, 469, 13), exit___262767, None, None, None)

        
        # Assigning a List to a Name (line 475):
        
        # Assigning a List to a Name (line 475):
        
        # Obtaining an instance of the builtin type 'list' (line 475)
        list_262769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 475)
        
        # Assigning a type to the variable 'toolitems' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'toolitems', list_262769)
        
        # Getting the type of 'cls' (line 476)
        cls_262770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 44), 'cls')
        # Obtaining the member 'ToolbarCls' of a type (line 476)
        ToolbarCls_262771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 44), cls_262770, 'ToolbarCls')
        # Obtaining the member 'toolitems' of a type (line 476)
        toolitems_262772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 44), ToolbarCls_262771, 'toolitems')
        # Testing the type of a for loop iterable (line 476)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 476, 8), toolitems_262772)
        # Getting the type of the for loop variable (line 476)
        for_loop_var_262773 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 476, 8), toolitems_262772)
        # Assigning a type to the variable 'name' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 8), for_loop_var_262773))
        # Assigning a type to the variable 'tooltip' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'tooltip', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 8), for_loop_var_262773))
        # Assigning a type to the variable 'image' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'image', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 8), for_loop_var_262773))
        # Assigning a type to the variable 'method' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'method', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 8), for_loop_var_262773))
        # SSA begins for a for statement (line 476)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 477)
        # Getting the type of 'name' (line 477)
        name_262774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'name')
        # Getting the type of 'None' (line 477)
        None_262775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 23), 'None')
        
        (may_be_262776, more_types_in_union_262777) = may_be_none(name_262774, None_262775)

        if may_be_262776:

            if more_types_in_union_262777:
                # Runtime conditional SSA (line 477)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 478)
            # Processing the call arguments (line 478)
            
            # Obtaining an instance of the builtin type 'list' (line 478)
            list_262780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 478)
            # Adding element type (line 478)
            unicode_262781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 34), 'unicode', u'')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 33), list_262780, unicode_262781)
            # Adding element type (line 478)
            unicode_262782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 38), 'unicode', u'')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 33), list_262780, unicode_262782)
            # Adding element type (line 478)
            unicode_262783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 42), 'unicode', u'')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 33), list_262780, unicode_262783)
            # Adding element type (line 478)
            unicode_262784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 46), 'unicode', u'')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 33), list_262780, unicode_262784)
            
            # Processing the call keyword arguments (line 478)
            kwargs_262785 = {}
            # Getting the type of 'toolitems' (line 478)
            toolitems_262778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 16), 'toolitems', False)
            # Obtaining the member 'append' of a type (line 478)
            append_262779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 16), toolitems_262778, 'append')
            # Calling append(args, kwargs) (line 478)
            append_call_result_262786 = invoke(stypy.reporting.localization.Localization(__file__, 478, 16), append_262779, *[list_262780], **kwargs_262785)
            

            if more_types_in_union_262777:
                # Runtime conditional SSA for else branch (line 477)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_262776) or more_types_in_union_262777):
            
            # Call to append(...): (line 480)
            # Processing the call arguments (line 480)
            
            # Obtaining an instance of the builtin type 'list' (line 480)
            list_262789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 33), 'list')
            # Adding type elements to the builtin type 'list' instance (line 480)
            # Adding element type (line 480)
            # Getting the type of 'name' (line 480)
            name_262790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 34), 'name', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 33), list_262789, name_262790)
            # Adding element type (line 480)
            # Getting the type of 'tooltip' (line 480)
            tooltip_262791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 40), 'tooltip', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 33), list_262789, tooltip_262791)
            # Adding element type (line 480)
            # Getting the type of 'image' (line 480)
            image_262792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 49), 'image', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 33), list_262789, image_262792)
            # Adding element type (line 480)
            # Getting the type of 'method' (line 480)
            method_262793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 56), 'method', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 33), list_262789, method_262793)
            
            # Processing the call keyword arguments (line 480)
            kwargs_262794 = {}
            # Getting the type of 'toolitems' (line 480)
            toolitems_262787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'toolitems', False)
            # Obtaining the member 'append' of a type (line 480)
            append_262788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 16), toolitems_262787, 'append')
            # Calling append(args, kwargs) (line 480)
            append_call_result_262795 = invoke(stypy.reporting.localization.Localization(__file__, 480, 16), append_262788, *[list_262789], **kwargs_262794)
            

            if (may_be_262776 and more_types_in_union_262777):
                # SSA join for if statement (line 477)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 481)
        # Processing the call arguments (line 481)
        
        # Call to format(...): (line 481)
        # Processing the call arguments (line 481)
        
        # Call to dumps(...): (line 482)
        # Processing the call arguments (line 482)
        # Getting the type of 'toolitems' (line 482)
        toolitems_262802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 23), 'toolitems', False)
        # Processing the call keyword arguments (line 482)
        kwargs_262803 = {}
        # Getting the type of 'json' (line 482)
        json_262800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'json', False)
        # Obtaining the member 'dumps' of a type (line 482)
        dumps_262801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 12), json_262800, 'dumps')
        # Calling dumps(args, kwargs) (line 482)
        dumps_call_result_262804 = invoke(stypy.reporting.localization.Localization(__file__, 482, 12), dumps_262801, *[toolitems_262802], **kwargs_262803)
        
        # Processing the call keyword arguments (line 481)
        kwargs_262805 = {}
        unicode_262798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 21), 'unicode', u'mpl.toolbar_items = {0};\n\n')
        # Obtaining the member 'format' of a type (line 481)
        format_262799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 21), unicode_262798, 'format')
        # Calling format(args, kwargs) (line 481)
        format_call_result_262806 = invoke(stypy.reporting.localization.Localization(__file__, 481, 21), format_262799, *[dumps_call_result_262804], **kwargs_262805)
        
        # Processing the call keyword arguments (line 481)
        kwargs_262807 = {}
        # Getting the type of 'output' (line 481)
        output_262796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'output', False)
        # Obtaining the member 'write' of a type (line 481)
        write_262797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 8), output_262796, 'write')
        # Calling write(args, kwargs) (line 481)
        write_call_result_262808 = invoke(stypy.reporting.localization.Localization(__file__, 481, 8), write_262797, *[format_call_result_262806], **kwargs_262807)
        
        
        # Assigning a List to a Name (line 484):
        
        # Assigning a List to a Name (line 484):
        
        # Obtaining an instance of the builtin type 'list' (line 484)
        list_262809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 484)
        
        # Assigning a type to the variable 'extensions' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'extensions', list_262809)
        
        
        # Call to sorted(...): (line 485)
        # Processing the call arguments (line 485)
        
        # Call to items(...): (line 485)
        # Processing the call keyword arguments (line 485)
        kwargs_262816 = {}
        
        # Call to get_supported_filetypes_grouped(...): (line 485)
        # Processing the call keyword arguments (line 485)
        kwargs_262813 = {}
        # Getting the type of 'FigureCanvasWebAggCore' (line 485)
        FigureCanvasWebAggCore_262811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 36), 'FigureCanvasWebAggCore', False)
        # Obtaining the member 'get_supported_filetypes_grouped' of a type (line 485)
        get_supported_filetypes_grouped_262812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 36), FigureCanvasWebAggCore_262811, 'get_supported_filetypes_grouped')
        # Calling get_supported_filetypes_grouped(args, kwargs) (line 485)
        get_supported_filetypes_grouped_call_result_262814 = invoke(stypy.reporting.localization.Localization(__file__, 485, 36), get_supported_filetypes_grouped_262812, *[], **kwargs_262813)
        
        # Obtaining the member 'items' of a type (line 485)
        items_262815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 36), get_supported_filetypes_grouped_call_result_262814, 'items')
        # Calling items(args, kwargs) (line 485)
        items_call_result_262817 = invoke(stypy.reporting.localization.Localization(__file__, 485, 36), items_262815, *[], **kwargs_262816)
        
        # Processing the call keyword arguments (line 485)
        kwargs_262818 = {}
        # Getting the type of 'sorted' (line 485)
        sorted_262810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 29), 'sorted', False)
        # Calling sorted(args, kwargs) (line 485)
        sorted_call_result_262819 = invoke(stypy.reporting.localization.Localization(__file__, 485, 29), sorted_262810, *[items_call_result_262817], **kwargs_262818)
        
        # Testing the type of a for loop iterable (line 485)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 485, 8), sorted_call_result_262819)
        # Getting the type of the for loop variable (line 485)
        for_loop_var_262820 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 485, 8), sorted_call_result_262819)
        # Assigning a type to the variable 'filetype' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'filetype', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 8), for_loop_var_262820))
        # Assigning a type to the variable 'ext' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'ext', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 485, 8), for_loop_var_262820))
        # SSA begins for a for statement (line 485)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        
        # Obtaining the type of the subscript
        int_262821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 23), 'int')
        # Getting the type of 'ext' (line 488)
        ext_262822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 19), 'ext')
        # Obtaining the member '__getitem__' of a type (line 488)
        getitem___262823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 19), ext_262822, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 488)
        subscript_call_result_262824 = invoke(stypy.reporting.localization.Localization(__file__, 488, 19), getitem___262823, int_262821)
        
        unicode_262825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 29), 'unicode', u'pgf')
        # Applying the binary operator '==' (line 488)
        result_eq_262826 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 19), '==', subscript_call_result_262824, unicode_262825)
        
        # Applying the 'not' unary operator (line 488)
        result_not__262827 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 15), 'not', result_eq_262826)
        
        # Testing the type of an if condition (line 488)
        if_condition_262828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 488, 12), result_not__262827)
        # Assigning a type to the variable 'if_condition_262828' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'if_condition_262828', if_condition_262828)
        # SSA begins for if statement (line 488)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 489)
        # Processing the call arguments (line 489)
        
        # Obtaining the type of the subscript
        int_262831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 38), 'int')
        # Getting the type of 'ext' (line 489)
        ext_262832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 34), 'ext', False)
        # Obtaining the member '__getitem__' of a type (line 489)
        getitem___262833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 34), ext_262832, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 489)
        subscript_call_result_262834 = invoke(stypy.reporting.localization.Localization(__file__, 489, 34), getitem___262833, int_262831)
        
        # Processing the call keyword arguments (line 489)
        kwargs_262835 = {}
        # Getting the type of 'extensions' (line 489)
        extensions_262829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 16), 'extensions', False)
        # Obtaining the member 'append' of a type (line 489)
        append_262830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 16), extensions_262829, 'append')
        # Calling append(args, kwargs) (line 489)
        append_call_result_262836 = invoke(stypy.reporting.localization.Localization(__file__, 489, 16), append_262830, *[subscript_call_result_262834], **kwargs_262835)
        
        # SSA join for if statement (line 488)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 490)
        # Processing the call arguments (line 490)
        
        # Call to format(...): (line 490)
        # Processing the call arguments (line 490)
        
        # Call to dumps(...): (line 491)
        # Processing the call arguments (line 491)
        # Getting the type of 'extensions' (line 491)
        extensions_262843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 23), 'extensions', False)
        # Processing the call keyword arguments (line 491)
        kwargs_262844 = {}
        # Getting the type of 'json' (line 491)
        json_262841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'json', False)
        # Obtaining the member 'dumps' of a type (line 491)
        dumps_262842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 12), json_262841, 'dumps')
        # Calling dumps(args, kwargs) (line 491)
        dumps_call_result_262845 = invoke(stypy.reporting.localization.Localization(__file__, 491, 12), dumps_262842, *[extensions_262843], **kwargs_262844)
        
        # Processing the call keyword arguments (line 490)
        kwargs_262846 = {}
        unicode_262839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 21), 'unicode', u'mpl.extensions = {0};\n\n')
        # Obtaining the member 'format' of a type (line 490)
        format_262840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 21), unicode_262839, 'format')
        # Calling format(args, kwargs) (line 490)
        format_call_result_262847 = invoke(stypy.reporting.localization.Localization(__file__, 490, 21), format_262840, *[dumps_call_result_262845], **kwargs_262846)
        
        # Processing the call keyword arguments (line 490)
        kwargs_262848 = {}
        # Getting the type of 'output' (line 490)
        output_262837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'output', False)
        # Obtaining the member 'write' of a type (line 490)
        write_262838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), output_262837, 'write')
        # Calling write(args, kwargs) (line 490)
        write_call_result_262849 = invoke(stypy.reporting.localization.Localization(__file__, 490, 8), write_262838, *[format_call_result_262847], **kwargs_262848)
        
        
        # Call to write(...): (line 493)
        # Processing the call arguments (line 493)
        
        # Call to format(...): (line 493)
        # Processing the call arguments (line 493)
        
        # Call to dumps(...): (line 494)
        # Processing the call arguments (line 494)
        
        # Call to get_default_filetype(...): (line 494)
        # Processing the call keyword arguments (line 494)
        kwargs_262858 = {}
        # Getting the type of 'FigureCanvasWebAggCore' (line 494)
        FigureCanvasWebAggCore_262856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 23), 'FigureCanvasWebAggCore', False)
        # Obtaining the member 'get_default_filetype' of a type (line 494)
        get_default_filetype_262857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 23), FigureCanvasWebAggCore_262856, 'get_default_filetype')
        # Calling get_default_filetype(args, kwargs) (line 494)
        get_default_filetype_call_result_262859 = invoke(stypy.reporting.localization.Localization(__file__, 494, 23), get_default_filetype_262857, *[], **kwargs_262858)
        
        # Processing the call keyword arguments (line 494)
        kwargs_262860 = {}
        # Getting the type of 'json' (line 494)
        json_262854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'json', False)
        # Obtaining the member 'dumps' of a type (line 494)
        dumps_262855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 12), json_262854, 'dumps')
        # Calling dumps(args, kwargs) (line 494)
        dumps_call_result_262861 = invoke(stypy.reporting.localization.Localization(__file__, 494, 12), dumps_262855, *[get_default_filetype_call_result_262859], **kwargs_262860)
        
        # Processing the call keyword arguments (line 493)
        kwargs_262862 = {}
        unicode_262852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 21), 'unicode', u'mpl.default_extension = {0};')
        # Obtaining the member 'format' of a type (line 493)
        format_262853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 21), unicode_262852, 'format')
        # Calling format(args, kwargs) (line 493)
        format_call_result_262863 = invoke(stypy.reporting.localization.Localization(__file__, 493, 21), format_262853, *[dumps_call_result_262861], **kwargs_262862)
        
        # Processing the call keyword arguments (line 493)
        kwargs_262864 = {}
        # Getting the type of 'output' (line 493)
        output_262850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'output', False)
        # Obtaining the member 'write' of a type (line 493)
        write_262851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), output_262850, 'write')
        # Calling write(args, kwargs) (line 493)
        write_call_result_262865 = invoke(stypy.reporting.localization.Localization(__file__, 493, 8), write_262851, *[format_call_result_262863], **kwargs_262864)
        
        
        # Type idiom detected: calculating its left and rigth part (line 496)
        # Getting the type of 'stream' (line 496)
        stream_262866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 11), 'stream')
        # Getting the type of 'None' (line 496)
        None_262867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 21), 'None')
        
        (may_be_262868, more_types_in_union_262869) = may_be_none(stream_262866, None_262867)

        if may_be_262868:

            if more_types_in_union_262869:
                # Runtime conditional SSA (line 496)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to getvalue(...): (line 497)
            # Processing the call keyword arguments (line 497)
            kwargs_262872 = {}
            # Getting the type of 'output' (line 497)
            output_262870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 19), 'output', False)
            # Obtaining the member 'getvalue' of a type (line 497)
            getvalue_262871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 19), output_262870, 'getvalue')
            # Calling getvalue(args, kwargs) (line 497)
            getvalue_call_result_262873 = invoke(stypy.reporting.localization.Localization(__file__, 497, 19), getvalue_262871, *[], **kwargs_262872)
            
            # Assigning a type to the variable 'stypy_return_type' (line 497)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'stypy_return_type', getvalue_call_result_262873)

            if more_types_in_union_262869:
                # SSA join for if statement (line 496)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'get_javascript(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_javascript' in the type store
        # Getting the type of 'stypy_return_type' (line 462)
        stypy_return_type_262874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262874)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_javascript'
        return stypy_return_type_262874


    @norecursion
    def get_static_file_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_static_file_path'
        module_type_store = module_type_store.open_function_context('get_static_file_path', 499, 4, False)
        # Assigning a type to the variable 'self' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerWebAgg.get_static_file_path.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerWebAgg.get_static_file_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerWebAgg.get_static_file_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerWebAgg.get_static_file_path.__dict__.__setitem__('stypy_function_name', 'FigureManagerWebAgg.get_static_file_path')
        FigureManagerWebAgg.get_static_file_path.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerWebAgg.get_static_file_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerWebAgg.get_static_file_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerWebAgg.get_static_file_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerWebAgg.get_static_file_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerWebAgg.get_static_file_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerWebAgg.get_static_file_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg.get_static_file_path', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_static_file_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_static_file_path(...)' code ##################

        
        # Call to join(...): (line 501)
        # Processing the call arguments (line 501)
        
        # Call to dirname(...): (line 501)
        # Processing the call arguments (line 501)
        # Getting the type of '__file__' (line 501)
        file___262881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 44), '__file__', False)
        # Processing the call keyword arguments (line 501)
        kwargs_262882 = {}
        # Getting the type of 'os' (line 501)
        os_262878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 501)
        path_262879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 28), os_262878, 'path')
        # Obtaining the member 'dirname' of a type (line 501)
        dirname_262880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 28), path_262879, 'dirname')
        # Calling dirname(args, kwargs) (line 501)
        dirname_call_result_262883 = invoke(stypy.reporting.localization.Localization(__file__, 501, 28), dirname_262880, *[file___262881], **kwargs_262882)
        
        unicode_262884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 55), 'unicode', u'web_backend')
        # Processing the call keyword arguments (line 501)
        kwargs_262885 = {}
        # Getting the type of 'os' (line 501)
        os_262875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 501)
        path_262876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 15), os_262875, 'path')
        # Obtaining the member 'join' of a type (line 501)
        join_262877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 15), path_262876, 'join')
        # Calling join(args, kwargs) (line 501)
        join_call_result_262886 = invoke(stypy.reporting.localization.Localization(__file__, 501, 15), join_262877, *[dirname_call_result_262883, unicode_262884], **kwargs_262885)
        
        # Assigning a type to the variable 'stypy_return_type' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'stypy_return_type', join_call_result_262886)
        
        # ################# End of 'get_static_file_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_static_file_path' in the type store
        # Getting the type of 'stypy_return_type' (line 499)
        stypy_return_type_262887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262887)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_static_file_path'
        return stypy_return_type_262887


    @norecursion
    def _send_event(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_send_event'
        module_type_store = module_type_store.open_function_context('_send_event', 503, 4, False)
        # Assigning a type to the variable 'self' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerWebAgg._send_event.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerWebAgg._send_event.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerWebAgg._send_event.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerWebAgg._send_event.__dict__.__setitem__('stypy_function_name', 'FigureManagerWebAgg._send_event')
        FigureManagerWebAgg._send_event.__dict__.__setitem__('stypy_param_names_list', ['event_type'])
        FigureManagerWebAgg._send_event.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerWebAgg._send_event.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureManagerWebAgg._send_event.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerWebAgg._send_event.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerWebAgg._send_event.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerWebAgg._send_event.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerWebAgg._send_event', ['event_type'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_send_event', localization, ['event_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_send_event(...)' code ##################

        
        # Assigning a Dict to a Name (line 504):
        
        # Assigning a Dict to a Name (line 504):
        
        # Obtaining an instance of the builtin type 'dict' (line 504)
        dict_262888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 504)
        # Adding element type (key, value) (line 504)
        unicode_262889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 19), 'unicode', u'type')
        # Getting the type of 'event_type' (line 504)
        event_type_262890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 27), 'event_type')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 18), dict_262888, (unicode_262889, event_type_262890))
        
        # Assigning a type to the variable 'payload' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'payload', dict_262888)
        
        # Call to update(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'kwargs' (line 505)
        kwargs_262893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 23), 'kwargs', False)
        # Processing the call keyword arguments (line 505)
        kwargs_262894 = {}
        # Getting the type of 'payload' (line 505)
        payload_262891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'payload', False)
        # Obtaining the member 'update' of a type (line 505)
        update_262892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 8), payload_262891, 'update')
        # Calling update(args, kwargs) (line 505)
        update_call_result_262895 = invoke(stypy.reporting.localization.Localization(__file__, 505, 8), update_262892, *[kwargs_262893], **kwargs_262894)
        
        
        # Getting the type of 'self' (line 506)
        self_262896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 17), 'self')
        # Obtaining the member 'web_sockets' of a type (line 506)
        web_sockets_262897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 17), self_262896, 'web_sockets')
        # Testing the type of a for loop iterable (line 506)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 506, 8), web_sockets_262897)
        # Getting the type of the for loop variable (line 506)
        for_loop_var_262898 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 506, 8), web_sockets_262897)
        # Assigning a type to the variable 's' (line 506)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 's', for_loop_var_262898)
        # SSA begins for a for statement (line 506)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to send_json(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'payload' (line 507)
        payload_262901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 24), 'payload', False)
        # Processing the call keyword arguments (line 507)
        kwargs_262902 = {}
        # Getting the type of 's' (line 507)
        s_262899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 12), 's', False)
        # Obtaining the member 'send_json' of a type (line 507)
        send_json_262900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 12), s_262899, 'send_json')
        # Calling send_json(args, kwargs) (line 507)
        send_json_call_result_262903 = invoke(stypy.reporting.localization.Localization(__file__, 507, 12), send_json_262900, *[payload_262901], **kwargs_262902)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_send_event(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_send_event' in the type store
        # Getting the type of 'stypy_return_type' (line 503)
        stypy_return_type_262904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262904)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_send_event'
        return stypy_return_type_262904


# Assigning a type to the variable 'FigureManagerWebAgg' (line 412)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 0), 'FigureManagerWebAgg', FigureManagerWebAgg)

# Assigning a Name to a Name (line 413):
# Getting the type of 'NavigationToolbar2WebAgg' (line 413)
NavigationToolbar2WebAgg_262905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 17), 'NavigationToolbar2WebAgg')
# Getting the type of 'FigureManagerWebAgg'
FigureManagerWebAgg_262906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureManagerWebAgg')
# Setting the type of the member 'ToolbarCls' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureManagerWebAgg_262906, 'ToolbarCls', NavigationToolbar2WebAgg_262905)
# Declaration of the 'TimerTornado' class
# Getting the type of 'backend_bases' (line 510)
backend_bases_262907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 19), 'backend_bases')
# Obtaining the member 'TimerBase' of a type (line 510)
TimerBase_262908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 19), backend_bases_262907, 'TimerBase')

class TimerTornado(TimerBase_262908, ):

    @norecursion
    def _timer_start(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_timer_start'
        module_type_store = module_type_store.open_function_context('_timer_start', 511, 4, False)
        # Assigning a type to the variable 'self' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimerTornado._timer_start.__dict__.__setitem__('stypy_localization', localization)
        TimerTornado._timer_start.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimerTornado._timer_start.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimerTornado._timer_start.__dict__.__setitem__('stypy_function_name', 'TimerTornado._timer_start')
        TimerTornado._timer_start.__dict__.__setitem__('stypy_param_names_list', [])
        TimerTornado._timer_start.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimerTornado._timer_start.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimerTornado._timer_start.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimerTornado._timer_start.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimerTornado._timer_start.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimerTornado._timer_start.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerTornado._timer_start', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to _timer_stop(...): (line 512)
        # Processing the call keyword arguments (line 512)
        kwargs_262911 = {}
        # Getting the type of 'self' (line 512)
        self_262909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'self', False)
        # Obtaining the member '_timer_stop' of a type (line 512)
        _timer_stop_262910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), self_262909, '_timer_stop')
        # Calling _timer_stop(args, kwargs) (line 512)
        _timer_stop_call_result_262912 = invoke(stypy.reporting.localization.Localization(__file__, 512, 8), _timer_stop_262910, *[], **kwargs_262911)
        
        
        # Getting the type of 'self' (line 513)
        self_262913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 11), 'self')
        # Obtaining the member '_single' of a type (line 513)
        _single_262914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 11), self_262913, '_single')
        # Testing the type of an if condition (line 513)
        if_condition_262915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 513, 8), _single_262914)
        # Assigning a type to the variable 'if_condition_262915' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'if_condition_262915', if_condition_262915)
        # SSA begins for if statement (line 513)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 514):
        
        # Assigning a Call to a Name (line 514):
        
        # Call to instance(...): (line 514)
        # Processing the call keyword arguments (line 514)
        kwargs_262920 = {}
        # Getting the type of 'tornado' (line 514)
        tornado_262916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 21), 'tornado', False)
        # Obtaining the member 'ioloop' of a type (line 514)
        ioloop_262917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 21), tornado_262916, 'ioloop')
        # Obtaining the member 'IOLoop' of a type (line 514)
        IOLoop_262918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 21), ioloop_262917, 'IOLoop')
        # Obtaining the member 'instance' of a type (line 514)
        instance_262919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 21), IOLoop_262918, 'instance')
        # Calling instance(args, kwargs) (line 514)
        instance_call_result_262921 = invoke(stypy.reporting.localization.Localization(__file__, 514, 21), instance_262919, *[], **kwargs_262920)
        
        # Assigning a type to the variable 'ioloop' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'ioloop', instance_call_result_262921)
        
        # Assigning a Call to a Attribute (line 515):
        
        # Assigning a Call to a Attribute (line 515):
        
        # Call to add_timeout(...): (line 515)
        # Processing the call arguments (line 515)
        
        # Call to timedelta(...): (line 516)
        # Processing the call keyword arguments (line 516)
        # Getting the type of 'self' (line 516)
        self_262926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 48), 'self', False)
        # Obtaining the member 'interval' of a type (line 516)
        interval_262927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 48), self_262926, 'interval')
        keyword_262928 = interval_262927
        kwargs_262929 = {'milliseconds': keyword_262928}
        # Getting the type of 'datetime' (line 516)
        datetime_262924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 16), 'datetime', False)
        # Obtaining the member 'timedelta' of a type (line 516)
        timedelta_262925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 16), datetime_262924, 'timedelta')
        # Calling timedelta(args, kwargs) (line 516)
        timedelta_call_result_262930 = invoke(stypy.reporting.localization.Localization(__file__, 516, 16), timedelta_262925, *[], **kwargs_262929)
        
        # Getting the type of 'self' (line 517)
        self_262931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 16), 'self', False)
        # Obtaining the member '_on_timer' of a type (line 517)
        _on_timer_262932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 16), self_262931, '_on_timer')
        # Processing the call keyword arguments (line 515)
        kwargs_262933 = {}
        # Getting the type of 'ioloop' (line 515)
        ioloop_262922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 26), 'ioloop', False)
        # Obtaining the member 'add_timeout' of a type (line 515)
        add_timeout_262923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 26), ioloop_262922, 'add_timeout')
        # Calling add_timeout(args, kwargs) (line 515)
        add_timeout_call_result_262934 = invoke(stypy.reporting.localization.Localization(__file__, 515, 26), add_timeout_262923, *[timedelta_call_result_262930, _on_timer_262932], **kwargs_262933)
        
        # Getting the type of 'self' (line 515)
        self_262935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'self')
        # Setting the type of the member '_timer' of a type (line 515)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 12), self_262935, '_timer', add_timeout_call_result_262934)
        # SSA branch for the else part of an if statement (line 513)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 519):
        
        # Assigning a Call to a Attribute (line 519):
        
        # Call to PeriodicCallback(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'self' (line 520)
        self_262939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'self', False)
        # Obtaining the member '_on_timer' of a type (line 520)
        _on_timer_262940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 16), self_262939, '_on_timer')
        # Getting the type of 'self' (line 521)
        self_262941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 16), 'self', False)
        # Obtaining the member 'interval' of a type (line 521)
        interval_262942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 16), self_262941, 'interval')
        # Processing the call keyword arguments (line 519)
        kwargs_262943 = {}
        # Getting the type of 'tornado' (line 519)
        tornado_262936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 26), 'tornado', False)
        # Obtaining the member 'ioloop' of a type (line 519)
        ioloop_262937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 26), tornado_262936, 'ioloop')
        # Obtaining the member 'PeriodicCallback' of a type (line 519)
        PeriodicCallback_262938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 26), ioloop_262937, 'PeriodicCallback')
        # Calling PeriodicCallback(args, kwargs) (line 519)
        PeriodicCallback_call_result_262944 = invoke(stypy.reporting.localization.Localization(__file__, 519, 26), PeriodicCallback_262938, *[_on_timer_262940, interval_262942], **kwargs_262943)
        
        # Getting the type of 'self' (line 519)
        self_262945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'self')
        # Setting the type of the member '_timer' of a type (line 519)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 12), self_262945, '_timer', PeriodicCallback_call_result_262944)
        
        # Call to start(...): (line 522)
        # Processing the call keyword arguments (line 522)
        kwargs_262949 = {}
        # Getting the type of 'self' (line 522)
        self_262946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'self', False)
        # Obtaining the member '_timer' of a type (line 522)
        _timer_262947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 12), self_262946, '_timer')
        # Obtaining the member 'start' of a type (line 522)
        start_262948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 12), _timer_262947, 'start')
        # Calling start(args, kwargs) (line 522)
        start_call_result_262950 = invoke(stypy.reporting.localization.Localization(__file__, 522, 12), start_262948, *[], **kwargs_262949)
        
        # SSA join for if statement (line 513)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_timer_start(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_timer_start' in the type store
        # Getting the type of 'stypy_return_type' (line 511)
        stypy_return_type_262951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262951)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_timer_start'
        return stypy_return_type_262951


    @norecursion
    def _timer_stop(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_timer_stop'
        module_type_store = module_type_store.open_function_context('_timer_stop', 524, 4, False)
        # Assigning a type to the variable 'self' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimerTornado._timer_stop.__dict__.__setitem__('stypy_localization', localization)
        TimerTornado._timer_stop.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimerTornado._timer_stop.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimerTornado._timer_stop.__dict__.__setitem__('stypy_function_name', 'TimerTornado._timer_stop')
        TimerTornado._timer_stop.__dict__.__setitem__('stypy_param_names_list', [])
        TimerTornado._timer_stop.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimerTornado._timer_stop.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimerTornado._timer_stop.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimerTornado._timer_stop.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimerTornado._timer_stop.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimerTornado._timer_stop.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerTornado._timer_stop', [], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 525)
        # Getting the type of 'self' (line 525)
        self_262952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 11), 'self')
        # Obtaining the member '_timer' of a type (line 525)
        _timer_262953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 11), self_262952, '_timer')
        # Getting the type of 'None' (line 525)
        None_262954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 26), 'None')
        
        (may_be_262955, more_types_in_union_262956) = may_be_none(_timer_262953, None_262954)

        if may_be_262955:

            if more_types_in_union_262956:
                # Runtime conditional SSA (line 525)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'stypy_return_type' (line 526)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_262956:
                # Runtime conditional SSA for else branch (line 525)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_262955) or more_types_in_union_262956):
            
            # Getting the type of 'self' (line 527)
            self_262957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 13), 'self')
            # Obtaining the member '_single' of a type (line 527)
            _single_262958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 13), self_262957, '_single')
            # Testing the type of an if condition (line 527)
            if_condition_262959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 527, 13), _single_262958)
            # Assigning a type to the variable 'if_condition_262959' (line 527)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 13), 'if_condition_262959', if_condition_262959)
            # SSA begins for if statement (line 527)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 528):
            
            # Assigning a Call to a Name (line 528):
            
            # Call to instance(...): (line 528)
            # Processing the call keyword arguments (line 528)
            kwargs_262964 = {}
            # Getting the type of 'tornado' (line 528)
            tornado_262960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 21), 'tornado', False)
            # Obtaining the member 'ioloop' of a type (line 528)
            ioloop_262961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 21), tornado_262960, 'ioloop')
            # Obtaining the member 'IOLoop' of a type (line 528)
            IOLoop_262962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 21), ioloop_262961, 'IOLoop')
            # Obtaining the member 'instance' of a type (line 528)
            instance_262963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 21), IOLoop_262962, 'instance')
            # Calling instance(args, kwargs) (line 528)
            instance_call_result_262965 = invoke(stypy.reporting.localization.Localization(__file__, 528, 21), instance_262963, *[], **kwargs_262964)
            
            # Assigning a type to the variable 'ioloop' (line 528)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'ioloop', instance_call_result_262965)
            
            # Call to remove_timeout(...): (line 529)
            # Processing the call arguments (line 529)
            # Getting the type of 'self' (line 529)
            self_262968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 34), 'self', False)
            # Obtaining the member '_timer' of a type (line 529)
            _timer_262969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 34), self_262968, '_timer')
            # Processing the call keyword arguments (line 529)
            kwargs_262970 = {}
            # Getting the type of 'ioloop' (line 529)
            ioloop_262966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 12), 'ioloop', False)
            # Obtaining the member 'remove_timeout' of a type (line 529)
            remove_timeout_262967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 12), ioloop_262966, 'remove_timeout')
            # Calling remove_timeout(args, kwargs) (line 529)
            remove_timeout_call_result_262971 = invoke(stypy.reporting.localization.Localization(__file__, 529, 12), remove_timeout_262967, *[_timer_262969], **kwargs_262970)
            
            # SSA branch for the else part of an if statement (line 527)
            module_type_store.open_ssa_branch('else')
            
            # Call to stop(...): (line 531)
            # Processing the call keyword arguments (line 531)
            kwargs_262975 = {}
            # Getting the type of 'self' (line 531)
            self_262972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'self', False)
            # Obtaining the member '_timer' of a type (line 531)
            _timer_262973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 12), self_262972, '_timer')
            # Obtaining the member 'stop' of a type (line 531)
            stop_262974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 12), _timer_262973, 'stop')
            # Calling stop(args, kwargs) (line 531)
            stop_call_result_262976 = invoke(stypy.reporting.localization.Localization(__file__, 531, 12), stop_262974, *[], **kwargs_262975)
            
            # SSA join for if statement (line 527)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_262955 and more_types_in_union_262956):
                # SSA join for if statement (line 525)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 533):
        
        # Assigning a Name to a Attribute (line 533):
        # Getting the type of 'None' (line 533)
        None_262977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 22), 'None')
        # Getting the type of 'self' (line 533)
        self_262978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'self')
        # Setting the type of the member '_timer' of a type (line 533)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), self_262978, '_timer', None_262977)
        
        # ################# End of '_timer_stop(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_timer_stop' in the type store
        # Getting the type of 'stypy_return_type' (line 524)
        stypy_return_type_262979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262979)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_timer_stop'
        return stypy_return_type_262979


    @norecursion
    def _timer_set_interval(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_timer_set_interval'
        module_type_store = module_type_store.open_function_context('_timer_set_interval', 535, 4, False)
        # Assigning a type to the variable 'self' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TimerTornado._timer_set_interval.__dict__.__setitem__('stypy_localization', localization)
        TimerTornado._timer_set_interval.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TimerTornado._timer_set_interval.__dict__.__setitem__('stypy_type_store', module_type_store)
        TimerTornado._timer_set_interval.__dict__.__setitem__('stypy_function_name', 'TimerTornado._timer_set_interval')
        TimerTornado._timer_set_interval.__dict__.__setitem__('stypy_param_names_list', [])
        TimerTornado._timer_set_interval.__dict__.__setitem__('stypy_varargs_param_name', None)
        TimerTornado._timer_set_interval.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TimerTornado._timer_set_interval.__dict__.__setitem__('stypy_call_defaults', defaults)
        TimerTornado._timer_set_interval.__dict__.__setitem__('stypy_call_varargs', varargs)
        TimerTornado._timer_set_interval.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TimerTornado._timer_set_interval.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerTornado._timer_set_interval', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 537)
        self_262980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 11), 'self')
        # Obtaining the member '_timer' of a type (line 537)
        _timer_262981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 11), self_262980, '_timer')
        # Getting the type of 'None' (line 537)
        None_262982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 30), 'None')
        # Applying the binary operator 'isnot' (line 537)
        result_is_not_262983 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 11), 'isnot', _timer_262981, None_262982)
        
        # Testing the type of an if condition (line 537)
        if_condition_262984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 537, 8), result_is_not_262983)
        # Assigning a type to the variable 'if_condition_262984' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'if_condition_262984', if_condition_262984)
        # SSA begins for if statement (line 537)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _timer_stop(...): (line 538)
        # Processing the call keyword arguments (line 538)
        kwargs_262987 = {}
        # Getting the type of 'self' (line 538)
        self_262985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'self', False)
        # Obtaining the member '_timer_stop' of a type (line 538)
        _timer_stop_262986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 12), self_262985, '_timer_stop')
        # Calling _timer_stop(args, kwargs) (line 538)
        _timer_stop_call_result_262988 = invoke(stypy.reporting.localization.Localization(__file__, 538, 12), _timer_stop_262986, *[], **kwargs_262987)
        
        
        # Call to _timer_start(...): (line 539)
        # Processing the call keyword arguments (line 539)
        kwargs_262991 = {}
        # Getting the type of 'self' (line 539)
        self_262989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'self', False)
        # Obtaining the member '_timer_start' of a type (line 539)
        _timer_start_262990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 12), self_262989, '_timer_start')
        # Calling _timer_start(args, kwargs) (line 539)
        _timer_start_call_result_262992 = invoke(stypy.reporting.localization.Localization(__file__, 539, 12), _timer_start_262990, *[], **kwargs_262991)
        
        # SSA join for if statement (line 537)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_timer_set_interval(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_timer_set_interval' in the type store
        # Getting the type of 'stypy_return_type' (line 535)
        stypy_return_type_262993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_262993)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_timer_set_interval'
        return stypy_return_type_262993


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 510, 0, False)
        # Assigning a type to the variable 'self' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TimerTornado.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TimerTornado' (line 510)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 0), 'TimerTornado', TimerTornado)
# Declaration of the '_BackendWebAggCoreAgg' class
# Getting the type of '_Backend' (line 543)
_Backend_262994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 28), '_Backend')

class _BackendWebAggCoreAgg(_Backend_262994, ):
    
    # Assigning a Name to a Name (line 544):
    
    # Assigning a Name to a Name (line 545):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 542, 0, False)
        # Assigning a type to the variable 'self' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendWebAggCoreAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendWebAggCoreAgg' (line 542)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 0), '_BackendWebAggCoreAgg', _BackendWebAggCoreAgg)

# Assigning a Name to a Name (line 544):
# Getting the type of 'FigureCanvasWebAggCore' (line 544)
FigureCanvasWebAggCore_262995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 19), 'FigureCanvasWebAggCore')
# Getting the type of '_BackendWebAggCoreAgg'
_BackendWebAggCoreAgg_262996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendWebAggCoreAgg')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendWebAggCoreAgg_262996, 'FigureCanvas', FigureCanvasWebAggCore_262995)

# Assigning a Name to a Name (line 545):
# Getting the type of 'FigureManagerWebAgg' (line 545)
FigureManagerWebAgg_262997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 20), 'FigureManagerWebAgg')
# Getting the type of '_BackendWebAggCoreAgg'
_BackendWebAggCoreAgg_262998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendWebAggCoreAgg')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendWebAggCoreAgg_262998, 'FigureManager', FigureManagerWebAgg_262997)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
