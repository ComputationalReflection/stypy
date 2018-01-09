
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Displays Agg images in the browser, with interactivity
3: '''
4: from __future__ import (absolute_import, division, print_function,
5:                         unicode_literals)
6: 
7: # The WebAgg backend is divided into two modules:
8: #
9: # - `backend_webagg_core.py` contains code necessary to embed a WebAgg
10: #   plot inside of a web application, and communicate in an abstract
11: #   way over a web socket.
12: #
13: # - `backend_webagg.py` contains a concrete implementation of a basic
14: #   application, implemented with tornado.
15: 
16: import six
17: 
18: import errno
19: import json
20: import os
21: import random
22: import sys
23: import signal
24: import socket
25: import threading
26: from contextlib import contextmanager
27: 
28: try:
29:     import tornado
30: except ImportError:
31:     raise RuntimeError("The WebAgg backend requires Tornado.")
32: 
33: import tornado.web
34: import tornado.ioloop
35: import tornado.websocket
36: 
37: import matplotlib
38: from matplotlib import rcParams
39: from matplotlib import backend_bases
40: from matplotlib.backend_bases import _Backend
41: from matplotlib.figure import Figure
42: from matplotlib._pylab_helpers import Gcf
43: from . import backend_webagg_core as core
44: from .backend_webagg_core import TimerTornado
45: 
46: 
47: class ServerThread(threading.Thread):
48:     def run(self):
49:         tornado.ioloop.IOLoop.instance().start()
50: 
51: webagg_server_thread = ServerThread()
52: 
53: 
54: class FigureCanvasWebAgg(core.FigureCanvasWebAggCore):
55:     def show(self):
56:         # show the figure window
57:         show()
58: 
59:     def new_timer(self, *args, **kwargs):
60:         return TimerTornado(*args, **kwargs)
61: 
62: 
63: class WebAggApplication(tornado.web.Application):
64:     initialized = False
65:     started = False
66: 
67:     class FavIcon(tornado.web.RequestHandler):
68:         def get(self):
69:             image_path = os.path.join(
70:                 os.path.dirname(os.path.dirname(__file__)),
71:                 'mpl-data', 'images')
72: 
73:             self.set_header('Content-Type', 'image/png')
74:             with open(os.path.join(image_path,
75:                                    'matplotlib.png'), 'rb') as fd:
76:                 self.write(fd.read())
77: 
78:     class SingleFigurePage(tornado.web.RequestHandler):
79:         def __init__(self, application, request, **kwargs):
80:             self.url_prefix = kwargs.pop('url_prefix', '')
81:             return tornado.web.RequestHandler.__init__(self, application,
82:                                                        request, **kwargs)
83: 
84:         def get(self, fignum):
85:             fignum = int(fignum)
86:             manager = Gcf.get_fig_manager(fignum)
87: 
88:             ws_uri = 'ws://{req.host}{prefix}/'.format(req=self.request,
89:                                                        prefix=self.url_prefix)
90:             self.render(
91:                 "single_figure.html",
92:                 prefix=self.url_prefix,
93:                 ws_uri=ws_uri,
94:                 fig_id=fignum,
95:                 toolitems=core.NavigationToolbar2WebAgg.toolitems,
96:                 canvas=manager.canvas)
97: 
98:     class AllFiguresPage(tornado.web.RequestHandler):
99:         def __init__(self, application, request, **kwargs):
100:             self.url_prefix = kwargs.pop('url_prefix', '')
101:             return tornado.web.RequestHandler.__init__(self, application,
102:                                                        request, **kwargs)
103: 
104:         def get(self):
105:             ws_uri = 'ws://{req.host}{prefix}/'.format(req=self.request,
106:                                                        prefix=self.url_prefix)
107:             self.render(
108:                 "all_figures.html",
109:                 prefix=self.url_prefix,
110:                 ws_uri=ws_uri,
111:                 figures=sorted(
112:                     list(Gcf.figs.items()), key=lambda item: item[0]),
113:                 toolitems=core.NavigationToolbar2WebAgg.toolitems)
114: 
115:     class MplJs(tornado.web.RequestHandler):
116:         def get(self):
117:             self.set_header('Content-Type', 'application/javascript')
118: 
119:             js_content = core.FigureManagerWebAgg.get_javascript()
120: 
121:             self.write(js_content)
122: 
123:     class Download(tornado.web.RequestHandler):
124:         def get(self, fignum, fmt):
125:             fignum = int(fignum)
126:             manager = Gcf.get_fig_manager(fignum)
127: 
128:             # TODO: Move this to a central location
129:             mimetypes = {
130:                 'ps': 'application/postscript',
131:                 'eps': 'application/postscript',
132:                 'pdf': 'application/pdf',
133:                 'svg': 'image/svg+xml',
134:                 'png': 'image/png',
135:                 'jpeg': 'image/jpeg',
136:                 'tif': 'image/tiff',
137:                 'emf': 'application/emf'
138:             }
139: 
140:             self.set_header('Content-Type', mimetypes.get(fmt, 'binary'))
141: 
142:             buff = six.BytesIO()
143:             manager.canvas.figure.savefig(buff, format=fmt)
144:             self.write(buff.getvalue())
145: 
146:     class WebSocket(tornado.websocket.WebSocketHandler):
147:         supports_binary = True
148: 
149:         def open(self, fignum):
150:             self.fignum = int(fignum)
151:             self.manager = Gcf.get_fig_manager(self.fignum)
152:             self.manager.add_web_socket(self)
153:             if hasattr(self, 'set_nodelay'):
154:                 self.set_nodelay(True)
155: 
156:         def on_close(self):
157:             self.manager.remove_web_socket(self)
158: 
159:         def on_message(self, message):
160:             message = json.loads(message)
161:             # The 'supports_binary' message is on a client-by-client
162:             # basis.  The others affect the (shared) canvas as a
163:             # whole.
164:             if message['type'] == 'supports_binary':
165:                 self.supports_binary = message['value']
166:             else:
167:                 manager = Gcf.get_fig_manager(self.fignum)
168:                 # It is possible for a figure to be closed,
169:                 # but a stale figure UI is still sending messages
170:                 # from the browser.
171:                 if manager is not None:
172:                     manager.handle_json(message)
173: 
174:         def send_json(self, content):
175:             self.write_message(json.dumps(content))
176: 
177:         def send_binary(self, blob):
178:             if self.supports_binary:
179:                 self.write_message(blob, binary=True)
180:             else:
181:                 data_uri = "data:image/png;base64,{0}".format(
182:                     blob.encode('base64').replace('\n', ''))
183:                 self.write_message(data_uri)
184: 
185:     def __init__(self, url_prefix=''):
186:         if url_prefix:
187:             assert url_prefix[0] == '/' and url_prefix[-1] != '/', \
188:                 'url_prefix must start with a "/" and not end with one.'
189: 
190:         super(WebAggApplication, self).__init__(
191:             [
192:                 # Static files for the CSS and JS
193:                 (url_prefix + r'/_static/(.*)',
194:                  tornado.web.StaticFileHandler,
195:                  {'path': core.FigureManagerWebAgg.get_static_file_path()}),
196: 
197:                 # An MPL favicon
198:                 (url_prefix + r'/favicon.ico', self.FavIcon),
199: 
200:                 # The page that contains all of the pieces
201:                 (url_prefix + r'/([0-9]+)', self.SingleFigurePage,
202:                  {'url_prefix': url_prefix}),
203: 
204:                 # The page that contains all of the figures
205:                 (url_prefix + r'/?', self.AllFiguresPage,
206:                  {'url_prefix': url_prefix}),
207: 
208:                 (url_prefix + r'/js/mpl.js', self.MplJs),
209: 
210:                 # Sends images and events to the browser, and receives
211:                 # events from the browser
212:                 (url_prefix + r'/([0-9]+)/ws', self.WebSocket),
213: 
214:                 # Handles the downloading (i.e., saving) of static images
215:                 (url_prefix + r'/([0-9]+)/download.([a-z0-9.]+)',
216:                  self.Download),
217:             ],
218:             template_path=core.FigureManagerWebAgg.get_static_file_path())
219: 
220:     @classmethod
221:     def initialize(cls, url_prefix='', port=None):
222:         if cls.initialized:
223:             return
224: 
225:         # Create the class instance
226:         app = cls(url_prefix=url_prefix)
227: 
228:         cls.url_prefix = url_prefix
229: 
230:         # This port selection algorithm is borrowed, more or less
231:         # verbatim, from IPython.
232:         def random_ports(port, n):
233:             '''
234:             Generate a list of n random ports near the given port.
235: 
236:             The first 5 ports will be sequential, and the remaining n-5 will be
237:             randomly selected in the range [port-2*n, port+2*n].
238:             '''
239:             for i in range(min(5, n)):
240:                 yield port + i
241:             for i in range(n - 5):
242:                 yield port + random.randint(-2 * n, 2 * n)
243: 
244:         success = None
245:         cls.port = rcParams['webagg.port']
246:         for port in random_ports(cls.port, rcParams['webagg.port_retries']):
247:             try:
248:                 app.listen(port)
249:             except socket.error as e:
250:                 if e.errno != errno.EADDRINUSE:
251:                     raise
252:             else:
253:                 cls.port = port
254:                 success = True
255:                 break
256: 
257:         if not success:
258:             raise SystemExit(
259:                 "The webagg server could not be started because an available "
260:                 "port could not be found")
261: 
262:         cls.initialized = True
263: 
264:     @classmethod
265:     def start(cls):
266:         if cls.started:
267:             return
268: 
269:         '''
270:         IOLoop.running() was removed as of Tornado 2.4; see for example
271:         https://groups.google.com/forum/#!topic/python-tornado/QLMzkpQBGOY
272:         Thus there is no correct way to check if the loop has already been
273:         launched. We may end up with two concurrently running loops in that
274:         unlucky case with all the expected consequences.
275:         '''
276:         ioloop = tornado.ioloop.IOLoop.instance()
277: 
278:         def shutdown():
279:             ioloop.stop()
280:             print("Server is stopped")
281:             sys.stdout.flush()
282:             cls.started = False
283: 
284:         @contextmanager
285:         def catch_sigint():
286:             old_handler = signal.signal(
287:                 signal.SIGINT,
288:                 lambda sig, frame: ioloop.add_callback_from_signal(shutdown))
289:             try:
290:                 yield
291:             finally:
292:                 signal.signal(signal.SIGINT, old_handler)
293: 
294:         # Set the flag to True *before* blocking on ioloop.start()
295:         cls.started = True
296: 
297:         print("Press Ctrl+C to stop WebAgg server")
298:         sys.stdout.flush()
299:         with catch_sigint():
300:             ioloop.start()
301: 
302: 
303: def ipython_inline_display(figure):
304:     import tornado.template
305: 
306:     WebAggApplication.initialize()
307:     if not webagg_server_thread.is_alive():
308:         webagg_server_thread.start()
309: 
310:     with open(os.path.join(
311:             core.FigureManagerWebAgg.get_static_file_path(),
312:             'ipython_inline_figure.html')) as fd:
313:         tpl = fd.read()
314: 
315:     fignum = figure.number
316: 
317:     t = tornado.template.Template(tpl)
318:     return t.generate(
319:         prefix=WebAggApplication.url_prefix,
320:         fig_id=fignum,
321:         toolitems=core.NavigationToolbar2WebAgg.toolitems,
322:         canvas=figure.canvas,
323:         port=WebAggApplication.port).decode('utf-8')
324: 
325: 
326: @_Backend.export
327: class _BackendWebAgg(_Backend):
328:     FigureCanvas = FigureCanvasWebAgg
329:     FigureManager = FigureManagerWebAgg
330: 
331:     @staticmethod
332:     def trigger_manager_draw(manager):
333:         manager.canvas.draw_idle()
334: 
335:     @staticmethod
336:     def show():
337:         WebAggApplication.initialize()
338: 
339:         url = "http://127.0.0.1:{port}{prefix}".format(
340:             port=WebAggApplication.port,
341:             prefix=WebAggApplication.url_prefix)
342: 
343:         if rcParams['webagg.open_in_browser']:
344:             import webbrowser
345:             webbrowser.open(url)
346:         else:
347:             print("To view figure, visit {0}".format(url))
348: 
349:         WebAggApplication.start()
350: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_260647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nDisplays Agg images in the browser, with interactivity\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import six' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260648 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six')

if (type(import_260648) is not StypyTypeError):

    if (import_260648 != 'pyd_module'):
        __import__(import_260648)
        sys_modules_260649 = sys.modules[import_260648]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', sys_modules_260649.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', import_260648)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import errno' statement (line 18)
import errno

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'errno', errno, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import json' statement (line 19)
import json

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'json', json, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import os' statement (line 20)
import os

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import random' statement (line 21)
import random

import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'random', random, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import sys' statement (line 22)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import signal' statement (line 23)
import signal

import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'signal', signal, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import socket' statement (line 24)
import socket

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'socket', socket, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'import threading' statement (line 25)
import threading

import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'threading', threading, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from contextlib import contextmanager' statement (line 26)
try:
    from contextlib import contextmanager

except:
    contextmanager = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'contextlib', None, module_type_store, ['contextmanager'], [contextmanager])



# SSA begins for try-except statement (line 28)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 4))

# 'import tornado' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260650 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 4), 'tornado')

if (type(import_260650) is not StypyTypeError):

    if (import_260650 != 'pyd_module'):
        __import__(import_260650)
        sys_modules_260651 = sys.modules[import_260650]
        import_module(stypy.reporting.localization.Localization(__file__, 29, 4), 'tornado', sys_modules_260651.module_type_store, module_type_store)
    else:
        import tornado

        import_module(stypy.reporting.localization.Localization(__file__, 29, 4), 'tornado', tornado, module_type_store)

else:
    # Assigning a type to the variable 'tornado' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'tornado', import_260650)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# SSA branch for the except part of a try statement (line 28)
# SSA branch for the except 'ImportError' branch of a try statement (line 28)
module_type_store.open_ssa_branch('except')

# Call to RuntimeError(...): (line 31)
# Processing the call arguments (line 31)
unicode_260653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'unicode', u'The WebAgg backend requires Tornado.')
# Processing the call keyword arguments (line 31)
kwargs_260654 = {}
# Getting the type of 'RuntimeError' (line 31)
RuntimeError_260652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'RuntimeError', False)
# Calling RuntimeError(args, kwargs) (line 31)
RuntimeError_call_result_260655 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), RuntimeError_260652, *[unicode_260653], **kwargs_260654)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 31, 4), RuntimeError_call_result_260655, 'raise parameter', BaseException)
# SSA join for try-except statement (line 28)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'import tornado.web' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260656 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'tornado.web')

if (type(import_260656) is not StypyTypeError):

    if (import_260656 != 'pyd_module'):
        __import__(import_260656)
        sys_modules_260657 = sys.modules[import_260656]
        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'tornado.web', sys_modules_260657.module_type_store, module_type_store)
    else:
        import tornado.web

        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'tornado.web', tornado.web, module_type_store)

else:
    # Assigning a type to the variable 'tornado.web' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'tornado.web', import_260656)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'import tornado.ioloop' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260658 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'tornado.ioloop')

if (type(import_260658) is not StypyTypeError):

    if (import_260658 != 'pyd_module'):
        __import__(import_260658)
        sys_modules_260659 = sys.modules[import_260658]
        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'tornado.ioloop', sys_modules_260659.module_type_store, module_type_store)
    else:
        import tornado.ioloop

        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'tornado.ioloop', tornado.ioloop, module_type_store)

else:
    # Assigning a type to the variable 'tornado.ioloop' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'tornado.ioloop', import_260658)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'import tornado.websocket' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260660 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'tornado.websocket')

if (type(import_260660) is not StypyTypeError):

    if (import_260660 != 'pyd_module'):
        __import__(import_260660)
        sys_modules_260661 = sys.modules[import_260660]
        import_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'tornado.websocket', sys_modules_260661.module_type_store, module_type_store)
    else:
        import tornado.websocket

        import_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'tornado.websocket', tornado.websocket, module_type_store)

else:
    # Assigning a type to the variable 'tornado.websocket' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'tornado.websocket', import_260660)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'import matplotlib' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260662 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib')

if (type(import_260662) is not StypyTypeError):

    if (import_260662 != 'pyd_module'):
        __import__(import_260662)
        sys_modules_260663 = sys.modules[import_260662]
        import_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib', sys_modules_260663.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib', import_260662)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'from matplotlib import rcParams' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260664 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'matplotlib')

if (type(import_260664) is not StypyTypeError):

    if (import_260664 != 'pyd_module'):
        __import__(import_260664)
        sys_modules_260665 = sys.modules[import_260664]
        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'matplotlib', sys_modules_260665.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 38, 0), __file__, sys_modules_260665, sys_modules_260665.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'matplotlib', import_260664)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'from matplotlib import backend_bases' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260666 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib')

if (type(import_260666) is not StypyTypeError):

    if (import_260666 != 'pyd_module'):
        __import__(import_260666)
        sys_modules_260667 = sys.modules[import_260666]
        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib', sys_modules_260667.module_type_store, module_type_store, ['backend_bases'])
        nest_module(stypy.reporting.localization.Localization(__file__, 39, 0), __file__, sys_modules_260667, sys_modules_260667.module_type_store, module_type_store)
    else:
        from matplotlib import backend_bases

        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib', None, module_type_store, ['backend_bases'], [backend_bases])

else:
    # Assigning a type to the variable 'matplotlib' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'matplotlib', import_260666)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'from matplotlib.backend_bases import _Backend' statement (line 40)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260668 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.backend_bases')

if (type(import_260668) is not StypyTypeError):

    if (import_260668 != 'pyd_module'):
        __import__(import_260668)
        sys_modules_260669 = sys.modules[import_260668]
        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.backend_bases', sys_modules_260669.module_type_store, module_type_store, ['_Backend'])
        nest_module(stypy.reporting.localization.Localization(__file__, 40, 0), __file__, sys_modules_260669, sys_modules_260669.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import _Backend

        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.backend_bases', None, module_type_store, ['_Backend'], [_Backend])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'matplotlib.backend_bases', import_260668)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 0))

# 'from matplotlib.figure import Figure' statement (line 41)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260670 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.figure')

if (type(import_260670) is not StypyTypeError):

    if (import_260670 != 'pyd_module'):
        __import__(import_260670)
        sys_modules_260671 = sys.modules[import_260670]
        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.figure', sys_modules_260671.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 41, 0), __file__, sys_modules_260671, sys_modules_260671.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'matplotlib.figure', import_260670)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 42, 0))

# 'from matplotlib._pylab_helpers import Gcf' statement (line 42)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260672 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'matplotlib._pylab_helpers')

if (type(import_260672) is not StypyTypeError):

    if (import_260672 != 'pyd_module'):
        __import__(import_260672)
        sys_modules_260673 = sys.modules[import_260672]
        import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'matplotlib._pylab_helpers', sys_modules_260673.module_type_store, module_type_store, ['Gcf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 42, 0), __file__, sys_modules_260673, sys_modules_260673.module_type_store, module_type_store)
    else:
        from matplotlib._pylab_helpers import Gcf

        import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'matplotlib._pylab_helpers', None, module_type_store, ['Gcf'], [Gcf])

else:
    # Assigning a type to the variable 'matplotlib._pylab_helpers' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'matplotlib._pylab_helpers', import_260672)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 43, 0))

# 'from matplotlib.backends import core' statement (line 43)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260674 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'matplotlib.backends')

if (type(import_260674) is not StypyTypeError):

    if (import_260674 != 'pyd_module'):
        __import__(import_260674)
        sys_modules_260675 = sys.modules[import_260674]
        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'matplotlib.backends', sys_modules_260675.module_type_store, module_type_store, ['backend_webagg_core'])
        nest_module(stypy.reporting.localization.Localization(__file__, 43, 0), __file__, sys_modules_260675, sys_modules_260675.module_type_store, module_type_store)
    else:
        from matplotlib.backends import backend_webagg_core as core

        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'matplotlib.backends', None, module_type_store, ['backend_webagg_core'], [core])

else:
    # Assigning a type to the variable 'matplotlib.backends' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'matplotlib.backends', import_260674)

# Adding an alias
module_type_store.add_alias('core', 'backend_webagg_core')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'from matplotlib.backends.backend_webagg_core import TimerTornado' statement (line 44)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_260676 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.backends.backend_webagg_core')

if (type(import_260676) is not StypyTypeError):

    if (import_260676 != 'pyd_module'):
        __import__(import_260676)
        sys_modules_260677 = sys.modules[import_260676]
        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.backends.backend_webagg_core', sys_modules_260677.module_type_store, module_type_store, ['TimerTornado'])
        nest_module(stypy.reporting.localization.Localization(__file__, 44, 0), __file__, sys_modules_260677, sys_modules_260677.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_webagg_core import TimerTornado

        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.backends.backend_webagg_core', None, module_type_store, ['TimerTornado'], [TimerTornado])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_webagg_core' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'matplotlib.backends.backend_webagg_core', import_260676)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# Declaration of the 'ServerThread' class
# Getting the type of 'threading' (line 47)
threading_260678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'threading')
# Obtaining the member 'Thread' of a type (line 47)
Thread_260679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 19), threading_260678, 'Thread')

class ServerThread(Thread_260679, ):

    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ServerThread.run.__dict__.__setitem__('stypy_localization', localization)
        ServerThread.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ServerThread.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        ServerThread.run.__dict__.__setitem__('stypy_function_name', 'ServerThread.run')
        ServerThread.run.__dict__.__setitem__('stypy_param_names_list', [])
        ServerThread.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        ServerThread.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ServerThread.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        ServerThread.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        ServerThread.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ServerThread.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ServerThread.run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        # Call to start(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_260687 = {}
        
        # Call to instance(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_260684 = {}
        # Getting the type of 'tornado' (line 49)
        tornado_260680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tornado', False)
        # Obtaining the member 'ioloop' of a type (line 49)
        ioloop_260681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), tornado_260680, 'ioloop')
        # Obtaining the member 'IOLoop' of a type (line 49)
        IOLoop_260682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), ioloop_260681, 'IOLoop')
        # Obtaining the member 'instance' of a type (line 49)
        instance_260683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), IOLoop_260682, 'instance')
        # Calling instance(args, kwargs) (line 49)
        instance_call_result_260685 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), instance_260683, *[], **kwargs_260684)
        
        # Obtaining the member 'start' of a type (line 49)
        start_260686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), instance_call_result_260685, 'start')
        # Calling start(args, kwargs) (line 49)
        start_call_result_260688 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), start_260686, *[], **kwargs_260687)
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_260689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_260689)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_260689


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 47, 0, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ServerThread.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ServerThread' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'ServerThread', ServerThread)

# Assigning a Call to a Name (line 51):

# Call to ServerThread(...): (line 51)
# Processing the call keyword arguments (line 51)
kwargs_260691 = {}
# Getting the type of 'ServerThread' (line 51)
ServerThread_260690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'ServerThread', False)
# Calling ServerThread(args, kwargs) (line 51)
ServerThread_call_result_260692 = invoke(stypy.reporting.localization.Localization(__file__, 51, 23), ServerThread_260690, *[], **kwargs_260691)

# Assigning a type to the variable 'webagg_server_thread' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'webagg_server_thread', ServerThread_call_result_260692)
# Declaration of the 'FigureCanvasWebAgg' class
# Getting the type of 'core' (line 54)
core_260693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'core')
# Obtaining the member 'FigureCanvasWebAggCore' of a type (line 54)
FigureCanvasWebAggCore_260694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 25), core_260693, 'FigureCanvasWebAggCore')

class FigureCanvasWebAgg(FigureCanvasWebAggCore_260694, ):

    @norecursion
    def show(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'show'
        module_type_store = module_type_store.open_function_context('show', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAgg.show.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAgg.show.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAgg.show.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAgg.show.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAgg.show')
        FigureCanvasWebAgg.show.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasWebAgg.show.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureCanvasWebAgg.show.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureCanvasWebAgg.show.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAgg.show.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAgg.show.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAgg.show.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAgg.show', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to show(...): (line 57)
        # Processing the call keyword arguments (line 57)
        kwargs_260696 = {}
        # Getting the type of 'show' (line 57)
        show_260695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'show', False)
        # Calling show(args, kwargs) (line 57)
        show_call_result_260697 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), show_260695, *[], **kwargs_260696)
        
        
        # ################# End of 'show(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'show' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_260698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_260698)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'show'
        return stypy_return_type_260698


    @norecursion
    def new_timer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_timer'
        module_type_store = module_type_store.open_function_context('new_timer', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasWebAgg.new_timer.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasWebAgg.new_timer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasWebAgg.new_timer.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasWebAgg.new_timer.__dict__.__setitem__('stypy_function_name', 'FigureCanvasWebAgg.new_timer')
        FigureCanvasWebAgg.new_timer.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasWebAgg.new_timer.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasWebAgg.new_timer.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasWebAgg.new_timer.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasWebAgg.new_timer.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasWebAgg.new_timer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasWebAgg.new_timer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAgg.new_timer', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Call to TimerTornado(...): (line 60)
        # Getting the type of 'args' (line 60)
        args_260700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'args', False)
        # Processing the call keyword arguments (line 60)
        # Getting the type of 'kwargs' (line 60)
        kwargs_260701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'kwargs', False)
        kwargs_260702 = {'kwargs_260701': kwargs_260701}
        # Getting the type of 'TimerTornado' (line 60)
        TimerTornado_260699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'TimerTornado', False)
        # Calling TimerTornado(args, kwargs) (line 60)
        TimerTornado_call_result_260703 = invoke(stypy.reporting.localization.Localization(__file__, 60, 15), TimerTornado_260699, *[args_260700], **kwargs_260702)
        
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', TimerTornado_call_result_260703)
        
        # ################# End of 'new_timer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_timer' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_260704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_260704)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_timer'
        return stypy_return_type_260704


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 54, 0, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasWebAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureCanvasWebAgg' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'FigureCanvasWebAgg', FigureCanvasWebAgg)
# Declaration of the 'WebAggApplication' class
# Getting the type of 'tornado' (line 63)
tornado_260705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'tornado')
# Obtaining the member 'web' of a type (line 63)
web_260706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), tornado_260705, 'web')
# Obtaining the member 'Application' of a type (line 63)
Application_260707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), web_260706, 'Application')

class WebAggApplication(Application_260707, ):
    # Declaration of the 'FavIcon' class
    # Getting the type of 'tornado' (line 67)
    tornado_260708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'tornado')
    # Obtaining the member 'web' of a type (line 67)
    web_260709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 18), tornado_260708, 'web')
    # Obtaining the member 'RequestHandler' of a type (line 67)
    RequestHandler_260710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 18), web_260709, 'RequestHandler')

    class FavIcon(RequestHandler_260710, ):

        @norecursion
        def get(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'get'
            module_type_store = module_type_store.open_function_context('get', 68, 8, False)
            # Assigning a type to the variable 'self' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            FavIcon.get.__dict__.__setitem__('stypy_localization', localization)
            FavIcon.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            FavIcon.get.__dict__.__setitem__('stypy_type_store', module_type_store)
            FavIcon.get.__dict__.__setitem__('stypy_function_name', 'FavIcon.get')
            FavIcon.get.__dict__.__setitem__('stypy_param_names_list', [])
            FavIcon.get.__dict__.__setitem__('stypy_varargs_param_name', None)
            FavIcon.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
            FavIcon.get.__dict__.__setitem__('stypy_call_defaults', defaults)
            FavIcon.get.__dict__.__setitem__('stypy_call_varargs', varargs)
            FavIcon.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            FavIcon.get.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'FavIcon.get', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'get', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'get(...)' code ##################

            
            # Assigning a Call to a Name (line 69):
            
            # Call to join(...): (line 69)
            # Processing the call arguments (line 69)
            
            # Call to dirname(...): (line 70)
            # Processing the call arguments (line 70)
            
            # Call to dirname(...): (line 70)
            # Processing the call arguments (line 70)
            # Getting the type of '__file__' (line 70)
            file___260720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 48), '__file__', False)
            # Processing the call keyword arguments (line 70)
            kwargs_260721 = {}
            # Getting the type of 'os' (line 70)
            os_260717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'os', False)
            # Obtaining the member 'path' of a type (line 70)
            path_260718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 32), os_260717, 'path')
            # Obtaining the member 'dirname' of a type (line 70)
            dirname_260719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 32), path_260718, 'dirname')
            # Calling dirname(args, kwargs) (line 70)
            dirname_call_result_260722 = invoke(stypy.reporting.localization.Localization(__file__, 70, 32), dirname_260719, *[file___260720], **kwargs_260721)
            
            # Processing the call keyword arguments (line 70)
            kwargs_260723 = {}
            # Getting the type of 'os' (line 70)
            os_260714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'os', False)
            # Obtaining the member 'path' of a type (line 70)
            path_260715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), os_260714, 'path')
            # Obtaining the member 'dirname' of a type (line 70)
            dirname_260716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), path_260715, 'dirname')
            # Calling dirname(args, kwargs) (line 70)
            dirname_call_result_260724 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), dirname_260716, *[dirname_call_result_260722], **kwargs_260723)
            
            unicode_260725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 16), 'unicode', u'mpl-data')
            unicode_260726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'unicode', u'images')
            # Processing the call keyword arguments (line 69)
            kwargs_260727 = {}
            # Getting the type of 'os' (line 69)
            os_260711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'os', False)
            # Obtaining the member 'path' of a type (line 69)
            path_260712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), os_260711, 'path')
            # Obtaining the member 'join' of a type (line 69)
            join_260713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), path_260712, 'join')
            # Calling join(args, kwargs) (line 69)
            join_call_result_260728 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), join_260713, *[dirname_call_result_260724, unicode_260725, unicode_260726], **kwargs_260727)
            
            # Assigning a type to the variable 'image_path' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'image_path', join_call_result_260728)
            
            # Call to set_header(...): (line 73)
            # Processing the call arguments (line 73)
            unicode_260731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'unicode', u'Content-Type')
            unicode_260732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 44), 'unicode', u'image/png')
            # Processing the call keyword arguments (line 73)
            kwargs_260733 = {}
            # Getting the type of 'self' (line 73)
            self_260729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'self', False)
            # Obtaining the member 'set_header' of a type (line 73)
            set_header_260730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), self_260729, 'set_header')
            # Calling set_header(args, kwargs) (line 73)
            set_header_call_result_260734 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), set_header_260730, *[unicode_260731, unicode_260732], **kwargs_260733)
            
            
            # Call to open(...): (line 74)
            # Processing the call arguments (line 74)
            
            # Call to join(...): (line 74)
            # Processing the call arguments (line 74)
            # Getting the type of 'image_path' (line 74)
            image_path_260739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 35), 'image_path', False)
            unicode_260740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 35), 'unicode', u'matplotlib.png')
            # Processing the call keyword arguments (line 74)
            kwargs_260741 = {}
            # Getting the type of 'os' (line 74)
            os_260736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'os', False)
            # Obtaining the member 'path' of a type (line 74)
            path_260737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 22), os_260736, 'path')
            # Obtaining the member 'join' of a type (line 74)
            join_260738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 22), path_260737, 'join')
            # Calling join(args, kwargs) (line 74)
            join_call_result_260742 = invoke(stypy.reporting.localization.Localization(__file__, 74, 22), join_260738, *[image_path_260739, unicode_260740], **kwargs_260741)
            
            unicode_260743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 54), 'unicode', u'rb')
            # Processing the call keyword arguments (line 74)
            kwargs_260744 = {}
            # Getting the type of 'open' (line 74)
            open_260735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'open', False)
            # Calling open(args, kwargs) (line 74)
            open_call_result_260745 = invoke(stypy.reporting.localization.Localization(__file__, 74, 17), open_260735, *[join_call_result_260742, unicode_260743], **kwargs_260744)
            
            with_260746 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 74, 17), open_call_result_260745, 'with parameter', '__enter__', '__exit__')

            if with_260746:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 74)
                enter___260747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 17), open_call_result_260745, '__enter__')
                with_enter_260748 = invoke(stypy.reporting.localization.Localization(__file__, 74, 17), enter___260747)
                # Assigning a type to the variable 'fd' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'fd', with_enter_260748)
                
                # Call to write(...): (line 76)
                # Processing the call arguments (line 76)
                
                # Call to read(...): (line 76)
                # Processing the call keyword arguments (line 76)
                kwargs_260753 = {}
                # Getting the type of 'fd' (line 76)
                fd_260751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'fd', False)
                # Obtaining the member 'read' of a type (line 76)
                read_260752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 27), fd_260751, 'read')
                # Calling read(args, kwargs) (line 76)
                read_call_result_260754 = invoke(stypy.reporting.localization.Localization(__file__, 76, 27), read_260752, *[], **kwargs_260753)
                
                # Processing the call keyword arguments (line 76)
                kwargs_260755 = {}
                # Getting the type of 'self' (line 76)
                self_260749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 76)
                write_260750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 16), self_260749, 'write')
                # Calling write(args, kwargs) (line 76)
                write_call_result_260756 = invoke(stypy.reporting.localization.Localization(__file__, 76, 16), write_260750, *[read_call_result_260754], **kwargs_260755)
                
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 74)
                exit___260757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 17), open_call_result_260745, '__exit__')
                with_exit_260758 = invoke(stypy.reporting.localization.Localization(__file__, 74, 17), exit___260757, None, None, None)

            
            # ################# End of 'get(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'get' in the type store
            # Getting the type of 'stypy_return_type' (line 68)
            stypy_return_type_260759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_260759)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'get'
            return stypy_return_type_260759

    
    # Assigning a type to the variable 'FavIcon' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'FavIcon', FavIcon)
    # Declaration of the 'SingleFigurePage' class
    # Getting the type of 'tornado' (line 78)
    tornado_260760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'tornado')
    # Obtaining the member 'web' of a type (line 78)
    web_260761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 27), tornado_260760, 'web')
    # Obtaining the member 'RequestHandler' of a type (line 78)
    RequestHandler_260762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 27), web_260761, 'RequestHandler')

    class SingleFigurePage(RequestHandler_260762, ):

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 79, 8, False)
            # Assigning a type to the variable 'self' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'SingleFigurePage.__init__', ['application', 'request'], None, 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['application', 'request'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Assigning a Call to a Attribute (line 80):
            
            # Call to pop(...): (line 80)
            # Processing the call arguments (line 80)
            unicode_260765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 41), 'unicode', u'url_prefix')
            unicode_260766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 55), 'unicode', u'')
            # Processing the call keyword arguments (line 80)
            kwargs_260767 = {}
            # Getting the type of 'kwargs' (line 80)
            kwargs_260763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'kwargs', False)
            # Obtaining the member 'pop' of a type (line 80)
            pop_260764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 30), kwargs_260763, 'pop')
            # Calling pop(args, kwargs) (line 80)
            pop_call_result_260768 = invoke(stypy.reporting.localization.Localization(__file__, 80, 30), pop_260764, *[unicode_260765, unicode_260766], **kwargs_260767)
            
            # Getting the type of 'self' (line 80)
            self_260769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'self')
            # Setting the type of the member 'url_prefix' of a type (line 80)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), self_260769, 'url_prefix', pop_call_result_260768)
            
            # Call to __init__(...): (line 81)
            # Processing the call arguments (line 81)
            # Getting the type of 'self' (line 81)
            self_260774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 55), 'self', False)
            # Getting the type of 'application' (line 81)
            application_260775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 61), 'application', False)
            # Getting the type of 'request' (line 82)
            request_260776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 55), 'request', False)
            # Processing the call keyword arguments (line 81)
            # Getting the type of 'kwargs' (line 82)
            kwargs_260777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 66), 'kwargs', False)
            kwargs_260778 = {'kwargs_260777': kwargs_260777}
            # Getting the type of 'tornado' (line 81)
            tornado_260770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'tornado', False)
            # Obtaining the member 'web' of a type (line 81)
            web_260771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 19), tornado_260770, 'web')
            # Obtaining the member 'RequestHandler' of a type (line 81)
            RequestHandler_260772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 19), web_260771, 'RequestHandler')
            # Obtaining the member '__init__' of a type (line 81)
            init___260773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 19), RequestHandler_260772, '__init__')
            # Calling __init__(args, kwargs) (line 81)
            init___call_result_260779 = invoke(stypy.reporting.localization.Localization(__file__, 81, 19), init___260773, *[self_260774, application_260775, request_260776], **kwargs_260778)
            
            # Assigning a type to the variable 'stypy_return_type' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', init___call_result_260779)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def get(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'get'
            module_type_store = module_type_store.open_function_context('get', 84, 8, False)
            # Assigning a type to the variable 'self' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            SingleFigurePage.get.__dict__.__setitem__('stypy_localization', localization)
            SingleFigurePage.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            SingleFigurePage.get.__dict__.__setitem__('stypy_type_store', module_type_store)
            SingleFigurePage.get.__dict__.__setitem__('stypy_function_name', 'SingleFigurePage.get')
            SingleFigurePage.get.__dict__.__setitem__('stypy_param_names_list', ['fignum'])
            SingleFigurePage.get.__dict__.__setitem__('stypy_varargs_param_name', None)
            SingleFigurePage.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
            SingleFigurePage.get.__dict__.__setitem__('stypy_call_defaults', defaults)
            SingleFigurePage.get.__dict__.__setitem__('stypy_call_varargs', varargs)
            SingleFigurePage.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            SingleFigurePage.get.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'SingleFigurePage.get', ['fignum'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'get', localization, ['fignum'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'get(...)' code ##################

            
            # Assigning a Call to a Name (line 85):
            
            # Call to int(...): (line 85)
            # Processing the call arguments (line 85)
            # Getting the type of 'fignum' (line 85)
            fignum_260781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'fignum', False)
            # Processing the call keyword arguments (line 85)
            kwargs_260782 = {}
            # Getting the type of 'int' (line 85)
            int_260780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'int', False)
            # Calling int(args, kwargs) (line 85)
            int_call_result_260783 = invoke(stypy.reporting.localization.Localization(__file__, 85, 21), int_260780, *[fignum_260781], **kwargs_260782)
            
            # Assigning a type to the variable 'fignum' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'fignum', int_call_result_260783)
            
            # Assigning a Call to a Name (line 86):
            
            # Call to get_fig_manager(...): (line 86)
            # Processing the call arguments (line 86)
            # Getting the type of 'fignum' (line 86)
            fignum_260786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 42), 'fignum', False)
            # Processing the call keyword arguments (line 86)
            kwargs_260787 = {}
            # Getting the type of 'Gcf' (line 86)
            Gcf_260784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), 'Gcf', False)
            # Obtaining the member 'get_fig_manager' of a type (line 86)
            get_fig_manager_260785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 22), Gcf_260784, 'get_fig_manager')
            # Calling get_fig_manager(args, kwargs) (line 86)
            get_fig_manager_call_result_260788 = invoke(stypy.reporting.localization.Localization(__file__, 86, 22), get_fig_manager_260785, *[fignum_260786], **kwargs_260787)
            
            # Assigning a type to the variable 'manager' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'manager', get_fig_manager_call_result_260788)
            
            # Assigning a Call to a Name (line 88):
            
            # Call to format(...): (line 88)
            # Processing the call keyword arguments (line 88)
            # Getting the type of 'self' (line 88)
            self_260791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 59), 'self', False)
            # Obtaining the member 'request' of a type (line 88)
            request_260792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 59), self_260791, 'request')
            keyword_260793 = request_260792
            # Getting the type of 'self' (line 89)
            self_260794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 62), 'self', False)
            # Obtaining the member 'url_prefix' of a type (line 89)
            url_prefix_260795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 62), self_260794, 'url_prefix')
            keyword_260796 = url_prefix_260795
            kwargs_260797 = {'prefix': keyword_260796, 'req': keyword_260793}
            unicode_260789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'unicode', u'ws://{req.host}{prefix}/')
            # Obtaining the member 'format' of a type (line 88)
            format_260790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 21), unicode_260789, 'format')
            # Calling format(args, kwargs) (line 88)
            format_call_result_260798 = invoke(stypy.reporting.localization.Localization(__file__, 88, 21), format_260790, *[], **kwargs_260797)
            
            # Assigning a type to the variable 'ws_uri' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'ws_uri', format_call_result_260798)
            
            # Call to render(...): (line 90)
            # Processing the call arguments (line 90)
            unicode_260801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 16), 'unicode', u'single_figure.html')
            # Processing the call keyword arguments (line 90)
            # Getting the type of 'self' (line 92)
            self_260802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'self', False)
            # Obtaining the member 'url_prefix' of a type (line 92)
            url_prefix_260803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 23), self_260802, 'url_prefix')
            keyword_260804 = url_prefix_260803
            # Getting the type of 'ws_uri' (line 93)
            ws_uri_260805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'ws_uri', False)
            keyword_260806 = ws_uri_260805
            # Getting the type of 'fignum' (line 94)
            fignum_260807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'fignum', False)
            keyword_260808 = fignum_260807
            # Getting the type of 'core' (line 95)
            core_260809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'core', False)
            # Obtaining the member 'NavigationToolbar2WebAgg' of a type (line 95)
            NavigationToolbar2WebAgg_260810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 26), core_260809, 'NavigationToolbar2WebAgg')
            # Obtaining the member 'toolitems' of a type (line 95)
            toolitems_260811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 26), NavigationToolbar2WebAgg_260810, 'toolitems')
            keyword_260812 = toolitems_260811
            # Getting the type of 'manager' (line 96)
            manager_260813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'manager', False)
            # Obtaining the member 'canvas' of a type (line 96)
            canvas_260814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 23), manager_260813, 'canvas')
            keyword_260815 = canvas_260814
            kwargs_260816 = {'fig_id': keyword_260808, 'toolitems': keyword_260812, 'prefix': keyword_260804, 'ws_uri': keyword_260806, 'canvas': keyword_260815}
            # Getting the type of 'self' (line 90)
            self_260799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'self', False)
            # Obtaining the member 'render' of a type (line 90)
            render_260800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), self_260799, 'render')
            # Calling render(args, kwargs) (line 90)
            render_call_result_260817 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), render_260800, *[unicode_260801], **kwargs_260816)
            
            
            # ################# End of 'get(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'get' in the type store
            # Getting the type of 'stypy_return_type' (line 84)
            stypy_return_type_260818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_260818)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'get'
            return stypy_return_type_260818

    
    # Assigning a type to the variable 'SingleFigurePage' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'SingleFigurePage', SingleFigurePage)
    # Declaration of the 'AllFiguresPage' class
    # Getting the type of 'tornado' (line 98)
    tornado_260819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'tornado')
    # Obtaining the member 'web' of a type (line 98)
    web_260820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 25), tornado_260819, 'web')
    # Obtaining the member 'RequestHandler' of a type (line 98)
    RequestHandler_260821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 25), web_260820, 'RequestHandler')

    class AllFiguresPage(RequestHandler_260821, ):

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 99, 8, False)
            # Assigning a type to the variable 'self' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'AllFiguresPage.__init__', ['application', 'request'], None, 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['application', 'request'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Assigning a Call to a Attribute (line 100):
            
            # Call to pop(...): (line 100)
            # Processing the call arguments (line 100)
            unicode_260824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 41), 'unicode', u'url_prefix')
            unicode_260825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 55), 'unicode', u'')
            # Processing the call keyword arguments (line 100)
            kwargs_260826 = {}
            # Getting the type of 'kwargs' (line 100)
            kwargs_260822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'kwargs', False)
            # Obtaining the member 'pop' of a type (line 100)
            pop_260823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 30), kwargs_260822, 'pop')
            # Calling pop(args, kwargs) (line 100)
            pop_call_result_260827 = invoke(stypy.reporting.localization.Localization(__file__, 100, 30), pop_260823, *[unicode_260824, unicode_260825], **kwargs_260826)
            
            # Getting the type of 'self' (line 100)
            self_260828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self')
            # Setting the type of the member 'url_prefix' of a type (line 100)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_260828, 'url_prefix', pop_call_result_260827)
            
            # Call to __init__(...): (line 101)
            # Processing the call arguments (line 101)
            # Getting the type of 'self' (line 101)
            self_260833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 55), 'self', False)
            # Getting the type of 'application' (line 101)
            application_260834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 61), 'application', False)
            # Getting the type of 'request' (line 102)
            request_260835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 55), 'request', False)
            # Processing the call keyword arguments (line 101)
            # Getting the type of 'kwargs' (line 102)
            kwargs_260836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 66), 'kwargs', False)
            kwargs_260837 = {'kwargs_260836': kwargs_260836}
            # Getting the type of 'tornado' (line 101)
            tornado_260829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'tornado', False)
            # Obtaining the member 'web' of a type (line 101)
            web_260830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 19), tornado_260829, 'web')
            # Obtaining the member 'RequestHandler' of a type (line 101)
            RequestHandler_260831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 19), web_260830, 'RequestHandler')
            # Obtaining the member '__init__' of a type (line 101)
            init___260832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 19), RequestHandler_260831, '__init__')
            # Calling __init__(args, kwargs) (line 101)
            init___call_result_260838 = invoke(stypy.reporting.localization.Localization(__file__, 101, 19), init___260832, *[self_260833, application_260834, request_260835], **kwargs_260837)
            
            # Assigning a type to the variable 'stypy_return_type' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type', init___call_result_260838)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def get(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'get'
            module_type_store = module_type_store.open_function_context('get', 104, 8, False)
            # Assigning a type to the variable 'self' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            AllFiguresPage.get.__dict__.__setitem__('stypy_localization', localization)
            AllFiguresPage.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            AllFiguresPage.get.__dict__.__setitem__('stypy_type_store', module_type_store)
            AllFiguresPage.get.__dict__.__setitem__('stypy_function_name', 'AllFiguresPage.get')
            AllFiguresPage.get.__dict__.__setitem__('stypy_param_names_list', [])
            AllFiguresPage.get.__dict__.__setitem__('stypy_varargs_param_name', None)
            AllFiguresPage.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
            AllFiguresPage.get.__dict__.__setitem__('stypy_call_defaults', defaults)
            AllFiguresPage.get.__dict__.__setitem__('stypy_call_varargs', varargs)
            AllFiguresPage.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            AllFiguresPage.get.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'AllFiguresPage.get', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'get', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'get(...)' code ##################

            
            # Assigning a Call to a Name (line 105):
            
            # Call to format(...): (line 105)
            # Processing the call keyword arguments (line 105)
            # Getting the type of 'self' (line 105)
            self_260841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 59), 'self', False)
            # Obtaining the member 'request' of a type (line 105)
            request_260842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 59), self_260841, 'request')
            keyword_260843 = request_260842
            # Getting the type of 'self' (line 106)
            self_260844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 62), 'self', False)
            # Obtaining the member 'url_prefix' of a type (line 106)
            url_prefix_260845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 62), self_260844, 'url_prefix')
            keyword_260846 = url_prefix_260845
            kwargs_260847 = {'prefix': keyword_260846, 'req': keyword_260843}
            unicode_260839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 21), 'unicode', u'ws://{req.host}{prefix}/')
            # Obtaining the member 'format' of a type (line 105)
            format_260840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 21), unicode_260839, 'format')
            # Calling format(args, kwargs) (line 105)
            format_call_result_260848 = invoke(stypy.reporting.localization.Localization(__file__, 105, 21), format_260840, *[], **kwargs_260847)
            
            # Assigning a type to the variable 'ws_uri' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'ws_uri', format_call_result_260848)
            
            # Call to render(...): (line 107)
            # Processing the call arguments (line 107)
            unicode_260851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 16), 'unicode', u'all_figures.html')
            # Processing the call keyword arguments (line 107)
            # Getting the type of 'self' (line 109)
            self_260852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'self', False)
            # Obtaining the member 'url_prefix' of a type (line 109)
            url_prefix_260853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 23), self_260852, 'url_prefix')
            keyword_260854 = url_prefix_260853
            # Getting the type of 'ws_uri' (line 110)
            ws_uri_260855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'ws_uri', False)
            keyword_260856 = ws_uri_260855
            
            # Call to sorted(...): (line 111)
            # Processing the call arguments (line 111)
            
            # Call to list(...): (line 112)
            # Processing the call arguments (line 112)
            
            # Call to items(...): (line 112)
            # Processing the call keyword arguments (line 112)
            kwargs_260862 = {}
            # Getting the type of 'Gcf' (line 112)
            Gcf_260859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 25), 'Gcf', False)
            # Obtaining the member 'figs' of a type (line 112)
            figs_260860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 25), Gcf_260859, 'figs')
            # Obtaining the member 'items' of a type (line 112)
            items_260861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 25), figs_260860, 'items')
            # Calling items(args, kwargs) (line 112)
            items_call_result_260863 = invoke(stypy.reporting.localization.Localization(__file__, 112, 25), items_260861, *[], **kwargs_260862)
            
            # Processing the call keyword arguments (line 112)
            kwargs_260864 = {}
            # Getting the type of 'list' (line 112)
            list_260858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'list', False)
            # Calling list(args, kwargs) (line 112)
            list_call_result_260865 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), list_260858, *[items_call_result_260863], **kwargs_260864)
            
            # Processing the call keyword arguments (line 111)

            @norecursion
            def _stypy_temp_lambda_111(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_111'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_111', 112, 48, True)
                # Passed parameters checking function
                _stypy_temp_lambda_111.stypy_localization = localization
                _stypy_temp_lambda_111.stypy_type_of_self = None
                _stypy_temp_lambda_111.stypy_type_store = module_type_store
                _stypy_temp_lambda_111.stypy_function_name = '_stypy_temp_lambda_111'
                _stypy_temp_lambda_111.stypy_param_names_list = ['item']
                _stypy_temp_lambda_111.stypy_varargs_param_name = None
                _stypy_temp_lambda_111.stypy_kwargs_param_name = None
                _stypy_temp_lambda_111.stypy_call_defaults = defaults
                _stypy_temp_lambda_111.stypy_call_varargs = varargs
                _stypy_temp_lambda_111.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_111', ['item'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_111', ['item'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Obtaining the type of the subscript
                int_260866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 66), 'int')
                # Getting the type of 'item' (line 112)
                item_260867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 61), 'item', False)
                # Obtaining the member '__getitem__' of a type (line 112)
                getitem___260868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 61), item_260867, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 112)
                subscript_call_result_260869 = invoke(stypy.reporting.localization.Localization(__file__, 112, 61), getitem___260868, int_260866)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 112)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 48), 'stypy_return_type', subscript_call_result_260869)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_111' in the type store
                # Getting the type of 'stypy_return_type' (line 112)
                stypy_return_type_260870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 48), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_260870)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_111'
                return stypy_return_type_260870

            # Assigning a type to the variable '_stypy_temp_lambda_111' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 48), '_stypy_temp_lambda_111', _stypy_temp_lambda_111)
            # Getting the type of '_stypy_temp_lambda_111' (line 112)
            _stypy_temp_lambda_111_260871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 48), '_stypy_temp_lambda_111')
            keyword_260872 = _stypy_temp_lambda_111_260871
            kwargs_260873 = {'key': keyword_260872}
            # Getting the type of 'sorted' (line 111)
            sorted_260857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'sorted', False)
            # Calling sorted(args, kwargs) (line 111)
            sorted_call_result_260874 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), sorted_260857, *[list_call_result_260865], **kwargs_260873)
            
            keyword_260875 = sorted_call_result_260874
            # Getting the type of 'core' (line 113)
            core_260876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 26), 'core', False)
            # Obtaining the member 'NavigationToolbar2WebAgg' of a type (line 113)
            NavigationToolbar2WebAgg_260877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 26), core_260876, 'NavigationToolbar2WebAgg')
            # Obtaining the member 'toolitems' of a type (line 113)
            toolitems_260878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 26), NavigationToolbar2WebAgg_260877, 'toolitems')
            keyword_260879 = toolitems_260878
            kwargs_260880 = {'toolitems': keyword_260879, 'prefix': keyword_260854, 'ws_uri': keyword_260856, 'figures': keyword_260875}
            # Getting the type of 'self' (line 107)
            self_260849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'self', False)
            # Obtaining the member 'render' of a type (line 107)
            render_260850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), self_260849, 'render')
            # Calling render(args, kwargs) (line 107)
            render_call_result_260881 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), render_260850, *[unicode_260851], **kwargs_260880)
            
            
            # ################# End of 'get(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'get' in the type store
            # Getting the type of 'stypy_return_type' (line 104)
            stypy_return_type_260882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_260882)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'get'
            return stypy_return_type_260882

    
    # Assigning a type to the variable 'AllFiguresPage' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'AllFiguresPage', AllFiguresPage)
    # Declaration of the 'MplJs' class
    # Getting the type of 'tornado' (line 115)
    tornado_260883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'tornado')
    # Obtaining the member 'web' of a type (line 115)
    web_260884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), tornado_260883, 'web')
    # Obtaining the member 'RequestHandler' of a type (line 115)
    RequestHandler_260885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), web_260884, 'RequestHandler')

    class MplJs(RequestHandler_260885, ):

        @norecursion
        def get(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'get'
            module_type_store = module_type_store.open_function_context('get', 116, 8, False)
            # Assigning a type to the variable 'self' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            MplJs.get.__dict__.__setitem__('stypy_localization', localization)
            MplJs.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            MplJs.get.__dict__.__setitem__('stypy_type_store', module_type_store)
            MplJs.get.__dict__.__setitem__('stypy_function_name', 'MplJs.get')
            MplJs.get.__dict__.__setitem__('stypy_param_names_list', [])
            MplJs.get.__dict__.__setitem__('stypy_varargs_param_name', None)
            MplJs.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
            MplJs.get.__dict__.__setitem__('stypy_call_defaults', defaults)
            MplJs.get.__dict__.__setitem__('stypy_call_varargs', varargs)
            MplJs.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            MplJs.get.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'MplJs.get', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'get', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'get(...)' code ##################

            
            # Call to set_header(...): (line 117)
            # Processing the call arguments (line 117)
            unicode_260888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 28), 'unicode', u'Content-Type')
            unicode_260889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 44), 'unicode', u'application/javascript')
            # Processing the call keyword arguments (line 117)
            kwargs_260890 = {}
            # Getting the type of 'self' (line 117)
            self_260886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'self', False)
            # Obtaining the member 'set_header' of a type (line 117)
            set_header_260887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), self_260886, 'set_header')
            # Calling set_header(args, kwargs) (line 117)
            set_header_call_result_260891 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), set_header_260887, *[unicode_260888, unicode_260889], **kwargs_260890)
            
            
            # Assigning a Call to a Name (line 119):
            
            # Call to get_javascript(...): (line 119)
            # Processing the call keyword arguments (line 119)
            kwargs_260895 = {}
            # Getting the type of 'core' (line 119)
            core_260892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'core', False)
            # Obtaining the member 'FigureManagerWebAgg' of a type (line 119)
            FigureManagerWebAgg_260893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 25), core_260892, 'FigureManagerWebAgg')
            # Obtaining the member 'get_javascript' of a type (line 119)
            get_javascript_260894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 25), FigureManagerWebAgg_260893, 'get_javascript')
            # Calling get_javascript(args, kwargs) (line 119)
            get_javascript_call_result_260896 = invoke(stypy.reporting.localization.Localization(__file__, 119, 25), get_javascript_260894, *[], **kwargs_260895)
            
            # Assigning a type to the variable 'js_content' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'js_content', get_javascript_call_result_260896)
            
            # Call to write(...): (line 121)
            # Processing the call arguments (line 121)
            # Getting the type of 'js_content' (line 121)
            js_content_260899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'js_content', False)
            # Processing the call keyword arguments (line 121)
            kwargs_260900 = {}
            # Getting the type of 'self' (line 121)
            self_260897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 121)
            write_260898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), self_260897, 'write')
            # Calling write(args, kwargs) (line 121)
            write_call_result_260901 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), write_260898, *[js_content_260899], **kwargs_260900)
            
            
            # ################# End of 'get(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'get' in the type store
            # Getting the type of 'stypy_return_type' (line 116)
            stypy_return_type_260902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_260902)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'get'
            return stypy_return_type_260902

    
    # Assigning a type to the variable 'MplJs' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'MplJs', MplJs)
    # Declaration of the 'Download' class
    # Getting the type of 'tornado' (line 123)
    tornado_260903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 19), 'tornado')
    # Obtaining the member 'web' of a type (line 123)
    web_260904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 19), tornado_260903, 'web')
    # Obtaining the member 'RequestHandler' of a type (line 123)
    RequestHandler_260905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 19), web_260904, 'RequestHandler')

    class Download(RequestHandler_260905, ):

        @norecursion
        def get(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'get'
            module_type_store = module_type_store.open_function_context('get', 124, 8, False)
            # Assigning a type to the variable 'self' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Download.get.__dict__.__setitem__('stypy_localization', localization)
            Download.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Download.get.__dict__.__setitem__('stypy_type_store', module_type_store)
            Download.get.__dict__.__setitem__('stypy_function_name', 'Download.get')
            Download.get.__dict__.__setitem__('stypy_param_names_list', ['fignum', 'fmt'])
            Download.get.__dict__.__setitem__('stypy_varargs_param_name', None)
            Download.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Download.get.__dict__.__setitem__('stypy_call_defaults', defaults)
            Download.get.__dict__.__setitem__('stypy_call_varargs', varargs)
            Download.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Download.get.__dict__.__setitem__('stypy_declared_arg_number', 3)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Download.get', ['fignum', 'fmt'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'get', localization, ['fignum', 'fmt'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'get(...)' code ##################

            
            # Assigning a Call to a Name (line 125):
            
            # Call to int(...): (line 125)
            # Processing the call arguments (line 125)
            # Getting the type of 'fignum' (line 125)
            fignum_260907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'fignum', False)
            # Processing the call keyword arguments (line 125)
            kwargs_260908 = {}
            # Getting the type of 'int' (line 125)
            int_260906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 21), 'int', False)
            # Calling int(args, kwargs) (line 125)
            int_call_result_260909 = invoke(stypy.reporting.localization.Localization(__file__, 125, 21), int_260906, *[fignum_260907], **kwargs_260908)
            
            # Assigning a type to the variable 'fignum' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'fignum', int_call_result_260909)
            
            # Assigning a Call to a Name (line 126):
            
            # Call to get_fig_manager(...): (line 126)
            # Processing the call arguments (line 126)
            # Getting the type of 'fignum' (line 126)
            fignum_260912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 42), 'fignum', False)
            # Processing the call keyword arguments (line 126)
            kwargs_260913 = {}
            # Getting the type of 'Gcf' (line 126)
            Gcf_260910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'Gcf', False)
            # Obtaining the member 'get_fig_manager' of a type (line 126)
            get_fig_manager_260911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 22), Gcf_260910, 'get_fig_manager')
            # Calling get_fig_manager(args, kwargs) (line 126)
            get_fig_manager_call_result_260914 = invoke(stypy.reporting.localization.Localization(__file__, 126, 22), get_fig_manager_260911, *[fignum_260912], **kwargs_260913)
            
            # Assigning a type to the variable 'manager' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'manager', get_fig_manager_call_result_260914)
            
            # Assigning a Dict to a Name (line 129):
            
            # Obtaining an instance of the builtin type 'dict' (line 129)
            dict_260915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 24), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 129)
            # Adding element type (key, value) (line 129)
            unicode_260916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'unicode', u'ps')
            unicode_260917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 22), 'unicode', u'application/postscript')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), dict_260915, (unicode_260916, unicode_260917))
            # Adding element type (key, value) (line 129)
            unicode_260918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 16), 'unicode', u'eps')
            unicode_260919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 23), 'unicode', u'application/postscript')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), dict_260915, (unicode_260918, unicode_260919))
            # Adding element type (key, value) (line 129)
            unicode_260920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 16), 'unicode', u'pdf')
            unicode_260921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 23), 'unicode', u'application/pdf')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), dict_260915, (unicode_260920, unicode_260921))
            # Adding element type (key, value) (line 129)
            unicode_260922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 16), 'unicode', u'svg')
            unicode_260923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 23), 'unicode', u'image/svg+xml')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), dict_260915, (unicode_260922, unicode_260923))
            # Adding element type (key, value) (line 129)
            unicode_260924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 16), 'unicode', u'png')
            unicode_260925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 23), 'unicode', u'image/png')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), dict_260915, (unicode_260924, unicode_260925))
            # Adding element type (key, value) (line 129)
            unicode_260926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 16), 'unicode', u'jpeg')
            unicode_260927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 24), 'unicode', u'image/jpeg')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), dict_260915, (unicode_260926, unicode_260927))
            # Adding element type (key, value) (line 129)
            unicode_260928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 16), 'unicode', u'tif')
            unicode_260929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'unicode', u'image/tiff')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), dict_260915, (unicode_260928, unicode_260929))
            # Adding element type (key, value) (line 129)
            unicode_260930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 16), 'unicode', u'emf')
            unicode_260931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'unicode', u'application/emf')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), dict_260915, (unicode_260930, unicode_260931))
            
            # Assigning a type to the variable 'mimetypes' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'mimetypes', dict_260915)
            
            # Call to set_header(...): (line 140)
            # Processing the call arguments (line 140)
            unicode_260934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 28), 'unicode', u'Content-Type')
            
            # Call to get(...): (line 140)
            # Processing the call arguments (line 140)
            # Getting the type of 'fmt' (line 140)
            fmt_260937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 58), 'fmt', False)
            unicode_260938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 63), 'unicode', u'binary')
            # Processing the call keyword arguments (line 140)
            kwargs_260939 = {}
            # Getting the type of 'mimetypes' (line 140)
            mimetypes_260935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 44), 'mimetypes', False)
            # Obtaining the member 'get' of a type (line 140)
            get_260936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 44), mimetypes_260935, 'get')
            # Calling get(args, kwargs) (line 140)
            get_call_result_260940 = invoke(stypy.reporting.localization.Localization(__file__, 140, 44), get_260936, *[fmt_260937, unicode_260938], **kwargs_260939)
            
            # Processing the call keyword arguments (line 140)
            kwargs_260941 = {}
            # Getting the type of 'self' (line 140)
            self_260932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'self', False)
            # Obtaining the member 'set_header' of a type (line 140)
            set_header_260933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), self_260932, 'set_header')
            # Calling set_header(args, kwargs) (line 140)
            set_header_call_result_260942 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), set_header_260933, *[unicode_260934, get_call_result_260940], **kwargs_260941)
            
            
            # Assigning a Call to a Name (line 142):
            
            # Call to BytesIO(...): (line 142)
            # Processing the call keyword arguments (line 142)
            kwargs_260945 = {}
            # Getting the type of 'six' (line 142)
            six_260943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'six', False)
            # Obtaining the member 'BytesIO' of a type (line 142)
            BytesIO_260944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 19), six_260943, 'BytesIO')
            # Calling BytesIO(args, kwargs) (line 142)
            BytesIO_call_result_260946 = invoke(stypy.reporting.localization.Localization(__file__, 142, 19), BytesIO_260944, *[], **kwargs_260945)
            
            # Assigning a type to the variable 'buff' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'buff', BytesIO_call_result_260946)
            
            # Call to savefig(...): (line 143)
            # Processing the call arguments (line 143)
            # Getting the type of 'buff' (line 143)
            buff_260951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 42), 'buff', False)
            # Processing the call keyword arguments (line 143)
            # Getting the type of 'fmt' (line 143)
            fmt_260952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 55), 'fmt', False)
            keyword_260953 = fmt_260952
            kwargs_260954 = {'format': keyword_260953}
            # Getting the type of 'manager' (line 143)
            manager_260947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'manager', False)
            # Obtaining the member 'canvas' of a type (line 143)
            canvas_260948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), manager_260947, 'canvas')
            # Obtaining the member 'figure' of a type (line 143)
            figure_260949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), canvas_260948, 'figure')
            # Obtaining the member 'savefig' of a type (line 143)
            savefig_260950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), figure_260949, 'savefig')
            # Calling savefig(args, kwargs) (line 143)
            savefig_call_result_260955 = invoke(stypy.reporting.localization.Localization(__file__, 143, 12), savefig_260950, *[buff_260951], **kwargs_260954)
            
            
            # Call to write(...): (line 144)
            # Processing the call arguments (line 144)
            
            # Call to getvalue(...): (line 144)
            # Processing the call keyword arguments (line 144)
            kwargs_260960 = {}
            # Getting the type of 'buff' (line 144)
            buff_260958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 23), 'buff', False)
            # Obtaining the member 'getvalue' of a type (line 144)
            getvalue_260959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 23), buff_260958, 'getvalue')
            # Calling getvalue(args, kwargs) (line 144)
            getvalue_call_result_260961 = invoke(stypy.reporting.localization.Localization(__file__, 144, 23), getvalue_260959, *[], **kwargs_260960)
            
            # Processing the call keyword arguments (line 144)
            kwargs_260962 = {}
            # Getting the type of 'self' (line 144)
            self_260956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 144)
            write_260957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), self_260956, 'write')
            # Calling write(args, kwargs) (line 144)
            write_call_result_260963 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), write_260957, *[getvalue_call_result_260961], **kwargs_260962)
            
            
            # ################# End of 'get(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'get' in the type store
            # Getting the type of 'stypy_return_type' (line 124)
            stypy_return_type_260964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_260964)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'get'
            return stypy_return_type_260964

    
    # Assigning a type to the variable 'Download' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'Download', Download)
    # Declaration of the 'WebSocket' class
    # Getting the type of 'tornado' (line 146)
    tornado_260965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'tornado')
    # Obtaining the member 'websocket' of a type (line 146)
    websocket_260966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 20), tornado_260965, 'websocket')
    # Obtaining the member 'WebSocketHandler' of a type (line 146)
    WebSocketHandler_260967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 20), websocket_260966, 'WebSocketHandler')

    class WebSocket(WebSocketHandler_260967, ):
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'True' (line 147)
        True_260968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 26), 'True')
        # Assigning a type to the variable 'supports_binary' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'supports_binary', True_260968)

        @norecursion
        def open(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'open'
            module_type_store = module_type_store.open_function_context('open', 149, 8, False)
            # Assigning a type to the variable 'self' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            WebSocket.open.__dict__.__setitem__('stypy_localization', localization)
            WebSocket.open.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            WebSocket.open.__dict__.__setitem__('stypy_type_store', module_type_store)
            WebSocket.open.__dict__.__setitem__('stypy_function_name', 'WebSocket.open')
            WebSocket.open.__dict__.__setitem__('stypy_param_names_list', ['fignum'])
            WebSocket.open.__dict__.__setitem__('stypy_varargs_param_name', None)
            WebSocket.open.__dict__.__setitem__('stypy_kwargs_param_name', None)
            WebSocket.open.__dict__.__setitem__('stypy_call_defaults', defaults)
            WebSocket.open.__dict__.__setitem__('stypy_call_varargs', varargs)
            WebSocket.open.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            WebSocket.open.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'WebSocket.open', ['fignum'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'open', localization, ['fignum'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'open(...)' code ##################

            
            # Assigning a Call to a Attribute (line 150):
            
            # Call to int(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'fignum' (line 150)
            fignum_260970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'fignum', False)
            # Processing the call keyword arguments (line 150)
            kwargs_260971 = {}
            # Getting the type of 'int' (line 150)
            int_260969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'int', False)
            # Calling int(args, kwargs) (line 150)
            int_call_result_260972 = invoke(stypy.reporting.localization.Localization(__file__, 150, 26), int_260969, *[fignum_260970], **kwargs_260971)
            
            # Getting the type of 'self' (line 150)
            self_260973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'self')
            # Setting the type of the member 'fignum' of a type (line 150)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), self_260973, 'fignum', int_call_result_260972)
            
            # Assigning a Call to a Attribute (line 151):
            
            # Call to get_fig_manager(...): (line 151)
            # Processing the call arguments (line 151)
            # Getting the type of 'self' (line 151)
            self_260976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 47), 'self', False)
            # Obtaining the member 'fignum' of a type (line 151)
            fignum_260977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 47), self_260976, 'fignum')
            # Processing the call keyword arguments (line 151)
            kwargs_260978 = {}
            # Getting the type of 'Gcf' (line 151)
            Gcf_260974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 27), 'Gcf', False)
            # Obtaining the member 'get_fig_manager' of a type (line 151)
            get_fig_manager_260975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 27), Gcf_260974, 'get_fig_manager')
            # Calling get_fig_manager(args, kwargs) (line 151)
            get_fig_manager_call_result_260979 = invoke(stypy.reporting.localization.Localization(__file__, 151, 27), get_fig_manager_260975, *[fignum_260977], **kwargs_260978)
            
            # Getting the type of 'self' (line 151)
            self_260980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'self')
            # Setting the type of the member 'manager' of a type (line 151)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), self_260980, 'manager', get_fig_manager_call_result_260979)
            
            # Call to add_web_socket(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'self' (line 152)
            self_260984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 40), 'self', False)
            # Processing the call keyword arguments (line 152)
            kwargs_260985 = {}
            # Getting the type of 'self' (line 152)
            self_260981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'self', False)
            # Obtaining the member 'manager' of a type (line 152)
            manager_260982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), self_260981, 'manager')
            # Obtaining the member 'add_web_socket' of a type (line 152)
            add_web_socket_260983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), manager_260982, 'add_web_socket')
            # Calling add_web_socket(args, kwargs) (line 152)
            add_web_socket_call_result_260986 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), add_web_socket_260983, *[self_260984], **kwargs_260985)
            
            
            # Type idiom detected: calculating its left and rigth part (line 153)
            unicode_260987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 29), 'unicode', u'set_nodelay')
            # Getting the type of 'self' (line 153)
            self_260988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'self')
            
            (may_be_260989, more_types_in_union_260990) = may_provide_member(unicode_260987, self_260988)

            if may_be_260989:

                if more_types_in_union_260990:
                    # Runtime conditional SSA (line 153)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'self' (line 153)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'self', remove_not_member_provider_from_union(self_260988, u'set_nodelay'))
                
                # Call to set_nodelay(...): (line 154)
                # Processing the call arguments (line 154)
                # Getting the type of 'True' (line 154)
                True_260993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 33), 'True', False)
                # Processing the call keyword arguments (line 154)
                kwargs_260994 = {}
                # Getting the type of 'self' (line 154)
                self_260991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'self', False)
                # Obtaining the member 'set_nodelay' of a type (line 154)
                set_nodelay_260992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 16), self_260991, 'set_nodelay')
                # Calling set_nodelay(args, kwargs) (line 154)
                set_nodelay_call_result_260995 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), set_nodelay_260992, *[True_260993], **kwargs_260994)
                

                if more_types_in_union_260990:
                    # SSA join for if statement (line 153)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # ################# End of 'open(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'open' in the type store
            # Getting the type of 'stypy_return_type' (line 149)
            stypy_return_type_260996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_260996)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'open'
            return stypy_return_type_260996


        @norecursion
        def on_close(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'on_close'
            module_type_store = module_type_store.open_function_context('on_close', 156, 8, False)
            # Assigning a type to the variable 'self' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            WebSocket.on_close.__dict__.__setitem__('stypy_localization', localization)
            WebSocket.on_close.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            WebSocket.on_close.__dict__.__setitem__('stypy_type_store', module_type_store)
            WebSocket.on_close.__dict__.__setitem__('stypy_function_name', 'WebSocket.on_close')
            WebSocket.on_close.__dict__.__setitem__('stypy_param_names_list', [])
            WebSocket.on_close.__dict__.__setitem__('stypy_varargs_param_name', None)
            WebSocket.on_close.__dict__.__setitem__('stypy_kwargs_param_name', None)
            WebSocket.on_close.__dict__.__setitem__('stypy_call_defaults', defaults)
            WebSocket.on_close.__dict__.__setitem__('stypy_call_varargs', varargs)
            WebSocket.on_close.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            WebSocket.on_close.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'WebSocket.on_close', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'on_close', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'on_close(...)' code ##################

            
            # Call to remove_web_socket(...): (line 157)
            # Processing the call arguments (line 157)
            # Getting the type of 'self' (line 157)
            self_261000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 43), 'self', False)
            # Processing the call keyword arguments (line 157)
            kwargs_261001 = {}
            # Getting the type of 'self' (line 157)
            self_260997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'self', False)
            # Obtaining the member 'manager' of a type (line 157)
            manager_260998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), self_260997, 'manager')
            # Obtaining the member 'remove_web_socket' of a type (line 157)
            remove_web_socket_260999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), manager_260998, 'remove_web_socket')
            # Calling remove_web_socket(args, kwargs) (line 157)
            remove_web_socket_call_result_261002 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), remove_web_socket_260999, *[self_261000], **kwargs_261001)
            
            
            # ################# End of 'on_close(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'on_close' in the type store
            # Getting the type of 'stypy_return_type' (line 156)
            stypy_return_type_261003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_261003)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'on_close'
            return stypy_return_type_261003


        @norecursion
        def on_message(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'on_message'
            module_type_store = module_type_store.open_function_context('on_message', 159, 8, False)
            # Assigning a type to the variable 'self' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            WebSocket.on_message.__dict__.__setitem__('stypy_localization', localization)
            WebSocket.on_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            WebSocket.on_message.__dict__.__setitem__('stypy_type_store', module_type_store)
            WebSocket.on_message.__dict__.__setitem__('stypy_function_name', 'WebSocket.on_message')
            WebSocket.on_message.__dict__.__setitem__('stypy_param_names_list', ['message'])
            WebSocket.on_message.__dict__.__setitem__('stypy_varargs_param_name', None)
            WebSocket.on_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
            WebSocket.on_message.__dict__.__setitem__('stypy_call_defaults', defaults)
            WebSocket.on_message.__dict__.__setitem__('stypy_call_varargs', varargs)
            WebSocket.on_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            WebSocket.on_message.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'WebSocket.on_message', ['message'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'on_message', localization, ['message'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'on_message(...)' code ##################

            
            # Assigning a Call to a Name (line 160):
            
            # Call to loads(...): (line 160)
            # Processing the call arguments (line 160)
            # Getting the type of 'message' (line 160)
            message_261006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 33), 'message', False)
            # Processing the call keyword arguments (line 160)
            kwargs_261007 = {}
            # Getting the type of 'json' (line 160)
            json_261004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 22), 'json', False)
            # Obtaining the member 'loads' of a type (line 160)
            loads_261005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 22), json_261004, 'loads')
            # Calling loads(args, kwargs) (line 160)
            loads_call_result_261008 = invoke(stypy.reporting.localization.Localization(__file__, 160, 22), loads_261005, *[message_261006], **kwargs_261007)
            
            # Assigning a type to the variable 'message' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'message', loads_call_result_261008)
            
            
            
            # Obtaining the type of the subscript
            unicode_261009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 23), 'unicode', u'type')
            # Getting the type of 'message' (line 164)
            message_261010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'message')
            # Obtaining the member '__getitem__' of a type (line 164)
            getitem___261011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 15), message_261010, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 164)
            subscript_call_result_261012 = invoke(stypy.reporting.localization.Localization(__file__, 164, 15), getitem___261011, unicode_261009)
            
            unicode_261013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 34), 'unicode', u'supports_binary')
            # Applying the binary operator '==' (line 164)
            result_eq_261014 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 15), '==', subscript_call_result_261012, unicode_261013)
            
            # Testing the type of an if condition (line 164)
            if_condition_261015 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 12), result_eq_261014)
            # Assigning a type to the variable 'if_condition_261015' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'if_condition_261015', if_condition_261015)
            # SSA begins for if statement (line 164)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Attribute (line 165):
            
            # Obtaining the type of the subscript
            unicode_261016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 47), 'unicode', u'value')
            # Getting the type of 'message' (line 165)
            message_261017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 39), 'message')
            # Obtaining the member '__getitem__' of a type (line 165)
            getitem___261018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 39), message_261017, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 165)
            subscript_call_result_261019 = invoke(stypy.reporting.localization.Localization(__file__, 165, 39), getitem___261018, unicode_261016)
            
            # Getting the type of 'self' (line 165)
            self_261020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'self')
            # Setting the type of the member 'supports_binary' of a type (line 165)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), self_261020, 'supports_binary', subscript_call_result_261019)
            # SSA branch for the else part of an if statement (line 164)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 167):
            
            # Call to get_fig_manager(...): (line 167)
            # Processing the call arguments (line 167)
            # Getting the type of 'self' (line 167)
            self_261023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 46), 'self', False)
            # Obtaining the member 'fignum' of a type (line 167)
            fignum_261024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 46), self_261023, 'fignum')
            # Processing the call keyword arguments (line 167)
            kwargs_261025 = {}
            # Getting the type of 'Gcf' (line 167)
            Gcf_261021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 26), 'Gcf', False)
            # Obtaining the member 'get_fig_manager' of a type (line 167)
            get_fig_manager_261022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 26), Gcf_261021, 'get_fig_manager')
            # Calling get_fig_manager(args, kwargs) (line 167)
            get_fig_manager_call_result_261026 = invoke(stypy.reporting.localization.Localization(__file__, 167, 26), get_fig_manager_261022, *[fignum_261024], **kwargs_261025)
            
            # Assigning a type to the variable 'manager' (line 167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'manager', get_fig_manager_call_result_261026)
            
            # Type idiom detected: calculating its left and rigth part (line 171)
            # Getting the type of 'manager' (line 171)
            manager_261027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'manager')
            # Getting the type of 'None' (line 171)
            None_261028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 34), 'None')
            
            (may_be_261029, more_types_in_union_261030) = may_not_be_none(manager_261027, None_261028)

            if may_be_261029:

                if more_types_in_union_261030:
                    # Runtime conditional SSA (line 171)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to handle_json(...): (line 172)
                # Processing the call arguments (line 172)
                # Getting the type of 'message' (line 172)
                message_261033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 40), 'message', False)
                # Processing the call keyword arguments (line 172)
                kwargs_261034 = {}
                # Getting the type of 'manager' (line 172)
                manager_261031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'manager', False)
                # Obtaining the member 'handle_json' of a type (line 172)
                handle_json_261032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 20), manager_261031, 'handle_json')
                # Calling handle_json(args, kwargs) (line 172)
                handle_json_call_result_261035 = invoke(stypy.reporting.localization.Localization(__file__, 172, 20), handle_json_261032, *[message_261033], **kwargs_261034)
                

                if more_types_in_union_261030:
                    # SSA join for if statement (line 171)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 164)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'on_message(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'on_message' in the type store
            # Getting the type of 'stypy_return_type' (line 159)
            stypy_return_type_261036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_261036)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'on_message'
            return stypy_return_type_261036


        @norecursion
        def send_json(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'send_json'
            module_type_store = module_type_store.open_function_context('send_json', 174, 8, False)
            # Assigning a type to the variable 'self' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            WebSocket.send_json.__dict__.__setitem__('stypy_localization', localization)
            WebSocket.send_json.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            WebSocket.send_json.__dict__.__setitem__('stypy_type_store', module_type_store)
            WebSocket.send_json.__dict__.__setitem__('stypy_function_name', 'WebSocket.send_json')
            WebSocket.send_json.__dict__.__setitem__('stypy_param_names_list', ['content'])
            WebSocket.send_json.__dict__.__setitem__('stypy_varargs_param_name', None)
            WebSocket.send_json.__dict__.__setitem__('stypy_kwargs_param_name', None)
            WebSocket.send_json.__dict__.__setitem__('stypy_call_defaults', defaults)
            WebSocket.send_json.__dict__.__setitem__('stypy_call_varargs', varargs)
            WebSocket.send_json.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            WebSocket.send_json.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'WebSocket.send_json', ['content'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'send_json', localization, ['content'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'send_json(...)' code ##################

            
            # Call to write_message(...): (line 175)
            # Processing the call arguments (line 175)
            
            # Call to dumps(...): (line 175)
            # Processing the call arguments (line 175)
            # Getting the type of 'content' (line 175)
            content_261041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 42), 'content', False)
            # Processing the call keyword arguments (line 175)
            kwargs_261042 = {}
            # Getting the type of 'json' (line 175)
            json_261039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 31), 'json', False)
            # Obtaining the member 'dumps' of a type (line 175)
            dumps_261040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 31), json_261039, 'dumps')
            # Calling dumps(args, kwargs) (line 175)
            dumps_call_result_261043 = invoke(stypy.reporting.localization.Localization(__file__, 175, 31), dumps_261040, *[content_261041], **kwargs_261042)
            
            # Processing the call keyword arguments (line 175)
            kwargs_261044 = {}
            # Getting the type of 'self' (line 175)
            self_261037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'self', False)
            # Obtaining the member 'write_message' of a type (line 175)
            write_message_261038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), self_261037, 'write_message')
            # Calling write_message(args, kwargs) (line 175)
            write_message_call_result_261045 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), write_message_261038, *[dumps_call_result_261043], **kwargs_261044)
            
            
            # ################# End of 'send_json(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'send_json' in the type store
            # Getting the type of 'stypy_return_type' (line 174)
            stypy_return_type_261046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_261046)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'send_json'
            return stypy_return_type_261046


        @norecursion
        def send_binary(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'send_binary'
            module_type_store = module_type_store.open_function_context('send_binary', 177, 8, False)
            # Assigning a type to the variable 'self' (line 178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            WebSocket.send_binary.__dict__.__setitem__('stypy_localization', localization)
            WebSocket.send_binary.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            WebSocket.send_binary.__dict__.__setitem__('stypy_type_store', module_type_store)
            WebSocket.send_binary.__dict__.__setitem__('stypy_function_name', 'WebSocket.send_binary')
            WebSocket.send_binary.__dict__.__setitem__('stypy_param_names_list', ['blob'])
            WebSocket.send_binary.__dict__.__setitem__('stypy_varargs_param_name', None)
            WebSocket.send_binary.__dict__.__setitem__('stypy_kwargs_param_name', None)
            WebSocket.send_binary.__dict__.__setitem__('stypy_call_defaults', defaults)
            WebSocket.send_binary.__dict__.__setitem__('stypy_call_varargs', varargs)
            WebSocket.send_binary.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            WebSocket.send_binary.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'WebSocket.send_binary', ['blob'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'send_binary', localization, ['blob'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'send_binary(...)' code ##################

            
            # Getting the type of 'self' (line 178)
            self_261047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'self')
            # Obtaining the member 'supports_binary' of a type (line 178)
            supports_binary_261048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 15), self_261047, 'supports_binary')
            # Testing the type of an if condition (line 178)
            if_condition_261049 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 12), supports_binary_261048)
            # Assigning a type to the variable 'if_condition_261049' (line 178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'if_condition_261049', if_condition_261049)
            # SSA begins for if statement (line 178)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write_message(...): (line 179)
            # Processing the call arguments (line 179)
            # Getting the type of 'blob' (line 179)
            blob_261052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 35), 'blob', False)
            # Processing the call keyword arguments (line 179)
            # Getting the type of 'True' (line 179)
            True_261053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 48), 'True', False)
            keyword_261054 = True_261053
            kwargs_261055 = {'binary': keyword_261054}
            # Getting the type of 'self' (line 179)
            self_261050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'self', False)
            # Obtaining the member 'write_message' of a type (line 179)
            write_message_261051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 16), self_261050, 'write_message')
            # Calling write_message(args, kwargs) (line 179)
            write_message_call_result_261056 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), write_message_261051, *[blob_261052], **kwargs_261055)
            
            # SSA branch for the else part of an if statement (line 178)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 181):
            
            # Call to format(...): (line 181)
            # Processing the call arguments (line 181)
            
            # Call to replace(...): (line 182)
            # Processing the call arguments (line 182)
            unicode_261065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 50), 'unicode', u'\n')
            unicode_261066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 56), 'unicode', u'')
            # Processing the call keyword arguments (line 182)
            kwargs_261067 = {}
            
            # Call to encode(...): (line 182)
            # Processing the call arguments (line 182)
            unicode_261061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 32), 'unicode', u'base64')
            # Processing the call keyword arguments (line 182)
            kwargs_261062 = {}
            # Getting the type of 'blob' (line 182)
            blob_261059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'blob', False)
            # Obtaining the member 'encode' of a type (line 182)
            encode_261060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 20), blob_261059, 'encode')
            # Calling encode(args, kwargs) (line 182)
            encode_call_result_261063 = invoke(stypy.reporting.localization.Localization(__file__, 182, 20), encode_261060, *[unicode_261061], **kwargs_261062)
            
            # Obtaining the member 'replace' of a type (line 182)
            replace_261064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 20), encode_call_result_261063, 'replace')
            # Calling replace(args, kwargs) (line 182)
            replace_call_result_261068 = invoke(stypy.reporting.localization.Localization(__file__, 182, 20), replace_261064, *[unicode_261065, unicode_261066], **kwargs_261067)
            
            # Processing the call keyword arguments (line 181)
            kwargs_261069 = {}
            unicode_261057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 27), 'unicode', u'data:image/png;base64,{0}')
            # Obtaining the member 'format' of a type (line 181)
            format_261058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 27), unicode_261057, 'format')
            # Calling format(args, kwargs) (line 181)
            format_call_result_261070 = invoke(stypy.reporting.localization.Localization(__file__, 181, 27), format_261058, *[replace_call_result_261068], **kwargs_261069)
            
            # Assigning a type to the variable 'data_uri' (line 181)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'data_uri', format_call_result_261070)
            
            # Call to write_message(...): (line 183)
            # Processing the call arguments (line 183)
            # Getting the type of 'data_uri' (line 183)
            data_uri_261073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 35), 'data_uri', False)
            # Processing the call keyword arguments (line 183)
            kwargs_261074 = {}
            # Getting the type of 'self' (line 183)
            self_261071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'self', False)
            # Obtaining the member 'write_message' of a type (line 183)
            write_message_261072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 16), self_261071, 'write_message')
            # Calling write_message(args, kwargs) (line 183)
            write_message_call_result_261075 = invoke(stypy.reporting.localization.Localization(__file__, 183, 16), write_message_261072, *[data_uri_261073], **kwargs_261074)
            
            # SSA join for if statement (line 178)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'send_binary(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'send_binary' in the type store
            # Getting the type of 'stypy_return_type' (line 177)
            stypy_return_type_261076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_261076)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'send_binary'
            return stypy_return_type_261076

    
    # Assigning a type to the variable 'WebSocket' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'WebSocket', WebSocket)

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_261077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 34), 'unicode', u'')
        defaults = [unicode_261077]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'WebAggApplication.__init__', ['url_prefix'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['url_prefix'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Getting the type of 'url_prefix' (line 186)
        url_prefix_261078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'url_prefix')
        # Testing the type of an if condition (line 186)
        if_condition_261079 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 8), url_prefix_261078)
        # Assigning a type to the variable 'if_condition_261079' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'if_condition_261079', if_condition_261079)
        # SSA begins for if statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Evaluating assert statement condition
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        int_261080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 30), 'int')
        # Getting the type of 'url_prefix' (line 187)
        url_prefix_261081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'url_prefix')
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___261082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 19), url_prefix_261081, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_261083 = invoke(stypy.reporting.localization.Localization(__file__, 187, 19), getitem___261082, int_261080)
        
        unicode_261084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 36), 'unicode', u'/')
        # Applying the binary operator '==' (line 187)
        result_eq_261085 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 19), '==', subscript_call_result_261083, unicode_261084)
        
        
        
        # Obtaining the type of the subscript
        int_261086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 55), 'int')
        # Getting the type of 'url_prefix' (line 187)
        url_prefix_261087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 44), 'url_prefix')
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___261088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 44), url_prefix_261087, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_261089 = invoke(stypy.reporting.localization.Localization(__file__, 187, 44), getitem___261088, int_261086)
        
        unicode_261090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 62), 'unicode', u'/')
        # Applying the binary operator '!=' (line 187)
        result_ne_261091 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 44), '!=', subscript_call_result_261089, unicode_261090)
        
        # Applying the binary operator 'and' (line 187)
        result_and_keyword_261092 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 19), 'and', result_eq_261085, result_ne_261091)
        
        # SSA join for if statement (line 186)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __init__(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Obtaining an instance of the builtin type 'list' (line 191)
        list_261099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 191)
        # Adding element type (line 191)
        
        # Obtaining an instance of the builtin type 'tuple' (line 193)
        tuple_261100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 193)
        # Adding element type (line 193)
        # Getting the type of 'url_prefix' (line 193)
        url_prefix_261101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 17), 'url_prefix', False)
        unicode_261102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 30), 'unicode', u'/_static/(.*)')
        # Applying the binary operator '+' (line 193)
        result_add_261103 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 17), '+', url_prefix_261101, unicode_261102)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 17), tuple_261100, result_add_261103)
        # Adding element type (line 193)
        # Getting the type of 'tornado' (line 194)
        tornado_261104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'tornado', False)
        # Obtaining the member 'web' of a type (line 194)
        web_261105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), tornado_261104, 'web')
        # Obtaining the member 'StaticFileHandler' of a type (line 194)
        StaticFileHandler_261106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), web_261105, 'StaticFileHandler')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 17), tuple_261100, StaticFileHandler_261106)
        # Adding element type (line 193)
        
        # Obtaining an instance of the builtin type 'dict' (line 195)
        dict_261107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 17), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 195)
        # Adding element type (key, value) (line 195)
        unicode_261108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'unicode', u'path')
        
        # Call to get_static_file_path(...): (line 195)
        # Processing the call keyword arguments (line 195)
        kwargs_261112 = {}
        # Getting the type of 'core' (line 195)
        core_261109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 26), 'core', False)
        # Obtaining the member 'FigureManagerWebAgg' of a type (line 195)
        FigureManagerWebAgg_261110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 26), core_261109, 'FigureManagerWebAgg')
        # Obtaining the member 'get_static_file_path' of a type (line 195)
        get_static_file_path_261111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 26), FigureManagerWebAgg_261110, 'get_static_file_path')
        # Calling get_static_file_path(args, kwargs) (line 195)
        get_static_file_path_call_result_261113 = invoke(stypy.reporting.localization.Localization(__file__, 195, 26), get_static_file_path_261111, *[], **kwargs_261112)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 17), dict_261107, (unicode_261108, get_static_file_path_call_result_261113))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 17), tuple_261100, dict_261107)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_261099, tuple_261100)
        # Adding element type (line 191)
        
        # Obtaining an instance of the builtin type 'tuple' (line 198)
        tuple_261114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 198)
        # Adding element type (line 198)
        # Getting the type of 'url_prefix' (line 198)
        url_prefix_261115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'url_prefix', False)
        unicode_261116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 30), 'unicode', u'/favicon.ico')
        # Applying the binary operator '+' (line 198)
        result_add_261117 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 17), '+', url_prefix_261115, unicode_261116)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 17), tuple_261114, result_add_261117)
        # Adding element type (line 198)
        # Getting the type of 'self' (line 198)
        self_261118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 47), 'self', False)
        # Obtaining the member 'FavIcon' of a type (line 198)
        FavIcon_261119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 47), self_261118, 'FavIcon')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 17), tuple_261114, FavIcon_261119)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_261099, tuple_261114)
        # Adding element type (line 191)
        
        # Obtaining an instance of the builtin type 'tuple' (line 201)
        tuple_261120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 201)
        # Adding element type (line 201)
        # Getting the type of 'url_prefix' (line 201)
        url_prefix_261121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 17), 'url_prefix', False)
        unicode_261122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 30), 'unicode', u'/([0-9]+)')
        # Applying the binary operator '+' (line 201)
        result_add_261123 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 17), '+', url_prefix_261121, unicode_261122)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 17), tuple_261120, result_add_261123)
        # Adding element type (line 201)
        # Getting the type of 'self' (line 201)
        self_261124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 44), 'self', False)
        # Obtaining the member 'SingleFigurePage' of a type (line 201)
        SingleFigurePage_261125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 44), self_261124, 'SingleFigurePage')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 17), tuple_261120, SingleFigurePage_261125)
        # Adding element type (line 201)
        
        # Obtaining an instance of the builtin type 'dict' (line 202)
        dict_261126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 17), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 202)
        # Adding element type (key, value) (line 202)
        unicode_261127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 18), 'unicode', u'url_prefix')
        # Getting the type of 'url_prefix' (line 202)
        url_prefix_261128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 32), 'url_prefix', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 17), dict_261126, (unicode_261127, url_prefix_261128))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 17), tuple_261120, dict_261126)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_261099, tuple_261120)
        # Adding element type (line 191)
        
        # Obtaining an instance of the builtin type 'tuple' (line 205)
        tuple_261129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 205)
        # Adding element type (line 205)
        # Getting the type of 'url_prefix' (line 205)
        url_prefix_261130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'url_prefix', False)
        unicode_261131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 30), 'unicode', u'/?')
        # Applying the binary operator '+' (line 205)
        result_add_261132 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 17), '+', url_prefix_261130, unicode_261131)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 17), tuple_261129, result_add_261132)
        # Adding element type (line 205)
        # Getting the type of 'self' (line 205)
        self_261133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 37), 'self', False)
        # Obtaining the member 'AllFiguresPage' of a type (line 205)
        AllFiguresPage_261134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 37), self_261133, 'AllFiguresPage')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 17), tuple_261129, AllFiguresPage_261134)
        # Adding element type (line 205)
        
        # Obtaining an instance of the builtin type 'dict' (line 206)
        dict_261135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 17), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 206)
        # Adding element type (key, value) (line 206)
        unicode_261136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 18), 'unicode', u'url_prefix')
        # Getting the type of 'url_prefix' (line 206)
        url_prefix_261137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 32), 'url_prefix', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 17), dict_261135, (unicode_261136, url_prefix_261137))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 17), tuple_261129, dict_261135)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_261099, tuple_261129)
        # Adding element type (line 191)
        
        # Obtaining an instance of the builtin type 'tuple' (line 208)
        tuple_261138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 208)
        # Adding element type (line 208)
        # Getting the type of 'url_prefix' (line 208)
        url_prefix_261139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 17), 'url_prefix', False)
        unicode_261140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 30), 'unicode', u'/js/mpl.js')
        # Applying the binary operator '+' (line 208)
        result_add_261141 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 17), '+', url_prefix_261139, unicode_261140)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 17), tuple_261138, result_add_261141)
        # Adding element type (line 208)
        # Getting the type of 'self' (line 208)
        self_261142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 45), 'self', False)
        # Obtaining the member 'MplJs' of a type (line 208)
        MplJs_261143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 45), self_261142, 'MplJs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 17), tuple_261138, MplJs_261143)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_261099, tuple_261138)
        # Adding element type (line 191)
        
        # Obtaining an instance of the builtin type 'tuple' (line 212)
        tuple_261144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 212)
        # Adding element type (line 212)
        # Getting the type of 'url_prefix' (line 212)
        url_prefix_261145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 17), 'url_prefix', False)
        unicode_261146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 30), 'unicode', u'/([0-9]+)/ws')
        # Applying the binary operator '+' (line 212)
        result_add_261147 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 17), '+', url_prefix_261145, unicode_261146)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 17), tuple_261144, result_add_261147)
        # Adding element type (line 212)
        # Getting the type of 'self' (line 212)
        self_261148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 47), 'self', False)
        # Obtaining the member 'WebSocket' of a type (line 212)
        WebSocket_261149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 47), self_261148, 'WebSocket')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 17), tuple_261144, WebSocket_261149)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_261099, tuple_261144)
        # Adding element type (line 191)
        
        # Obtaining an instance of the builtin type 'tuple' (line 215)
        tuple_261150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 215)
        # Adding element type (line 215)
        # Getting the type of 'url_prefix' (line 215)
        url_prefix_261151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 17), 'url_prefix', False)
        unicode_261152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 30), 'unicode', u'/([0-9]+)/download.([a-z0-9.]+)')
        # Applying the binary operator '+' (line 215)
        result_add_261153 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 17), '+', url_prefix_261151, unicode_261152)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 17), tuple_261150, result_add_261153)
        # Adding element type (line 215)
        # Getting the type of 'self' (line 216)
        self_261154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 17), 'self', False)
        # Obtaining the member 'Download' of a type (line 216)
        Download_261155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 17), self_261154, 'Download')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 17), tuple_261150, Download_261155)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), list_261099, tuple_261150)
        
        # Processing the call keyword arguments (line 190)
        
        # Call to get_static_file_path(...): (line 218)
        # Processing the call keyword arguments (line 218)
        kwargs_261159 = {}
        # Getting the type of 'core' (line 218)
        core_261156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 26), 'core', False)
        # Obtaining the member 'FigureManagerWebAgg' of a type (line 218)
        FigureManagerWebAgg_261157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 26), core_261156, 'FigureManagerWebAgg')
        # Obtaining the member 'get_static_file_path' of a type (line 218)
        get_static_file_path_261158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 26), FigureManagerWebAgg_261157, 'get_static_file_path')
        # Calling get_static_file_path(args, kwargs) (line 218)
        get_static_file_path_call_result_261160 = invoke(stypy.reporting.localization.Localization(__file__, 218, 26), get_static_file_path_261158, *[], **kwargs_261159)
        
        keyword_261161 = get_static_file_path_call_result_261160
        kwargs_261162 = {'template_path': keyword_261161}
        
        # Call to super(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'WebAggApplication' (line 190)
        WebAggApplication_261094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 14), 'WebAggApplication', False)
        # Getting the type of 'self' (line 190)
        self_261095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 33), 'self', False)
        # Processing the call keyword arguments (line 190)
        kwargs_261096 = {}
        # Getting the type of 'super' (line 190)
        super_261093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'super', False)
        # Calling super(args, kwargs) (line 190)
        super_call_result_261097 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), super_261093, *[WebAggApplication_261094, self_261095], **kwargs_261096)
        
        # Obtaining the member '__init__' of a type (line 190)
        init___261098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), super_call_result_261097, '__init__')
        # Calling __init__(args, kwargs) (line 190)
        init___call_result_261163 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), init___261098, *[list_261099], **kwargs_261162)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def initialize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        unicode_261164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 35), 'unicode', u'')
        # Getting the type of 'None' (line 221)
        None_261165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'None')
        defaults = [unicode_261164, None_261165]
        # Create a new context for function 'initialize'
        module_type_store = module_type_store.open_function_context('initialize', 220, 4, False)
        # Assigning a type to the variable 'self' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        WebAggApplication.initialize.__dict__.__setitem__('stypy_localization', localization)
        WebAggApplication.initialize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        WebAggApplication.initialize.__dict__.__setitem__('stypy_type_store', module_type_store)
        WebAggApplication.initialize.__dict__.__setitem__('stypy_function_name', 'WebAggApplication.initialize')
        WebAggApplication.initialize.__dict__.__setitem__('stypy_param_names_list', ['url_prefix', 'port'])
        WebAggApplication.initialize.__dict__.__setitem__('stypy_varargs_param_name', None)
        WebAggApplication.initialize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        WebAggApplication.initialize.__dict__.__setitem__('stypy_call_defaults', defaults)
        WebAggApplication.initialize.__dict__.__setitem__('stypy_call_varargs', varargs)
        WebAggApplication.initialize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        WebAggApplication.initialize.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'WebAggApplication.initialize', ['url_prefix', 'port'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize', localization, ['url_prefix', 'port'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize(...)' code ##################

        
        # Getting the type of 'cls' (line 222)
        cls_261166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'cls')
        # Obtaining the member 'initialized' of a type (line 222)
        initialized_261167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 11), cls_261166, 'initialized')
        # Testing the type of an if condition (line 222)
        if_condition_261168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 8), initialized_261167)
        # Assigning a type to the variable 'if_condition_261168' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'if_condition_261168', if_condition_261168)
        # SSA begins for if statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 222)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 226):
        
        # Call to cls(...): (line 226)
        # Processing the call keyword arguments (line 226)
        # Getting the type of 'url_prefix' (line 226)
        url_prefix_261170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'url_prefix', False)
        keyword_261171 = url_prefix_261170
        kwargs_261172 = {'url_prefix': keyword_261171}
        # Getting the type of 'cls' (line 226)
        cls_261169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 14), 'cls', False)
        # Calling cls(args, kwargs) (line 226)
        cls_call_result_261173 = invoke(stypy.reporting.localization.Localization(__file__, 226, 14), cls_261169, *[], **kwargs_261172)
        
        # Assigning a type to the variable 'app' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'app', cls_call_result_261173)
        
        # Assigning a Name to a Attribute (line 228):
        # Getting the type of 'url_prefix' (line 228)
        url_prefix_261174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 25), 'url_prefix')
        # Getting the type of 'cls' (line 228)
        cls_261175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'cls')
        # Setting the type of the member 'url_prefix' of a type (line 228)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 8), cls_261175, 'url_prefix', url_prefix_261174)

        @norecursion
        def random_ports(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'random_ports'
            module_type_store = module_type_store.open_function_context('random_ports', 232, 8, False)
            
            # Passed parameters checking function
            random_ports.stypy_localization = localization
            random_ports.stypy_type_of_self = None
            random_ports.stypy_type_store = module_type_store
            random_ports.stypy_function_name = 'random_ports'
            random_ports.stypy_param_names_list = ['port', 'n']
            random_ports.stypy_varargs_param_name = None
            random_ports.stypy_kwargs_param_name = None
            random_ports.stypy_call_defaults = defaults
            random_ports.stypy_call_varargs = varargs
            random_ports.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'random_ports', ['port', 'n'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'random_ports', localization, ['port', 'n'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'random_ports(...)' code ##################

            unicode_261176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, (-1)), 'unicode', u'\n            Generate a list of n random ports near the given port.\n\n            The first 5 ports will be sequential, and the remaining n-5 will be\n            randomly selected in the range [port-2*n, port+2*n].\n            ')
            
            
            # Call to range(...): (line 239)
            # Processing the call arguments (line 239)
            
            # Call to min(...): (line 239)
            # Processing the call arguments (line 239)
            int_261179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 31), 'int')
            # Getting the type of 'n' (line 239)
            n_261180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 34), 'n', False)
            # Processing the call keyword arguments (line 239)
            kwargs_261181 = {}
            # Getting the type of 'min' (line 239)
            min_261178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'min', False)
            # Calling min(args, kwargs) (line 239)
            min_call_result_261182 = invoke(stypy.reporting.localization.Localization(__file__, 239, 27), min_261178, *[int_261179, n_261180], **kwargs_261181)
            
            # Processing the call keyword arguments (line 239)
            kwargs_261183 = {}
            # Getting the type of 'range' (line 239)
            range_261177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 21), 'range', False)
            # Calling range(args, kwargs) (line 239)
            range_call_result_261184 = invoke(stypy.reporting.localization.Localization(__file__, 239, 21), range_261177, *[min_call_result_261182], **kwargs_261183)
            
            # Testing the type of a for loop iterable (line 239)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 239, 12), range_call_result_261184)
            # Getting the type of the for loop variable (line 239)
            for_loop_var_261185 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 239, 12), range_call_result_261184)
            # Assigning a type to the variable 'i' (line 239)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'i', for_loop_var_261185)
            # SSA begins for a for statement (line 239)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'port' (line 240)
            port_261186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 22), 'port')
            # Getting the type of 'i' (line 240)
            i_261187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 29), 'i')
            # Applying the binary operator '+' (line 240)
            result_add_261188 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 22), '+', port_261186, i_261187)
            
            GeneratorType_261189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 16), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 16), GeneratorType_261189, result_add_261188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'stypy_return_type', GeneratorType_261189)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Call to range(...): (line 241)
            # Processing the call arguments (line 241)
            # Getting the type of 'n' (line 241)
            n_261191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 27), 'n', False)
            int_261192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 31), 'int')
            # Applying the binary operator '-' (line 241)
            result_sub_261193 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 27), '-', n_261191, int_261192)
            
            # Processing the call keyword arguments (line 241)
            kwargs_261194 = {}
            # Getting the type of 'range' (line 241)
            range_261190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'range', False)
            # Calling range(args, kwargs) (line 241)
            range_call_result_261195 = invoke(stypy.reporting.localization.Localization(__file__, 241, 21), range_261190, *[result_sub_261193], **kwargs_261194)
            
            # Testing the type of a for loop iterable (line 241)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 241, 12), range_call_result_261195)
            # Getting the type of the for loop variable (line 241)
            for_loop_var_261196 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 241, 12), range_call_result_261195)
            # Assigning a type to the variable 'i' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'i', for_loop_var_261196)
            # SSA begins for a for statement (line 241)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'port' (line 242)
            port_261197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 22), 'port')
            
            # Call to randint(...): (line 242)
            # Processing the call arguments (line 242)
            int_261200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 44), 'int')
            # Getting the type of 'n' (line 242)
            n_261201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 49), 'n', False)
            # Applying the binary operator '*' (line 242)
            result_mul_261202 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 44), '*', int_261200, n_261201)
            
            int_261203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 52), 'int')
            # Getting the type of 'n' (line 242)
            n_261204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 56), 'n', False)
            # Applying the binary operator '*' (line 242)
            result_mul_261205 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 52), '*', int_261203, n_261204)
            
            # Processing the call keyword arguments (line 242)
            kwargs_261206 = {}
            # Getting the type of 'random' (line 242)
            random_261198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 29), 'random', False)
            # Obtaining the member 'randint' of a type (line 242)
            randint_261199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 29), random_261198, 'randint')
            # Calling randint(args, kwargs) (line 242)
            randint_call_result_261207 = invoke(stypy.reporting.localization.Localization(__file__, 242, 29), randint_261199, *[result_mul_261202, result_mul_261205], **kwargs_261206)
            
            # Applying the binary operator '+' (line 242)
            result_add_261208 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 22), '+', port_261197, randint_call_result_261207)
            
            GeneratorType_261209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 16), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 16), GeneratorType_261209, result_add_261208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'stypy_return_type', GeneratorType_261209)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'random_ports(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'random_ports' in the type store
            # Getting the type of 'stypy_return_type' (line 232)
            stypy_return_type_261210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_261210)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'random_ports'
            return stypy_return_type_261210

        # Assigning a type to the variable 'random_ports' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'random_ports', random_ports)
        
        # Assigning a Name to a Name (line 244):
        # Getting the type of 'None' (line 244)
        None_261211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 18), 'None')
        # Assigning a type to the variable 'success' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'success', None_261211)
        
        # Assigning a Subscript to a Attribute (line 245):
        
        # Obtaining the type of the subscript
        unicode_261212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 28), 'unicode', u'webagg.port')
        # Getting the type of 'rcParams' (line 245)
        rcParams_261213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 19), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___261214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 19), rcParams_261213, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_261215 = invoke(stypy.reporting.localization.Localization(__file__, 245, 19), getitem___261214, unicode_261212)
        
        # Getting the type of 'cls' (line 245)
        cls_261216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'cls')
        # Setting the type of the member 'port' of a type (line 245)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), cls_261216, 'port', subscript_call_result_261215)
        
        
        # Call to random_ports(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'cls' (line 246)
        cls_261218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 33), 'cls', False)
        # Obtaining the member 'port' of a type (line 246)
        port_261219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 33), cls_261218, 'port')
        
        # Obtaining the type of the subscript
        unicode_261220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 52), 'unicode', u'webagg.port_retries')
        # Getting the type of 'rcParams' (line 246)
        rcParams_261221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 43), 'rcParams', False)
        # Obtaining the member '__getitem__' of a type (line 246)
        getitem___261222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 43), rcParams_261221, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 246)
        subscript_call_result_261223 = invoke(stypy.reporting.localization.Localization(__file__, 246, 43), getitem___261222, unicode_261220)
        
        # Processing the call keyword arguments (line 246)
        kwargs_261224 = {}
        # Getting the type of 'random_ports' (line 246)
        random_ports_261217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'random_ports', False)
        # Calling random_ports(args, kwargs) (line 246)
        random_ports_call_result_261225 = invoke(stypy.reporting.localization.Localization(__file__, 246, 20), random_ports_261217, *[port_261219, subscript_call_result_261223], **kwargs_261224)
        
        # Testing the type of a for loop iterable (line 246)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 246, 8), random_ports_call_result_261225)
        # Getting the type of the for loop variable (line 246)
        for_loop_var_261226 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 246, 8), random_ports_call_result_261225)
        # Assigning a type to the variable 'port' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'port', for_loop_var_261226)
        # SSA begins for a for statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 247)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to listen(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'port' (line 248)
        port_261229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 27), 'port', False)
        # Processing the call keyword arguments (line 248)
        kwargs_261230 = {}
        # Getting the type of 'app' (line 248)
        app_261227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'app', False)
        # Obtaining the member 'listen' of a type (line 248)
        listen_261228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 16), app_261227, 'listen')
        # Calling listen(args, kwargs) (line 248)
        listen_call_result_261231 = invoke(stypy.reporting.localization.Localization(__file__, 248, 16), listen_261228, *[port_261229], **kwargs_261230)
        
        # SSA branch for the except part of a try statement (line 247)
        # SSA branch for the except 'Attribute' branch of a try statement (line 247)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'socket' (line 249)
        socket_261232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'socket')
        # Obtaining the member 'error' of a type (line 249)
        error_261233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 19), socket_261232, 'error')
        # Assigning a type to the variable 'e' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'e', error_261233)
        
        
        # Getting the type of 'e' (line 250)
        e_261234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 19), 'e')
        # Obtaining the member 'errno' of a type (line 250)
        errno_261235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 19), e_261234, 'errno')
        # Getting the type of 'errno' (line 250)
        errno_261236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 30), 'errno')
        # Obtaining the member 'EADDRINUSE' of a type (line 250)
        EADDRINUSE_261237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 30), errno_261236, 'EADDRINUSE')
        # Applying the binary operator '!=' (line 250)
        result_ne_261238 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 19), '!=', errno_261235, EADDRINUSE_261237)
        
        # Testing the type of an if condition (line 250)
        if_condition_261239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 16), result_ne_261238)
        # Assigning a type to the variable 'if_condition_261239' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'if_condition_261239', if_condition_261239)
        # SSA begins for if statement (line 250)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 250)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else branch of a try statement (line 247)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a Name to a Attribute (line 253):
        # Getting the type of 'port' (line 253)
        port_261240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 27), 'port')
        # Getting the type of 'cls' (line 253)
        cls_261241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'cls')
        # Setting the type of the member 'port' of a type (line 253)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 16), cls_261241, 'port', port_261240)
        
        # Assigning a Name to a Name (line 254):
        # Getting the type of 'True' (line 254)
        True_261242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'True')
        # Assigning a type to the variable 'success' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'success', True_261242)
        # SSA join for try-except statement (line 247)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'success' (line 257)
        success_261243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'success')
        # Applying the 'not' unary operator (line 257)
        result_not__261244 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 11), 'not', success_261243)
        
        # Testing the type of an if condition (line 257)
        if_condition_261245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 8), result_not__261244)
        # Assigning a type to the variable 'if_condition_261245' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'if_condition_261245', if_condition_261245)
        # SSA begins for if statement (line 257)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to SystemExit(...): (line 258)
        # Processing the call arguments (line 258)
        unicode_261247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 16), 'unicode', u'The webagg server could not be started because an available port could not be found')
        # Processing the call keyword arguments (line 258)
        kwargs_261248 = {}
        # Getting the type of 'SystemExit' (line 258)
        SystemExit_261246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 18), 'SystemExit', False)
        # Calling SystemExit(args, kwargs) (line 258)
        SystemExit_call_result_261249 = invoke(stypy.reporting.localization.Localization(__file__, 258, 18), SystemExit_261246, *[unicode_261247], **kwargs_261248)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 258, 12), SystemExit_call_result_261249, 'raise parameter', BaseException)
        # SSA join for if statement (line 257)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 262):
        # Getting the type of 'True' (line 262)
        True_261250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 26), 'True')
        # Getting the type of 'cls' (line 262)
        cls_261251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'cls')
        # Setting the type of the member 'initialized' of a type (line 262)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), cls_261251, 'initialized', True_261250)
        
        # ################# End of 'initialize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize' in the type store
        # Getting the type of 'stypy_return_type' (line 220)
        stypy_return_type_261252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_261252)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize'
        return stypy_return_type_261252


    @norecursion
    def start(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'start'
        module_type_store = module_type_store.open_function_context('start', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        WebAggApplication.start.__dict__.__setitem__('stypy_localization', localization)
        WebAggApplication.start.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        WebAggApplication.start.__dict__.__setitem__('stypy_type_store', module_type_store)
        WebAggApplication.start.__dict__.__setitem__('stypy_function_name', 'WebAggApplication.start')
        WebAggApplication.start.__dict__.__setitem__('stypy_param_names_list', [])
        WebAggApplication.start.__dict__.__setitem__('stypy_varargs_param_name', None)
        WebAggApplication.start.__dict__.__setitem__('stypy_kwargs_param_name', None)
        WebAggApplication.start.__dict__.__setitem__('stypy_call_defaults', defaults)
        WebAggApplication.start.__dict__.__setitem__('stypy_call_varargs', varargs)
        WebAggApplication.start.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        WebAggApplication.start.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'WebAggApplication.start', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'start', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'start(...)' code ##################

        
        # Getting the type of 'cls' (line 266)
        cls_261253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 11), 'cls')
        # Obtaining the member 'started' of a type (line 266)
        started_261254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 11), cls_261253, 'started')
        # Testing the type of an if condition (line 266)
        if_condition_261255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 8), started_261254)
        # Assigning a type to the variable 'if_condition_261255' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'if_condition_261255', if_condition_261255)
        # SSA begins for if statement (line 266)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 266)
        module_type_store = module_type_store.join_ssa_context()
        
        unicode_261256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, (-1)), 'unicode', u'\n        IOLoop.running() was removed as of Tornado 2.4; see for example\n        https://groups.google.com/forum/#!topic/python-tornado/QLMzkpQBGOY\n        Thus there is no correct way to check if the loop has already been\n        launched. We may end up with two concurrently running loops in that\n        unlucky case with all the expected consequences.\n        ')
        
        # Assigning a Call to a Name (line 276):
        
        # Call to instance(...): (line 276)
        # Processing the call keyword arguments (line 276)
        kwargs_261261 = {}
        # Getting the type of 'tornado' (line 276)
        tornado_261257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 17), 'tornado', False)
        # Obtaining the member 'ioloop' of a type (line 276)
        ioloop_261258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 17), tornado_261257, 'ioloop')
        # Obtaining the member 'IOLoop' of a type (line 276)
        IOLoop_261259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 17), ioloop_261258, 'IOLoop')
        # Obtaining the member 'instance' of a type (line 276)
        instance_261260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 17), IOLoop_261259, 'instance')
        # Calling instance(args, kwargs) (line 276)
        instance_call_result_261262 = invoke(stypy.reporting.localization.Localization(__file__, 276, 17), instance_261260, *[], **kwargs_261261)
        
        # Assigning a type to the variable 'ioloop' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'ioloop', instance_call_result_261262)

        @norecursion
        def shutdown(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'shutdown'
            module_type_store = module_type_store.open_function_context('shutdown', 278, 8, False)
            
            # Passed parameters checking function
            shutdown.stypy_localization = localization
            shutdown.stypy_type_of_self = None
            shutdown.stypy_type_store = module_type_store
            shutdown.stypy_function_name = 'shutdown'
            shutdown.stypy_param_names_list = []
            shutdown.stypy_varargs_param_name = None
            shutdown.stypy_kwargs_param_name = None
            shutdown.stypy_call_defaults = defaults
            shutdown.stypy_call_varargs = varargs
            shutdown.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'shutdown', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'shutdown', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'shutdown(...)' code ##################

            
            # Call to stop(...): (line 279)
            # Processing the call keyword arguments (line 279)
            kwargs_261265 = {}
            # Getting the type of 'ioloop' (line 279)
            ioloop_261263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'ioloop', False)
            # Obtaining the member 'stop' of a type (line 279)
            stop_261264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), ioloop_261263, 'stop')
            # Calling stop(args, kwargs) (line 279)
            stop_call_result_261266 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), stop_261264, *[], **kwargs_261265)
            
            
            # Call to print(...): (line 280)
            # Processing the call arguments (line 280)
            unicode_261268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 18), 'unicode', u'Server is stopped')
            # Processing the call keyword arguments (line 280)
            kwargs_261269 = {}
            # Getting the type of 'print' (line 280)
            print_261267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'print', False)
            # Calling print(args, kwargs) (line 280)
            print_call_result_261270 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), print_261267, *[unicode_261268], **kwargs_261269)
            
            
            # Call to flush(...): (line 281)
            # Processing the call keyword arguments (line 281)
            kwargs_261274 = {}
            # Getting the type of 'sys' (line 281)
            sys_261271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'sys', False)
            # Obtaining the member 'stdout' of a type (line 281)
            stdout_261272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), sys_261271, 'stdout')
            # Obtaining the member 'flush' of a type (line 281)
            flush_261273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), stdout_261272, 'flush')
            # Calling flush(args, kwargs) (line 281)
            flush_call_result_261275 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), flush_261273, *[], **kwargs_261274)
            
            
            # Assigning a Name to a Attribute (line 282):
            # Getting the type of 'False' (line 282)
            False_261276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 26), 'False')
            # Getting the type of 'cls' (line 282)
            cls_261277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'cls')
            # Setting the type of the member 'started' of a type (line 282)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), cls_261277, 'started', False_261276)
            
            # ################# End of 'shutdown(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'shutdown' in the type store
            # Getting the type of 'stypy_return_type' (line 278)
            stypy_return_type_261278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_261278)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'shutdown'
            return stypy_return_type_261278

        # Assigning a type to the variable 'shutdown' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'shutdown', shutdown)

        @norecursion
        def catch_sigint(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'catch_sigint'
            module_type_store = module_type_store.open_function_context('catch_sigint', 284, 8, False)
            
            # Passed parameters checking function
            catch_sigint.stypy_localization = localization
            catch_sigint.stypy_type_of_self = None
            catch_sigint.stypy_type_store = module_type_store
            catch_sigint.stypy_function_name = 'catch_sigint'
            catch_sigint.stypy_param_names_list = []
            catch_sigint.stypy_varargs_param_name = None
            catch_sigint.stypy_kwargs_param_name = None
            catch_sigint.stypy_call_defaults = defaults
            catch_sigint.stypy_call_varargs = varargs
            catch_sigint.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'catch_sigint', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'catch_sigint', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'catch_sigint(...)' code ##################

            
            # Assigning a Call to a Name (line 286):
            
            # Call to signal(...): (line 286)
            # Processing the call arguments (line 286)
            # Getting the type of 'signal' (line 287)
            signal_261281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'signal', False)
            # Obtaining the member 'SIGINT' of a type (line 287)
            SIGINT_261282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 16), signal_261281, 'SIGINT')

            @norecursion
            def _stypy_temp_lambda_112(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_112'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_112', 288, 16, True)
                # Passed parameters checking function
                _stypy_temp_lambda_112.stypy_localization = localization
                _stypy_temp_lambda_112.stypy_type_of_self = None
                _stypy_temp_lambda_112.stypy_type_store = module_type_store
                _stypy_temp_lambda_112.stypy_function_name = '_stypy_temp_lambda_112'
                _stypy_temp_lambda_112.stypy_param_names_list = ['sig', 'frame']
                _stypy_temp_lambda_112.stypy_varargs_param_name = None
                _stypy_temp_lambda_112.stypy_kwargs_param_name = None
                _stypy_temp_lambda_112.stypy_call_defaults = defaults
                _stypy_temp_lambda_112.stypy_call_varargs = varargs
                _stypy_temp_lambda_112.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_112', ['sig', 'frame'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_112', ['sig', 'frame'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to add_callback_from_signal(...): (line 288)
                # Processing the call arguments (line 288)
                # Getting the type of 'shutdown' (line 288)
                shutdown_261285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 67), 'shutdown', False)
                # Processing the call keyword arguments (line 288)
                kwargs_261286 = {}
                # Getting the type of 'ioloop' (line 288)
                ioloop_261283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 35), 'ioloop', False)
                # Obtaining the member 'add_callback_from_signal' of a type (line 288)
                add_callback_from_signal_261284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 35), ioloop_261283, 'add_callback_from_signal')
                # Calling add_callback_from_signal(args, kwargs) (line 288)
                add_callback_from_signal_call_result_261287 = invoke(stypy.reporting.localization.Localization(__file__, 288, 35), add_callback_from_signal_261284, *[shutdown_261285], **kwargs_261286)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 288)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'stypy_return_type', add_callback_from_signal_call_result_261287)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_112' in the type store
                # Getting the type of 'stypy_return_type' (line 288)
                stypy_return_type_261288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_261288)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_112'
                return stypy_return_type_261288

            # Assigning a type to the variable '_stypy_temp_lambda_112' (line 288)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), '_stypy_temp_lambda_112', _stypy_temp_lambda_112)
            # Getting the type of '_stypy_temp_lambda_112' (line 288)
            _stypy_temp_lambda_112_261289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), '_stypy_temp_lambda_112')
            # Processing the call keyword arguments (line 286)
            kwargs_261290 = {}
            # Getting the type of 'signal' (line 286)
            signal_261279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'signal', False)
            # Obtaining the member 'signal' of a type (line 286)
            signal_261280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 26), signal_261279, 'signal')
            # Calling signal(args, kwargs) (line 286)
            signal_call_result_261291 = invoke(stypy.reporting.localization.Localization(__file__, 286, 26), signal_261280, *[SIGINT_261282, _stypy_temp_lambda_112_261289], **kwargs_261290)
            
            # Assigning a type to the variable 'old_handler' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'old_handler', signal_call_result_261291)
            
            # Try-finally block (line 289)
            # Creating a generator
            GeneratorType_261292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 16), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 16), GeneratorType_261292, None)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'stypy_return_type', GeneratorType_261292)
            
            # finally branch of the try-finally block (line 289)
            
            # Call to signal(...): (line 292)
            # Processing the call arguments (line 292)
            # Getting the type of 'signal' (line 292)
            signal_261295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'signal', False)
            # Obtaining the member 'SIGINT' of a type (line 292)
            SIGINT_261296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 30), signal_261295, 'SIGINT')
            # Getting the type of 'old_handler' (line 292)
            old_handler_261297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 45), 'old_handler', False)
            # Processing the call keyword arguments (line 292)
            kwargs_261298 = {}
            # Getting the type of 'signal' (line 292)
            signal_261293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'signal', False)
            # Obtaining the member 'signal' of a type (line 292)
            signal_261294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 16), signal_261293, 'signal')
            # Calling signal(args, kwargs) (line 292)
            signal_call_result_261299 = invoke(stypy.reporting.localization.Localization(__file__, 292, 16), signal_261294, *[SIGINT_261296, old_handler_261297], **kwargs_261298)
            
            
            
            # ################# End of 'catch_sigint(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'catch_sigint' in the type store
            # Getting the type of 'stypy_return_type' (line 284)
            stypy_return_type_261300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_261300)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'catch_sigint'
            return stypy_return_type_261300

        # Assigning a type to the variable 'catch_sigint' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'catch_sigint', catch_sigint)
        
        # Assigning a Name to a Attribute (line 295):
        # Getting the type of 'True' (line 295)
        True_261301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 22), 'True')
        # Getting the type of 'cls' (line 295)
        cls_261302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'cls')
        # Setting the type of the member 'started' of a type (line 295)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), cls_261302, 'started', True_261301)
        
        # Call to print(...): (line 297)
        # Processing the call arguments (line 297)
        unicode_261304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 14), 'unicode', u'Press Ctrl+C to stop WebAgg server')
        # Processing the call keyword arguments (line 297)
        kwargs_261305 = {}
        # Getting the type of 'print' (line 297)
        print_261303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'print', False)
        # Calling print(args, kwargs) (line 297)
        print_call_result_261306 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), print_261303, *[unicode_261304], **kwargs_261305)
        
        
        # Call to flush(...): (line 298)
        # Processing the call keyword arguments (line 298)
        kwargs_261310 = {}
        # Getting the type of 'sys' (line 298)
        sys_261307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'sys', False)
        # Obtaining the member 'stdout' of a type (line 298)
        stdout_261308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), sys_261307, 'stdout')
        # Obtaining the member 'flush' of a type (line 298)
        flush_261309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), stdout_261308, 'flush')
        # Calling flush(args, kwargs) (line 298)
        flush_call_result_261311 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), flush_261309, *[], **kwargs_261310)
        
        
        # Call to catch_sigint(...): (line 299)
        # Processing the call keyword arguments (line 299)
        kwargs_261313 = {}
        # Getting the type of 'catch_sigint' (line 299)
        catch_sigint_261312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 13), 'catch_sigint', False)
        # Calling catch_sigint(args, kwargs) (line 299)
        catch_sigint_call_result_261314 = invoke(stypy.reporting.localization.Localization(__file__, 299, 13), catch_sigint_261312, *[], **kwargs_261313)
        
        with_261315 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 299, 13), catch_sigint_call_result_261314, 'with parameter', '__enter__', '__exit__')

        if with_261315:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 299)
            enter___261316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 13), catch_sigint_call_result_261314, '__enter__')
            with_enter_261317 = invoke(stypy.reporting.localization.Localization(__file__, 299, 13), enter___261316)
            
            # Call to start(...): (line 300)
            # Processing the call keyword arguments (line 300)
            kwargs_261320 = {}
            # Getting the type of 'ioloop' (line 300)
            ioloop_261318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'ioloop', False)
            # Obtaining the member 'start' of a type (line 300)
            start_261319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), ioloop_261318, 'start')
            # Calling start(args, kwargs) (line 300)
            start_call_result_261321 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), start_261319, *[], **kwargs_261320)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 299)
            exit___261322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 13), catch_sigint_call_result_261314, '__exit__')
            with_exit_261323 = invoke(stypy.reporting.localization.Localization(__file__, 299, 13), exit___261322, None, None, None)

        
        # ################# End of 'start(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'start' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_261324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_261324)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'start'
        return stypy_return_type_261324


# Assigning a type to the variable 'WebAggApplication' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'WebAggApplication', WebAggApplication)

# Assigning a Name to a Name (line 64):
# Getting the type of 'False' (line 64)
False_261325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'False')
# Getting the type of 'WebAggApplication'
WebAggApplication_261326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'WebAggApplication')
# Setting the type of the member 'initialized' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), WebAggApplication_261326, 'initialized', False_261325)

# Assigning a Name to a Name (line 65):
# Getting the type of 'False' (line 65)
False_261327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 14), 'False')
# Getting the type of 'WebAggApplication'
WebAggApplication_261328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'WebAggApplication')
# Setting the type of the member 'started' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), WebAggApplication_261328, 'started', False_261327)

@norecursion
def ipython_inline_display(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ipython_inline_display'
    module_type_store = module_type_store.open_function_context('ipython_inline_display', 303, 0, False)
    
    # Passed parameters checking function
    ipython_inline_display.stypy_localization = localization
    ipython_inline_display.stypy_type_of_self = None
    ipython_inline_display.stypy_type_store = module_type_store
    ipython_inline_display.stypy_function_name = 'ipython_inline_display'
    ipython_inline_display.stypy_param_names_list = ['figure']
    ipython_inline_display.stypy_varargs_param_name = None
    ipython_inline_display.stypy_kwargs_param_name = None
    ipython_inline_display.stypy_call_defaults = defaults
    ipython_inline_display.stypy_call_varargs = varargs
    ipython_inline_display.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ipython_inline_display', ['figure'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ipython_inline_display', localization, ['figure'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ipython_inline_display(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 304, 4))
    
    # 'import tornado.template' statement (line 304)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
    import_261329 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 304, 4), 'tornado.template')

    if (type(import_261329) is not StypyTypeError):

        if (import_261329 != 'pyd_module'):
            __import__(import_261329)
            sys_modules_261330 = sys.modules[import_261329]
            import_module(stypy.reporting.localization.Localization(__file__, 304, 4), 'tornado.template', sys_modules_261330.module_type_store, module_type_store)
        else:
            import tornado.template

            import_module(stypy.reporting.localization.Localization(__file__, 304, 4), 'tornado.template', tornado.template, module_type_store)

    else:
        # Assigning a type to the variable 'tornado.template' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'tornado.template', import_261329)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')
    
    
    # Call to initialize(...): (line 306)
    # Processing the call keyword arguments (line 306)
    kwargs_261333 = {}
    # Getting the type of 'WebAggApplication' (line 306)
    WebAggApplication_261331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'WebAggApplication', False)
    # Obtaining the member 'initialize' of a type (line 306)
    initialize_261332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 4), WebAggApplication_261331, 'initialize')
    # Calling initialize(args, kwargs) (line 306)
    initialize_call_result_261334 = invoke(stypy.reporting.localization.Localization(__file__, 306, 4), initialize_261332, *[], **kwargs_261333)
    
    
    
    
    # Call to is_alive(...): (line 307)
    # Processing the call keyword arguments (line 307)
    kwargs_261337 = {}
    # Getting the type of 'webagg_server_thread' (line 307)
    webagg_server_thread_261335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 11), 'webagg_server_thread', False)
    # Obtaining the member 'is_alive' of a type (line 307)
    is_alive_261336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 11), webagg_server_thread_261335, 'is_alive')
    # Calling is_alive(args, kwargs) (line 307)
    is_alive_call_result_261338 = invoke(stypy.reporting.localization.Localization(__file__, 307, 11), is_alive_261336, *[], **kwargs_261337)
    
    # Applying the 'not' unary operator (line 307)
    result_not__261339 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 7), 'not', is_alive_call_result_261338)
    
    # Testing the type of an if condition (line 307)
    if_condition_261340 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 4), result_not__261339)
    # Assigning a type to the variable 'if_condition_261340' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'if_condition_261340', if_condition_261340)
    # SSA begins for if statement (line 307)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to start(...): (line 308)
    # Processing the call keyword arguments (line 308)
    kwargs_261343 = {}
    # Getting the type of 'webagg_server_thread' (line 308)
    webagg_server_thread_261341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'webagg_server_thread', False)
    # Obtaining the member 'start' of a type (line 308)
    start_261342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), webagg_server_thread_261341, 'start')
    # Calling start(args, kwargs) (line 308)
    start_call_result_261344 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), start_261342, *[], **kwargs_261343)
    
    # SSA join for if statement (line 307)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to open(...): (line 310)
    # Processing the call arguments (line 310)
    
    # Call to join(...): (line 310)
    # Processing the call arguments (line 310)
    
    # Call to get_static_file_path(...): (line 311)
    # Processing the call keyword arguments (line 311)
    kwargs_261352 = {}
    # Getting the type of 'core' (line 311)
    core_261349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'core', False)
    # Obtaining the member 'FigureManagerWebAgg' of a type (line 311)
    FigureManagerWebAgg_261350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), core_261349, 'FigureManagerWebAgg')
    # Obtaining the member 'get_static_file_path' of a type (line 311)
    get_static_file_path_261351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), FigureManagerWebAgg_261350, 'get_static_file_path')
    # Calling get_static_file_path(args, kwargs) (line 311)
    get_static_file_path_call_result_261353 = invoke(stypy.reporting.localization.Localization(__file__, 311, 12), get_static_file_path_261351, *[], **kwargs_261352)
    
    unicode_261354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 12), 'unicode', u'ipython_inline_figure.html')
    # Processing the call keyword arguments (line 310)
    kwargs_261355 = {}
    # Getting the type of 'os' (line 310)
    os_261346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 310)
    path_261347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 14), os_261346, 'path')
    # Obtaining the member 'join' of a type (line 310)
    join_261348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 14), path_261347, 'join')
    # Calling join(args, kwargs) (line 310)
    join_call_result_261356 = invoke(stypy.reporting.localization.Localization(__file__, 310, 14), join_261348, *[get_static_file_path_call_result_261353, unicode_261354], **kwargs_261355)
    
    # Processing the call keyword arguments (line 310)
    kwargs_261357 = {}
    # Getting the type of 'open' (line 310)
    open_261345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 9), 'open', False)
    # Calling open(args, kwargs) (line 310)
    open_call_result_261358 = invoke(stypy.reporting.localization.Localization(__file__, 310, 9), open_261345, *[join_call_result_261356], **kwargs_261357)
    
    with_261359 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 310, 9), open_call_result_261358, 'with parameter', '__enter__', '__exit__')

    if with_261359:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 310)
        enter___261360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 9), open_call_result_261358, '__enter__')
        with_enter_261361 = invoke(stypy.reporting.localization.Localization(__file__, 310, 9), enter___261360)
        # Assigning a type to the variable 'fd' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 9), 'fd', with_enter_261361)
        
        # Assigning a Call to a Name (line 313):
        
        # Call to read(...): (line 313)
        # Processing the call keyword arguments (line 313)
        kwargs_261364 = {}
        # Getting the type of 'fd' (line 313)
        fd_261362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 14), 'fd', False)
        # Obtaining the member 'read' of a type (line 313)
        read_261363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 14), fd_261362, 'read')
        # Calling read(args, kwargs) (line 313)
        read_call_result_261365 = invoke(stypy.reporting.localization.Localization(__file__, 313, 14), read_261363, *[], **kwargs_261364)
        
        # Assigning a type to the variable 'tpl' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'tpl', read_call_result_261365)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 310)
        exit___261366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 9), open_call_result_261358, '__exit__')
        with_exit_261367 = invoke(stypy.reporting.localization.Localization(__file__, 310, 9), exit___261366, None, None, None)

    
    # Assigning a Attribute to a Name (line 315):
    # Getting the type of 'figure' (line 315)
    figure_261368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 13), 'figure')
    # Obtaining the member 'number' of a type (line 315)
    number_261369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 13), figure_261368, 'number')
    # Assigning a type to the variable 'fignum' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'fignum', number_261369)
    
    # Assigning a Call to a Name (line 317):
    
    # Call to Template(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'tpl' (line 317)
    tpl_261373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 34), 'tpl', False)
    # Processing the call keyword arguments (line 317)
    kwargs_261374 = {}
    # Getting the type of 'tornado' (line 317)
    tornado_261370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'tornado', False)
    # Obtaining the member 'template' of a type (line 317)
    template_261371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), tornado_261370, 'template')
    # Obtaining the member 'Template' of a type (line 317)
    Template_261372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), template_261371, 'Template')
    # Calling Template(args, kwargs) (line 317)
    Template_call_result_261375 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), Template_261372, *[tpl_261373], **kwargs_261374)
    
    # Assigning a type to the variable 't' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 't', Template_call_result_261375)
    
    # Call to decode(...): (line 318)
    # Processing the call arguments (line 318)
    unicode_261396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 44), 'unicode', u'utf-8')
    # Processing the call keyword arguments (line 318)
    kwargs_261397 = {}
    
    # Call to generate(...): (line 318)
    # Processing the call keyword arguments (line 318)
    # Getting the type of 'WebAggApplication' (line 319)
    WebAggApplication_261378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 15), 'WebAggApplication', False)
    # Obtaining the member 'url_prefix' of a type (line 319)
    url_prefix_261379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 15), WebAggApplication_261378, 'url_prefix')
    keyword_261380 = url_prefix_261379
    # Getting the type of 'fignum' (line 320)
    fignum_261381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'fignum', False)
    keyword_261382 = fignum_261381
    # Getting the type of 'core' (line 321)
    core_261383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 18), 'core', False)
    # Obtaining the member 'NavigationToolbar2WebAgg' of a type (line 321)
    NavigationToolbar2WebAgg_261384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 18), core_261383, 'NavigationToolbar2WebAgg')
    # Obtaining the member 'toolitems' of a type (line 321)
    toolitems_261385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 18), NavigationToolbar2WebAgg_261384, 'toolitems')
    keyword_261386 = toolitems_261385
    # Getting the type of 'figure' (line 322)
    figure_261387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'figure', False)
    # Obtaining the member 'canvas' of a type (line 322)
    canvas_261388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 15), figure_261387, 'canvas')
    keyword_261389 = canvas_261388
    # Getting the type of 'WebAggApplication' (line 323)
    WebAggApplication_261390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'WebAggApplication', False)
    # Obtaining the member 'port' of a type (line 323)
    port_261391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 13), WebAggApplication_261390, 'port')
    keyword_261392 = port_261391
    kwargs_261393 = {'fig_id': keyword_261382, 'prefix': keyword_261380, 'toolitems': keyword_261386, 'canvas': keyword_261389, 'port': keyword_261392}
    # Getting the type of 't' (line 318)
    t_261376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 't', False)
    # Obtaining the member 'generate' of a type (line 318)
    generate_261377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 11), t_261376, 'generate')
    # Calling generate(args, kwargs) (line 318)
    generate_call_result_261394 = invoke(stypy.reporting.localization.Localization(__file__, 318, 11), generate_261377, *[], **kwargs_261393)
    
    # Obtaining the member 'decode' of a type (line 318)
    decode_261395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 11), generate_call_result_261394, 'decode')
    # Calling decode(args, kwargs) (line 318)
    decode_call_result_261398 = invoke(stypy.reporting.localization.Localization(__file__, 318, 11), decode_261395, *[unicode_261396], **kwargs_261397)
    
    # Assigning a type to the variable 'stypy_return_type' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type', decode_call_result_261398)
    
    # ################# End of 'ipython_inline_display(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ipython_inline_display' in the type store
    # Getting the type of 'stypy_return_type' (line 303)
    stypy_return_type_261399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_261399)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ipython_inline_display'
    return stypy_return_type_261399

# Assigning a type to the variable 'ipython_inline_display' (line 303)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 0), 'ipython_inline_display', ipython_inline_display)
# Declaration of the '_BackendWebAgg' class
# Getting the type of '_Backend' (line 327)
_Backend_261400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 21), '_Backend')

class _BackendWebAgg(_Backend_261400, ):

    @staticmethod
    @norecursion
    def trigger_manager_draw(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'trigger_manager_draw'
        module_type_store = module_type_store.open_function_context('trigger_manager_draw', 331, 4, False)
        
        # Passed parameters checking function
        _BackendWebAgg.trigger_manager_draw.__dict__.__setitem__('stypy_localization', localization)
        _BackendWebAgg.trigger_manager_draw.__dict__.__setitem__('stypy_type_of_self', None)
        _BackendWebAgg.trigger_manager_draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        _BackendWebAgg.trigger_manager_draw.__dict__.__setitem__('stypy_function_name', 'trigger_manager_draw')
        _BackendWebAgg.trigger_manager_draw.__dict__.__setitem__('stypy_param_names_list', ['manager'])
        _BackendWebAgg.trigger_manager_draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        _BackendWebAgg.trigger_manager_draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _BackendWebAgg.trigger_manager_draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        _BackendWebAgg.trigger_manager_draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        _BackendWebAgg.trigger_manager_draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _BackendWebAgg.trigger_manager_draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
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

        
        # Call to draw_idle(...): (line 333)
        # Processing the call keyword arguments (line 333)
        kwargs_261404 = {}
        # Getting the type of 'manager' (line 333)
        manager_261401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'manager', False)
        # Obtaining the member 'canvas' of a type (line 333)
        canvas_261402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), manager_261401, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 333)
        draw_idle_261403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), canvas_261402, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 333)
        draw_idle_call_result_261405 = invoke(stypy.reporting.localization.Localization(__file__, 333, 8), draw_idle_261403, *[], **kwargs_261404)
        
        
        # ################# End of 'trigger_manager_draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger_manager_draw' in the type store
        # Getting the type of 'stypy_return_type' (line 331)
        stypy_return_type_261406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_261406)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger_manager_draw'
        return stypy_return_type_261406


    @staticmethod
    @norecursion
    def show(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'show'
        module_type_store = module_type_store.open_function_context('show', 335, 4, False)
        
        # Passed parameters checking function
        _BackendWebAgg.show.__dict__.__setitem__('stypy_localization', localization)
        _BackendWebAgg.show.__dict__.__setitem__('stypy_type_of_self', None)
        _BackendWebAgg.show.__dict__.__setitem__('stypy_type_store', module_type_store)
        _BackendWebAgg.show.__dict__.__setitem__('stypy_function_name', 'show')
        _BackendWebAgg.show.__dict__.__setitem__('stypy_param_names_list', [])
        _BackendWebAgg.show.__dict__.__setitem__('stypy_varargs_param_name', None)
        _BackendWebAgg.show.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _BackendWebAgg.show.__dict__.__setitem__('stypy_call_defaults', defaults)
        _BackendWebAgg.show.__dict__.__setitem__('stypy_call_varargs', varargs)
        _BackendWebAgg.show.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _BackendWebAgg.show.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'show', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to initialize(...): (line 337)
        # Processing the call keyword arguments (line 337)
        kwargs_261409 = {}
        # Getting the type of 'WebAggApplication' (line 337)
        WebAggApplication_261407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'WebAggApplication', False)
        # Obtaining the member 'initialize' of a type (line 337)
        initialize_261408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), WebAggApplication_261407, 'initialize')
        # Calling initialize(args, kwargs) (line 337)
        initialize_call_result_261410 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), initialize_261408, *[], **kwargs_261409)
        
        
        # Assigning a Call to a Name (line 339):
        
        # Call to format(...): (line 339)
        # Processing the call keyword arguments (line 339)
        # Getting the type of 'WebAggApplication' (line 340)
        WebAggApplication_261413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 17), 'WebAggApplication', False)
        # Obtaining the member 'port' of a type (line 340)
        port_261414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 17), WebAggApplication_261413, 'port')
        keyword_261415 = port_261414
        # Getting the type of 'WebAggApplication' (line 341)
        WebAggApplication_261416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 19), 'WebAggApplication', False)
        # Obtaining the member 'url_prefix' of a type (line 341)
        url_prefix_261417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 19), WebAggApplication_261416, 'url_prefix')
        keyword_261418 = url_prefix_261417
        kwargs_261419 = {'prefix': keyword_261418, 'port': keyword_261415}
        unicode_261411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 14), 'unicode', u'http://127.0.0.1:{port}{prefix}')
        # Obtaining the member 'format' of a type (line 339)
        format_261412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 14), unicode_261411, 'format')
        # Calling format(args, kwargs) (line 339)
        format_call_result_261420 = invoke(stypy.reporting.localization.Localization(__file__, 339, 14), format_261412, *[], **kwargs_261419)
        
        # Assigning a type to the variable 'url' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'url', format_call_result_261420)
        
        
        # Obtaining the type of the subscript
        unicode_261421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 20), 'unicode', u'webagg.open_in_browser')
        # Getting the type of 'rcParams' (line 343)
        rcParams_261422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___261423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 11), rcParams_261422, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_261424 = invoke(stypy.reporting.localization.Localization(__file__, 343, 11), getitem___261423, unicode_261421)
        
        # Testing the type of an if condition (line 343)
        if_condition_261425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 8), subscript_call_result_261424)
        # Assigning a type to the variable 'if_condition_261425' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'if_condition_261425', if_condition_261425)
        # SSA begins for if statement (line 343)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 344, 12))
        
        # 'import webbrowser' statement (line 344)
        import webbrowser

        import_module(stypy.reporting.localization.Localization(__file__, 344, 12), 'webbrowser', webbrowser, module_type_store)
        
        
        # Call to open(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'url' (line 345)
        url_261428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 28), 'url', False)
        # Processing the call keyword arguments (line 345)
        kwargs_261429 = {}
        # Getting the type of 'webbrowser' (line 345)
        webbrowser_261426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'webbrowser', False)
        # Obtaining the member 'open' of a type (line 345)
        open_261427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), webbrowser_261426, 'open')
        # Calling open(args, kwargs) (line 345)
        open_call_result_261430 = invoke(stypy.reporting.localization.Localization(__file__, 345, 12), open_261427, *[url_261428], **kwargs_261429)
        
        # SSA branch for the else part of an if statement (line 343)
        module_type_store.open_ssa_branch('else')
        
        # Call to print(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Call to format(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'url' (line 347)
        url_261434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 53), 'url', False)
        # Processing the call keyword arguments (line 347)
        kwargs_261435 = {}
        unicode_261432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 18), 'unicode', u'To view figure, visit {0}')
        # Obtaining the member 'format' of a type (line 347)
        format_261433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 18), unicode_261432, 'format')
        # Calling format(args, kwargs) (line 347)
        format_call_result_261436 = invoke(stypy.reporting.localization.Localization(__file__, 347, 18), format_261433, *[url_261434], **kwargs_261435)
        
        # Processing the call keyword arguments (line 347)
        kwargs_261437 = {}
        # Getting the type of 'print' (line 347)
        print_261431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'print', False)
        # Calling print(args, kwargs) (line 347)
        print_call_result_261438 = invoke(stypy.reporting.localization.Localization(__file__, 347, 12), print_261431, *[format_call_result_261436], **kwargs_261437)
        
        # SSA join for if statement (line 343)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to start(...): (line 349)
        # Processing the call keyword arguments (line 349)
        kwargs_261441 = {}
        # Getting the type of 'WebAggApplication' (line 349)
        WebAggApplication_261439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'WebAggApplication', False)
        # Obtaining the member 'start' of a type (line 349)
        start_261440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), WebAggApplication_261439, 'start')
        # Calling start(args, kwargs) (line 349)
        start_call_result_261442 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), start_261440, *[], **kwargs_261441)
        
        
        # ################# End of 'show(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'show' in the type store
        # Getting the type of 'stypy_return_type' (line 335)
        stypy_return_type_261443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_261443)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'show'
        return stypy_return_type_261443


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 326, 0, False)
        # Assigning a type to the variable 'self' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendWebAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendWebAgg' (line 326)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 0), '_BackendWebAgg', _BackendWebAgg)

# Assigning a Name to a Name (line 328):
# Getting the type of 'FigureCanvasWebAgg' (line 328)
FigureCanvasWebAgg_261444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 19), 'FigureCanvasWebAgg')
# Getting the type of '_BackendWebAgg'
_BackendWebAgg_261445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendWebAgg')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendWebAgg_261445, 'FigureCanvas', FigureCanvasWebAgg_261444)

# Assigning a Name to a Name (line 329):
# Getting the type of 'FigureManagerWebAgg' (line 329)
FigureManagerWebAgg_261446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'FigureManagerWebAgg')
# Getting the type of '_BackendWebAgg'
_BackendWebAgg_261447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendWebAgg')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendWebAgg_261447, 'FigureManager', FigureManagerWebAgg_261446)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
