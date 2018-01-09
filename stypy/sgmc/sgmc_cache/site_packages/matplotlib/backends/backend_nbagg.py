
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Interactive figures in the IPython notebook'''
2: # Note: There is a notebook in
3: # lib/matplotlib/backends/web_backend/nbagg_uat.ipynb to help verify
4: # that changes made maintain expected behaviour.
5: 
6: import datetime
7: from base64 import b64encode
8: import json
9: import io
10: import os
11: import six
12: from uuid import uuid4 as uuid
13: 
14: import tornado.ioloop
15: 
16: from IPython.display import display, Javascript, HTML
17: try:
18:     # Jupyter/IPython 4.x or later
19:     from ipykernel.comm import Comm
20: except ImportError:
21:     # Jupyter/IPython 3.x or earlier
22:     from IPython.kernel.comm import Comm
23: 
24: from matplotlib import rcParams, is_interactive
25: from matplotlib._pylab_helpers import Gcf
26: from matplotlib.backends.backend_webagg_core import (
27:     FigureCanvasWebAggCore, FigureManagerWebAgg, NavigationToolbar2WebAgg,
28:     TimerTornado)
29: from matplotlib.backend_bases import (
30:     _Backend, FigureCanvasBase, NavigationToolbar2)
31: from matplotlib.figure import Figure
32: from matplotlib import is_interactive
33: from matplotlib.backends.backend_webagg_core import (FigureManagerWebAgg,
34:                                                      FigureCanvasWebAggCore,
35:                                                      NavigationToolbar2WebAgg,
36:                                                      TimerTornado)
37: from matplotlib.backend_bases import (ShowBase, NavigationToolbar2,
38:                                       FigureCanvasBase)
39: 
40: 
41: def connection_info():
42:     '''
43:     Return a string showing the figure and connection status for
44:     the backend. This is intended as a diagnostic tool, and not for general
45:     use.
46: 
47:     '''
48:     result = []
49:     for manager in Gcf.get_all_fig_managers():
50:         fig = manager.canvas.figure
51:         result.append('{0} - {0}'.format((fig.get_label() or
52:                                           "Figure {0}".format(manager.num)),
53:                                          manager.web_sockets))
54:     if not is_interactive():
55:         result.append('Figures pending show: {0}'.format(len(Gcf._activeQue)))
56:     return '\n'.join(result)
57: 
58: 
59: # Note: Version 3.2 and 4.x icons
60: # http://fontawesome.io/3.2.1/icons/
61: # http://fontawesome.io/
62: # the `fa fa-xxx` part targets font-awesome 4, (IPython 3.x)
63: # the icon-xxx targets font awesome 3.21 (IPython 2.x)
64: _FONT_AWESOME_CLASSES = {
65:     'home': 'fa fa-home icon-home',
66:     'back': 'fa fa-arrow-left icon-arrow-left',
67:     'forward': 'fa fa-arrow-right icon-arrow-right',
68:     'zoom_to_rect': 'fa fa-square-o icon-check-empty',
69:     'move': 'fa fa-arrows icon-move',
70:     'download': 'fa fa-floppy-o icon-save',
71:     None: None
72: }
73: 
74: 
75: class NavigationIPy(NavigationToolbar2WebAgg):
76: 
77:     # Use the standard toolbar items + download button
78:     toolitems = [(text, tooltip_text,
79:                   _FONT_AWESOME_CLASSES[image_file], name_of_method)
80:                  for text, tooltip_text, image_file, name_of_method
81:                  in (NavigationToolbar2.toolitems +
82:                      (('Download', 'Download plot', 'download', 'download'),))
83:                  if image_file in _FONT_AWESOME_CLASSES]
84: 
85: 
86: class FigureManagerNbAgg(FigureManagerWebAgg):
87:     ToolbarCls = NavigationIPy
88: 
89:     def __init__(self, canvas, num):
90:         self._shown = False
91:         FigureManagerWebAgg.__init__(self, canvas, num)
92: 
93:     def display_js(self):
94:         # XXX How to do this just once? It has to deal with multiple
95:         # browser instances using the same kernel (require.js - but the
96:         # file isn't static?).
97:         display(Javascript(FigureManagerNbAgg.get_javascript()))
98: 
99:     def show(self):
100:         if not self._shown:
101:             self.display_js()
102:             self._create_comm()
103:         else:
104:             self.canvas.draw_idle()
105:         self._shown = True
106: 
107:     def reshow(self):
108:         '''
109:         A special method to re-show the figure in the notebook.
110: 
111:         '''
112:         self._shown = False
113:         self.show()
114: 
115:     @property
116:     def connected(self):
117:         return bool(self.web_sockets)
118: 
119:     @classmethod
120:     def get_javascript(cls, stream=None):
121:         if stream is None:
122:             output = io.StringIO()
123:         else:
124:             output = stream
125:         super(FigureManagerNbAgg, cls).get_javascript(stream=output)
126:         with io.open(os.path.join(
127:                 os.path.dirname(__file__),
128:                 "web_backend",
129:                 "nbagg_mpl.js"), encoding='utf8') as fd:
130:             output.write(fd.read())
131:         if stream is None:
132:             return output.getvalue()
133: 
134:     def _create_comm(self):
135:         comm = CommSocket(self)
136:         self.add_web_socket(comm)
137:         return comm
138: 
139:     def destroy(self):
140:         self._send_event('close')
141:         # need to copy comms as callbacks will modify this list
142:         for comm in list(self.web_sockets):
143:             comm.on_close()
144:         self.clearup_closed()
145: 
146:     def clearup_closed(self):
147:         '''Clear up any closed Comms.'''
148:         self.web_sockets = set([socket for socket in self.web_sockets
149:                                 if socket.is_open()])
150: 
151:         if len(self.web_sockets) == 0:
152:             self.canvas.close_event()
153: 
154:     def remove_comm(self, comm_id):
155:         self.web_sockets = set([socket for socket in self.web_sockets
156:                                 if not socket.comm.comm_id == comm_id])
157: 
158: 
159: class FigureCanvasNbAgg(FigureCanvasWebAggCore):
160:     def new_timer(self, *args, **kwargs):
161:         return TimerTornado(*args, **kwargs)
162: 
163: 
164: class CommSocket(object):
165:     '''
166:     Manages the Comm connection between IPython and the browser (client).
167: 
168:     Comms are 2 way, with the CommSocket being able to publish a message
169:     via the send_json method, and handle a message with on_message. On the
170:     JS side figure.send_message and figure.ws.onmessage do the sending and
171:     receiving respectively.
172: 
173:     '''
174:     def __init__(self, manager):
175:         self.supports_binary = None
176:         self.manager = manager
177:         self.uuid = str(uuid())
178:         # Publish an output area with a unique ID. The javascript can then
179:         # hook into this area.
180:         display(HTML("<div id=%r></div>" % self.uuid))
181:         try:
182:             self.comm = Comm('matplotlib', data={'id': self.uuid})
183:         except AttributeError:
184:             raise RuntimeError('Unable to create an IPython notebook Comm '
185:                                'instance. Are you in the IPython notebook?')
186:         self.comm.on_msg(self.on_message)
187: 
188:         manager = self.manager
189:         self._ext_close = False
190: 
191:         def _on_close(close_message):
192:             self._ext_close = True
193:             manager.remove_comm(close_message['content']['comm_id'])
194:             manager.clearup_closed()
195: 
196:         self.comm.on_close(_on_close)
197: 
198:     def is_open(self):
199:         return not (self._ext_close or self.comm._closed)
200: 
201:     def on_close(self):
202:         # When the socket is closed, deregister the websocket with
203:         # the FigureManager.
204:         if self.is_open():
205:             try:
206:                 self.comm.close()
207:             except KeyError:
208:                 # apparently already cleaned it up?
209:                 pass
210: 
211:     def send_json(self, content):
212:         self.comm.send({'data': json.dumps(content)})
213: 
214:     def send_binary(self, blob):
215:         # The comm is ascii, so we always send the image in base64
216:         # encoded data URL form.
217:         data = b64encode(blob)
218:         if six.PY3:
219:             data = data.decode('ascii')
220:         data_uri = "data:image/png;base64,{0}".format(data)
221:         self.comm.send({'data': data_uri})
222: 
223:     def on_message(self, message):
224:         # The 'supports_binary' message is relevant to the
225:         # websocket itself.  The other messages get passed along
226:         # to matplotlib as-is.
227: 
228:         # Every message has a "type" and a "figure_id".
229:         message = json.loads(message['content']['data'])
230:         if message['type'] == 'closing':
231:             self.on_close()
232:             self.manager.clearup_closed()
233:         elif message['type'] == 'supports_binary':
234:             self.supports_binary = message['value']
235:         else:
236:             self.manager.handle_json(message)
237: 
238: 
239: @_Backend.export
240: class _BackendNbAgg(_Backend):
241:     FigureCanvas = FigureCanvasNbAgg
242:     FigureManager = FigureManagerNbAgg
243: 
244:     @staticmethod
245:     def new_figure_manager_given_figure(num, figure):
246:         canvas = FigureCanvasNbAgg(figure)
247:         if rcParams['nbagg.transparent']:
248:             figure.patch.set_alpha(0)
249:         manager = FigureManagerNbAgg(canvas, num)
250:         if is_interactive():
251:             manager.show()
252:             figure.canvas.draw_idle()
253:         canvas.mpl_connect('close_event', lambda event: Gcf.destroy(num))
254:         return manager
255: 
256:     @staticmethod
257:     def trigger_manager_draw(manager):
258:         manager.show()
259: 
260:     @staticmethod
261:     def show():
262:         from matplotlib._pylab_helpers import Gcf
263: 
264:         managers = Gcf.get_all_fig_managers()
265:         if not managers:
266:             return
267: 
268:         interactive = is_interactive()
269: 
270:         for manager in managers:
271:             manager.show()
272: 
273:             # plt.figure adds an event which puts the figure in focus
274:             # in the activeQue. Disable this behaviour, as it results in
275:             # figures being put as the active figure after they have been
276:             # shown, even in non-interactive mode.
277:             if hasattr(manager, '_cidgcf'):
278:                 manager.canvas.mpl_disconnect(manager._cidgcf)
279: 
280:             if not interactive and manager in Gcf._activeQue:
281:                 Gcf._activeQue.remove(manager)
282: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_231057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Interactive figures in the IPython notebook')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import datetime' statement (line 6)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'datetime', datetime, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from base64 import b64encode' statement (line 7)
try:
    from base64 import b64encode

except:
    b64encode = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'base64', None, module_type_store, ['b64encode'], [b64encode])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import json' statement (line 8)
import json

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'json', json, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import io' statement (line 9)
import io

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'io', io, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import os' statement (line 10)
import os

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import six' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231058 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'six')

if (type(import_231058) is not StypyTypeError):

    if (import_231058 != 'pyd_module'):
        __import__(import_231058)
        sys_modules_231059 = sys.modules[import_231058]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'six', sys_modules_231059.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'six', import_231058)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from uuid import uuid' statement (line 12)
try:
    from uuid import uuid4 as uuid

except:
    uuid = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'uuid', None, module_type_store, ['uuid4'], [uuid])
# Adding an alias
module_type_store.add_alias('uuid', 'uuid4')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import tornado.ioloop' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231060 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'tornado.ioloop')

if (type(import_231060) is not StypyTypeError):

    if (import_231060 != 'pyd_module'):
        __import__(import_231060)
        sys_modules_231061 = sys.modules[import_231060]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'tornado.ioloop', sys_modules_231061.module_type_store, module_type_store)
    else:
        import tornado.ioloop

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'tornado.ioloop', tornado.ioloop, module_type_store)

else:
    # Assigning a type to the variable 'tornado.ioloop' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'tornado.ioloop', import_231060)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from IPython.display import display, Javascript, HTML' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231062 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'IPython.display')

if (type(import_231062) is not StypyTypeError):

    if (import_231062 != 'pyd_module'):
        __import__(import_231062)
        sys_modules_231063 = sys.modules[import_231062]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'IPython.display', sys_modules_231063.module_type_store, module_type_store, ['display', 'Javascript', 'HTML'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_231063, sys_modules_231063.module_type_store, module_type_store)
    else:
        from IPython.display import display, Javascript, HTML

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'IPython.display', None, module_type_store, ['display', 'Javascript', 'HTML'], [display, Javascript, HTML])

else:
    # Assigning a type to the variable 'IPython.display' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'IPython.display', import_231062)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')



# SSA begins for try-except statement (line 17)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 4))

# 'from ipykernel.comm import Comm' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231064 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'ipykernel.comm')

if (type(import_231064) is not StypyTypeError):

    if (import_231064 != 'pyd_module'):
        __import__(import_231064)
        sys_modules_231065 = sys.modules[import_231064]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'ipykernel.comm', sys_modules_231065.module_type_store, module_type_store, ['Comm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 4), __file__, sys_modules_231065, sys_modules_231065.module_type_store, module_type_store)
    else:
        from ipykernel.comm import Comm

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'ipykernel.comm', None, module_type_store, ['Comm'], [Comm])

else:
    # Assigning a type to the variable 'ipykernel.comm' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ipykernel.comm', import_231064)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# SSA branch for the except part of a try statement (line 17)
# SSA branch for the except 'ImportError' branch of a try statement (line 17)
module_type_store.open_ssa_branch('except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 4))

# 'from IPython.kernel.comm import Comm' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231066 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 4), 'IPython.kernel.comm')

if (type(import_231066) is not StypyTypeError):

    if (import_231066 != 'pyd_module'):
        __import__(import_231066)
        sys_modules_231067 = sys.modules[import_231066]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 4), 'IPython.kernel.comm', sys_modules_231067.module_type_store, module_type_store, ['Comm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 4), __file__, sys_modules_231067, sys_modules_231067.module_type_store, module_type_store)
    else:
        from IPython.kernel.comm import Comm

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 4), 'IPython.kernel.comm', None, module_type_store, ['Comm'], [Comm])

else:
    # Assigning a type to the variable 'IPython.kernel.comm' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'IPython.kernel.comm', import_231066)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# SSA join for try-except statement (line 17)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from matplotlib import rcParams, is_interactive' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231068 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib')

if (type(import_231068) is not StypyTypeError):

    if (import_231068 != 'pyd_module'):
        __import__(import_231068)
        sys_modules_231069 = sys.modules[import_231068]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', sys_modules_231069.module_type_store, module_type_store, ['rcParams', 'is_interactive'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_231069, sys_modules_231069.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams, is_interactive

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', None, module_type_store, ['rcParams', 'is_interactive'], [rcParams, is_interactive])

else:
    # Assigning a type to the variable 'matplotlib' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'matplotlib', import_231068)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from matplotlib._pylab_helpers import Gcf' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231070 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib._pylab_helpers')

if (type(import_231070) is not StypyTypeError):

    if (import_231070 != 'pyd_module'):
        __import__(import_231070)
        sys_modules_231071 = sys.modules[import_231070]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib._pylab_helpers', sys_modules_231071.module_type_store, module_type_store, ['Gcf'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_231071, sys_modules_231071.module_type_store, module_type_store)
    else:
        from matplotlib._pylab_helpers import Gcf

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib._pylab_helpers', None, module_type_store, ['Gcf'], [Gcf])

else:
    # Assigning a type to the variable 'matplotlib._pylab_helpers' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'matplotlib._pylab_helpers', import_231070)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from matplotlib.backends.backend_webagg_core import FigureCanvasWebAggCore, FigureManagerWebAgg, NavigationToolbar2WebAgg, TimerTornado' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231072 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.backends.backend_webagg_core')

if (type(import_231072) is not StypyTypeError):

    if (import_231072 != 'pyd_module'):
        __import__(import_231072)
        sys_modules_231073 = sys.modules[import_231072]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.backends.backend_webagg_core', sys_modules_231073.module_type_store, module_type_store, ['FigureCanvasWebAggCore', 'FigureManagerWebAgg', 'NavigationToolbar2WebAgg', 'TimerTornado'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_231073, sys_modules_231073.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_webagg_core import FigureCanvasWebAggCore, FigureManagerWebAgg, NavigationToolbar2WebAgg, TimerTornado

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.backends.backend_webagg_core', None, module_type_store, ['FigureCanvasWebAggCore', 'FigureManagerWebAgg', 'NavigationToolbar2WebAgg', 'TimerTornado'], [FigureCanvasWebAggCore, FigureManagerWebAgg, NavigationToolbar2WebAgg, TimerTornado])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_webagg_core' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'matplotlib.backends.backend_webagg_core', import_231072)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from matplotlib.backend_bases import _Backend, FigureCanvasBase, NavigationToolbar2' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231074 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.backend_bases')

if (type(import_231074) is not StypyTypeError):

    if (import_231074 != 'pyd_module'):
        __import__(import_231074)
        sys_modules_231075 = sys.modules[import_231074]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.backend_bases', sys_modules_231075.module_type_store, module_type_store, ['_Backend', 'FigureCanvasBase', 'NavigationToolbar2'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_231075, sys_modules_231075.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import _Backend, FigureCanvasBase, NavigationToolbar2

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.backend_bases', None, module_type_store, ['_Backend', 'FigureCanvasBase', 'NavigationToolbar2'], [_Backend, FigureCanvasBase, NavigationToolbar2])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'matplotlib.backend_bases', import_231074)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from matplotlib.figure import Figure' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231076 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.figure')

if (type(import_231076) is not StypyTypeError):

    if (import_231076 != 'pyd_module'):
        __import__(import_231076)
        sys_modules_231077 = sys.modules[import_231076]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.figure', sys_modules_231077.module_type_store, module_type_store, ['Figure'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_231077, sys_modules_231077.module_type_store, module_type_store)
    else:
        from matplotlib.figure import Figure

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.figure', None, module_type_store, ['Figure'], [Figure])

else:
    # Assigning a type to the variable 'matplotlib.figure' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'matplotlib.figure', import_231076)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from matplotlib import is_interactive' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231078 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib')

if (type(import_231078) is not StypyTypeError):

    if (import_231078 != 'pyd_module'):
        __import__(import_231078)
        sys_modules_231079 = sys.modules[import_231078]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib', sys_modules_231079.module_type_store, module_type_store, ['is_interactive'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_231079, sys_modules_231079.module_type_store, module_type_store)
    else:
        from matplotlib import is_interactive

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib', None, module_type_store, ['is_interactive'], [is_interactive])

else:
    # Assigning a type to the variable 'matplotlib' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'matplotlib', import_231078)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'from matplotlib.backends.backend_webagg_core import FigureManagerWebAgg, FigureCanvasWebAggCore, NavigationToolbar2WebAgg, TimerTornado' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231080 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backends.backend_webagg_core')

if (type(import_231080) is not StypyTypeError):

    if (import_231080 != 'pyd_module'):
        __import__(import_231080)
        sys_modules_231081 = sys.modules[import_231080]
        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backends.backend_webagg_core', sys_modules_231081.module_type_store, module_type_store, ['FigureManagerWebAgg', 'FigureCanvasWebAggCore', 'NavigationToolbar2WebAgg', 'TimerTornado'])
        nest_module(stypy.reporting.localization.Localization(__file__, 33, 0), __file__, sys_modules_231081, sys_modules_231081.module_type_store, module_type_store)
    else:
        from matplotlib.backends.backend_webagg_core import FigureManagerWebAgg, FigureCanvasWebAggCore, NavigationToolbar2WebAgg, TimerTornado

        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backends.backend_webagg_core', None, module_type_store, ['FigureManagerWebAgg', 'FigureCanvasWebAggCore', 'NavigationToolbar2WebAgg', 'TimerTornado'], [FigureManagerWebAgg, FigureCanvasWebAggCore, NavigationToolbar2WebAgg, TimerTornado])

else:
    # Assigning a type to the variable 'matplotlib.backends.backend_webagg_core' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'matplotlib.backends.backend_webagg_core', import_231080)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from matplotlib.backend_bases import ShowBase, NavigationToolbar2, FigureCanvasBase' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_231082 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.backend_bases')

if (type(import_231082) is not StypyTypeError):

    if (import_231082 != 'pyd_module'):
        __import__(import_231082)
        sys_modules_231083 = sys.modules[import_231082]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.backend_bases', sys_modules_231083.module_type_store, module_type_store, ['ShowBase', 'NavigationToolbar2', 'FigureCanvasBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_231083, sys_modules_231083.module_type_store, module_type_store)
    else:
        from matplotlib.backend_bases import ShowBase, NavigationToolbar2, FigureCanvasBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.backend_bases', None, module_type_store, ['ShowBase', 'NavigationToolbar2', 'FigureCanvasBase'], [ShowBase, NavigationToolbar2, FigureCanvasBase])

else:
    # Assigning a type to the variable 'matplotlib.backend_bases' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'matplotlib.backend_bases', import_231082)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')


@norecursion
def connection_info(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'connection_info'
    module_type_store = module_type_store.open_function_context('connection_info', 41, 0, False)
    
    # Passed parameters checking function
    connection_info.stypy_localization = localization
    connection_info.stypy_type_of_self = None
    connection_info.stypy_type_store = module_type_store
    connection_info.stypy_function_name = 'connection_info'
    connection_info.stypy_param_names_list = []
    connection_info.stypy_varargs_param_name = None
    connection_info.stypy_kwargs_param_name = None
    connection_info.stypy_call_defaults = defaults
    connection_info.stypy_call_varargs = varargs
    connection_info.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'connection_info', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'connection_info', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'connection_info(...)' code ##################

    str_231084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, (-1)), 'str', '\n    Return a string showing the figure and connection status for\n    the backend. This is intended as a diagnostic tool, and not for general\n    use.\n\n    ')
    
    # Assigning a List to a Name (line 48):
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_231085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    
    # Assigning a type to the variable 'result' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'result', list_231085)
    
    
    # Call to get_all_fig_managers(...): (line 49)
    # Processing the call keyword arguments (line 49)
    kwargs_231088 = {}
    # Getting the type of 'Gcf' (line 49)
    Gcf_231086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'Gcf', False)
    # Obtaining the member 'get_all_fig_managers' of a type (line 49)
    get_all_fig_managers_231087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), Gcf_231086, 'get_all_fig_managers')
    # Calling get_all_fig_managers(args, kwargs) (line 49)
    get_all_fig_managers_call_result_231089 = invoke(stypy.reporting.localization.Localization(__file__, 49, 19), get_all_fig_managers_231087, *[], **kwargs_231088)
    
    # Testing the type of a for loop iterable (line 49)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 4), get_all_fig_managers_call_result_231089)
    # Getting the type of the for loop variable (line 49)
    for_loop_var_231090 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 4), get_all_fig_managers_call_result_231089)
    # Assigning a type to the variable 'manager' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'manager', for_loop_var_231090)
    # SSA begins for a for statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Attribute to a Name (line 50):
    # Getting the type of 'manager' (line 50)
    manager_231091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'manager')
    # Obtaining the member 'canvas' of a type (line 50)
    canvas_231092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 14), manager_231091, 'canvas')
    # Obtaining the member 'figure' of a type (line 50)
    figure_231093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 14), canvas_231092, 'figure')
    # Assigning a type to the variable 'fig' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'fig', figure_231093)
    
    # Call to append(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to format(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Evaluating a boolean operation
    
    # Call to get_label(...): (line 51)
    # Processing the call keyword arguments (line 51)
    kwargs_231100 = {}
    # Getting the type of 'fig' (line 51)
    fig_231098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'fig', False)
    # Obtaining the member 'get_label' of a type (line 51)
    get_label_231099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 42), fig_231098, 'get_label')
    # Calling get_label(args, kwargs) (line 51)
    get_label_call_result_231101 = invoke(stypy.reporting.localization.Localization(__file__, 51, 42), get_label_231099, *[], **kwargs_231100)
    
    
    # Call to format(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'manager' (line 52)
    manager_231104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 62), 'manager', False)
    # Obtaining the member 'num' of a type (line 52)
    num_231105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 62), manager_231104, 'num')
    # Processing the call keyword arguments (line 52)
    kwargs_231106 = {}
    str_231102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 42), 'str', 'Figure {0}')
    # Obtaining the member 'format' of a type (line 52)
    format_231103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 42), str_231102, 'format')
    # Calling format(args, kwargs) (line 52)
    format_call_result_231107 = invoke(stypy.reporting.localization.Localization(__file__, 52, 42), format_231103, *[num_231105], **kwargs_231106)
    
    # Applying the binary operator 'or' (line 51)
    result_or_keyword_231108 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 42), 'or', get_label_call_result_231101, format_call_result_231107)
    
    # Getting the type of 'manager' (line 53)
    manager_231109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 41), 'manager', False)
    # Obtaining the member 'web_sockets' of a type (line 53)
    web_sockets_231110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 41), manager_231109, 'web_sockets')
    # Processing the call keyword arguments (line 51)
    kwargs_231111 = {}
    str_231096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 22), 'str', '{0} - {0}')
    # Obtaining the member 'format' of a type (line 51)
    format_231097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 22), str_231096, 'format')
    # Calling format(args, kwargs) (line 51)
    format_call_result_231112 = invoke(stypy.reporting.localization.Localization(__file__, 51, 22), format_231097, *[result_or_keyword_231108, web_sockets_231110], **kwargs_231111)
    
    # Processing the call keyword arguments (line 51)
    kwargs_231113 = {}
    # Getting the type of 'result' (line 51)
    result_231094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 51)
    append_231095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), result_231094, 'append')
    # Calling append(args, kwargs) (line 51)
    append_call_result_231114 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), append_231095, *[format_call_result_231112], **kwargs_231113)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to is_interactive(...): (line 54)
    # Processing the call keyword arguments (line 54)
    kwargs_231116 = {}
    # Getting the type of 'is_interactive' (line 54)
    is_interactive_231115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'is_interactive', False)
    # Calling is_interactive(args, kwargs) (line 54)
    is_interactive_call_result_231117 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), is_interactive_231115, *[], **kwargs_231116)
    
    # Applying the 'not' unary operator (line 54)
    result_not__231118 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), 'not', is_interactive_call_result_231117)
    
    # Testing the type of an if condition (line 54)
    if_condition_231119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), result_not__231118)
    # Assigning a type to the variable 'if_condition_231119' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_231119', if_condition_231119)
    # SSA begins for if statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Call to format(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Call to len(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'Gcf' (line 55)
    Gcf_231125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 61), 'Gcf', False)
    # Obtaining the member '_activeQue' of a type (line 55)
    _activeQue_231126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 61), Gcf_231125, '_activeQue')
    # Processing the call keyword arguments (line 55)
    kwargs_231127 = {}
    # Getting the type of 'len' (line 55)
    len_231124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 57), 'len', False)
    # Calling len(args, kwargs) (line 55)
    len_call_result_231128 = invoke(stypy.reporting.localization.Localization(__file__, 55, 57), len_231124, *[_activeQue_231126], **kwargs_231127)
    
    # Processing the call keyword arguments (line 55)
    kwargs_231129 = {}
    str_231122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'str', 'Figures pending show: {0}')
    # Obtaining the member 'format' of a type (line 55)
    format_231123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 22), str_231122, 'format')
    # Calling format(args, kwargs) (line 55)
    format_call_result_231130 = invoke(stypy.reporting.localization.Localization(__file__, 55, 22), format_231123, *[len_call_result_231128], **kwargs_231129)
    
    # Processing the call keyword arguments (line 55)
    kwargs_231131 = {}
    # Getting the type of 'result' (line 55)
    result_231120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 55)
    append_231121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), result_231120, 'append')
    # Calling append(args, kwargs) (line 55)
    append_call_result_231132 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), append_231121, *[format_call_result_231130], **kwargs_231131)
    
    # SSA join for if statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'result' (line 56)
    result_231135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'result', False)
    # Processing the call keyword arguments (line 56)
    kwargs_231136 = {}
    str_231133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 11), 'str', '\n')
    # Obtaining the member 'join' of a type (line 56)
    join_231134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), str_231133, 'join')
    # Calling join(args, kwargs) (line 56)
    join_call_result_231137 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), join_231134, *[result_231135], **kwargs_231136)
    
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type', join_call_result_231137)
    
    # ################# End of 'connection_info(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'connection_info' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_231138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_231138)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'connection_info'
    return stypy_return_type_231138

# Assigning a type to the variable 'connection_info' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'connection_info', connection_info)

# Assigning a Dict to a Name (line 64):

# Obtaining an instance of the builtin type 'dict' (line 64)
dict_231139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 24), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 64)
# Adding element type (key, value) (line 64)
str_231140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'str', 'home')
str_231141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 12), 'str', 'fa fa-home icon-home')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), dict_231139, (str_231140, str_231141))
# Adding element type (key, value) (line 64)
str_231142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'str', 'back')
str_231143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 12), 'str', 'fa fa-arrow-left icon-arrow-left')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), dict_231139, (str_231142, str_231143))
# Adding element type (key, value) (line 64)
str_231144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'str', 'forward')
str_231145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 15), 'str', 'fa fa-arrow-right icon-arrow-right')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), dict_231139, (str_231144, str_231145))
# Adding element type (key, value) (line 64)
str_231146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'str', 'zoom_to_rect')
str_231147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'str', 'fa fa-square-o icon-check-empty')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), dict_231139, (str_231146, str_231147))
# Adding element type (key, value) (line 64)
str_231148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'str', 'move')
str_231149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 12), 'str', 'fa fa-arrows icon-move')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), dict_231139, (str_231148, str_231149))
# Adding element type (key, value) (line 64)
str_231150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'str', 'download')
str_231151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 16), 'str', 'fa fa-floppy-o icon-save')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), dict_231139, (str_231150, str_231151))
# Adding element type (key, value) (line 64)
# Getting the type of 'None' (line 71)
None_231152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'None')
# Getting the type of 'None' (line 71)
None_231153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), dict_231139, (None_231152, None_231153))

# Assigning a type to the variable '_FONT_AWESOME_CLASSES' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), '_FONT_AWESOME_CLASSES', dict_231139)
# Declaration of the 'NavigationIPy' class
# Getting the type of 'NavigationToolbar2WebAgg' (line 75)
NavigationToolbar2WebAgg_231154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'NavigationToolbar2WebAgg')

class NavigationIPy(NavigationToolbar2WebAgg_231154, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 75, 0, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NavigationIPy.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NavigationIPy' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'NavigationIPy', NavigationIPy)

# Assigning a ListComp to a Name (line 78):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'NavigationToolbar2' (line 81)
NavigationToolbar2_231166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'NavigationToolbar2')
# Obtaining the member 'toolitems' of a type (line 81)
toolitems_231167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 21), NavigationToolbar2_231166, 'toolitems')

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_231168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)
# Adding element type (line 82)

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_231169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)
# Adding element type (line 82)
str_231170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'str', 'Download')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 23), tuple_231169, str_231170)
# Adding element type (line 82)
str_231171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 35), 'str', 'Download plot')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 23), tuple_231169, str_231171)
# Adding element type (line 82)
str_231172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 52), 'str', 'download')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 23), tuple_231169, str_231172)
# Adding element type (line 82)
str_231173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 64), 'str', 'download')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 23), tuple_231169, str_231173)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), tuple_231168, tuple_231169)

# Applying the binary operator '+' (line 81)
result_add_231174 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 21), '+', toolitems_231167, tuple_231168)

comprehension_231175 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 17), result_add_231174)
# Assigning a type to the variable 'text' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'text', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 17), comprehension_231175))
# Assigning a type to the variable 'tooltip_text' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'tooltip_text', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 17), comprehension_231175))
# Assigning a type to the variable 'image_file' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'image_file', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 17), comprehension_231175))
# Assigning a type to the variable 'name_of_method' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'name_of_method', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 17), comprehension_231175))

# Getting the type of 'image_file' (line 83)
image_file_231163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'image_file')
# Getting the type of '_FONT_AWESOME_CLASSES' (line 83)
_FONT_AWESOME_CLASSES_231164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), '_FONT_AWESOME_CLASSES')
# Applying the binary operator 'in' (line 83)
result_contains_231165 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 20), 'in', image_file_231163, _FONT_AWESOME_CLASSES_231164)


# Obtaining an instance of the builtin type 'tuple' (line 78)
tuple_231155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 78)
# Adding element type (line 78)
# Getting the type of 'text' (line 78)
text_231156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 18), 'text')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 18), tuple_231155, text_231156)
# Adding element type (line 78)
# Getting the type of 'tooltip_text' (line 78)
tooltip_text_231157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'tooltip_text')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 18), tuple_231155, tooltip_text_231157)
# Adding element type (line 78)

# Obtaining the type of the subscript
# Getting the type of 'image_file' (line 79)
image_file_231158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 40), 'image_file')
# Getting the type of '_FONT_AWESOME_CLASSES' (line 79)
_FONT_AWESOME_CLASSES_231159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), '_FONT_AWESOME_CLASSES')
# Obtaining the member '__getitem__' of a type (line 79)
getitem___231160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 18), _FONT_AWESOME_CLASSES_231159, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 79)
subscript_call_result_231161 = invoke(stypy.reporting.localization.Localization(__file__, 79, 18), getitem___231160, image_file_231158)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 18), tuple_231155, subscript_call_result_231161)
# Adding element type (line 78)
# Getting the type of 'name_of_method' (line 79)
name_of_method_231162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 53), 'name_of_method')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 18), tuple_231155, name_of_method_231162)

list_231176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 17), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 17), list_231176, tuple_231155)
# Getting the type of 'NavigationIPy'
NavigationIPy_231177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NavigationIPy')
# Setting the type of the member 'toolitems' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NavigationIPy_231177, 'toolitems', list_231176)
# Declaration of the 'FigureManagerNbAgg' class
# Getting the type of 'FigureManagerWebAgg' (line 86)
FigureManagerWebAgg_231178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'FigureManagerWebAgg')

class FigureManagerNbAgg(FigureManagerWebAgg_231178, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerNbAgg.__init__', ['canvas', 'num'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 90):
        # Getting the type of 'False' (line 90)
        False_231179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 22), 'False')
        # Getting the type of 'self' (line 90)
        self_231180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member '_shown' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_231180, '_shown', False_231179)
        
        # Call to __init__(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'self' (line 91)
        self_231183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'self', False)
        # Getting the type of 'canvas' (line 91)
        canvas_231184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'canvas', False)
        # Getting the type of 'num' (line 91)
        num_231185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 51), 'num', False)
        # Processing the call keyword arguments (line 91)
        kwargs_231186 = {}
        # Getting the type of 'FigureManagerWebAgg' (line 91)
        FigureManagerWebAgg_231181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'FigureManagerWebAgg', False)
        # Obtaining the member '__init__' of a type (line 91)
        init___231182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), FigureManagerWebAgg_231181, '__init__')
        # Calling __init__(args, kwargs) (line 91)
        init___call_result_231187 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), init___231182, *[self_231183, canvas_231184, num_231185], **kwargs_231186)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def display_js(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'display_js'
        module_type_store = module_type_store.open_function_context('display_js', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerNbAgg.display_js.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerNbAgg.display_js.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerNbAgg.display_js.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerNbAgg.display_js.__dict__.__setitem__('stypy_function_name', 'FigureManagerNbAgg.display_js')
        FigureManagerNbAgg.display_js.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerNbAgg.display_js.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerNbAgg.display_js.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerNbAgg.display_js.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerNbAgg.display_js.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerNbAgg.display_js.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerNbAgg.display_js.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerNbAgg.display_js', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'display_js', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'display_js(...)' code ##################

        
        # Call to display(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to Javascript(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to get_javascript(...): (line 97)
        # Processing the call keyword arguments (line 97)
        kwargs_231192 = {}
        # Getting the type of 'FigureManagerNbAgg' (line 97)
        FigureManagerNbAgg_231190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'FigureManagerNbAgg', False)
        # Obtaining the member 'get_javascript' of a type (line 97)
        get_javascript_231191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 27), FigureManagerNbAgg_231190, 'get_javascript')
        # Calling get_javascript(args, kwargs) (line 97)
        get_javascript_call_result_231193 = invoke(stypy.reporting.localization.Localization(__file__, 97, 27), get_javascript_231191, *[], **kwargs_231192)
        
        # Processing the call keyword arguments (line 97)
        kwargs_231194 = {}
        # Getting the type of 'Javascript' (line 97)
        Javascript_231189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'Javascript', False)
        # Calling Javascript(args, kwargs) (line 97)
        Javascript_call_result_231195 = invoke(stypy.reporting.localization.Localization(__file__, 97, 16), Javascript_231189, *[get_javascript_call_result_231193], **kwargs_231194)
        
        # Processing the call keyword arguments (line 97)
        kwargs_231196 = {}
        # Getting the type of 'display' (line 97)
        display_231188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'display', False)
        # Calling display(args, kwargs) (line 97)
        display_call_result_231197 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), display_231188, *[Javascript_call_result_231195], **kwargs_231196)
        
        
        # ################# End of 'display_js(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'display_js' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_231198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231198)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'display_js'
        return stypy_return_type_231198


    @norecursion
    def show(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'show'
        module_type_store = module_type_store.open_function_context('show', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerNbAgg.show.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerNbAgg.show.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerNbAgg.show.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerNbAgg.show.__dict__.__setitem__('stypy_function_name', 'FigureManagerNbAgg.show')
        FigureManagerNbAgg.show.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerNbAgg.show.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerNbAgg.show.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerNbAgg.show.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerNbAgg.show.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerNbAgg.show.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerNbAgg.show.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerNbAgg.show', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 100)
        self_231199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'self')
        # Obtaining the member '_shown' of a type (line 100)
        _shown_231200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), self_231199, '_shown')
        # Applying the 'not' unary operator (line 100)
        result_not__231201 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 11), 'not', _shown_231200)
        
        # Testing the type of an if condition (line 100)
        if_condition_231202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 8), result_not__231201)
        # Assigning a type to the variable 'if_condition_231202' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'if_condition_231202', if_condition_231202)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to display_js(...): (line 101)
        # Processing the call keyword arguments (line 101)
        kwargs_231205 = {}
        # Getting the type of 'self' (line 101)
        self_231203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self', False)
        # Obtaining the member 'display_js' of a type (line 101)
        display_js_231204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_231203, 'display_js')
        # Calling display_js(args, kwargs) (line 101)
        display_js_call_result_231206 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), display_js_231204, *[], **kwargs_231205)
        
        
        # Call to _create_comm(...): (line 102)
        # Processing the call keyword arguments (line 102)
        kwargs_231209 = {}
        # Getting the type of 'self' (line 102)
        self_231207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'self', False)
        # Obtaining the member '_create_comm' of a type (line 102)
        _create_comm_231208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), self_231207, '_create_comm')
        # Calling _create_comm(args, kwargs) (line 102)
        _create_comm_call_result_231210 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), _create_comm_231208, *[], **kwargs_231209)
        
        # SSA branch for the else part of an if statement (line 100)
        module_type_store.open_ssa_branch('else')
        
        # Call to draw_idle(...): (line 104)
        # Processing the call keyword arguments (line 104)
        kwargs_231214 = {}
        # Getting the type of 'self' (line 104)
        self_231211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'self', False)
        # Obtaining the member 'canvas' of a type (line 104)
        canvas_231212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), self_231211, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 104)
        draw_idle_231213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), canvas_231212, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 104)
        draw_idle_call_result_231215 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), draw_idle_231213, *[], **kwargs_231214)
        
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 105):
        # Getting the type of 'True' (line 105)
        True_231216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'True')
        # Getting the type of 'self' (line 105)
        self_231217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self')
        # Setting the type of the member '_shown' of a type (line 105)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_231217, '_shown', True_231216)
        
        # ################# End of 'show(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'show' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_231218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231218)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'show'
        return stypy_return_type_231218


    @norecursion
    def reshow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reshow'
        module_type_store = module_type_store.open_function_context('reshow', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerNbAgg.reshow.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerNbAgg.reshow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerNbAgg.reshow.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerNbAgg.reshow.__dict__.__setitem__('stypy_function_name', 'FigureManagerNbAgg.reshow')
        FigureManagerNbAgg.reshow.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerNbAgg.reshow.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerNbAgg.reshow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerNbAgg.reshow.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerNbAgg.reshow.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerNbAgg.reshow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerNbAgg.reshow.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerNbAgg.reshow', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reshow', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reshow(...)' code ##################

        str_231219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, (-1)), 'str', '\n        A special method to re-show the figure in the notebook.\n\n        ')
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'False' (line 112)
        False_231220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'False')
        # Getting the type of 'self' (line 112)
        self_231221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self')
        # Setting the type of the member '_shown' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_231221, '_shown', False_231220)
        
        # Call to show(...): (line 113)
        # Processing the call keyword arguments (line 113)
        kwargs_231224 = {}
        # Getting the type of 'self' (line 113)
        self_231222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self', False)
        # Obtaining the member 'show' of a type (line 113)
        show_231223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_231222, 'show')
        # Calling show(args, kwargs) (line 113)
        show_call_result_231225 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), show_231223, *[], **kwargs_231224)
        
        
        # ################# End of 'reshow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reshow' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_231226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231226)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reshow'
        return stypy_return_type_231226


    @norecursion
    def connected(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'connected'
        module_type_store = module_type_store.open_function_context('connected', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerNbAgg.connected.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerNbAgg.connected.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerNbAgg.connected.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerNbAgg.connected.__dict__.__setitem__('stypy_function_name', 'FigureManagerNbAgg.connected')
        FigureManagerNbAgg.connected.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerNbAgg.connected.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerNbAgg.connected.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerNbAgg.connected.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerNbAgg.connected.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerNbAgg.connected.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerNbAgg.connected.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerNbAgg.connected', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'connected', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'connected(...)' code ##################

        
        # Call to bool(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'self' (line 117)
        self_231228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 20), 'self', False)
        # Obtaining the member 'web_sockets' of a type (line 117)
        web_sockets_231229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 20), self_231228, 'web_sockets')
        # Processing the call keyword arguments (line 117)
        kwargs_231230 = {}
        # Getting the type of 'bool' (line 117)
        bool_231227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'bool', False)
        # Calling bool(args, kwargs) (line 117)
        bool_call_result_231231 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), bool_231227, *[web_sockets_231229], **kwargs_231230)
        
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', bool_call_result_231231)
        
        # ################# End of 'connected(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'connected' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_231232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231232)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'connected'
        return stypy_return_type_231232


    @norecursion
    def get_javascript(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 120)
        None_231233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 35), 'None')
        defaults = [None_231233]
        # Create a new context for function 'get_javascript'
        module_type_store = module_type_store.open_function_context('get_javascript', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerNbAgg.get_javascript.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerNbAgg.get_javascript.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerNbAgg.get_javascript.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerNbAgg.get_javascript.__dict__.__setitem__('stypy_function_name', 'FigureManagerNbAgg.get_javascript')
        FigureManagerNbAgg.get_javascript.__dict__.__setitem__('stypy_param_names_list', ['stream'])
        FigureManagerNbAgg.get_javascript.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerNbAgg.get_javascript.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerNbAgg.get_javascript.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerNbAgg.get_javascript.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerNbAgg.get_javascript.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerNbAgg.get_javascript.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerNbAgg.get_javascript', ['stream'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 121)
        # Getting the type of 'stream' (line 121)
        stream_231234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'stream')
        # Getting the type of 'None' (line 121)
        None_231235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'None')
        
        (may_be_231236, more_types_in_union_231237) = may_be_none(stream_231234, None_231235)

        if may_be_231236:

            if more_types_in_union_231237:
                # Runtime conditional SSA (line 121)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 122):
            
            # Call to StringIO(...): (line 122)
            # Processing the call keyword arguments (line 122)
            kwargs_231240 = {}
            # Getting the type of 'io' (line 122)
            io_231238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'io', False)
            # Obtaining the member 'StringIO' of a type (line 122)
            StringIO_231239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 21), io_231238, 'StringIO')
            # Calling StringIO(args, kwargs) (line 122)
            StringIO_call_result_231241 = invoke(stypy.reporting.localization.Localization(__file__, 122, 21), StringIO_231239, *[], **kwargs_231240)
            
            # Assigning a type to the variable 'output' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'output', StringIO_call_result_231241)

            if more_types_in_union_231237:
                # Runtime conditional SSA for else branch (line 121)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_231236) or more_types_in_union_231237):
            
            # Assigning a Name to a Name (line 124):
            # Getting the type of 'stream' (line 124)
            stream_231242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'stream')
            # Assigning a type to the variable 'output' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'output', stream_231242)

            if (may_be_231236 and more_types_in_union_231237):
                # SSA join for if statement (line 121)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to get_javascript(...): (line 125)
        # Processing the call keyword arguments (line 125)
        # Getting the type of 'output' (line 125)
        output_231249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 61), 'output', False)
        keyword_231250 = output_231249
        kwargs_231251 = {'stream': keyword_231250}
        
        # Call to super(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'FigureManagerNbAgg' (line 125)
        FigureManagerNbAgg_231244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 14), 'FigureManagerNbAgg', False)
        # Getting the type of 'cls' (line 125)
        cls_231245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 34), 'cls', False)
        # Processing the call keyword arguments (line 125)
        kwargs_231246 = {}
        # Getting the type of 'super' (line 125)
        super_231243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'super', False)
        # Calling super(args, kwargs) (line 125)
        super_call_result_231247 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), super_231243, *[FigureManagerNbAgg_231244, cls_231245], **kwargs_231246)
        
        # Obtaining the member 'get_javascript' of a type (line 125)
        get_javascript_231248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), super_call_result_231247, 'get_javascript')
        # Calling get_javascript(args, kwargs) (line 125)
        get_javascript_call_result_231252 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), get_javascript_231248, *[], **kwargs_231251)
        
        
        # Call to open(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Call to join(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Call to dirname(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of '__file__' (line 127)
        file___231261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 32), '__file__', False)
        # Processing the call keyword arguments (line 127)
        kwargs_231262 = {}
        # Getting the type of 'os' (line 127)
        os_231258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 127)
        path_231259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), os_231258, 'path')
        # Obtaining the member 'dirname' of a type (line 127)
        dirname_231260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), path_231259, 'dirname')
        # Calling dirname(args, kwargs) (line 127)
        dirname_call_result_231263 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), dirname_231260, *[file___231261], **kwargs_231262)
        
        str_231264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 16), 'str', 'web_backend')
        str_231265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 16), 'str', 'nbagg_mpl.js')
        # Processing the call keyword arguments (line 126)
        kwargs_231266 = {}
        # Getting the type of 'os' (line 126)
        os_231255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 126)
        path_231256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 21), os_231255, 'path')
        # Obtaining the member 'join' of a type (line 126)
        join_231257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 21), path_231256, 'join')
        # Calling join(args, kwargs) (line 126)
        join_call_result_231267 = invoke(stypy.reporting.localization.Localization(__file__, 126, 21), join_231257, *[dirname_call_result_231263, str_231264, str_231265], **kwargs_231266)
        
        # Processing the call keyword arguments (line 126)
        str_231268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 42), 'str', 'utf8')
        keyword_231269 = str_231268
        kwargs_231270 = {'encoding': keyword_231269}
        # Getting the type of 'io' (line 126)
        io_231253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'io', False)
        # Obtaining the member 'open' of a type (line 126)
        open_231254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 13), io_231253, 'open')
        # Calling open(args, kwargs) (line 126)
        open_call_result_231271 = invoke(stypy.reporting.localization.Localization(__file__, 126, 13), open_231254, *[join_call_result_231267], **kwargs_231270)
        
        with_231272 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 126, 13), open_call_result_231271, 'with parameter', '__enter__', '__exit__')

        if with_231272:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 126)
            enter___231273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 13), open_call_result_231271, '__enter__')
            with_enter_231274 = invoke(stypy.reporting.localization.Localization(__file__, 126, 13), enter___231273)
            # Assigning a type to the variable 'fd' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'fd', with_enter_231274)
            
            # Call to write(...): (line 130)
            # Processing the call arguments (line 130)
            
            # Call to read(...): (line 130)
            # Processing the call keyword arguments (line 130)
            kwargs_231279 = {}
            # Getting the type of 'fd' (line 130)
            fd_231277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'fd', False)
            # Obtaining the member 'read' of a type (line 130)
            read_231278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 25), fd_231277, 'read')
            # Calling read(args, kwargs) (line 130)
            read_call_result_231280 = invoke(stypy.reporting.localization.Localization(__file__, 130, 25), read_231278, *[], **kwargs_231279)
            
            # Processing the call keyword arguments (line 130)
            kwargs_231281 = {}
            # Getting the type of 'output' (line 130)
            output_231275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'output', False)
            # Obtaining the member 'write' of a type (line 130)
            write_231276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), output_231275, 'write')
            # Calling write(args, kwargs) (line 130)
            write_call_result_231282 = invoke(stypy.reporting.localization.Localization(__file__, 130, 12), write_231276, *[read_call_result_231280], **kwargs_231281)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 126)
            exit___231283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 13), open_call_result_231271, '__exit__')
            with_exit_231284 = invoke(stypy.reporting.localization.Localization(__file__, 126, 13), exit___231283, None, None, None)

        
        # Type idiom detected: calculating its left and rigth part (line 131)
        # Getting the type of 'stream' (line 131)
        stream_231285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'stream')
        # Getting the type of 'None' (line 131)
        None_231286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'None')
        
        (may_be_231287, more_types_in_union_231288) = may_be_none(stream_231285, None_231286)

        if may_be_231287:

            if more_types_in_union_231288:
                # Runtime conditional SSA (line 131)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to getvalue(...): (line 132)
            # Processing the call keyword arguments (line 132)
            kwargs_231291 = {}
            # Getting the type of 'output' (line 132)
            output_231289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'output', False)
            # Obtaining the member 'getvalue' of a type (line 132)
            getvalue_231290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 19), output_231289, 'getvalue')
            # Calling getvalue(args, kwargs) (line 132)
            getvalue_call_result_231292 = invoke(stypy.reporting.localization.Localization(__file__, 132, 19), getvalue_231290, *[], **kwargs_231291)
            
            # Assigning a type to the variable 'stypy_return_type' (line 132)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'stypy_return_type', getvalue_call_result_231292)

            if more_types_in_union_231288:
                # SSA join for if statement (line 131)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'get_javascript(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_javascript' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_231293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231293)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_javascript'
        return stypy_return_type_231293


    @norecursion
    def _create_comm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_create_comm'
        module_type_store = module_type_store.open_function_context('_create_comm', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerNbAgg._create_comm.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerNbAgg._create_comm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerNbAgg._create_comm.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerNbAgg._create_comm.__dict__.__setitem__('stypy_function_name', 'FigureManagerNbAgg._create_comm')
        FigureManagerNbAgg._create_comm.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerNbAgg._create_comm.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerNbAgg._create_comm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerNbAgg._create_comm.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerNbAgg._create_comm.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerNbAgg._create_comm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerNbAgg._create_comm.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerNbAgg._create_comm', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_create_comm', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_create_comm(...)' code ##################

        
        # Assigning a Call to a Name (line 135):
        
        # Call to CommSocket(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'self' (line 135)
        self_231295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 26), 'self', False)
        # Processing the call keyword arguments (line 135)
        kwargs_231296 = {}
        # Getting the type of 'CommSocket' (line 135)
        CommSocket_231294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'CommSocket', False)
        # Calling CommSocket(args, kwargs) (line 135)
        CommSocket_call_result_231297 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), CommSocket_231294, *[self_231295], **kwargs_231296)
        
        # Assigning a type to the variable 'comm' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'comm', CommSocket_call_result_231297)
        
        # Call to add_web_socket(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'comm' (line 136)
        comm_231300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'comm', False)
        # Processing the call keyword arguments (line 136)
        kwargs_231301 = {}
        # Getting the type of 'self' (line 136)
        self_231298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'self', False)
        # Obtaining the member 'add_web_socket' of a type (line 136)
        add_web_socket_231299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), self_231298, 'add_web_socket')
        # Calling add_web_socket(args, kwargs) (line 136)
        add_web_socket_call_result_231302 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), add_web_socket_231299, *[comm_231300], **kwargs_231301)
        
        # Getting the type of 'comm' (line 137)
        comm_231303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'comm')
        # Assigning a type to the variable 'stypy_return_type' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'stypy_return_type', comm_231303)
        
        # ################# End of '_create_comm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_create_comm' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_231304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231304)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_create_comm'
        return stypy_return_type_231304


    @norecursion
    def destroy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'destroy'
        module_type_store = module_type_store.open_function_context('destroy', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerNbAgg.destroy.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerNbAgg.destroy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerNbAgg.destroy.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerNbAgg.destroy.__dict__.__setitem__('stypy_function_name', 'FigureManagerNbAgg.destroy')
        FigureManagerNbAgg.destroy.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerNbAgg.destroy.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerNbAgg.destroy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerNbAgg.destroy.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerNbAgg.destroy.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerNbAgg.destroy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerNbAgg.destroy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerNbAgg.destroy', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to _send_event(...): (line 140)
        # Processing the call arguments (line 140)
        str_231307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 25), 'str', 'close')
        # Processing the call keyword arguments (line 140)
        kwargs_231308 = {}
        # Getting the type of 'self' (line 140)
        self_231305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self', False)
        # Obtaining the member '_send_event' of a type (line 140)
        _send_event_231306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_231305, '_send_event')
        # Calling _send_event(args, kwargs) (line 140)
        _send_event_call_result_231309 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), _send_event_231306, *[str_231307], **kwargs_231308)
        
        
        
        # Call to list(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_231311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), 'self', False)
        # Obtaining the member 'web_sockets' of a type (line 142)
        web_sockets_231312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 25), self_231311, 'web_sockets')
        # Processing the call keyword arguments (line 142)
        kwargs_231313 = {}
        # Getting the type of 'list' (line 142)
        list_231310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'list', False)
        # Calling list(args, kwargs) (line 142)
        list_call_result_231314 = invoke(stypy.reporting.localization.Localization(__file__, 142, 20), list_231310, *[web_sockets_231312], **kwargs_231313)
        
        # Testing the type of a for loop iterable (line 142)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 142, 8), list_call_result_231314)
        # Getting the type of the for loop variable (line 142)
        for_loop_var_231315 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 142, 8), list_call_result_231314)
        # Assigning a type to the variable 'comm' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'comm', for_loop_var_231315)
        # SSA begins for a for statement (line 142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to on_close(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_231318 = {}
        # Getting the type of 'comm' (line 143)
        comm_231316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'comm', False)
        # Obtaining the member 'on_close' of a type (line 143)
        on_close_231317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), comm_231316, 'on_close')
        # Calling on_close(args, kwargs) (line 143)
        on_close_call_result_231319 = invoke(stypy.reporting.localization.Localization(__file__, 143, 12), on_close_231317, *[], **kwargs_231318)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to clearup_closed(...): (line 144)
        # Processing the call keyword arguments (line 144)
        kwargs_231322 = {}
        # Getting the type of 'self' (line 144)
        self_231320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self', False)
        # Obtaining the member 'clearup_closed' of a type (line 144)
        clearup_closed_231321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_231320, 'clearup_closed')
        # Calling clearup_closed(args, kwargs) (line 144)
        clearup_closed_call_result_231323 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), clearup_closed_231321, *[], **kwargs_231322)
        
        
        # ################# End of 'destroy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'destroy' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_231324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231324)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'destroy'
        return stypy_return_type_231324


    @norecursion
    def clearup_closed(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clearup_closed'
        module_type_store = module_type_store.open_function_context('clearup_closed', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerNbAgg.clearup_closed.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerNbAgg.clearup_closed.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerNbAgg.clearup_closed.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerNbAgg.clearup_closed.__dict__.__setitem__('stypy_function_name', 'FigureManagerNbAgg.clearup_closed')
        FigureManagerNbAgg.clearup_closed.__dict__.__setitem__('stypy_param_names_list', [])
        FigureManagerNbAgg.clearup_closed.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerNbAgg.clearup_closed.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerNbAgg.clearup_closed.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerNbAgg.clearup_closed.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerNbAgg.clearup_closed.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerNbAgg.clearup_closed.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerNbAgg.clearup_closed', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clearup_closed', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clearup_closed(...)' code ##################

        str_231325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 8), 'str', 'Clear up any closed Comms.')
        
        # Assigning a Call to a Attribute (line 148):
        
        # Call to set(...): (line 148)
        # Processing the call arguments (line 148)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 148)
        self_231332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 53), 'self', False)
        # Obtaining the member 'web_sockets' of a type (line 148)
        web_sockets_231333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 53), self_231332, 'web_sockets')
        comprehension_231334 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 32), web_sockets_231333)
        # Assigning a type to the variable 'socket' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 32), 'socket', comprehension_231334)
        
        # Call to is_open(...): (line 149)
        # Processing the call keyword arguments (line 149)
        kwargs_231330 = {}
        # Getting the type of 'socket' (line 149)
        socket_231328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 35), 'socket', False)
        # Obtaining the member 'is_open' of a type (line 149)
        is_open_231329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 35), socket_231328, 'is_open')
        # Calling is_open(args, kwargs) (line 149)
        is_open_call_result_231331 = invoke(stypy.reporting.localization.Localization(__file__, 149, 35), is_open_231329, *[], **kwargs_231330)
        
        # Getting the type of 'socket' (line 148)
        socket_231327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 32), 'socket', False)
        list_231335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 32), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 32), list_231335, socket_231327)
        # Processing the call keyword arguments (line 148)
        kwargs_231336 = {}
        # Getting the type of 'set' (line 148)
        set_231326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'set', False)
        # Calling set(args, kwargs) (line 148)
        set_call_result_231337 = invoke(stypy.reporting.localization.Localization(__file__, 148, 27), set_231326, *[list_231335], **kwargs_231336)
        
        # Getting the type of 'self' (line 148)
        self_231338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self')
        # Setting the type of the member 'web_sockets' of a type (line 148)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_231338, 'web_sockets', set_call_result_231337)
        
        
        
        # Call to len(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'self' (line 151)
        self_231340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'self', False)
        # Obtaining the member 'web_sockets' of a type (line 151)
        web_sockets_231341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 15), self_231340, 'web_sockets')
        # Processing the call keyword arguments (line 151)
        kwargs_231342 = {}
        # Getting the type of 'len' (line 151)
        len_231339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'len', False)
        # Calling len(args, kwargs) (line 151)
        len_call_result_231343 = invoke(stypy.reporting.localization.Localization(__file__, 151, 11), len_231339, *[web_sockets_231341], **kwargs_231342)
        
        int_231344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 36), 'int')
        # Applying the binary operator '==' (line 151)
        result_eq_231345 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), '==', len_call_result_231343, int_231344)
        
        # Testing the type of an if condition (line 151)
        if_condition_231346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_eq_231345)
        # Assigning a type to the variable 'if_condition_231346' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_231346', if_condition_231346)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close_event(...): (line 152)
        # Processing the call keyword arguments (line 152)
        kwargs_231350 = {}
        # Getting the type of 'self' (line 152)
        self_231347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'self', False)
        # Obtaining the member 'canvas' of a type (line 152)
        canvas_231348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), self_231347, 'canvas')
        # Obtaining the member 'close_event' of a type (line 152)
        close_event_231349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), canvas_231348, 'close_event')
        # Calling close_event(args, kwargs) (line 152)
        close_event_call_result_231351 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), close_event_231349, *[], **kwargs_231350)
        
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'clearup_closed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clearup_closed' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_231352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231352)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clearup_closed'
        return stypy_return_type_231352


    @norecursion
    def remove_comm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove_comm'
        module_type_store = module_type_store.open_function_context('remove_comm', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureManagerNbAgg.remove_comm.__dict__.__setitem__('stypy_localization', localization)
        FigureManagerNbAgg.remove_comm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureManagerNbAgg.remove_comm.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureManagerNbAgg.remove_comm.__dict__.__setitem__('stypy_function_name', 'FigureManagerNbAgg.remove_comm')
        FigureManagerNbAgg.remove_comm.__dict__.__setitem__('stypy_param_names_list', ['comm_id'])
        FigureManagerNbAgg.remove_comm.__dict__.__setitem__('stypy_varargs_param_name', None)
        FigureManagerNbAgg.remove_comm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FigureManagerNbAgg.remove_comm.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureManagerNbAgg.remove_comm.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureManagerNbAgg.remove_comm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureManagerNbAgg.remove_comm.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureManagerNbAgg.remove_comm', ['comm_id'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove_comm', localization, ['comm_id'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove_comm(...)' code ##################

        
        # Assigning a Call to a Attribute (line 155):
        
        # Call to set(...): (line 155)
        # Processing the call arguments (line 155)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 155)
        self_231361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 53), 'self', False)
        # Obtaining the member 'web_sockets' of a type (line 155)
        web_sockets_231362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 53), self_231361, 'web_sockets')
        comprehension_231363 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 32), web_sockets_231362)
        # Assigning a type to the variable 'socket' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'socket', comprehension_231363)
        
        
        # Getting the type of 'socket' (line 156)
        socket_231355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 39), 'socket', False)
        # Obtaining the member 'comm' of a type (line 156)
        comm_231356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 39), socket_231355, 'comm')
        # Obtaining the member 'comm_id' of a type (line 156)
        comm_id_231357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 39), comm_231356, 'comm_id')
        # Getting the type of 'comm_id' (line 156)
        comm_id_231358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 62), 'comm_id', False)
        # Applying the binary operator '==' (line 156)
        result_eq_231359 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 39), '==', comm_id_231357, comm_id_231358)
        
        # Applying the 'not' unary operator (line 156)
        result_not__231360 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 35), 'not', result_eq_231359)
        
        # Getting the type of 'socket' (line 155)
        socket_231354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'socket', False)
        list_231364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 32), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 32), list_231364, socket_231354)
        # Processing the call keyword arguments (line 155)
        kwargs_231365 = {}
        # Getting the type of 'set' (line 155)
        set_231353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'set', False)
        # Calling set(args, kwargs) (line 155)
        set_call_result_231366 = invoke(stypy.reporting.localization.Localization(__file__, 155, 27), set_231353, *[list_231364], **kwargs_231365)
        
        # Getting the type of 'self' (line 155)
        self_231367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self')
        # Setting the type of the member 'web_sockets' of a type (line 155)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_231367, 'web_sockets', set_call_result_231366)
        
        # ################# End of 'remove_comm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove_comm' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_231368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231368)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove_comm'
        return stypy_return_type_231368


# Assigning a type to the variable 'FigureManagerNbAgg' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'FigureManagerNbAgg', FigureManagerNbAgg)

# Assigning a Name to a Name (line 87):
# Getting the type of 'NavigationIPy' (line 87)
NavigationIPy_231369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'NavigationIPy')
# Getting the type of 'FigureManagerNbAgg'
FigureManagerNbAgg_231370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FigureManagerNbAgg')
# Setting the type of the member 'ToolbarCls' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FigureManagerNbAgg_231370, 'ToolbarCls', NavigationIPy_231369)
# Declaration of the 'FigureCanvasNbAgg' class
# Getting the type of 'FigureCanvasWebAggCore' (line 159)
FigureCanvasWebAggCore_231371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'FigureCanvasWebAggCore')

class FigureCanvasNbAgg(FigureCanvasWebAggCore_231371, ):

    @norecursion
    def new_timer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_timer'
        module_type_store = module_type_store.open_function_context('new_timer', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FigureCanvasNbAgg.new_timer.__dict__.__setitem__('stypy_localization', localization)
        FigureCanvasNbAgg.new_timer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FigureCanvasNbAgg.new_timer.__dict__.__setitem__('stypy_type_store', module_type_store)
        FigureCanvasNbAgg.new_timer.__dict__.__setitem__('stypy_function_name', 'FigureCanvasNbAgg.new_timer')
        FigureCanvasNbAgg.new_timer.__dict__.__setitem__('stypy_param_names_list', [])
        FigureCanvasNbAgg.new_timer.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        FigureCanvasNbAgg.new_timer.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        FigureCanvasNbAgg.new_timer.__dict__.__setitem__('stypy_call_defaults', defaults)
        FigureCanvasNbAgg.new_timer.__dict__.__setitem__('stypy_call_varargs', varargs)
        FigureCanvasNbAgg.new_timer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FigureCanvasNbAgg.new_timer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasNbAgg.new_timer', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        
        # Call to TimerTornado(...): (line 161)
        # Getting the type of 'args' (line 161)
        args_231373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'args', False)
        # Processing the call keyword arguments (line 161)
        # Getting the type of 'kwargs' (line 161)
        kwargs_231374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'kwargs', False)
        kwargs_231375 = {'kwargs_231374': kwargs_231374}
        # Getting the type of 'TimerTornado' (line 161)
        TimerTornado_231372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'TimerTornado', False)
        # Calling TimerTornado(args, kwargs) (line 161)
        TimerTornado_call_result_231376 = invoke(stypy.reporting.localization.Localization(__file__, 161, 15), TimerTornado_231372, *[args_231373], **kwargs_231375)
        
        # Assigning a type to the variable 'stypy_return_type' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'stypy_return_type', TimerTornado_call_result_231376)
        
        # ################# End of 'new_timer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_timer' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_231377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231377)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_timer'
        return stypy_return_type_231377


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 159, 0, False)
        # Assigning a type to the variable 'self' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FigureCanvasNbAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'FigureCanvasNbAgg' (line 159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), 'FigureCanvasNbAgg', FigureCanvasNbAgg)
# Declaration of the 'CommSocket' class

class CommSocket(object, ):
    str_231378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, (-1)), 'str', '\n    Manages the Comm connection between IPython and the browser (client).\n\n    Comms are 2 way, with the CommSocket being able to publish a message\n    via the send_json method, and handle a message with on_message. On the\n    JS side figure.send_message and figure.ws.onmessage do the sending and\n    receiving respectively.\n\n    ')

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommSocket.__init__', ['manager'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['manager'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 175):
        # Getting the type of 'None' (line 175)
        None_231379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 31), 'None')
        # Getting the type of 'self' (line 175)
        self_231380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self')
        # Setting the type of the member 'supports_binary' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_231380, 'supports_binary', None_231379)
        
        # Assigning a Name to a Attribute (line 176):
        # Getting the type of 'manager' (line 176)
        manager_231381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 'manager')
        # Getting the type of 'self' (line 176)
        self_231382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self')
        # Setting the type of the member 'manager' of a type (line 176)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_231382, 'manager', manager_231381)
        
        # Assigning a Call to a Attribute (line 177):
        
        # Call to str(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to uuid(...): (line 177)
        # Processing the call keyword arguments (line 177)
        kwargs_231385 = {}
        # Getting the type of 'uuid' (line 177)
        uuid_231384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'uuid', False)
        # Calling uuid(args, kwargs) (line 177)
        uuid_call_result_231386 = invoke(stypy.reporting.localization.Localization(__file__, 177, 24), uuid_231384, *[], **kwargs_231385)
        
        # Processing the call keyword arguments (line 177)
        kwargs_231387 = {}
        # Getting the type of 'str' (line 177)
        str_231383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'str', False)
        # Calling str(args, kwargs) (line 177)
        str_call_result_231388 = invoke(stypy.reporting.localization.Localization(__file__, 177, 20), str_231383, *[uuid_call_result_231386], **kwargs_231387)
        
        # Getting the type of 'self' (line 177)
        self_231389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self')
        # Setting the type of the member 'uuid' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_231389, 'uuid', str_call_result_231388)
        
        # Call to display(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Call to HTML(...): (line 180)
        # Processing the call arguments (line 180)
        str_231392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 21), 'str', '<div id=%r></div>')
        # Getting the type of 'self' (line 180)
        self_231393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 43), 'self', False)
        # Obtaining the member 'uuid' of a type (line 180)
        uuid_231394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 43), self_231393, 'uuid')
        # Applying the binary operator '%' (line 180)
        result_mod_231395 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 21), '%', str_231392, uuid_231394)
        
        # Processing the call keyword arguments (line 180)
        kwargs_231396 = {}
        # Getting the type of 'HTML' (line 180)
        HTML_231391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'HTML', False)
        # Calling HTML(args, kwargs) (line 180)
        HTML_call_result_231397 = invoke(stypy.reporting.localization.Localization(__file__, 180, 16), HTML_231391, *[result_mod_231395], **kwargs_231396)
        
        # Processing the call keyword arguments (line 180)
        kwargs_231398 = {}
        # Getting the type of 'display' (line 180)
        display_231390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'display', False)
        # Calling display(args, kwargs) (line 180)
        display_call_result_231399 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), display_231390, *[HTML_call_result_231397], **kwargs_231398)
        
        
        
        # SSA begins for try-except statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Attribute (line 182):
        
        # Call to Comm(...): (line 182)
        # Processing the call arguments (line 182)
        str_231401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 29), 'str', 'matplotlib')
        # Processing the call keyword arguments (line 182)
        
        # Obtaining an instance of the builtin type 'dict' (line 182)
        dict_231402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 48), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 182)
        # Adding element type (key, value) (line 182)
        str_231403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 49), 'str', 'id')
        # Getting the type of 'self' (line 182)
        self_231404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 55), 'self', False)
        # Obtaining the member 'uuid' of a type (line 182)
        uuid_231405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 55), self_231404, 'uuid')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 48), dict_231402, (str_231403, uuid_231405))
        
        keyword_231406 = dict_231402
        kwargs_231407 = {'data': keyword_231406}
        # Getting the type of 'Comm' (line 182)
        Comm_231400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'Comm', False)
        # Calling Comm(args, kwargs) (line 182)
        Comm_call_result_231408 = invoke(stypy.reporting.localization.Localization(__file__, 182, 24), Comm_231400, *[str_231401], **kwargs_231407)
        
        # Getting the type of 'self' (line 182)
        self_231409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'self')
        # Setting the type of the member 'comm' of a type (line 182)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), self_231409, 'comm', Comm_call_result_231408)
        # SSA branch for the except part of a try statement (line 181)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 181)
        module_type_store.open_ssa_branch('except')
        
        # Call to RuntimeError(...): (line 184)
        # Processing the call arguments (line 184)
        str_231411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 31), 'str', 'Unable to create an IPython notebook Comm instance. Are you in the IPython notebook?')
        # Processing the call keyword arguments (line 184)
        kwargs_231412 = {}
        # Getting the type of 'RuntimeError' (line 184)
        RuntimeError_231410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 184)
        RuntimeError_call_result_231413 = invoke(stypy.reporting.localization.Localization(__file__, 184, 18), RuntimeError_231410, *[str_231411], **kwargs_231412)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 184, 12), RuntimeError_call_result_231413, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 181)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to on_msg(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'self' (line 186)
        self_231417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'self', False)
        # Obtaining the member 'on_message' of a type (line 186)
        on_message_231418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 25), self_231417, 'on_message')
        # Processing the call keyword arguments (line 186)
        kwargs_231419 = {}
        # Getting the type of 'self' (line 186)
        self_231414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self', False)
        # Obtaining the member 'comm' of a type (line 186)
        comm_231415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_231414, 'comm')
        # Obtaining the member 'on_msg' of a type (line 186)
        on_msg_231416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), comm_231415, 'on_msg')
        # Calling on_msg(args, kwargs) (line 186)
        on_msg_call_result_231420 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), on_msg_231416, *[on_message_231418], **kwargs_231419)
        
        
        # Assigning a Attribute to a Name (line 188):
        # Getting the type of 'self' (line 188)
        self_231421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 'self')
        # Obtaining the member 'manager' of a type (line 188)
        manager_231422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 18), self_231421, 'manager')
        # Assigning a type to the variable 'manager' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'manager', manager_231422)
        
        # Assigning a Name to a Attribute (line 189):
        # Getting the type of 'False' (line 189)
        False_231423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 26), 'False')
        # Getting the type of 'self' (line 189)
        self_231424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'self')
        # Setting the type of the member '_ext_close' of a type (line 189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), self_231424, '_ext_close', False_231423)

        @norecursion
        def _on_close(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_on_close'
            module_type_store = module_type_store.open_function_context('_on_close', 191, 8, False)
            
            # Passed parameters checking function
            _on_close.stypy_localization = localization
            _on_close.stypy_type_of_self = None
            _on_close.stypy_type_store = module_type_store
            _on_close.stypy_function_name = '_on_close'
            _on_close.stypy_param_names_list = ['close_message']
            _on_close.stypy_varargs_param_name = None
            _on_close.stypy_kwargs_param_name = None
            _on_close.stypy_call_defaults = defaults
            _on_close.stypy_call_varargs = varargs
            _on_close.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_on_close', ['close_message'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_on_close', localization, ['close_message'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_on_close(...)' code ##################

            
            # Assigning a Name to a Attribute (line 192):
            # Getting the type of 'True' (line 192)
            True_231425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 30), 'True')
            # Getting the type of 'self' (line 192)
            self_231426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'self')
            # Setting the type of the member '_ext_close' of a type (line 192)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), self_231426, '_ext_close', True_231425)
            
            # Call to remove_comm(...): (line 193)
            # Processing the call arguments (line 193)
            
            # Obtaining the type of the subscript
            str_231429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 57), 'str', 'comm_id')
            
            # Obtaining the type of the subscript
            str_231430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 46), 'str', 'content')
            # Getting the type of 'close_message' (line 193)
            close_message_231431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 32), 'close_message', False)
            # Obtaining the member '__getitem__' of a type (line 193)
            getitem___231432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 32), close_message_231431, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 193)
            subscript_call_result_231433 = invoke(stypy.reporting.localization.Localization(__file__, 193, 32), getitem___231432, str_231430)
            
            # Obtaining the member '__getitem__' of a type (line 193)
            getitem___231434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 32), subscript_call_result_231433, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 193)
            subscript_call_result_231435 = invoke(stypy.reporting.localization.Localization(__file__, 193, 32), getitem___231434, str_231429)
            
            # Processing the call keyword arguments (line 193)
            kwargs_231436 = {}
            # Getting the type of 'manager' (line 193)
            manager_231427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'manager', False)
            # Obtaining the member 'remove_comm' of a type (line 193)
            remove_comm_231428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), manager_231427, 'remove_comm')
            # Calling remove_comm(args, kwargs) (line 193)
            remove_comm_call_result_231437 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), remove_comm_231428, *[subscript_call_result_231435], **kwargs_231436)
            
            
            # Call to clearup_closed(...): (line 194)
            # Processing the call keyword arguments (line 194)
            kwargs_231440 = {}
            # Getting the type of 'manager' (line 194)
            manager_231438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'manager', False)
            # Obtaining the member 'clearup_closed' of a type (line 194)
            clearup_closed_231439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), manager_231438, 'clearup_closed')
            # Calling clearup_closed(args, kwargs) (line 194)
            clearup_closed_call_result_231441 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), clearup_closed_231439, *[], **kwargs_231440)
            
            
            # ################# End of '_on_close(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_on_close' in the type store
            # Getting the type of 'stypy_return_type' (line 191)
            stypy_return_type_231442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231442)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_on_close'
            return stypy_return_type_231442

        # Assigning a type to the variable '_on_close' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), '_on_close', _on_close)
        
        # Call to on_close(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of '_on_close' (line 196)
        _on_close_231446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), '_on_close', False)
        # Processing the call keyword arguments (line 196)
        kwargs_231447 = {}
        # Getting the type of 'self' (line 196)
        self_231443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self', False)
        # Obtaining the member 'comm' of a type (line 196)
        comm_231444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_231443, 'comm')
        # Obtaining the member 'on_close' of a type (line 196)
        on_close_231445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), comm_231444, 'on_close')
        # Calling on_close(args, kwargs) (line 196)
        on_close_call_result_231448 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), on_close_231445, *[_on_close_231446], **kwargs_231447)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def is_open(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_open'
        module_type_store = module_type_store.open_function_context('is_open', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommSocket.is_open.__dict__.__setitem__('stypy_localization', localization)
        CommSocket.is_open.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommSocket.is_open.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommSocket.is_open.__dict__.__setitem__('stypy_function_name', 'CommSocket.is_open')
        CommSocket.is_open.__dict__.__setitem__('stypy_param_names_list', [])
        CommSocket.is_open.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommSocket.is_open.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommSocket.is_open.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommSocket.is_open.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommSocket.is_open.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommSocket.is_open.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommSocket.is_open', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_open', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_open(...)' code ##################

        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 199)
        self_231449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'self')
        # Obtaining the member '_ext_close' of a type (line 199)
        _ext_close_231450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 20), self_231449, '_ext_close')
        # Getting the type of 'self' (line 199)
        self_231451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'self')
        # Obtaining the member 'comm' of a type (line 199)
        comm_231452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 39), self_231451, 'comm')
        # Obtaining the member '_closed' of a type (line 199)
        _closed_231453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 39), comm_231452, '_closed')
        # Applying the binary operator 'or' (line 199)
        result_or_keyword_231454 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 20), 'or', _ext_close_231450, _closed_231453)
        
        # Applying the 'not' unary operator (line 199)
        result_not__231455 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 15), 'not', result_or_keyword_231454)
        
        # Assigning a type to the variable 'stypy_return_type' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'stypy_return_type', result_not__231455)
        
        # ################# End of 'is_open(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_open' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_231456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231456)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_open'
        return stypy_return_type_231456


    @norecursion
    def on_close(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'on_close'
        module_type_store = module_type_store.open_function_context('on_close', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommSocket.on_close.__dict__.__setitem__('stypy_localization', localization)
        CommSocket.on_close.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommSocket.on_close.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommSocket.on_close.__dict__.__setitem__('stypy_function_name', 'CommSocket.on_close')
        CommSocket.on_close.__dict__.__setitem__('stypy_param_names_list', [])
        CommSocket.on_close.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommSocket.on_close.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommSocket.on_close.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommSocket.on_close.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommSocket.on_close.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommSocket.on_close.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommSocket.on_close', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to is_open(...): (line 204)
        # Processing the call keyword arguments (line 204)
        kwargs_231459 = {}
        # Getting the type of 'self' (line 204)
        self_231457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'self', False)
        # Obtaining the member 'is_open' of a type (line 204)
        is_open_231458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 11), self_231457, 'is_open')
        # Calling is_open(args, kwargs) (line 204)
        is_open_call_result_231460 = invoke(stypy.reporting.localization.Localization(__file__, 204, 11), is_open_231458, *[], **kwargs_231459)
        
        # Testing the type of an if condition (line 204)
        if_condition_231461 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), is_open_call_result_231460)
        # Assigning a type to the variable 'if_condition_231461' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'if_condition_231461', if_condition_231461)
        # SSA begins for if statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to close(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_231465 = {}
        # Getting the type of 'self' (line 206)
        self_231462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'self', False)
        # Obtaining the member 'comm' of a type (line 206)
        comm_231463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 16), self_231462, 'comm')
        # Obtaining the member 'close' of a type (line 206)
        close_231464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 16), comm_231463, 'close')
        # Calling close(args, kwargs) (line 206)
        close_call_result_231466 = invoke(stypy.reporting.localization.Localization(__file__, 206, 16), close_231464, *[], **kwargs_231465)
        
        # SSA branch for the except part of a try statement (line 205)
        # SSA branch for the except 'KeyError' branch of a try statement (line 205)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 204)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'on_close(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'on_close' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_231467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231467)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'on_close'
        return stypy_return_type_231467


    @norecursion
    def send_json(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'send_json'
        module_type_store = module_type_store.open_function_context('send_json', 211, 4, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommSocket.send_json.__dict__.__setitem__('stypy_localization', localization)
        CommSocket.send_json.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommSocket.send_json.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommSocket.send_json.__dict__.__setitem__('stypy_function_name', 'CommSocket.send_json')
        CommSocket.send_json.__dict__.__setitem__('stypy_param_names_list', ['content'])
        CommSocket.send_json.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommSocket.send_json.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommSocket.send_json.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommSocket.send_json.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommSocket.send_json.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommSocket.send_json.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommSocket.send_json', ['content'], None, None, defaults, varargs, kwargs)

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

        
        # Call to send(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Obtaining an instance of the builtin type 'dict' (line 212)
        dict_231471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 212)
        # Adding element type (key, value) (line 212)
        str_231472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 24), 'str', 'data')
        
        # Call to dumps(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'content' (line 212)
        content_231475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 43), 'content', False)
        # Processing the call keyword arguments (line 212)
        kwargs_231476 = {}
        # Getting the type of 'json' (line 212)
        json_231473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 32), 'json', False)
        # Obtaining the member 'dumps' of a type (line 212)
        dumps_231474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 32), json_231473, 'dumps')
        # Calling dumps(args, kwargs) (line 212)
        dumps_call_result_231477 = invoke(stypy.reporting.localization.Localization(__file__, 212, 32), dumps_231474, *[content_231475], **kwargs_231476)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 23), dict_231471, (str_231472, dumps_call_result_231477))
        
        # Processing the call keyword arguments (line 212)
        kwargs_231478 = {}
        # Getting the type of 'self' (line 212)
        self_231468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'self', False)
        # Obtaining the member 'comm' of a type (line 212)
        comm_231469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), self_231468, 'comm')
        # Obtaining the member 'send' of a type (line 212)
        send_231470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), comm_231469, 'send')
        # Calling send(args, kwargs) (line 212)
        send_call_result_231479 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), send_231470, *[dict_231471], **kwargs_231478)
        
        
        # ################# End of 'send_json(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'send_json' in the type store
        # Getting the type of 'stypy_return_type' (line 211)
        stypy_return_type_231480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231480)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'send_json'
        return stypy_return_type_231480


    @norecursion
    def send_binary(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'send_binary'
        module_type_store = module_type_store.open_function_context('send_binary', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommSocket.send_binary.__dict__.__setitem__('stypy_localization', localization)
        CommSocket.send_binary.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommSocket.send_binary.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommSocket.send_binary.__dict__.__setitem__('stypy_function_name', 'CommSocket.send_binary')
        CommSocket.send_binary.__dict__.__setitem__('stypy_param_names_list', ['blob'])
        CommSocket.send_binary.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommSocket.send_binary.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommSocket.send_binary.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommSocket.send_binary.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommSocket.send_binary.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommSocket.send_binary.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommSocket.send_binary', ['blob'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 217):
        
        # Call to b64encode(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'blob' (line 217)
        blob_231482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 'blob', False)
        # Processing the call keyword arguments (line 217)
        kwargs_231483 = {}
        # Getting the type of 'b64encode' (line 217)
        b64encode_231481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'b64encode', False)
        # Calling b64encode(args, kwargs) (line 217)
        b64encode_call_result_231484 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), b64encode_231481, *[blob_231482], **kwargs_231483)
        
        # Assigning a type to the variable 'data' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'data', b64encode_call_result_231484)
        
        # Getting the type of 'six' (line 218)
        six_231485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'six')
        # Obtaining the member 'PY3' of a type (line 218)
        PY3_231486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 11), six_231485, 'PY3')
        # Testing the type of an if condition (line 218)
        if_condition_231487 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 8), PY3_231486)
        # Assigning a type to the variable 'if_condition_231487' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'if_condition_231487', if_condition_231487)
        # SSA begins for if statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 219):
        
        # Call to decode(...): (line 219)
        # Processing the call arguments (line 219)
        str_231490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 31), 'str', 'ascii')
        # Processing the call keyword arguments (line 219)
        kwargs_231491 = {}
        # Getting the type of 'data' (line 219)
        data_231488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 19), 'data', False)
        # Obtaining the member 'decode' of a type (line 219)
        decode_231489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 19), data_231488, 'decode')
        # Calling decode(args, kwargs) (line 219)
        decode_call_result_231492 = invoke(stypy.reporting.localization.Localization(__file__, 219, 19), decode_231489, *[str_231490], **kwargs_231491)
        
        # Assigning a type to the variable 'data' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'data', decode_call_result_231492)
        # SSA join for if statement (line 218)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 220):
        
        # Call to format(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'data' (line 220)
        data_231495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 54), 'data', False)
        # Processing the call keyword arguments (line 220)
        kwargs_231496 = {}
        str_231493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 19), 'str', 'data:image/png;base64,{0}')
        # Obtaining the member 'format' of a type (line 220)
        format_231494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), str_231493, 'format')
        # Calling format(args, kwargs) (line 220)
        format_call_result_231497 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), format_231494, *[data_231495], **kwargs_231496)
        
        # Assigning a type to the variable 'data_uri' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'data_uri', format_call_result_231497)
        
        # Call to send(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining an instance of the builtin type 'dict' (line 221)
        dict_231501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 221)
        # Adding element type (key, value) (line 221)
        str_231502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 24), 'str', 'data')
        # Getting the type of 'data_uri' (line 221)
        data_uri_231503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 32), 'data_uri', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 23), dict_231501, (str_231502, data_uri_231503))
        
        # Processing the call keyword arguments (line 221)
        kwargs_231504 = {}
        # Getting the type of 'self' (line 221)
        self_231498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self', False)
        # Obtaining the member 'comm' of a type (line 221)
        comm_231499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_231498, 'comm')
        # Obtaining the member 'send' of a type (line 221)
        send_231500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), comm_231499, 'send')
        # Calling send(args, kwargs) (line 221)
        send_call_result_231505 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), send_231500, *[dict_231501], **kwargs_231504)
        
        
        # ################# End of 'send_binary(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'send_binary' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_231506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231506)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'send_binary'
        return stypy_return_type_231506


    @norecursion
    def on_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'on_message'
        module_type_store = module_type_store.open_function_context('on_message', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CommSocket.on_message.__dict__.__setitem__('stypy_localization', localization)
        CommSocket.on_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CommSocket.on_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        CommSocket.on_message.__dict__.__setitem__('stypy_function_name', 'CommSocket.on_message')
        CommSocket.on_message.__dict__.__setitem__('stypy_param_names_list', ['message'])
        CommSocket.on_message.__dict__.__setitem__('stypy_varargs_param_name', None)
        CommSocket.on_message.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CommSocket.on_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        CommSocket.on_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        CommSocket.on_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CommSocket.on_message.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CommSocket.on_message', ['message'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 229):
        
        # Call to loads(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Obtaining the type of the subscript
        str_231509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 48), 'str', 'data')
        
        # Obtaining the type of the subscript
        str_231510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 37), 'str', 'content')
        # Getting the type of 'message' (line 229)
        message_231511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 29), 'message', False)
        # Obtaining the member '__getitem__' of a type (line 229)
        getitem___231512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 29), message_231511, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 229)
        subscript_call_result_231513 = invoke(stypy.reporting.localization.Localization(__file__, 229, 29), getitem___231512, str_231510)
        
        # Obtaining the member '__getitem__' of a type (line 229)
        getitem___231514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 29), subscript_call_result_231513, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 229)
        subscript_call_result_231515 = invoke(stypy.reporting.localization.Localization(__file__, 229, 29), getitem___231514, str_231509)
        
        # Processing the call keyword arguments (line 229)
        kwargs_231516 = {}
        # Getting the type of 'json' (line 229)
        json_231507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 18), 'json', False)
        # Obtaining the member 'loads' of a type (line 229)
        loads_231508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 18), json_231507, 'loads')
        # Calling loads(args, kwargs) (line 229)
        loads_call_result_231517 = invoke(stypy.reporting.localization.Localization(__file__, 229, 18), loads_231508, *[subscript_call_result_231515], **kwargs_231516)
        
        # Assigning a type to the variable 'message' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'message', loads_call_result_231517)
        
        
        
        # Obtaining the type of the subscript
        str_231518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 19), 'str', 'type')
        # Getting the type of 'message' (line 230)
        message_231519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'message')
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___231520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 11), message_231519, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 230)
        subscript_call_result_231521 = invoke(stypy.reporting.localization.Localization(__file__, 230, 11), getitem___231520, str_231518)
        
        str_231522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 30), 'str', 'closing')
        # Applying the binary operator '==' (line 230)
        result_eq_231523 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 11), '==', subscript_call_result_231521, str_231522)
        
        # Testing the type of an if condition (line 230)
        if_condition_231524 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), result_eq_231523)
        # Assigning a type to the variable 'if_condition_231524' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_231524', if_condition_231524)
        # SSA begins for if statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to on_close(...): (line 231)
        # Processing the call keyword arguments (line 231)
        kwargs_231527 = {}
        # Getting the type of 'self' (line 231)
        self_231525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'self', False)
        # Obtaining the member 'on_close' of a type (line 231)
        on_close_231526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), self_231525, 'on_close')
        # Calling on_close(args, kwargs) (line 231)
        on_close_call_result_231528 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), on_close_231526, *[], **kwargs_231527)
        
        
        # Call to clearup_closed(...): (line 232)
        # Processing the call keyword arguments (line 232)
        kwargs_231532 = {}
        # Getting the type of 'self' (line 232)
        self_231529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'self', False)
        # Obtaining the member 'manager' of a type (line 232)
        manager_231530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), self_231529, 'manager')
        # Obtaining the member 'clearup_closed' of a type (line 232)
        clearup_closed_231531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), manager_231530, 'clearup_closed')
        # Calling clearup_closed(args, kwargs) (line 232)
        clearup_closed_call_result_231533 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), clearup_closed_231531, *[], **kwargs_231532)
        
        # SSA branch for the else part of an if statement (line 230)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        str_231534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 21), 'str', 'type')
        # Getting the type of 'message' (line 233)
        message_231535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 13), 'message')
        # Obtaining the member '__getitem__' of a type (line 233)
        getitem___231536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 13), message_231535, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 233)
        subscript_call_result_231537 = invoke(stypy.reporting.localization.Localization(__file__, 233, 13), getitem___231536, str_231534)
        
        str_231538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 32), 'str', 'supports_binary')
        # Applying the binary operator '==' (line 233)
        result_eq_231539 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 13), '==', subscript_call_result_231537, str_231538)
        
        # Testing the type of an if condition (line 233)
        if_condition_231540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 13), result_eq_231539)
        # Assigning a type to the variable 'if_condition_231540' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 13), 'if_condition_231540', if_condition_231540)
        # SSA begins for if statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Attribute (line 234):
        
        # Obtaining the type of the subscript
        str_231541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 43), 'str', 'value')
        # Getting the type of 'message' (line 234)
        message_231542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'message')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___231543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 35), message_231542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_231544 = invoke(stypy.reporting.localization.Localization(__file__, 234, 35), getitem___231543, str_231541)
        
        # Getting the type of 'self' (line 234)
        self_231545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'self')
        # Setting the type of the member 'supports_binary' of a type (line 234)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), self_231545, 'supports_binary', subscript_call_result_231544)
        # SSA branch for the else part of an if statement (line 233)
        module_type_store.open_ssa_branch('else')
        
        # Call to handle_json(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'message' (line 236)
        message_231549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 37), 'message', False)
        # Processing the call keyword arguments (line 236)
        kwargs_231550 = {}
        # Getting the type of 'self' (line 236)
        self_231546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'self', False)
        # Obtaining the member 'manager' of a type (line 236)
        manager_231547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), self_231546, 'manager')
        # Obtaining the member 'handle_json' of a type (line 236)
        handle_json_231548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), manager_231547, 'handle_json')
        # Calling handle_json(args, kwargs) (line 236)
        handle_json_call_result_231551 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), handle_json_231548, *[message_231549], **kwargs_231550)
        
        # SSA join for if statement (line 233)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 230)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'on_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'on_message' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_231552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'on_message'
        return stypy_return_type_231552


# Assigning a type to the variable 'CommSocket' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'CommSocket', CommSocket)
# Declaration of the '_BackendNbAgg' class
# Getting the type of '_Backend' (line 240)
_Backend_231553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), '_Backend')

class _BackendNbAgg(_Backend_231553, ):

    @staticmethod
    @norecursion
    def new_figure_manager_given_figure(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_figure_manager_given_figure'
        module_type_store = module_type_store.open_function_context('new_figure_manager_given_figure', 244, 4, False)
        
        # Passed parameters checking function
        _BackendNbAgg.new_figure_manager_given_figure.__dict__.__setitem__('stypy_localization', localization)
        _BackendNbAgg.new_figure_manager_given_figure.__dict__.__setitem__('stypy_type_of_self', None)
        _BackendNbAgg.new_figure_manager_given_figure.__dict__.__setitem__('stypy_type_store', module_type_store)
        _BackendNbAgg.new_figure_manager_given_figure.__dict__.__setitem__('stypy_function_name', 'new_figure_manager_given_figure')
        _BackendNbAgg.new_figure_manager_given_figure.__dict__.__setitem__('stypy_param_names_list', ['num', 'figure'])
        _BackendNbAgg.new_figure_manager_given_figure.__dict__.__setitem__('stypy_varargs_param_name', None)
        _BackendNbAgg.new_figure_manager_given_figure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _BackendNbAgg.new_figure_manager_given_figure.__dict__.__setitem__('stypy_call_defaults', defaults)
        _BackendNbAgg.new_figure_manager_given_figure.__dict__.__setitem__('stypy_call_varargs', varargs)
        _BackendNbAgg.new_figure_manager_given_figure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _BackendNbAgg.new_figure_manager_given_figure.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'new_figure_manager_given_figure', ['num', 'figure'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_figure_manager_given_figure', localization, ['figure'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_figure_manager_given_figure(...)' code ##################

        
        # Assigning a Call to a Name (line 246):
        
        # Call to FigureCanvasNbAgg(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'figure' (line 246)
        figure_231555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 35), 'figure', False)
        # Processing the call keyword arguments (line 246)
        kwargs_231556 = {}
        # Getting the type of 'FigureCanvasNbAgg' (line 246)
        FigureCanvasNbAgg_231554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 17), 'FigureCanvasNbAgg', False)
        # Calling FigureCanvasNbAgg(args, kwargs) (line 246)
        FigureCanvasNbAgg_call_result_231557 = invoke(stypy.reporting.localization.Localization(__file__, 246, 17), FigureCanvasNbAgg_231554, *[figure_231555], **kwargs_231556)
        
        # Assigning a type to the variable 'canvas' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'canvas', FigureCanvasNbAgg_call_result_231557)
        
        
        # Obtaining the type of the subscript
        str_231558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 20), 'str', 'nbagg.transparent')
        # Getting the type of 'rcParams' (line 247)
        rcParams_231559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'rcParams')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___231560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 11), rcParams_231559, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_231561 = invoke(stypy.reporting.localization.Localization(__file__, 247, 11), getitem___231560, str_231558)
        
        # Testing the type of an if condition (line 247)
        if_condition_231562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 8), subscript_call_result_231561)
        # Assigning a type to the variable 'if_condition_231562' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'if_condition_231562', if_condition_231562)
        # SSA begins for if statement (line 247)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_alpha(...): (line 248)
        # Processing the call arguments (line 248)
        int_231566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 35), 'int')
        # Processing the call keyword arguments (line 248)
        kwargs_231567 = {}
        # Getting the type of 'figure' (line 248)
        figure_231563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'figure', False)
        # Obtaining the member 'patch' of a type (line 248)
        patch_231564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), figure_231563, 'patch')
        # Obtaining the member 'set_alpha' of a type (line 248)
        set_alpha_231565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), patch_231564, 'set_alpha')
        # Calling set_alpha(args, kwargs) (line 248)
        set_alpha_call_result_231568 = invoke(stypy.reporting.localization.Localization(__file__, 248, 12), set_alpha_231565, *[int_231566], **kwargs_231567)
        
        # SSA join for if statement (line 247)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 249):
        
        # Call to FigureManagerNbAgg(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'canvas' (line 249)
        canvas_231570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 37), 'canvas', False)
        # Getting the type of 'num' (line 249)
        num_231571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 45), 'num', False)
        # Processing the call keyword arguments (line 249)
        kwargs_231572 = {}
        # Getting the type of 'FigureManagerNbAgg' (line 249)
        FigureManagerNbAgg_231569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 18), 'FigureManagerNbAgg', False)
        # Calling FigureManagerNbAgg(args, kwargs) (line 249)
        FigureManagerNbAgg_call_result_231573 = invoke(stypy.reporting.localization.Localization(__file__, 249, 18), FigureManagerNbAgg_231569, *[canvas_231570, num_231571], **kwargs_231572)
        
        # Assigning a type to the variable 'manager' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'manager', FigureManagerNbAgg_call_result_231573)
        
        
        # Call to is_interactive(...): (line 250)
        # Processing the call keyword arguments (line 250)
        kwargs_231575 = {}
        # Getting the type of 'is_interactive' (line 250)
        is_interactive_231574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 11), 'is_interactive', False)
        # Calling is_interactive(args, kwargs) (line 250)
        is_interactive_call_result_231576 = invoke(stypy.reporting.localization.Localization(__file__, 250, 11), is_interactive_231574, *[], **kwargs_231575)
        
        # Testing the type of an if condition (line 250)
        if_condition_231577 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 8), is_interactive_call_result_231576)
        # Assigning a type to the variable 'if_condition_231577' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'if_condition_231577', if_condition_231577)
        # SSA begins for if statement (line 250)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to show(...): (line 251)
        # Processing the call keyword arguments (line 251)
        kwargs_231580 = {}
        # Getting the type of 'manager' (line 251)
        manager_231578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'manager', False)
        # Obtaining the member 'show' of a type (line 251)
        show_231579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 12), manager_231578, 'show')
        # Calling show(args, kwargs) (line 251)
        show_call_result_231581 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), show_231579, *[], **kwargs_231580)
        
        
        # Call to draw_idle(...): (line 252)
        # Processing the call keyword arguments (line 252)
        kwargs_231585 = {}
        # Getting the type of 'figure' (line 252)
        figure_231582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'figure', False)
        # Obtaining the member 'canvas' of a type (line 252)
        canvas_231583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 12), figure_231582, 'canvas')
        # Obtaining the member 'draw_idle' of a type (line 252)
        draw_idle_231584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 12), canvas_231583, 'draw_idle')
        # Calling draw_idle(args, kwargs) (line 252)
        draw_idle_call_result_231586 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), draw_idle_231584, *[], **kwargs_231585)
        
        # SSA join for if statement (line 250)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to mpl_connect(...): (line 253)
        # Processing the call arguments (line 253)
        str_231589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 27), 'str', 'close_event')

        @norecursion
        def _stypy_temp_lambda_100(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_100'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_100', 253, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_100.stypy_localization = localization
            _stypy_temp_lambda_100.stypy_type_of_self = None
            _stypy_temp_lambda_100.stypy_type_store = module_type_store
            _stypy_temp_lambda_100.stypy_function_name = '_stypy_temp_lambda_100'
            _stypy_temp_lambda_100.stypy_param_names_list = ['event']
            _stypy_temp_lambda_100.stypy_varargs_param_name = None
            _stypy_temp_lambda_100.stypy_kwargs_param_name = None
            _stypy_temp_lambda_100.stypy_call_defaults = defaults
            _stypy_temp_lambda_100.stypy_call_varargs = varargs
            _stypy_temp_lambda_100.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_100', ['event'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_100', ['event'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to destroy(...): (line 253)
            # Processing the call arguments (line 253)
            # Getting the type of 'num' (line 253)
            num_231592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 68), 'num', False)
            # Processing the call keyword arguments (line 253)
            kwargs_231593 = {}
            # Getting the type of 'Gcf' (line 253)
            Gcf_231590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 56), 'Gcf', False)
            # Obtaining the member 'destroy' of a type (line 253)
            destroy_231591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 56), Gcf_231590, 'destroy')
            # Calling destroy(args, kwargs) (line 253)
            destroy_call_result_231594 = invoke(stypy.reporting.localization.Localization(__file__, 253, 56), destroy_231591, *[num_231592], **kwargs_231593)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 253)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 42), 'stypy_return_type', destroy_call_result_231594)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_100' in the type store
            # Getting the type of 'stypy_return_type' (line 253)
            stypy_return_type_231595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_231595)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_100'
            return stypy_return_type_231595

        # Assigning a type to the variable '_stypy_temp_lambda_100' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 42), '_stypy_temp_lambda_100', _stypy_temp_lambda_100)
        # Getting the type of '_stypy_temp_lambda_100' (line 253)
        _stypy_temp_lambda_100_231596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 42), '_stypy_temp_lambda_100')
        # Processing the call keyword arguments (line 253)
        kwargs_231597 = {}
        # Getting the type of 'canvas' (line 253)
        canvas_231587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'canvas', False)
        # Obtaining the member 'mpl_connect' of a type (line 253)
        mpl_connect_231588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), canvas_231587, 'mpl_connect')
        # Calling mpl_connect(args, kwargs) (line 253)
        mpl_connect_call_result_231598 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), mpl_connect_231588, *[str_231589, _stypy_temp_lambda_100_231596], **kwargs_231597)
        
        # Getting the type of 'manager' (line 254)
        manager_231599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'manager')
        # Assigning a type to the variable 'stypy_return_type' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'stypy_return_type', manager_231599)
        
        # ################# End of 'new_figure_manager_given_figure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_figure_manager_given_figure' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_231600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231600)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_figure_manager_given_figure'
        return stypy_return_type_231600


    @staticmethod
    @norecursion
    def trigger_manager_draw(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'trigger_manager_draw'
        module_type_store = module_type_store.open_function_context('trigger_manager_draw', 256, 4, False)
        
        # Passed parameters checking function
        _BackendNbAgg.trigger_manager_draw.__dict__.__setitem__('stypy_localization', localization)
        _BackendNbAgg.trigger_manager_draw.__dict__.__setitem__('stypy_type_of_self', None)
        _BackendNbAgg.trigger_manager_draw.__dict__.__setitem__('stypy_type_store', module_type_store)
        _BackendNbAgg.trigger_manager_draw.__dict__.__setitem__('stypy_function_name', 'trigger_manager_draw')
        _BackendNbAgg.trigger_manager_draw.__dict__.__setitem__('stypy_param_names_list', ['manager'])
        _BackendNbAgg.trigger_manager_draw.__dict__.__setitem__('stypy_varargs_param_name', None)
        _BackendNbAgg.trigger_manager_draw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _BackendNbAgg.trigger_manager_draw.__dict__.__setitem__('stypy_call_defaults', defaults)
        _BackendNbAgg.trigger_manager_draw.__dict__.__setitem__('stypy_call_varargs', varargs)
        _BackendNbAgg.trigger_manager_draw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _BackendNbAgg.trigger_manager_draw.__dict__.__setitem__('stypy_declared_arg_number', 1)
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

        
        # Call to show(...): (line 258)
        # Processing the call keyword arguments (line 258)
        kwargs_231603 = {}
        # Getting the type of 'manager' (line 258)
        manager_231601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'manager', False)
        # Obtaining the member 'show' of a type (line 258)
        show_231602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), manager_231601, 'show')
        # Calling show(args, kwargs) (line 258)
        show_call_result_231604 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), show_231602, *[], **kwargs_231603)
        
        
        # ################# End of 'trigger_manager_draw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trigger_manager_draw' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_231605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231605)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trigger_manager_draw'
        return stypy_return_type_231605


    @staticmethod
    @norecursion
    def show(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'show'
        module_type_store = module_type_store.open_function_context('show', 260, 4, False)
        
        # Passed parameters checking function
        _BackendNbAgg.show.__dict__.__setitem__('stypy_localization', localization)
        _BackendNbAgg.show.__dict__.__setitem__('stypy_type_of_self', None)
        _BackendNbAgg.show.__dict__.__setitem__('stypy_type_store', module_type_store)
        _BackendNbAgg.show.__dict__.__setitem__('stypy_function_name', 'show')
        _BackendNbAgg.show.__dict__.__setitem__('stypy_param_names_list', [])
        _BackendNbAgg.show.__dict__.__setitem__('stypy_varargs_param_name', None)
        _BackendNbAgg.show.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _BackendNbAgg.show.__dict__.__setitem__('stypy_call_defaults', defaults)
        _BackendNbAgg.show.__dict__.__setitem__('stypy_call_varargs', varargs)
        _BackendNbAgg.show.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _BackendNbAgg.show.__dict__.__setitem__('stypy_declared_arg_number', 0)
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

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 262, 8))
        
        # 'from matplotlib._pylab_helpers import Gcf' statement (line 262)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
        import_231606 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 262, 8), 'matplotlib._pylab_helpers')

        if (type(import_231606) is not StypyTypeError):

            if (import_231606 != 'pyd_module'):
                __import__(import_231606)
                sys_modules_231607 = sys.modules[import_231606]
                import_from_module(stypy.reporting.localization.Localization(__file__, 262, 8), 'matplotlib._pylab_helpers', sys_modules_231607.module_type_store, module_type_store, ['Gcf'])
                nest_module(stypy.reporting.localization.Localization(__file__, 262, 8), __file__, sys_modules_231607, sys_modules_231607.module_type_store, module_type_store)
            else:
                from matplotlib._pylab_helpers import Gcf

                import_from_module(stypy.reporting.localization.Localization(__file__, 262, 8), 'matplotlib._pylab_helpers', None, module_type_store, ['Gcf'], [Gcf])

        else:
            # Assigning a type to the variable 'matplotlib._pylab_helpers' (line 262)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'matplotlib._pylab_helpers', import_231606)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')
        
        
        # Assigning a Call to a Name (line 264):
        
        # Call to get_all_fig_managers(...): (line 264)
        # Processing the call keyword arguments (line 264)
        kwargs_231610 = {}
        # Getting the type of 'Gcf' (line 264)
        Gcf_231608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 19), 'Gcf', False)
        # Obtaining the member 'get_all_fig_managers' of a type (line 264)
        get_all_fig_managers_231609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 19), Gcf_231608, 'get_all_fig_managers')
        # Calling get_all_fig_managers(args, kwargs) (line 264)
        get_all_fig_managers_call_result_231611 = invoke(stypy.reporting.localization.Localization(__file__, 264, 19), get_all_fig_managers_231609, *[], **kwargs_231610)
        
        # Assigning a type to the variable 'managers' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'managers', get_all_fig_managers_call_result_231611)
        
        
        # Getting the type of 'managers' (line 265)
        managers_231612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'managers')
        # Applying the 'not' unary operator (line 265)
        result_not__231613 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 11), 'not', managers_231612)
        
        # Testing the type of an if condition (line 265)
        if_condition_231614 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), result_not__231613)
        # Assigning a type to the variable 'if_condition_231614' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_231614', if_condition_231614)
        # SSA begins for if statement (line 265)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 265)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 268):
        
        # Call to is_interactive(...): (line 268)
        # Processing the call keyword arguments (line 268)
        kwargs_231616 = {}
        # Getting the type of 'is_interactive' (line 268)
        is_interactive_231615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'is_interactive', False)
        # Calling is_interactive(args, kwargs) (line 268)
        is_interactive_call_result_231617 = invoke(stypy.reporting.localization.Localization(__file__, 268, 22), is_interactive_231615, *[], **kwargs_231616)
        
        # Assigning a type to the variable 'interactive' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'interactive', is_interactive_call_result_231617)
        
        # Getting the type of 'managers' (line 270)
        managers_231618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 23), 'managers')
        # Testing the type of a for loop iterable (line 270)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 270, 8), managers_231618)
        # Getting the type of the for loop variable (line 270)
        for_loop_var_231619 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 270, 8), managers_231618)
        # Assigning a type to the variable 'manager' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'manager', for_loop_var_231619)
        # SSA begins for a for statement (line 270)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to show(...): (line 271)
        # Processing the call keyword arguments (line 271)
        kwargs_231622 = {}
        # Getting the type of 'manager' (line 271)
        manager_231620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'manager', False)
        # Obtaining the member 'show' of a type (line 271)
        show_231621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), manager_231620, 'show')
        # Calling show(args, kwargs) (line 271)
        show_call_result_231623 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), show_231621, *[], **kwargs_231622)
        
        
        # Type idiom detected: calculating its left and rigth part (line 277)
        str_231624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 32), 'str', '_cidgcf')
        # Getting the type of 'manager' (line 277)
        manager_231625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 23), 'manager')
        
        (may_be_231626, more_types_in_union_231627) = may_provide_member(str_231624, manager_231625)

        if may_be_231626:

            if more_types_in_union_231627:
                # Runtime conditional SSA (line 277)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'manager' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'manager', remove_not_member_provider_from_union(manager_231625, '_cidgcf'))
            
            # Call to mpl_disconnect(...): (line 278)
            # Processing the call arguments (line 278)
            # Getting the type of 'manager' (line 278)
            manager_231631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 46), 'manager', False)
            # Obtaining the member '_cidgcf' of a type (line 278)
            _cidgcf_231632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 46), manager_231631, '_cidgcf')
            # Processing the call keyword arguments (line 278)
            kwargs_231633 = {}
            # Getting the type of 'manager' (line 278)
            manager_231628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'manager', False)
            # Obtaining the member 'canvas' of a type (line 278)
            canvas_231629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 16), manager_231628, 'canvas')
            # Obtaining the member 'mpl_disconnect' of a type (line 278)
            mpl_disconnect_231630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 16), canvas_231629, 'mpl_disconnect')
            # Calling mpl_disconnect(args, kwargs) (line 278)
            mpl_disconnect_call_result_231634 = invoke(stypy.reporting.localization.Localization(__file__, 278, 16), mpl_disconnect_231630, *[_cidgcf_231632], **kwargs_231633)
            

            if more_types_in_union_231627:
                # SSA join for if statement (line 277)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'interactive' (line 280)
        interactive_231635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'interactive')
        # Applying the 'not' unary operator (line 280)
        result_not__231636 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 15), 'not', interactive_231635)
        
        
        # Getting the type of 'manager' (line 280)
        manager_231637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 35), 'manager')
        # Getting the type of 'Gcf' (line 280)
        Gcf_231638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 46), 'Gcf')
        # Obtaining the member '_activeQue' of a type (line 280)
        _activeQue_231639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 46), Gcf_231638, '_activeQue')
        # Applying the binary operator 'in' (line 280)
        result_contains_231640 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 35), 'in', manager_231637, _activeQue_231639)
        
        # Applying the binary operator 'and' (line 280)
        result_and_keyword_231641 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 15), 'and', result_not__231636, result_contains_231640)
        
        # Testing the type of an if condition (line 280)
        if_condition_231642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 12), result_and_keyword_231641)
        # Assigning a type to the variable 'if_condition_231642' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'if_condition_231642', if_condition_231642)
        # SSA begins for if statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'manager' (line 281)
        manager_231646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 38), 'manager', False)
        # Processing the call keyword arguments (line 281)
        kwargs_231647 = {}
        # Getting the type of 'Gcf' (line 281)
        Gcf_231643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'Gcf', False)
        # Obtaining the member '_activeQue' of a type (line 281)
        _activeQue_231644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), Gcf_231643, '_activeQue')
        # Obtaining the member 'remove' of a type (line 281)
        remove_231645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), _activeQue_231644, 'remove')
        # Calling remove(args, kwargs) (line 281)
        remove_call_result_231648 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), remove_231645, *[manager_231646], **kwargs_231647)
        
        # SSA join for if statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'show(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'show' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_231649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231649)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'show'
        return stypy_return_type_231649


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 239, 0, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_BackendNbAgg.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_BackendNbAgg' (line 239)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), '_BackendNbAgg', _BackendNbAgg)

# Assigning a Name to a Name (line 241):
# Getting the type of 'FigureCanvasNbAgg' (line 241)
FigureCanvasNbAgg_231650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 19), 'FigureCanvasNbAgg')
# Getting the type of '_BackendNbAgg'
_BackendNbAgg_231651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendNbAgg')
# Setting the type of the member 'FigureCanvas' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendNbAgg_231651, 'FigureCanvas', FigureCanvasNbAgg_231650)

# Assigning a Name to a Name (line 242):
# Getting the type of 'FigureManagerNbAgg' (line 242)
FigureManagerNbAgg_231652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'FigureManagerNbAgg')
# Getting the type of '_BackendNbAgg'
_BackendNbAgg_231653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_BackendNbAgg')
# Setting the type of the member 'FigureManager' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _BackendNbAgg_231653, 'FigureManager', FigureManagerNbAgg_231652)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
